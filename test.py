import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
from types import MethodType
from tools.env import init_dist
from tqdm import tqdm

from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label

from models.HAMMER import HAMMER

def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger

def parse_csv_arg(value):
    if value is None:
        return None
    return [x.strip() for x in value.split(',') if x.strip()]

def apply_config_overrides(config, args):
    if args.data_root:
        config['data_root'] = args.data_root
    train_file = parse_csv_arg(args.train_file)
    if train_file:
        config['train_file'] = train_file
    val_file = parse_csv_arg(args.val_file)
    if val_file:
        config['val_file'] = val_file
    val_sources = parse_csv_arg(args.val_sources)
    if val_sources is not None:
        config['val_sources'] = val_sources

def safe_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def _safe_multilabel_metrics(meter):
    if meter.scores.numel() == 0:
        return float('nan'), float('nan'), float('nan')
    map_tensor = meter.value()
    map_score = map_tensor.mean().item() if torch.is_tensor(map_tensor) else float(map_tensor)
    overall = meter.overall()
    if isinstance(overall, tuple):
        _, _, of1, _, _, cf1 = overall
    else:
        of1 = float('nan')
        cf1 = float('nan')
    return map_score, cf1, of1


def _safe_binary_metrics(y_true, y_score, y_pred_label):
    if y_true.size == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float('nan')

    acc = float(np.mean(y_pred_label == y_true))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        fnr = 1.0 - tpr
        diff = fpr - fnr
        cross_idx = np.where(diff[:-1] * diff[1:] <= 0)[0]
        if cross_idx.size > 0:
            i = int(cross_idx[0])
            x0, x1 = diff[i], diff[i + 1]
            y0, y1 = fpr[i], fpr[i + 1]
            if x1 == x0:
                eer = float(y0)
            else:
                t = float(-x0 / (x1 - x0))
                eer = float(y0 + t * (y1 - y0))
        else:
            i = int(np.argmin(np.abs(diff)))
            eer = float((fpr[i] + fnr[i]) / 2.0)
    except ValueError:
        eer = float('nan')
    f1 = f1_score(y_true, y_pred_label, zero_division=0)
    return auc, acc, eer, f1


def text_input_adjust(text_input, fake_word_pos, device):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers

        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) # get the sub-word position (token position)
        
        # transfer the fake word position into fake token position
        for i in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch

  

@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    print_freq = 200 

    y_true, y_pred, y_pred_label = [], [], []

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()
    multi_nums_all = 0
    multi_exact_correct_all = 0

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

        logits_real_fake, logits_multicls, output_coord, logits_tok = model(image, label, text_input, fake_image_box, fake_token_pos, is_train=False)

        ##================= real/fake cls ========================## 
        cls_label = torch.ones(len(label), dtype=torch.long).to(image.device) 
        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        cls_label[real_label_pos] = 0

        y_pred.extend(F.softmax(logits_real_fake,dim=1)[:,1].cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        pred_acc = logits_real_fake.argmax(1)
        y_pred_label.extend(pred_acc.cpu().flatten().tolist())

        # ----- multi metrics -----
        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)
        pred_multi = (torch.sigmoid(logits_multicls) >= 0.5).long()
        multi_nums_all += target.shape[0]
        multi_exact_correct_all += torch.sum(torch.all(pred_multi == target, dim=1)).item()
        
        
    ##================= real/fake cls ========================## 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_label = np.array(y_pred_label)
    AUC_cls, ACC_cls, EER_cls, BINARY_F1 = _safe_binary_metrics(y_true, y_pred, y_pred_label)
    ##================= multi-label cls ========================## 
    MAP, CF1, OC1 = _safe_multilabel_metrics(multi_label_meter)
    ERR_multi = 1.0 - (multi_exact_correct_all / multi_nums_all) if multi_nums_all > 0 else float('nan')

    return {
        "AUC_cls": AUC_cls,
        "ACC_cls": ACC_cls,
        "EER_cls": EER_cls,
        "Binary_F1": BINARY_F1,
        "MAP": MAP,
        "ERR_multi": ERR_multi,
        "CF1": CF1,
        "OC1": OC1,
    }
    
def main_worker(gpu, args, config):

    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    if config.get('val_sources'):
        eval_type = "_".join([str(x).lower().replace(" ", "_") for x in config['val_sources']])
    else:
        eval_type = os.path.basename(config['val_file'][0]).split('.')[0]
        if eval_type == 'test':
            eval_type = 'all'
    log_dir = os.path.join(args.output_dir, args.log_num, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'shell_{eval_type}.txt')
    logger = setlogger(log_file)
    
    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')

    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


    #### Model #### 
    tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)
    if args.log:
        print(f"Creating MAMMER")
    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    
    model = model.to(device)   

    checkpoint_dir = f'{args.output_dir}/{args.log_num}/checkpoint_{args.test_epoch}.pth'
    checkpoint = torch.load(checkpoint_dir, map_location='cpu') 
    state_dict = checkpoint['model']                       

    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
                   
    # model.load_state_dict(state_dict)  
    if args.log:
        print('load checkpoint from %s'%checkpoint_dir)  
    msg = model.load_state_dict(state_dict, strict=False)
    if args.log:
        print(msg)  

    #### Dataset #### 
    if args.log:
        print("Creating dataset")
    _, val_dataset = create_dataset(config)
    
    if args.distributed:  
        samplers = create_sampler([val_dataset], [True], args.world_size, args.rank) + [None]    
    else:
        samplers = [None]

    val_loader = create_loader([val_dataset],
                                samplers,
                                batch_size=[config['batch_size_val']], 
                                num_workers=[4], 
                                is_trains=[False], 
                                collate_fns=[None])[0]

    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.log:
        print("Start evaluation")

    metrics = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config)
    #============ evaluation info ============#
    val_stats = {"AUC_cls": "{:.4f}".format(metrics["AUC_cls"]*100),
                    "ACC_cls": "{:.4f}".format(metrics["ACC_cls"]*100),
                    "EER_cls": "{:.4f}".format(metrics["EER_cls"]*100),
                    "Binary_F1": "{:.4f}".format(metrics["Binary_F1"]*100),
                    "MAP": "{:.4f}".format(metrics["MAP"]*100),
                    "ERR_multi": "{:.4f}".format(metrics["ERR_multi"]*100),
                    "CF1": "{:.4f}".format(metrics["CF1"]*100),
                    "OC1": "{:.4f}".format(metrics["OC1"]*100),
    }
    
    if utils.is_main_process(): 
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': args.test_epoch,
                    }             
        with open(os.path.join(log_dir, f"results_{eval_type}.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")
    safe_barrier()

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='/mnt/lustre/share/rshao/data/FakeNews/Ours/results')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    # parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=5)
    parser.add_argument('--token_momentum', default=False, action='store_true')
    parser.add_argument('--test_epoch', default='best', type=str)
    parser.add_argument('--data_root', default=None, type=str)
    parser.add_argument('--train_file', default=None, type=str, help='comma-separated json paths')
    parser.add_argument('--val_file', default=None, type=str, help='comma-separated json paths')
    parser.add_argument('--val_sources', default=None, type=str, help='comma-separated sources')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    apply_config_overrides(config, args)
 
    main_worker(0, args, config)
