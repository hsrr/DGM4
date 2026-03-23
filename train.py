import warnings
warnings.filterwarnings("ignore")

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
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

from sklearn.metrics import roc_auc_score
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label
from models.HAMMER import HAMMER


def _parse_csv_arg(value):
    if value is None:
        return None
    items = [x.strip() for x in value.split(',') if x.strip()]
    return items


def apply_config_overrides(config, args):
    if args.data_root:
        config['data_root'] = args.data_root
    if args.train_file:
        config['train_file'] = [x.strip() for x in args.train_file.split(',') if x.strip()]
    if args.val_file:
        config['val_file'] = [x.strip() for x in args.val_file.split(',') if x.strip()]

    train_sources = _parse_csv_arg(args.train_sources)
    if train_sources is not None:
        config['train_sources'] = train_sources
    val_sources = _parse_csv_arg(args.val_sources)
    if val_sources is not None:
        config['val_sources'] = val_sources


def safe_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def _safe_multilabel_metrics(meter):
    if meter.scores.numel() == 0:
        return float('nan'), float('nan')
    map_tensor = meter.value()
    map_score = map_tensor.mean().item() if torch.is_tensor(map_tensor) else float(map_tensor)
    overall = meter.overall()
    if isinstance(overall, tuple):
        _, _, _, _, _, cf1 = overall
    else:
        cf1 = float('nan')
    return map_score, cf1

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


def train(args, model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, summary_writer):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_MAC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_BIC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_MLC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 100   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    global_step = epoch*len(data_loader)
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):

        if config['schedular']['sched'] == 'cosine_in_step':
            scheduler.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args, config)        

        optimizer.zero_grad()
  
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)
 
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC = model(image, label, text_input, fake_image_box, fake_token_pos, alpha = alpha)  
            
        # bbox and token branches still run forward, but their GT-based losses are disabled in model.forward.
        loss = config['loss_MAC_wgt']*loss_MAC \
             + config['loss_BIC_wgt']*loss_BIC \
             + config['loss_MLC_wgt']*loss_MLC \
          
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_MAC=loss_MAC.item())
        metric_logger.update(loss_BIC=loss_BIC.item())
        metric_logger.update(loss_MLC=loss_MLC.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations and config['schedular']['sched'] != 'cosine_in_step': 
            scheduler.step(i//step_size)   

        global_step+=1
        

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'lr': optimizer.param_groups[0]["lr"],                                                                                                  
                'loss_MAC': loss_MAC.item(),                                                                                                  
                'loss_BIC': loss_BIC.item(),                                                                                                  
                'loss_MLC': loss_MLC.item(),                                                                                                  
                'loss': loss.item(),                                                                                                  
                    } 
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, global_step)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if args.log:
        print("Averaged stats:", metric_logger.global_avg(), flush=True)     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    



@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    print_freq = 200 

    y_true, y_pred = [], []
    cls_nums_all = 0
    cls_acc_all = 0   
    val_loss_sum = 0.0
    val_sample_count = 0

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

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
        cls_nums_all += cls_label.shape[0]
        cls_acc_all += torch.sum(pred_acc == cls_label).item()

        # ----- multi metrics -----
        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)
        loss_BIC = F.cross_entropy(logits_real_fake, cls_label)
        loss_MLC = F.binary_cross_entropy_with_logits(logits_multicls, target.type(torch.float))
        # Validation loss follows the active label-supervision objective.
        batch_val_loss = config['loss_BIC_wgt'] * loss_BIC + config['loss_MLC_wgt'] * loss_MLC
        batch_size = cls_label.shape[0]
        val_loss_sum += batch_val_loss.item() * batch_size
        val_sample_count += batch_size
        
    ##================= real/fake cls ========================## 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        AUC_cls = roc_auc_score(y_true, y_pred)
    except ValueError:
        AUC_cls = float('nan')
    ACC_cls = cls_acc_all / cls_nums_all if cls_nums_all > 0 else float('nan')
    
    ##================= multi-label cls ========================## 
    MAP, CF1 = _safe_multilabel_metrics(multi_label_meter)
    val_loss = val_loss_sum / val_sample_count if val_sample_count > 0 else float('nan')
    return AUC_cls, ACC_cls, MAP, CF1, val_loss
    
def main_worker(gpu, args, config):

    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    log_dir = os.path.join(args.output_dir, 'log'+ args.log_num)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'shell.txt')
    logger = setlogger(log_file)
    yaml.dump(config, open(os.path.join(log_dir, 'config.yaml'), 'w')) 
    
    if args.log:
        summary_writer = SummaryWriter(log_dir)
    else:
        summary_writer = None

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
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']  
    best = float('-inf')
    best_epoch = 0  
    best_val_loss = float('inf')
    best_val_loss_epoch = 0
    early_stop_patience = 3
    no_improve_epochs = 0

    #### Dataset #### 
    if args.log:
        print("Creating dataset")
    train_dataset, val_dataset = create_dataset(config)
    
    if args.distributed:
        samplers = create_sampler([train_dataset], [True], args.world_size, args.rank) + [None]    
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader([train_dataset, val_dataset],
                                samplers,
                                batch_size=[config['batch_size_train']]+[config['batch_size_val']], 
                                num_workers=[4, 4], 
                                is_trains=[True, False], 
                                collate_fns=[None, None])

    tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)

    #### Model #### 
    if args.log:
        print(f"Creating MAMMER")
    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    model = model.to(device)   
        
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    if config['schedular']['sched'] == 'cosine_in_step':
        args.lr = config['optimizer']['lr']
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1         
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
        # model.load_state_dict(state_dict)  
        if args.log:
            print('load checkpoint from %s'%args.checkpoint)  
        msg = model.load_state_dict(state_dict, strict=False)
        if args.log:
            print(msg)  

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.log:
        print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
            
        train_stats = train(args, model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, summary_writer) 
        AUC_cls, ACC_cls, MAP, CF1, val_loss = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config)

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'AUC_cls': round(AUC_cls*100, 4),                                                                                                  
                'ACC_cls': round(ACC_cls*100, 4),                                                                                                  
                'MAP': round(MAP*100, 4),                                                                                                  
                'CF1': round(CF1*100, 4),
                'val_loss': round(val_loss, 6),
                    } 
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, epoch)

        #============ evaluation info ============#
        val_stats = {"AUC_cls": "{:.4f}".format(AUC_cls*100),
                     "ACC_cls": "{:.4f}".format(ACC_cls*100),
                     "MAP": "{:.4f}".format(MAP*100),
                     "CF1": "{:.4f}".format(CF1*100),
                     "loss": "{:.6f}".format(val_loss),
        }
        stop_training = False
        
        if utils.is_main_process(): 
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            'epoch': epoch,
                        }             
            with open(os.path.join(log_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if config['schedular']['sched'] != 'cosine_in_step':
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
            else:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]["lr"],
                    'config': config,
                    'epoch': epoch,
                }                    
            if (epoch % args.model_save_epoch == 0 and epoch!=0):
                torch.save(save_obj, os.path.join(log_dir, 'checkpoint_%02d.pth'%epoch)) 
            current_score = AUC_cls if np.isfinite(AUC_cls) else ACC_cls
            if current_score > best:
                torch.save(save_obj, os.path.join(log_dir, 'checkpoint_best.pth')) 
                best = current_score
                best_epoch = epoch 
            if np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= early_stop_patience:
                stop_training = True
                if args.log:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best val loss {best_val_loss:.6f} at epoch {best_val_loss_epoch}."
                    )

        if args.distributed:
            stop_tensor = torch.tensor(int(stop_training), device=device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            safe_barrier()
            break

        if config['schedular']['sched'] != 'cosine_in_step':
            lr_scheduler.step(epoch+warmup_steps+1)  
        safe_barrier() 

    if utils.is_main_process():
        torch.save(save_obj, os.path.join(log_dir, 'checkpoint_%02d.pth'%epoch))   
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.log:
        print('Training time {}'.format(total_time_str)) 
        with open(os.path.join(log_dir, "log.txt"),"a") as f:
            f.write("best epoch: {}, Training time: {}".format(best_epoch, total_time_str))    
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23459', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=20)
    parser.add_argument('--token_momentum', default=False, action='store_true')
    parser.add_argument('--data_root', default=None, type=str)
    parser.add_argument('--train_file', default=None, type=str, help='comma-separated json paths')
    parser.add_argument('--val_file', default=None, type=str, help='comma-separated json paths')
    parser.add_argument('--train_sources', default=None, type=str, help='comma-separated sources')
    parser.add_argument('--val_sources', default=None, type=str, help='comma-separated sources')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    apply_config_overrides(config, args)

    # main(args, config)
    if args.launcher == 'none':
        args.launcher = 'pytorch'
        main_worker(0, args, config)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, config))