import warnings
warnings.filterwarnings("ignore")

import argparse
import copy
import csv
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    try:
        from ruamel import yaml  # type: ignore
    except ModuleNotFoundError:
        import yaml  # type: ignore
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils
from dataset.dataset import DGM4_Dataset
from models import box_ops
from models.HAMMER import HAMMER
from models.vit import interpolate_pos_embed
from tools.env import init_dist
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label


FOUR_SET_NAMES = ["guardian", "bbc", "usa_today", "washington_post"]


def text_input_adjust(text_input, fake_word_pos, device):
    # input_ids adaptation
    input_ids_remove_sep = [x[:-1] for x in text_input.input_ids]
    maxlen = max(len(x) for x in text_input.input_ids) - 1
    input_ids_remove_sep_pad = [
        x + [0] * (maxlen - len(x)) for x in input_ids_remove_sep
    ]
    text_input.input_ids = torch.LongTensor(input_ids_remove_sep_pad).to(device)

    # attention_mask adaptation
    attention_mask_remove_sep = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_sep_pad = [
        x + [0] * (maxlen - len(x)) for x in attention_mask_remove_sep
    ]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_sep_pad).to(device)

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    for idx in range(len(fake_word_pos)):
        fake_token_pos = []
        fake_word_pos_decimal = np.where(fake_word_pos[idx].numpy() == 1)[0].tolist()

        subword_idx = text_input.word_ids(idx)
        subword_idx_rm_cls_sep = np.array(subword_idx[1:-1])

        for pos in fake_word_pos_decimal:
            fake_token_pos.extend(np.where(subword_idx_rm_cls_sep == pos)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def _to_percent_str(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
        return ""
    return f"{float(value) * 100.0:.4f}"


def _compute_eer(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    try:
        return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))
    except Exception:
        return np.nan


def _resolve_checkpoint_path(args, checkpoint_log_num):
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        return args.checkpoint

    ckpt_dir = os.path.join(args.output_dir, checkpoint_log_num)
    tried = []
    if args.test_epoch == "best":
        tried.append(os.path.join(ckpt_dir, "checkpoint_best.pth"))
    else:
        tried.append(os.path.join(ckpt_dir, f"checkpoint_{args.test_epoch}.pth"))
        if str(args.test_epoch).isdigit():
            tried.append(os.path.join(ckpt_dir, f"checkpoint_{int(args.test_epoch):02d}.pth"))

    for path in tried:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "No checkpoint found. Tried:\n" + "\n".join(f"  - {x}" for x in tried)
    )


def _create_val_loader(config, ann_files, args):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    test_transform = transforms.Compose([
        transforms.Resize((config["image_res"], config["image_res"]), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = DGM4_Dataset(
        config=config,
        ann_file=ann_files,
        transform=test_transform,
        max_words=config["max_words"],
        is_train=False,
    )

    sampler = None
    if args.distributed and dist.is_available() and dist.is_initialized():
        sampler = torch.utils.data.DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        )

    loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size_val"],
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        sampler=sampler,
        shuffle=False,
        drop_last=False,
    )
    return loader


@torch.no_grad()
def evaluate_one_split(args, model, tokenizer, device, config, ann_files):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"
    print_freq = 200

    y_true, y_pred, iou_pred, iou_50, iou_75, iou_95 = [], [], [], [], [], []
    y_true_multicls, y_pred_multicls = [], []
    multicls_codes = torch.tensor(
        [
            [0, 0, 0, 0],  # orig
            [1, 0, 0, 0],  # face_swap
            [0, 1, 0, 0],  # face_attribute
            [0, 0, 1, 0],  # text_swap
            [0, 0, 0, 1],  # text_attribute
            [1, 0, 1, 0],  # face_swap&text_swap
            [1, 0, 0, 1],  # face_swap&text_attribute
            [0, 1, 1, 0],  # face_attribute&text_swap
            [0, 1, 0, 1],  # face_attribute&text_attribute
        ],
        device=device,
        dtype=torch.long,
    )

    tp_all = 0
    tn_all = 0
    fp_all = 0
    fn_all = 0

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    val_loader = _create_val_loader(config, ann_files, args)
    print("Computing features for evaluation...")
    start_time = time.time()

    for image, label, text, fake_image_box, fake_word_pos, _, _ in metric_logger.log_every(
        args, val_loader, print_freq, header
    ):
        image = image.to(device, non_blocking=True)

        text_input = tokenizer(
            text,
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

        logits_real_fake, logits_multicls, output_coord, logits_tok = model(
            image, label, text_input, fake_image_box, fake_token_pos, is_train=False
        )

        # real/fake metrics
        cls_label = torch.ones(len(label), dtype=torch.long).to(image.device)
        real_label_pos = np.where(np.array(label) == "orig")[0].tolist()
        cls_label[real_label_pos] = 0

        y_pred.extend(F.softmax(logits_real_fake, dim=1)[:, 1].cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        # multi-label metrics
        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)

        # 9-way multiclass metrics derived from 4-d code
        log_p1 = F.logsigmoid(logits_multicls).unsqueeze(1)
        log_p0 = F.logsigmoid(-logits_multicls).unsqueeze(1)
        code_float = multicls_codes.unsqueeze(0).float()
        class_logprob = (code_float * log_p1 + (1 - code_float) * log_p0).sum(-1)
        pred_multicls = class_logprob.argmax(dim=1)

        target_match = (target.unsqueeze(1) == multicls_codes.unsqueeze(0)).all(-1)
        true_multicls = target_match.float().argmax(dim=1)

        y_pred_multicls.extend(pred_multicls.cpu().tolist())
        y_true_multicls.extend(true_multicls.cpu().tolist())

        # bbox metrics
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)
        iou, _ = box_ops.box_iou(boxes1, boxes2.to(device), test=True)

        iou_pred.extend(iou.cpu().tolist())
        iou_50.extend((iou > 0.5).long().cpu().tolist())
        iou_75.extend((iou > 0.75).long().cpu().tolist())
        iou_95.extend((iou > 0.95).long().cpu().tolist())

        # token metrics
        token_label = text_input.attention_mask[:, 1:].clone()
        token_label[token_label == 0] = -100
        token_label[token_label == 1] = 0
        for batch_idx in range(len(fake_token_pos)):
            for pos in fake_token_pos[batch_idx]:
                token_label[batch_idx, pos] = 1

        logits_tok_reshape = logits_tok.view(-1, 2)
        logits_tok_pred = logits_tok_reshape.argmax(1)
        token_label_reshape = token_label.view(-1)

        tp_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
        tn_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
        fp_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
        fn_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    if args.log:
        print(f"Evaluation time {elapsed}")

    # real/fake cls
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(np.unique(y_true)) < 2:
        auc_cls = np.nan
        eer_cls = np.nan
    else:
        auc_cls = roc_auc_score(y_true, y_pred)
        eer_cls = _compute_eer(y_true, y_pred)

    cls_threshold = float(config.get("cls_threshold", 0.5))
    pred_label = (y_pred >= cls_threshold).astype(np.int64)
    acc_cls = float((pred_label == y_true).mean())
    err_cls = 1.0 - acc_cls

    tp = int(np.sum((pred_label == 1) & (y_true == 1)))
    tn = int(np.sum((pred_label == 0) & (y_true == 0)))
    fp = int(np.sum((pred_label == 1) & (y_true == 0)))
    fn = int(np.sum((pred_label == 0) & (y_true == 1)))
    precision_cls = _safe_div(tp, tp + fp)
    recall_cls = _safe_div(tp, tp + fn)
    f1_cls = _safe_div(2 * precision_cls * recall_cls, precision_cls + recall_cls)
    specificity_cls = _safe_div(tn, tn + fp)
    bacc_cls = 0.5 * (recall_cls + specificity_cls)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_cls = float((tp * tn - fp * fn) / mcc_den) if mcc_den > 0 else 0.0

    # multi-label cls
    map_value = multi_label_meter.value()
    map_score = float(map_value.mean().item()) if torch.is_tensor(map_value) else 0.0
    op, or_, of1, cp, cr, cf1 = multi_label_meter.overall()

    # multi-class cls (9 classes represented by 4-d code)
    y_true_multicls = np.array(y_true_multicls, dtype=np.int64)
    y_pred_multicls = np.array(y_pred_multicls, dtype=np.int64)
    acc_multicls = float((y_true_multicls == y_pred_multicls).mean())
    macro_f1_multicls = f1_score(y_true_multicls, y_pred_multicls, average="macro", zero_division=0)
    weighted_f1_multicls = f1_score(
        y_true_multicls, y_pred_multicls, average="weighted", zero_division=0
    )

    # bbox cls
    iou_score = _safe_div(sum(iou_pred), len(iou_pred))
    iou_acc_50 = _safe_div(sum(iou_50), len(iou_50))
    iou_acc_75 = _safe_div(sum(iou_75), len(iou_75))
    iou_acc_95 = _safe_div(sum(iou_95), len(iou_95))

    # token cls
    acc_tok = _safe_div(tp_all + tn_all, tp_all + tn_all + fp_all + fn_all)
    precision_tok = _safe_div(tp_all, tp_all + fp_all)
    recall_tok = _safe_div(tp_all, tp_all + fn_all)
    f1_tok = _safe_div(2 * precision_tok * recall_tok, precision_tok + recall_tok)

    raw_stats = {
        "AUC_cls": auc_cls,
        "ACC_cls": acc_cls,
        "ERR_cls": err_cls,
        "EER_cls": eer_cls,
        "Precision_cls": precision_cls,
        "Recall_cls": recall_cls,
        "F1_cls": f1_cls,
        "MCC_cls": mcc_cls,
        "Specificity_cls": specificity_cls,
        "BACC_cls": bacc_cls,
        "ACC_multicls": acc_multicls,
        "Macro_F1_multicls": macro_f1_multicls,
        "Weighted_F1_multicls": weighted_f1_multicls,
        "MAP": map_score,
        "OP": op,
        "OR": or_,
        "OF1": of1,
        "CP": cp,
        "CR": cr,
        "CF1": cf1,
        "IOU_score": iou_score,
        "IOU_ACC_50": iou_acc_50,
        "IOU_ACC_75": iou_acc_75,
        "IOU_ACC_95": iou_acc_95,
        "ACC_tok": acc_tok,
        "Precision_tok": precision_tok,
        "Recall_tok": recall_tok,
        "F1_tok": f1_tok,
    }
    return raw_stats


def _save_one_result(output_dir, log_num, raw_stats):
    eval_dir = os.path.join(output_dir, log_num, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    result_path = os.path.join(eval_dir, "results_all.txt")

    val_stats = {f"val_{k}": _to_percent_str(v) for k, v in raw_stats.items()}
    val_stats["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")

    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(val_stats, ensure_ascii=False) + "\n")

    return val_stats, result_path


def _print_core_metrics(tag, val_stats):
    print(f"\n===== {tag} =====")
    print(f"val_AUC_cls             : {val_stats.get('val_AUC_cls', '')}")
    print(f"val_EER_cls             : {val_stats.get('val_EER_cls', '')}")
    print(f"val_ACC_cls             : {val_stats.get('val_ACC_cls', '')}")
    print(f"val_F1_cls              : {val_stats.get('val_F1_cls', '')}")
    print(f"val_MCC_cls             : {val_stats.get('val_MCC_cls', '')}")
    print(f"val_ACC_multicls        : {val_stats.get('val_ACC_multicls', '')}")
    print(f"val_Macro_F1_multicls   : {val_stats.get('val_Macro_F1_multicls', '')}")
    print(f"val_Weighted_F1_multicls: {val_stats.get('val_Weighted_F1_multicls', '')}")
    print(f"val_MAP                 : {val_stats.get('val_MAP', '')}")
    print(f"val_CF1                 : {val_stats.get('val_CF1', '')}")
    print(f"val_OF1                 : {val_stats.get('val_OF1', '')}")


def _write_four_set_summary(output_dir, base_log_num, rows, summary_csv=None):
    summary_cols = [
        "dataset",
        "val_AUC_cls", "val_EER_cls", "val_ACC_cls", "val_F1_cls", "val_MCC_cls",
        "val_ACC_multicls", "val_Macro_F1_multicls", "val_Weighted_F1_multicls",
        "val_MAP", "val_CF1", "val_OF1",
    ]
    if summary_csv is None:
        summary_csv = os.path.join(output_dir, f"{base_log_num}_4sets_summary.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary_csv


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        # ruamel.yaml package API
        if hasattr(yaml, "YAML"):
            return yaml.YAML(typ="safe").load(f)

        # ruamel_yaml / PyYAML style API
        loader = getattr(yaml, "Loader", None) or getattr(yaml, "SafeLoader", None)
        if loader is not None:
            return yaml.load(f, Loader=loader)
        return yaml.safe_load(f)


def _prepare_runtime(args):
    args.log = True
    args.distributed = False

    if args.launcher != "none" and torch.cuda.is_available():
        if args.rank < 0:
            args.rank = 0
        args.gpu = 0
        init_dist(args)
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def _build_model_and_tokenizer(args, config, device, checkpoint_path):
    try:
        tokenizer = BertTokenizerFast.from_pretrained(
            args.text_encoder,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load tokenizer. For offline mode, pass a local --text_encoder path and "
            "enable --local_files_only."
        ) from exc

    model = HAMMER(
        args=args,
        config=config,
        text_encoder=args.text_encoder,
        tokenizer=tokenizer,
        init_deit=(not args.no_deit_init),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    if "visual_encoder.pos_embed" in state_dict:
        pos_embed_reshaped = interpolate_pos_embed(state_dict["visual_encoder.pos_embed"], model.visual_encoder)
        state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped

    msg = model.load_state_dict(state_dict, strict=False)
    if args.log:
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(msg)

    return model, tokenizer


def run_single_eval(args, model, tokenizer, device, config):
    if not args.log_num:
        raise ValueError("--log_num is required in single-set mode.")

    raw_stats = evaluate_one_split(args, model, tokenizer, device, config, config["val_file"])
    val_stats, result_path = _save_one_result(args.output_dir, args.log_num, raw_stats)
    _print_core_metrics(args.log_num, val_stats)
    print(f"Saved: {result_path}")


def run_four_set_eval(args, model, tokenizer, device, config):
    if not args.log_num:
        raise ValueError("--log_num is required in 4-set mode.")
    if not args.data_root:
        raise ValueError("--data_root is required in 4-set mode.")

    dataset_names = [x.strip() for x in args.eval_set_names.split(",") if x.strip()]
    if not dataset_names:
        dataset_names = FOUR_SET_NAMES

    rows = []
    for name in dataset_names:
        ann = os.path.join(args.data_root, name, "test.json")
        if not os.path.isfile(ann):
            raise FileNotFoundError(f"Missing test file for {name}: {ann}")

        eval_log = f"{args.log_num}_{name}"
        cfg_this = copy.deepcopy(config)
        cfg_this["val_file"] = [ann]

        raw_stats = evaluate_one_split(args, model, tokenizer, device, cfg_this, [ann])
        val_stats, result_path = _save_one_result(args.output_dir, eval_log, raw_stats)
        _print_core_metrics(eval_log, val_stats)
        print(f"Saved: {result_path}")

        row = {"dataset": name}
        for key in [
            "val_AUC_cls", "val_EER_cls", "val_ACC_cls", "val_F1_cls", "val_MCC_cls",
            "val_ACC_multicls", "val_Macro_F1_multicls", "val_Weighted_F1_multicls",
            "val_MAP", "val_CF1", "val_OF1",
        ]:
            row[key] = val_stats.get(key, "")
        rows.append(row)

    summary_csv = _write_four_set_summary(
        output_dir=args.output_dir,
        base_log_num=args.log_num,
        rows=rows,
        summary_csv=(args.summary_csv or None),
    )
    print(f"Summary CSV saved: {summary_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--test_epoch", default="best", help="best or explicit checkpoint suffix")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--log_num", "-l", type=str, default="")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--local_files_only", default=False, action="store_true")
    parser.add_argument("--no_deit_init", default=False, action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument("--rank", default=-1, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", "--dist-url", default="tcp://127.0.0.1:23459", type=str, dest="dist_url")
    parser.add_argument("--dist_backend", "--dist-backend", default="nccl", type=str, dest="dist_backend")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--token_momentum", default=False, action="store_true")

    # One-shot 4-subset evaluation mode.
    parser.add_argument("--eval_4sets", action="store_true")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--eval_set_names", default="guardian,bbc,usa_today,washington_post")
    parser.add_argument("--summary_csv", default="")

    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    config = _load_config(args.config)

    device = _prepare_runtime(args)
    if args.log:
        print(f"Using device: {device}")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    checkpoint_log_num = args.log_num
    checkpoint_path = _resolve_checkpoint_path(args, checkpoint_log_num)
    model, tokenizer = _build_model_and_tokenizer(args, config, device, checkpoint_path)

    if args.eval_4sets:
        run_four_set_eval(args, model, tokenizer, device, config)
    else:
        run_single_eval(args, model, tokenizer, device, config)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
