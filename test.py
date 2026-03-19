import os
import re
import argparse
import numpy as np
from tqdm import tqdm
import warnings
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoProcessor

# === 导入 Dataset ===
from dgm4Datasets5class import DGM4_Dataset 

warnings.filterwarnings("ignore")

# ==========================================
#             工具函数 (来自你的 train.py)
# ==========================================

def parse_prediction_vector(pred_str):
    text = str(pred_str).upper().strip()
    vec = [0, 0, 0, 0]  # [B, C, D, E]
    
    # 先检查是否为纯 A（真实）
    if re.fullmatch(r'\s*A\s*', text):
        return vec  # 全0 = 真实
    
    # 严格匹配：字母必须独立（前后非字母数字）
    if re.search(r'(?<![A-Z])B(?![A-Z])', text): vec[0] = 1
    if re.search(r'(?<![A-Z])C(?![A-Z])', text): vec[1] = 1
    if re.search(r'(?<![A-Z])D(?![A-Z])', text): vec[2] = 1
    if re.search(r'(?<![A-Z])E(?![A-Z])', text): vec[3] = 1
    
    # 额外安全：若检测到 A 且无其他字母 → 强制设为真实
    if 'A' in text and sum(vec) == 0:
        return [0, 0, 0, 0]
    
    return vec

def _resolve_letter_token_ids(tokenizer, letter):
    candidate_ids = []
    for candidate in (letter, f" {letter}", letter.lower(), f" {letter.lower()}"):
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) == 1:
            candidate_ids.append(token_ids[0])
    if len(candidate_ids) == 0:
        candidate_ids = [tokenizer.encode(letter, add_special_tokens=False)[0]]
    return sorted(set(candidate_ids))

def calculate_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)

def _extract_fake_prob_from_outputs(outputs, letter_token_ids):
    num_steps = len(outputs.scores)
    bsz = outputs.sequences.size(0)
    seq_steps = outputs.sequences[:, 1:1 + num_steps]

    letter_union = set()
    for key in ["A", "B", "C", "D", "E"]:
        letter_union.update(letter_token_ids[key])

    fake_probs = []
    for i in range(bsz):
        step_idx = None
        for t in range(num_steps):
            tok_id = int(seq_steps[i, t].item())
            if tok_id in letter_union:
                step_idx = t
                break
        if step_idx is None:
            step_idx = 0

        step_probs = torch.softmax(outputs.scores[step_idx][i], dim=-1)
        p_a = step_probs[letter_token_ids["A"]].sum()
        p_b = step_probs[letter_token_ids["B"]].sum()
        p_c = step_probs[letter_token_ids["C"]].sum()
        p_d = step_probs[letter_token_ids["D"]].sum()
        p_e = step_probs[letter_token_ids["E"]].sum()
        denom = p_a + p_b + p_c + p_d + p_e
        
        if float(denom.item()) <= 1e-12:
            fake_probs.append(0.5)
        else:
            # Fake的概率 = 1 - P(A)
            fake_probs.append(float((1.0 - (p_a / denom)).item()))

    return np.array(fake_probs, dtype=np.float32)

# ==========================================
#             主测试流程
# ==========================================

def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")
    
    # 1. 自动处理分片并加载模型
    print(f"[*] 正在加载模型和处理器: {args.checkpoint}")
    # HuggingFace 的 from_pretrained 能够自动识别目录下的 .index.json 和分片的 .bin / .safetensors
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True).to(device)
    model.eval()

    # 2. 准备数据集
    CFG = {
        "max_seq_len": 1024,
        "image_size": 768,
        "batch_size": args.batch_size
    }
    print(f"[*] 加载测试数据: {args.test_file}")
    test_dataset = DGM4_Dataset(config=CFG, is_train=False, is_test=True, ann_files=[args.test_file])
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CFG["batch_size"], 
        shuffle=False, 
        num_workers=4, 
        collate_fn=test_dataset.collate_fn
    )

    # 3. 提取所需的 Token IDs
    token_ids = {
        letter: _resolve_letter_token_ids(processor.tokenizer, letter)
        for letter in ["A", "B", "C", "D", "E"]
    }

    all_gts = []
    all_preds = []
    all_fake_scores = []

    # 4. 推理过程
    print("[*] 开始评估推理...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            prompts = batch["prompts"]
            images = batch["images"]
            gt_vectors = batch["muti_ans"].cpu().numpy() # Ground Truth shape [B, 4]

            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                do_rescale=True
            ).to(device)

            # 生成文本并输出 logits (用于计算概率)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20,
                num_beams=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

            # 提取连续值假新闻概率 (0~1)
            p_fake = _extract_fake_prob_from_outputs(outputs, token_ids)
            all_fake_scores.append(p_fake)

            # 解析生成的文本
            generated_texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
            pred_vecs = []
            for txt in generated_texts:
                pred_vecs.append(parse_prediction_vector(txt))
            
            all_preds.append(pred_vecs)
            all_gts.append(gt_vectors)

    # 5. 合并并计算指标
    all_preds = np.concatenate(all_preds, axis=0)        # Shape: [N, 4]
    all_gts = np.concatenate(all_gts, axis=0)            # Shape: [N, 4]
    fake_scores = np.concatenate(all_fake_scores, axis=0) # Shape: [N]

    # ========== 指标计算核心区 ==========

    print("\n" + "="*40)
    print("           DGM4 测试结果汇总")
    print("="*40)

    # 【1】 二分类指标 (基于连续值 fake_scores 计算)
    # Ground Truth 二分类化: 如果 4 个维度(B,C,D,E)全为0则是真(0)，只要有1则是假(1)
    bin_true = (all_gts.sum(axis=1) > 0).astype(int)
    
    # ACC: 利用连续概率通过 0.5 阈值进行判断
    bin_pred_continuous = (fake_scores >= 0.5).astype(int)
    acc_bin = accuracy_score(bin_true, bin_pred_continuous)
    
    # AUC
    auc_score = roc_auc_score(bin_true, fake_scores)
    
    # EER (ERR)
    eer_score = calculate_eer(bin_true, fake_scores)

    print("\n--- 【二分类 (真 vs 假)】 ---")
    print(f"AUC (连续值): {auc_score:.4f}")
    print(f"ERR (EER)   : {eer_score:.4f}")
    print(f"ACC (连续值): {acc_bin:.4f}")


    # 【2】 多分类/多标签指标 (基于解析后的 A/B/C/D/E 预测向量计算)
    # ACC: 多标签下的 exact match accuracy (完全匹配率)
    acc_multi = accuracy_score(all_gts, all_preds)
    
    # Macro F1
    macro_f1 = f1_score(all_gts, all_preds, average='macro', zero_division=0)
    
    # Weighted F1
    weighted_f1 = f1_score(all_gts, all_preds, average='weighted', zero_division=0)

    print("\n--- 【多分类 (四种篡改类型)】 ---")
    print(f"Exact ACC   : {acc_multi:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 你的分片权重目录 (比如 train.py 中保存 best_model 的路径)
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help="HuggingFace 格式的模型保存目录 (存放着 .bin / .safetensors 和 index.json 的文件夹)")
    # 测试集路径
    parser.add_argument('--test_file', type=str, required=True, 
                        help="测试集 json 文件路径")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default="cuda:0")

    args = parser.parse_args()
    evaluate(args)
