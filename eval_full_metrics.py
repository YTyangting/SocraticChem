import json
import re
import argparse
import pandas as pd
import jieba
import torch
from rouge_chinese import Rouge
from bert_score import score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
import os
import logging
import glob
import sys
import torch.multiprocessing as mp
import numpy as np

# --- 配置 ---
jieba.setLogLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLPEvaluator:
    def __init__(self, model_type="bert-base-chinese", device=None, batch_size=32):
        self.rouge = Rouge()
        self.batch_size = batch_size
        self.device = device
        self.bert_model_type = model_type
        print(f"   [Worker {self.device}] Evaluator initialized.", flush=True)

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'```\w*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def compute_lexical_metrics(self, preds, refs):
        """计算 ROUGE 和 BLEU (语料库级别)"""
        preds_tokens, refs_tokens = [], []
        preds_spaced, refs_spaced = [], []
        
        for p, r in zip(preds, refs):
            p_cut = list(jieba.cut(p)) if p.strip() else ["空"]
            r_cut = list(jieba.cut(r)) if r.strip() else ["空"]
            preds_tokens.append(p_cut); refs_tokens.append([r_cut])
            preds_spaced.append(' '.join(p_cut)); refs_spaced.append(' '.join(r_cut))

        metrics = {}
        # ROUGE
        try:
            rouge_scores = self.rouge.get_scores(preds_spaced, refs_spaced, avg=True)
            metrics["ROUGE-1"] = rouge_scores['rouge-1']['f'] * 100
            metrics["ROUGE-2"] = rouge_scores['rouge-2']['f'] * 100
            metrics["ROUGE-L"] = rouge_scores['rouge-l']['f'] * 100
        except: 
            metrics.update({"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0})

        # BLEU
        try:
            smooth = SmoothingFunction().method1
            metrics["BLEU-4"] = corpus_bleu(refs_tokens, preds_tokens, smoothing_function=smooth) * 100
        except: metrics["BLEU-4"] = 0.0
            
        return metrics

    def compute_bertscore(self, preds, refs):
        """
        计算 BERTScore
        返回: (平均分, 分数列表)
        """
        if not preds: return 0.0, []
        try:
            # P, R, F1 (F1 是一个 tensor，包含每一个样本的分数)
            P, R, F1 = score(preds, refs, lang="zh", verbose=False, 
                             device=self.device, batch_size=self.batch_size,
                             model_type=self.bert_model_type)
            
            mean_score = F1.mean().item() * 100
            # 将 Tensor 转为 Python list，保留2位小数
            instance_scores = [round(s.item() * 100, 2) for s in F1]
            
            return mean_score, instance_scores
        except Exception as e:
            logger.error(f"BERTScore error on {self.device}: {e}")
            return 0.0, [0.0] * len(preds)

    def process_file(self, file_path, output_csv):
        print(f"⚙️  [GPU {self.device}] Processing {os.path.basename(file_path)}...", flush=True)
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    l = line.strip()
                    if l: data.append(json.loads(l))
                except: continue
        
        if not data: return

        df = pd.DataFrame(data)
        
        pred_col = next((c for c in ['generated_output', 'response', 'generated_response'] if c in df.columns), None)
        ref_col = next((c for c in ['ground_truth', 'reference', 'label'] if c in df.columns), None)

        if not pred_col or not ref_col: return

        raw_preds = df[pred_col].fillna("").astype(str).tolist()
        raw_refs = df[ref_col].fillna("").astype(str).tolist()
        
        clean_preds = [self.clean_text(p) for p in raw_preds]
        clean_refs = [self.clean_text(r) for r in raw_refs]

        # 1. 计算整体指标
        lex_metrics = self.compute_lexical_metrics(clean_preds, clean_refs)
        
        # 2. 计算 BERTScore (获取平均分 和 每一行的分数)
        bs_mean, bs_list = self.compute_bertscore(clean_preds, clean_refs)

        # 3. 打印报告
        report = (
            f"\n" + "="*45 + "\n"
            f"📄 Evaluation Report: {os.path.basename(file_path)}\n"
            f"💻 Worker: GPU {self.device}\n"
            f"📊 Sample Count: {len(df)}\n"
            f"="*45 + "\n"
            f"🔹 Cleaned BLEU-4:   {lex_metrics['BLEU-4']:.2f}\n"
            f"-" * 45 + "\n"
            f"🔹 Cleaned ROUGE-1:  {lex_metrics['ROUGE-1']:.2f}\n"
            f"🔹 Cleaned ROUGE-2:  {lex_metrics['ROUGE-2']:.2f}\n"
            f"🔹 Cleaned ROUGE-L:  {lex_metrics['ROUGE-L']:.2f}\n"
            f"-" * 45 + "\n"
            f"🔥 BERTScore (F1):   {bs_mean:.2f}\n"
            f"="*45 + "\n"
        )
        print(report, flush=True)

        # --- 4. 保存 CSV (包含每一行的 BERTScore) ---
        df['cleaned_pred'] = clean_preds
        df['cleaned_ref'] = clean_refs
        df['bert_score'] = bs_list  # <--- 新增列：每一行的具体分数
        df.to_csv(output_csv, index=False)
        
        # --- 5. 保存汇总 JSON (保存你在屏幕上看到的总分) ---
        summary_data = {
            "filename": os.path.basename(file_path),
            "sample_count": len(df),
            "metrics": {
                "BLEU-4": lex_metrics['BLEU-4'],
                "ROUGE-1": lex_metrics['ROUGE-1'],
                "ROUGE-2": lex_metrics['ROUGE-2'],
                "ROUGE-L": lex_metrics['ROUGE-L'],
                "BERTScore_Mean": bs_mean
            }
        }
        
        summary_path = file_path.replace(".jsonl", "_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
            
        print(f"💾 Saved summary to: {summary_path}", flush=True)

def worker_process(gpu_id, file_list, batch_size):
    device_str = f"cuda:{gpu_id}"
    jieba.initialize()
    
    try:
        evaluator = NLPEvaluator(device=device_str, batch_size=batch_size)
    except Exception as e:
        print(f"❌ Worker {gpu_id} failed: {e}", flush=True)
        return

    for f_path in file_list:
        out_csv = f_path.replace(".jsonl", "_full_metrics.csv")
        # 如果你想强制重新运行以生成新的汇总文件，可以注释掉下面这两行
        # if os.path.exists(out_csv):
        #      print(f"⏩ [GPU {gpu_id}] Skipping {os.path.basename(f_path)}", flush=True)
        #      continue
        evaluator.process_file(f_path, out_csv)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpus", type=int, nargs='+', help="Specific GPU IDs")
    args = parser.parse_args()

    print("🛠️  Initializing Jieba in Main Process...", flush=True)
    jieba.initialize()
    list(jieba.cut("Test"))
    print("✅ Jieba initialized.", flush=True)

    if args.gpus:
        available_gpus = args.gpus
    else:
        num_gpus = torch.cuda.device_count()
        available_gpus = list(range(num_gpus))
    
    if not available_gpus:
        print("❌ No GPUs found!")
        return

    print(f"🚀 Detected {len(available_gpus)} GPUs: {available_gpus}", flush=True)

    files = glob.glob(os.path.join(args.data_dir, "*.jsonl"))
    if not files:
        print("❌ No files found.")
        return
    print(f"📂 Found {len(files)} files.", flush=True)

    file_chunks = np.array_split(files, len(available_gpus))
    mp.set_start_method('spawn', force=True)
    
    processes = []
    print("⚡ Starting Parallel Evaluation...", flush=True)
    
    for i, gpu_id in enumerate(available_gpus):
        chunk = file_chunks[i].tolist()
        if not chunk: continue 
        p = mp.Process(target=worker_process, args=(gpu_id, chunk, args.batch_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("🎉 All tasks finished!")

if __name__ == "__main__":
    main()