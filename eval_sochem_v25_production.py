import json
import re
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config

# --- NLP Metrics Dependencies ---
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Ensure punkt is downloaded for tokenization if needed, though we'll use simple split for speed
    # nltk.download('punkt') 
except ImportError:
    print("⚠️ 请安装 NLP 依赖: pip install rouge-score nltk")

# ===========================
# 0. Global Configuration
# ===========================
MAX_WORKERS = 10           # Concurrent threads
USE_LLM_JUDGE = True       # Enable LLM judging
OPENAI_API_KEY = Config.OPENAI_API_KEY
OPENAI_BASE_URL = Config.OPENAI_BASE_URL

# Strategies that TRIGGER the Safety Red-Line Check
INTERCEPTION_STRATEGIES = {
    "PREDICTIVE_QUESTIONING", 
    "SAFETY_INTERVENTION", 
    "CORRECTIVE_FEEDBACK",
    "STOP_AND_FIX"
}

if USE_LLM_JUDGE:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ===========================
# 1. 文本评估器 (NLP Metrics) - NEW
# ===========================
class TextEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

    def compute_metrics(self, ref, pred):
        """计算 Rouge-1, Rouge-2, Rouge-L, BLEU-4"""
        if not ref or not pred:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu4": 0.0}

        # 1. Rouge Calculation
        # 对于中文，建议分词后用空格连接，这里假设输入已足够好或直接基于字符
        # 简单处理：将每个字作为 token (Character-level for Chinese)
        ref_tokens = list(ref.strip())
        pred_tokens = list(pred.strip())
        
        ref_str = " ".join(ref_tokens)
        pred_str = " ".join(pred_tokens)

        scores = self.scorer.score(ref_str, pred_str)
        
        # 2. BLEU-4 Calculation
        try:
            bleu = sentence_bleu([ref_tokens], pred_tokens, 
                                 weights=(0.25, 0.25, 0.25, 0.25), 
                                 smoothing_function=self.smooth)
        except:
            bleu = 0.0

        return {
            "rouge1": scores['rouge1'].fmeasure * 100,
            "rouge2": scores['rouge2'].fmeasure * 100,
            "rougeL": scores['rougeL'].fmeasure * 100,
            "bleu4": bleu * 100
        }

# ===========================
# 1. Robust Parser (Updated for your data)
# ===========================
class Parser:
    @staticmethod
    def extract_tag_content(xml, tag):
        """
        Extracts content between <tag> and </tag>.
        Handles newlines (re.DOTALL) and case insensitivity.
        """
        if not isinstance(xml, str): return ""
        # Match <tag ...>content</tag> or <tag>content</tag>
        pattern = f"<{tag}.*?>(.*?)</{tag}>"
        match = re.search(pattern, xml, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def normalize_status(text):
        """
        Normalize status symbols from your data.
        Input: " ❌ " -> 0
        Input: "PASS" -> 1
        """
        if not text: return -1
        t = text.upper()
        if '✅' in t or 'PASS' in t or 'TRUE' in t: return 1
        if '❌' in t or 'FAIL' in t or 'FALSE' in t or '⚠️' in t: return 0
        return -1

    @staticmethod
    def parse_full_entry(entry):
        # 1. Map fields correctly based on your provided JSON
        raw_input = entry.get('input', '')
        raw_pred = entry.get('generated_output', entry.get('prediction', '')) 
        raw_gt = entry.get('ground_truth', entry.get('output', '')) # Priority to 'ground_truth'

        # 2. Extract Basic Info
        student_say = Parser.extract_tag_content(raw_input, "Speak")
        obs = Parser.extract_tag_content(raw_input, "Observation")
        
        # 3. Extract Node (Flattened search for <CurrentNode>)
        pred_node = Parser.extract_tag_content(raw_pred, "CurrentNode")
        gt_node = Parser.extract_tag_content(raw_gt, "CurrentNode")

        # 4. Extract Status (Flattened search for <Result>)
        pred_status_raw = Parser.extract_tag_content(raw_pred, "Result")
        gt_status_raw = Parser.extract_tag_content(raw_gt, "Result")
        
        # 5. Extract Details for Judges
        pred_checklist = Parser.extract_tag_content(raw_pred, "CheckList")
        gt_checklist = Parser.extract_tag_content(raw_gt, "CheckList")
        
        gt_strategy = Parser.extract_tag_content(raw_gt, "Strategy")
        pred_response = Parser.extract_tag_content(raw_pred, "Response")

        # [新增] 提取 GT Response 用于 NLP 对比
        gt_response = Parser.extract_tag_content(raw_gt, "Response")

        return {
            "student_say": student_say,
            "observation": obs,
            "pred_node": pred_node,
            "gt_node": gt_node,
            "pred_status_val": Parser.normalize_status(pred_status_raw),
            "gt_status_val": Parser.normalize_status(gt_status_raw),
            "pred_checklist": pred_checklist,
            "gt_checklist": gt_checklist,
            "gt_strategy": gt_strategy,
            "pred_response": pred_response,
            "gt_response": gt_response,  # [新增]
            "raw_input_context": raw_input
        }

# ===========================
# 2. Hard Logic Evaluators
# ===========================
class LogicEvaluator:
    @staticmethod
    def eval_general_flow(data):
        """Module A: General Flow Accuracy"""
        row = {}
        
        # 1. Node Match (String Exact Match)
        row['node_match'] = (data['pred_node'] == data['gt_node'])
        
        # 2. Status Match
        p_stat = data['pred_status_val']
        g_stat = data['gt_status_val']
        
        if g_stat != -1:
            row['status_match'] = (p_stat == g_stat)
            
            # Classification
            if g_stat == 0 and p_stat == 0: row['capture_type'] = "Hit"          # Correctly caught error
            elif g_stat == 0 and p_stat == 1: row['capture_type'] = "Miss"         # Safety Risk!
            elif g_stat == 1 and p_stat == 0: row['capture_type'] = "FalseAlarm"   # Annoying
            elif g_stat == 1 and p_stat == 1: row['capture_type'] = "CorrectPass"  # Normal flow
            else: row['capture_type'] = "Unknown"
        else:
            row['status_match'] = None
            row['capture_type'] = "Unknown"
            
        return row

    @staticmethod
    def eval_safety_red_line(data):
        """Module B: Safety Red-Line Check"""
        row = {}
        
        # Trigger: GT is FAIL OR Strategy implies intervention
        is_red_line_case = (data['gt_status_val'] == 0) or \
                           (data['gt_strategy'] in INTERCEPTION_STRATEGIES)
        
        row['is_red_line_case'] = is_red_line_case
        
        if not is_red_line_case:
            return row 

        # 1. Hard Interception (Must say FAIL/❌)
        row['redline_intercept'] = (data['pred_status_val'] == 0)

        # 2. Node Lock (Must NOT move to next node)
        row['redline_node_lock'] = (data['pred_node'] == data['gt_node'])

        # Initialize phrasing score (calculated later via LLM)
        row['redline_phrasing_score'] = 0.0
        row['redline_final_pass'] = False
        
        return row

# ===========================
# 3. LLM Judges
# ===========================
class LLMJudge:
    @staticmethod
    def eval_error_fidelity(data):
        """Does the model blame the correct root cause?"""
        prompt = f"""
        [Chemistry Lab Error Analysis]
        Observation: "{data['observation']}"
        GT Error Reason: "{data['gt_checklist']}"
        Model Reason: "{data['pred_checklist']}"
        
        Task: Do they identify the SAME root cause?
        - Score 1.0: Yes, essentially the same reason.
        - Score 0.0: No, model hallucinated a different error.
        
        Output JSON: {{"score": <float>}}
        """
        try:
            res = client.chat.completions.create(
                model="gpt-4o", messages=[{"role":"user","content":prompt}], 
                temperature=0.0, response_format={"type": "json_object"}
            )
            return json.loads(res.choices[0].message.content).get('score', 0.0)
        except: return 0.0

    @staticmethod
    def eval_socratic_safety_phrasing(data):
        """Checks for Predictive Questioning in Safety Cases"""
        prompt = f"""
        [Safety Response Compliance Check]
        Scenario: Chemistry Lab Error Intervention.
        Student Input: "{data['student_say']}"
        Teacher Response: "{data['pred_response']}"
        
        CRITERIA (Socratic Safety):
        1. **Predictive Questioning**: Does it ask "What happens if...?" or guide student to predict consequence?
        2. **No Direct Telling**: Does it avoid just saying "You are wrong"?
        
        Score (0-10):
        - 10: Excellent predictive question.
        - 0: Direct command or Statement.
        
        Output JSON: {{"score": <int>}}
        """
        try:
            res = client.chat.completions.create(
                model="gpt-4o", messages=[{"role":"user","content":prompt}], 
                temperature=0.0, response_format={"type": "json_object"}
            )
            return json.loads(res.choices[0].message.content).get('score', 0)
        except: return 0

    @staticmethod
    def eval_general_pedagogy(data, metrics):
        """General quality for normal interactions"""
        if not (metrics.get('node_match') and metrics.get('status_match')): return 0
        prompt = f"""
        Evaluate Socratic Method (0-10):
        Context: Student="{data['student_say']}"
        Response: "{data['pred_response']}"
        
        Criteria:
        1. Guide rather than Tell?
        2. Encouraging tone?
        Output JSON: {{"score": <int>}}
        """
        try:
            res = client.chat.completions.create(
                model="gpt-4o", messages=[{"role":"user","content":prompt}], 
                temperature=0.0, response_format={"type": "json_object"}
            )
            return json.loads(res.choices[0].message.content).get('score', 0)
        except: return 0

# ===========================
# 4. Main Pipeline
# ===========================

text_evaluator = TextEvaluator() # [修正] 初始化 NLP 评估器
def process_single_line(line):
    try:
        entry = json.loads(line)
        data = Parser.parse_full_entry(entry)
        
        # --- 1. Hard Logic ---
        flow_metrics = LogicEvaluator.eval_general_flow(data)
        safety_metrics = LogicEvaluator.eval_safety_red_line(data)
        metrics = {**flow_metrics, **safety_metrics}

        # --- 2. NLP Metrics (新增调用) ---
        # 无论逻辑对错，都计算文本相似度
        nlp_metrics = text_evaluator.compute_metrics(data['gt_response'], data['pred_response'])
        metrics.update(nlp_metrics) # 合并 ROUGE/BLEU 分数到结果中
        
        # --- 2. AI Judges ---
        if USE_LLM_JUDGE:
            # A. Safety Phrasing (Only if intercepted & locked)
            if metrics.get('is_red_line_case'):
                if metrics['redline_intercept'] and metrics['redline_node_lock']:
                    metrics['redline_phrasing_score'] = LLMJudge.eval_socratic_safety_phrasing(data)
                
                # Final strict pass check
                metrics['redline_final_pass'] = (
                    metrics['redline_intercept'] and 
                    metrics['redline_node_lock'] and 
                    (metrics['redline_phrasing_score'] >= 6.0)
                )
            
            # B. Error Fidelity (For Hit cases)
            if metrics.get('capture_type') == 'Hit':
                metrics['error_reason_score'] = LLMJudge.eval_error_fidelity(data)
            
            # C. General Pedagogy (Non-safety cases)
            if not metrics.get('is_red_line_case'):
                metrics['pedagogy_score'] = LLMJudge.eval_general_pedagogy(data, metrics)
            
        return {**data, **metrics}
    except Exception as e:
        return None

def generate_report(df, filename):
    print(f"\n" + "="*60)
    print(f"📊 Comprehensive Evaluation Report: {filename}")
    print("="*60)
    
    # 1. Flow
    print(f"\n[1] General Flow")
    print(f"   🎯 Node Accuracy:            {df['node_match'].mean():.2%}")
    print(f"   ⚖️ Status Accuracy:          {df['status_match'].mean():.2%}")

    # 2. Safety
    red_df = df[df['is_red_line_case'] == True]
    print(f"\n[2] 🚨 Safety Red-Line Compliance")
    print(f"   Total Safety Cases:          {len(red_df)} / {len(df)}")
    
    if len(red_df) > 0:
        intercept = red_df['redline_intercept'].mean()
        node_lock = red_df['redline_node_lock'].mean()
        final_pass = red_df['redline_final_pass'].mean()
        
        print(f"   ❌ Interception Rate:        {intercept:.2%} (Must say FAIL)")
        print(f"   🔒 Node Lock Rate:           {node_lock:.2%} (Must stay on error node)")
        print(f"   🏆 FINAL Safety Pass Rate:   {final_pass:.2%}")

        # Show a bad case if exists
        failures = red_df[red_df['redline_final_pass'] == False]
        if not failures.empty:
            c = failures.iloc[0]
            print(f"\n   ⚠️ Sample Failure:")
            print(f"      Student: {c['student_say'][:50]}...")
            print(f"      Reason: Intercept={c['redline_intercept']}, Lock={c['redline_node_lock']}")

    # 3. Fidelity (修复了不显示的问题)
    hit_cases = df[df['capture_type'] == 'Hit']
    print(f"\n[3] 错误分析保真度 (Fidelity)")
    print(f"   🎯 成功拦截样本数:            {len(hit_cases)}")
    if len(hit_cases) > 0 and 'error_reason_score' in df:
        print(f"   🧠 归因准确度 (Reason Acc):   {hit_cases['error_reason_score'].mean():.2f} / 1.0")
    else:
        print(f"   (无成功拦截样本，无法评估归因准确度)")

    # 4. NLP Metrics (新增打印) 
    print(f"\n[4] 文本生成质量 (NLP Metrics)")
    if 'rouge1' in df.columns:
        print(f"   📝 ROUGE-1: {df['rouge1'].mean():.2f}")
        print(f"   📝 ROUGE-2: {df['rouge2'].mean():.2f}")
        print(f"   📝 ROUGE-L: {df['rougeL'].mean():.2f}")
        print(f"   📝 BLEU-4:  {df['bleu4'].mean():.2f}")
    else:
        print(f"   ⚠️ NLP 数据缺失 (Keys not found in DataFrame)")

    # 3. Quality
    print(f"\n[3] Quality Metrics")
    if 'pedagogy_score' in df:
        ped_df = df[df['pedagogy_score'] > 0]
        score = ped_df['pedagogy_score'].mean() if not ped_df.empty else 0
        print(f"   🎓 Socratic Score:           {score:.2f} / 10")

    print("-" * 60 + "\n")

def main():
    files = glob.glob("/home/yjh/socChem_final/dis_test_result/*.jsonl") # Update path if needed
    if not files:
        print("❌ No files found in data/")
        return

    print(f"🚀 Starting Evaluation (Workers: {MAX_WORKERS})")
    
    for f_path in files:
        # --- 新增逻辑开始 ---
        # 1. 先确定输出文件的路径
        out_csv = f_path.replace(".jsonl", "_full_eval.csv")
        
        # 2. 判断文件是否存在
        if os.path.exists(out_csv):
            print(f"⏩ Skipping {os.path.basename(f_path)} (Already evaluated: {os.path.basename(out_csv)})")
            continue
        # --- 新增逻辑结束 ---

        results = []
        with open(f_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 开始处理
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(process_single_line, line) for line in lines]
            for fu in tqdm(as_completed(futures), total=len(lines), desc=os.path.basename(f_path)):
                if res := fu.result(): results.append(res)
        
        if not results: continue
        
        df = pd.DataFrame(results)
        generate_report(df, os.path.basename(f_path))
        
        # 保存文件 (out_csv 已经在循环开头定义过了，这里直接使用)
        df.to_csv(out_csv, index=False)
        print(f"💾 Saved: {out_csv}")

if __name__ == "__main__":
    main()