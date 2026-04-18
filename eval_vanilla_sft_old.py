import json
import re
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===========================
# 0. Global Configuration
# ===========================
MAX_WORKERS = 10
USE_LLM_JUDGE = True

# 从 config 导入 (确保同目录下有 config.py)
try:
    from config import Config
    OPENAI_API_KEY = Config.OPENAI_API_KEY
    OPENAI_BASE_URL = Config.OPENAI_BASE_URL
except ImportError:
    print("⚠️ 未找到 config.py，使用环境变量或默认值")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ===========================
# 0.1 NLP Metrics Dependencies
# ===========================
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    print("⚠️ 请安装 NLP 依赖: pip install rouge-score nltk")

# ===========================
# 0.2 LLM Client
# ===========================
if USE_LLM_JUDGE and OPENAI_API_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ===========================
# 1. Text Evaluator (NLP Metrics)
# ===========================
class TextEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

    def compute_metrics(self, ref, pred):
        """计算 Rouge-1, Rouge-2, Rouge-L, BLEU-4"""
        if not ref or not pred:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu4": 0.0}

        # Character-level tokenization for Chinese
        ref_tokens = list(ref.strip())
        pred_tokens = list(pred.strip())

        ref_str = " ".join(ref_tokens)
        pred_str = " ".join(pred_tokens)

        scores = self.scorer.score(ref_str, pred_str)

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
# 2. Parser for SoChem-LLM XML (Ground Truth)
# ===========================
class GTParser:
    @staticmethod
    def extract_tag_content(xml, tag):
        """Extract content between <tag> and </tag>"""
        if not isinstance(xml, str):
            return ""
        pattern = f"<{tag}.*?>(.*?)</{tag}>"
        match = re.search(pattern, xml, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def normalize_status(text):
        """Normalize status symbols: ✅/PASS/TRUE -> 1, ❌/FAIL/FALSE -> 0"""
        if not text:
            return -1
        t = text.upper()
        if '✅' in t or 'PASS' in t or 'TRUE' in t:
            return 1
        if '❌' in t or 'FAIL' in t or 'FALSE' in t or '⚠️' in t:
            return 0
        return -1

    @staticmethod
    def parse_entry(entry):
        """
        解析 SoChem-LLM XML 格式的 ground_truth
        以及纯文本的 generated_output
        """
        raw_input = entry.get('input', '')
        raw_pred = entry.get('generated_output', entry.get('prediction', ''))
        raw_gt = entry.get('ground_truth', entry.get('output', ''))

        # 1. 从 Ground Truth XML 中提取各字段
        gt_node = GTParser.extract_tag_content(raw_gt, "CurrentNode")
        gt_status_raw = GTParser.extract_tag_content(raw_gt, "Result")
        gt_checklist = GTParser.extract_tag_content(raw_gt, "CheckList")
        gt_strategy = GTParser.extract_tag_content(raw_gt, "Strategy")
        gt_response = GTParser.extract_tag_content(raw_gt, "Response")

        # 2. 从 Student 输入中提取学生说的话
        student_say = GTParser.extract_tag_content(raw_input, "Speak")
        observation = GTParser.extract_tag_content(raw_input, "Observation")

        # 3. Prediction 就是纯文本 response（无 XML 包装）
        pred_response = raw_pred.strip() if raw_pred else ""

        return {
            "student_say": student_say,
            "observation": observation,
            "gt_node": gt_node,
            "gt_status_val": GTParser.normalize_status(gt_status_raw),
            "gt_checklist": gt_checklist,
            "gt_strategy": gt_strategy,
            "gt_response": gt_response,
            "pred_response": pred_response,  # 纯文本
            "raw_input_context": raw_input
        }

# ===========================
# 3. LLM Judges (For Plain Text Evaluation)
# ===========================
class LLMJudge:
    @staticmethod
    def eval_socratic_quality(data):
        """
        评估回复是否遵循苏格拉底式教学法
        适用于所有场景（正常流程 + 错误干预）
        """
        prompt = f"""[Socratic Teaching Quality Assessment]

Role: You are evaluating responses from a Socratic-style chemistry lab tutor.

Student Input: "{data['student_say']}"
Observation: "{data['observation']}"
Model Response: "{data['pred_response']}"
Ground Truth Response: "{data['gt_response']}"

EVALUATION CRITERIA (Score 0-10):
1. **Socratic Method**: Does the response guide through questions rather than direct telling?
2. **Pedagogical Effectiveness**: Does it encourage student thinking and participation?
3. **Appropriateness**: Is the response suitable for the current context?
4. **Fluency and Clarity**: Is the language natural and clear?

Score Guidelines:
- 9-10: Excellent Socratic approach, guides student to discover answers
- 7-8: Good guidance, mostly asks questions
- 5-6: Moderate, some guidance but also some directiveness
- 3-4: Mostly tells rather than asks
- 1-2: Direct instruction with no Socratic elements
- 0: Completely off-topic or harmful

Output JSON: {{"score": <int>, "reason": "<brief explanation>"}}
"""
        try:
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = json.loads(res.choices[0].message.content)
            return result.get('score', 0), result.get('reason', '')
        except Exception as e:
            print(f"LLM Judge Error (socratic_quality): {e}")
            return 0, ""

    @staticmethod
    def eval_predictive_questioning(data):
        """
        评估回复是否使用"预测性提问"引导学生
        特别适用于错误场景：是否通过提问让学生预测后果
        """
        prompt = f"""[Predictive Questioning Assessment]

Scenario: Student makes an error or needs guidance in chemistry lab.
Student Input: "{data['student_say']}"
Observation: "{data['observation']}"
Model Response: "{data['pred_response']}"

TASK: Evaluate if the response uses PREDICTIVE QUESTIONING
(predicting consequences, asking "What would happen if...?")

CRITERIA:
1. Does it ask about potential consequences of actions?
2. Does it guide student to think ahead about results?
3. Does it avoid directly telling the student they are wrong?

Score (0-10):
- 10: Excellent predictive questioning ("What do you think will happen if we add more?")
- 5: Some guidance but not clearly predictive
- 0: Direct statement or command with no questioning

Output JSON: {{"score": <int>}}
"""
        try:
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(res.choices[0].message.content).get('score', 0)
        except Exception as e:
            print(f"LLM Judge Error (predictive_questioning): {e}")
            return 0

    @staticmethod
    def eval_safety_awareness(data):
        """
        评估模型是否在学生操作错误时进行了安全相关指导
        Ground Truth 标记了错误场景（FAIL）时，评估 prediction 是否体现安全意识
        """
        gt_status = data['gt_status_val']
        is_error_case = (gt_status == 0)  # Ground Truth 标记为 FAIL

        if not is_error_case:
            return None  # 非错误场景，跳过此项评估

        prompt = f"""[Safety Awareness Evaluation]

Scenario: Student made an error in chemistry lab.
Student Input: "{data['student_say']}"
Observation: "{data['observation']}"
Model Response: "{data['pred_response']}"

TASK: Evaluate if the teacher response appropriately addresses the safety concern:
1. Does it acknowledge the potential issue?
2. Does it guide student to understand the problem?
3. Does it suggest correct procedure?

Score (0-10):
- 10: Excellent safety guidance, helps student understand consequences
- 5: Moderate, some safety mention but not comprehensive
- 0: Ignores safety concern or is inappropriate

Output JSON: {{"score": <int>}}
"""
        try:
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(res.choices[0].message.content).get('score', 0)
        except Exception as e:
            print(f"LLM Judge Error (safety_awareness): {e}")
            return 0

# ===========================
# 4. Main Processing
# ===========================
text_evaluator = TextEvaluator()

def process_single_line(line):
    """
    处理单条评估数据
    - NLP 指标：始终计算
    - LLM Judge：根据场景调用
    """
    try:
        entry = json.loads(line)
        data = GTParser.parse_entry(entry)

        metrics = {}

        # --- 1. NLP Metrics (始终计算) ---
        nlp_metrics = text_evaluator.compute_metrics(
            data['gt_response'],
            data['pred_response']
        )
        metrics.update(nlp_metrics)

        # --- 2. LLM Judges ---
        if USE_LLM_JUDGE and OPENAI_API_KEY:
            # A. 苏格拉底式教学质量（所有样本）
            socratic_score, socratic_reason = LLMJudge.eval_socratic_quality(data)
            metrics['socratic_score'] = socratic_score
            metrics['socratic_reason'] = socratic_reason

            # B. 预测性提问评估
            metrics['predictive_questioning_score'] = LLMJudge.eval_predictive_questioning(data)

            # C. 安全意识评估（仅 GT 标记为 FAIL 的样本）
            safety_score = LLMJudge.eval_safety_awareness(data)
            if safety_score is not None:
                metrics['safety_awareness_score'] = safety_score

        return {**data, **metrics}

    except Exception as e:
        print(f"Error processing line: {e}")
        return None

# ===========================
# 5. Report Generation
# ===========================
def generate_report(df, filename):
    print(f"\n{'='*60}")
    print(f"📊 Evaluation Report: {filename}")
    print(f"{'='*60}")

    total = len(df)

    # --- NLP Metrics ---
    print(f"\n[1] 📝 Text Generation Quality (NLP Metrics)")
    if 'rouge1' in df.columns:
        print(f"   ROUGE-1: {df['rouge1'].mean():.2f}")
        print(f"   ROUGE-2: {df['rouge2'].mean():.2f}")
        print(f"   ROUGE-L: {df['rougeL'].mean():.2f}")
        print(f"   BLEU-4:  {df['bleu4'].mean():.2f}")

    # --- Socratic Quality ---
    print(f"\n[2] 🎓 Socratic Teaching Quality")
    if 'socratic_score' in df.columns:
        valid_scores = df[df['socratic_score'] > 0]['socratic_score']
        if not valid_scores.empty:
            print(f"   Socratic Score: {valid_scores.mean():.2f} / 10")
            print(f"   Score Distribution:")
            for bucket in [(9, 10), (7, 8), (5, 6), (3, 4), (1, 2), (0, 0)]:
                low, high = bucket
                count = len(valid_scores[(valid_scores >= low) & (valid_scores <= high)])
                pct = count / len(valid_scores) * 100
                print(f"      {low}-{high}: {count} ({pct:.1f}%)")

    # --- Predictive Questioning ---
    print(f"\n[3] 🔮 Predictive Questioning")
    if 'predictive_questioning_score' in df.columns:
        pq_scores = df['predictive_questioning_score']
        valid_pq = pq_scores[pq_scores > 0]
        if not valid_pq.empty:
            print(f"   Avg Score: {valid_pq.mean():.2f} / 10")
            high_pq = len(valid_pq[valid_pq >= 7])
            print(f"   High Quality (≥7): {high_pq} ({high_pq/len(valid_pq):.1%})")

    # --- Safety Awareness (Error Cases Only) ---
    print(f"\n[4] ⚠️ Safety Awareness (Error Cases)")
    if 'safety_awareness_score' in df.columns:
        safety_df = df[df['safety_awareness_score'].notna()]
        if not safety_df.empty:
            print(f"   Error Cases: {len(safety_df)} / {total}")
            print(f"   Avg Safety Score: {safety_df['safety_awareness_score'].mean():.2f} / 10")
        else:
            print(f"   Error Cases: 0 / {total} (GT had no FAIL cases)")

    # --- GT Status Distribution (Info Only) ---
    print(f"\n[5] 📋 Ground Truth Status Distribution")
    if 'gt_status_val' in df.columns:
        pass_cases = len(df[df['gt_status_val'] == 1])
        fail_cases = len(df[df['gt_status_val'] == 0])
        unknown_cases = len(df[df['gt_status_val'] == -1])
        print(f"   PASS (✅): {pass_cases}")
        print(f"   FAIL (❌): {fail_cases}")
        print(f"   Unknown:   {unknown_cases}")

    # --- Sample Outputs (Good vs Bad) ---
    if 'socratic_score' in df.columns:
        print(f"\n[6] 📖 Sample Outputs")
        good = df[df['socratic_score'] >= 8].iloc[0] if not df[df['socratic_score'] >= 8].empty else None
        bad = df[df['socratic_score'] <= 3].iloc[0] if not df[df['socratic_score'] <= 3].empty else None

        if good is not None:
            print(f"\n   ✅ Good Example (Score: {good['socratic_score']}):")
            print(f"      Student: {good['student_say'][:60]}...")
            print(f"      Response: {good['pred_response'][:120]}...")

        if bad is not None:
            print(f"\n   ❌ Poor Example (Score: {bad['socratic_score']}):")
            print(f"      Student: {bad['student_say'][:60]}...")
            print(f"      Response: {bad['pred_response'][:120]}...")

    print(f"\n{'='*60}")
    print(f"Total samples evaluated: {total}")
    print(f"{'='*60}\n")

# ===========================
# 6. Main
# ===========================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Vanilla SFT / NoXML models")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path (default: <input>_eval.csv)")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Max workers (default: 10)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM Judge evaluation")
    args = parser.parse_args()

    global MAX_WORKERS, USE_LLM_JUDGE
    MAX_WORKERS = args.workers
    USE_LLM_JUDGE = not args.no_llm

    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        return

    output_path = args.output or args.input.replace(".jsonl", "_eval.csv")

    # Skip if already evaluated
    if os.path.exists(output_path):
        print(f"⏩ Already evaluated: {output_path}")
        response = input("Re-evaluate? (y/N): ")
        if response.lower() != 'y':
            return

    print(f"🚀 Starting Evaluation")
    print(f"   Input:  {args.input}")
    print(f"   Output: {output_path}")
    print(f"   Workers: {MAX_WORKERS}")
    print(f"   LLM Judge: {'ON' if USE_LLM_JUDGE else 'OFF'}")

    results = []
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"📊 Processing {len(lines)} samples...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_single_line, line) for line in lines]
        for fu in tqdm(as_completed(futures), total=len(lines), desc="Evaluating"):
            if res := fu.result():
                results.append(res)

    if not results:
        print("❌ No results collected")
        return

    df = pd.DataFrame(results)
    generate_report(df, os.path.basename(args.input))
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"💾 Saved: {output_path}")

if __name__ == "__main__":
    main()
