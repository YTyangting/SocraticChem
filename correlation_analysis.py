"""
人类评分与GPT-4o评分相关性分析
================================
计算 Cohen's Kappa (分类) 和 Pearson相关系数 (连续)

使用方法:
    python correlation_analysis.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 配置
# ===========================
HUMAN_SCORE_FILE = r"/home/yjh/socChem_final/eval_form(1).csv.xlsx"
GPT4O_SCORE_FILE = "gpt4o_scores.json"
OUTPUT_FILE = "correlation_results.txt"

# ===========================
# 数据加载
# ===========================
def load_human_scores():
    """加载人类评分"""
    # 使用 utf-8 编码读取，engine 指定 openpyxl
    df = pd.read_excel(HUMAN_SCORE_FILE, engine='openpyxl')
    
    # 由于列名可能有编码问题，我们按位置访问
    # 列顺序: 0=样本ID, 1=原始索引, 2=实验名称, ..., 11=安全性评分, 12=苏格拉底评分, 13=评语
    print(f"[Human Scores] Shape: {df.shape}")
    
    scores = {}
    for idx, row in df.iterrows():
        # 按位置获取 sample_id (第0列)
        sample_id = str(row.iloc[0]).strip()
        
        # 按位置获取评分 (第11列=安全性, 第12列=苏格拉底式)
        try:
            safety_score = int(float(row.iloc[11]))
            socratic_score = int(float(row.iloc[12]))
        except (ValueError, IndexError) as e:
            print(f"  [WARN] Row {idx}: {sample_id} - failed to parse scores: {e}")
            continue
        
        if sample_id and (1 <= safety_score <= 10) and (1 <= socratic_score <= 10):
            scores[sample_id] = {
                'safety': safety_score,
                'socratic': socratic_score
            }
    
    return scores

def load_gpt4o_scores():
    """加载GPT-4o评分"""
    script_dir = Path(__file__).parent
    gpt4o_path = script_dir / GPT4O_SCORE_FILE
    
    if not gpt4o_path.exists():
        print(f"[ERROR] GPT-4o score file not found: {gpt4o_path}")
        return {}
    
    with open(gpt4o_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scores = {}
    for item in data:
        sample_id = item['sample_id']
        scores[sample_id] = {
            'safety': item['safety_score'],
            'socratic': item['socratic_score']
        }
    
    return scores

# ===========================
# 相关性分析
# ===========================
def categorize_scores(scores, threshold=5):
    """
    将连续分数转为二分类(0/1)用于Cohen's Kappa
    threshold: >=threshold 为"好"(1), <threshold 为"差"(0)
    """
    return [1 if s >= threshold else 0 for s in scores]

def analyze_correlation(human_scores, gpt4o_scores):
    """计算相关性"""
    
    # 找出共同样本
    common_ids = set(human_scores.keys()) & set(gpt4o_scores.keys())
    print(f"\n[匹配] Human: {len(human_scores)}, GPT-4o: {len(gpt4o_scores)}, Common: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("[ERROR] No common samples found!")
        return
    
    # 提取配对数据
    human_safety = [human_scores[sid]['safety'] for sid in sorted(common_ids)]
    human_socratic = [human_scores[sid]['socratic'] for sid in sorted(common_ids)]
    gpt4o_safety = [gpt4o_scores[sid]['safety'] for sid in sorted(common_ids)]
    gpt4o_socratic = [gpt4o_scores[sid]['socratic'] for sid in sorted(common_ids)]
    
    # 1. Pearson相关系数 (连续变量)
    print("\n" + "=" * 60)
    print("📊 Pearson 相关系数 (连续评分)")
    print("=" * 60)
    
    # 安全性
    if len(set(human_safety)) > 1 and len(set(gpt4o_safety)) > 1:
        r_safety, p_safety = pearsonr(human_safety, gpt4o_safety)
        print(f"\n  安全性评分:")
        print(f"    Pearson r = {r_safety:.4f}")
        print(f"    p-value  = {p_safety:.4e}")
        print(f"    解释: {'强相关' if abs(r_safety) > 0.7 else '中等相关' if abs(r_safety) > 0.4 else '弱相关'}")
    else:
        print("  安全性评分: 无法计算(方差为0)")
    
    # 苏格拉底式教学
    if len(set(human_socratic)) > 1 and len(set(gpt4o_socratic)) > 1:
        r_socratic, p_socratic = pearsonr(human_socratic, gpt4o_socratic)
        print(f"\n  苏格拉底式教学评分:")
        print(f"    Pearson r = {r_socratic:.4f}")
        print(f"    p-value  = {p_socratic:.4e}")
        print(f"    解释: {'强相关' if abs(r_socratic) > 0.7 else '中等相关' if abs(r_socratic) > 0.4 else '弱相关'}")
    else:
        print("  苏格拉底式教学评分: 无法计算(方差为0)")
    
    # 2. Cohen's Kappa (二分类)
    print("\n" + "=" * 60)
    print("📊 Cohen's Kappa (二分类, threshold=5)")
    print("=" * 60)
    
    # 安全性二分类
    human_safety_bin = categorize_scores(human_safety)
    gpt4o_safety_bin = categorize_scores(gpt4o_safety)
    kappa_safety = cohen_kappa_score(human_safety_bin, gpt4o_safety_bin)
    print(f"\n  安全性评分:")
    print(f"    Kappa = {kappa_safety:.4f}")
    print(f"    解释: {'几乎完全一致' if kappa_safety > 0.8 else '高度一致' if kappa_safety > 0.6 else '中等一致' if kappa_safety > 0.4 else '一致性较低'}")
    
    # 苏格拉底式教学二分类
    human_socratic_bin = categorize_scores(human_socratic)
    gpt4o_socratic_bin = categorize_scores(gpt4o_socratic)
    kappa_socratic = cohen_kappa_score(human_socratic_bin, gpt4o_socratic_bin)
    print(f"\n  苏格拉底式教学评分:")
    print(f"    Kappa = {kappa_socratic:.4f}")
    print(f"    解释: {'几乎完全一致' if kappa_socratic > 0.8 else '高度一致' if kappa_socratic > 0.6 else '中等一致' if kappa_socratic > 0.4 else '一致性较低'}")
    
    # 3. 详细对比表
    print("\n" + "=" * 60)
    print("📋 详细对比 (前20条)")
    print("=" * 60)
    print(f"{'Sample':<8} {'Human_S':<10} {'GPT4o_S':<10} {'Human_Soc':<10} {'GPT4o_Soc':<10}")
    print("-" * 60)
    
    for sid in sorted(common_ids)[:20]:
        h_s = human_scores[sid]['safety']
        g_s = gpt4o_scores[sid]['safety']
        h_soc = human_scores[sid]['socratic']
        g_soc = gpt4o_scores[sid]['socratic']
        print(f"{sid:<8} {h_s:<10} {g_s:<10} {h_soc:<10} {g_soc:<10}")
    
    # 4. 保存结果
    return {
        'pearson_safety': r_safety if len(set(human_safety)) > 1 else None,
        'pearson_socratic': r_socratic if len(set(human_socratic)) > 1 else None,
        'kappa_safety': kappa_safety,
        'kappa_socratic': kappa_socratic,
        'common_samples': len(common_ids)
    }

# ===========================
# 主函数
# ===========================
def main():
    print("=" * 60)
    print("🔍 人类评分 vs GPT-4o评分 相关性分析")
    print("=" * 60)
    
    # 加载数据
    print("\n[1] 加载人类评分...")
    human_scores = load_human_scores()
    print(f"    Loaded {len(human_scores)} human scores")
    
    print("\n[2] 加载GPT-4o评分...")
    gpt4o_scores = load_gpt4o_scores()
    print(f"    Loaded {len(gpt4o_scores)} GPT-4o scores")
    
    if not human_scores or not gpt4o_scores:
        print("[ERROR] Missing score data")
        return
    
    # 分析相关性
    print("\n[3] 计算相关性...")
    results = analyze_correlation(human_scores, gpt4o_scores)
    
    # 保存结果到文件
    script_dir = Path(__file__).parent
    output_path = script_dir / OUTPUT_FILE
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("人类评分 vs GPT-4o评分 相关性分析结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"共同样本数: {results['common_samples']}\n\n")
        f.write("Pearson 相关系数:\n")
        f.write(f"  安全性评分: r = {results['pearson_safety']:.4f}\n" if results['pearson_safety'] else "  安全性评分: 无法计算\n")
        f.write(f"  苏格拉底式: r = {results['pearson_socratic']:.4f}\n" if results['pearson_socratic'] else "  苏格拉底式: 无法计算\n")
        f.write("\nCohen's Kappa:\n")
        f.write(f"  安全性评分: κ = {results['kappa_safety']:.4f}\n")
        f.write(f"  苏格拉底式: κ = {results['kappa_socratic']:.4f}\n")
    
    print(f"\n✅ 结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
