"""
human_eval_collector.py
========================
Layer 3: 人工评估验证框架

功能：
1. 生成供人工评估的样本对（XDL-DAG 对应关系）
2. 专家评分问卷生成
3. 跨领域泛化测试
4. 评分汇总与统计分析

Usage:
    python human_eval_collector.py --mode generate --input xdl_results.json
    python human_eval_collector.py --mode cross_domain
    python human_eval_collector.py --mode analyze --scores expert_scores.json
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import math

# === 评分量表定义 ===
EXPERT_EVALUATION_CRITERIA = {
    "xdl_quality": {
        "name": "XDL Structure Quality",
        "description": "Is the XDL structurally valid and follows the specification?",
        "scores": {
            5: "Perfect - follows spec exactly, all required elements present",
            4: "Good - minor issues, overall structure correct",
            3: "Acceptable - some structural issues but comprehensible",
            2: "Poor - significant structural problems",
            1: "Fail - invalid or completely incorrect"
        }
    },
    "xdl_completeness": {
        "name": "XDL Completeness",
        "description": "Are all necessary steps, reagents, and hardware properly defined?",
        "scores": {
            5: "Complete - nothing missing",
            4: "Nearly complete - minor omissions",
            3: "Partial - some important elements missing",
            2: "Incomplete - many elements missing",
            1: "Missing critical components"
        }
    },
    "dag_quality": {
        "name": "DAG Logic Quality",
        "description": "Does the DAG logically represent the experimental procedure?",
        "scores": {
            5: "Excellent - clear logical flow, proper dependencies",
            4: "Good - mostly correct logic with minor issues",
            3: "Acceptable - logic present but some errors",
            2: "Poor - significant logical flaws",
            1: "Fail - illogical or meaningless"
        }
    },
    "dag_coverage": {
        "name": "DAG Coverage",
        "description": "Does the DAG cover all essential steps in the XDL?",
        "scores": {
            5: "Complete coverage",
            4: "Minor steps missed",
            3: "Some important steps missed",
            2: "Many steps missing",
            1: "Critical steps not represented"
        }
    },
    "expert_verification": {
        "name": "Expert Verification",
        "description": "Would this XDL/DAG pair successfully guide a chemistry experiment?",
        "scores": {
            5: "Definitely yes - can execute directly",
            4: "Probably yes - minor fixes needed",
            3: "Maybe - significant fixes needed",
            2: "Probably not - major rework required",
            1: "Cannot be used"
        }
    }
}


# === 数据类 ===
@dataclass
class ExpertSample:
    """供专家评估的样本"""
    sample_id: str
    chem_text: str  # 原始化学实验描述
    xdl_from_model: Dict[str, str]  # {model_name: xdl_text}
    dag_from_model: Dict[str, str]  # {model_name: dag_text}
    ground_truth: Optional[str] = None  # 专家编写的参考答案（如果有）
    
    def to_expert_sheet(self) -> Dict:
        """转换为专家评估表格式"""
        return {
            "sample_id": self.sample_id,
            "chem_text": self.chem_text,
            "xdl_options": self.xdl_from_model,
            "dag_options": self.dag_from_model
        }


@dataclass
class ExpertScores:
    """专家评分记录"""
    sample_id: str
    expert_id: str
    timestamp: str
    ratings: Dict[str, int]  # {criterion: score}
    comments: str
    overall_notes: str = ""


# === 采样策略 ===
def stratified_sampling(results: List[Dict], n_samples: int = 30) -> List[Dict]:
    """
    分层采样：从 XDL 生成结果中选择最具代表性的样本
    
    分层依据：
    1. 一致性高（多模型生成结果相似）
    2. 一致性低（多模型生成结果差异大）
    3. 边界情况（相似度在阈值附近）
    """
    if not results or n_samples >= len(results):
        return results
    
    # 计算每个案例的一致性分数
    scored_results = []
    for r in results:
        agreement = r.get("xdl_agreement", 0)
        scored_results.append((agreement, r))
    
    # 分成三个层次
    high_agreement = [r for score, r in scored_results if score >= 0.95]
    mid_agreement = [r for score, r in scored_results if 0.80 <= score < 0.95]
    low_agreement = [r for score, r in scored_results if score < 0.80]
    
    # 分配采样数量（约 60% 高一致，30% 中等，10% 低一致）
    n_high = int(n_samples * 0.6)
    n_mid = int(n_samples * 0.3)
    n_low = n_samples - n_high - n_mid
    
    samples = []
    
    # 从每层随机采样
    if high_agreement and n_high > 0:
        samples.extend(random.sample(high_agreement, min(n_high, len(high_agreement))))
    if mid_agreement and n_mid > 0:
        samples.extend(random.sample(mid_agreement, min(n_mid, len(mid_agreement))))
    if low_agreement and n_low > 0:
        samples.extend(random.sample(low_agreement, min(n_low, len(low_agreement))))
    
    return samples


def select_samples_for_human_eval(xdl_results: List[Dict], dag_results: List[Dict] = None,
                                   n_expert_samples: int = 30) -> List[Dict]:
    """
    选择供人工评估的样本
    - 包含高/中/低一致性的案例
    - 包含跨模型对比
    """
    
    # 对 XDL 结果进行分层采样
    sampled_xdl = stratified_sampling(xdl_results, n_expert_samples)
    
    # 构建评估样本
    eval_samples = []
    for i, xdl_case in enumerate(sampled_xdl):
        sample = {
            "sample_id": f"eval_{i:03d}",
            "chem_text": xdl_case.get("chem_text", ""),
            "case_id": xdl_case.get("case_id", i),
            "xdl_models": {},
            "dag_models": {}
        }
        
        # 收集各模型的 XDL
        for model_name, result in xdl_case.get("model_results", {}).items():
            if "xdl" in result:
                sample["xdl_models"][model_name] = result["xdl"]
        
        # 如果有 DAG 结果，也收集
        if dag_results:
            # 找到对应的 DAG 结果
            dag_case = next((d for d in dag_results if d.get("case_id") == xdl_case.get("case_id")), None)
            if dag_case:
                for model_name, result in dag_case.get("model_results", {}).items():
                    if "dag" in result:
                        sample["dag_models"][model_name] = result["dag"]
        
        eval_samples.append(sample)
    
    return eval_samples


def generate_expert_questionnaire(samples: List[Dict], output_path: str,
                                   include_comparison: bool = True):
    """
    生成专家评估问卷（Markdown 格式，可打印）
    """
    
    questionnaire = []
    questionnaire.append("# Expert Evaluation Questionnaire")
    questionnaire.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d')}")
    questionnaire.append(f"\n**Total Samples:** {len(samples)}")
    questionnaire.append("\n---\n")
    
    # 评分说明
    questionnaire.append("## Evaluation Criteria\n")
    for key, criterion in EXPERT_EVALUATION_CRITERIA.items():
        questionnaire.append(f"### {criterion['name']}")
        questionnaire.append(f"*{criterion['description']}*\n")
        for score, desc in criterion['scores'].items():
            questionnaire.append(f"- **{score}:** {desc}")
        questionnaire.append("")
    
    questionnaire.append("\n---\n")
    
    # 逐个样本
    for i, sample in enumerate(samples):
        questionnaire.append(f"## Sample {i+1}: {sample['sample_id']}\n")
        questionnaire.append(f"**Original Description:**\n>{sample['chem_text']}\n")
        
        # XDL 评估（每个模型）
        if sample.get("xdl_models"):
            questionnaire.append("\n### XDL Generation\n")
            for model_name, xdl_text in sample["xdl_models"].items():
                questionnaire.append(f"#### Model: {model_name}\n")
                questionnaire.append(f"```xml\n{xdl_text[:2000]}")  # 截断
                if len(xdl_text) > 2000:
                    questionnaire.append("\n... (truncated)")
                questionnaire.append("```\n")
        
        # DAG 评估（每个模型）
        if sample.get("dag_models"):
            questionnaire.append("\n### DAG Generation\n")
            for model_name, dag_text in sample["dag_models"].items():
                questionnaire.append(f"#### Model: {model_name}\n")
                if isinstance(dag_text, str):
                    try:
                        dag_json = json.loads(dag_text)
                        dag_text = json.dumps(dag_json, indent=2, ensure_ascii=False)
                    except:
                        pass
                questionnaire.append(f"```json\n{dag_text[:1500]}\n```\n")
        
        # 评分表
        questionnaire.append("\n### Scores\n")
        questionnaire.append("| Criterion | Score (1-5) | Comments |")
        questionnaire.append("|-----------|-------------|---------|")
        for key, criterion in EXPERT_EVALUATION_CRITERIA.items():
            questionnaire.append(f"| {criterion['name']} | | |")
        questionnaire.append("| **Overall** | | |")
        
        # 总体评价
        questionnaire.append("\n**Overall Notes:**\n")
        questionnaire.append("_" * 50 + "\n\n")
        
        questionnaire.append("---\n")
    
    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(questionnaire))
    
    print(f"✅ Expert questionnaire saved to: {output_path}")
    return output_path


# === 评分汇总分析 ===
def load_expert_scores(scores_file: str) -> List[ExpertScores]:
    """加载专家评分数据"""
    with open(scores_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    scores = []
    for entry in data:
        scores.append(ExpertScores(
            sample_id=entry["sample_id"],
            expert_id=entry["expert_id"],
            timestamp=entry.get("timestamp", ""),
            ratings=entry.get("ratings", {}),
            comments=entry.get("comments", ""),
            overall_notes=entry.get("overall_notes", "")
        ))
    
    return scores


def compute_inter_rater_reliability(scores: List[ExpertScores]) -> Dict[str, float]:
    """
    计算评分者间信度 (Inter-Rater Reliability)
    使用 Weighted Kappa 或 ICC
    """
    
    # 按样本分组
    samples = defaultdict(list)
    for s in scores:
        samples[s.sample_id].append(s)
    
    # 计算每对评分者的相关性
    from itertools import combinations
    
    correlations = []
    
    for sample_id, sample_scores in samples.items():
        if len(sample_scores) < 2:
            continue
        
        # 两两对比
        for (s1, s2) in combinations(sample_scores, 2):
            # 计算每个指标的差异
            for criterion in EXPERT_EVALUATION_CRITERIA.keys():
                r1 = s1.ratings.get(criterion, 3)
                r2 = s2.ratings.get(criterion, 3)
                
                # 一致率
                if r1 == r2:
                    correlations.append(1.0)
                else:
                    diff = abs(r1 - r2)
                    correlations.append(max(0, 1 - diff / 4))  # 归一化差异
    
    avg_correlation = sum(correlations) / len(correlations) if correlations else 0
    
    return {
        "avg_agreement": avg_correlation,
        "total_comparisons": len(correlations),
        "interpretation": "Substantial" if avg_correlation >= 0.8 else "Moderate" if avg_correlation >= 0.6 else "Fair/Poor"
    }


def compute_criterion_statistics(scores: List[ExpertScores]) -> Dict[str, Dict]:
    """计算每个评价指标的统计数据"""
    
    stats = {}
    
    for criterion_key in EXPERT_EVALUATION_CRITERIA.keys():
        ratings = [s.ratings.get(criterion_key, 0) for s in scores if criterion_key in s.ratings]
        
        if not ratings:
            continue
        
        mean = sum(ratings) / len(ratings)
        variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
        std_dev = math.sqrt(variance)
        
        counter = Counter(ratings)
        
        stats[criterion_key] = {
            "mean": round(mean, 3),
            "std_dev": round(std_dev, 3),
            "min": min(ratings),
            "max": max(ratings),
            "distribution": dict(sorted(counter.items())),
            "n_ratings": len(ratings)
        }
    
    return stats


def generate_human_eval_report(scores: List[ExpertScores], output_path: str) -> str:
    """生成人工评估报告"""
    
    report = []
    report.append("# Human Evaluation Analysis Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Total Scores Collected:** {len(scores)}")
    report.append(f"\n**Unique Experts:** {len(set(s.expert_id for s in scores))}")
    report.append(f"\n**Unique Samples:** {len(set(s.sample_id for s in scores))}")
    
    # 评分者间信度
    irr = compute_inter_rater_reliability(scores)
    report.append("\n## Inter-Rater Reliability (IRR)")
    report.append(f"- **Average Agreement:** {irr['avg_agreement']:.3f}")
    report.append(f"- **Total Comparisons:** {irr['total_comparisons']}")
    report.append(f"- **Interpretation:** {irr['interpretation']}")
    
    # 每个指标的统计
    criterion_stats = compute_criterion_statistics(scores)
    
    report.append("\n## Criterion Statistics\n")
    report.append("| Criterion | Mean | Std Dev | Min | Max | Distribution |")
    report.append("|-----------|------|---------|-----|-----|--------------|")
    
    for criterion_key, stats in criterion_stats.items():
        criterion_name = EXPERT_EVALUATION_CRITERIA[criterion_key]["name"]
        dist_str = " ".join([f"{k}:{v}" for k, v in stats["distribution"].items()])
        report.append(
            f"| {criterion_name} | {stats['mean']:.2f} | {stats['std_dev']:.2f} | "
            f"{stats['min']} | {stats['max']} | {dist_str} |"
        )
    
    # 总体评分
    all_ratings = []
    for s in scores:
        overall_key = "expert_verification"  # 使用专家验证作为总体指标
        if overall_key in s.ratings:
            all_ratings.append(s.ratings[overall_key])
    
    if all_ratings:
        overall_mean = sum(all_ratings) / len(all_ratings)
        report.append(f"\n## Overall Quality Score")
        report.append(f"- **Mean:** {overall_mean:.2f} / 5.0")
        report.append(f"- **Percentage:** {overall_mean/5*100:.1f}%")
        
        if overall_mean >= 4.5:
            report.append(f"- **Rating:** ⭐⭐⭐⭐⭐ Excellent")
        elif overall_mean >= 3.5:
            report.append(f"- **Rating:** ⭐⭐⭐⭐ Good")
        elif overall_mean >= 2.5:
            report.append(f"- **Rating:** ⭐⭐⭐⭐⚠ Acceptable")
        elif overall_mean >= 1.5:
            report.append(f"- **Rating:** ⚠️ Needs Improvement")
        else:
            report.append(f"- **Rating:** ❌ Unsatisfactory")
    
    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"✅ Human evaluation report saved to: {output_path}")
    return "\n".join(report)


# === 跨领域泛化测试 ===
CROSS_DOMAIN_TESTS = {
    "chemistry_basic": {
        "name": "Chemistry - Basic Reactions",
        "category": "chemistry",
        "test_cases": [
            "Mix dilute HCl with NaOH solution and measure temperature change.",
            "Add AgNO3 solution to NaCl solution and observe white precipitate.",
            "Heat copper sulfate crystals until they turn white.",
            "Perform titration of HCl with NaOH using phenolphthalein indicator.",
            "Prepare Fe(OH)3 colloid by adding FeCl3 to boiling water.",
        ]
    },
    "chemistry_organic": {
        "name": "Chemistry - Organic Reactions", 
        "category": "chemistry",
        "test_cases": [
            "Synthesize aspirin from salicylic acid and acetic anhydride.",
            "Perform saponification of vegetable oil with NaOH.",
            "Test for unsaturation in alkenes using bromine water.",
            "Separate a mixture of inks using paper chromatography.",
            "Prepare a soap and test its cleaning properties.",
        ]
    },
    "physics_mechanics": {
        "name": "Physics - Mechanics (NEW DOMAIN)",
        "category": "physics",
        "test_cases": [
            "Set up a simple pendulum and measure its period for different lengths.",
            "Verify Hooke's law using a spring and various masses.",
            "Measure the coefficient of friction between wood and different surfaces.",
            "Determine the acceleration due to gravity using a free-fall apparatus.",
            "Study the conservation of momentum in elastic and inelastic collisions.",
        ]
    },
    "physics_optics": {
        "name": "Physics - Optics (NEW DOMAIN)",
        "category": "physics",
        "test_cases": [
            "Set up an optical bench to measure the focal length of a convex lens.",
            "Verify the law of reflection using a plane mirror and optical pins.",
            "Observe and measure the angle of refraction for light in glass.",
            "Determine the wavelength of laser light using a diffraction grating.",
            "Study the formation of images by a concave mirror.",
        ]
    },
    "biology_microscopy": {
        "name": "Biology - Microscopy (NEW DOMAIN)",
        "category": "biology",
        "test_cases": [
            "Prepare and stain a wet mount of onion epithelial cells.",
            "Observe plasmolysis in plant cells using salt solution.",
            "Identify different types of blood cells from a prepared slide.",
            "Study the structure of a flower and identify its parts.",
            "Observe and sketch bacteria from a yogurt sample.",
        ]
    },
    "earth_science": {
        "name": "Earth Science - Geology (NEW DOMAIN)",
        "category": "earth_science",
        "test_cases": [
            "Identify minerals using physical properties like color and hardness.",
            "Study the process of weathering using rock samples.",
            "Examine fossil types and determine relative age of rock layers.",
            "Separate components of a soil sample using sieving.",
            "Study the water cycle through a simple evaporation-condensation model.",
        ]
    }
}


def run_cross_domain_evaluation(models: List[str], domains: List[str] = None) -> Dict:
    """
    在新领域上测试 XDL 生成能力
    """
    from multi_model_generator import MultiModelGenerator, generate_xdl_single
    
    if domains is None:
        domains = list(CROSS_DOMAIN_TESTS.keys())
    
    results = {}
    
    for domain_key in domains:
        if domain_key not in CROSS_DOMAIN_TESTS:
            continue
        
        domain = CROSS_DOMAIN_TESTS[domain_key]
        print(f"\n{'='*60}")
        print(f"Testing Domain: {domain['name']}")
        print(f"Category: {domain['category']} (NEW)" if domain["category"] != "chemistry" else "Category: chemistry (IN-DOMAIN)")
        print(f"{'='*60}")
        
        domain_results = {
            "domain_name": domain["name"],
            "category": domain["category"],
            "is_novel_domain": domain["category"] != "chemistry",
            "test_cases": []
        }
        
        generator = MultiModelGenerator(models)
        
        for i, chem_text in enumerate(domain["test_cases"]):
            print(f"\n[{i+1}/{len(domain['test_cases'])}] {chem_text[:60]}...")
            
            case_result = {
                "case_text": chem_text,
                "model_results": {}
            }
            
            for model_name in models:
                if model_name not in generator.clients:
                    continue
                
                print(f"  → {model_name}...", end=" ")
                try:
                    result = generate_xdl_single(
                        generator.clients[model_name],
                        model_name,
                        chem_text
                    )
                    case_result["model_results"][model_name] = {
                        "xdl": result["xdl"],
                        "hash": result["hash"],
                        "status": "success"
                    }
                    print("✓")
                except Exception as e:
                    print(f"✗ {e}")
                    case_result["model_results"][model_name] = {
                        "error": str(e),
                        "status": "failed"
                    }
            
            # 计算该案例的一致性
            successful = [r for r in case_result["model_results"].values() if r.get("status") == "success"]
            if len(successful) >= 2:
                hashes = [r["hash"] for r in successful]
                unique = len(set(hashes))
                case_result["agreement"] = 1.0 if unique == 1 else 0.5  # 简化计算
            else:
                case_result["agreement"] = 0.0
            
            domain_results["test_cases"].append(case_result)
        
        # 计算该领域的统计
        successes = sum(1 for tc in domain_results["test_cases"] 
                       if any(r.get("status") == "success" for r in tc["model_results"].values()))
        domain_results["success_rate"] = successes / len(domain_results["test_cases"])
        
        agreements = [tc["agreement"] for tc in domain_results["test_cases"]]
        domain_results["avg_agreement"] = sum(agreements) / len(agreements) if agreements else 0
        
        results[domain_key] = domain_results
        
        print(f"\n→ Domain Summary: Success={domain_results['success_rate']:.1%}, "
              f"Agreement={domain_results['avg_agreement']:.2f}")
    
    return results


def generate_cross_domain_report(results: Dict, output_path: str) -> str:
    """生成跨领域测试报告"""
    
    report = []
    report.append("# Cross-Domain Generalization Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Domains Tested:** {len(results)}")
    
    # 按类别分组
    by_category = defaultdict(list)
    for domain_key, domain_result in results.items():
        by_category[domain_result["category"]].append(domain_result)
    
    # 总体统计
    report.append("\n## Overall Summary\n")
    report.append("| Domain | Category | Novel? | Success Rate | Avg Agreement |")
    report.append("|--------|----------|--------|--------------|---------------|")
    
    for domain_key, domain_result in results.items():
        is_novel = "✅ NEW" if domain_result.get("is_novel_domain") else "❌ In-domain"
        success = domain_result.get("success_rate", 0)
        agreement = domain_result.get("avg_agreement", 0)
        report.append(f"| {domain_result['domain_name']} | {domain_result['category']} | {is_novel} | {success:.1%} | {agreement:.2f} |")
    
    # 类别平均
    report.append("\n## Category Averages\n")
    report.append("| Category | Avg Success Rate | Avg Agreement |")
    report.append("|----------|-------------------|---------------|")
    
    for category, domain_results in by_category.items():
        avg_success = sum(r.get("success_rate", 0) for r in domain_results) / len(domain_results)
        avg_agreement = sum(r.get("avg_agreement", 0) for r in domain_results) / len(domain_results)
        is_novel = "NEW" if category != "chemistry" else "In-domain"
        report.append(f"| {category} ({is_novel}) | {avg_success:.1%} | {avg_agreement:.2f} |")
    
    # 泛化能力评估
    novel_domains = [r for r in results.values() if r.get("is_novel_domain")]
    in_domain = [r for r in results.values() if not r.get("is_novel_domain")]
    
    if novel_domains:
        novel_success = sum(r.get("success_rate", 0) for r in novel_domains) / len(novel_domains)
        novel_agreement = sum(r.get("avg_agreement", 0) for r in novel_domains) / len(novel_domains)
        
        report.append("\n## Generalization Analysis\n")
        report.append(f"- **Novel Domains Tested:** {len(novel_domains)}")
        report.append(f"- **Novel Domain Success Rate:** {novel_success:.1%}")
        report.append(f"- **Novel Domain Agreement:** {novel_agreement:.2f}")
        
        if in_domain:
            in_success = sum(r.get("success_rate", 0) for r in in_domain) / len(in_domain)
            in_agreement = sum(r.get("avg_agreement", 0) for r in in_domain) / len(in_domain)
            report.append(f"- **In-Domain Success Rate:** {in_success:.1%}")
            report.append(f"- **In-Domain Agreement:** {in_agreement:.2f}")
            
            # 计算泛化损失
            success_drop = in_success - novel_success
            agreement_drop = in_agreement - novel_agreement
            
            report.append(f"\n### Generalization Gap")
            report.append(f"- **Success Rate Drop:** {success_drop:.1%} ({'+' if success_drop > 0 else ''}{success_drop*100:.1f}pp)")
            report.append(f"- **Agreement Drop:** {agreement_drop:.2f}")
            
            if abs(success_drop) < 0.1 and abs(agreement_drop) < 0.1:
                conclusion = "✅ **EXCELLENT GENERALIZATION**: Performance on novel domains is comparable to in-domain."
            elif abs(success_drop) < 0.2:
                conclusion = "⚠️ **GOOD GENERALIZATION**: Some degradation on novel domains, but acceptable."
            else:
                conclusion = "❌ **LIMITED GENERALIZATION**: Significant degradation on novel domains. Method may be domain-specific."
            
            report.append(f"\n**Conclusion:** {conclusion}")
    
    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"✅ Cross-domain report saved to: {output_path}")
    return "\n".join(report)


# === 命令行入口 ===
def main():
    parser = argparse.ArgumentParser(description="Human Evaluation & Cross-Domain Testing Framework")
    parser.add_argument("--mode", choices=["generate", "analyze", "cross_domain", "full"],
                        default="full", help="Operation mode")
    parser.add_argument("--input", type=str, help="Input results file (XDL/DAG results)")
    parser.add_argument("--dag-input", type=str, help="DAG results file")
    parser.add_argument("--scores", type=str, help="Expert scores file (JSON)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=30, help="Number of samples for human eval")
    parser.add_argument("--models", type=str, default="gpt-4.1,deepseek-chat,qwen-plus",
                        help="Comma-separated model list")
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated domain keys to test (for cross_domain mode)")
    
    args = parser.parse()
    
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models = [m.strip() for m in args.models.split(",")]
    
    if args.mode in ["generate", "full"]:
        if not args.input:
            print("❌ --input required for generate mode")
            return
        
        print(f"Loading XDL results from: {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            xdl_results = json.load(f)
        
        dag_results = None
        if args.dag_input:
            print(f"Loading DAG results from: {args.dag_input}")
            with open(args.dag_input, "r", encoding="utf-8") as f:
                dag_results = json.load(f)
        
        # 选择样本
        samples = select_samples_for_human_eval(xdl_results, dag_results, args.n_samples)
        
        # 保存样本
        samples_path = f"{args.output}/human_eval_samples_{timestamp}.json"
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        # 生成问卷
        questionnaire_path = f"{args.output}/expert_questionnaire_{timestamp}.md"
        generate_expert_questionnaire(samples, questionnaire_path)
        
        print(f"\n✅ Human evaluation materials generated:")
        print(f"   - Samples: {samples_path}")
        print(f"   - Questionnaire: {questionnaire_path}")
    
    if args.mode in ["analyze", "full"]:
        if not args.scores:
            print("❌ --scores required for analyze mode")
            return
        
        scores = load_expert_scores(args.scores)
        report_path = f"{args.output}/human_eval_report_{timestamp}.md"
        generate_human_eval_report(scores, report_path)
        
        print(f"\n✅ Human evaluation report: {report_path}")
    
    if args.mode in ["cross_domain", "full"]:
        domains = [d.strip() for d in args.domains.split(",")] if args.domains else None
        
        results = run_cross_domain_evaluation(models, domains)
        
        # 保存结果
        results_path = f"{args.output}/cross_domain_results_{timestamp}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成报告
        report_path = f"{args.output}/cross_domain_report_{timestamp}.md"
        generate_cross_domain_report(results, report_path)
        
        print(f"\n✅ Cross-domain test completed:")
        print(f"   - Results: {results_path}")
        print(f"   - Report: {report_path}")


if __name__ == "__main__":
    main()
