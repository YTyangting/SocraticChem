"""
consistency_checker.py
======================
自动分析多模型生成结果，计算一致性指标，生成详细报告。

功能：
1. 计算 XDL 文本相似度
2. 计算 DAG 结构相似度
3. 识别分歧案例
4. 生成可视化报告（Markdown 格式）

Usage:
    python consistency_checker.py --xdl-results xdl_results.json
    python consistency_checker.py --xdl-results xdl_results.json --dag-results dag_results.json
    python consistency_checker.py --all-results .  # 扫描目录下所有结果文件
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import difflib
import hashlib

# === 配置 ===
SIMILARITY_THRESHOLDS = {
    "xdl_text": 0.90,      # XDL 文本相似度阈值
    "xdl_structure": 0.85, # XDL 结构相似度阈值
    "dag_structure": 0.80, # DAG 结构相似度阈值
    "dag_semantic": 0.75   # DAG 语义相似度阈值
}


# === 辅助函数 ===
def compute_text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度"""
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def extract_xml_structure(xml_text: str) -> Dict[str, Any]:
    """从 XML 中提取结构信息"""
    import xml.etree.ElementTree as ET
    
    try:
        root = ET.fromstring(xml_text)
        
        # 提取硬件定义
        hardware = []
        for comp in root.findall(".//Hardware/Component"):
            hardware.append({
                "type": comp.get("type", ""),
                "id": comp.get("id", "")
            })
        
        # 提取试剂
        reagents = []
        for reagent in root.findall(".//Reagents/Reagent"):
            reagents.append({
                "name": reagent.get("name", ""),
                "formula": reagent.get("formula", ""),
                "state": reagent.get("state", "")
            })
        
        # 提取步骤
        steps = []
        for step in root.findall(".//Procedure//"):
            steps.append({
                "action": step.tag,
                "attributes": step.attrib
            })
        
        return {
            "hardware": hardware,
            "reagents": reagents,
            "steps": steps,
            "step_count": len(steps)
        }
    except Exception as e:
        return {"error": str(e)}


def extract_dag_structure(dag_json: Dict) -> Dict[str, Any]:
    """从 DAG JSON 中提取结构信息"""
    if "graph_nodes" not in dag_json:
        return {"error": "Invalid DAG format"}
    
    nodes = dag_json["graph_nodes"]
    
    # 节点信息
    node_ids = [n["id"] for n in nodes]
    node_descs = [n.get("description", "") for n in nodes]
    
    # 边信息
    edges = []
    for n in nodes:
        for dep in n.get("dependencies", []):
            edges.append((dep, n["id"]))
    
    # 拓扑层（根据依赖深度计算）
    layers = {}
    visited = set()
    
    def get_depth(node_id: str) -> int:
        if node_id in visited:
            return layers.get(node_id, 0)
        
        for n in nodes:
            if n["id"] == node_id:
                deps = n.get("dependencies", [])
                if not deps:
                    layers[node_id] = 0
                else:
                    layers[node_id] = max(get_depth(d) for d in deps) + 1
                visited.add(node_id)
                return layers[node_id]
        return 0
    
    for n in nodes:
        get_depth(n["id"])
    
    layer_groups = defaultdict(list)
    for node_id, layer in layers.items():
        layer_groups[layer].append(node_id)
    
    return {
        "node_ids": node_ids,
        "node_count": len(node_ids),
        "edges": edges,
        "edge_count": len(edges),
        "layers": dict(layer_groups),
        "layer_count": len(layer_groups)
    }


def analyze_xdl_consistency(model_results: Dict[str, Dict]) -> Dict[str, Any]:
    """分析多个模型生成的 XDL 之间的一致性"""
    
    successful = {m: r for m, r in model_results.items() if "xdl" in r}
    
    if len(successful) < 2:
        return {
            "status": "insufficient_data",
            "reason": f"Only {len(successful)} successful generation(s)"
        }
    
    models = list(successful.keys())
    xdls = [successful[m]["xdl"] for m in models]
    
    # 1. 文本相似度矩阵
    similarity_matrix = {}
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i >= j:
                continue
            sim = compute_text_similarity(xdls[i], xdls[j])
            similarity_matrix[f"{m1}_vs_{m2}"] = round(sim, 4)
    
    # 2. 结构相似度
    structures = [extract_xml_structure(xdl) for xdl in xdls]
    structure_similarities = []
    
    for i in range(len(structures)):
        for j in range(i + 1, len(structures)):
            s1, s2 = structures[i], structures[j]
            if "error" in s1 or "error" in s2:
                continue
            
            # 步骤数量相似度
            step_sim = 1 - abs(s1["step_count"] - s2["step_count"]) / max(s1["step_count"], s2["step_count"], 1)
            
            # 试剂数量相似度
            reagent_sim = 1 - abs(len(s1["reagents"]) - len(s2["reagents"])) / max(len(s1["reagents"]), len(s2["reagents"]), 1)
            
            # 硬件数量相似度
            hw_sim = 1 - abs(len(s1["hardware"]) - len(s2["hardware"])) / max(len(s1["hardware"]), len(s2["hardware"]), 1)
            
            structure_similarities.append({
                "models": (models[i], models[j]),
                "step_similarity": round(step_sim, 4),
                "reagent_similarity": round(reagent_sim, 4),
                "hardware_similarity": round(hw_sim, 4),
                "avg_similarity": round((step_sim + reagent_sim + hw_sim) / 3, 4)
            })
    
    # 3. Hash 一致性
    hashes = [successful[m]["hash"] for m in models]
    unique_hashes = set(hashes)
    hash_agreement = len(unique_hashes) == 1
    
    # 4. 综合评分
    if structure_similarities:
        avg_struct_sim = sum(s["avg_similarity"] for s in structure_similarities) / len(structure_similarities)
    else:
        avg_struct_sim = 0
    
    text_sims = list(similarity_matrix.values())
    avg_text_sim = sum(text_sims) / len(text_sims) if text_sims else 0
    
    overall_score = (avg_text_sim + avg_struct_sim) / 2
    
    return {
        "status": "analyzed",
        "models_tested": models,
        "hash_agreement": hash_agreement,
        "unique_hashes": list(unique_hashes),
        "text_similarity": {
            "matrix": similarity_matrix,
            "average": round(avg_text_sim, 4),
            "min": round(min(text_sims), 4) if text_sims else 0,
            "max": round(max(text_sims), 4) if text_sims else 0
        },
        "structure_similarity": structure_similarities,
        "avg_structure_similarity": round(avg_struct_sim, 4),
        "overall_score": round(overall_score, 4),
        "passed": overall_score >= SIMILARITY_THRESHOLDS["xdl_structure"]
    }


def analyze_dag_consistency(model_results: Dict[str, Dict]) -> Dict[str, Any]:
    """分析多个模型生成的 DAG 之间的一致性"""
    
    successful = {m: r for m, r in model_results.items() if "dag" in r}
    
    if len(successful) < 2:
        return {
            "status": "insufficient_data",
            "reason": f"Only {len(successful)} successful generation(s)"
        }
    
    models = list(successful.keys())
    dags = [successful[m]["dag"] for m in models]
    
    # 1. 结构指纹对比
    structures = [extract_dag_structure(dag) for dag in dags]
    
    structure_similarities = []
    for i in range(len(structures)):
        for j in range(i + 1, len(structures)):
            s1, s2 = structures[i], structures[j]
            if "error" in s1 or "error" in s2:
                continue
            
            # 节点数量相似度
            node_sim = 1 - abs(s1["node_count"] - s2["node_count"]) / max(s1["node_count"], s2["node_count"], 1)
            
            # 边数量相似度
            edge_sim = 1 - abs(s1["edge_count"] - s2["edge_count"]) / max(s1["edge_count"], s2["edge_count"], 1)
            
            # 层级数量相似度
            layer_sim = 1 - abs(s1["layer_count"] - s2["layer_count"]) / max(s1["layer_count"], s2["layer_count"], 1)
            
            # 节点描述语义相似度
            desc_sims = []
            for d1 in s1.get("node_ids", []):
                # 找到对应的描述
                desc1 = ""
                for dag in dags:
                    for n in dag.get("graph_nodes", []):
                        if n["id"] == d1:
                            desc1 = n.get("description", "")
                            break
                
                best_match = 0
                for dag2 in dags[j:i+j+1] if i < len(dags) - 1 else dags[j+1:]:
                    for n in dag2.get("graph_nodes", []):
                        if n["id"] == d1:
                            sim = compute_text_similarity(desc1, n.get("description", ""))
                            best_match = max(best_match, sim)
                desc_sims.append(best_match)
            
            avg_desc_sim = sum(desc_sims) / len(desc_sims) if desc_sims else 0
            
            structure_similarities.append({
                "models": (models[i], models[j]),
                "node_similarity": round(node_sim, 4),
                "edge_similarity": round(edge_sim, 4),
                "layer_similarity": round(layer_sim, 4),
                "desc_semantic_similarity": round(avg_desc_sim, 4),
                "overall": round((node_sim + edge_sim + layer_sim + avg_desc_sim) / 4, 4)
            })
    
    # 2. Hash 一致性
    structure_hashes = [successful[m]["structure_hash"] for m in models]
    unique_hashes = set(structure_hashes)
    hash_agreement = len(unique_hashes) == 1
    
    # 3. 拓扑顺序一致性（如果都是 DAG）
    topological_consistencies = []
    for i, dag in enumerate(dags):
        if "graph_nodes" not in dag:
            continue
        
        try:
            import networkx as nx
            G = nx.DiGraph()
            for n in dag["graph_nodes"]:
                G.add_node(n["id"])
                for dep in n.get("dependencies", []):
                    G.add_edge(dep, n["id"])
            
            topo_order = list(nx.topological_sort(G))
            topological_consistencies.append((models[i], tuple(topo_order)))
        except:
            pass
    
    # 4. 综合评分
    if structure_similarities:
        avg_overall = sum(s["overall"] for s in structure_similarities) / len(structure_similarities)
    else:
        avg_overall = 0
    
    return {
        "status": "analyzed",
        "models_tested": models,
        "hash_agreement": hash_agreement,
        "structure_hashes": list(unique_hashes),
        "structure_similarity": structure_similarities,
        "avg_structure_similarity": round(avg_overall, 4),
        "topological_orders": topological_consistencies,
        "overall_score": round(avg_overall, 4),
        "passed": avg_overall >= SIMILARITY_THRESHOLDS["dag_structure"]
    }


# === 报告生成 ===
def generate_markdown_report(xdl_analysis: List[Dict], dag_analysis: List[Dict] = None,
                              cross_domain_results: Dict = None) -> str:
    """生成 Markdown 格式的报告"""
    
    report = []
    report.append("# Multi-Model Consistency Validation Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Models Tested:** {', '.join(xdl_analysis[0]['models_tested']) if xdl_analysis and 'models_tested' in xdl_analysis[0] else 'N/A'}")
    
    # === Section 1: XDL Consistency ===
    report.append("\n" + "="*70)
    report.append("## Layer 1: XDL Generation Consistency")
    report.append("="*70)
    
    xdl_passed = sum(1 for a in xdl_analysis if a.get("passed", False))
    xdl_avg_score = sum(a.get("overall_score", 0) for a in xdl_analysis) / len(xdl_analysis) if xdl_analysis else 0
    
    report.append(f"\n### Summary")
    report.append(f"- **Total Cases:** {len(xdl_analysis)}")
    report.append(f"- **Passed (score ≥ {SIMILARITY_THRESHOLDS['xdl_structure']}):** {xdl_passed} ({xdl_passed/len(xdl_analysis)*100:.1f}%)" if xdl_analysis else "- No data")
    report.append(f"- **Average Score:** {xdl_avg_score:.4f}")
    
    # 详细结果表格
    report.append(f"\n### Detailed Results")
    report.append(f"\n| Case ID | Hash Agreement | Text Sim (avg) | Struct Sim | Overall | Status |")
    report.append(f"|---------|----------------|----------------|-------------|---------|--------|")
    
    for a in xdl_analysis:
        case_id = a.get("case_id", "N/A")
        hash_agree = "✅" if a.get("hash_agreement", False) else "❌"
        text_sim = a.get("text_similarity", {}).get("average", "N/A")
        struct_sim = a.get("avg_structure_similarity", "N/A")
        overall = a.get("overall_score", "N/A")
        status = "✅ PASS" if a.get("passed", False) else "❌ FAIL"
        
        if isinstance(text_sim, float):
            text_sim = f"{text_sim:.3f}"
        if isinstance(struct_sim, float):
            struct_sim = f"{struct_sim:.3f}"
        if isinstance(overall, float):
            overall = f"{overall:.3f}"
        
        report.append(f"| {case_id} | {hash_agree} | {text_sim} | {struct_sim} | {overall} | {status} |")
    
    # 分歧案例详情
    failed_cases = [a for a in xdl_analysis if not a.get("passed", False)]
    if failed_cases:
        report.append(f"\n### Cases Below Threshold ({len(failed_cases)} cases)")
        for a in failed_cases[:5]:  # 最多显示5个
            report.append(f"\n**Case {a.get('case_id', 'N/A')}:**")
            report.append(f"- Overall Score: {a.get('overall_score', 0):.4f}")
            report.append(f"- Text Similarities: {json.dumps(a.get('text_similarity', {}).get('matrix', {}), indent=2)}")
    
    # === Section 2: DAG Consistency ===
    if dag_analysis:
        report.append("\n" + "="*70)
        report.append("## Layer 2: DAG Generation Consistency")
        report.append("="*70)
        
        dag_passed = sum(1 for a in dag_analysis if a.get("passed", False))
        dag_avg_score = sum(a.get("overall_score", 0) for a in dag_analysis) / len(dag_analysis) if dag_analysis else 0
        
        report.append(f"\n### Summary")
        report.append(f"- **Total Cases:** {len(dag_analysis)}")
        report.append(f"- **Passed (score ≥ {SIMILARITY_THRESHOLDS['dag_structure']}):** {dag_passed} ({dag_passed/len(dag_analysis)*100:.1f}%)" if dag_analysis else "- No data")
        report.append(f"- **Average Score:** {dag_avg_score:.4f}")
        
        # 详细结果表格
        report.append(f"\n### Detailed Results")
        report.append(f"\n| Case ID | Hash Agreement | Struct Sim | Semantic Sim | Overall | Status |")
        report.append(f"|---------|----------------|-------------|--------------|---------|--------|")
        
        for a in dag_analysis:
            case_id = a.get("case_id", "N/A")
            hash_agree = "✅" if a.get("hash_agreement", False) else "❌"
            struct_sim = a.get("avg_structure_similarity", "N/A")
            
            # 计算语义相似度平均
            struct_details = a.get("structure_similarity", [])
            if struct_details:
                sem_sim = sum(s.get("desc_semantic_similarity", 0) for s in struct_details) / len(struct_details)
            else:
                sem_sim = "N/A"
            
            overall = a.get("overall_score", "N/A")
            status = "✅ PASS" if a.get("passed", False) else "❌ FAIL"
            
            if isinstance(struct_sim, float):
                struct_sim = f"{struct_sim:.3f}"
            if isinstance(sem_sim, float):
                sem_sim = f"{sem_sim:.3f}"
            if isinstance(overall, float):
                overall = f"{overall:.3f}"
            
            report.append(f"| {case_id} | {hash_agree} | {struct_sim} | {sem_sim} | {overall} | {status} |")
    
    # === Section 3: Cross-Domain Results ===
    if cross_domain_results:
        report.append("\n" + "="*70)
        report.append("## Layer 3: Cross-Domain Generalization")
        report.append("="*70)
        
        report.append(f"\n### Summary")
        for domain, result in cross_domain_results.items():
            name = result.get("domain_name", domain)
            rate = result.get("success_rate", 0)
            success = result.get("successful_generations", 0)
            total = result.get("total_cases", 0)
            
            bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            status_icon = "✅" if rate >= 0.8 else "⚠️" if rate >= 0.5 else "❌"
            
            report.append(f"\n**{name}** {status_icon}")
            report.append(f"- Success Rate: {rate:.1%} ({success}/{total})")
            report.append(f"- [{bar}]")
    
    # === Section 4: Conclusion ===
    report.append("\n" + "="*70)
    report.append("## Conclusion & Recommendations")
    report.append("="*70)
    
    overall_pass = xdl_passed / len(xdl_analysis) if xdl_analysis else 0
    if dag_analysis:
        dag_overall_pass = dag_passed / len(dag_analysis)
        combined_pass = (overall_pass + dag_overall_pass) / 2
    else:
        combined_pass = overall_pass
    
    if combined_pass >= 0.90:
        conclusion = "**EXCELLENT**: The pipeline shows strong cross-model consistency."
        recommendation = "The method is robust and does not depend on a single model's idiosyncrasies."
    elif combined_pass >= 0.75:
        conclusion = "**GOOD**: The pipeline shows reasonable consistency with minor variations."
        recommendation = "Consider investigating the failing cases to improve robustness."
    elif combined_pass >= 0.50:
        conclusion = "**MODERATE**: Significant variations observed across models."
        recommendation = "Further analysis needed. May require prompt refinement or additional constraints."
    else:
        conclusion = "**CONCERNING**: Low consistency across models."
        recommendation = "The pipeline may be too sensitive to model choice. Review prompts and generation strategy."
    
    report.append(f"\n{conclusion}")
    report.append(f"\n**Combined Pass Rate:** {combined_pass:.1%}")
    report.append(f"\n**Recommendation:** {recommendation}")
    
    return "\n".join(report)


# === 主程序 ===
def main():
    parser = argparse.ArgumentParser(description="Consistency Checker & Report Generator")
    parser.add_argument("--xdl-results", type=str, help="XDL generation results JSON file")
    parser.add_argument("--dag-results", type=str, help="DAG generation results JSON file")
    parser.add_argument("--cross-domain", type=str, help="Cross-domain test results JSON file")
    parser.add_argument("--output", type=str, default="consistency_report.md", help="Output report file")
    parser.add_argument("--threshold-xdl", type=float, default=SIMILARITY_THRESHOLDS["xdl_structure"],
                        help="XDL similarity threshold")
    parser.add_argument("--threshold-dag", type=float, default=SIMILARITY_THRESHOLDS["dag_structure"],
                        help="DAG similarity threshold")
    
    args = parser.parse_args()
    
    # 更新阈值
    if args.threshold_xdl:
        SIMILARITY_THRESHOLDS["xdl_structure"] = args.threshold_xdl
    if args.threshold_dag:
        SIMILARITY_THRESHOLDS["dag_structure"] = args.threshold_dag
    
    xdl_analysis = []
    dag_analysis = []
    cross_domain_results = None
    
    # 分析 XDL 结果
    if args.xdl_results:
        print(f"Loading XDL results from: {args.xdl_results}")
        with open(args.xdl_results, "r", encoding="utf-8") as f:
            xdl_data = json.load(f)
        
        for case in xdl_data:
            analysis = analyze_xdl_consistency(case.get("model_results", {}))
            analysis["case_id"] = case.get("case_id", case.get("id", "N/A"))
            if "models_tested" not in analysis:
                analysis["models_tested"] = list(case.get("model_results", {}).keys())
            xdl_analysis.append(analysis)
        
        print(f"Analyzed {len(xdl_analysis)} XDL cases")
    
    # 分析 DAG 结果
    if args.dag_results:
        print(f"Loading DAG results from: {args.dag_results}")
        with open(args.dag_results, "r", encoding="utf-8") as f:
            dag_data = json.load(f)
        
        for case in dag_data:
            analysis = analyze_dag_consistency(case.get("model_results", {}))
            analysis["case_id"] = case.get("case_id", "N/A")
            if "models_tested" not in analysis:
                analysis["models_tested"] = list(case.get("model_results", {}).keys())
            dag_analysis.append(analysis)
        
        print(f"Analyzed {len(dag_analysis)} DAG cases")
    
    # 加载跨领域测试结果
    if args.cross_domain:
        print(f"Loading cross-domain results from: {args.cross_domain}")
        with open(args.cross_domain, "r", encoding="utf-8") as f:
            cross_domain_results = json.load(f)
    
    # 生成报告
    report = generate_markdown_report(xdl_analysis, dag_analysis, cross_domain_results)
    
    # 保存报告
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {args.output}")
    
    # 同时打印到控制台
    print("\n" + "="*70)
    print("REPORT PREVIEW")
    print("="*70)
    print(report[:3000])  # 打印前3000字符
    if len(report) > 3000:
        print(f"\n... (truncated, full report in {args.output})")


if __name__ == "__main__":
    main()
