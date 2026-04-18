"""
multi_model_generator.py
========================
多模型 XDL & DAG 生成与一致性验证框架

支持：
1. 多模型并行/串行生成 XDL
2. 多模型从同一 XDL 生成 DAG
3. 自动一致性检查
4. 跨领域泛化测试

Usage:
    python multi_model_generator.py --mode xdl --input test_cases.json
    python multi_model_generator.py --mode dag --input experiments.xdl
    python multi_model_generator.py --mode full --input test_cases.json
"""

import os
import sys
import json
import hashlib
import argparse
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
import difflib

# === 配置 ===
@dataclass
class ModelConfig:
    name: str
    provider: str  # "openai", "anthropic", "deepseek"
    model_id: str
    api_key_env: str
    base_url: Optional[str] = None

# 支持的模型列表
MODELS = {
    "gpt-4.1": ModelConfig(
        name="GPT-4.1",
        provider="openai", 
        model_id="gpt-4.1",
        api_key_env="OPENAI_API_KEY"
    ),
    "claude-3-5-sonnet": ModelConfig(
        name="Claude-3.5-Sonnet",
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        api_key_env="ANTHROPIC_API_KEY"
    ),
    "deepseek-chat": ModelConfig(
        name="DeepSeek-Chat",
        provider="deepseek",
        model_id="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com"
    ),
    "qwen-plus": ModelConfig(
        name="Qwen-Plus",
        provider="openai",  # Qwen 也兼容 OpenAI API
        model_id="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    "qwq-32b": ModelConfig(
        name="QwQ-32B",
        provider="openai",
        model_id="qwq-32b",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
}

# 默认测试的模型组合
DEFAULT_MODELS = ["gpt-4.1", "deepseek-chat", "qwen-plus"]


# === 辅助函数 ===
def get_client(model_config: ModelConfig):
    """根据模型配置创建 API 客户端"""
    api_key = os.environ.get(model_config.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key for {model_config.name}. Set {model_config.api_key_env}")
    
    if model_config.provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=model_config.base_url)
    elif model_config.provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)
    elif model_config.provider == "deepseek":
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=model_config.base_url or "https://api.deepseek.com")
    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")


def compute_text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度 (0-1)"""
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def compute_hash(text: str) -> str:
    """计算文本的 MD5 hash"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# === XDL 生成相关 ===
XDL_SPEC_SYSTEM_PROMPT = """
You are an expert in chemical XDL synthesis file generation (v3.0).

[OFFICIAL SPECIFICATION - STRICTLY FOLLOW]
1. Structure: Wrap <Hardware>, <Reagents>, and <Procedure> inside a <Synthesis> tag
2. Hardware: Define all vessels (test_tube, beaker, water_trough, etc.)
3. Reagents: List all chemicals with name, formula, and state
4. Procedure: Use valid actions only: Add, Transfer, Heat, Cool, Wait, Insert, Attach, Wash, Filter
5. Anti-Hallucination: Do NOT invent tags like <MeasureObservation> or <Check>
6. Action constraints:
   - <Add>: Liquid -> volume (ml), Solid -> mass (g)
   - <Transfer>: Only for liquids between vessels
7. Use <Wait time="..."/> for qualitative observations

Output: Pure XML only, no markdown.
"""

XDL_GENERATION_USER_PROMPT = """
Convert this experiment description into XDL v3.0 XML:
"{chem_text}"

Requirements:
- Parse the 'goal' carefully
- Ensure every vessel used in Procedure is defined in Hardware
- Output pure XML (no markdown wrappers)
"""


def generate_xdl_single(client, model_name: str, chem_text: str, xdl_spec: str = None) -> Dict[str, Any]:
    """用单个模型生成 XDL"""
    system_prompt = xdl_spec or XDL_SPEC_SYSTEM_PROMPT
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": XDL_GENERATION_USER_PROMPT.format(chem_text=chem_text)}
    ]
    
    # 不同 provider 的调用方式略有不同
    if "claude" in model_name.lower():
        # Anthropic
        response = client.messages.create(
            model=MODELS[model_name].model_id,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": XDL_GENERATION_USER_PROMPT.format(chem_text=chem_text)}]
        )
        xdl_text = response.content[0].text
    else:
        # OpenAI compatible
        response = client.chat.completions.create(
            model=MODELS[model_name].model_id,
            messages=messages,
            temperature=0.2,
            max_tokens=4096
        )
        xdl_text = response.choices[0].message.content
    
    # 清理 XML
    xdl_text = xdl_text.replace("```xml", "").replace("```", "").strip()
    if "<?xml" not in xdl_text:
        xdl_text = '<?xml version="1.0" encoding="UTF-8"?>\n' + xdl_text
    
    return {
        "xdl": xdl_text,
        "hash": compute_hash(xdl_text),
        "model": model_name,
        "raw_response": xdl_text
    }


# === DAG 生成相关 ===
DAG_COMPILER_SYSTEM_PROMPT = """
# Role
You are a chemical experiment logic compiler. Your task is to convert XDL experiment steps 
into a structured logical milestone graph (DAG).

# Context
The XDL contains step IDs like step_0, step_1, etc.

# Task
1. Cluster physical operations into logical milestones
2. Map each milestone to its required step IDs
3. Define dependencies between milestones

# Output Format (JSON Only)
{{
    "graph_nodes": [
        {{
            "id": "node descriptive name",
            "description": "What this milestone achieves",
            "required_steps": ["step_0", "step_1"],
            "dependencies": ["parent_node_id"]
        }}
    ]
}}

Output pure JSON only, no markdown.
"""


def generate_dag_single(client, model_name: str, enriched_xdl: str) -> Dict[str, Any]:
    """用单个模型从 XDL 生成 DAG"""
    messages = [
        {"role": "system", "content": DAG_COMPILER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Generate DAG for this XDL:\n{enriched_xdl}"}
    ]
    
    if "claude" in model_name.lower():
        response = client.messages.create(
            model=MODELS[model_name].model_id,
            max_tokens=4096,
            system=DAG_COMPILER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Generate DAG for this XDL:\n{enriched_xdl}"}]
        )
        dag_text = response.content[0].text
    else:
        response = client.chat.completions.create(
            model=MODELS[model_name].model_id,
            messages=messages,
            temperature=0.1,
            max_tokens=4096
        )
        dag_text = response.choices[0].message.content
    
    # 解析 JSON
    dag_text = dag_text.strip()
    if "```json" in dag_text:
        dag_text = dag_text.split("```json")[1].split("```")[0]
    elif "```" in dag_text:
        dag_text = dag_text.split("```")[1].split("```")[0]
    
    try:
        dag_json = json.loads(dag_text)
    except:
        dag_json = {"graph_nodes": [], "raw": dag_text}
    
    # 计算 DAG 的结构指纹
    if "graph_nodes" in dag_json:
        nodes = dag_json["graph_nodes"]
        node_ids = sorted([n["id"] for n in nodes])
        edges = []
        for n in nodes:
            for dep in n.get("dependencies", []):
                edges.append((dep, n["id"]))
        structure_hash = compute_hash(str(sorted(edges)))
    else:
        structure_hash = compute_hash(dag_text)
    
    return {
        "dag": dag_json,
        "structure_hash": structure_hash,
        "model": model_name,
        "raw_response": dag_text
    }


# === 一致性验证 ===
@dataclass
class ConsistencyReport:
    """一致性验证报告"""
    total_cases: int
    xdl_agreement_rate: float
    dag_agreement_rate: float
    xdl_fully_agreed_cases: List[int]  # case indices
    xdl_disagreed_cases: List[Dict]    # details
    dag_fully_agreed_cases: List[int]
    dag_disagreed_cases: List[Dict]
    
    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "xdl_agreement_rate": self.xdl_agreement_rate,
            "dag_agreement_rate": self.dag_agreement_rate,
            "xdl_fully_agreed": len(self.xdl_fully_agreed_cases),
            "xdl_disagreed": len(self.xdl_disagreed_cases),
            "dag_fully_agreed": len(self.dag_fully_agreed_cases),
            "dag_disagreed": len(self.dag_disagreed_cases),
            "xdl_disagreement_details": self.xdl_disagreed_cases[:5],  # 前5个详情
            "dag_disagreement_details": self.dag_disagreed_cases[:5]
        }


def check_xdl_equivalence(results: List[Dict]) -> Tuple[float, List[Dict]]:
    """
    检查多模型 XDL 生成的一致性
    返回: (一致率, 不一致案例详情)
    """
    hashes = [r["hash"] for r in results]
    unique_hashes = set(hashes)
    
    if len(unique_hashes) == 1:
        return 1.0, []
    
    # 计算两两相似度
    similarities = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            sim = compute_text_similarity(results[i]["xdl"], results[j]["xdl"])
            similarities.append((results[i]["model"], results[j]["model"], sim))
    
    avg_similarity = sum(s[2] for s in similarities) / len(similarities)
    
    # 找出分歧最大的案例
    disagreement_detail = {
        "hashes": Counter(hashes),
        "pairwise_similarities": similarities,
        "avg_similarity": avg_similarity
    }
    
    return avg_similarity, [disagreement_detail]


def check_dag_equivalence(results: List[Dict]) -> Tuple[float, List[Dict]]:
    """
    检查多模型 DAG 生成的一致性
    返回: (一致率, 不一致案例详情)
    """
    structure_hashes = [r["structure_hash"] for r in results]
    unique_hashes = set(structure_hashes)
    
    if len(unique_hashes) == 1:
        return 1.0, []
    
    # 计算结构相似度（考虑节点对应关系）
    all_similarities = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            sim = compute_dag_similarity(results[i]["dag"], results[j]["dag"])
            all_similarities.append((results[i]["model"], results[j]["model"], sim))
    
    avg_similarity = sum(s[2] for s in all_similarities) / len(all_similarities) if all_similarities else 0
    
    return avg_similarity, [{"pairwise": all_similarities, "avg": avg_similarity}]


def compute_dag_similarity(dag1: Dict, dag2: Dict) -> float:
    """计算两个 DAG 的结构相似度"""
    if "graph_nodes" not in dag1 or "graph_nodes" not in dag2:
        return 0.0
    
    nodes1 = dag1["graph_nodes"]
    nodes2 = dag2["graph_nodes"]
    
    # 1. 节点数量相似度
    node_count_sim = 1 - abs(len(nodes1) - len(nodes2)) / max(len(nodes1), len(nodes2), 1)
    
    # 2. 边数量相似度
    edges1 = []
    for n in nodes1:
        for dep in n.get("dependencies", []):
            edges1.append((dep, n["id"]))
    edges2 = []
    for n in nodes2:
        for dep in n.get("dependencies", []):
            edges2.append((dep, n["id"]))
    
    edge_count_sim = 1 - abs(len(edges1) - len(edges2)) / max(len(edges1), len(edges2), 1)
    
    # 3. 节点描述相似度（通过字符串匹配）
    descs1 = [n["description"].lower() for n in nodes1]
    descs2 = [n["description"].lower() for n in nodes2]
    
    desc_similarities = []
    for d1 in descs1:
        best_match = max([compute_text_similarity(d1, d2) for d2 in descs2], default=0)
        desc_similarities.append(best_match)
    desc_sim = sum(desc_similarities) / len(desc_similarities) if desc_similarities else 0
    
    # 综合相似度
    return (node_count_sim + edge_count_sim + desc_sim) / 3


# === 主运行逻辑 ===
class MultiModelGenerator:
    """多模型生成与验证主类"""
    
    def __init__(self, models: List[str] = None):
        self.models = models or DEFAULT_MODELS
        self.clients = {}
        self._init_clients()
    
    def _init_clients(self):
        """初始化各模型的 API 客户端"""
        for model_name in self.models:
            if model_name not in MODELS:
                print(f"⚠️ Unknown model: {model_name}, skipping...")
                continue
            try:
                self.clients[model_name] = get_client(MODELS[model_name])
                print(f"✅ Initialized {MODELS[model_name].name}")
            except Exception as e:
                print(f"❌ Failed to init {model_name}: {e}")
    
    def run_xdl_generation(self, test_cases: List[Dict]) -> List[Dict]:
        """
        对所有测试案例运行多模型 XDL 生成
        test_cases: [{"id": 1, "chem_text": "..."}, ...]
        """
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"\n[{i+1}/{len(test_cases)}] Processing case: {case.get('id', i)}")
            case_results = {
                "case_id": case.get("id", i),
                "chem_text": case["chem_text"],
                "model_results": {}
            }
            
            for model_name in self.models:
                if model_name not in self.clients:
                    continue
                    
                print(f"  → {model_name}...", end=" ")
                try:
                    result = generate_xdl_single(
                        self.clients[model_name],
                        model_name,
                        case["chem_text"]
                    )
                    case_results["model_results"][model_name] = result
                    print(f"✓ (hash: {result['hash'][:8]}...)")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    case_results["model_results"][model_name] = {"error": str(e)}
            
            # 一致性检查
            successful_results = [
                r for r in case_results["model_results"].values() 
                if "xdl" in r
            ]
            if len(successful_results) >= 2:
                agreement, details = check_xdl_equivalence(successful_results)
                case_results["xdl_agreement"] = agreement
                case_results["xdl_disagreement_detail"] = details
            else:
                case_results["xdl_agreement"] = 0.0
            
            results.append(case_results)
        
        return results
    
    def run_dag_generation(self, xdl_results: List[Dict]) -> List[Dict]:
        """
        对已生成的 XDL 运行多模型 DAG 生成
        """
        results = []
        
        for i, xdl_result in enumerate(xdl_results):
            print(f"\n[{i+1}/{len(xdl_results)}] Generating DAG for case: {xdl_result.get('case_id', i)}")
            
            # 选取第一个成功生成的 XDL 作为输入
            xdl_input = None
            source_model = None
            for model_name, model_result in xdl_result["model_results"].items():
                if "xdl" in model_result:
                    xdl_input = model_result["xdl"]
                    source_model = model_name
                    break
            
            if not xdl_input:
                print("  ⚠️ No valid XDL found, skipping DAG generation")
                continue
            
            print(f"  Using XDL from {source_model}")
            
            case_results = {
                "case_id": xdl_result["case_id"],
                "source_xdl_model": source_model,
                "model_results": {}
            }
            
            for model_name in self.models:
                if model_name not in self.clients:
                    continue
                    
                print(f"  → {model_name}...", end=" ")
                try:
                    result = generate_dag_single(
                        self.clients[model_name],
                        model_name,
                        xdl_input
                    )
                    case_results["model_results"][model_name] = result
                    print(f"✓ (structure_hash: {result['structure_hash'][:8]}...)")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    case_results["model_results"][model_name] = {"error": str(e)}
            
            # 一致性检查
            successful_results = [
                r for r in case_results["model_results"].values()
                if "dag" in r
            ]
            if len(successful_results) >= 2:
                agreement, details = check_dag_equivalence(successful_results)
                case_results["dag_agreement"] = agreement
                case_results["dag_disagreement_detail"] = details
            else:
                case_results["dag_agreement"] = 0.0
            
            results.append(case_results)
        
        return results
    
    def generate_full_report(self, xdl_results: List[Dict], dag_results: List[Dict] = None) -> Dict:
        """生成完整的验证报告"""
        
        # XDL 一致性统计
        xdl_agreed = sum(1 for r in xdl_results if r.get("xdl_agreement", 0) >= 0.95)
        xdl_agreement_rate = xdl_agreed / len(xdl_results) if xdl_results else 0
        
        avg_xdl_agreement = sum(r.get("xdl_agreement", 0) for r in xdl_results) / len(xdl_results) if xdl_results else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "models_tested": self.models,
            "total_cases": len(xdl_results),
            "xdl_consistency": {
                "fully_agreed_cases": xdl_agreed,
                "agreement_rate": xdl_agreement_rate,
                "avg_agreement_score": avg_xdl_agreement,
                "cases_below_threshold": [
                    {"case_id": r["case_id"], "agreement": r.get("xdl_agreement", 0)}
                    for r in xdl_results if r.get("xdl_agreement", 0) < 0.95
                ]
            },
            "dag_consistency": None
        }
        
        if dag_results:
            dag_agreed = sum(1 for r in dag_results if r.get("dag_agreement", 0) >= 0.90)
            dag_agreement_rate = dag_agreed / len(dag_results) if dag_results else 0
            avg_dag_agreement = sum(r.get("dag_agreement", 0) for r in dag_results) / len(dag_results) if dag_results else 0
            
            report["dag_consistency"] = {
                "fully_agreed_cases": dag_agreed,
                "agreement_rate": dag_agreement_rate,
                "avg_agreement_score": avg_dag_agreement,
                "cases_below_threshold": [
                    {"case_id": r["case_id"], "agreement": r.get("dag_agreement", 0)}
                    for r in dag_results if r.get("dag_agreement", 0) < 0.90
                ]
            }
        
        return report


# === 跨领域泛化测试 ===
@dataclass
class DomainGeneralizationTest:
    """跨领域泛化测试"""
    name: str
    description: str
    test_cases: List[Dict]
    expected_capabilities: List[str] = field(default_factory=list)


# 预定义的跨领域测试集
DOMAIN_TESTS = {
    "chemistry_basic": {
        "name": "Chemistry - Basic Reactions",
        "description": "Basic chemistry experiments (acid-base, precipitation)",
        "examples": [
            {"chem_text": "Mix dilute HCl with NaOH solution, observe temperature change."},
            {"chem_text": "Add AgNO3 solution to NaCl solution, observe white precipitate formation."},
            {"chem_text": "Heat copper sulfate crystals until they turn white, then add water to restore blue color."},
        ]
    },
    "chemistry_physics": {
        "name": "Chemistry - Physical Separation",
        "description": "Physical separation techniques in chemistry",
        "examples": [
            {"chem_text": "Separate a mixture of sand and salt by dissolving, filtration, and evaporation."},
            {"chem_text": "Extract iodine from a mixture using liquid-liquid extraction with immiscible solvents."},
            {"chem_text": "Separate components of ink using paper chromatography."},
        ]
    },
    "physics_optics": {
        "name": "Physics - Optics",
        "description": "Light and optics experiments (NEW DOMAIN)",
        "examples": [
            {"chem_text": "Set up a simple optical bench to measure focal length of a convex lens."},
            {"chem_text": "Observe and measure the angle of refraction for light passing from air into water."},
            {"chem_text": "Verify the law of reflection using a plane mirror and optical pins."},
        ]
    },
    "physics_electricity": {
        "name": "Physics - Electricity",
        "description": "Electrical circuits experiments (NEW DOMAIN)",
        "examples": [
            {"chem_text": "Set up a simple circuit with a battery, resistor, and LED in series to verify Ohm's law."},
            {"chem_text": "Measure the equivalent resistance of two resistors connected in parallel."},
            {"chem_text": "Build a voltage divider circuit and measure output voltages at different points."},
        ]
    },
    "biology_microscopy": {
        "name": "Biology - Microscopy",
        "description": "Biological microscopy techniques (NEW DOMAIN)",
        "examples": [
            {"chem_text": "Prepare a wet mount of onion cells and observe under microscope after staining."},
            {"chem_text": "Observe the phenomenon of plasmolysis in plant cells using salt solution."},
            {"chem_text": "Identify and sketch different types of epithelial cells from a prepared slide."},
        ]
    }
}


def run_cross_domain_test(models: List[str], domain_key: str = None) -> Dict:
    """
    运行跨领域泛化测试
    """
    generator = MultiModelGenerator(models)
    
    results = {}
    
    domains_to_test = [domain_key] if domain_key else list(DOMAIN_TESTS.keys())
    
    for domain in domains_to_test:
        if domain not in DOMAIN_TESTS:
            continue
            
        test_set = DOMAIN_TESTS[domain]
        print(f"\n{'='*60}")
        print(f"Testing Domain: {test_set['name']}")
        print(f"Description: {test_set['description']}")
        print(f"{'='*60}")
        
        domain_results = generator.run_xdl_generation(test_set["examples"])
        
        # 计算该领域的生成成功率
        success_count = 0
        for case_result in domain_results:
            if any("xdl" in r for r in case_result.get("model_results", {}).values()):
                success_count += 1
        
        results[domain] = {
            "domain_name": test_set["name"],
            "total_cases": len(test_set["examples"]),
            "successful_generations": success_count,
            "success_rate": success_count / len(test_set["examples"]),
            "detailed_results": domain_results
        }
    
    return results


# === 命令行入口 ===
def main():
    parser = argparse.ArgumentParser(description="Multi-Model XDL & DAG Generator")
    parser.add_argument("--mode", choices=["xdl", "dag", "full", "cross_domain"], default="full",
                        help="Generation mode")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON file with test cases")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                        help=f"Comma-separated model list. Available: {list(MODELS.keys())}")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--xdl-results", type=str, default=None,
                        help="Previous XDL results file (for DAG mode)")
    
    args = parser.parse_args()
    
    # 解析模型列表
    models = [m.strip() for m in args.models.split(",")]
    print(f"Models to test: {models}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.mode in ["xdl", "full"]:
        # 加载测试用例
        with open(args.input, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        
        print(f"Loaded {len(test_cases)} test cases")
        
        # 运行 XDL 生成
        generator = MultiModelGenerator(models)
        xdl_results = generator.run_xdl_generation(test_cases)
        
        # 保存 XDL 结果
        xdl_output = f"{args.output}/xdl_results_{timestamp}.json"
        with open(xdl_output, "w", encoding="utf-8") as f:
            json.dump(xdl_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ XDL results saved to: {xdl_output}")
        
        # 生成报告
        if args.mode == "full":
            dag_results = generator.run_dag_generation(xdl_results)
            
            dag_output = f"{args.output}/dag_results_{timestamp}.json"
            with open(dag_output, "w", encoding="utf-8") as f:
                json.dump(dag_results, f, ensure_ascii=False, indent=2)
            
            report = generator.generate_full_report(xdl_results, dag_results)
            report_output = f"{args.output}/validation_report_{timestamp}.json"
            with open(report_output, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"✅ DAG results saved to: {dag_output}")
            print(f"✅ Validation report saved to: {report_output}")
            
            # 打印摘要
            print("\n" + "="*60)
            print("VALIDATION SUMMARY")
            print("="*60)
            print(f"Total cases: {report['total_cases']}")
            print(f"Models tested: {report['models_tested']}")
            print(f"\nXDL Consistency:")
            print(f"  - Agreement Rate: {report['xdl_consistency']['agreement_rate']:.1%}")
            print(f"  - Avg Agreement Score: {report['xdl_consistency']['avg_agreement_score']:.3f}")
            if report['xdl_consistency']['cases_below_threshold']:
                print(f"  - Cases below 0.95: {len(report['xdl_consistency']['cases_below_threshold'])}")
            
            if report['dag_consistency']:
                print(f"\nDAG Consistency:")
                print(f"  - Agreement Rate: {report['dag_consistency']['agreement_rate']:.1%}")
                print(f"  - Avg Agreement Score: {report['dag_consistency']['avg_agreement_score']:.3f}")
    
    elif args.mode == "dag":
        if not args.xdl_results:
            print("❌ --xdl-results is required for DAG mode")
            return
        
        with open(args.xdl_results, "r", encoding="utf-8") as f:
            xdl_results = json.load(f)
        
        generator = MultiModelGenerator(models)
        dag_results = generator.run_dag_generation(xdl_results)
        
        dag_output = f"{args.output}/dag_results_{timestamp}.json"
        with open(dag_output, "w", encoding="utf-8") as f:
            json.dump(dag_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ DAG results saved to: {dag_output}")
    
    elif args.mode == "cross_domain":
        domain = None  # 测试所有领域
        if os.path.exists(args.input):
            # 如果提供了输入文件，作为单个领域测试
            with open(args.input, "r", encoding="utf-8") as f:
                custom_cases = json.load(f)
            domain_results = run_cross_domain_test(models)
        else:
            # 使用预定义领域测试
            domain_results = run_cross_domain_test(models)
        
        # 保存跨领域测试结果
        cross_output = f"{args.output}/cross_domain_results_{timestamp}.json"
        with open(cross_output, "w", encoding="utf-8") as f:
            json.dump(domain_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Cross-domain results saved to: {cross_output}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("CROSS-DOMAIN GENERALIZATION SUMMARY")
        print("="*60)
        for domain, result in domain_results.items():
            print(f"\n{result['domain_name']}:")
            print(f"  - Success Rate: {result['success_rate']:.1%} ({result['successful_generations']}/{result['total_cases']})")


if __name__ == "__main__":
    main()
