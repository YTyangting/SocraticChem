import os
import glob
import json
import re
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from tqdm import tqdm
from colorama import Fore, Style, init
from config import Config

# === 初始化 ===
init(autoreset=True)

# === 配置区域 ===
# 请在此处填入你的 API Key，或者设置环境变量 OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", Config.OPENAI_API_KEY)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", Config.OPENAI_BASE_URL) # 如果用中转，请修改此项
MODEL_NAME = "gpt-4.1"  # 建议使用 gpt-4o 或 gpt-4-turbo 以保证化学推理的准确性

# 路径配置
INPUT_DIR = "experiments"    # 放 XDL 文件的目录
OUTPUT_DIR = "database"     # 输出 JSON 的目录
BATCH_SIZE = 5              # 每次发给 GPT 处理的 XDL 文件数量

# === 核心逻辑 ===

try:
    from openai import OpenAI
except ImportError:
    print(Fore.RED + "错误: 请先安装 openai 库: pip install openai")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

class ChemKnowledgeMiner:
    def __init__(self):
        self.ensure_directories()
        
        # 内存中的数据库缓存
        self.master_reagent_map = {}
        self.master_substances = {}
        self.master_reactions = []
        self.seen_reaction_equations = set()

    def ensure_directories(self):
        if not os.path.exists(INPUT_DIR):
            os.makedirs(INPUT_DIR)
            print(Fore.YELLOW + f"创建了输入目录: {INPUT_DIR} (请将 .xdl 文件放入此处)")
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

    def load_existing_db(self):
        """加载已有的数据，避免重复或覆盖"""
        print(Fore.CYAN + "正在检查现有数据库...")
        
        # 1. Reagent Map
        path = os.path.join(OUTPUT_DIR, "reagent_map.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.master_reagent_map = json.load(f)
        
        # 2. Substances
        path = os.path.join(OUTPUT_DIR, "substances.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.master_substances = json.load(f)

        # 3. Reactions
        path = os.path.join(OUTPUT_DIR, "reactions.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.master_reactions = json.load(f)
                # 建立哈希集用于去重
                for r in self.master_reactions:
                    if "equation" in r:
                        self.seen_reaction_equations.add(r["equation"].replace(" ", ""))

        print(Fore.GREEN + f"已加载: {len(self.master_reagent_map)} 映射, {len(self.master_substances)} 物质, {len(self.master_reactions)} 反应")

    def parse_xdl_batch(self, file_paths: List[str]) -> List[Dict]:
        """批量解析 XDL 提取元数据"""
        batch_data = []
        for fp in file_paths:
            try:
                tree = ET.parse(fp)
                root = tree.getroot()
                
                # 提取元数据
                meta = root.find(".//Metadata")
                title = meta.get("title", "Unknown Experiment") if meta is not None else "Unknown"
                goal = meta.get("goal", "") if meta is not None else ""
                
                # 提取试剂
                reagents = []
                for r in root.findall(".//Reagents/Reagent"):
                    reagents.append(r.get("name", "Unknown"))
                
                batch_data.append({
                    "file": os.path.basename(fp),
                    "title": title,
                    "goal": goal,
                    "reagents": reagents
                })
            except Exception as e:
                print(Fore.RED + f"解析 {fp} 失败: {e}")
        return batch_data

    def construct_prompt(self, batch_data: List[Dict]) -> str:
        """构建生产级 Prompt，强制 Schema 对齐"""
        
        # 将批次数据转为文本
        experiments_text = json.dumps(batch_data, indent=2, ensure_ascii=False)
        
        return f"""
You are an Expert Chemical Database Engineer.
Your task is to analyze the following chemistry experiments (XDL metadata) and generate the configuration data for a physics-based chemical simulation engine.

# INPUT EXPERIMENTS
{experiments_text}

# REQUIRED OUTPUT FORMAT (JSON)
You must output a SINGLE valid JSON object with exactly these three keys:
1. "reagent_map": Map natural language names from input to strict chemical formulas.
2. "substances": Physics properties for ALL formulas (reactants AND products).
3. "reactions": Balanced chemical equations and kinetics.

# SCHEMA SPECIFICATIONS

## 1. reagent_map
Key: Reagent name in XDL (e.g., "Marble", "Dilute HCl").
Value: Standard Chemical Formula (e.g., "CaCO3", "HCl").
*Note: Treat dilute acids as the solute formula (HCl).*

## 2. substances
Key: Chemical Formula.
Value Object Schema:
{{
  "molar_mass": float,
  "state": "s" (solid) | "l" (liquid/aq) | "g" (gas),
  "type": "salt" | "oxide" | "acid" | "base" | "metal" | "solvent",
  "solubility": float (g/100g H2O, use -1 for infinite/miscible, 0.0001 for insoluble),
  "visual": {{
    "base_color": "string" (e.g., "无色", "蓝", "紫红"),
    "state_solid_desc": "string" (e.g., "白色粉末"),
    "solution_rgb": [r, g, b] (approximate color),
    "intensity_factor": float (0.0 for colorless, 1.0 for normal, 10.0 for strong dyes like KMnO4),
    "is_transparent": boolean
  }}
}}

## 3. reactions
List of Objects Schema:
{{
  "equation": "Reactants -> Products" (Must be chemically BALANCED),
  "type": "decomposition" | "neutralization" | "combustion" | "replacement" | "double_decomposition",
  "temp_threshold": float (Celsius, use -273.0 for spontaneous),
  "exothermic": float (kJ/mol, positive means heat release/exothermic, negative means endothermic),
  "phenomena": "string" (Brief Chinese description of visible effects like bubbles, precipitate, color change)
}}

# CRITICAL RULES
1. **Infer Products**: If input mentions "Marble + HCl", you MUST include "CaCl2", "H2O", "CO2" in the `substances` section, even if they aren't in the input list.
2. **Missing Properties**: Estimate physical properties (solubility, color) based on chemical truth.
3. **Language**: Use Chinese for visual descriptions and phenomena.
4. **Valid JSON**: Do not output markdown code blocks, just the raw JSON string if possible, or wrap in ```json```.
"""

    def call_llm(self, prompt: str) -> Dict:
        """调用 OpenAI API，含重试逻辑"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a precise chemistry engine configuration assistant. Output valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1, # 低温度确保事实准确性
                    response_format={"type": "json_object"} # 强制 JSON 模式
                )
                content = response.choices[0].message.content
                return self.clean_and_parse_json(content)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(Fore.YELLOW + f"API 调用失败 ({e})，正在重试 {attempt+1}/{max_retries}...")
                    time.sleep(2)
                else:
                    print(Fore.RED + f"API 调用最终失败: {e}")
                    return {}
        return {}

    def clean_and_parse_json(self, text: str) -> Dict:
        """清洗 LLM 返回的可能包含 Markdown 的 JSON"""
        text = text.strip()
        # 移除 ```json 和 ```
        if text.startswith("```"):
            text = re.sub(r"^```(json)?", "", text)
            text = re.sub(r"```$", "", text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(Fore.RED + "JSON 解析失败，原始返回如下:")
            print(text[:200] + "...")
            return {}

    def merge_data(self, new_data: Dict):
        """将新生成的数据合并到主数据库"""
        if not new_data: return

        # 1. Merge Reagent Map
        new_map = new_data.get("reagent_map", {})
        for k, v in new_map.items():
            if k not in self.master_reagent_map:
                self.master_reagent_map[k] = v

        # 2. Merge Substances
        new_subs = new_data.get("substances", {})
        for k, v in new_subs.items():
            # 简单的策略：如果不存在，则添加。如果存在，假设旧的准确（或者你可以改为覆盖）
            if k not in self.master_substances:
                self.master_substances[k] = v

        # 3. Merge Reactions
        new_rxns = new_data.get("reactions", [])
        for rxn in new_rxns:
            eq = rxn.get("equation", "").strip()
            # 简单去重：移除空格后比较字符串
            eq_key = eq.replace(" ", "")
            if eq_key and eq_key not in self.seen_reaction_equations:
                self.master_reactions.append(rxn)
                self.seen_reaction_equations.add(eq_key)

    def save_db(self):
        """保存最终结果"""
        print(Fore.CYAN + "\n正在保存数据库...")
        
        def save_json(filename, data):
            path = os.path.join(OUTPUT_DIR, filename)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(Fore.GREEN + f"已保存: {path}")

        save_json("reagent_map.json", self.master_reagent_map)
        save_json("substances.json", self.master_substances)
        save_json("reactions.json", self.master_reactions)

    def run(self):
        # 1. 扫描文件
        xdl_files = glob.glob(os.path.join(INPUT_DIR, "*.xdl"))
        if not xdl_files:
            print(Fore.RED + f"在 {INPUT_DIR} 中未找到 .xdl 文件。")
            return

        print(Fore.YELLOW + f"找到 {len(xdl_files)} 个 XDL 文件，准备处理...")
        
        # 2. 加载旧数据
        self.load_existing_db()

        # 3. 批处理循环
        # 使用 tqdm 显示进度条
        for i in tqdm(range(0, len(xdl_files), BATCH_SIZE), desc="Processing Batches"):
            batch_files = xdl_files[i : i + BATCH_SIZE]
            
            # A. 解析 XDL
            batch_meta = self.parse_xdl_batch(batch_files)
            if not batch_meta: continue

            # B. 构建 Prompt
            prompt = self.construct_prompt(batch_meta)
            
            # C. 调用 LLM
            # print(f"正在向 GPT 发送 {len(batch_meta)} 个实验...") 
            result = self.call_llm(prompt)
            
            # D. 合并数据
            self.merge_data(result)

        # 4. 保存
        self.save_db()
        print(Fore.MAGENTA + "=== 处理完成 ===")
        print(Fore.WHITE + f"总计包含: {len(self.master_substances)} 种物质, {len(self.master_reactions)} 条反应规则。")

# === 入口 ===
if __name__ == "__main__":
    miner = ChemKnowledgeMiner()
    miner.run()