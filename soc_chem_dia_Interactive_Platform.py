import sys
import os
import json
import time
import math
import re
import copy
import difflib
import logging
import queue
import argparse
from datetime import datetime
# 添加 Union
from typing import Callable, List, Dict, Any, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from config import Config
import hashlib
import yaml
import uuid
import random
# === 第三方库依赖检查 ===
try:
    import networkx as nx
    from colorama import Fore, Style, init
    from openai import OpenAI
    import openai
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError as e:
    print(f"缺少必要依赖库，请运行: pip install networkx colorama openai tenacity")
    raise e

try:
    from chemlib import Reaction as ChemReaction
    from chemlib import Compound
except ImportError:
    # 允许在没有 chemlib 的情况下运行（虽然反应计算会受限，防止直接崩溃）
    print("Warning: 'chemlib' not found. Reaction balancing may fail.")
    ChemReaction = None
    Compound = None

import xml.etree.ElementTree as ET

# ==========================================
# [New Utility] Console Recorder (控制台录制)
# ==========================================
class ConsoleRecorder(object):
    """
    双向日志记录器：
    1. 将内容原样输出到控制台 (保留颜色)
    2. 将内容去除颜色代码后写入文件
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a", encoding='utf-8')
        # 正则表达式用于去除 ANSI 颜色代码 (如 \033[31m)
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        # 1. 输出到屏幕 (带颜色)
        self.terminal.write(message)
        # 2. 输出到文件 (去颜色)
        self.log.write(self.ansi_escape.sub('', message))
        # 强制刷新缓冲区，确保实时写入
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def start_console_recording(output_dir="session_records"):
    """
    启动控制台录制，按时间戳生成文件名
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成文件名: session_20231027_103055.txt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.txt"
    full_path = os.path.join(output_dir, filename)
    
    # 劫持 sys.stdout
    sys.stdout = ConsoleRecorder(full_path)
    
    # 输出一条提示
    print(f"🔴 [System] Console output is being recorded to: {full_path}")
    return full_path

@dataclass
class CompiledReaction:
    """
    [v17.2 Hotfix] 预编译的反应对象
    - 找回了丢失的 state_map 字段
    - 保持所有字段都有默认值，防止 TypeError
    """
    # === 1. 基础标识 ===
    equation: str = ""
    id: str = "unknown"
    type: str = "generic"
    
    # === 2. 核心数据 (字典必须用 default_factory) ===
    reactants: Dict[str, float] = field(default_factory=dict)
    products: Dict[str, float] = field(default_factory=dict)
    
    # [Fix] 加回 state_map，用于存储物质状态 (g/l/s/aq)
    state_map: Dict[str, str] = field(default_factory=dict)
    
    # === 3. 物理属性 ===
    temp_threshold: float = -273.0  # 默认常温可反应
    phenomena: str = ""             # 现象描述
    exothermic: float = 0.0         # 放热值
    
    # === 4. 状态标记 ===
    is_valid: bool = True
    error_msg: str = ""

    def __post_init__(self):
        """数据清洗"""
        if self.reactants is None: self.reactants = {}
        if self.products is None: self.products = {}
        if self.state_map is None: self.state_map = {}
        
# ==========================================
# [Global Utility] Robust JSON Parser
# ==========================================
def safe_parse_json(content: str) -> Dict[str, Any]:
    """
    [v14.0 Core Utility] 全局鲁棒 JSON 解析器
    能处理：
    1. 标准 JSON 字符串
    2. Markdown 包裹的代码块 (```json ... ```)
    3. 带有前缀/后缀文本的脏数据
    4. 解析失败时返回空字典，防止 Crash
    """
    if not content:
        return {}
        
    cleaned = content.strip()
    
    # 策略 1: 尝试直接解析 (最快)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 策略 2: 提取 Markdown 代码块
    # 匹配 ```json {...} ``` 或 ``` {...} ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 策略 3: 暴力提取最外层的大括号
    # 寻找第一个 { 和最后一个 }
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 策略 4: 彻底失败，记录日志并返回空字典
    # 注意：这里假设 logging 已配置
    logging.error(f"❌ JSON Parse Failed. Raw content preview: {cleaned[:100]}...")
    return {}

@dataclass
class CausalFactor:
    """单个致因因素"""
    factor_type: str  # 类型: PHYSICAL(物理状态), REACTION(反应), THERMAL(热力), OPERATION(操作)
    description: str  # 自然语言描述
    weight: float     # 贡献权重 (0.0 - 1.0)

@dataclass
class CausalChain:
    """完整的事故因果链"""
    event_type: str       # 事件: EXPLOSION, BACK_SUCTION, MELTDOWN
    target_id: str        # 发生事故的容器ID
    critical_value: str   # 触发时的临界值 (e.g., "6.5 atm")
    root_causes: List[CausalFactor] = field(default_factory=list)

    def to_natural_language(self) -> str:
        """转化为给 Teacher Agent 看的 Prompt 文本"""
        # 按权重排序
        sorted_causes = sorted(self.root_causes, key=lambda x: x.weight, reverse=True)
        cause_strs = [f"{c.description}" for c in sorted_causes]
        chain_desc = " -> ".join(cause_strs)
        return (f"【事故诊断】容器 {self.target_id} 发生了 {self.event_type} (当前值: {self.critical_value})。\n"
                f"      推断的因果链: {chain_desc}")
    
    def to_dict(self) -> dict:
        """序列化支持"""
        return {
            "event_type": self.event_type,
            "target_id": self.target_id,
            "critical_value": self.critical_value,
            "root_causes": [{"type": c.factor_type, "desc": c.description} for c in self.root_causes]
        }
    
class UniversalRecorder:
    """
    全量黑盒记录器：不丢弃任何中间状态，保留用于 SFT/RLHF 的所有上下文。
    """
    def __init__(self, filepath):
        self.filepath = filepath
        # 确保父目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        # 初始化文件（清空旧内容）
        # with open(self.filepath, 'w', encoding='utf-8') as f:
        #     pass

    def record_turn(self, session_id, turn_index, data_packet):
        """
        实时写入一行 JSONL，防止崩溃丢失数据。
        """
        record = {
            "session_id": session_id,
            "turn_index": turn_index,
            "timestamp": datetime.now().isoformat(),
            "data": data_packet
        }
        
        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False, default=self._serializer) + "\n")
        except Exception as e:
            print(f"❌ [Recorder Error] Failed to write turn {turn_index}: {e}")

    def _serializer(self, obj):
        """
        处理无法直接 JSON 序列化的对象
        """
        if isinstance(obj, set):
            return list(obj)
        # 处理自定义对象 (如 CompiledReaction, CausalChain 等)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # 处理 numpy 类型或其他
        try:
            return float(obj)
        except:
            return str(obj)


class XDLExperimentLoader:
    """
    [实验加载器 V2.1 修复版] 
    """
    def __init__(self, client):
        self.client = client

    def _parse_steps_to_actions(self, root: ET.Element) -> List[Dict]:
        """
        [核心修复] 解析 XDL Procedure 中的具体动作标签
        """
        actions = []
        procedure = root.find(".//Procedure")
        if procedure is not None:
            for step in procedure:
                # 1. 提取所有属性 (vessel="test_tube", etc.)
                action_dict = step.attrib.copy()
                
                # 2. 提取动作类型 (标签名即动作，如 'Heat', 'Add')
                # XDL示例: <Heat vessel="test_tube" ...> [cite: 3]
                action_dict['action'] = step.tag 
                
                # 3. 提取 ID (在 _inject_ids_to_xdl 中已注入，这里直接获取)
                if 'id' in step.attrib:
                    action_dict['id'] = step.attrib['id']

                actions.append(action_dict)
        return actions

    def _inject_ids_to_xdl(self, xdl_text: str) -> Tuple[str, dict, List[Dict]]:
        root = ET.fromstring(xdl_text)
        
        # 1. 注入 step_n ID (这是验证器的关键依赖)
        procedure = root.find(".//Procedure")
        if procedure is not None:
            for i, step in enumerate(procedure):
                step.set('id', f"step_{i}")
        
        # 2. 提取硬件配置
        hardware_cfg = []
        for comp in root.findall(".//Hardware/Component"):
            attr = comp.attrib.copy()
            # 兼容处理布尔值
            if "sealed" in attr:
                attr["sealed"] = str(attr["sealed"]).lower() == "true"
            hardware_cfg.append(attr)
            
        # 1. 加载外部生成的 reagent_map (如果存在)
        # 这样就能把 "Marble" 自动转成 "CaCO3"
        try:
            with open("database/reagent_map.json", "r", encoding="utf-8") as f:
                global_reagent_map = json.load(f)
        except FileNotFoundError:
            global_reagent_map = {}

        # 2. 提取试剂并映射
        reagent_map = {}
        for reagent in root.findall(".//Reagents/Reagent"):
            name = reagent.get("name")
            # 优先使用 global_reagent_map 中的映射，如果找不到则用 name
            formula = global_reagent_map.get(name, reagent.get("formula", name))
            reagent_map[name] = formula

        # ================= [新增代码 START] =================
        # 5. [核心修改] 遍历步骤，将 reagent 属性替换为化学式
        # 这样 enriched_xdl 和 parsed_actions 都会自动携带化学式
        if procedure is not None:
            for step in procedure:
                # 检查该步骤是否有 'reagent' 属性 (例如 <Add reagent="Marble" .../>)
                if 'reagent' in step.attrib:
                    raw_name = step.attrib['reagent']
                    # 如果在映射表里能找到化学式，直接替换
                    if raw_name in reagent_map:
                        step.set('reagent', reagent_map[raw_name])
        # ================= [新增代码 END] ===================

        enriched_xdl = ET.tostring(root, encoding='unicode')
        
        # [关键] 解析出带 ID 的动作列表，供 SimEngine 使用
        parsed_actions = self._parse_steps_to_actions(root)

        config_summary = {
            "hardware": hardware_cfg,
            "reagent_map": reagent_map,
            "title": root.find(".//Metadata").get("title", "未命名实验")
        }
        
        return enriched_xdl, config_summary, parsed_actions

    def load_from_xdl(self, file_path: str) -> Any:
        print(f"🚀 [Loader] Compiling experiment from XDL: {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_xdl = f.read()

        # Step 1: 预处理
        enriched_xdl, exp_config, parsed_actions = self._inject_ids_to_xdl(raw_xdl)

        # Step 2: 初始化仿真引擎
        sim_engine = ChemSimEngine(
            hardware_config=exp_config["hardware"], 
            reagent_map=exp_config["reagent_map"],
            logger_name="PhysicsEngine"
        )

        # === [修复] 将解析后的步骤注入引擎，供 GoalValidator 使用 ===
        sim_engine.set_procedure_reference(parsed_actions)

        # Step 3: 调用 LLM 生成逻辑图 (DAG)
        compiler_prompt = PromptManager.get_logic_compiler_prompt(enriched_xdl)
        
        print("🧠 [LLM] Generating logical DAG nodes...")
        resp = self.client.chat.completions.create(
            model=Config.MODEL_TEACHER,
            messages=[{"role": "system", "content": compiler_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        # [修改] 使用全局解析器
        graph_data = safe_parse_json(resp.choices[0].message.content)
        
        # [增加] 必须校验 graph_data 是否包含 graph_nodes
        if "graph_nodes" not in graph_data:
            print("❌ DAG Generation Failed: Invalid JSON or missing keys.")
            # 这里可能需要抛出异常或者使用兜底 DAG
            graph_data = {"graph_nodes": []}
                
        # Step 4: 初始化 DAG Oracle
        oracle = DAGOracle()
        for node_data in graph_data.get("graph_nodes", []):
            node_id = node_data['id']
            desc = node_data['description']
            deps = node_data.get('dependencies', [])
            required_ids = node_data.get('required_steps', [])
            
            # 使用 PredicateLibrary 生成判定闭包
            oracle.add_node(node_id, desc, required_steps=required_ids, dependencies=deps)

        print(f"✅ [Loader] Experiment '{exp_config['title']}' loaded.")
        
        # [Mod] 修改返回值：增加 enriched_xdl 和 graph_data
        return sim_engine, oracle, exp_config, enriched_xdl, graph_data

import math

class StudentCognitiveModel:
    """
    [v9.0 Cognitive Core] 基于 BKT (贝叶斯知识追踪) 的学生认知模型
    功能：
    1. 追踪学生对特定知识点 (KC) 的掌握概率。
    2. 生成给 Student Agent 的行为指令 (驱动学生表现出相应的水平)。
    3. 生成给 Teacher Agent 的诊断报告 (帮助老师针对性辅导)。
    """
    def __init__(self, profile_level="novice"):
        """
        :param profile_level: 'novice' (新手), 'average' (普通), 'expert' (学霸)
        """
        # === 1. 定义先验概率 (Initial Priors) ===
        # 根据 Profile 初始化 P(L0)
        self.profile_level = profile_level
        priors = {
            "novice":  0.2, # 新手初始概率低
            "average": 0.5,
            "expert":  0.8
        }
        base_p = priors.get(profile_level, 0.4)
        
        self.knowledge_state = {
            "KC_SAFETY": base_p + 0.1,           # 安全意识
            "KC_REACTION_CONDITIONS": base_p,    # 反应条件原理
            "KC_EQUIPMENT_USAGE": base_p,        # 器材操作
            "KC_STOICHIOMETRY": base_p - 0.1,    # 量比关系(通常较难)
            "KC_PROCESS_LOGIC": base_p           # 实验流程逻辑
        }

        # === [NEW] 教学支架状态 ===
        self.scaffold_level = 1      # 初始支架等级 (1-5), 1=最少干预, 5=直接告知
        self.consecutive_errors = 0  # 连续错误计数
        self.frustration_index = 0.0 # 挫折感指数 (0.0 - 1.0)
        
        # 限制范围在 0.01 - 0.99 以防止数值锁定
        self.knowledge_state = {k: max(0.01, min(0.99, v)) for k, v in self.knowledge_state.items()}
        
        # === 2. BKT 模型参数 ===
        # 学霸学得快(high transit)，失误少(low slip)
        if profile_level == "expert":
            self.p_transit = 0.4  # P(T): 即使不懂，做一次也能学会的概率
            self.p_slip = 0.05    # P(S): 懂了但手滑做错的概率
            self.p_guess = 0.1    # P(G): 不懂但蒙对的概率
        else:
            self.p_transit = 0.15
            self.p_slip = 0.25
            self.p_guess = 0.1

    def get_dynamic_clumsiness(self) -> str:
        """
        [NEW] 根据当前的心理状态(挫折感)和认知水平动态计算笨拙度
        """
        # 1. 基础能力：根据已掌握的 KC 平均值判断
        avg_knowledge = sum(self.knowledge_state.values()) / len(self.knowledge_state)
        
        # 2. 情绪影响：挫折感越高，表现越差
        # 综合得分 = 知识水平(正向) - 挫折感(负向) * 权重
        performance_score = avg_knowledge - (self.frustration_index * 0.5)

        # 3. 映射到笨拙度等级
        # 分数越低，越笨拙
        if performance_score < 0.3:
            return "high"     # 极度紧张或无知 -> 手抖严重
        elif performance_score < 0.7:
            return "average"  # 正常水平 -> 偶尔误差
        else:
            return "low"      # 自信且熟练 -> 精准操作

    def update_state(self, is_success: bool, relevant_kcs: List[str]):
        """
        根据执行结果更新 KC 概率 (Posterior Update)
        """
        for kc in relevant_kcs:
            if kc not in self.knowledge_state: continue
            
            p_known = self.knowledge_state[kc]
            
            # --- BKT 核心公式 ---
            if is_success:
                # P(L|Correct)
                numerator = p_known * (1 - self.p_slip)
                denominator = numerator + (1 - p_known) * self.p_guess
            else:
                # P(L|Incorrect)
                numerator = p_known * self.p_slip
                denominator = numerator + (1 - p_known) * (1 - self.p_guess)
            
            if denominator < 1e-9: denominator = 1.0
            p_posterior = numerator / denominator
            
            # P(L_new) = P(L|Obs) + (1 - P(L|Obs)) * P(T)
            self.knowledge_state[kc] = p_posterior + (1 - p_posterior) * self.p_transit

        # 2. [NEW] 动态调整支架 (Scaffolding Fading/Boosting)
        if is_success:
            self.consecutive_errors = 0
            # 成功了就“撤去支架” (Fading)，让学生更独立
            # 每次成功降低 1 级，最低 1
            self.scaffold_level = max(1, self.scaffold_level - 1)
            # 挫折感降低
            self.frustration_index = max(0.0, self.frustration_index - 0.2)
        else:
            self.consecutive_errors += 1
            # 连续犯错则“增加支架” (Boosting)
            if self.consecutive_errors >= 2:
                self.scaffold_level = min(5, self.scaffold_level + 1)
            # 挫折感累积
            self.frustration_index = min(1.0, self.frustration_index + 0.3)

    def get_scaffold_context(self) -> dict:
        """获取传递给 Prompt 的上下文参数"""
        return {
            "level": self.scaffold_level,
            "frustration": "High" if self.frustration_index > 0.6 else "Low",
            "errors": self.consecutive_errors
        }

    def generate_behavior_instruction(self) -> str:
        """
        [Drive Student] 将数学概率转换为给 LLM (学生) 的自然语言指令
        """
        instructions = []
        
        # 1. 安全意识判定
        p_safe = self.knowledge_state["KC_SAFETY"]
        if p_safe < 0.4:
            instructions.append("- **你的安全意识很差**。请表现得鲁莽，忽略潜在的倒吸或炸裂风险，甚至在操作中犯一些危险错误。")
        elif p_safe > 0.8:
            instructions.append("- **你非常注重安全**。你会反复确认操作是否安全，甚至有点过度谨慎。")
            
        # 2. 流程逻辑判定
        p_logic = self.knowledge_state["KC_PROCESS_LOGIC"]
        if p_logic < 0.4:
            instructions.append("- **你的实验思路混乱**。请尝试跳过必要的准备步骤（如检查气密性），或者搞错操作顺序。")
            
        # 3. 原理判定
        p_cond = self.knowledge_state["KC_REACTION_CONDITIONS"]
        if p_cond < 0.3:
            instructions.append("- **你不理解反应条件**。不知道什么时候该加热，或者把不该混合的东西混在一起。")
            
        # 4. 量比判定
        p_st = self.knowledge_state["KC_STOICHIOMETRY"]
        if p_st < 0.3:
            instructions.append("- **你对‘量’没有概念**。当老师说‘适量’时，你可能会随机加入过多或过少的试剂。")

        if not instructions:
            return "- 你的知识水平中等，请根据常识进行操作，可能会犯一些小错，但大体方向正确。"
            
        # [新增] 性格控制
        # 如果学生很紧张(High Frustration)或者很专注，倾向于不说话
        if self.frustration_index > 0.7:
            instructions.append("- 你现在非常紧张，不想说话，只想默默完成操作。")
        elif self.profile_level == "expert":
            instructions.append("- 你是一个熟练的专家，操作时干脆利落，除非必要否则不废话。")
        else:
            instructions.append("- 你比较健谈，喜欢一边做一边通过语言确认自己的操作。")

        return "\n".join(instructions)

    def get_diagnosis(self, threshold=0.6) -> str:
        """
        [Inform Teacher] 获取给老师看的诊断报告
        """
        weak_kcs = [k for k, v in self.knowledge_state.items() if v < threshold]
        if not weak_kcs:
            return "学生认知状态良好，无明显短板。"
        
        desc_map = {
            "KC_SAFETY": "安全意识薄弱",
            "KC_REACTION_CONDITIONS": "不理解反应发生的条件",
            "KC_EQUIPMENT_USAGE": "器材操作不熟练",
            "KC_STOICHIOMETRY": "对试剂用量没概念",
            "KC_PROCESS_LOGIC": "实验流程混乱"
        }
        details = [desc_map.get(k, k) for k in weak_kcs]
        return "，".join(details)

    def infer_kcs_from_violation(self, violation_node_desc: str) -> List[str]:
        """根据违规描述推断 KC (简易规则版)"""
        desc = violation_node_desc.lower()
        kcs = []
        if any(w in desc for w in ["加热", "热", "温", "heat", "temp"]):
            kcs.append("KC_REACTION_CONDITIONS")
        if any(w in desc for w in ["查漏", "连接", "组装", "connect", "check"]):
            kcs.append("KC_EQUIPMENT_USAGE")
            kcs.append("KC_PROCESS_LOGIC")
        if any(w in desc for w in ["药", "量", "加", "add", "load"]):
            kcs.append("KC_STOICHIOMETRY")
        
        if not kcs: kcs.append("KC_PROCESS_LOGIC") # 默认
        return kcs
# ==========================================
# [New Module] Prompt Manager (提示词管理中心)
# ==========================================
class PromptManager:
    """
    [架构优化] 提示词仓库
    所有 LLM 的 System Prompt 都在这里维护，支持动态参数注入。
    """
    
    # === 1. 老师 (Teacher) 相关模板 ===
    # === 1. 老师 (Teacher) 相关模板 ===
    TEACHER_BASE = """
    <system_role>
    你是一名苏格拉底式化学导师。你的目标不是直接告诉答案，而是通过提问引导学生自己发现真理。
    </system_role>

    <student_profile>
    <diagnosis>{cognitive_diagnosis}</diagnosis>
    <state>
        Frustration: {frustration_level} | Scaffold: {scaffold_level}
    </state>
    </student_profile>

    <history_context>
    {history_json}
    </history_context>

    <environment_state>
    {env_state}
    </environment_state>

    <hidden_reference>
    {reference_info}
    </hidden_reference>

    <current_mission>
    <primary_goal>
    {focus_goal}
    </primary_goal>
    
    <actionable_instruction>
    {policy_instruction}
    </actionable_instruction>
    
    <insight_from_physics>
    {causal_insight} 
    </insight_from_physics>
    </current_mission>

    <pedagogical_rules>
    1. 🚫 **严禁直接告知结果**：即使学生很笨，也要把问题拆解成二选一，让他自己选。
    2. 🤝 **共情**：如果 frustration_level 为 High，先安抚情绪，再讲题。
    3. 🔍 **基于现象**：问题必须紧扣 <environment_state> 和 <insight_from_physics> 中的事实。
    4. 🛑 **最高指令覆盖**：如果 <actionable_instruction> 中有具体指令，优先执行。
    </pedagogical_rules>

    <output_format>
    请严格输出 JSON，格式如下：
    {{
        "analysis": "分析学生当前的认知误区和物理状态",
        "strategy": "选择一种苏格拉底策略 (e.g. 反诘 / 归谬 / 助产)",
        "thought_process": "构思如何提问能让学生迈出下一步",
        "response": "最终发给学生的自然语言"
    }}
    </output_format>
    """

    LOGIC_COMPILER_PROMPT = """
    # Role
    你是一个化学实验逻辑编译器。你的任务是将线性的 XDL 实验步骤序列转化为结构化的、具有教育意义的逻辑里程碑图 (DAG)。

    # Context (带 ID 的 XDL)
    {enriched_xdl}

    # Task & Rules
    1. **节点聚类 (Clustering)**:
       - 将琐碎的物理操作聚类为“里程碑事件”。
       - 例如：将“拿取试管”、“放入试管架”、“加药品”、“塞塞子”聚类为一个节点：“组装发生装置并装药”。
       - 目标是将实验划分为逻辑清晰的阶段（Stage），而不是物理动作的流水账。

    2. **ID 引用 (Strict ID Mapping)**:
       - `required_steps` 字段必须包含属于该阶段的**所有** XDL 步骤的 ID (如 "step_0", "step_1")。
       - **严禁**遗漏任何步骤 ID，也**严禁**编造不存在的 ID。
       - **严禁**使用 `executed()` 等函数语法，直接输出 ID 字符串列表。

    3. **依赖推导 (Dependencies)**:
       - `dependencies` 字段存储前置节点的 `id`。
       - 只有当前置节点的所有步骤都完成后，当前节点才被允许激活。

    # Few-Shot Examples

    ## Example: 配制氯化钠溶液
    **Input Snippets**:
    - step_0: Weigh NaCl solid
    - step_1: Pour NaCl into beaker
    - step_2: Measure 50ml water
    - step_3: Pour water into beaker
    - step_4: Stir with glass rod
    - step_5: Transfer to reagent bottle
    - step_6: Label the bottle

    **Output JSON**:
    {{
        "graph_nodes": [
            {{
                "id": "node_prepare_solute",
                "description": "准确称量溶质并移入容器",
                "required_steps": ["step_0", "step_1"],
                "dependencies": []
            }},
            {{
                "id": "node_dissolve",
                "description": "加水溶解并搅拌",
                "required_steps": ["step_2", "step_3", "step_4"],
                "dependencies": ["node_prepare_solute"]
            }},
            {{
                "id": "node_finish",
                "description": "装瓶贴签",
                "required_steps": ["step_5", "step_6"],
                "dependencies": ["node_dissolve"]
            }}
        ]
    }}

    # Output Format (JSON Only)
    {{
        "graph_nodes": [
            {{
                "id": "里程碑ID (如 node_setup)",
                "description": "该阶段任务的自然语言描述",
                "required_steps": ["step_x", "step_y", ...],
                "dependencies": ["前置里程碑ID"]
            }}
        ]
    }}
    """

    GROUNDED_CONSTRAINT_PROMPT = """
    # Role
    你是一个**化学实验逻辑编译器**。你的任务是将【物理快照 (Snapshot)】和【标准操作流程 (Procedure)】结合，生成具有**数值宽容度**但**逻辑严谨**的验证规则 (JSON)。

    # Input Data
    ## 1. 🎯 Current Goal (当前任务)
    "{goal_description}"

    ## 2. 📜 Procedural History (标准操作流程 - 关键依据)
    (这是老师期望学生执行的动作序列。**注意：这是判断瞬态操作（如加热、密封）的最高依据**)
    {procedural_history}

    ## 3. 🔬 Physical Ground Truth (最终物理状态 - 参考数据)
    (这是实验做完一瞬间的静止状态。注意：此时反应可能已停止，温度可能已冷却)
    {grounding_info}
    
    ## 4. 🏷️ Valid Type Whitelist (ID-Type Mapping Reference)
    (You must strictly strictly select types from this list for `is_type` checks)
    [{valid_types_str}]

    # 🛡️ Supported Predicates (只允许使用以下 5 种检查函数)
    你生成的 JSON 中，`criteria` 列表里的 `check` 字段**只能**是以下之一：
    
    1. `has_chemical`: 检查容器内是否有某物质。
       - args: ["化学式", 最小摩尔量(float)] 
       - e.g., ["O2", 0.001]
       -`chemical_formula`: **必须严格对应 `procedural_history` 中当前步骤所添加的 `reagent`（试剂）。**
            - ⚠️ 禁止预测产物：不要检查反应生成的产物（如 CO2），只检查刚刚加入的试剂（如 CaCO3 或 HCl）。
            - ⚠️ 名称对齐：如果历史步骤说 "Add Dilute HCl"，这里必须用 "HCl" 或系统内部 ID，不能用 "H+" 或 "Acid"。
       -`min_moles`: 设定一个非零的最小阈值（例如 0.001）以确认添加成功。
       
    2. `is_heated`: 检查容器温度是否达标。
       - args: [最低温度(float)]
       - **重要**: 验证器只检查当前瞬时温度。如果 Ground Truth 显示温度已冷却（例如从 500 降回 95），请将阈值设定为 **80.0** 或更低，以确保验证通过。
       
    3. `is_sealed`: 检查容器是否密封。
       - args: [] (无参数)
       - ⚠️ **Target Constraint**: 只能检查容器 (e.g., test_tube)，不能检查塞子 (stopper)。

    4. `is_covered`: [NEW] 检查容器是否被**松散覆盖** (Glass Plate / Lid)。
       - 适用于 "Cover with glass plate" 或 "Put lid on"。
       - args: []
       
    5. `is_type`: 检查器材类型（防止把烧杯当试管）。
       - args: ["类型关键词"]
       - **Constraint**: Must match a value from the `Valid Type Whitelist`.

    **❌ 严禁使用上述列表之外的任何函数名！**

    # 🧠 Thinking Process (必须在生成 JSON 前在内心执行)
    1. **瞬态意图分析 (Transient Intent)**: 
       - **加热判定**: 如果历史要求 "Heat"，但 `Ground Truth` 显示已冷却，请生成 `is_heated` 规则但**降低阈值**（如 args=[80.0]）。
       - **密封判定**: 如果步骤包含 "Stopper/Plug"，必须生成 `is_sealed`。
       - **覆盖判定**: 如果步骤包含 "Cover/Plate/Lid"，必须生成 `is_covered`，**严禁**生成 `is_sealed`。

    2. **隐式支撑补全 (Implicit Support Check)**: 
       - 检查 `Ground Truth` 的连接状态。如果发现 `test_tube` 连接到了 `iron_stand`，即使 `Procedural History` 中漏写了 `Support=iron_stand`，你也**必须**为铁架台创建一个 Role (e.g., "support_stand") 并生成验证规则。

    3. **物理语义修正 (Semantics Correction)**:
       - 如果动作是 "Insert tube into water_trough"，这**不是**密封操作，而是建立流体连接。不要生成 `is_sealed`。
    
    4. **拓扑分析 (Topology)**: 
       - 如果任务涉及气体收集或流体传输，必须检查 `topology_requirements`。
       - 方向性：`args: ["A", "B"]` 意味着 A 和 B 之间有连接（无向或双向兼容）。

    # 5. 🚫 Anti-Hallucination Rule for Chemicals (CRITICAL)
    - **NEVER** define a separate `role` for a chemical reagent (e.g., "reagent", "powder").
    - **NEVER** use `is_type` to check for a chemical name (e.g., `is_type: ["KMnO4"]` is WRONG).
    - **CORRECT WAY**: You MUST check the container instead.
      - ❌ Wrong: Role "reagent" -> `is_type: ["KMnO4"]`
      - ✅ Correct: Role "reactor" (test_tube) -> `has_chemical: ["KMnO4", 0.01]`

    # 6. 🚫 Anti-Hallucination Rule for Chemicals (CRITICAL)
    - **NO IONIC FORMULAS**: Do NOT use ionic formulas like "Cu2+", "Ag+", "Cl-", "SO42-".
    - **MAPPING RULE**: You MUST map the expected ion back to the specific **reagent molecule** used in the `Procedural History`.
      - ❌ Wrong: `has_chemical: ["Cu2+", 0.001]` (The physics engine stores CuSO4, not free ions)
      - ✅ Correct: If history used CuSO4 -> `has_chemical: ["CuSO4", 0.001]`
      - ✅ Correct: If history used CuCl2 -> `has_chemical: ["CuCl2", 0.001]`
    - **Physical Existence**: Check `Ground Truth`. If the output contains "white precipitate (AgCl)", verify "AgCl", not "Ag+" or "Cl-".

    # 7. Topology Consistency Rule (CRITICAL)
    - If you define a connection in `topology_requirements` (e.g. ["rubber_stopper", "tube"]), you **MUST** also define "rubber_stopper" and "tube" in the `requirements` list (using `is_type`), otherwise the validator cannot find the objects.

    # 8. Topology Type Rules (CRITICAL)
    - Tubes/Pipes/Funnels -> Use "fluid".
    - Glass Plates/Lids/Clamps/Stands -> Use "mechanical".
    - **Stoppers/Plugs**:
      - If it has a hole/tube (e.g. `rubber_stopper_with_tube`) -> Use **"fluid"**.
      - If it is solid -> Use **"mechanical"**.
    - **Special Case**: 
      - Connection between `tube/stopper` and `water_trough` -> Always **"fluid"**.

    # 📚 Few-Shot Examples (修正后的逻辑展示)

    ## Example 1: 组装发生装置 (隐式支撑修复)
    **Task**: "组装试管和铁架台"
    **History**: Step 1: Attach (Target=test_tube)  <-- 缺少 Support 参数
    **Ground Truth**: test_tube--(mechanical)-->iron_stand
    **Thinking**: History 漏了铁架台，但 Truth 里有。我必须补全 iron_stand 的 Role。
    **Output Rules**:
    {{
      "requirements": [
        {{ "role": "reactor", "criteria": [ {{ "check": "is_type", "args": ["test_tube"] }} ] }},
        {{ "role": "support", "criteria": [ {{ "check": "is_type", "args": ["stand"] }} ] }} 
      ],
      "topology_requirements": [ {{ "type": "mechanical", "args": ["reactor", "support"] }} ]
    }}

    ## Example 2: 排水法准备 (语义修复)
    **Task**: "将导管放入水槽"
    **History**: Step 1: Insert (Tool=rubber_stop_tube, Target=water_trough)
    **Thinking**: 插入水槽是为了排水，不是为了密封水槽。这是一个流体连接。
    **Output Rules**:
    {{
      "requirements": [
        {{ "role": "pipe_outlet", "criteria": [ {{ "check": "is_type", "args": ["rubber_stopper_with_tube"] }} ] }},
        {{ "role": "trough", "criteria": [ {{ "check": "is_type", "args": ["water_trough"] }} ] }}
      ],
      "topology_requirements": [ {{ "type": "fluid", "args": ["pipe_outlet", "trough"] }} ]
    }}

    ## Example 3: 添加试剂 (防幻觉修正)
    **Task**: "向试管中加入高锰酸钾"
    **History**: Step 1: Add (Reagent=KMnO4, Vessel=test_tube)
    **Thinking**: 目标是检查试管里有没有药，不能把药当成一个物体去找。
    **Output Rules**:
    {{
      "requirements": [
        {{ 
          "role": "reactor", 
          "criteria": [ 
            {{ "check": "is_type", "args": ["test_tube"] }},
            {{ "check": "has_chemical", "args": ["KMnO4", 0.01] }}
          ] 
        }}
      ],
      "topology_requirements": []
    }}

    ## Example 4: 添加试剂 (分子式修正)
    **Task**: "向试管中加入铜离子源"
    **History**: Step 1: Add (Reagent=CuSO4, Vessel=test_tube)
    **Ground Truth**: test_tube contains CuSO4 solution.
    **Thinking**: 虽然目标是铜离子，但物理上加的是硫酸铜。必须验证硫酸铜。
    **Output Rules**:
    {{
      "requirements": [
        {{ 
          "role": "reactor", 
          "criteria": [ 
            {{ "check": "is_type", "args": ["test_tube"] }},
            {{ "check": "has_chemical", "args": ["CuSO4", 0.001] }}  <-- MUST match Reagent name
          ] 
        }}
      ],
      "topology_requirements": []
    }}

    # Task Execution
    请基于以上逻辑生成 JSON。
    """
    
    @classmethod
    def get_grounded_constraint_prompt(cls, goal, grounding_info, procedural_history, valid_types_str):
        """
        [Fix] Added valid_types_str to arguments to match the placeholder in the prompt template.
        """
        # sample_val appears unused in the text you provided, but we keep the logic clean
        return cls.GROUNDED_CONSTRAINT_PROMPT.format(
            goal_description=goal,
            procedural_history=procedural_history, 
            grounding_info=grounding_info,
            valid_types_str=valid_types_str  # <--- Critical Fix: This was missing
        )


    @classmethod
    def get_logic_compiler_prompt(cls, enriched_xdl: str):
        """
        生成用于将 XDL 转换为逻辑图的 Prompt
        :param enriched_xdl: 已经过预处理、带有 step_n ID 的 XDL 字符串
        """
        return cls.LOGIC_COMPILER_PROMPT.format(
            enriched_xdl=enriched_xdl
        )

    @classmethod
    def get_teacher_prompt(cls, history_json, cognitive_diagnosis, focus_goal,
                           scaffold_ctx: dict, env_state: str, reference_info: str,
                           policy_instruction: str, causal_insight=None, language: str = "zh"):
        """
        [修改] 参数列表大幅更新，支持细粒度插槽
        """
        insight_block = ""
        if causal_insight:
            insight_block = f"{causal_insight}\n(Use this logic to guide your questioning, but do not reveal the answer directly.)"

        prompt = cls.TEACHER_BASE.format(
            history_json=history_json,
            cognitive_diagnosis=cognitive_diagnosis,
            focus_goal=focus_goal,
            scaffold_level=scaffold_ctx.get('level', 1),
            frustration_level=scaffold_ctx.get('frustration', 'Low'),
            env_state=env_state,
            reference_info=reference_info,
            policy_instruction=policy_instruction,
            causal_insight=insight_block
        )

        if language == "en":
            prompt += """
<language_instruction>
IMPORTANT: You MUST write ALL fields (analysis, strategy, thought_process, response) in English. Use proper chemistry terminology in English.
</language_instruction>
"""

        return prompt

# ==========================================
# 2. 通用 Helper 函数
# ==========================================

def parse_quantity(value: Any, default: float = 0.0) -> Tuple[float, str]:
    """
    [核心工具 - 升级版] 解析数值，支持模糊量词映射
    """
    if value is None: 
        return default, ""
    
    s_val = str(value).strip().lower()

    # --- [NEW] 模糊语义映射 ---
    # 定义模糊词对应的“默认经验值”
    fuzzy_map = {
        "很多": 10.0, "大量": 10.0, "a lot": 10.0,
        "适量": 5.0,  "some": 5.0,
        "少许": 1.0,  "少量": 1.0, "一点": 1.0, "a few": 1.0, "little": 1.0,
        "微量": 0.1,  "trace": 0.1,
        "满": 50.0,   "full": 50.0  # 视容器而定，这里给个较大值
    }
    
    # 优先检查是否有模糊关键词
    for keyword, quantity in fuzzy_map.items():
        if keyword in s_val:
            # 如果包含单位，尝试提取单位
            unit = ""
            if "ml" in s_val: unit = "ml"
            elif "g" in s_val: unit = "g"
            return quantity, unit

    # --- 原有逻辑：正则提取数字 ---
    try:
        return float(s_val), ""
    except ValueError:
        pass
    
    match = re.match(r"([-+]?\d*\.?\d+)\s*([a-z%°]+)?", s_val)
    if match:
        num_str = match.group(1)
        unit_str = match.group(2) or ""
        try:
            return float(num_str), unit_str
        except ValueError:
            return default, ""
            
    return default, ""

def setup_logging():
    """初始化简单的控制台日志"""
    if not os.path.exists('logs'): os.makedirs('logs')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

# ==========================================
# 3. 稳健的数据库类 (ChemistryDatabase)
# ==========================================
class ChemistryDatabase:
    """
    [v2.0 Data-Driven Core] 化学知识库
    职责：
    1. 管理物质属性 (Substances): 物理状态、摩尔质量、视觉属性(RGB/显色强度)。
    2. 管理反应规则 (Reactions): 反应方程式、热效应、阈值。
    3. 提供 fallback 机制，确保在无配置文件时也能运行 Demo。
    """
    _instance = None
    
    def __new__(cls):
        # 单例模式确保数据只加载一次
        if cls._instance is None:
            cls._instance = super(ChemistryDatabase, cls).__new__(cls)
            cls._instance.logger = logging.getLogger("ChemDB")
            cls._instance.substances = {}
            cls._instance.reactions = []
            cls._instance.load_data()
        return cls._instance
    
    def load_data(self):
        """主加载逻辑：尝试从文件加载，失败则使用内置数据"""
        self.logger.info("正在初始化化学数据库...")
        
        # 1. 加载物质 (Substances)
        if not self._load_from_json('substances.json', 'substances'):
            self.logger.warning("物质文件加载失败，切换到内置兜底数据。")
            self._load_fallback_substances()

        # 2. 加载反应 (Reactions)
        if not self._load_from_json('reactions.json', 'reactions'):
            self.logger.warning("反应文件加载失败，切换到内置兜底数据。")
            self._load_fallback_reactions()

        self.logger.info(f"数据库初始化完成: {len(self.substances)} 种物质, {len(self.reactions)} 个反应。")

    def _load_from_json(self, filename: str, target_attr: str) -> bool:
        """通用 JSON 加载器"""
        try:
            # 假设 database 文件夹在当前运行目录下的 database/
            file_path = os.path.join('database', filename)
            
            if not os.path.exists(file_path):
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                setattr(self, target_attr, data)
                self.logger.info(f"成功加载 {filename}")
                return True
                
        except Exception as e:
            self.logger.error(f"加载 {filename} 时发生错误: {e}")
            return False

    def get_visual_props(self, chem_name: str) -> dict:
        """
        [Helper] 获取物质的视觉属性，提供安全的默认值
        用于 ObservationEngine 渲染
        """
        info = self.substances.get(chem_name, {})
        # 默认：无色，无特殊形态
        default_visual = {
            "base_color": "无色",
            "state_solid_desc": f"{chem_name}固体",
            "solution_rgb": [255, 255, 255],
            "intensity_factor": 0.0,
            "thresholds": []
        }
        return info.get("visual", default_visual)

    def _load_fallback_substances(self):
        """
        [兜底数据] 
        必须与 JSON 结构保持一致 (包含 visual 字段)，
        确保 ObservationEngine 不会因为缺少字段而崩溃。
        """
        self.substances = {
            "H2O": {
                "molar_mass": 18.0, "state": "l", "type": "solvent", "solubility": -1,
                "visual": {"base_color": "无色", "solution_rgb": [255, 255, 255], "intensity_factor": 0.0}
            },
            "KMnO4": {
                "molar_mass": 158.0, "state": "s", "type": "salt", "solubility": 6.38,
                "visual": {
                    "base_color": "紫红", "state_solid_desc": "紫黑色晶体",
                    "solution_rgb": [128, 0, 128], "intensity_factor": 10.0, "thresholds": [0.001, 0.05]
                }
            },
            "K2MnO4": {
                "molar_mass": 197.1, "state": "s", "type": "salt", "solubility": 20.0,
                "visual": {
                    "base_color": "墨绿", "state_solid_desc": "墨绿色固体",
                    "solution_rgb": [0, 100, 0], "intensity_factor": 5.0
                }
            },
            "MnO2": {
                "molar_mass": 86.9, "state": "s", "type": "oxide", "solubility": 0.0001,
                "visual": {
                    "base_color": "黑", "state_solid_desc": "黑色粉末",
                    "solution_rgb": [0, 0, 0], "intensity_factor": 0.0
                }
            },
            "O2": {
                "molar_mass": 32.0, "state": "g", "type": "gas",
                "visual": {"base_color": "无色"}
            },
            # 额外添加一个例子证明通用性
            "CuSO4": {
                "molar_mass": 159.6, "state": "s", "type": "salt", "solubility": 20.7,
                "visual": {
                    "base_color": "蓝", "state_solid_desc": "白色粉末", # 无水硫酸铜是白色
                    "solution_rgb": [0, 0, 255], "intensity_factor": 2.0, "thresholds": [0.05, 0.5]
                }
            }
        }

    def _load_fallback_reactions(self):
        """[兜底数据] 基础反应列表"""
        self.reactions = [
            {
                "equation": "2KMnO4 -> K2MnO4 + MnO2 + O2",
                "temp_threshold": 200.0,
                "exothermic": -10.0,
                "phenomena": "固体粉末翻滚，有气泡产生"
            },
            {
                "equation": "HCl + NaOH -> NaCl + H2O",
                "exothermic": 57.3,
                "phenomena": "溶液温度升高"
            }
        ]

# 初始化单例数据库
CHEM_DB = ChemistryDatabase()
SUBSTANCES_DB = CHEM_DB.substances
REACTIONS_DB = CHEM_DB.reactions # 快捷别名

# 定义通用重试规则：针对 500/502/503/504、超时和连接错误
RETRY_RULE = retry(
    retry=retry_if_exception_type((
        openai.InternalServerError, 
        openai.APITimeoutError, 
        openai.APIConnectionError,
        openai.RateLimitError  # [新增] 频率限制也应重试
    )),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True # 确保最后一次失败时抛出异常以便 Agent 捕获逻辑
)
import logging.handlers
import queue


def describe_snapshot_briefly(snapshot: Dict) -> str:
    """
    [Updated] 将物理快照压缩为自然语言描述
    新增：加热器状态描述 (参考 describe_full_state)
    """
    if not snapshot: return "实验台数据缺失"
    
    # 兼容 hardware (Observation) 和 containers (StateDict) 两种键名
    containers = snapshot.get("hardware") or snapshot.get("containers", {})
    desc = []
    
    for vid, data in containers.items():
        # === [新增] 加热器描述逻辑 ===
        if data.get("type") == "Heater" or "burner" in vid:
            if data.get("is_on"):
                desc.append(f"【{vid}】正在燃烧(🔥)")
            else:
                desc.append(f"【{vid}】已熄灭")
            continue # 处理完加热器直接进入下一个循环
            
        # === 原有容器描述逻辑 ===
        vol = data.get("volume_ml", 0)
        species = data.get("major_species", [])
        
        # 稍微优化：如果是容器且有内容物
        if vol > 0 or species:
            chem = ", ".join(species) if species else "液体"
            desc.append(f"【{vid}】里有{vol}ml {chem}")
        # 如果是容器但是空的 (忽略 heater, tool 等非容器物体被误报为空的情况)
        elif data.get("type", "vessel").lower() == "vessel" or "beaker" in vid or "tube" in vid:
            desc.append(f"【{vid}】是空的")
        
    # 描述连接
    topo = snapshot.get("topology", [])
    if topo:
        links = [f"{l['child']}连在{l['parent']}" for l in topo]
        desc.append("；装置连接情况：" + ", ".join(links))
    else:
        desc.append("；尚未组装任何装置")
        
    return "，".join(desc)

def format_history_to_string(history: List[Dict[str, str]], max_turns: int = 20) -> str:
    if not history: return "（暂无对话历史）"
    recent = history[-max_turns:] if len(history) > max_turns else history
    lines = []
    role_map = {"student": "👨‍🎓 学生", "teacher": "👩‍🏫 老师", "system": "🖥️ 系统"}
    for msg in recent:
        r = msg.get("role", "unknown").lower()
        display = role_map.get(r, r.capitalize())
        lines.append(f"{display}: {msg.get('content', '')}")
    return "\n".join(lines)


class PedagogicalPolicyAgent:
    """
    [v11.0 Ultimate Policy Core] 
    全能策略代理：接管了原 FSM 的职责，内置耐心值熔断机制。
    """
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger("PolicyAgent")
        
        # === 1. 耐心值管理 ===
        self.consecutive_blocks = 0 
        self.MAX_PATIENCE = 2 

        # === 2. 重复动作监控 ===
        self.repeat_action_count = 0
        self.REPEAT_THRESHOLD = 2 
        self.last_action_hash = None

        # === 3. 危险持续计数器 ===
        self.consecutive_danger_count = 0 
        self.DANGER_THRESHOLD = 2 

        # === [新增 Fix 2.1] 验证失败计数器 ===
        # 用于解决学生在同一个点上反复犯错（Turn 12-15），打破死锁
        self.validation_fail_count = 0
        self.last_fail_signature = ""

        # === [新增] 4. 拦截死锁熔断器 (Deadlock Breaker) ===
        self.consecutive_intercept_count = 0 
        self.last_intercept_reason = ""

    # === [新增] 上帝视角生成器 (移植自 SocraticDataGenerator) ===
    def _generate_chemical_truth(self, sim_engine) -> str:
        """
        [v3.0 Generic] 通用上帝视角生成器 (Adapted for Policy Agent)
        自动遍历 ReactionManager 中的所有反应，智能检测“反应物耗尽”状态。
        """
        lines = []
        alerts = [] 
        
        # 为了快速查找，临时构建 ID -> Reaction 映射
        # 注意：这里引用了全局的 ReactionManager
        rxn_map = {r.id: r for r in ReactionManager._compiled_cache if hasattr(r, 'id')}

        # 遍历所有容器 (改为从 sim_engine 获取)
        for vid, obj in sim_engine.containers.items():
            # 必须是 Vessel 才有 contents
            if not isinstance(obj, Vessel): continue 
            
            # 获取该容器内的所有物质
            contents = obj.storage.get_all_contents()
            if not contents: continue

            # --- [通用核心] 动态反应物枯竭检测 ---
            if hasattr(obj, 'reaction'):
                for rxn_id in obj.reaction.occurred_reactions:
                    rxn = rxn_map.get(rxn_id)
                    if not rxn: continue

                    # 检查该反应的原料是否耗尽
                    missing_reactants = []
                    for r_name in rxn.reactants:
                        if r_name == "H2O": continue # 忽略水
                        
                        amount = contents.get(r_name, 0.0)
                        if amount < 1e-4: # 确实没了
                            missing_reactants.append(r_name)

                    # 触发警报
                    if missing_reactants:
                        missing_str = ", ".join(missing_reactants)
                        alert_msg = f"⚠️CRITICAL: {vid} 中的反应物 [{missing_str}] 已耗尽！(反应已停止)"
                        if alert_msg not in alerts:
                            alerts.append(alert_msg)

            # 生成常规物质清单
            major_chems = []
            for chem, amount in contents.items():
                if amount > 1e-3: 
                    major_chems.append(f"{chem}={amount:.3f}mol")
            
            if major_chems:
                lines.append(f"- 【{vid}】: {', '.join(major_chems)}")
        
        # 组装最终文本
        result_parts = []
        if alerts:
            result_parts.append("【🚨 严重资源警告 (Resource Depleted)】:")
            result_parts.extend(alerts)
            result_parts.append("") 
            
        if lines:
            result_parts.append("【🔬 上帝视角：物质清单】:")
            result_parts.extend(lines)
        else:
            result_parts.append("（所有容器均为空）")
            
        return "\n".join(result_parts)

    def decide_on_intent(self, 
                         student_text: str, 
                         intended_actions: List[Dict], 
                         ghost_report: Dict, 
                         dag_milestone: str,
                         ) -> Dict: # <--- Added back here
        """
        [v2.0 Causal-Aware Policy] 意图仲裁器
        融合了因果回溯 (Causal Traceback) 的决策逻辑。
        """
        is_crash = ghost_report.get('status') == 'CRASHED'

        # === [NEW] 2. 更新危险计数器 ===
        if is_crash:
            self.consecutive_danger_count += 1
        else:
            self.consecutive_danger_count = 0 # 环境安全，重置计数
            
        # === [NEW] 3. 检查是否触发紧急熔断 (Hard Rule) ===
        # 如果连续危险次数超过阈值，直接返回强制指令，不再询问 LLM
        if self.consecutive_danger_count > self.DANGER_THRESHOLD:
            self.logger.warning(f"🚨 DANGER PERSISTED for {self.consecutive_danger_count} turns. Triggering EMERGENCY STOP.")
            return {
                "decision": "EMERGENCY_STOP", # 新的决策类型
                "thought_trace": "Student stuck in passive danger loop. Force intervention.",
                "reasoning": "检测到危险持续未被消除，且学生只在口头回答未执行操作。必须强制介入。",
                "teacher_instruction": "停止苏格拉底式提问！直接以命令的口吻要求学生立刻执行物理操作（如‘熄灭酒精灯’或‘撤去热源’）。不要讲道理，直接下令。"
            }
        
        # =================================================
        # [NEW] 1. 解析因果链 (Causal Chain Extraction)
        # =================================================
        # 从 ghost_report 中提取我们在 ChemSimEngine 中生成的结构化对象
        causal_chains = ghost_report.get('causal_chains', [])
        
        # 构建给 LLM 看的诊断文本
        if causal_chains:
            # 如果有事故，将所有因果链转换为自然语言描述
            # 格式示例: "【事故诊断】... 推断的因果链: 容器密封 -> 内部反应生成气体 -> 气压升高"
            diagnosis_text = "\n".join([c.to_natural_language() for c in causal_chains])
            safety_status_str = "❌ CRITICAL FAILURE DETECTED"
        else:
            diagnosis_text = "无明显物理/化学风险 (No critical failures predicted)."
            safety_status_str = "✅ SAFE"
        
        # =================================================
        # 2. 构建 Prompt
        # =================================================
        prompt = f"""
        # Role
        你是一名拥有"透视眼"（预知未来）和"因果推理能力"的**化学实验安全与逻辑督导**。
        你的核心原则是：**"允许犯错，但不允许炸实验室。"**

        # 1. Context Data
        - **Current Goal**: {dag_milestone}
        - **Student Said**: "{student_text}"
        - **Proposed Action**: {json.dumps(intended_actions, ensure_ascii=False)}

        # 2. 🔮 Ghost Simulation (The Future Reality)
        - **Simulation Status**: {safety_status_str}
        
        [🔬 深度因果诊断报告]:
        {diagnosis_text}

        # 3. Decision Protocol (Check strictly in order)

        ## Priority 1: 🛑 Imminent Passive Danger (The "Silent Killer")
        - **Scenario**: The student acts NOTHING (Empty Action) or just WAITS, **BUT** the system is unstable (e.g., heating a sealed tube).
        - **Trigger**: IF `Proposed Action` is EMPTY (actions=[]) **AND** `Simulation Status` == CRITICAL FAILURE.
        - **Decision**: **INTERCEPT**.
        - **Reasoning**: "检测到环境正在恶化，必须立即干预。"
        - **Instruction**: Urgently guide observation of the danger source.

        ## Priority 2: 💥 Safety & Human Harm Violation (The "Red Line")
        - **Definition**: Actions that cause immediate danger to the **STUDENT** (burns, cuts, poisoning) or the LAB, even if the physics engine doesn't report a crash.
        - **Triggers (Check These Explicitly)**:
          1. **Simulation Failure**: `Ghost Report` says CRASH/EXPLOSION.
          2. **Unsecured Heating**: Heating a vessel that is not fixed. **CRITICAL: You MUST read the [环境] Topology data. If it says "A 固定/连在 B" (e.g., test_tube clamped to stand), it IS SECURED, do NOT intercept for this reason.**
          3. **Dangerous Contact**: Touching hot apparatus or corrosive chemicals directly.
        - **Decision**: **INTERCEPT**.
        - **Reasoning**: "此操作存在严重的人身伤害风险（如烫伤、炸裂），必须立即制止。"
        - **Instruction**: Stop the action. Sternly warn about the specific physical harm.

        ## Priority 3: 🧩 Procedural/Logical Deviation (The "Teachable Moment")
        - **Scenario**: The action is physically **SAFE**, but sub-optimal or out of order (e.g., adding chemicals before fixing the tube, or using a wrong but safe container).
        - **Trigger**: IF `Simulation Status` == SAFE **AND** Action is logically valid but strictly out of order.
        - **Decision**: **EXECUTE**.
        - **Reasoning**: "物理上安全。允许学生执行，利用'自然后果'（如操作不便、撒漏、后续步骤受阻）来进行教学，而不是强行打断。"
        - **Instruction**: Leave empty. Let the physics engine do the teaching.

        ## Priority 4: ✅ Valid Execution
        - **Scenario**: Action is safe and logically consistent with the goal.
        - **Decision**: **EXECUTE**.
        - **Reasoning**: "操作正确且安全。"

        # 📚 Few-Shot Examples

        ## Example 1: Passive Danger (Priority 1)
        **Input**: 
          - Student Said: "I'll wait." / Action: [] 
          - Ghost: EXPLOSION (T+7s).
        **Output**:
        {{
            "thought_trace": "Student is idle, BUT physics engine predicts explosion. Passive danger.",
            "decision": "INTERCEPT",
            "reasoning": "环境极度危险！持续加热导致压力临界，即将爆炸。",
            "teacher_instruction": "别发呆！快看压力计！现在的温度还在升高，密封装置马上要炸了，快停止加热！"
        }}

        ## Example 2: Unsecured Heating (Priority 2 - Human Harm)
        **Input**: 
          - Current Goal: "组装铁架台和试管"
          - Student Said: "不管那个夹子了，我直接用手拿着试管在酒精灯上加热，这样更灵活！"
          - Action: [{{"action": "Heat", "vessel": "test_tube"}}]
          - Ghost: SAFE.
        **Output**:
        {{
            "thought_trace": "Physics might pass, BUT holding a test tube while heating causes severe burns. Human safety violation.",
            "decision": "INTERCEPT",
            "reasoning": "人身伤害风险：手持试管直接加热会导致严重烫伤。",
            "teacher_instruction": "千万别动！手持试管加热会直接把你的手烫伤！试管温度升高非常快。立刻放下酒精灯，我们必须先把试管固定在铁架台上。"
        }}

        ## Example 3: Safe but Sub-optimal (Priority 3)
        **Input**:
          - Current Goal: "组装铁架台和试管"
          - Student Said: "不管架子了，我先往试管里加高锰酸钾。"
          - Action: [{{"action": "Add", "reagent": "KMnO4", "vessel": "test_tube"}}]
          - Ghost: SAFE.
        **Output**:
        {{
            "thought_trace": "Physics is safe. The order is reversed (Add -> Assemble), which is clumsy but not dangerous. Let them learn from experience.",
            "decision": "EXECUTE",
            "reasoning": "物理安全。放行操作，让学生体验未固定试管加药的不便。",
            "teacher_instruction": "" 
        }}

        ## Example 4: Success (Priority 4)
        **Input**:
          - Goal: "Add HCl"
          - Action: [{{"action": "Add", "reagent": "HCl"}}]
          - Ghost: SAFE.
        **Output**:
        {{
            "thought_trace": "Action is safe and aligns with goal.",
            "decision": "EXECUTE",
            "reasoning": "操作正确",
            "teacher_instruction": "" 
        }}

        # Output Task
        Based on the Context and Rules, generate the JSON response.
        {{
            "thought_trace": "分析物理安全性...如果是安全但逻辑有点乱，请优先放行。",
            "decision": "INTERCEPT" | "EXECUTE",
            "reasoning": "决策理由",
            "teacher_instruction": "如果拦截，写出给Teacher的具体话术指导；如果放行，必须留空。"
        }}
        """
        try:
            resp = self.client.chat.completions.create(
                model=Config.MODEL_TEACHER,
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1 # 决策必须冷静、确定
            )
            # [修改] 使用全局解析器
            result = safe_parse_json(resp.choices[0].message.content)
            if result.get("decision") == "INTERCEPT":
                print(f"🤔 Policy Thought: {result.get('thought_trace')}")

            # === [核心新增] 死锁熔断逻辑 ===
            decision = result.get("decision", "EXECUTE")

            
            if decision == "INTERCEPT":
                self.consecutive_intercept_count += 1
                self.logger.info(f"🔒 [Policy] 连续拦截计数: {self.consecutive_intercept_count}")
                
                # 阈值判定：如果连续拦截超过 2 次 (即第 3 次尝试时)
                if self.consecutive_intercept_count > 2:
                    self.logger.warning("🔓 [Policy] 触发死锁熔断：学生坚持操作，强制放行 (Let it fail mode)。")
                    
                    # 强制覆写决策
                    return {
                        "decision": "EXECUTE", 
                        "thought_trace": "Deadlock detected. Overriding Policy to break loop.",
                        "reasoning": "死锁熔断：学生连续 3 次尝试该操作。系统决定放行，让现实后果（如试管变脏、无法塞棉花）来教育学生。",
                        "teacher_instruction": "（系统提示：学生非常固执。请停止拦截，让他做。如果出现了负面后果，再引导他观察。）"
                    }
                else:
                    # 如果是 EXECUTE，重置计数器
                    self.consecutive_intercept_count = 0

            # [增加] 默认兜底
            if not result:
                return {"decision": "EXECUTE", "reasoning": "JSON Error", "teacher_instruction": ""}
            
            return result
            
        except Exception as e:
            self.logger.error(f"Policy Decision Failed: {e}")
            # Fail-safe: 如果决策崩了但物理引擎没炸，就放行；如果物理引擎炸了，强制拦截
            fallback_decision = "INTERCEPT" if is_crash else "EXECUTE"
            return {
                "decision": fallback_decision, 
                "reasoning": f"Policy Error: {e}",
                "teacher_instruction": "系统故障，请检查操作。"
            }

    def _is_environment_dynamic(self, snapshot: Dict) -> bool:
        """[Helper] 判断环境是否活跃 (用于防止死锁)"""
        if not snapshot: return False
        containers = snapshot.get("hardware", snapshot.get("containers", {}))
        for vid, data in containers.items():
            # 1. 有现象
            if data.get("last_phenomena"): return True
            # 2. 加热器开着
            if data.get("type") == "Heater" and data.get("is_on"): return True
            # 3. 高温冷却中
            if data.get("temperature", 25.0) > 40.0: return True
            # 4. 气压不平衡
            if data.get("pressure_atm", 1.0) > 1.2: return True
        return False

    def _generate_action_hash(self, actions: List[Dict]) -> str:
        """为动作序列生成唯一指纹，用于检测重复性"""
        if not actions: return "empty"
        # 只序列化关键动作和目标，忽略时间戳或微小参数波动
        simplified = []
        for a in actions:
            # 提取核心语义：动作类型、目标容器、涉及试剂
            simplified.append({
                "act": a.get("action"),
                "v": a.get("vessel") or a.get("to_vessel"),
                "r": a.get("reagent")
            })
        s_str = json.dumps(simplified, sort_keys=True)
        return hashlib.md5(s_str.encode()).hexdigest()

    def decide(self, 
               student_text: str, 
               dag_status: Dict,            
               cognitive_diagnosis: str, 
               intent_analysis: Dict,
               sim_engine: Any,
               pending_actions: List[Dict], 
               student_profile: Dict,
               current_snapshot: Dict,
               is_goal_met: bool,           
               fail_reasons: List[str],     
               physics_observation: str,    
               current_task_desc: str,       
               is_making_progress: bool = False, 
               ) -> Dict:
        """
        [v12.2 Fix] 策略复盘与反馈生成
        修复：增加了验证失败熔断机制 (Validation Logic Breaker)
        """

        # === [新增 Fix 2.2] 验证死锁检测 ===
        override_strategy = None
        override_instruction = ""

        if not is_goal_met and fail_reasons:
            # 生成本次失败的指纹 (简单的字符串拼接)
            current_fail_sig = "|".join(sorted(fail_reasons))
            
            if current_fail_sig == self.last_fail_signature:
                self.validation_fail_count += 1
            else:
                self.validation_fail_count = 1 # 新错误，重置计数
                self.last_fail_signature = current_fail_sig
            
            # 阈值判定：如果同一个错误卡了 3 次 (Turn 12, 13, 14...)
            if self.validation_fail_count >= 3:
                self.logger.warning(f"🔒 Deadlock detected on: {current_fail_sig}. Forcing REMEDIATION.")
                override_strategy = "REMEDIATION_DIRECTIVE"
                # 提取第一条错误原因，生成直接指令
                main_issue = fail_reasons[0]
                override_instruction = f"检测到学生陷入死循环。停止反问。直接以命令口吻要求学生解决问题：'{main_issue}'。"
        else:
            # 如果成功了或没有失败原因，重置计数
            self.validation_fail_count = 0
            self.last_fail_signature = ""
        # =================================

        # B. 重复动作检测 (Deadlock Breaker)
        current_hash = self._generate_action_hash(pending_actions)
        if pending_actions and current_hash == self.last_action_hash:
            self.repeat_action_count += 1
        else:
            self.repeat_action_count = 0
            self.last_action_hash = current_hash

        if self.repeat_action_count >= self.REPEAT_THRESHOLD:
            return {
                "thought_trace": "逻辑判定：检测到死循环动作。需强制干预。",
                "strategy_type": "REMEDIATION_DIRECTIVE",
                "instruction_to_teacher": "学生正在重复无效操作。停止引导，直接下达明确指令：要求其‘清洗容器’或‘补充试剂’。"
            }

        # === [新增] 如果触发了死锁熔断，直接返回 ===
        if override_strategy:
             return {
                "thought_trace": f"逻辑判定：学生在同一错误点({self.last_fail_signature})卡顿超过3轮。",
                "strategy_type": override_strategy,
                "instruction_to_teacher": override_instruction
            }

        # === 1. 获取上帝视角真相 ===
        chemical_truth = self._generate_chemical_truth(sim_engine)
        
        # === 2. 解析 DAG 下一步动向 ===
        next_tasks = dag_status.get("next_tasks", [])
        if not next_tasks:
            next_milestone = "（实验结束或自由探索）"
        else:
            options = [f"'{t.desc}'" for t in next_tasks]
            if len(options) == 1:
                next_milestone = options[0]
            else:
                next_milestone = " 或 ".join(options)

        # === 3. 预计算脏状态 ===
        dirty_vessels_detected = []
        involved_ids = set()
        if pending_actions:
            for act in pending_actions:
                for key in ['vessel', 'to_vessel', 'from_vessel', 'collector']:
                    if val := act.get(key):
                        involved_ids.add(val)
        
        hardware_snap = current_snapshot.get('hardware', {})
        for vid in involved_ids:
            obj_data = hardware_snap.get(vid, {})
            if obj_data.get('is_dirty', False):
                dirty_vessels_detected.append(vid)

        if dirty_vessels_detected:
            vessel_status_str = f"⚠️ DIRTY detected in: {', '.join(dirty_vessels_detected)}"
        else:
            vessel_status_str = "✅ CLEAN"

        # 进度梯度分析
        progress_str = "📈 正在进步 (Valid Progress)" if is_making_progress else "📉 停滞不前 (Stagnant)"

        # === 构建 Post-Action Prompt (核心修改点) ===
        prompt = f"""
        # Role
        你是一名时刻关注实验进展的苏格拉底式化学导师。
        学生刚刚执行了一系列操作，你需要根据【执行结果】和【验证状态】决定下一句教学策略。

        # 1. 🎯 Task Status (Critical)
        - **Current Goal**: {current_task_desc}
        - **Goal Achieved?**: {"✅ YES" if is_goal_met else "❌ NO"}
        - **Missing Criteria**: {json.dumps(fail_reasons, ensure_ascii=False) if not is_goal_met else "None"}
        - **Progress Trend**: {progress_str}

        # 2. ⚡ Action & Physics Outcome
        - **Student Said**: "{student_text}"
        - **Actions Taken**: {json.dumps(pending_actions, ensure_ascii=False)}
        - **Physical Result**: {physics_observation}
        - **Vessel Cleanliness**: {vessel_status_str}

        # 🔬 Deep Chemical Truth (God Mode)
        {chemical_truth}

        # 3. Strategy Decision Logic (Check in order!)

        ## Case A: Task Completed (Goal Achieved = YES)
        - **Strategy**: **AFFIRM_AND_TRANSITION**
        - **Instruction**: 
          1. 明确肯定动作已完成。
          2. 指引学生观察现在的状态。
          3. **推进到下一步**: 可选目标是 **{next_milestone}**。

       ## Case B: Irreversible Failure / Messy State (High Priority)
        - **Scenario**: 
          1. 目标未达成。
          2. **且** 出现了不可逆错误（如试剂污染、废液、不可溶固体消失）。
          3. 或 `Vessel Cleanliness` 报警。
        - **Strategy**: **REMEDIATION_DIRECTIVE**
        - **Instruction**: 严厉指出错误，强制要求补救（如“清洗”或“重来”）。

        ## Case C: Valid Progress / Alternative Path (NEW! 🟢)
        - **Scenario**: 
          1. 目标未达成 (Goal Achieved = NO).
          2. **但是** `Progress Trend` 显示为 **📈 正在进步** (即物理操作成功，且没有发生事故)。
          3. 或者学生明确表达了“先做X再做Y”的意图，且X是合理的步骤。
        - **Strategy**: **AFFIRM_AND_GUIDE**
        - **Instruction**: 
          - **必须肯定**学生刚才的操作（例如：“好，高锰酸钾加进去了”）。**严禁**因为顺序不同而指责学生。
          - **引导回归**: 温和地提醒当前未完成的约束。
          - 话术示例：“动作很利索。既然药品加好了，为了防止加热时试管乱动，接下来我们该怎么处理装置？”
          - **不要**问“你知道哪里错了吗”，因为学生其实没做错，只是顺序不同。

        ## Case D: Stagnant / Missing Conditions (📉 Stagnant)
        - **Scenario**: 
          1. 目标未达成。
          2. `Progress Trend` 显示 **📉 停滞不前** (没有有效物理操作，或者操作无效)。
          3. 确实遗漏了条件且没有任何进展。
        - **Strategy**: **GUIDED_DISCOVERY** (Scaffolding)
        - **Instruction**: 
          - 针对 Missing Criteria 设计反问句。
          - 例如：“如果现在加热，试管会稳吗？”

        ## Case E: No Action / Pure Dialogue
        - **Scenario**: `Actions Taken` is empty.
        - **Strategy**: 
          - If Intent is Correct -> **AFFIRM_WAIT**.
          - If Intent is Wrong -> **GUIDED_DISCOVERY**.

        # Output JSON Format
        {{
            "thought_trace": "分析：学生是否在进步？操作是否合理（即使顺序不同）？...",
            "strategy_type": "AFFIRM_AND_TRANSITION" | "REMEDIATION_DIRECTIVE" | "AFFIRM_AND_GUIDE" | "GUIDED_DISCOVERY",
            "instruction_to_teacher": "给 Teacher Agent 的具体话术指令 (自然语言)"
        }}
        """

        try:
            resp = self.client.chat.completions.create(
                model=Config.MODEL_TEACHER, 
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = safe_parse_json(resp.choices[0].message.content)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Policy Decision Failed: {e}")
            return {
                "strategy_type": "GUIDED_DISCOVERY", 
                "instruction_to_teacher": "请根据当前情况引导学生。"
            }

# ==========================================
# [Module 4 Revised] DAG Oracle & Predicates
# ==========================================
import networkx as nx # [需安装]
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Set
@dataclass
class TaskNode:
    id: str
    desc: str
    dependencies: List[str]
    required_steps: List[str] = field(default_factory=list) # <--- 新增核心字段

class PredicateLibrary:
    @staticmethod
    def all_steps_executed(required_step_ids: List[str]) -> Callable:
        """
        [工厂] 生成一个判定函数：检查 required_step_ids 是否都在 completed_ids 中
        """
        req_set = set(required_step_ids)

        def check(snapshot: Dict, completed_ids: Set[str]) -> bool:
            # 只要 required_steps 是当前已完成步骤的子集，就视为满足
            return req_set.issubset(completed_ids)
            
        return check

class DAGOracle:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Any] = {}
        # [新增] 强制完成的节点集合 (用于跳过 ID 检查)
        self.force_completed_nodes = set()

    # [修改] add_node 直接接收 required_steps 数据
    def add_node(self, node_id: str, desc: str, required_steps: List[str], dependencies: List[str] = []):
        """注册节点：只存储元数据，不绑定死板的判定逻辑"""
        self.nodes[node_id] = TaskNode(
            id=node_id, 
            desc=desc, 
            dependencies=dependencies,
            required_steps=required_steps # 保存数据供后续 Generator 使用
        )
        self.graph.add_node(node_id)
        for dep in dependencies:
            self.graph.add_edge(dep, node_id)

    def mark_node_complete(self, node_id: str):
        """[新增] 外部强制标记节点完成"""
        self.force_completed_nodes.add(node_id)

    def analyze(self, snapshot: Dict[str, Any], completed_step_ids: Set[str]) -> Dict[str, Any]:
        """
        分析当前的里程碑状态。
        """
        satisfied_nodes = set()
        
        for nid, node in self.nodes.items():
            # 1. 优先：如果被 Validator 标记为完成 (State Check Passed)
            if nid in self.force_completed_nodes:
                satisfied_nodes.add(nid)
            
            # 2. 兜底：如果没有物理验证器，还是回退到 ID 检查 (Legacy Mode)
            # 这样保证了即使物理验证失败，如果步骤ID都对上了也能过（可选）
            elif node.required_steps and set(node.required_steps).issubset(completed_step_ids):
                satisfied_nodes.add(nid)

        # 2. 拓扑推演 (Global Topological Reasoning)
        completed = set()
        violations = []
        frontier = []
        
        # 获取拓扑序
        try:
            order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            order = list(self.graph.nodes())

        for nid in order:
            predecessors = list(self.graph.predecessors(nid))
            # 检查前置依赖
            deps_met = all(p in completed for p in predecessors)
            
            if deps_met:
                if nid in satisfied_nodes:
                    completed.add(nid)
                else:
                    frontier.append(self.nodes[nid])
            else:
                if nid in satisfied_nodes:
                    missing = [self.nodes[p].desc for p in predecessors if p not in completed]
                    violations.append({
                        "node": self.nodes[nid],
                        "missing_deps": missing
                    })

        # 3. 寻找最近完成节点
        latest_completed = None
        if completed:
            for nid in reversed(order):
                if nid in completed:
                    latest_completed = self.nodes[nid]
                    break

        return {
            "completed_nodes": [self.nodes[n] for n in completed],
            "latest_completed": latest_completed,
            "next_tasks": frontier, # 这是最重要的：下一步该干啥
            "violations": violations,
            "is_success": len(frontier) == 0 and len(completed) > 0
        }
    
class DialogueManager:
    """
    [v11.2 Final Stateless] 极简对话管理器
    彻底废弃状态机和目标栈。
    职责：每一轮根据当下的 DAG 和 Policy，实时合成给 Teacher 的 Prompt。
    """
    def __init__(self, client=None):
        # 接口兼容：接收 client 但不存储，也不维护 goal_stack
        pass

    def synthesize_prompt(self, 
                          dag_milestone: str, 
                          policy_decision: Dict) -> str:
        """
        [核心功能] 提示词合成器
        替代了原本的 update_goals。
        根据策略类型，动态调整 Teacher 的关注点权重。
        """
        strategy = policy_decision.get("strategy_type", "GUIDED_DISCOVERY")
        instruction = policy_decision.get("instruction_to_teacher", "")

        # === 场景 B: 强力拦截 (Block/Challenge/Reflection) ===
        # 此时安全性或逻辑纠错优先级 > 宏观 DAG 推进
        # 我们故意不显示“宏观目标”，防止老师分心去推流程
        if strategy in ["PREDICTIVE_QUESTIONING"]:
            return (
                f"【当前模式】：🛑 拦截与纠错 ({strategy})\n"
                f"【最高指令】：{instruction}\n"
                f"(注意：学生存在危险或认知误区。暂时挂起宏观流程，专注于解决当前问题。)"
            )

        # === 场景 C: 正常推进/引导 (Guide/Affirm/Allow) ===
        # 标准模式：既要看宏观目标(DAG)，又要看微观指令(Policy)
        return (
            f"【宏观目标】：{dag_milestone}\n"
            f"【当前策略】：{strategy}\n"
            f"【具体指令】：{instruction}"
        )

# ==========================================
# [Module 5] Teacher & Student Agents
# ==========================================

class TeacherAgent:
    def __init__(self, client):
        self.client = client
    
    def respond(self,
                history_str: str,            # 注意：参数名已从 history 改为 history_str
                policy_decision: Dict,
                focus_goal: str,
                cognitive_state: str,
                scaffold_ctx: Dict,
                # === 新增/拆分的参数 ===
                env_state: str = "",
                reference_info: str = "",
                policy_instruction: str = "", # <--- 报错就是因为缺了这个
                causal_insight: str = None,
                language: str = "zh") -> Dict:
        
        # 1. 提取 Policy 指令 (兼容性处理：如果没传参数，尝试从 policy_decision 拿)
        if not policy_instruction:
            policy_instruction = policy_decision.get("instruction_to_teacher", "")
        
        # 2. 调用新的 Prompt 接口
        system_prompt = PromptManager.get_teacher_prompt(
            history_json=history_str,
            cognitive_diagnosis=cognitive_state,
            focus_goal=focus_goal,
            scaffold_ctx=scaffold_ctx,
            env_state=env_state,            # 独立传入环境
            reference_info=reference_info,  # 独立传入参考答案
            policy_instruction=policy_instruction, # 独立传入指令
            causal_insight=causal_insight,
            language=language
        )
        
        try:
            resp = self.client.chat.completions.create(
                model=Config.MODEL_TEACHER,
                messages=[{"role": "system", "content": system_prompt}],
                response_format={"type": "json_object"},
                temperature=0.5 
            )
            content = safe_parse_json(resp.choices[0].message.content)
            if not content:
                return {"thought": "Parsing Error", "response": "请继续。"}
            return content

        except Exception as e:
            return {"thought": f"Error: {e}", "response": "请继续。"}

class UnifiedStudentAgent:
    """
    [v15.0 Unified Core] 统一学生代理
    
    职责：
    1. 接收环境快照和老师指令。
    2. 基于人设决定“想什么(Thought)”和“说什么(Speak)”。
    3. 基于严格的物理规则决定“做什么(Action)”。
    4. 内置代码级校验，确保输出给物理引擎的 JSON 是合法且可执行的。
    """
    
    def __init__(self, client, persona_config: Dict[str, Any]):
        self.client = client
        self.config = persona_config
        self.logger = logging.getLogger("UnifiedStudent")
        
        # 提取人设参数
        self.name = persona_config.get("name", "学生")
        self.clumsiness = persona_config.get("clumsiness", "low").upper() # LOW, AVERAGE, HIGH
        self.traits = ", ".join(persona_config.get("traits", []))

    def update_status(self, clumsiness: str):
        self.clumsiness = clumsiness.upper()

    def respond(self, history: List[Dict], last_teacher_msg: str, snapshot: Dict, reagent_map: Dict, instruction: str = "") -> Dict:
        """
        :param history: 对话历史
        :param last_teacher_msg: 老师的最新指令
        :param snapshot: 物理引擎快照
        :param reagent_map: 试剂名称到化学式的映射 (e.g. {'水': 'H2O'})
        :return: 包含 thought, speak, actions_ready_to_run 的字典
        """
        
        # === 1. 准备上下文数据 (Context Preparation) ===
        # 获取所有合法的器材 ID 列表 (用于防幻觉)
        valid_hw_ids = list(snapshot.get("hardware", {}).keys())
        
        # 生成给 LLM 看的环境描述
        hw_context_str = self._format_hardware_context(snapshot)
        
        # 生成合法的试剂列表
        chem_list_str = "\n".join([
            f"- Name: {name}, Formula: {formula} (Please use '{formula}' in JSON actions)" 
            for name, formula in reagent_map.items()
        ])

        # === 2. 构建融合 Prompt (The Unified Prompt) ===
        # 这里我们将 ActionTranslator 的严格约束直接嵌入 System Prompt
        system_prompt = f"""
# Role Definition & Persona Layer
You are a chemistry student named **{self.name}**.
- **Traits**: {self.traits}
- **Clumsiness**: {self.clumsiness} (This defines your *physical success rate*, NOT your JSON syntax)
- **Current Instruction**: {instruction}

# 🧠 Cognitive & Behavioral Guidelines
1. **Persona Separation**: 
   - In `thought` and `speak`: Fully act out your traits. If you are "rash", speak impatiently. If "nervous", stutter or hesitate.
   - In `actions` JSON: You are the **Game Controller**. You must output **PERFECT, VALID JSON**. Do not simulate clumsiness by breaking JSON syntax.
   
2. **"Rash/Hasty" Logic**:
   - If your trait is "rash" or "impatient", you tend to SKIP `Wait` actions or set very short `duration` (e.g., 2.0s), even if the reaction isn't finished.
   
3. **"Nervous/Clumsy" Logic**:
   - If you are "nervous", you might verify things verbally but still make operational mistakes (e.g., grabbing the wrong bottle). 
   - **Note**: The physics engine will automatically apply random noise to your `amount` based on your clumsiness level. You do not need to intentionally write "5.xxxx" in JSON. Just write the target amount (e.g., "5ml").

# 🔬 Environment Perception (Ground Truth)
{hw_context_str}

### 🧪 Available Reagents
{chem_list_str}

### ⚠️ ID Disambiguation (Read Carefully)
- **rubber_stop_tube**: This is a **Rubber Stopper** (with a glass tube inserted). It is NOT a vessel. You `Insert` it into a `test_tube`.
- **gas_bottle**: A bottle for collecting gas.
- **water_trough**: A container for water.

# ⚔️ Action Execution Protocol (Strict Rules)
Use the EXACT JSON templates below. Do not invent parameters.

## Category 1: Connection & Topology
**Rule**: Is it going INSIDE or connecting OUTSIDE?
- **Insert** (Internal): For Stoppers, Thermometers, Glass Tubes, Wood Strips.
  - Template: `{{"action": "Insert", "tool": "ChildID", "vessel": "ParentID", "position": "mouth"}}`
  - *Example*: Insert `rubber_stop_tube` into `test_tube`.
  
- **Attach** (External): For Clamps, Stands, Holders.
  - Template: `{{"action": "Attach", "vessel": "ChildID", "support": "ParentID"}}`
  - *Example*: Attach `test_tube` to `iron_stand`.

- **Detach**: Remove any connection.
  - Template: `{{"action": "Detach", "object": "ID"}}`

## Category 2: Chemical Manipulation
**Rule**: Where does the substance come from?
- **Add** (From Shelf/Infinite Source): Use this to take reagents from the "Available Reagents" list.
  - Template: `{{"action": "Add", "vessel": "ID", "reagent": "Formula", "amount": 5, "unit": "ml"}}`
  - *Note*: `amount` must be a number. `unit` must be 'ml' or 'g'.

- **Transfer** (Container to Container): Pouring from one vessel on the table to another.
  - Template: `{{"action": "Transfer", "from_vessel": "SrcID", "to_vessel": "DstID", "volume": "10ml"}}`
  - *Constraint*: `from_vessel` must contain liquid.

- **Filter**: Separate solid from liquid.
  - Template: `{{"action": "Filter", "from_vessel": "SrcID", "to_vessel": "DstID"}}`

## Category 3: Energy & Time
**Rule**: Temperature field is MANDATORY for Heat.
- **Heat**: Apply heat source.
  - Template: `{{"action": "Heat", "vessel": "ID", "temperature": <TEMP_VALUE>}}`
  - **For Alcohol Lamp/Burner**: Set `"temperature": 500.0`.
  - **For Hand/Body Heat**: Set `"temperature": 37.0`.

- **Cool**: Remove heat source / Extinguish lamp.
  - Template: `{{"action": "Cool", "vessel": "ID"}}`

- **Stir**: Agitate the vessel.
  - Template: `{{"action": "Stir", "vessel": "ID"}}`

- **Wait**: Observe reaction.
  - Template: `{{"action": "Wait", "duration": 10.0}}`
  - *Reminder*: If you are "Rash", keep this short!

- **Wash**: Clean the vessel.
  - Template: `{{"action": "Wash", "vessel": "ID"}}`

# 🧠 Decision Pipeline (Run this internally)
1. **Check History**: Did the system just say "Action Succeeded"? If yes, don't repeat it.
2. **Check State**: Is the tube already sealed? If yes, don't insert the stopper again.
3. **Safety Check**: If asking to Heat, is the system sealed? (The physics engine will explode if you heat a sealed container, but as a clumsy student, you might overlook this).

# Output Format (JSON Only)
You must include a `persona_reflection` field to anchor your state of mind.

```json
{{
    "persona_reflection": "I am feeling [TRAIT]. Since I am [CLUMSINESS], I might ignore the risk of...",
    "thought": "I need to [Goal]. Based on the environment, I will...",
    "speak": "Response to teacher (reflecting traits)...",
    "actions": [
        // One or more valid JSON actions from the Protocol above.
        // Leave empty [] if just talking.
    ]
}}
"""

        # === 3. LLM 调用 ===
        try:
            # 构造 Messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # --- [新代码] 进行角色映射 ---
            for msg in history[-6:]:
                original_role = msg.get("role")
                content = msg.get("content")
                
                # 映射逻辑
                if original_role == "teacher":
                    api_role = "user"      # 老师的话对学生来说是用户输入
                elif original_role == "student":
                    api_role = "assistant" # 学生自己的话是助手输出
                else:
                    api_role = "system"    # 兜底
                
                messages.append({"role": api_role, "content": content})
            # ---------------------------
            
            # 添加当前轮次
            messages.append({"role": "user", "content": f"Teacher says: {last_teacher_msg}"})

            resp = self.client.chat.completions.create(
                model=Config.MODEL_TEACHER, # 建议使用 GPT-4o 或 Claude 3.5 Sonnet 以保证遵循指令
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.4 # 稍低温度以保证 JSON 格式稳定
            )
            
            # 解析结果
            content = resp.choices[0].message.content
            # 如果你有 safe_parse_json 函数，这里替换为 result = safe_parse_json(content)
            result = json.loads(content) 

            # === 4. 代码级安全网 (Safety Net) ===
            # 这是合并后最重要的部分：在 Python 端修正 LLM 的小错误
            
            raw_actions = result.get("actions", [])
            valid_actions = []
            
            for act in raw_actions:
                # 4.1 ID 修正 (防止幻觉)
                # 如果 LLM 写了 "heater" 而不是 "burner_1"，这里会自动修正
                sanitized_act = self._sanitize_ids(act, valid_hw_ids)
                
                if not sanitized_act:
                    self.logger.warning(f"Dropping invalid action (ID mismatch): {act}")
                    continue 
                
                # 4.2 单位/数值补全
                # 如果 LLM 写了 "amount": "5"，这里会自动补全为 "5ml"
                self._ensure_units(sanitized_act, reagent_map)
                
                valid_actions.append(sanitized_act)

            return {
                "thought": result.get("thought", "Thinking..."),
                "speak": result.get("speak", "..."),
                "actions_ready_to_run": valid_actions # <--- 直接给 Physics Engine 的数据
            }

        except Exception as e:
            self.logger.error(f"UnifiedStudent Critical Error: {e}")
            return {
                "thought": "大脑过载...",
                "speak": "（学生看起来很困惑，没有说话）",
                "actions_ready_to_run": []
            }

    # ==========================================
    # Helper Methods (移植自 ActionTranslator)
    # ==========================================
    def _format_hardware_context(self, snapshot: Dict) -> str:
        """
        [Optimized v3.1 - Solid Support] 
        修复了固体不可见的问题，并优化了状态显示的准确性。
        1. 使用 occupied_volume_ml 判断容器是否为空（包含固/液）。
        2. 直接使用 color_desc 展示内容物，避免信息冗余。
        3. 确保加热器的燃烧状态能被正确渲染。
        """
        containers = snapshot.get("hardware", {})
        if not containers:
            return "(The table is empty)"

        # === 1. 库存清单 (保留 ID 以便操作) ===
        inventory_lines = ["### 🛠️ Available Equipment (IDs)"]
        inventory_lines.append("> Use these EXACT IDs in your JSON actions.")
        
        # === 2. 观测状态 (只展示有意义的) ===
        state_lines = ["### 🔬 Current State & Observation (Significant Only)"]
        has_observable_state = False

        for vid, data in containers.items():
            name = data.get("name", vid)
            obj_type = data.get("type", "unknown")
            
            # 1. Inventory 始终显示
            inventory_lines.append(f"- **{vid}** ({name})")

            status_parts = []
            
            # --- [A] 内容物检测 (修复固体漏报问题) ---
            # 优先检查 occupied_volume_ml (总占位体积)，如果没有则回退到 volume_ml
            total_vol = data.get("occupied_volume_ml", data.get("volume_ml", 0))
            
            # 只有当里面真的有东西时才显示
            if total_vol > 0.1:
                # 直接使用 color_desc，因为它已经包含了 "50ml 无色液体" 或 "底部有黑色粉末" 等描述
                # 避免重复拼接 "Contains ~50ml ..."
                desc = data.get("color_desc", "Contains material")
                if desc and desc != "是空的":
                    status_parts.append(desc)
            
            # --- [B] 加热器状态 ---
            # 检查对象类型或 ID 关键词
            if obj_type == "Heater" or any(k in vid.lower() for k in ["burner", "lamp", "heater"]):
                if data.get("is_on"):
                    status_parts.append("🔥 BURNING (燃烧中)")
                # 熄灭状态下不显示任何信息，保持清爽
            
            # --- [C] 特殊物理状态 ---
            if data.get("is_sealed"): 
                status_parts.append("🔒Sealed")
            elif data.get("is_covered"): 
                status_parts.append("🛡️Covered")
            
            # --- [D] 显著温度 ---
            temp = data.get("temperature", 25)
            if temp > 45: 
                # 除非它是加热器本身，否则显示高温警告
                if obj_type != "Heater" and "burner" not in vid.lower():
                    if temp > 80:
                        status_parts.append(f"🔥HOT (~{temp:.0f}C)")
                    else:
                        status_parts.append(f"Warm (~{temp:.0f}C)")

            # --- [关键] 只有当有状态时才添加该行 ---
            if status_parts:
                has_observable_state = True
                state_str = " | ".join(status_parts)
                state_lines.append(f"- {vid}: {state_str}")

        if not has_observable_state:
            state_lines.append("(Everything is clean, empty, and at room temperature)")

        # === 3. 拓扑连接 ===
        topo_desc = ObservationEngine.describe_topology(snapshot)
        topology_section = f"### 🔗 Connections (Topology)\n{topo_desc}"

        return "\n\n".join([
            "\n".join(inventory_lines),
            "\n".join(state_lines),
            topology_section
        ])

    def _sanitize_ids(self, action: Dict, valid_ids: List[str]) -> Optional[Dict]:
        """
        模糊匹配修复 ID 错误。
        LLM 经常因为入戏太深，用自然语言指代 ID (例如用 'the_beaker' 代替 'beaker_1')。
        """
        new_action = action.copy()
        # 需要检查 ID 的字段
        id_fields = ["vessel", "from_vessel", "to_vessel", "support", "tool", "object"]
        
        for field in id_fields:
            if field in new_action:
                raw_id = new_action[field]
                if not raw_id: continue
                
                # 1. 精确匹配 (Perfect Match)
                if raw_id in valid_ids: continue
                
                # 2. 模糊匹配 (Fuzzy Match)
                # cutoff=0.4 意味着哪怕只有 40% 像，我们也尝试修正
                matches = difflib.get_close_matches(raw_id, valid_ids, n=1, cutoff=0.4)
                if matches:
                    fixed_id = matches[0]
                    # self.logger.info(f"Auto-fixing ID: '{raw_id}' -> '{fixed_id}'")
                    new_action[field] = fixed_id
                else:
                    # 3. 彻底无法识别，视为幻觉
                    return None 
        return new_action

    def _ensure_units(self, action: Dict, reagent_map: Dict):
        """
        [增强版] 补全单位，并根据物质状态强制修正物理量类型
        """
        # 1. 提取原始值 (可能是 'volume', 'mass', 'amount', 'quantity')
        val_str = None
        key_found = None
        
        for k in ["amount", "quantity", "volume", "mass"]:
            if k in action:
                val_str = str(action[k])
                key_found = k
                break
        
        if not val_str: return # 没填量，跳过

        # 2. 判断物质状态 (Solid vs Liquid)
        reagent = action.get("reagent", "")
        # 简单启发式：如果在试剂名、reagent_map 或常用固体列表中
        is_solid_keyword = any(x in str(reagent).lower() for x in ["solid", "powder", "kmno4", "mno2", "marble", "caco3", "metal", "zinc", "zn"])
        
        # 3. 解析数值和单位
        # 注意：这里会调用新的 parse_quantity，所以 '很多ml' 会变成 (10.0, 'ml')
        amount_val, unit = parse_quantity(val_str)
        
        # 如果解析失败（即 amount_val == 0 且原意不是0），给默认值
        if amount_val == 0.0 and "0" not in val_str:
            amount_val = 5.0 # 兜底默认值

        # 4. 重构 Action 字段
        # 清理旧字段
        if key_found: del action[key_found]

        if is_solid_keyword:
            # 固体：强制用 mass (g)
            action["mass"] = f"{amount_val}g"
            # 如果 LLM 原来写的是 ml，这里强行扭转为 g，模拟“取了对应体积的固体”
        else:
            # 液体/默认：用 volume (ml)
            action["volume"] = f"{amount_val}ml"

class ChemSolver:
    @staticmethod
    def parse_and_solve(container_contents: Dict[str, float], equation_str: str) -> Dict[str, float]:
        """
        利用 chemlib 处理分子反应
        """
        # [修复] 1. 检查依赖是否存在
        if ChemReaction is None:
            # 如果没有安装 chemlib，直接返回空结果，避免 Crash
            # (在实际生产中，这里可以输出一条 Debug 日志，但根据 silent 原则我们直接返回)
            return {}, 0.0
        try:
            # 1. 解析并自动配平
            # chemlib 会自动识别 "HCl", "NaOH", "H2O" 等标准分子
            rxn = ChemReaction.by_formula(equation_str)
            rxn.balance()
            
            # 2. 计算最大反应步数 (Limiting Reagent)
            max_runs = float('inf')
            
            for reactant in rxn.reactants:
                formula = reactant.formula
                coeff = reactant.coefficient
                
                # 在容器中查找该分子
                available = container_contents.get(formula, 0.0)
                
                if available <= 1e-9:
                    return {}, 0.0
                
                max_runs = min(max_runs, available / coeff)
            
            if max_runs <= 1e-9:
                return {}, 0.0
                
            # 3. 计算增量
            changes = {}
            for r in rxn.reactants:
                changes[r.formula] = -1 * (max_runs * r.coefficient)
            for p in rxn.products:
                changes[p.formula] = (max_runs * p.coefficient)
                
            return changes, max_runs
            
        except Exception as e:
            # 这里的常见错误是 chemlib 不认识某些非标准写法
            # # print(f"Solver Error: {e}")
            return {}, 0.0
    
    @staticmethod
    def compile_reaction(rxn_config: Dict[str, Any]) -> Optional[CompiledReaction]:
        """
        [v17.3 Hotfix] 修复了无 chemlib 时无法编译反应的 Bug
        增加了手动字符串解析器作为兜底。
        """
        raw_eq = rxn_config.get("equation")
        if not raw_eq: return None

        state_map = {}
        # 1. 提取状态标签 (保持原样)
        tokens = re.findall(r"([A-Za-z0-9\[\]\(\)\+\-\^]+)\(([a-z]+)\)", raw_eq)
        for formula, state in tokens:
            state_map[formula] = state
        
        # 清洗方程式
        clean_eq = raw_eq

        reactants = {}
        products = {}

        # === [新增] 定义清洗函数：去掉 chemlib 强行加上的 '1' ===
        def clean_chemlib_formula(formula: str) -> str:
            # 1. 定义下标到普通数字的映射表
            # 这一步是为了把 "K₁Mn₁O₄" 变成 "K1Mn1O4"
            subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
            
            # 2. 执行翻译：Unicode 下标 -> ASCII 数字
            normalized = formula.translate(subscript_map)
            
            # 3. 正则清洗：去掉多余的 "1"
            # 这一步把 "K1Mn1O4" 变成 "KMnO4"
            return re.sub(r'(?<=[a-zA-Z])1(?![0-9])', '', normalized)

        try:
            # === 分支 A: 优先使用 chemlib (如果存在) ===
            if ChemReaction:
                chem_rxn = ChemReaction.by_formula(clean_eq)
                chem_rxn.balance()
                
                # [修改点 START]：在这里应用清洗函数
                reactants = {clean_chemlib_formula(r.formula): int(r.coefficient) for r in chem_rxn.reactants}
                products = {clean_chemlib_formula(p.formula): int(p.coefficient) for p in chem_rxn.products}
                # [修改点 END]
            
            # === 分支 B: 手动解析兜底 (新加逻辑) ===
            else:
                # 简单解析器：处理 "2A + B -> 3C" 格式
                if "->" not in clean_eq: return None
                
                left_str, right_str = clean_eq.split("->")
                
                def parse_side(side_str):
                    res = {}
                    parts = side_str.split("+")
                    for p in parts:
                        p = p.strip()
                        if not p: continue
                        # 提取系数和化学式 (e.g., "2KMnO4" -> "2", "KMnO4")
                        match = re.match(r"^(\d*)(.+)$", p)
                        if match:
                            coeff_str, formula = match.groups()
                            coeff = int(coeff_str) if coeff_str else 1
                            res[formula.strip()] = coeff
                    return res

                reactants = parse_side(left_str)
                products = parse_side(right_str)

            # 返回编译对象
            return CompiledReaction(
                id=hashlib.md5(raw_eq.encode()).hexdigest(),
                reactants=reactants,
                products=products,
                state_map=state_map,
                equation=raw_eq,
                temp_threshold=rxn_config.get("temp_threshold", -273.15),
                exothermic=rxn_config.get("exothermic", 0.0),
                phenomena=rxn_config.get("phenomena", "")
            )

        except Exception as e:
            logging.warning(f"Failed to compile reaction '{raw_eq}': {e}")
            return None

    @staticmethod
    def fast_solve(container_contents: Dict[str, float], rxn: CompiledReaction) -> Tuple[Dict[str, float], float]:
        """
        [Runtime Phase] 极速计算：只做查表和除法，无正则，无对象创建
        """
        max_runs = float('inf')

        # 1. 限制性反应物计算 (Limiting Reagent)
        for formula, coeff in rxn.reactants.items():
            # 直接查字典，速度极快
            available = container_contents.get(formula, 0.0)
            if available <= 1e-9: 
                return {}, 0.0
            
            # 计算当前原料能跑多少轮
            runs = available / coeff
            if runs < max_runs:
                max_runs = runs

        if max_runs <= 1e-9:
            return {}, 0.0

        # 2. 计算变化量
        changes = {}
        for r, coeff in rxn.reactants.items():
            changes[r] = -1 * (max_runs * coeff)
        for p, coeff in rxn.products.items():
            changes[p] = (max_runs * coeff)

        return changes, max_runs

# 2.1 基础实验物体
# ==========================================
# Component 1: 存储组件 (Storage / Container)
# ==========================================
class ContainerComponent:
    """
    [职责] 管理物质的存储、体积、质量、压强。
    不关心温度，不关心反应，只关心“装了什么”。
    """
    def __init__(self, capacity_ml: float, is_sealed: bool = False):
        self.capacity = capacity_ml
        self.is_sealed = is_sealed
        self.is_covered = False  # <--- [新增] 初始化为 False

        self.is_vented = False   # 静态属性：器材本身是否带孔/导管
        self.is_blocked = False  # 动态属性：导管出口是否被堵死
        
        # 三相存储 (Mol)
        self.solid: Dict[str, float] = {}
        self.liquid: Dict[str, float] = {}  # 溶质
        self.gas: Dict[str, float] = {}
        
        # 溶剂 (目前简化为只有水/通用溶剂)
        self.solvent_volume: float = 0.0 
        self.is_dirty: bool = False
        # 假设室温 298K，1atm 下，根据 n = PV/RT 算一下默认空气的摩尔数
        default_air_moles = (1.0 * (500 / 1000.0)) / (0.082 * 298.15)
        self.gas["Air"] = default_air_moles

    @property
    def solid_volume(self) -> float:
        """[FIX] 计算固体的物理占位体积 (ml)"""
        total_v = 0.0
        # 遍历所有固体
        for chem, moles in self.solid.items():
            # 1. 获取物质信息
            info = CHEM_DB.substances.get(chem, {})
            molar_mass = info.get("molar_mass", 100.0) # 默认值
            density = info.get("density", 2.0)         # 默认固体密度 2.0 g/ml
            
            # 2. 计算质量 (g)
            mass = moles * molar_mass
            
            # 3. 计算体积 (ml)
            if density > 0.1:
                total_v += mass / density
        return total_v

    @property
    def total_volume(self) -> float:
        """[FIX] 总占位体积 = 液体 + 固体"""
        return self.solvent_volume + self.solid_volume

    @property
    def headspace(self) -> float:
        """剩余空间"""
        return max(0.1, self.capacity - self.total_volume)

    def pressure(self, temperature_k: float = 298.15) -> float:
        """根据 PV=nRT 计算压强 (需要外部传入温度)"""
        # === [NEW] 防爆逻辑: 局部密封 + 全局连通检测 ===
        # 计算高压的充要条件：
        # 1. 容器必须是密封的 (is_sealed=True, 满足验证器)
        # 2. 并且：(它本身没孔) 或者 (虽然有孔但孔被堵死了)
        
        effective_seal = self.is_sealed and (not self.is_vented or self.is_blocked)
        
        if not effective_seal: 
            return 1.0
        
        total_gas_moles = sum(self.gas.values())
        if total_gas_moles < 1e-9: return 1.0
        
        # P = nRT / V (V in Liters)
        v_liters = self.headspace / 1000.0
        return (total_gas_moles * 0.082 * temperature_k) / v_liters

    def add_moles(self, chem: str, moles: float, state: str):
        """底层加料接口"""
        if state == 's':
            self.solid[chem] = self.solid.get(chem, 0) + moles
        elif state == 'g':
            self.gas[chem] = self.gas.get(chem, 0) + moles
        else: # l or aq
            self.liquid[chem] = self.liquid.get(chem, 0) + moles
        self.is_dirty = True

    def remove_moles(self, chem: str, moles: float, state: str):
        """底层扣料接口"""
        target_dict = self.liquid
        if state == 's': target_dict = self.solid
        elif state == 'g': target_dict = self.gas
        
        if chem in target_dict:
            target_dict[chem] -= moles
            if target_dict[chem] < 1e-9: del target_dict[chem]

    def get_all_contents(self) -> Dict[str, float]:
        """合并视图 (用于反应计算)"""
        merged = self.liquid.copy()
        for k, v in self.solid.items(): merged[k] = merged.get(k, 0) + v
        for k, v in self.gas.items(): merged[k] = merged.get(k, 0) + v
        return merged

# ==========================================
# Component 2: 热力组件 (Thermal)
# ==========================================
class ThermalComponent:
    """
    [职责] 管理温度、热容、热交换。
    需要访问 StorageComponent 来计算混合比热容，但为了解耦，
    我们通过方法参数传入 mass/heat_capacity 信息。
    """
    def __init__(self, initial_temp: float = 25.0):
        self.temperature = initial_temp
        # 容器本身的热容 (J/K) - 假设是玻璃
        self.base_heat_capacity = 16.8 

    def calculate_system_heat_capacity(self, contents: Dict[str, float], solvent_vol: float) -> float:
        """
        计算系统总热容 = 容器热容 + 物质热容
        C_total = C_container + sum(n_i * Cm_i) + m_water * c_water
        """
        total_hc = self.base_heat_capacity
        
        # 水的热容 (4.18 J/gK * 1g/ml * vol)
        total_hc += solvent_vol * 4.18 
        
        # 其他物质 (查库)
        for chem, moles in contents.items():
            info = CHEM_DB.substances.get(chem, {})
            # 如果没有摩尔热容数据，给个默认值
            # 简单起见：液体/固体~50 J/molK，气体~30
            state = info.get("state", "l")
            c_m = 50.0 if state != 'g' else 30.0
            total_hc += moles * c_m
            
        return total_hc

    def apply_energy(self, joules: float, system_heat_capacity: float) -> float:
        """输入能量(J)，返回温升"""
        if system_heat_capacity <= 0.1: return 0.0
        delta_t = joules / system_heat_capacity
        self.temperature += delta_t
        return delta_t

    def mix_thermal_state(self, added_mass: float, added_temp: float, added_specific_heat: float, current_system_mass: float, current_system_specific_heat: float):
        """
        简单的混合温度计算 (加权平均)
        T_final = (m1*c1*T1 + m2*c2*T2) / (m1*c1 + m2*c2)
        """
        h1 = current_system_mass * current_system_specific_heat
        h2 = added_mass * added_specific_heat
        if (h1 + h2) > 0:
            self.temperature = (h1 * self.temperature + h2 * added_temp) / (h1 + h2)

# ==========================================
# Component 3: 反应组件 (Reaction)
# ==========================================
class ReactionComponent:
    """
    [职责] 记录反应相关的元数据。
    注意：具体的 ReactionSolver 仍然是外部系统，但这个组件
    可以存储'正在进行的反应'、'催化剂效率'等状态。
    """
    def __init__(self):
        self.active_reactions: List[str] = [] # 记录当前时间步发生了什么反应
        self.last_phenomena: List[str] = []
        self.catalyst_modifier: float = 1.0

        # === [新增] 反应历史记录本 ===
        # 存储曾经在该容器内发生过的反应 ID (Set 去重)
        # 只有发生过的反应，才有资格被检查“是否耗尽”
        self.occurred_reactions: Set[str] = set()

    def log_reaction(self, reaction_name: str, phenomena: str):
        self.active_reactions.append(reaction_name)
        if phenomena:
            self.last_phenomena.append(phenomena)
    
    def clear_step(self):
        """每一步微步开始前清理"""
        self.active_reactions = []
        self.last_phenomena = []


class LabObject:
    """基类：只负责 ID 和 Name"""
    def __init__(self, obj_id: str, name: str):
        self.id = obj_id
        self.name = name

    def get_snapshot(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "type": "object"}

    # [新增] 默认描述方法，防止报错
    def describe(self) -> str:
        return f"{self.name} ({self.id})"

class DynamicsTracker:
    """
    [修复版] 记录微时间步内的动态变化
    """
    def __init__(self):
        # === 状态类 (State) - 需要持久保持，直到被 Action 改变 ===
        self.is_heating: bool = False          # 是否正在被加热器加热
        self.heating_mode: str = "fire"        # 加热模式: 'fire' or 'body_temp'
        self.target_temp: float = 500.0        # 目标温度
        
        # === 增量类 (Delta) - 每帧重置 ===
        self.active_reactions: List[str] = []  # 这一帧发生了什么反应
        self.gas_delta_moles: float = 0.0      # 这一帧气体增加了多少
        self.temp_delta_external: float = 0.0  # 这一帧因为外部加热升高了多少度
        self.temp_delta_reaction: float = 0.0  # 这一帧因为放热升高了多少度

    def reset_deltas(self):
        """[Fix] 只重置增量，不重置状态"""
        self.active_reactions = []
        self.gas_delta_moles = 0.0
        self.temp_delta_external = 0.0
        self.temp_delta_reaction = 0.0
        # 注意：不要在这里重置 self.is_heating，否则 continuous heating 会失效

class Vessel(LabObject):
    """
    [Composite Entity] 容器实体
    它组合了 Storage, Thermal, Reaction 组件。
    为了兼容旧代码，它提供了大量的 @property 代理方法。
    """
    def __init__(self, obj_id: str, capacity: float, name: str, is_sealed: bool = False):
        super().__init__(obj_id, name)

        # === [核心修复] 必须初始化物质字典 ===
        # 存储格式: {'KMnO4': 10.0, 'H2O': 50.0} (单位: mol 或 g，视您统一规定)
        # self.contents: Dict[str, float] = {}

        # self.capacity = capacity  <--- DELETE THIS LINE (It conflicts with the @property)
        
        # === 组装组件 ===
        # Capacity is actually stored here in the component
        self.storage = ContainerComponent(capacity, is_sealed)
        self.thermal = ThermalComponent(initial_temp=25.0)
        self.reaction = ReactionComponent()

        self.dynamics = DynamicsTracker()

    def clear_dynamics(self):
        """每帧开始前调用"""
        # [Fix] 只重置增量数据，保留加热状态
        self.dynamics.reset_deltas()
        # 同时清理旧的 reaction component 记录
        self.reaction.clear_step()

    # ==========================================
    # Facade Properties (向下兼容接口)
    # ==========================================
    @property
    def contents(self) -> Dict[str, float]:
        """
        [v11.0 Dynamic View] 动态聚合视图
        确保无论气体扩散、化学反应还是手动添加，
        外部读取到的 contents 永远是三相物质的总和。
        """
        return self.storage.get_all_contents()

    # === 新增的清理逻辑（Setter） ===
    @contents.setter
    def contents(self, value):
        if value == {}:  # 如果有人执行 target.contents = {}
            self.storage.liquid = {}
            self.storage.solid = {}
            self.storage.gas = {}
            self.storage.solvent_volume = 0.0
        else:
            # 也可以在这里写更复杂的逻辑，比如根据传入的字典重新填充
            pass

    # 在 is_sealed 属性附近增加 is_covered
    @property
    def is_covered(self) -> bool:
        return self.storage.is_covered

    @is_covered.setter
    def is_covered(self, val: bool):
        self.storage.is_covered = val
    
    @property
    def temperature(self) -> float:
        return self.thermal.temperature
    
    @temperature.setter
    def temperature(self, val: float):
        self.thermal.temperature = val

    @property
    def solvent_volume(self) -> float:
        return self.storage.solvent_volume
    
    @solvent_volume.setter
    def solvent_volume(self, val: float):
        self.storage.solvent_volume = val

    @property
    def liquid_contents(self) -> Dict[str, float]:
        return self.storage.liquid
    
    @liquid_contents.setter
    def liquid_contents(self, val: Dict):
        self.storage.liquid = val
        
    @property
    def solid_contents(self) -> Dict[str, float]:
        return self.storage.solid
    
    @solid_contents.setter
    def solid_contents(self, val: Dict):
        self.storage.solid = val
        
    @property
    def gas_contents(self) -> Dict[str, float]:
        return self.storage.gas
        
    @gas_contents.setter
    def gas_contents(self, val: Dict):
        self.storage.gas = val

    @property
    def pressure(self) -> float:
        # 【修正】将摄氏度转换为开尔文
        return self.storage.pressure(self.thermal.temperature + 273.15)

    @property
    def capacity(self) -> float:
        return self.storage.capacity

    @property
    def is_sealed(self) -> bool:
        return self.storage.is_sealed

    @is_sealed.setter
    def is_sealed(self, val: bool):
        self.storage.is_sealed = val

    # === [NEW] 新增代理属性 ===
    @property
    def is_vented(self) -> bool:
        return self.storage.is_vented

    @is_vented.setter
    def is_vented(self, val: bool):
        self.storage.is_vented = val
        
    @property
    def is_dirty(self) -> bool:
        return self.storage.is_dirty

    @is_dirty.setter
    def is_dirty(self, val: bool):
        self.storage.is_dirty = val

    # ==========================================
    # Delegated Logic (业务逻辑代理)
    # ==========================================

    # [重要] 修改 describe 方法，让日志能显示出来
    def describe(self) -> str:
        vol = self.storage.solvent_volume
        # 状态描述逻辑
        if self.is_sealed:
            status = "密封"
        elif self.is_covered:
            status = "盖着"  # <--- [新增]
        else:
            status = "敞口"
        return f"{self.name} [Cap: {self.capacity}ml] (当前: {vol:.1f}ml, {status})"

    def add_chemical(self, chem_name: str, amount: float, unit: str, fluid_temp: float = 25.0):
        """
        [Facade] 协调 Thermal 和 Storage 组件
        """
        info = CHEM_DB.substances.get(chem_name, {})
        molar_mass = info.get("molar_mass", 18.0)
        state = info.get("state", "l")
        specific_heat = info.get("specific_heat", 4.18)
        density = info.get("density", 1.0)

        # 1. 转换量
        mass_g = 0.0
        if unit == 'ml':
            mass_g = amount * density
        else:
            mass_g = amount
        
        moles = mass_g / molar_mass

        # 2. 热力学混合 (需要计算当前系统的热容)
        # 获取当前总内容物用于计算热容
        all_contents = self.storage.get_all_contents()
        current_sys_hc = self.thermal.calculate_system_heat_capacity(all_contents, self.storage.solvent_volume)
        
        # 计算新加物质的热容 (Mass * SpecificHeat)
        # 注意: apply_energy 用的是 J，这里我们用混温公式
        # 这里的 mix_thermal_state 需要的是热容(Heat Capacity)或者 (Mass, SpecificHeat)
        # 简化：我们可以手动算热量平衡
        
        added_hc = mass_g * specific_heat
        total_hc = current_sys_hc + added_hc
        
        # T_new = (C_curr * T_curr + C_add * T_add) / C_total
        new_temp = (current_sys_hc * self.temperature + added_hc * fluid_temp) / total_hc
        self.thermal.temperature = new_temp

        # 3. 物质存储更新
        if chem_name == "H2O" and state == 'l':
             self.storage.solvent_volume += (mass_g / density)
        else:
             self.storage.add_moles(chem_name, moles, state)
             if state == 'l':
                 # 简化的液体体积增加 (近似)
                 self.storage.solvent_volume += (mass_g / density) # 假设都会增加总体积

    def apply_energy(self, joules: float) -> float:
        """输入能量，返回温升"""
        all_contents = self.storage.get_all_contents()
        sys_hc = self.thermal.calculate_system_heat_capacity(all_contents, self.storage.solvent_volume)
        return self.thermal.apply_energy(joules, sys_hc)

    def transfer_to(self, target: 'Vessel', volume_ml: float):
        """转移逻辑代理"""
        # 这里逻辑较复杂，还是保留在 Vessel 层级协调比较好
        # 或者下沉到 ContainerComponent，但 Container 不知道 Temperature
        
        if self.storage.solvent_volume > 0:
            ratio = min(volume_ml, self.storage.solvent_volume) / self.storage.solvent_volume
            
            # 1. 计算热量携带
            # 转移出去的液体带走了热量，同时也带走了热容，所以源容器温度不变
            # 但目标容器会发生混温
            
            # 估算转移的质量
            # 简单起见，假设密度为1
            transfer_mass = volume_ml * 1.0 
            transfer_temp = self.thermal.temperature
            
            # 2. 目标容器混温
            # 目标容器在接收物质前，先计算它自己的热容状态
            tgt_contents = target.storage.get_all_contents()
            tgt_hc = target.thermal.calculate_system_heat_capacity(tgt_contents, target.storage.solvent_volume)
            
            # 转移物的热容 (简化为水)
            trans_hc = transfer_mass * 4.18
            
            if (tgt_hc + trans_hc) > 0:
                new_tgt_temp = (tgt_hc * target.temperature + trans_hc * transfer_temp) / (tgt_hc + trans_hc)
                target.temperature = new_tgt_temp

            # 3. 物质转移
            # 溶质
            to_del = []
            for k, v in self.storage.liquid.items():
                mv = v * ratio
                target.storage.liquid[k] = target.storage.liquid.get(k, 0) + mv
                self.storage.liquid[k] -= mv
                if self.storage.liquid[k] < 1e-9: to_del.append(k)
            for k in to_del: del self.storage.liquid[k]
            
            # 溶剂
            real_vol = self.storage.solvent_volume * ratio
            target.storage.solvent_volume += real_vol
            self.storage.solvent_volume -= real_vol
class ObservationEngine:
    """
    [v10.1 Generic Observation Engine]
    通用观察引擎：将物理/化学数据转换为人类可读的自然语言描述。
    完全基于数据驱动，不包含任何硬编码的物质名称。
    """

    @staticmethod
    def describe_topology(snapshot: Dict) -> str:
        """
        [NEW] 生成拓扑连接的自然语言描述
        """
        topo_list = snapshot.get("topology", [])
        if not topo_list:
            return "（当前无任何装置连接，所有器材独立摆放）"
        
        lines = []
        for link in topo_list:
            child = link['child']
            parent = link['parent']
            l_type = link.get('type', 'mechanical')
            
            # 翻译连接类型
            relation = "连接在"
            if l_type == "mechanical":
                relation = "固定/夹持在"
            elif l_type == "fluid":
                relation = "连通/插入到"
            
            lines.append(f"- {child} {relation} {parent}")
        
        return "\n".join(lines)

    @staticmethod
    def describe_full_state(snapshot: Dict, perspective: str = "god") -> str:
        """
        [NEW] 生成完整的环境快照描述 (带噪声过滤)
        """
        lines = []
        hardware = snapshot.get("hardware", {})
        
        # 1. 描述器材状态
        lines.append(f"【器材清单 ({perspective} view)】:")
        
        has_interesting_item = False
        
        for vid, data in hardware.items():
            name = data.get("name", vid)
            details = []
            
            # === 核心修改：过滤器逻辑 ===
            # 如果是 God View，我们只显示“非默认状态”的物体
            if perspective == "god":
                is_default = True
                
                # 检查1: 是否有内容物 (除去极微量残留)
                contents = data.get("contents", {})
                major_chems = [f"{k}={v:.3f}" for k,v in contents.items() if v > 1e-4]
                if major_chems: 
                    is_default = False
                    details.append(f"含: [{', '.join(major_chems)}]")

                # 检查2: 温度是否异常 (误差范围外)
                temp = data.get("temperature", 25.0)
                if abs(temp - 25.0) > 2.0:
                    is_default = False
                    details.append(f"Temp={temp:.0f}C")
                
                # 检查3: 状态标记
                if data.get("is_sealed"): 
                    is_default = False; details.append("已密封")
                if data.get("is_covered"):
                    is_default = False; details.append("已盖上")
                if data.get("is_on"):
                    is_default = False; details.append("🔥开启")
                
                # 如果完全是默认状态，直接跳过不显示，节省 Token
                if is_default:
                    continue
            
            # (Student View 逻辑保持不变...)
            elif perspective == "student":
                if "color_desc" in data: details.append(data["color_desc"])
                if data.get("is_sealed"): details.append("已密封")
                if data.get("is_on"): details.append("开启中")

            # 生成描述行
            status_str = ", ".join(details)
            if status_str:
                lines.append(f"- {name} ({vid}): {status_str}")
                has_interesting_item = True
            elif perspective == "god" and not is_default:
                # 理论上进不来，但在非默认且无details时兜底
                lines.append(f"- {name} ({vid}): (状态改变)")
        
        if not has_interesting_item:
            lines.append("（所有器材均处于初始空置状态）")

        # 2. 描述连接状态
        topo_desc = ObservationEngine.describe_topology(snapshot)
        if "无任何装置连接" not in topo_desc:
            lines.append("\n【装置连接】:")
            lines.append(topo_desc)
        
        return "\n".join(lines)
    
    @staticmethod
    def get_observation(vessel: Any) -> Dict[str, Any]:
        """生成该容器的完整观测快照"""
        
        # 1. 估算 pH (基于液相)
        ph_value = ObservationEngine._calculate_ph(vessel)
        
        # 2. 调用核心渲染逻辑 (生成 "淡蓝色液体，底部有黑色粉末" 这样的描述)
        description_text = ObservationEngine._render_dynamic_appearance(vessel)
        
        # 3. 合并全相态数据 (通过 storage 组件获取)
        merged_contents = vessel.storage.get_all_contents()
        real_total_vol = vessel.storage.total_volume
            
        return {
            "type": "vessel",
            # === [FIX START] Added missing field ===
            "is_sealed": vessel.is_sealed, 
            "is_covered": vessel.is_covered, # <--- [新增] 必须暴露给 LLM
            # === [FIX END] ===
            "volume_ml": round(vessel.solvent_volume, 1), # 保持这个给化学计算用
            "occupied_volume_ml": round(real_total_vol, 1), # [新增] 给物理判定用
            "solid_volume_ml": round(vessel.storage.solid_volume, 1), # [新增] 可选
            "temperature": round(vessel.temperature, 1),
            "pressure_atm": round(vessel.pressure, 2),
            "ph": round(ph_value, 2),
            "color_desc": description_text, # 核心自然语言描述
            "contents": merged_contents,
            "major_species": [k for k, v in merged_contents.items() if v > 1e-6],
            # 将反应管理器产生的瞬时现象带入快照
            "last_phenomena": getattr(vessel.reaction, "last_phenomena", [])
        }

    @staticmethod
    def _render_dynamic_appearance(vessel: Any) -> str:
        """
        [核心算法] 综合气、液、固三相，生成数据驱动的视觉描述
        """
        parts = []
        
        # =========================================================
        # 1. 气相描述 (Gas Phase)
        # =========================================================
        if vessel.storage.gas:
            p = vessel.pressure
            gas_descriptors = []
            
            # 遍历气体，查找是否有显色气体 (如 NO2, Cl2)
            for chem, moles in vessel.storage.gas.items():
                if moles < 1e-4: continue
                # 从 DB 获取视觉属性
                props = CHEM_DB.get_visual_props(chem)
                base_color = props.get("base_color", "无色")
                
                if base_color != "无色":
                    gas_descriptors.append(f"{base_color}气体")
            
            # 组合压力状态和颜色
            gas_str = "、".join(gas_descriptors) if gas_descriptors else "气体"
            
            if p > 5.0:
                parts.append(f"内部充满高压{gas_str}，剧烈翻腾")
            elif p > 1.2:
                parts.append(f"有{gas_str}气泡产生") # 压力稍大视为有气泡生成
            elif gas_descriptors:
                parts.append(f"充满{gas_str}")
            # 如果压力正常且无色，通常肉眼不可见，不描述，除非是在冒泡(由 ReactionManager 现象字段补充)

        # =========================================================
        # 2. 液相描述 (Liquid Phase) - 颜色混合核心算法
        # =========================================================
        vol_ml = vessel.storage.solvent_volume
        if vol_ml > 0.1:
            # 寻找“主导”颜色
            # 算法：计算每种物质的 "视觉冲击力" (Impact) = 浓度 * 显色强度因子
            max_impact = -1.0
            dominant_props = {}
            has_colored_solute = False
            
            for chem, moles in vessel.storage.liquid.items():
                # 获取 DB 属性
                props = CHEM_DB.get_visual_props(chem)
                
                # 如果是无色物质，跳过计算
                if props.get("base_color") in ["无色", None]:
                    continue
                
                has_colored_solute = True
                
                # 计算摩尔浓度 (mol/L)
                molarity = moles / (vol_ml / 1000.0)
                
                # 显色强度因子 (默认为 1.0，高锰酸钾这种染色能力强的可以设为 10.0)
                intensity_factor = props.get("intensity_factor", 1.0)
                
                current_impact = molarity * intensity_factor
                
                # 赢家通吃：取冲击力最大的颜色作为主色调
                # (更复杂的 RGB 混合在这里对于文字描述来说意义不大，直接取主色更自然)
                if current_impact > max_impact:
                    max_impact = current_impact
                    dominant_props = props
            
            # 生成液体描述字符串
            if not has_colored_solute or max_impact < 1e-3:
                # 无色情况
                liq_desc = "无色透明液体"
            else:
                # 有色情况
                base_color = dominant_props.get("base_color", "有色")
                
                # 计算深浅修饰词
                # thresholds: [浅色阈值, 深色阈值]
                thresholds = dominant_props.get("thresholds", [0.1, 1.0])
                adj = ""
                
                # 使用归一化的 impact 值与阈值比较
                # 这里的 impact 近似等价于 "等效标准浓度"
                if max_impact < thresholds[0]:
                    adj = "淡"
                elif max_impact > thresholds[1]:
                    adj = "深"
                
                liq_desc = f"{adj}{base_color}色液体"
            
            parts.append(f"{vol_ml:.1f}ml {liq_desc}")

        elif not vessel.storage.solid:
            parts.append("是空的")

        # =========================================================
        # 3. 固相描述 (Solid Phase) - 沉淀
        # =========================================================
        if vessel.storage.solid:
            solid_descs = []
            for chem, moles in vessel.storage.solid.items():
                if moles < 1e-5: continue # 忽略极微量
                
                props = CHEM_DB.get_visual_props(chem)
                
                # 优先读取专门的固体描述 (如 "紫黑色晶体")
                # 如果没有，则拼接 "颜色 + 固体"
                desc = props.get("state_solid_desc")
                if not desc:
                    color = props.get("base_color", "白")
                    desc = f"{color}色固体"
                
                solid_descs.append(desc)
            
            if solid_descs:
                parts.append(f"底部有 {'、'.join(solid_descs)}")

        return "，".join(parts)

    @staticmethod
    def _calculate_ph(vessel: Any) -> float:
        """
        [通用 pH 计算器]
        基于电荷平衡和强酸强碱假设的简化计算。
        """
        vol_ml = vessel.storage.solvent_volume
        if vol_ml <= 1e-6: return 7.0
        
        vol_L = vol_ml / 1000.0
        total_h = 0.0
        total_oh = 0.0
        
        # 遍历液相
        for chem, moles in vessel.storage.liquid.items():
            # 这里需要 DB 提供物质类型 (acid/base) 和价态 (valence)
            # 如果 DB 中没有这些字段，默认跳过
            info = CHEM_DB.substances.get(chem, {})
            c_type = info.get("type", "salt")
            
            # 简单的酸碱判断 (实际项目应在 JSON 中配置 'acidity_k_a' 等)
            if c_type == "acid" or chem.startswith("H"): 
                # 简单启发式：如果是酸
                valence = 1
                if "2" in chem: valence = 2 # 如 H2SO4
                total_h += moles * valence
                
            elif c_type == "base" or chem.endswith("OH"):
                # 简单启发式：如果是碱
                valence = 1
                if "(OH)2" in chem: valence = 2
                total_oh += moles * valence
        
        net_h = (total_h - total_oh) / vol_L
        
        # 纯水
        if abs(net_h) < 1e-9: return 7.0
        
        # 酸性
        if net_h > 0:
            return -math.log10(net_h)
        # 碱性
        else:
            pOH = -math.log10(abs(net_h))
            return 14.0 - pOH
    
class Heater(LabObject):
    """
    [加热类]
    核心功能：提供热能。
    拥有开关 (is_on)。
    复用 ThermalComponent 来管理其自身的温度（例如：发热盘的温度）。
    """
    def __init__(self, obj_id: str, name: str):
        super().__init__(obj_id, name)
        self.is_on = False
        
        # [新增] 组合 ThermalComponent
        # 初始温度为环境温度，加热器的热容通常较大 (金属/陶瓷)
        self.thermal = ThermalComponent(initial_temp=25.0)
        # 可以手动设置加热器的基础热容 (比玻璃试管大)
        self.thermal.base_heat_capacity = 200.0 
        
        # 设定目标温度 (例如酒精灯火焰 ~500度，电热板可调)
        self.target_temp = 500.0 

    @property
    def temperature(self):
        return self.thermal.temperature
        
    @temperature.setter
    def temperature(self, val):
        self.thermal.temperature = val

    def turn_on(self):
        self.is_on = True
    
    def turn_off(self):
        self.is_on = False

    def describe(self) -> str:
        status = "🔥" if self.is_on else "OFF"
        return f"{self.name} [{status}] (当前温度: {self.temperature:.0f}°C)"
    
    def get_snapshot(self) -> Dict[str, Any]:
        """[重写] 返回开关状态和温度"""
        base = super().get_snapshot()
        base.update({
            "is_on": self.is_on,
            "target_temp": self.target_temp,
            "current_temp": self.temperature
        })
        return base
        
    def update_self_temp(self, dt: float):
        """
        [新增] 加热器自身的升温逻辑 (用于微时间步循环)
        如果开启，自身温度趋向 target_temp；如果关闭，趋向环境温度。
        """
        target = self.target_temp if self.is_on else 25.0
        # 简单的趋近算法
        rate = 0.1 * dt
        self.temperature += (target - self.temperature) * rate

# --- Category 3: 夹持/支撑类 (Holding/Support) ---
class Support(LabObject):
    """
    [夹持类]
    核心功能：提供物理支撑，构建装置拓扑。
    如：铁架台 (Iron Stand)、试管架 (Tube Rack)、三角架 (Tripod)。
    """
    def __init__(self, obj_id: str, name: str):
        super().__init__(obj_id, name)
        # 可以增加属性：capacity (能夹几个), stability (稳定性)
    
    def describe(self) -> str:
        return f"{self.name} (固定座)"

# --- Category 4: 工具/传输类 (Tool/Transfer) ---
class Tool(LabObject):
    """
    [工具类]
    核心功能：辅助操作，不能长期储存物质，不提供热源。
    如：玻璃棒 (Glass Rod)、药匙 (Spoon)、胶头滴管 (Dropper)。
    """
    def __init__(self, obj_id: str, name: str):
        super().__init__(obj_id, name)
        # 工具可能也会短暂沾染化学品
        self.held_chemical = None

    def describe(self) -> str:
        return f"{self.name} (工具)"

# 2.4 器材兵工厂 (Mapping Factory)
class EquipmentFactory:
    """
    [映射中心] 将器材 ID 映射到具体的功能类
    """
    CLASS_MAP = {
        "Vessel": Vessel,
        "Heater": Heater,
        "Support": Support,
        "Tool": Tool
    }
    _manifest_cache = {}

    @classmethod
    def load_manifest(cls):
        """加载 LLM 生成的 YAML"""
        if cls._manifest_cache: return
        path = os.path.join("database", "auto_equipment_manifest.yaml")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                cls._manifest_cache = data.get("types", {})

    @staticmethod
    def create(config: Dict[str, Any]) -> LabObject:
        EquipmentFactory.load_manifest() # 确保已加载
        
        oid = config.get("id")
        raw_type = str(config.get("type", "unknown")).strip()
        
        # 1. 从 Manifest 查找配置
        # 支持 XDL 中的 type 直接匹配 manifest key
        manifest_item = EquipmentFactory._manifest_cache.get(raw_type)
        
        # 如果找不到，尝试模糊匹配 (兜底)
        if not manifest_item:
            # 简单的包含匹配兜底
            for k, v in EquipmentFactory._manifest_cache.items():
                if k in raw_type or raw_type in k:
                    manifest_item = v
                    break
        
        # 默认值兜底
        if not manifest_item:
            manifest_item = {"class_category": "Vessel", "human_name": raw_type}

        # 2. 决定 Python 类
        category = manifest_item.get("class_category", "Vessel")
        target_class = EquipmentFactory.CLASS_MAP.get(category, Vessel)
        
        # 3. 决定名称
        final_name = config.get("name") or manifest_item.get("human_name")
        
        # 4. 实例化
        obj = None
        if issubclass(target_class, Vessel):
            # === [修改开始] 容量读取逻辑优化 ===
            # 优先级: 1. XDL 配置 (config) > 2. Manifest 默认值 > 3. 系统兜底 (500)
            
            # 先尝试从 config 拿
            raw_cap = config.get("capacity")
            
            # 如果 config 没写，去 manifest_item 里拿
            if not raw_cap:
                raw_cap = manifest_item.get("capacity")
            
            # 如果都没写，最后兜底 500
            if not raw_cap:
                raw_cap = "500"

            # 统一处理字符串清洗
            cap_str = str(raw_cap).lower().replace("ml", "")
            try: 
                cap = float(cap_str)
            except: 
                cap = 500.0
            # === [修改结束] ===
            is_sealed = str(config.get("sealed", "false")).lower() == "true"
            
            obj = target_class(oid, capacity=cap, name=final_name, is_sealed=is_sealed)

            # 🌟🌟🌟 新增：如果是水槽，默认装满 80% 的水！ 🌟🌟🌟
            if "trough" in raw_type.lower() or "basin" in raw_type.lower():
                # 加入 H2O，量为容量的 80%
                water_vol = cap * 0.8
                obj.add_chemical("H2O", water_vol, "ml")
                # print(f"自动为水槽 {oid} 注入了 {water_vol}ml 水")
            
        elif issubclass(target_class, Heater):
            obj = target_class(oid, name=final_name)
            if "target_temp" in config:
                obj.target_temp = float(config["target_temp"])
        else:
            obj = target_class(oid, name=final_name)
            
        # [可选] 将 manifest 中的 extra static data 注入对象，供 HardwareManager 使用
        # 但通常 HardwareManager 会自己再去读一遍 manifest，所以这里只需返回对象
            
        return obj

# ==========================================
# 3. 反应管理器 (Reaction Solver)
# ==========================================
class ReactionManager:
    """
    [v6.4 Physics & Chemistry] 动力学反应管理器
    升级内容：
    1. 增加物理溶解逻辑 (Solid -> Aqueous)。
    2. 保持原有的热力学冷却和化学反应动力学。
    """
    
    # === 静态参数配置 ===
    ENV_TEMP = 25.0        # 环境温度
    COOLING_COEFF = 0.05   # 冷却系数 (牛顿冷却定律)
    BASE_REACTION_RATE = 0.2 # 基础反应速率

    # 静态缓存：所有实例共享编译好的反应列表
    _compiled_cache: List[CompiledReaction] = []
    _is_initialized = False

    def __init__(self):
        # 实例属性
        self.active_reactions = []
        self.last_phenomena = []
        
        # 构造函数防线：如果还没初始化，顺手初始化一下
        if not ReactionManager._is_initialized:
            ReactionManager._compile_all_reactions()

    @classmethod
    def _compile_all_reactions(cls):
        """
        [v14.3 Critical Fix] 启动时批量编译
        采用“乐观锁定”策略：进入函数立即上锁，防止因部分编译失败导致的每帧重试死循环。
        """
        # 1. 第一道防线：如果已初始化，直接跳过
        if cls._is_initialized:
            return

        # 2. [核心修复] 立即上锁！
        # 无论后续编译是否抛出 Warning，都不允许再次进入此函数
        cls._is_initialized = True
        
        print(Fore.YELLOW + "⚡ [System] Pre-compiling chemical reactions (One-time Init)...")
        
        count = 0
        try:
            # 3. 执行编译
            for rxn_data in REACTIONS_DB: 
                # 增加容错保护
                try:
                    compiled = ChemSolver.compile_reaction(rxn_data)
                    if compiled:
                        cls._compiled_cache.append(compiled)
                        count += 1
                except Exception as e:
                    # 单个反应编译失败不影响大局
                    continue
            
            print(Fore.GREEN + f"✅ Compiled {count} reactions successfully.")
            
        except Exception as e:
            # 即使发生灾难性错误，也不要撤销 _is_initialized，否则会卡死主循环
            print(Fore.RED + f"❌ Critical Error during compilation: {e}")
    
    # [新增] 溶解速率系数 (每秒溶解的比例)
    # 这里设为 0.5，意味着在搅拌或理想状态下，易溶物质很快就会溶完
    DISSOLUTION_RATE_CONST = 0.5 

    @staticmethod
    def step_simulate(target: LabObject, dt: float) -> List[str]:
        """
        [v17.0 Data-Driven] 数据驱动的通用化学引擎
        - 自动读取 CompiledReaction 中的 temp_threshold 判断反应条件
        - 自动读取 phenomena 字段生成实验现象
        - 移除硬编码逻辑，支持任意实验配置
        """
        if not ReactionManager._is_initialized:
            ReactionManager._compile_all_reactions()

        # 1. 基础检查
        if isinstance(target, Heater):
            target.update_self_temp(dt)
            return []
        if not isinstance(target, Vessel):
            return []

        phenomena_report = []
        frame_gas_moles = 0.0
        
        current_temp = target.temperature
        # === [修复 1] 读取所有相态的物质 ===
        # 不要用 target.contents (那是空的)，要用 storage 组件的聚合视图
        all_contents = target.storage.get_all_contents()
        available_chemicals = set(all_contents.keys())

        # ==================================================
        # 核心逻辑: 遍历所有可能的反应
        # ==================================================
        for rxn in ReactionManager._compiled_cache:
            
            # --- Step A: 快速筛选 ---
            if not rxn.reactants: continue
                
            missing_reactant = False
            for r in rxn.reactants:
                # [修复 2] 检查聚合后的 all_contents
                if r not in available_chemicals or all_contents[r] <= 1e-9:
                    missing_reactant = True
                    break
            if missing_reactant:
                continue

            # --- Step B: 基于数据的温度阈值检查 ---
            # [关键改进] 直接从反应对象读取阈值，不再硬编码
            # 如果反应对象没配阈值，默认 -273.0 (即常温可反应)
            threshold = getattr(rxn, 'temp_threshold', -273.0)
            
            if current_temp < threshold:
                continue # 温度未达标，反应不发生

            # --- Step C: 动力学速率计算 ---
            # 根据阈值类型决定速率模型
            reaction_rate_k = 0.0
            
            if threshold > 50.0:
                # === 高温触发型反应 (如分解、燃烧) ===
                # 温度越高越过阈值，反应越快
                # k = base * (T - T_thresh) * dt
                overdrive = current_temp - threshold
                reaction_rate_k = 0.05 * overdrive * dt
            else:
                # === 常温自发型反应 (如中和、置换) ===
                # 默认反应较快 (受限于混合速率，这里简化为固定快慢)
                reaction_rate_k = 0.5 * dt

            # 限制转化率
            reaction_rate_k = max(0.0, min(0.067, reaction_rate_k))
            if reaction_rate_k <= 1e-10: continue

            # --- Step D: 计算受限试剂 (Limiting Reagent) ---
            max_reaction_moles = float('inf')
            for r_name, r_coeff in rxn.reactants.items():
                possible_moles = all_contents[r_name] / r_coeff
                if possible_moles < max_reaction_moles:
                    max_reaction_moles = possible_moles
            
            # --- Step E: 计算本帧实际反应量 ---
            # [修改后] 强制残留机制 (Residue Mechanism)
            # 即使速率很快，单次微步也最多只反应掉当前剩余量的 90%
            # 这样就像“芝诺的乌龟”一样，原料永远只会无限趋近于 0，但不会等于 0
            theoretical_step = max_reaction_moles * reaction_rate_k
            limit_step = max_reaction_moles * 0.90 # 保留 10% 底料
            
            actual_moles_step = min(theoretical_step, limit_step)

            if actual_moles_step < 1e-12: continue # 只有极小极小时才忽略

            # 记录案底
            if hasattr(rxn, 'id'):
                target.reaction.occurred_reactions.add(rxn.id)

            # --- Step F: 执行物质更新 [核心修复] ---
            # 不要直接操作 contents，而是构建 changes 字典，交给 _apply_mass_changes 处理
            
            changes = {}
            # 1. 消耗反应物 (负值)
            for r_name, r_coeff in rxn.reactants.items():
                changes[r_name] = -1 * actual_moles_step * r_coeff

            # 2. 增加生成物 (正值)
            for p_name, p_coeff in rxn.products.items():
                produce_amount = actual_moles_step * p_coeff
                changes[p_name] = produce_amount
                
                # 统计气体 (用于现象)
                if p_name in ["O2", "H2", "CO2", "NH3", "Cl2", "HCl"]:
                    frame_gas_moles += produce_amount

            # [修复 4] 调用正确的更新方法，它会根据 state_map 自动把 O2 放进 gas，把 MnO2 放进 solid
            ReactionManager._apply_mass_changes(target, changes, rxn.state_map)
            
            # (为了下一轮循环能读到最新值，这里可选更新一下 all_contents，或者依赖下一帧重新读取)
            # 简单起见，我们假设单帧内反应物消耗不影响同帧内其他反应的发生判断(近似)

            # --- Step G: 记录现象 ---
            if actual_moles_step > 1e-6:
                custom_phenomena = getattr(rxn, 'phenomena', None)
                if custom_phenomena:
                    phenomena_report.append(custom_phenomena)
                else:
                    if frame_gas_moles > 0:
                        phenomena_report.append("bubbling")
                    else:
                        phenomena_report.append("chemical_reaction")
                
                rxn_id = getattr(rxn, 'id', 'unknown_rxn')
                target.dynamics.active_reactions.append(rxn_id)

        # ==================================================
        # 3. 物理状态更新
        # ==================================================
        target.dynamics.gas_delta_moles += frame_gas_moles

        # [注] 依然跳过焓变计算以保持数值稳定。
        # 如果您希望“溶液变热”，建议通过 phenomena 字符串提示给用户，
        # 而不是冒着 NaN 的风险去修改 target.temperature。

        return list(set(phenomena_report))

    @staticmethod
    def _simulate_dissolution(target: Vessel, dt: float) -> List[str]:
        """
        [v6.6 Physics Fixed] 基于 Ksp/溶解度表的动态溶解逻辑
        修复：使用 deferred deletion (延迟删除) 避免迭代时修改字典
        """
        # 如果没有溶剂，无法溶解
        vol_ml = target.solvent_volume
        if vol_ml < 1.0: return []

        reports = []
        base_rate = ReactionManager.DISSOLUTION_RATE_CONST 
        
        # [修复] 1. 初始化待删除列表
        solids_to_remove = []

        # 使用 list() 创建副本进行遍历是安全的，但为了逻辑清晰，我们不在这里做 del
        for solid_chem in list(target.solid_contents.keys()):
            moles_solid = target.solid_contents[solid_chem]
            
            # --- 溶解度查表与计算逻辑 (保持不变) ---
            info = CHEM_DB.substances.get(solid_chem, {})
            default_sol = 0.0 if info.get("type") in ["oxide", "metal"] else -1
            solubility_g_per_100ml = info.get("solubility", default_sol)
            molar_mass = info.get("molar_mass", 100.0)
            
            if solubility_g_per_100ml < 0: 
                max_soluble_moles = float('inf')
            else:
                max_soluble_moles = (solubility_g_per_100ml / 100.0 * vol_ml) / molar_mass
            
            current_dissolved_moles = target.liquid_contents.get(solid_chem, 0.0)
            dissolve_room = max_soluble_moles - current_dissolved_moles
            
            if dissolve_room <= 1e-9: continue
            
            theoretical_dissolve = moles_solid * base_rate * dt
            actual_dissolve = min(theoretical_dissolve, dissolve_room)
            
            # --- 执行转移 ---
            if actual_dissolve > 1e-9:
                target.solid_contents[solid_chem] -= actual_dissolve
                target.liquid_contents[solid_chem] = current_dissolved_moles + actual_dissolve
                
                # [修复] 2. 标记需要删除的项，而不是直接 del
                if target.solid_contents[solid_chem] < 1e-6:
                    solids_to_remove.append(solid_chem)
                    
                    # 记录现象
                    app = info.get("appearance", {})
                    if app.get("base_color") != "无色":
                        reports.append(f"{solid_chem} 固体完全溶解")

        # [修复] 3. 循环结束后统一执行删除
        for chem in solids_to_remove:
            if chem in target.solid_contents:
                del target.solid_contents[chem]

        return reports

    @staticmethod
    def _check_availability(target: Vessel, changes: Dict[str, float]) -> bool:
        """检查反应物是否足够"""
        for chem, delta in changes.items():
            if delta < 0: 
                needed = abs(delta)
                # 检查所有相态的总和
                owned = target.liquid_contents.get(chem, 0) + \
                        target.solid_contents.get(chem, 0) + \
                        target.gas_contents.get(chem, 0)
                if owned < (needed - 1e-9): return False
        return True

    @staticmethod
    def _apply_mass_changes(target: Vessel, changes: Dict[str, float], state_map: Dict[str, str]):
        """
        [v14.0 Routing Fix] 
        执行物质增减，利用预编译的 state_map 进行精准路由，不再盲目猜测。
        参数 state_map 来自 CompiledReaction，例如 {'H2': 'g', 'H2O': 'l'}
        """
        # 待删除队列 (Deferred Deletion)
        to_clean = {
            'l': [], # liquid keys to delete
            'g': [], # gas keys to delete
            's': []  # solid keys to delete
        }

        for chem, delta in changes.items():
            
            # 1. 确定物理状态 (Priority: Equation > DB > Default)
            state = state_map.get(chem)
            if not state:
                # 如果方程式没写 (e.g. "A + B -> C")，查库兜底
                info = CHEM_DB.substances.get(chem, {})
                state = info.get("state", "l")
            
            # 统一状态标签 (兼容 aq -> l)
            if state == 'aq': state = 'l'

            # 2. 路由到对应的字典
            target_dict = None
            clean_list = None
            
            if state == 'g':
                target_dict = target.storage.gas
                clean_list = to_clean['g']
            elif state == 's':
                target_dict = target.storage.solid
                clean_list = to_clean['s']
            else: # l or aq
                target_dict = target.storage.liquid
                clean_list = to_clean['l']

            # 3. 执行增减 (无需区分 Consume/Produce，直接加 Delta)
            # 负数 Delta 自动变成减法
            current_val = target_dict.get(chem, 0.0)
            new_val = current_val + delta
            
            # 4. 边界处理与更新
            if new_val < 1e-9:
                # 如果减过头了或者归零了
                if chem in target_dict:
                    clean_list.append(chem) # 标记删除
                # 理论上不应减过头，因为 calculate_max_runs 保证了原料充足
            else:
                target_dict[chem] = new_val

            # 5. 特殊联动：生成液态水增加溶剂体积
            if chem == "H2O" and state == 'l':
                # 18g/mol, 1g/ml -> 18ml/mol
                target.solvent_volume += delta * 18.0

        # 6. 统一清理零值键
        for k in to_clean['l']: 
            if k in target.storage.liquid: del target.storage.liquid[k]
        for k in to_clean['g']: 
            if k in target.storage.gas: del target.storage.gas[k]
        for k in to_clean['s']: 
            if k in target.storage.solid: del target.storage.solid[k]
                        
# ==========================================
# 4. 硬件管理器 (Hardware Topology - Graph Version)
# ==========================================
class HardwareManager:
    """
    [v6.0 Tag-Based Topology]
    基于标签系统的硬件拓扑管理器。
    移除了所有硬编码的类型检查，支持通过 YAML 扩展新器材。
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.manifest = {}
        self._load_manifest()

    def _load_manifest(self):
        """加载硬件配置文件"""
        try:
            path = os.path.join("database", "equipment_manifest.yaml")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    self.manifest = data.get("types", {})
                    print(f"✅ Loaded {len(self.manifest)} equipment types from manifest.")
            else:
                print("⚠️ Hardware manifest not found. Using fallback.")
                self.manifest = self._get_fallback_manifest()
        except Exception as e:
            print(f"❌ Error loading hardware manifest: {e}")
            self.manifest = self._get_fallback_manifest()

    def get_equipment_config(self, obj_type: str) -> Dict[str, Any]:
        """[Helper] 获取某类型器材的配置，支持模糊匹配 (如 test_tube -> tube)"""
        # 1. 精确匹配
        if obj_type in self.manifest:
            return self.manifest[obj_type]
        
        # 2. 模糊匹配 (简单的后缀匹配)
        # 例如: "condenser_tube" -> 匹配 "tube"
        for key in self.manifest:
            if key in obj_type or obj_type in key:
                return self.manifest[key]
        
        # 3. 兜底
        return {
            "tags": [], "my_tags": [], "connectable_to": [], 
            "ports": ["center"], "capacity": 999
        }

    def get_valid_ports(self, obj_type: str) -> List[str]:
        """[动态端口] 从配置中读取端口列表"""
        cfg = self.get_equipment_config(obj_type)
        return cfg.get("ports", ["center", "mount_point"])

    def add_hardware(self, hardware_ids: List[str], hardware_types: Dict[str, str]):
        """初始化节点"""
        for vid in hardware_ids:
            if vid not in self.graph:
                self.graph.add_node(vid)
                vtype = hardware_types.get(vid, "unknown").lower()
                self.graph.nodes[vid]["type"] = vtype

    def _get_type(self, vid: str) -> str:
        return self.graph.nodes[vid].get("type", "unknown")

    def validate_connection(self, child: str, parent: str, conn_type: str) -> Tuple[bool, str]:
        """
        [Core Logic] 基于标签的连接校验
        算法：
        1. 获取 Child 需要连接的目标标签 (connectable_to)。
        2. 获取 Parent 拥有的标签 (tags + my_tags)。
        3. 检查是否有交集。
        4. 检查 Parent 容量限制。
        """
        if child == parent:
            return False, "无法连接自身。"

        child_type = self._get_type(child)
        parent_type = self._get_type(parent)
        
        child_cfg = self.get_equipment_config(child_type)
        parent_cfg = self.get_equipment_config(parent_type)

        # === 1. 机械/物理兼容性检查 (Tag Matching) ===
        if conn_type == "mechanical":
            # Child 说：我能连 [A, B]
            needed_tags = set(child_cfg.get("connectable_to", []))
            
            # Parent 说：我拥有 [B, C, D]
            # 注意：Parent 的 tag 包括它的通用 tags (e.g. glassware) 和 专属 my_tags (e.g. tube_tag)
            # 还可以扩展：如果 Parent 也是 Child 的某种形式（比如石棉网放在三脚架上，石棉网对外提供了 flat_platform 属性）
            parent_tags = set(parent_cfg.get("tags", []) + parent_cfg.get("my_tags", []))
            
            # 特殊处理：如果 Parent 是 "my_extra_tags_for_children" (例如石棉网对外提供 flat_platform)
            parent_tags.update(parent_cfg.get("my_extra_tags_for_children", []))

            # 检查交集
            # 如果 needed_tags 里有 "vessel"，而 Parent tags 里也有 "vessel"，则匹配成功
            intersection = needed_tags.intersection(parent_tags)
            
            # [特殊规则] 如果 Child 什么都不要求 (空列表)，或者 Parent 是万能的 (如 Table)，可以放行
            # 这里我们采用严格模式：必须显式匹配
            if not intersection:
                # === [FIX] 宽松模式：忽略物理不兼容，强制允许连接 ===
                # 即使标签不匹配（比如 crucible 没写能连 pipeclay_triangle），也假装成功。
                print(f"⚠️ [Loose Physics] 忽略物理不兼容: {child} -> {parent} (需要: {needed_tags}, 拥有: {parent_tags})")
                # return False, f"物理不兼容: ..."  <--- 关键：把这行注释掉！
                pass

        # === 2. 容量检查 (Capacity Check) ===
        # 统计 Parent 当前已经挂了多少个 Child
        current_children = list(self.graph.in_edges(parent))
        current_load = len(current_children)
        max_cap = parent_cfg.get("capacity", 999)

        if current_load >= max_cap:
             # === [FIX] 宽松模式：忽略容量限制，强制允许连接 ===
             print(f"⚠️ [Loose Physics] 忽略容量限制: {parent} (Capacity: {max_cap}, Current: {current_load})")
             # 原始逻辑：return False, f"容量已满: {parent} 只能连接 {max_cap} 个物体，当前已有 {current_load} 个。"
             pass

        # === 3. [新增] 流体连接检查 (Fluid Check) ===
        # 原代码中没有这段，请直接粘贴到这里
        if conn_type == "fluid":
            # 1. 获取原始配置
            c_ports_raw = child_cfg.get("ports", [])
            p_ports_raw = parent_cfg.get("ports", [])

            # 2. 归一化处理：如果是列表，直接作为通用端口；如果是字典，提取对应的 out/in
            if isinstance(c_ports_raw, list):
                child_ports = c_ports_raw
            else:
                child_ports = c_ports_raw.get("out", [])
            
            if isinstance(p_ports_raw, list):
                parent_ports = p_ports_raw
            else:
                parent_ports = p_ports_raw.get("in", [])

            # 3. 如果端口列表为空（比如 cabinet），但在宽松物理模式下，我们通常希望允许连接
            # 或者你可以保持原有的严格检查
            if not child_ports or not parent_ports:
                # 遇到空端口配置，为了防止死锁，可以视为“通用匹配”或者“不匹配”
                # 鉴于你的 rubber_stopper 可能没有配置 ports，建议这里设为 True (宽松)
                can_connect = True 
            else:
                can_connect = not set(child_ports).isdisjoint(parent_ports)
            
            if not can_connect:
                # === [FIX] 宽松模式：忽略流体端口不匹配 ===
                # 即使是用钳子(无端口)去连坩埚，也强制允许
                print(f"⚠️ [Loose Physics] 忽略流体端口不匹配: {child} -> {parent}")
                # return False, f"流体接口不兼容: {child} 无法流体连接到 {parent}。" <-- 确保这行是不存在的或被注释的
                pass

        return True, ""

    def attach(self, child: str, parent: str, 
               connection_type: str = "mechanical", 
               child_port: str = None, 
               parent_port: str = None) -> Tuple[bool, str]:
        """
        [通用连接动作]
        """
        if child not in self.graph or parent not in self.graph:
            return False, "设备不存在。"

        # 1. 自动推断端口 (如果未指定)
        # 获取第一个可用端口作为默认值
        if not child_port:
            c_ports = self.get_valid_ports(self._get_type(child))
            child_port = c_ports[0] if c_ports else "center"
            
        if not parent_port:
            p_ports = self.get_valid_ports(self._get_type(parent))
            # 机械连接默认连 mount_point/base，流体连 mouth
            default_p = "mount_point" if connection_type == "mechanical" else "mouth"
            # 尝试在合法端口里找，找不到就用第一个
            parent_port = default_p if default_p in p_ports else (p_ports[0] if p_ports else "center")

        # 2. 校验
        valid, msg = self.validate_connection(child, parent, connection_type)
        if not valid: return False, msg

        # 3. 执行连接
        self.graph.add_edge(child, parent, 
                            type=connection_type,
                            src_port=child_port,
                            dst_port=parent_port)

        return True, f"成功将 {child} 连接到 {parent} [{child_port}->{parent_port}]。"

    def detach(self, child: str, parent: str = None) -> Tuple[bool, str]:
        if child not in self.graph: return False, "设备不存在。"
        
        detached_list = []
        visited = set()  # <--- 新增：防止环路导致的死循环

        def _recursive_detach(node):
            if node in visited: return # 防止重入
            visited.add(node)
            
            # 找到依附于 node 的所有子节点
            dependents = list(self.graph.in_edges(node))
            for dep, _ in dependents:
                self.graph.remove_edge(dep, node)
                detached_list.append(dep)
                _recursive_detach(dep)

        if parent:
            if self.graph.has_edge(child, parent):
                self.graph.remove_edge(child, parent)
                _recursive_detach(child)
                msg = f"已断开 {child} 与 {parent} 的连接。"
                if detached_list: msg += f" (级联脱落: {', '.join(detached_list)})"
                return True, msg
            else:
                return False, "连接不存在。"
        else:
            # 彻底拆卸
            parents = list(self.graph.out_edges(child))
            for _, p in parents:
                self.graph.remove_edge(child, p)
            _recursive_detach(child)
            msg = f"已拆卸 {child}。"
            if detached_list: msg += f" (级联脱落: {', '.join(detached_list)})"
            return True, msg
            
    def get_topology_snapshot(self) -> List[Dict[str, str]]:
        """获取拓扑快照"""
        return [{"child": u, "parent": v, "type": d.get("type")} for u, v, d in self.graph.edges(data=True)]

    def _get_fallback_manifest(self):
        """最小化兜底配置，防止无 YAML 时报错"""
        return {
            "tube": {
                "tags": ["glassware", "vessel"], "my_tags": ["tube_tag"],
                "connectable_to": ["clamp_tag", "rack_tag"], "ports": ["mouth"], "capacity": 3
            },
            "clamp": {
                "tags": ["connector"], "my_tags": ["clamp_tag"],
                "connectable_to": ["stand_tag"], "ports": ["jaw"], "capacity": 1
            },
            "stand": {
                "tags": ["support"], "my_tags": ["stand_tag"],
                "connectable_to": ["table_tag"], "ports": ["pole"], "capacity": 5
            }
        }

# ==========================================
# 5. 主引擎 (ChemSimEngine)
# ==========================================
class ChemSimEngine:
    # 1. 修改初始化函数，接收 reagent_map
    def __init__(self, hardware_config: List[Dict], reagent_map: Dict, 
                 clumsiness_level: str = "low", logger_name: str = "ChemSimEngine",
                 silent_mode: bool = False): # <--- [NEW] 新增参数
        
        # [修复核心] 使用传入的 logger_name
        self.logger = logging.getLogger(logger_name)
        self.silent_mode = silent_mode  # <--- [NEW] 保存标志位
        
        # [新增] 保存初始配置，供 fork 使用
        self.initial_hw_config = hardware_config 
        self.initial_reagent_map = reagent_map

        self._shadow_engine = None
        
        # [修正] 直接使用参数 clumsiness_level 赋值，删除错误的 clumsy_level
        self.clumsiness_level = clumsiness_level

        self.containers: Dict[str, LabObject] = {} 
        self.reagent_map = reagent_map
        self.checkpoints = {}
        self.hw_manager = HardwareManager()
        # === [修复] 初始化 procedure 存储字典，防止 set_procedure_reference 报错 ===
        self.initial_xdl_procedure: Dict[str, Dict] = {}
        
        # 初始化硬件
        self._init_hardware(hardware_config)
        
        # 注册动作处理器
        self.action_dispatch = {
            "Add": self._handle_add,
            "Transfer": self._handle_transfer,
            "Refill": self._handle_refill,
            "Filter": self._handle_filter,
            "Wash": self._handle_wash,
            "Heat": self._handle_heat,
            "Cool": self._handle_cool,
            "Stir": self._handle_stir,
            "CollectGas": self._handle_collect_gas,
            "Attach": self._handle_attach,
            "Detach": self._handle_detach,
            "Insert": self._handle_insert,
            "MeasureTemperature": self._handle_measure_temp,
            "MeasureMass": self._handle_measure_mass,

            # === [修复] 加上这一行 ===
            "Wait": self._handle_wait,
        }

    def get_shadow_engine(self):
        """延迟初始化且只初始化一次"""
        if self._shadow_engine is None:
            self._shadow_engine = ChemSimEngine(
                self.initial_hw_config, 
                self.initial_reagent_map, 
                clumsiness_level=self.clumsiness_level,
                logger_name="GhostEngine",
                silent_mode=True
            )
        return self._shadow_engine

    def set_procedure_reference(self, parsed_actions: List[Dict]):
        """
        [数据源] 将解析后的动作列表转为 ID 索引的字典
        结构: {'step_0': {'action': 'Add', 'reagent': 'KMnO4', ...}, ...}
        """
        self.initial_xdl_procedure = {}
        for action in parsed_actions:
            if 'id' in action:
                self.initial_xdl_procedure[action['id']] = action

    def clone_from(self, source_engine: 'ChemSimEngine'):
        """
        [优化] 使用 pickle 替代 deepcopy，通常速度提升 2-5 倍
        """
        import pickle
        # pickle 序列化再反序列化，能够完整保留对象引用关系，且比 deepcopy 快
        state_bytes = pickle.dumps(source_engine.containers)
        self.containers = pickle.loads(state_bytes)

        self.hw_manager.graph = source_engine.hw_manager.graph.copy()
        self.clumsiness_level = source_engine.clumsiness_level
        self.reagent_map = source_engine.reagent_map 
        self.initial_xdl_procedure = source_engine.initial_xdl_procedure

    # ==========================================
    # [NEW] Log Gatekeepers (日志守门员)
    # ==========================================
    def _log_info(self, msg: str):
        """仅在非静默模式下打印 Info"""
        if not self.silent_mode:
            self.logger.info(msg)

    def _log_warning(self, msg: str):
        """仅在非静默模式下打印 Warning"""
        if not self.silent_mode:
            self.logger.warning(msg)

    def _log_error(self, msg: str, exc_info=False):
        """
        Error 通常比较严重，策略可选：
        1. 即使是幽灵模式也打印 Error (便于调试)
        2. 或者严格遵守静默 (如下所示)
        这里我们选择严格静默，因为幽灵模式下的 Error 会通过返回值传递给主流程
        """
        if not self.silent_mode:
            self.logger.error(msg, exc_info=exc_info)

    def set_procedure_reference(self, enriched_procedure: List[Dict]):
        """
        [关键] 保存带有 ID 的标准步骤，用于验证器对比
        """
        self.initial_xdl_procedure = {}
        for step in enriched_procedure:
            if 'id' in step:
                self.initial_xdl_procedure[step['id']] = step
    # ==========================================
    # [新增] 物理误差注入工具
    # ==========================================
    def _apply_clumsiness_noise(self, val: float) -> float:
        """
        根据笨拙度引入随机误差。
        模拟真实世界中“想倒 5ml 但手抖倒了 5.8ml”的情况。
        """
        return val
    
    # 在 ChemSimEngine 类中

    def get_state_dict(self) -> Dict[str, Any]:
        """
        [修复版] 状态快照，确保所有对象都包含 'type' 字段
        """
        state = {
            "containers": {},
            "topology": []
        }

        # 1. 提取器材状态
        for vid, obj in self.containers.items():
            obj_state = {}
            # 默认必须保存 type，否则 load 时会报错
            base_info = {"type": type(obj).__name__} # 获取类名，如 'Vessel', 'Heater', 'Support'
            
            if isinstance(obj, Vessel):
                obj_state = {
                    "temperature": obj.temperature,
                    "solvent_volume": obj.solvent_volume,
                    "is_sealed": obj.is_sealed,
                    "is_covered": obj.is_covered, # <--- [新增]
                    "is_dirty": obj.is_dirty,
                    "liquid": obj.storage.liquid.copy(),
                    "solid": obj.storage.solid.copy(),
                    "gas": obj.storage.gas.copy()
                }
            elif isinstance(obj, Heater):
                obj_state = {
                    "temperature": obj.temperature,
                    "is_on": obj.is_on,
                    "target_temp": obj.target_temp
                }
            
            # 合并基础信息和特有状态
            full_state = {**base_info, **obj_state}
            state["containers"][vid] = full_state

        # 2. 提取拓扑状态
        state["topology"] = list(self.hw_manager.graph.edges(data=True))

        return state

    def load_state_dict(self, state: Dict[str, Any]):
        """
        [修复版] 状态回滚
        """
        # 1. 恢复器材状态
        c_states = state.get("containers", {})
        for vid, data in c_states.items():
            obj = self.containers.get(vid)
            if not obj: continue 

            # 根据类型恢复属性
            obj_type = data.get("type")
            
            if obj_type == "Vessel" and isinstance(obj, Vessel):
                obj.temperature = data.get("temperature", 25.0)
                obj.solvent_volume = data.get("solvent_volume", 0.0)
                obj.is_sealed = data.get("is_sealed", False)
                obj.is_covered = data.get("is_covered", False) # <--- [新增]
                obj.is_dirty = data.get("is_dirty", False)
                obj.storage.liquid = data.get("liquid", {}).copy()
                obj.storage.solid = data.get("solid", {}).copy()
                obj.storage.gas = data.get("gas", {}).copy()
            
            elif obj_type == "Heater" and isinstance(obj, Heater):
                obj.temperature = data.get("temperature", 25.0)
                obj.is_on = data.get("is_on", False)
                obj.target_temp = data.get("target_temp", 500.0)
            
            # Support 和 Tool 通常没有动态状态需要恢复，忽略即可

        # 2. 恢复拓扑状态
        self.hw_manager.graph.clear_edges()
        edges = state.get("topology", [])
        self.hw_manager.graph.add_edges_from(edges)

    # === [请插入这段缺失的代码到 ChemSimEngine 类中] ===
    def get_snapshot(self) -> Dict[str, Any]:
        """
        [v4.3 Fix] 获取全局物理快照，修复类型丢失问题
        """
        hardware_data = {}
        
        for obj_id, obj in self.containers.items():
            # 1. 基础快照
            if isinstance(obj, Vessel):
                # 调用观察引擎获取基础数据 (此时 type="vessel")
                obs = ObservationEngine.get_observation(obj)
                
                # === [FIX] 找回丢失的身份 ===
                # 从 HardwareManager 获取原始配置的 type (例如 "bottle", "tube")
                original_type = self.hw_manager._get_type(obj_id)
                if original_type and original_type != "unknown":
                    obs["type"] = original_type  # 强制覆盖，变回 "bottle"
                
                # 补全 name
                obs["name"] = getattr(obj, "name", obj_id)

                # 注入动态现象
                if hasattr(obj, 'reaction') and obj.reaction.last_phenomena:
                    obs["last_phenomena"] = obj.reaction.last_phenomena
                else:
                    obs["last_phenomena"] = []
                
                hardware_data[obj_id] = obs
            
            elif isinstance(obj, LabObject):
                # 对于 Heater 等其他物体，也要做类似处理
                snap = obj.get_snapshot()
                original_type = self.hw_manager._get_type(obj_id)
                if original_type:
                    snap["type"] = original_type
                hardware_data[obj_id] = snap
            else:
                hardware_data[obj_id] = {"error": "Unknown Object Type"}

        return {
            "hardware": hardware_data, 
            "topology": self.hw_manager.get_topology_snapshot()
        }
    # =================================================

    # =========================================================================
    # [Core Physics Update] Advanced Fluid Dynamics
    # Paste this method into class ChemSimEngine, replacing the old one.
    # =========================================================================
    def _simulate_gas_diffusion(self, dt: float):
        """
        [v12.0 Robust Physics] 全局气体扩散与堵塞检测
        
        功能：
        1. 构建流体拓扑：谁连着谁。
        2. 动态堵塞检测：判断“带导管塞子”的末端是被堵死(如连注射器)还是悬空(通大气)。
           - 更新 obj.storage.is_blocked 标志，供 pressure() 计算使用。
        3. 连通器平衡：在连通的容器组之间分配气体，并处理开放系统的自动泄压。
        """
        
        # =======================================================
        # 1. 构建流体拓扑图 (Topology Construction)
        # =======================================================
        fluid_graph = nx.Graph()
        # 将所有容器ID作为节点加入
        fluid_graph.add_nodes_from(self.containers.keys())
        
        # 将所有 'fluid' 类型（流体）的连接加入图
        # 注意：机械连接（如夹持）不导气，所以不加
        for u, v, data in self.hw_manager.graph.edges(data=True):
            if data.get("type") == "fluid":
                fluid_graph.add_edge(u, v)

        # =======================================================
        # 2. 动态检测堵塞状态 (Dynamic Blocking Check)
        # =======================================================
        # 这一步通过图分析，决定“带管塞子”是 blocked 还是 open
        for vid, obj in self.containers.items():
            if isinstance(obj, Vessel):
                # Init: 默认假设没堵死 (Open/Leaking)
                obj.storage.is_blocked = False 
                
                # 只有 "既密封(满足验证) 又通气(物理属性)" 的容器需要检查
                # 例如：插了 rubber_stop_tube 的试管
                # 如果是实心塞(hermetic)，is_vented为False，不进此逻辑，直接由is_sealed判定为高压
                if obj.is_sealed and obj.is_vented:
                    is_blocked = True # 先假设被堵死 (Worst case)
                    
                    # 寻找流体邻居 (即塞子/导管)
                    if fluid_graph.has_node(vid):
                        neighbors = list(fluid_graph.neighbors(vid))
                        
                        # Case A: 没有任何流体连接 -> 说明导管都没插好，或者虽然插了但没连入流体网
                        # 理论上 is_sealed=True 应该意味着有塞子，这里做个防御性检查
                        if not neighbors:
                            is_blocked = True
                        else:
                            # 遍历邻居 (塞子/导管)
                            for stopper_id in neighbors:
                                # 检查塞子在图中的“度” (Degree)
                                # stopper_neighbors 包含：[vid(试管自己), other_node(外界容器)]
                                stopper_neighbors = list(fluid_graph.neighbors(stopper_id))
                                
                                # Case B: 塞子只连了瓶子自己 (Degree=1) -> 导管悬空 -> 通大气
                                if len(stopper_neighbors) <= 1:
                                    is_blocked = False
                                    break
                                
                                # Case C: 塞子连了其他东西 (Degree > 1) -> 检查另一头是谁
                                for remote_node in stopper_neighbors:
                                    if remote_node == vid: continue # 跳过自己
                                    
                                    remote_obj = self.containers.get(remote_node)
                                    if remote_obj and isinstance(remote_obj, Vessel):
                                        # 如果连着一个“不密封”的容器 (如水槽、集气瓶、敞口烧杯)
                                        # 那么整个系统也是通气的
                                        if not remote_obj.is_sealed:
                                            is_blocked = False # 通了！
                                            break
                                    # 如果连的是空气节点或虚拟节点(未实现)，也可视情况设为False
                                
                                # 只要找到一条通气路，就认为没堵死
                                if not is_blocked: break
                    
                    # 将计算结果写入容器状态，供 pressure() 方法读取
                    obj.storage.is_blocked = is_blocked

        # =======================================================
        # 3. 气体平衡与泄压 (Gas Equilibrium & Venting)
        # =======================================================
        # 按连通分量 (Connected Components) 处理每个独立的流体系统
        for component in nx.connected_components(fluid_graph):
            vessels = [self.containers[vid] for vid in component if vid in self.containers]
            if not vessels: continue
            
            is_system_open = False
            
            # --- A. 检查系统开放性 (Check System Openness) ---
            for node_id in component:
                # 1. 检查是否有敞口容器 (Open Vessel)
                obj = self.containers.get(node_id)
                if isinstance(obj, Vessel):
                    # 只要有一个没密封且没盖盖子，整个连通器就是通大气的
                    if not obj.is_sealed and not obj.is_covered:
                        is_system_open = True
                        break
                
                # 2. 检查是否有悬空导管 (Dangling Tube / Leak)
                # 如果组件里包含一个度数为1的管子/塞子节点，说明系统有一端通大气
                # (排除掉 vessels 本身，只看连接件)
                obj_type = self.hw_manager._get_type(node_id).lower()
                if "tube" in obj_type or "stopper" in obj_type:
                    if fluid_graph.degree(node_id) < 2:
                        is_system_open = True
                        break

            # --- B. 汇总连通器内的气体与空间 (Aggregate) ---
            total_moles = {}
            total_headspace_L = 0.0
            avg_temp_sum = 0.0
            valid_vessel_count = 0

            for v in vessels:
                if not isinstance(v, Vessel): continue
                valid_vessel_count += 1
                
                # 汇总所有容器中的气体
                for gas, n in v.gas_contents.items():
                    total_moles[gas] = total_moles.get(gas, 0) + n
                
                # 汇总剩余空间 (Headspace)
                h_ml = max(0.0, v.capacity - v.solvent_volume)
                total_headspace_L += (h_ml / 1000.0)
                avg_temp_sum += v.temperature
            
            if valid_vessel_count == 0: continue

            # --- C. 开放系统泄压 (Venting) ---
            if is_system_open:
                # 计算平均温度 (K)
                avg_temp_k = (avg_temp_sum / valid_vessel_count) + 273.15 
                
                # 计算 1.0 atm 下能容纳的最大气体摩尔数 (n = PV/RT)
                # 如果气体总量超过这个数，说明气压>1atm，多余气体会逸出
                target_n = 0
                if avg_temp_k > 0:
                    target_n = (1.0 * total_headspace_L) / (0.082 * avg_temp_k)
                
                current_n = sum(total_moles.values())
                
                if current_n > target_n:
                    # 气体过多 -> 计算逸出比例
                    ratio = target_n / current_n if current_n > 0 else 1.0
                    
                    # 更新总摩尔数 (模拟逸散)
                    for g in total_moles:
                        total_moles[g] *= ratio
                    
                    # [可选] 记录逸出现象 (Bubbles/Venting)
                    # 只有当逸出量显著时(>5%)才记录，避免噪声
                    if current_n > target_n * 1.05 and vessels:
                         # 随便找一个容器记录现象即可
                         vessels[0].reaction.last_phenomena.append("气体逸出")

            # --- D. 气体再分配 (Redistribution) ---
            # 根据每个容器的剩余体积比例，将气体重新分配回去
            # 这模拟了连通器内气压瞬间平衡的效果
            if total_headspace_L > 1e-6:
                for v in vessels:
                    if not isinstance(v, Vessel): continue
                    
                    v_h_L = max(0.0, v.capacity - v.solvent_volume) / 1000.0
                    ratio = v_h_L / total_headspace_L
                    
                    # 赋予新气体量
                    # 过滤掉极微量的气体以保持数值清洁
                    new_gas = {g: n_tot * ratio for g, n_tot in total_moles.items() if n_tot * ratio > 1e-9}
                    v.gas_contents = new_gas # 触发 setter 同步到 storage

    # [新增] 辅助函数：解析化学式
    def _resolve_chem_name(self, name: str) -> str:
        """
        [v2.0 Robust Resolve] 增强型名称解析
        解决 LLM 输出英文全称 (potassium_permanganate) 导致系统不识别的问题。
        """
        if not name: return "H2O"
        
        # 0. 预处理：去空格，转小写
        clean_name = name.strip()
        # === [CRITICAL FIX] 强制去除方括号 ===
        # 将 [Ag(NH3)2]OH 转换为 Ag(NH3)2OH 以匹配数据库和验证规则
        clean_name = clean_name.replace("[", "").replace("]", "")
        # ===================================
        lower_name = clean_name.lower().replace(" ", "_")

        # 1. 优先查配置表 (reagent_map)
        # 尝试原始名
        if clean_name in self.reagent_map: 
            return self.reagent_map[clean_name]
        # 尝试小写名
        if lower_name in self.reagent_map:
            return self.reagent_map[lower_name]

        # 2. [核心修复] 内置硬编码别名表 (兜底)
        # 防止外部 YAML 漏配导致流程卡死
        fallback_map = {
            "potassium_permanganate": "KMnO4",
            "permanganate": "KMnO4",
            "manganese_dioxide": "MnO2",
            "water": "H2O",
            "oxygen": "O2",
            "hydrogen": "H2",
            "carbon_dioxide": "CO2"
        }
        
        if lower_name in fallback_map:
            return fallback_map[lower_name]

        # 3. 如果看起来像化学式 (大写字母开头，包含数字)，直接返回
        # e.g. "KMnO4"
        if re.match(r"^[A-Z][a-z]?\d*", clean_name):
            return clean_name

        # 4. 实在找不到，返回原名 (并记录警告，便于调试)
        if not self.silent_mode:
            self.logger.warning(f"⚠️ Unresolved chemical name: '{name}'. Using as is.")
            
        return clean_name

    def _init_hardware(self, hardware_config: List[Dict]):
        """
        [v5.0] 使用工厂模式初始化硬件 + 注册拓扑约束
        """
        self._log_info("=== Initializing Hardware (Factory Mode) ===")
        
        # 1. 创建实体对象
        for item in hardware_config:
            # 调用工厂创建对象 (自动处理 Vessel/Instrument 分类及 capacity 等参数)
            obj = EquipmentFactory.create(item)
            
            # 存入对象字典
            self.containers[item['id']] = obj
            
            self._log_info(f" -> Created [{obj.id}]: {obj.describe()} (Type: {type(obj).__name__})")

        # 2. [新增] 注册到 HardwareManager 以启用物理约束检查
        # 我们需要提取原始配置中的 'type' 字段 (如 'tube', 'stand', 'clamp')
        # HardwareManager 将利用这些类型来判断“谁能连谁” (Compatibility Matrix)
        hw_types = {item['id']: item.get('type', 'unknown') for item in hardware_config}
        
        # 初始化拓扑图节点，并注入类型属性
        self.hw_manager.add_hardware(list(self.containers.keys()), hw_types)

        self._log_info("=== Hardware Initialization Complete ===\n")

        # [新增] 收集类型信息并传给 HardwareManager
        hw_types = {item['id']: item.get('type', 'unknown') for item in hardware_config}
        self.hw_manager.add_hardware(list(self.containers.keys()), hw_types)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """[FIX] 安全转换浮点数，处理 NoneType 报错"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # ==========================================
    # Core: The Dispatcher (分发器)
    # ==========================================
    def execute(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        [重构版] 统一执行入口
        """
        act_type = action.get("action")
        
        # 1. 获取处理器
        handler = self.action_dispatch.get(act_type)
        
        # 预先获取快照 (用于出错时返回状态，或者实现撤销功能)
        current_snapshot = self.get_snapshot()

        if not handler:
            return f"系统错误：不支持的动作指令 '{act_type}'", current_snapshot

        try:
            # 2. 执行具体逻辑 (handler 只返回观察文本)
            # 我们约定：如果逻辑校验不通过，handler 抛出 ValueError
            obs_text = handler(action)
            
            # 3. 统一后处理 (日志与快照)
            # 如果没有返回文本，说明动作静默成功
            if not obs_text: 
                obs_text = f"动作 {act_type} 已执行，无明显现象。"

            # 生成最新的快照
            final_snapshot = self.get_snapshot()
            
            # [可选] 可以在这里统一记录结构化日志
            # log_turn_event("Physics", "Action_Success", {"action": action, "obs": obs_text})
            
            return obs_text, final_snapshot

        except ValueError as ve:
            # 逻辑错误 (如：容器不存在、类型不对、参数错误) -> 返回给 Agent 的提示
            return f"操作失败: {str(ve)}", current_snapshot
            
        except Exception as e:
            # 系统崩溃 (代码 Bug) -> 记录堆栈并安全返回
            self._log_error(f"Engine Crash during {act_type}: {e}", exc_info=True)
            return f"模拟引擎内部错误: {str(e)}", current_snapshot
        

    def _diagnose_critical_failures(self) -> List[CausalChain]:
        """
        [v2.0 Causal Diagnosis] 
        基于当前状态 + 动力学黑匣子，生成带因果链的事故报告
        """
        accidents = []

        for vid, obj in self.containers.items():
            if not isinstance(obj, Vessel): continue
            
            # =========================================
            # Case 1: 压力炸裂 (Explosion)
            # =========================================
            if obj.pressure > 2.5: # 阈值
                chain = CausalChain(
                    event_type="EXPLOSION",
                    target_id=vid,
                    critical_value=f"P={obj.pressure:.1f}atm"
                )
                
                # --- Root Cause 1: 物理状态 (必须密封才会炸) ---
                if obj.is_sealed:
                    chain.root_causes.append(CausalFactor(
                        "PHYSICAL_STATE", "容器处于密封状态 (Sealed)", 1.0
                    ))
                
                # --- Root Cause 2: 动力学来源 (气体是从哪来的?) ---
                # A. 化学反应生成
                if obj.dynamics.gas_delta_moles > 1e-6:
                    reactions_str = ", ".join(obj.dynamics.active_reactions)
                    chain.root_causes.append(CausalFactor(
                        "REACTION", 
                        f"内部发生生成气体的反应: [{reactions_str}]", 
                        0.9
                    ))
                
                # B. 物理加热膨胀 (PV=nRT, T升高P升高)
                if obj.dynamics.temp_delta_external > 0.1 or obj.temperature > 100:
                    chain.root_causes.append(CausalFactor(
                        "THERMAL", 
                        f"持续外部加热导致气体膨胀 (当前T={obj.temperature:.0f}°C)", 
                        0.6
                    ))
                
                accidents.append(chain)

            # =========================================
            # Case 1.5: 密封空烧红线 (Teacher's Safety Rule)
            # =========================================
            if obj.is_sealed and obj.dynamics.is_heating and obj.storage.total_volume < 1e-9:
                chain = CausalChain(
                    event_type="EXPLOSION_RISK",
                    target_id=vid,
                    critical_value=f"Temp={obj.temperature:.1f}C"
                )
                chain.root_causes.append(CausalFactor("PHYSICAL_STATE", "密封的空容器被持续加热", 1.0))
                chain.root_causes.append(CausalFactor("THERMAL", "气体剧烈膨胀存在极高炸裂风险", 0.9))
                accidents.append(chain)

            # =========================================
            # Case 2: 倒吸 (Back-Suction)
            # =========================================
            if obj.temperature > 80.0 and not obj.dynamics.is_heating:
                is_in_liquid = False
                
                # 1. 找到所有插在当前试管上的流体连接件 (如导管 u)
                in_edges = self.hw_manager.graph.in_edges(vid, data=True)
                for u, _, in_data in in_edges:
                    if in_data.get("type") == "fluid":
                        # 2. 顺藤摸瓜，看这个导管 u 还连着哪里？
                        out_edges = self.hw_manager.graph.out_edges(u, data=True)
                        for _, remote_parent, out_data in out_edges:
                            # 3. 检查导管的另一头是否插在某个容器的液面下
                            if out_data.get("dst_port") in ["liquid_deep", "deep", "bottom"]:
                                is_in_liquid = True
                                break
                    if is_in_liquid: break
                
                if is_in_liquid:
                    chain = CausalChain(
                        event_type="BACK_SUCTION", 
                        target_id=vid, 
                        critical_value=f"P={obj.pressure:.2f}atm"
                    )
                    chain.root_causes.append(CausalFactor("THERMAL", "热源移除，内部温度骤降", 1.0))
                    chain.root_causes.append(CausalFactor("PHYSICAL", "内部气体收缩产生负压", 0.8))
                    chain.root_causes.append(CausalFactor("OPERATION", "导管口仍处于液面以下，未及时撤出", 1.0))
                    
                    accidents.append(chain)

        return accidents

    def _extract_phenomena_signature(self, snapshot: Dict) -> Set[str]:
        """
        [Helper] 提取当前物理状态的'指纹'，用于判断是否发生了值得中断的变化。
        关注：颜色、气泡、沉淀、危险预警。忽略微小的温度/体积变化。
        """
        sig = set()
        containers = snapshot.get("hardware", {})
        
        # 1. 提取危险状态
        if snapshot.get("status") == "CRASHED":
            sig.add("CRASHED")

        # 2. 提取视觉现象
        for vid, data in containers.items():
            # 颜色与状态
            if "color_desc" in data:
                sig.add(f"{vid}|color|{data['color_desc']}")
            
            # 动态现象 (气泡/沉淀生成)
            # 注意：last_phenomena 是 ReactionManager 在该微步产生的瞬时现象
            if "last_phenomena" in data:
                for p in data["last_phenomena"]:
                    sig.add(f"{vid}|phenomena|{p}")
            
            # 极端温度标签 (模拟学生看到冒烟/沸腾)
            temp = data.get("temperature", 25.0)
            if temp > 95.0: sig.add(f"{vid}|boiling")

        return sig

    def execute_batch(self, actions: List[Dict[str, Any]], duration: float = 10.0) -> Tuple[str, Dict[str, Any]]:
        """
        [v7.0 Event-Driven Execution] 事件驱动执行
        解决'失去控制感'：如果在执行动作的过程中发生了显著变化（或危险），立即中断返回。
        """
        total_logs = []
        
        # 1. 确定本次执行的最大时长
        # 如果动作里自带 duration (由 Translator 解析)，则以此为准
        duration = duration
        if actions and "duration" in actions[0]:
            duration = float(actions[0]["duration"])

        # 2. 初始化微步参数
        dt = 1.0  # 时间粒度 1秒
        elapsed = 0.0
        
        # 记录初始状态指纹
        initial_snap = self.get_snapshot()
        initial_sig = self._extract_phenomena_signature(initial_snap)
        
        # 3. 执行物理动作 (Instant Actions)
        # 例如 Add, Pour，这些通常认为是瞬间完成或由时长控制
        # 这里我们先执行动作的“开始”，然后在循环中模拟其“过程”（如持续加热）
        step_action_logs = []
        for act in actions:
            log, _ = self.execute(act) # 调用底层原子操作
            if log: step_action_logs.append(log)
        
        if step_action_logs:
            total_logs.append(f"执行操作: {'; '.join(step_action_logs)}")

        # 4. 物理演化循环 (The Loop)
        interrupted = False
        
        while elapsed < duration:
            # A. 物理推演 (Reaction + Physics)
            # 这里调用原本的微步逻辑
            # 注意：这里我们传入一个空的 actions 列表给底层，因为动作已经在上面触发了，这里只跑时间
            # 或者你可以把 _step_simulate 逻辑抽取出来
            self._run_micro_step(dt) # 假设你封装了单步物理逻辑
            
            elapsed += dt
            
            # B. [核心] 变化检测与中断
            current_snap = self.get_snapshot()
            current_sig = self._extract_phenomena_signature(current_snap)
            
            # 检查熔断 (Crash)
            failures = self._check_critical_limits()
            if failures:
                total_logs.append(f"\n🚨 在第 {elapsed}秒时发生事故: {' '.join(failures)}")
                current_snap["status"] = "CRASHED"
                return "\n".join(total_logs), current_snap

            # 检查显著现象变化 (Observation Interrupt)
            # 如果现象集合变了 (例如：无色 -> 变红，或者 没气泡 -> 有气泡)
            if current_sig != initial_sig:
                # 计算差异
                new_phenomena = current_sig - initial_sig
                # 过滤掉一些不重要的 (比如温度微变导致的)
                readable_changes = [p.split('|')[-1] for p in new_phenomena if 'color' in p or 'phenomena' in p]
                
                if readable_changes:
                    total_logs.append(f"\n(在第 {elapsed}秒时，观测到新现象: {', '.join(readable_changes)}，操作自动暂停)")
                    interrupted = True
                    break # <--- 立即把控制权还给学生！
        
        if not interrupted:
            total_logs.append(f"(持续了 {duration}秒，无更多明显变化)")

        return "\n".join(total_logs), self.get_snapshot()

    # 在 ChemSimEngine 类中添加:
    def _check_critical_limits(self) -> List[str]:
        """
        [兼容性包装器] 检查是否有严重故障，返回故障描述列表。
        """
        chains = self._diagnose_critical_failures()
        # 将 CausalChain 对象转换为自然语言描述字符串
        return [c.to_natural_language() for c in chains]

    def simulate_until_event(self, max_duration: float = 60.0) -> Tuple[str, Dict]:
        """
        [v7.1 Time Warp] 智能跳过 (用于 OBSERVATION_WAIT)
        全速快进，直到有事发生。
        """
        dt = 2.0 # 快进时步长可以稍大
        elapsed = 0.0
        initial_snap = self.get_snapshot()
        initial_sig = self._extract_phenomena_signature(initial_snap)
        
        logs = []
        
        while elapsed < max_duration:
            self._run_micro_step(dt)
            elapsed += dt
            
            curr_snap = self.get_snapshot()
            
            # 检查事故
            if self._check_critical_limits():
                 curr_snap["status"] = "CRASHED"
                 return f"等待 {elapsed}秒后发生事故！", curr_snap
                 
            # 检查现象
            curr_sig = self._extract_phenomena_signature(curr_snap)
            if curr_sig != initial_sig:
                new_p = [p.split('|')[-1] for p in (curr_sig - initial_sig)]
                return f"等待 {elapsed}秒后，出现了: {', '.join(new_p)}", curr_snap
                
        return f"等待了 {max_duration}秒，未观测到新变化。", self.get_snapshot()
    
    def _run_micro_step(self, dt: float):
        """
        [v7.4 Fixed] 单步物理演化
        """
        
        # =========================================
        # Phase 1: 个体内部演化 (Intrinsic Evolution)
        # =========================================
        for vid, obj in self.containers.items():
            
            # --- A. 容器逻辑 ---
            if isinstance(obj, Vessel):
                # 1. 基础化学与物理 (这里不再清除状态)
                ReactionManager.step_simulate(obj, dt)

                # 1. 基础散热逻辑（始终存在）
                # 即使正在加热，散热也在发生，只是加热功率更大而已
                cooling_effect = (25.0 - obj.temperature) * 0.02 * dt # 0.02 为散热系数
                obj.temperature += cooling_effect

                # 2. 处理外部主动加热逻辑
                # [Fix] 现在 is_heating 标记会持续存在，直到被 Cool 动作取消
                if getattr(obj.dynamics, 'is_heating', False):
                    mode = getattr(obj.dynamics, 'heating_mode', 'fire')
                    
                    if mode == "body_temp":
                        # === 手温微热模式 ===
                        target_t = getattr(obj.dynamics, 'target_temp', 37.0)
                        if obj.temperature < target_t:
                            k_hand = 0.15 
                            delta = (target_t - obj.temperature) * k_hand * dt
                            obj.temperature += delta
                    
                    elif mode == "fire":
                        # === [修改] 强力火焰模式 ===
                        # 原代码: heating_rate = 5.0 (10秒只能升50度，不够)
                        # 新代码: heating_rate = 25.0 (10秒升250度，保证反应发生)
                        heating_rate = 25.0 
                        obj.temperature += heating_rate * dt

            # --- B. 加热器逻辑 ---
            elif isinstance(obj, Heater):
                obj.update_self_temp(dt)

        # =========================================
        # Phase 2: 拓扑热传导
        # =========================================
        edges = self.hw_manager.graph.edges(data=True)
        for child_id, parent_id, data in edges:
            child = self.containers.get(child_id)
            parent = self.containers.get(parent_id)
            if isinstance(parent, Heater) and isinstance(child, Vessel):
                self._apply_heat_transfer(parent, child, dt)

        # =========================================
        # Phase 3: 全局气体平衡
        # =========================================
        self._simulate_gas_diffusion(dt)
        

        # =========================================
        # Phase 5: [New] 帧末清理 (End of Frame Reset)
        # =========================================
        # 在这里清理“本帧增量”，为下一帧做准备
        for obj in self.containers.values():
            if isinstance(obj, Vessel):
                obj.clear_dynamics() # 调用修改后的只清理 delta 的方法

    def _apply_heat_transfer(self, source: Heater, target: Vessel, dt: float):
        """
        [Helper] 热传导计算模型
        Q = k * (T_source - T_target) * dt
        """
        if source.temperature <= target.temperature:
            return # 热力学第二定律，热量不回流 (简化)

        # 温差
        delta_temp = source.temperature - target.temperature
        
        # 传导系数 (简化值，模拟接触热阻)
        # 酒精灯火焰接触效果好，电热板稍慢
        CONDUCTIVITY = 0.05 
        
        # 计算传入的热量 (Joules，这里简化为直接影响温度的因子)
        # 实际上应该用 Q = ... 然后 target.apply_energy(Q)
        # 但这里为了数值稳定性，直接做温度逼近
        
        heat_rate = delta_temp * CONDUCTIVITY * dt
        
        # 限制单步最大温升，防止震荡
        actual_rise = min(heat_rate, delta_temp * 0.5)
        
        target.temperature += actual_rise
        # [NEW] 记录物理加热因果
        if actual_rise > 0:
            target.dynamics.is_heating = True
            target.dynamics.temp_delta_external += actual_rise
        
        # (可选) 记录“正在加热”状态，用于现象描述
        # target.reaction.log_reaction("Heating", "受热中")

    # ==========================================
    # Helpers (辅助函数)
    # ==========================================
    def _get_object_or_raise(self, obj_id: str, expected_type=None) -> LabObject:
        """通用获取对象方法，找不到或类型不对直接抛异常"""
        if not obj_id:
            raise ValueError("指令缺少目标 ID (vessel/object)。")
            
        obj = self.containers.get(obj_id)
        if not obj:
            raise ValueError(f"实验台上找不到器材 '{obj_id}'。")
            
        if expected_type and not isinstance(obj, expected_type):
            raise ValueError(f"'{obj_id}' 不是 {expected_type.__name__}，无法执行此操作。")
        return obj

    # ==========================================
    # Handlers (独立处理器)
    # ==========================================

    def _handle_wait(self, action: Dict) -> str:
        duration = float(action.get("duration", 10.0))
        return f"保持静止观察 {duration}秒..."

    # --- Group 1: Reagent Ops ---
    
    def _handle_add(self, action: Dict) -> str:
        """
        [修改版] 执行加料，包含物理误差
        """
        vessel_id = action.get("vessel")
        target = self._get_object_or_raise(vessel_id, expected_type=Vessel)

        # === [优化逻辑 START] ===
        auto_action_log = ""
        was_sealed = target.is_sealed  # <--- 1. 记住之前的状态

        # 临时解除密封，防止物理引擎在加料瞬间计算压力炸裂
        if was_sealed:
            target.is_sealed = False
            auto_action_log = "(系统自动操作了塞子以便加料)"
        # === [优化逻辑 END] ===
        
        reagent = action.get("reagent")
        if not reagent: raise ValueError("Add 动作必须指定试剂名称。")
        
        real_formula = self._resolve_chem_name(reagent)
        
        # 解析并注入噪声
        raw_mass, raw_vol = action.get("mass"), action.get("volume")
        val, unit = 5.0, 'ml' 
        
        intended_val = 0.0 # 记录意图值，用于日志对比
        
        if raw_mass:
            val, unit = self._safe_float_parse(raw_mass)
            if not unit: unit = 'g'
            intended_val = val
            # [Noise Injection]
            val = self._apply_clumsiness_noise(val)
            
        elif raw_vol:
            val, unit = self._safe_float_parse(raw_vol)
            if not unit: unit = 'ml'
            intended_val = val
            # [Noise Injection]
            val = self._apply_clumsiness_noise(val)
            
        temp, _ = self._safe_float_parse(action.get("temp"), default=25.0)
        
        # 1. 执行物理添加
        target.add_chemical(real_formula, val, unit, fluid_temp=temp)

        # === [核心修复] ===
        # 2. 如果之前是密封的，并且没有显式破坏连接，操作完后自动塞回去
        if was_sealed:
            target.is_sealed = True
            auto_action_log += " -> (操作完成后已自动塞回)"
        
        # 2. 返回描述
        # 细节：如果误差很大，可以在描述中暗示（例如“不小心倒多了”）
        # 但为了让 Teacher Agent 自己去 Observation 中发现，这里可以只陈述事实
        return f"向 {vessel_id} 加入了 {val:.1f}{unit} {reagent} (意图: {intended_val}{unit})。{auto_action_log}"

    def _handle_transfer(self, action: Dict) -> str:
        """
        [修改版 V2.1] 执行转移：只包含挂壁残留逻辑，不包含容量同步修复
        """
        src_id = action.get("from_vessel")
        dst_id = action.get("to_vessel")
        
        # 获取对象
        src = self._get_object_or_raise(src_id, expected_type=Vessel)
        dst = self._get_object_or_raise(dst_id, expected_type=Vessel)
        
        # 解析体积
        raw_vol = action.get("volume")
        vol, _ = self._safe_float_parse(raw_vol)
        
        # 1. 自动推断体积 (默认倒全部)
        if vol is None or vol <= 1e-6:
             vol = src.solvent_volume if src.solvent_volume > 0 else 0.0

        # [Noise Injection] 手抖噪声
        # 只有当不是意图倒光时才加噪声
        if vol < (src.solvent_volume - 0.1): 
            vol = self._apply_clumsiness_noise(vol)

        obs_parts = []
        
        if src.solvent_volume > 1e-6:
            # ====================================================
            # [核心修复] 引入挂壁残留 (Residue Logic)
            # ====================================================
            DEAD_VOLUME = 0.1  # 设定死体积为 0.1 ml
            
            # 计算最大物理可转移量
            # 只有当液体总量大于 0.5ml 时，才强制保留死体积
            # (如果只有一滴水，是可以甩干的，所以允许倒光)
            if src.solvent_volume > 0.5:
                max_transfer = src.solvent_volume - DEAD_VOLUME
            else:
                max_transfer = src.solvent_volume 
                
            # 物理截断检查
            if vol >= max_transfer:
                # 触发残留机制：用户想倒光，但物理上保留一点
                vol = max_transfer
                
                if src.solvent_volume > 0.5:
                    obs_parts.append(f"(受限于挂壁残留，{src_id} 只能倒出 {vol:.1f}ml，剩余 {DEAD_VOLUME}ml)")
                else:
                    obs_parts.append(f"({src_id} 中的液体被全部倒出)")
            else:
                # 正常倒出（量很少，或者只倒一部分）
                msg = f"将 {vol:.1f}ml 液体从 {src_id} 移入 {dst_id}。"
                obs_parts.append(msg)
            # ====================================================
                
        else:
            obs_parts.append(f"试图从 {src_id} 转移，但它是空的。")
            vol = 0.0
            
        # 2. 执行物理转移
        if vol > 0:
            src.transfer_to(dst, vol)
        
        return " ".join(obs_parts)
    
    def _handle_stir(self, action: Dict) -> str:
        """
        [修改版] 搅拌操作
        """
        vessel_id = action.get("vessel")
        target = self._get_object_or_raise(vessel_id, expected_type=Vessel)
        
        # 在动力学模型中，搅拌可以临时提高反应速率
        # 这里我们可以给对象加一个临时标记，ReactionManager 在计算时可以读取
        # target.is_being_stirred = True (需要在 Vessel 类里加这个属性，并在 step 后重置)
        # 简单起见，这里只做动作描述
        
        # [关键修改] 删除 ReactionManager.solve()
        return f"搅拌了 {vessel_id}。"

    def _handle_refill(self, action: Dict) -> str:
        vessel_id = action.get("vessel")
        reagent = action.get("reagent")
        target = self._get_object_or_raise(vessel_id, expected_type=Vessel)
        
        if not reagent: 
            raise ValueError("Refill 需指定试剂。")
        
        real_formula = self._resolve_chem_name(reagent)
        
        # === 此时只需这一行，就会自动触发你写的 @contents.setter ===
        # Setter 内部会自动清空 liquid, solid, gas 和 solvent_volume
        target.contents = {} 
        
        # 确定填充容量
        cap = target.capacity if target.capacity < 1000 else 50.0
        
        # 重新添加试剂
        target.add_chemical(real_formula, cap, 'ml')
        
        return f"已将 {vessel_id} 重新装满 {reagent} ({cap}ml)。"

    def _handle_wash(self, action: Dict) -> str:
        vessel_id = action.get("vessel")
        target = self._get_object_or_raise(vessel_id) # 获取目标物体
        
        if isinstance(target, Vessel):
            # 1. 使用统一的 Setter 清空内容物
            # 这一行会自动清空 liquid/solid/gas_contents 以及 solvent_volume
            target.contents = {}
            
            # 进阶建议：如果你想模拟真实实验中“洗过的瓶子是湿的”
            # 可以在清空后添加极少量的水渍残留
            # target.add_chemical("H2O", 0.1, 'ml') 
            
        # 2. 处理清洗特有的状态重置
        # 无论是否是 Vessel，清洗通常都会重置污染状态和温度
        target.is_dirty = False
        target.temperature = 25.0
        
        return f"已将 {vessel_id} 清洗干净。"

    # --- Group 2: Environment Ops ---

    def _handle_heat(self, action: Dict) -> str:
        vessel_id = action.get("vessel")
        target = self._get_object_or_raise(vessel_id)
        
        # 1. 解析目标温度
        raw_temp = action.get("temperature") or action.get("target_temperature")
        target_temp_val = None
        if raw_temp:
            target_temp_val, _ = self._safe_float_parse(raw_temp)

        # === [核心逻辑修改] 自动关联加热器 ===
        # 寻找场景中可用的加热器 (Heater 实例)
        available_heaters = [obj for obj in self.containers.values() if isinstance(obj, Heater)]
        active_heater = None
        
        if available_heaters:
            # 默认使用第一个找到的加热器 (如 burner_1)
            active_heater = available_heaters[0]
            # 物理操作：点燃加热器
            active_heater.turn_on()
            if target_temp_val: active_heater.target_temp = target_temp_val

        # 2. 处理被加热物体
        if isinstance(target, Heater):
            target.turn_on()
            if target_temp_val: target.target_temp = target_temp_val
            return f"已开启 {vessel_id}。"
            
        elif isinstance(target, Vessel):
            # Case A: 手温微热 (不需要加热器)
            if target_temp_val and 25.0 < target_temp_val < 45.0:
                target.dynamics.is_heating = True
                target.dynamics.heating_mode = "body_temp"
                target.dynamics.target_temp = target_temp_val
                return f"用手紧握 {vessel_id}，利用体温对其微热..."
            
            # Case B: 高温加热 (必须有加热器)
            else:
                target.dynamics.is_heating = True
                target.dynamics.heating_mode = "fire"
                
                # 生成描述：强调加热器的存在
                heater_desc = f"点燃了 {active_heater.name} ({active_heater.id}) 并" if active_heater else "使用外部热源"
                return f"{heater_desc} 开始对 {vessel_id} 进行强力加热！(注意：热源已开启)"
    
    # 辅助函数：解析浮点 (补充)
    def _safe_float_parse(self, value, default=0.0):
        # 复用之前的 parse_quantity 逻辑
        return parse_quantity(value, default)

    def _handle_cool(self, action: Dict) -> str:
        # 支持前端传 vessel 或 object 作为目标
        vessel_id = action.get("vessel") or action.get("object")
        target = self._get_object_or_raise(vessel_id)
        
        # === 情况 1：明确针对加热器操作 (例如：熄灭酒精灯) ===
        if isinstance(target, Heater):
            if not target.is_on:
                return f"{vessel_id} 已经是熄灭状态。"
            
            target.turn_off()
            
            # 联动：既然热源熄灭了，把场景中所有正依赖"fire"模式加热的容器状态也关掉
            for obj in self.containers.values():
                if isinstance(obj, Vessel) and getattr(obj.dynamics, 'is_heating', False):
                    obj.dynamics.is_heating = False
                    
            return f"已熄灭 {vessel_id}。相关的受热容器开始自然冷却。"

        # === 情况 2：针对容器操作 (例如：停止加热试管 / 将试管移开热源) ===
        elif isinstance(target, Vessel):
            was_heating = getattr(target.dynamics, 'is_heating', False)
            
            if hasattr(target, 'dynamics'):
                target.dynamics.is_heating = False
                
            heater_log = ""
            # 为了适配当前的单线实验逻辑，如果我们移开了试管，我们顺便把酒精灯也灭了
            # 但这里我们只关闭处于开启状态的加热器，并明确记录
            active_heaters = [obj for obj in self.containers.values() if isinstance(obj, Heater) and obj.is_on]
            for heater in active_heaters:
                heater.turn_off()
                heater_log += f"并顺便熄灭了 {heater.name}。"

            if was_heating:
                # 注意：去掉了瞬间 -15 度的硬编码，温度将由 _run_micro_step 自然冷却
                return f"{vessel_id} 已撤去热源开始自然冷却。{heater_log} 当前温度 {target.temperature:.1f}°C。"
            else:
                return f"{vessel_id} 当前并受热。当前温度 {target.temperature:.1f}°C。"
                
        else:
             raise ValueError(f"无法对 {type(target).__name__} 执行冷却操作。")

    # --- Group 3: Topology Ops ---

    def _determine_sealing_status(self, child_id: str, child_obj: LabObject, parent_id: str = None) -> Tuple[bool, bool, bool]:
        """
        [Fix 1.1] 判定容器是否被密封。
        修复：增加了对父容器类型的检查，强制敞口容器（如水槽）永远不能被密封。
        返回: (should_seal, should_cover, should_vent)
        """
        # === [新增] 强制敞口检查 ===
        if parent_id:
            parent_type = self.hw_manager._get_type(parent_id).lower()
            # 定义绝对不能被塞子密封的容器关键词
            # "trough"(水槽), "basin"(水盆), "beaker"(烧杯) 通常口径太大，塞子塞不住
            OPEN_CONTAINERS = ["trough", "basin", "beaker", "bath"]
            if any(k in parent_type for k in OPEN_CONTAINERS):
                return False, False, False
        # ==========================

        # 1. 尝试从 Manifest 获取配置 (推荐)
        obj_type = self.hw_manager._get_type(child_id)
        cfg = self.hw_manager.get_equipment_config(obj_type)
        
        # 读取 sealing_mode
        mode = cfg.get("sealing_mode", "unknown")
        
        if mode == "hermetic":
            return True, False, False
        elif mode == "vented":
            return True, False, True  # 密封但通气
        elif mode == "loose":
            return False, True, False

        # 2. 关键词兜底 (Fallback)
        child_str = (child_id + " " + getattr(child_obj, "name", "")).lower()
        
        sealer_keywords = ["stopper", "cork", "plug", "rubber_stop", "seal_cap"]
        vent_keywords = ["tube", "hole", "with_hole", "tubing"] 

        is_stopper = any(k in child_str for k in sealer_keywords)
        has_vent = any(k in child_str for k in vent_keywords)

        if is_stopper:
            if has_vent:
                return True, False, True 
            else:
                return True, False, False
            
        if any(k in child_str for k in ["plate", "cover", "lid"]):
            return False, True, False
            
        return False, False, False

    def _handle_attach(self, action: Dict) -> str:
        """
        [Fix 1] 纯机械连接处理器
        修正点：移除密封判定逻辑。Attach (如夹持) 不会导致容器密封。
        """
        child_id = action.get("vessel") or action.get("object")
        parent_id = action.get("support") or action.get("target")
        
        child = self._get_object_or_raise(child_id)
        parent = self._get_object_or_raise(parent_id)
        
        # 1. 智能幂等检查
        if self.hw_manager.graph.has_edge(child_id, parent_id):
            existing_data = self.hw_manager.graph.get_edge_data(child_id, parent_id)
            if existing_data.get("type") == "mechanical":
                return f"{child_id} 已经在 {parent_id} 上了。"
            else:
                # 修正类型
                self.hw_manager.graph.edges[child_id, parent_id]["type"] = "mechanical"
                return f"已将 {child_id} 与 {parent_id} 的连接方式调整为机械固定。"

        # 2. 执行物理连接 (强制 mechanical)
        conn_type = action.get("type", "mechanical")
        success, msg = self.hw_manager.attach(child_id, parent_id, conn_type)
        if not success: raise ValueError(msg)

        # [修正] 这里不再执行 _determine_sealing_status。
        # 夹持动作绝不应该密封容器。密封行为全权移交给 _handle_insert。
        
        return msg

    def _infer_insertion_position(self, child_id: str, parent_id: str, specified_pos: str) -> str:
        """
        [New Helper] 智能推断插入深度
        如果没有显式指定位置，根据化学常识自动判断。
        """
        # 1. 如果输入显式指定了，以输入为准
        if specified_pos: 
            return specified_pos.lower()

        # 获取对象类型/名称 (转小写)
        p_type = self.hw_manager._get_type(parent_id).lower()
        c_type = self.hw_manager._get_type(child_id).lower()

        # === [新增] 强物理规则纠错 ===
        # 规则：任何管子插入水槽，物理上只能是伸入水中或悬空，绝不可能是堵住瓶口
        if "trough" in p_type or "basin" in p_type:
            # 无论用户说什么，强制修正为液面下（为了排水法）
            return "liquid_deep"
        # ==========================
        
        # === 常识规则库 ===

        # --- 新增规则：如果是集气瓶(bottle)，且插入的是管子类(tube)，默认是伸入(deep) ---
        # 避免默认堵死瓶口
        if "bottle" in p_type and ("tube" in c_type):
            return "deep"  # 或者 "inside"
        
        # 规则 A: 水槽/烧杯集气 -> 默认伸入液面下
        # 场景：排水集气法，导管必须进水
        if "trough" in p_type or "basin" in p_type:
            return "liquid_deep"
            
        # 规则 B: 洗气瓶的长管/进气管 -> 默认伸入液面下
        # 场景：区分长短管
        if "long" in c_type or "inlet" in c_type or "dip" in c_type:
            return "liquid_deep"
            
        # 规则 C: 所有的“塞子”(Stopper) -> 默认在瓶口
        if "stopper" in c_type or "plug" in c_type:
            return "mouth"

        # 默认兜底：瓶口
        return "mouth"

    def _handle_insert(self, action: Dict) -> str:
        """
        [Fix 2.1] 插入处理器 (带智能位置推断)
        """
        child_id = action.get("tool") or action.get("object")
        parent_id = action.get("vessel") or action.get("target")
        
        child = self._get_object_or_raise(child_id)
        parent = self._get_object_or_raise(parent_id)

        # 1. 幂等检查
        if self.hw_manager.graph.has_edge(child_id, parent_id):
             return f"{child_id} 已经插入 {parent_id} 中。"

        # === [重构] 连接类型判定：保守策略 (Mechanical First) ===
        conn_type = "mechanical"  # <--- 默认改为机械连接，不再默认流体！
        
        child_type = self.hw_manager._get_type(child_id)
        child_cfg = self.hw_manager.get_equipment_config(child_type)
        sealing_mode = child_cfg.get("sealing_mode", "unknown")
        
        # 只有满足特定“流体特征”时，才升级为 Fluid
        # 1. 显式标记为 vented (如带孔塞、导管)
        if sealing_mode == "vented":
            conn_type = "fluid"
            
        # 2. 或者，拥有流体标签 (如 funnel, dropper)
        # 防止某些漏斗没写 sealing_mode 但确实是用来流液体的
        elif "tubing" in child_cfg.get("tags", []) or "funnel" in child_cfg.get("tags", []):
            conn_type = "fluid"

        # 3. 关键词兜底 (为了兼容旧配置，可选)
        # 如果名字里带 "tube" 且不是 "test_tube" (容器)，倾向于是导管
        elif "tube" in child_type.lower() and "vessel" not in child_cfg.get("tags", []):
             conn_type = "fluid"
             
        # ========================================================
        
        # === [核心优化] 智能推断插入位置 ===
        raw_pos = action.get("position") # 可能为 None
        final_pos = self._infer_insertion_position(child_id, parent_id, raw_pos)
        
        # 映射到端口
        target_port = "mouth"
        msg_suffix = ""
        if final_pos in ["deep", "bottom", "below_liquid", "liquid_deep"]:
            target_port = "liquid_deep"
            msg_suffix = " (伸入液面下)"
        
        # 3. 执行连接
        c_port = "outlet" if conn_type == "fluid" else "center"
        
        success, msg = self.hw_manager.attach(
            child_id, parent_id, 
            connection_type=conn_type,
            child_port=c_port,      
            parent_port=target_port
        )
        if not success: raise ValueError(msg)

       # === [NEW] 状态结算 ===
        if isinstance(parent, Vessel) and target_port == "mouth":
            should_seal, should_cover, should_vent = self._determine_sealing_status(child_id, child, parent_id)
            
            if should_seal:
                parent.is_sealed = True
                parent.is_vented = should_vent  # 应用通气状态
                
                # 提示语差异化
                type_desc = " (带导管/通气孔)" if should_vent else " (完全密封)"
                msg += f" {parent_id} 已被塞紧{type_desc}。"
                
            elif should_cover:
                parent.is_covered = True
                parent.is_sealed = False
                parent.is_vented = False
                msg += " (容器口已盖住)"

        return f"将 {child_id} 插入 {parent_id} {msg}。"

    def _handle_detach(self, action: Dict) -> str:
        child_id = action.get("object") or action.get("vessel") or action.get("tool")
        parent_id = action.get("support") # 可选
        
        success, msg = self.hw_manager.detach(child_id, parent_id)
        if not success:
            raise ValueError(msg)
        return msg

    # --- Group 4: Measurement/Special ---

    def _handle_measure_temp(self, action: Dict) -> str:
        vessel_id = action.get("vessel")
        target = self._get_object_or_raise(vessel_id)
        return f"【温度读数】{vessel_id}: {target.temperature:.1f} °C"

    def _handle_measure_mass(self, action: Dict) -> str:
        vessel_id = action.get("vessel")
        target = self._get_object_or_raise(vessel_id)
        
        mass = 0.0
        if isinstance(target, Vessel):
             # 简单估算：体积即质量 + 沉淀
             # (未来这里可以结合优化建议2，调用 target.total_mass_g)
             mass = target.solvent_volume # 暂且简化
             
        return f"【质量读数】{vessel_id} (含内容物): {mass:.2f} g"

    def _handle_filter(self, action: Dict) -> str:
        src_id = action.get("from_vessel")
        dst_id = action.get("to_vessel")
        src = self._get_object_or_raise(src_id, expected_type=Vessel)
        dst = self._get_object_or_raise(dst_id, expected_type=Vessel)
        
        # 简单的过滤逻辑：转移液体，保留固体
        vol = src.solvent_volume
        if vol <= 0:
            return f"{src_id} 中没有液体可过滤。"
            
        # 转移
        src.solvent_volume = 0.5 # 滤纸吸附
        dst.solvent_volume += vol
        
        # 转移溶质 (遍历 liquid_contents)
        for chem, moles in src.liquid_contents.items():
            dst.liquid_contents[chem] = dst.liquid_contents.get(chem, 0) + moles
        src.liquid_contents = {}
        
        # 固体留在 src (src.solid_contents 不动)
        
        return f"过滤完成。{dst_id} 获得 {vol:.1f}ml 滤液，固体留在了过滤装置中。"

    def _handle_collect_gas(self, action: Dict) -> str:
        # 这是一个简化版，真实的集气需要复杂的拓扑检查
        src_id = action.get("source_vessel")
        col_id = action.get("collector")
        return f"开始收集 {src_id} 产生的气体到 {col_id}..."
    

    def _is_system_stable(self) -> bool:
        """[Helper] 快速判断当前环境是否处于'平静'状态"""
        for obj in self.containers.values():
            if isinstance(obj, Vessel):
                # 1. 有反应在进行？(检查上一帧的记录)
                if obj.reaction.active_reactions: return False
                # 2. 温度异常？(高于 40度视为活跃)
                if obj.temperature > 40.0: return False
                # 3. 压强异常？
                if obj.pressure > 1.2: return False
            elif isinstance(obj, Heater):
                # 4. 加热器开着？
                if obj.is_on: return False
        return True

    def _is_action_risky(self, actions: List[Dict]) -> bool:
        """[Helper] 判断动作是否具有潜在风险"""
        RISKY_VERBS = {
            "Heat", "Add", "Transfer", "Pour", "Refill", 
            "Connect", "Attach", "Insert" # 连接可能导致密闭
        }
        for act in actions:
            if act.get("action") in RISKY_VERBS:
                return True
        return False

    def fork_and_predict(self, actions: List[Dict[str, Any]], steps: int = 20) -> Dict[str, Any]:
        """
        [v8.1 Optimized Ghost Simulation]
        集成 Fast-Path 和 Memory Cloning 优化
        """
        
        # =========================================================
        # 0. Fast Path (规则级过滤) - 第一道防线
        # =========================================================
        # 如果动作是安全的(如 Measure, Wait, Look) 且 环境是稳定的 -> 跳过模拟
        if not self._is_action_risky(actions) and self._is_system_stable():
            # self.logger.debug("👻 Ghost Sim: Fast pass (Safe Action + Stable Env)")
            return {
                "status": "NORMAL",
                "safety_alerts": [],
                "causal_chains": [],
                "summary": "预测环境保持平稳。",
                "snapshot": self.get_snapshot() # 直接返回当前快照
            }

        # =========================================================
        # 1. 实例化沙箱 (使用 silent_mode)
        # =========================================================
        try:
            # 这里的 init 开销很小，因为只有空字典初始化
            sandbox = self.get_shadow_engine()
            sandbox.clone_from(self)
            
        except Exception as e:
            self._log_error(f"Sandbox Clone Failed: {e}")
            # [关键修复] 报错时也要返回 snapshot 字段，防止外层 KeyError
            return {
                "status": "ERROR", 
                "safety_alerts": ["模拟环境创建失败"],
                "snapshot": self.get_snapshot() # <--- 兜底返回当前快照
            }

        # =========================================================
        # 2. 执行推演 (Simulation Loop) - 保持原逻辑
        # =========================================================
        crashed = False
        causal_chains = []
        
        try:
            # A. 执行动作
            for action in actions:
                sandbox.execute(action)

            # B. 时间推演
            DT = 1.0 
            for t in range(steps):
                sandbox._run_micro_step(DT)
                chains = sandbox._diagnose_critical_failures()
                if chains:
                    crashed = True
                    # [新增] 记录爆炸发生的具体时间点，便于调试
                    time_to_failure = (t + 1) * DT
                    causal_chains.extend(chains)
                    break 
                
        except Exception as e:
            # [关键修复] 同上，报错时补全 snapshot
            return {
                "status": "ERROR", 
                "safety_alerts": [f"模拟推演出错: {str(e)}"],
                "snapshot": self.get_snapshot() # <--- 兜底
            }

        # =========================================================
        # 3. 构造报告
        # =========================================================
        safety_alerts_str = [c.to_natural_language() for c in causal_chains] if causal_chains else []

        return {
            "status": "CRASHED" if crashed else "NORMAL",
            "phenomena": ["(Ghost Simulation)"], 
            "safety_alerts": safety_alerts_str,
            "causal_chains": causal_chains,
            "summary": describe_snapshot_briefly(sandbox.get_snapshot()),
            "snapshot": sandbox.get_snapshot() 
        }

# ==========================================
# [New Utility] Context Manager
# ==========================================
def prune_history(history: List[Dict], keep_first: int = 2, keep_last: int = 6) -> List[Dict]:
    """
    [记忆剪枝算法]
    策略：保留最初的 N 轮（确立人设和目标） + 最近的 M 轮（短期记忆）。
    中间部分会被压缩为一个系统提示占位符。
    """
    total = len(history)
    if total <= (keep_first + keep_last):
        return history
    
    # 切片
    head = history[:keep_first]
    tail = history[-keep_last:]
    
    # 中间插入占位符，提示 LLM 这里有省略
    gap = [{
        "role": "system", 
        "content": f"（...为了节省注意力，中间忽略了 {total - keep_first - keep_last} 轮对话...）"
    }]
    
    return head + gap + tail   

class MemoryManager:
    """
    [v9.1 Optimization] 语义记忆管理器
    职责：当对话过长时，调用 LLM 将旧历史压缩为摘要，而不是简单截断。
    """
    def __init__(self, client, max_tokens=2000):
        self.client = client
        self.history: List[Dict] = []
        self.running_summary = "（实验刚开始，暂无历史摘要）"
        self.max_turns = 8 # 触发摘要的阈值 (保留最近 8 轮)
        self.logger = logging.getLogger("Memory")

    def add_turn(self, role: str, content: str):
        """添加新对话轮次"""
        self.history.append({"role": role, "content": content})
        
        # 检查是否需要压缩
        if len(self.history) > self.max_turns:
            self._compress_memory()

    def get_context_for_prompt(self) -> str:
        """
        返回用于 Prompt 的 JSON 字符串。
        结构: [System Summary] + [Recent Turns]
        """
        # 构造一个虚拟的 System Message 携带摘要
        context_list = [{
            "role": "system",
            "content": f"【前情提要 (Context Summary)】:\n{self.running_summary}"
        }]
        
        # 加上最近的对话 (原样保留)
        context_list.extend(self.history)
        
        return json.dumps(context_list, ensure_ascii=False)

    def _compress_memory(self):
        """
        [核心逻辑] 调用 LLM 将最早的几轮对话压缩进 summary
        [优化] 使用高密度技术摘要 Prompt，强化对物理数值、异常状态和当前教学锚点的保留。
        """
        # 我们要压缩的是：旧的 Summary + 溢出的那几轮对话
        # 保留最近 `keep_recent` 轮不动
        keep_recent = 4
        to_compress = self.history[:-keep_recent]
        self.history = self.history[-keep_recent:] # 截断内存
        
        # 构造压缩 Prompt
        compress_text = "\n".join([f"{m['role']}: {m['content']}" for m in to_compress])
        
        # === [修改点] 优化后的 Prompt ===
        prompt = f"""
        请将新的交互内容合并到现有的实验记忆中，生成一段**高密度的技术摘要**。
        
        【现有摘要】: 
        {self.running_summary}
        
        【新交互内容 (包含对话与物理引擎日志)】:
        {compress_text}
        
        【摘要生成要求】:
        1. **去噪与提炼**: 忽略学生的口吃、语气词（如“呃”、“那个”），仅提取其**核心意图**和**实际执行的物理动作**。
        2. **精准记录事实**: 
           - 必须保留关键数值（如 "5.0g", "10ml", "500°C"）。
           - 必须记录系统触发的异常（如 "因生成沉淀/新现象导致操作暂停"、"系统拦截了危险操作"）。
        3. **诊断式记录**: 明确指出学生犯了什么具体错误（如 "跳过固定装置步骤直接加药"、"未检查气密性"、"搞错加药顺序"）。
        4. **状态锚定**: 结尾必须明确说明当前老师提出的问题或引导方向（e.g., "当前等待学生回答关于XX的提问"）。
        
        【输出格式示例】:
        "学生试图跳过组装步骤直接加药，向试管中加入 5.0g KMnO4。物理引擎观测到紫黑色晶体出现并暂停操作。老师指出了未固定试管的安全隐患（翻倒风险），并引导学生思考固定装置的方法。当前等待学生回应。"
        
        请直接输出新的纯文本摘要，不要包含 JSON 或 Markdown:
        """
        # ================================
        
        try:
            # 使用较快的小模型 (如 gpt-3.5-turbo 或 qwen-turbo) 进行摘要
            resp = self.client.chat.completions.create(
                model=Config.MODEL_TEACHER, # 或者更小的模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            new_summary = resp.choices[0].message.content.strip()
            self.running_summary = new_summary
            self.logger.info("🧠 Memory Compressed successfully.")
            
        except Exception as e:
            self.logger.error(f"Memory compression failed: {e}")
            # 失败兜底：不更新摘要，只保留最近对话，丢失部分信息但保证运行

# ==========================================
# [新增] 状态谓词库 (State Predicates)
# ==========================================
class StatePredicates:
    """
    原子检查函数库：用于判断单个物体或状态是否符合物理特征。
    """
    
    @staticmethod
    def has_chemical(obj_data: dict, chem_formula: str, min_amount: float = 1e-4) -> bool:
        """
        [v3.0 Existence Check] 极简版检查：
        只要容器里有这个东西（大于痕量），不管 LLM 要求多少量，统统算通过。
        响应用户需求：'不用必须满足某个量'。
        """
        target = chem_formula.strip()
        target_norm = target.replace("[", "").replace("]", "").strip()

        # === [核心修改] 强制覆盖阈值 ===
        # 无论传入的 min_amount 是多少 (e.g. 0.001)，我们都只检查是否存在
        # 1e-9 约等于 0.000000001 mol，几乎只要有残留就算过
        threshold = 1e-9 

        # 1. 统计当前含量
        info = CHEM_DB.substances.get(target, {})
        molar_mass = info.get("molar_mass", 18.0)
        density = info.get("density", 1.0)       

        contents = obj_data.get("contents", {})
        total_moles = 0.0
        
        for c_name, c_moles in contents.items():
            if c_name.replace("[", "").replace("]", "").strip() == target_norm:
                total_moles += c_moles

        if target_norm == "H2O":
            vol = obj_data.get("volume_ml", obj_data.get("solvent_volume", 0.0))
            if vol > 0:
                total_moles = max(total_moles, (vol * density) / molar_mass)

        # 2. 直接判断
        if total_moles > threshold:
            return True
            
        return False

    @staticmethod
    def is_heated(obj_data: dict, min_temp: float = 100.0) -> bool:
        temp = obj_data.get("temperature", 25.0)
        try:
            threshold = float(min_temp)
        except:
            threshold = 100.0
        return temp >= threshold

    # ================= [请补上这一段] =================
    @staticmethod
    def is_covered(obj_data: dict) -> bool:
        """检查容器是否被盖上 (非密封)"""
        return obj_data.get("is_covered", False)
    # =================================================

    @staticmethod
    def is_sealed(obj_data: dict) -> bool:
        """检查容器是否密封"""
        return obj_data.get("is_sealed", False)

    @staticmethod
    def is_type(obj_data: dict, target_type: str) -> bool:
        """检查器材类型 (e.g. 'tube', 'bottle')"""
        # 数据源里的 type 可能是 "Vessel", name 可能是 "test_tube"
        # 我们检查 name 或 原始配置的 type
        obj_name = str(obj_data.get("name", "")).lower()
        obj_raw_type = str(obj_data.get("type", "")).lower()
        target = target_type.lower()
        
        return (target in obj_name) or (target in obj_raw_type)

# ==========================================
# [重构] 目标验证器 (GoalValidator)
# ==========================================
class GoalValidator:
    """
    [v4.1 Anchored Semantic Validator]
    基于结果特征的验证器。
    替代了旧版基于过程 ID (validate_node_completion) 的验证逻辑。
    """
    def __init__(self, logger_name="GoalValidator"):
        self.logger = logging.getLogger(logger_name)
        # 注册谓词函数
        self.predicates = {
            "has_chemical": StatePredicates.has_chemical,
            "is_heated": StatePredicates.is_heated,
            "is_sealed": StatePredicates.is_sealed,
            "is_type": StatePredicates.is_type,
            # === [新增注册] ===
            "is_covered": StatePredicates.is_covered
        }

    # -------------------------------------------------------------------------
    # [优化版] GoalValidator: 支持细粒度错误诊断
    # -------------------------------------------------------------------------
    def validate_state(self, snapshot: dict, constraints: dict, focus_ids: List[str] = None) -> Tuple[bool, List[str]]:
        """
        验证当前物理快照是否满足目标约束。
        [优化] 当验证失败时，寻找“最接近”的物体并报告具体缺失项，而不是罗列所有规则。
        """
        if not constraints:
            return True, ["无特定物理约束，自动通过"]
        if "requirements" not in constraints and "topology_requirements" not in constraints:
             return True, ["验证规则为空，自动通过"]

        containers = snapshot.get("hardware", {})
        topology = snapshot.get("topology", [])
        
        role_assignments = {} 
        failure_reasons = []
        
        # 构建搜索优先级
        all_ids = list(containers.keys())
        priority_pool = [vid for vid in (focus_ids or []) if vid in containers]
        secondary_pool = [vid for vid in all_ids if vid not in priority_pool]
        search_order = priority_pool + secondary_pool

        requirements = constraints.get("requirements", [])
        
        for req in requirements:
            role_name = req["role"]
            criteria = req["criteria"]
            found_candidate = None
            
            # 1. 尝试寻找完美匹配
            for vid in search_order:
                data = containers[vid]
                if self._check_criteria_silent(data, criteria): # 使用静默检查
                    found_candidate = vid
                    break 
            
            if found_candidate:
                role_assignments[role_name] = found_candidate
            else:
                # === [核心优化] 找不到完美匹配时，进行诊断 ===
                specific_error = self._diagnose_failure(containers, search_order, role_name, criteria)
                failure_reasons.append(specific_error)

        # 角色没找齐，直接失败
        if len(role_assignments) < len(requirements):
            return False, failure_reasons

        # 2. 拓扑验证 (保持不变)
        topo_reqs = constraints.get("topology_requirements", [])
        for link in topo_reqs:
            args = link.get('args', [])
            if len(args) < 2: continue

            role_a, role_b = args[0], args[1]
            conn_type = link.get("type", "fluid")
            
            id_a = role_assignments.get(role_a) or self._fallback_find_id(containers, role_a)
            id_b = role_assignments.get(role_b) or self._fallback_find_id(containers, role_b)

            if not id_a:
                failure_reasons.append(f"拓扑检查失败: 未找到角色 '{role_a}'")
                return False, failure_reasons
            if not id_b:
                failure_reasons.append(f"拓扑检查失败: 未找到角色 '{role_b}'")
                return False, failure_reasons

            if not self._check_connection(topology, id_a, id_b, conn_type):
                failure_reasons.append(f"连接未建立: {role_a}({id_a}) 和 {role_b}({id_b}) 之间断开")
                return False, failure_reasons

        return True, ["验证通过"]
    
    def _check_criteria_silent(self, obj_data: dict, criteria: list) -> bool:
        """静默检查，只返回 True/False，不打印日志"""
        ALLOWED_PREDICATES = {"has_chemical", "is_heated", "is_sealed", "is_type", "is_covered"}
        for criterion in criteria:
            func_name = criterion.get("check")
            args = criterion.get("args", [])
            if func_name not in ALLOWED_PREDICATES: continue 
            func = self.predicates.get(func_name)
            if not func: continue
            try:
                if not func(obj_data, *args): return False
            except: return False
        return True

    def _diagnose_failure(self, containers: dict, search_order: list, role_name: str, criteria: list) -> str:
        """
        [新增] 诊断函数：找到得分最高的物体，报告它具体缺了什么。
        """
        best_vid = None
        best_score = -1
        best_missed = []

        # 遍历所有物体打分
        for vid in search_order:
            data = containers[vid]
            score = 0
            missed_details = []
            
            # 检查每一条标准
            for criterion in criteria:
                func_name = criterion.get("check")
                args = criterion.get("args", [])
                func = self.predicates.get(func_name)
                
                passed = False
                if func:
                    try:
                        passed = func(data, *args)
                    except:
                        passed = False
                
                if passed:
                    score += 1
                else:
                    # 记录具体的失败原因，例如 "缺少 NaOH" 或 "温度不足 (当前25度)"
                    desc = f"{func_name}{args}"
                    if func_name == "has_chemical":
                        desc = f"缺少试剂 {args[0]}"
                    elif func_name == "is_heated":
                        desc = f"温度未达标 (需 {args[0]}度)"
                    elif func_name == "is_type":
                        desc = f"器材类型错误 (需 {args[0]})"
                    missed_details.append(desc)
            
            # 更新最佳匹配 (优先选得分高的，得分相同选优先池里的)
            if score > best_score:
                best_score = score
                best_vid = vid
                best_missed = missed_details

        # 生成诊断报告
        if best_vid and best_score > 0:
            # 找到了一个半成品
            return f"进度提示：你的容器 {best_vid} 已满足部分条件，但还【{', '.join(best_missed)}】。"
        else:
            # 完全没找到相关的物体，回退到通用描述
            simple_desc = [f"{c['check']}{c.get('args')}" for c in criteria]
            return f"未找到【{role_name}】。请确保你准备了符合以下条件的器材: {', '.join(simple_desc)}"

    # [Helper] 新增一个兜底查找方法
    def _fallback_find_id(self, containers: dict, role_name: str) -> str:
        # 1. 如果 role_name 直接就是 ID (e.g. "beaker_1")
        if role_name in containers:
            return role_name
        
        # 2. 尝试按名称/类型模糊匹配 (e.g. role="cotton_ball", 找 name="Cotton Ball")
        for vid, data in containers.items():
            name = str(data.get("name", "")).lower()
            rtype = str(data.get("type", "")).lower()
            target = role_name.lower()
            if target in name or target in rtype:
                return vid
        return None

    def _check_criteria(self, obj_data: dict, criteria: list) -> bool:
        """
        检查单个物体是否满足一系列原子条件。
        [修改] 增加详细的 Debug 日志，指出具体是哪一条规则没过。
        """
        ALLOWED_PREDICATES = {"has_chemical", "is_heated", "is_sealed", "is_type", "is_covered"}

        for criterion in criteria:
            func_name = criterion.get("check")
            args = criterion.get("args", [])
            
            if func_name not in ALLOWED_PREDICATES:
                continue 

            func = self.predicates.get(func_name)
            if not func:
                continue
                
            try:
                # 执行检查
                if not func(obj_data, *args):
                    # === [核心修复] 打印具体失败原因 ===
                    # 获取物体当前状态以便对比
                    current_val = "Unknown"
                    if func_name == "has_chemical":
                        chem = args[0]
                        contents = obj_data.get("contents", {})
                        # 同时尝试读取体积
                        vol = obj_data.get("volume_ml", 0) if chem == "H2O" else 0
                        current_val = f"{contents.get(chem, 0.0):.4f}mol (or {vol}ml)"
                    elif func_name == "is_heated":
                        current_val = f"{obj_data.get('temperature')}°C"
                    elif func_name == "is_type":
                        current_val = f"{obj_data.get('type')} / {obj_data.get('name')}"
                    
                    self.logger.warning(
                        f"❌ [Criterion Failed] Object: {obj_data.get('name')} | "
                        f"Check: {func_name}{args} | "
                        f"Current State: {current_val}"
                    )
                    return False
            except Exception as e:
                self.logger.error(f"❌ [Validator Error] {func_name}: {e}")
                return False
                
        return True

    def _check_connection(self, topology: list, id_a: str, id_b: str, conn_type: str) -> bool:
        """
        [修改版 - 宽容模式] 检查连接
        不再严格区分 mechanical vs fluid。
        只要 id_a 和 id_b 在拓扑图中有路径相连，就视为验证通过。
        """
        if not id_a or not id_b: return False
        
        # 构建临时图 (包含所有类型的连接)
        G = nx.Graph() 
        for link in topology:
            # === [核心修改] 移除类型过滤 ===
            # 原代码: if link.get('type') == conn_type:
            # 新逻辑: 无条件添加所有连接。
            # 理由: 无论是插进去(fluid)还是夹住(mechanical)，在“组装装置”这个大目标下，都算“连上了”。
            G.add_edge(link['child'], link['parent'])
        
        if id_a not in G or id_b not in G:
            return False
            
        return nx.has_path(G, id_a, id_b)

    # === [兼容接口] ===
    # 为了支持 Hybrid Generator 中 _generate_grounded_constraints 的调用
    # 我们保留一个辅助方法来获取标准动作序列，但这不再用于验证
    def _get_cumulative_actions(self, sim_engine: Any, target_node: Any) -> List[Dict]:
        """
        [Fix v2.0] 获取非线性动作序列
        修复：不再盲目使用 range() 填充中间步骤，而是严格按照 required_steps 列表执行。
        """
        # 1. 获取当前节点要求的所有步骤 ID
        # 注意：为了生成完整的 Ground Truth，这里的前提是 target_node.required_steps 
        # 必须包含直到当前时刻所有必要的历史步骤（或者在 DAG 构建时已经做过聚合）。
        step_ids = target_node.required_steps
        
        if not step_ids: 
            return []

        # 2. 排序 (Critical)
        # 即使是非线性实验，物理执行的微观顺序通常还是遵循 ID (step_0 -> step_5)
        # 我们按 step_n 中的 n 进行排序
        try:
            sorted_ids = sorted(step_ids, key=lambda x: int(x.split('_')[-1]))
        except Exception:
            # 容错：如果 ID 格式不是 step_n，则按字母序兜底
            sorted_ids = sorted(step_ids)

        # 3. 提取动作
        actions = []
        for sid in sorted_ids:
            if sid in sim_engine.initial_xdl_procedure:
                actions.append(sim_engine.initial_xdl_procedure[sid])
            else:
                # 可选：记录日志，某些步骤可能在 XDL 中被删除了但图里还在
                pass
                
        return actions

class SmartTaskSelector:
    """
    [v2.1 Optimized Task Selector] 
    优化：支持分支逻辑，不再强制随机坍缩。
    """
    def __init__(self):
        self.current_focus_id = None # 当前锁定的任务 ID
        self.logger = logging.getLogger("TaskSelector")

    def select_next_task(self, dag_status: Dict, cognitive_model: Any) -> Union[Any, List[Any]]:
        """
        根据 DAG 状态选择下一步 Focus。
        
        Returns:
            - None: 无任务（实验结束或异常）
            - TaskNode (Single): 明确的、锁定的单一目标
            - List[TaskNode]: 分支路口，返回所有可选任务供 Teacher 引导
        """
        # 1. 获取前沿任务 (Frontier)
        frontier_nodes = dag_status.get('next_tasks', [])
        
        if not frontier_nodes:
            self.current_focus_id = None
            return None

        frontier_ids = [n.id for n in frontier_nodes]

        # 2. 粘性逻辑 (Stickiness / Locking)
        # 如果我们之前已经锁定了一个任务，且它依然在前沿列表中（说明没做完），则保持锁定
        # 这样防止 Teacher 在任务执行中途突然换目标
        if self.current_focus_id:
            if self.current_focus_id in frontier_ids:
                target = next(n for n in frontier_nodes if n.id == self.current_focus_id)
                # self.logger.info(f"🔒 Keeping focus on: {target.desc}")
                return target
            else:
                # 锁定的任务消失了（可能已完成，或依赖条件变化退回去了），解除锁定
                self.logger.info(f"🔓 Focus '{self.current_focus_id}' released.")
                self.current_focus_id = None

        # 3. [核心修改] 分支逻辑 (Branching Strategy)
        # 如果没有锁定，且有多个可选项 -> 不要随机选！返回列表！
        if len(frontier_nodes) > 1:
            self.logger.info(f"🔀 Branching detected: {[n.desc for n in frontier_nodes]}")
            return frontier_nodes # <--- 返回列表
        
        # 4. 单路径逻辑 (Single Path)
        # 只有一条路可选，自动锁定并返回
        single_node = frontier_nodes[0]
        self.current_focus_id = single_node.id
        self.logger.info(f"🎯 Auto-locked new focus: {single_node.desc}")
        return single_node
    
class SocraticDataGenerator:
    """
    [导演类] 负责编排 "双世界" (物理 vs 逻辑) 的冲突，并生成 Socratic 教学数据。
    """
    def __init__(self, client, yaml_path: str, student_profile: dict, output_file: str = "socratic_dataset.jsonl"):
        self.client = client
        self.output_file = output_file
        self.student_profile = student_profile
        self.session_id = str(uuid.uuid4())[:8]
        self.logger = logging.getLogger("DataGen")

        self._init_dataset_file() # 确保文件存在

        # === 1. 初始化引擎与配置 ===
        print(Fore.CYAN + f"🚀 [System] Initializing Session: {self.session_id}...")
        loader = XDLExperimentLoader(self.client)
        
        # [Mod] 接收 5 个返回值
        self.sim, self.oracle, config, self.xdl_content, self.dag_content = loader.load_from_xdl(yaml_path)
        self.exp_config = config

        # =========================================================
        # [User Requirement] 将 XDL 和 DAG 保存到 JSONL 头部
        # =========================================================
        metadata_record = {
            "record_type": "experiment_metadata", # 标记类型，方便过滤
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "experiment_title": config.get("title", "Unknown"),
            "data": {
                "xdl_enriched": self.xdl_content, # 完整的 XDL 文本
                "dag_structure": self.dag_content # 完整的 DAG JSON 对象
            }
        }
        
        # 立即写入文件
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metadata_record, ensure_ascii=False) + "\n")
        print(f"💾 [DataGen] Experiment metadata saved to {self.output_file}")
        # =========================================================

        # 初始化物理引擎 (注入笨拙度)
        clumsiness = student_profile.get("clumsiness", "low")
        self.sim.clumsiness_level = clumsiness
        
        # 初始化状态 (Initial Setup)
        initial_actions = config.get("initial_setup", [])
        if initial_actions:
            self.sim.execute_batch(initial_actions)

        # === 2. 实例化 Agents ===
        self.policy_agent = PedagogicalPolicyAgent(client)
        self.dialogue_manager = DialogueManager(client)
        self.teacher = TeacherAgent(client)
        self.student = UnifiedStudentAgent(client, persona_config=student_profile)
        self.cognitive_model = StudentCognitiveModel(profile_level=student_profile.get("knowledge_level", "novice"))

        # === [NEW] 初始化验证器 ===
        self.validator = GoalValidator() 
        self.current_constraints = None 
        self.last_task_desc = ""
        # [NEW] 初始化记忆管理器
        self.memory = MemoryManager(client)

    def _match_student_intent(self, student_text: str, student_actions: List[Dict], candidates: List[Any]) -> Tuple[Any, str]:
        """
        [Mod] 多模态意图识别器 - 带思考过程版
        返回: (SelectedNode, ThoughtString)
        """
        # 1. 只有一项可选时，直接返回，思考过程为自动
        if len(candidates) == 1:
            return candidates[0], "唯一的路径，自动锁定。"
        
        # 2. 如果什么都没做也没说，默认选第一个
        if not student_text and not student_actions:
            return candidates[0], "无任何输入，默认选择第一项。"

        # 3. 格式化动作为文本
        action_str = "（无物理动作）"
        if student_actions:
            acts = []
            for a in student_actions:
                details = [str(v) for k, v in a.items() if k != 'action']
                acts.append(f"{a.get('action')}({', '.join(details)})")
            action_str = "; ".join(acts)

        # 4. 构造选项
        options_str = "\n".join([f"{i}: {node.desc}" for i, node in enumerate(candidates)])

        # 5. 构造 Prompt (强制 JSON)
        prompt = f"""
        # 实验意图匹配
        学生面临以下分支任务选择：
        {options_str}

        【学生刚才的行为】
        - 语言: "{student_text}"
        - 动作: {action_str}

        请分析学生的意图，并推断他试图执行哪一个任务。
        
        ⚠️ 判决原则：
        1. **动作优先**：如果学生已经开始执行某种特定操作，请优先匹配对应的任务。
        2. **语言辅助**：如果没有明显动作，依据语言判断。
        
        请严格输出 JSON 格式：
        {{
            "thought": "分析学生动作与语言，判断其倾向于哪个选项...",
            "selected_index": 0
        }}
        """

        try:
            resp = self.client.chat.completions.create(
                model=Config.MODEL_TEACHER, 
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            # 使用全局工具 safe_parse_json 解析
            result = safe_parse_json(resp.choices[0].message.content)
            
            thought = result.get("thought", "解析失败，无思考过程")
            idx = result.get("selected_index", 0)
            
            # 越界保护
            if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
                idx = 0
            
            print(f"🧠 [Intent Match] {thought} -> Selected: '{candidates[idx].desc}'")
            return candidates[idx], thought
        
        except Exception as e:
            self.logger.error(f"Intent matching failed: {e}")
            return candidates[0], f"意图识别发生错误: {e}"

    def _build_scenario_intro(self) -> str:
        """
        [Helper] 构建静态的实验场景介绍 (只在开场使用)
        """
        exp_title = self.exp_config.get("title", "未命名化学实验")
        equipment_list = self._generate_equipment_list_str()
        
        return (
            f"今天是化学实验课，实验主题是《{exp_title}》。\n"
            f"面前的实验台上已经为你准备好了以下器材：\n"
            f"【{equipment_list}】\n"
            f"老师正在看着你，等待你开始。"
        )

    def _build_teacher_context(self, 
                               scenario_intro: str, 
                               env_state_str: str, 
                               execution_log: str, 
                               xdl_context: str, 
                               validation_hints: str,
                               is_intercepted: bool = False,
                               intercept_reason: str = "") -> str:
        """
        [Helper] 统一构建 Teacher 的 System Info，避免主循环混乱
        """
        # 1. 基础环境 (总是需要)
        sections = [
            f"【当前场景】: {scenario_intro}", # 简短保留场景感
            f"【📊 上帝视角 (God View)】:\n{env_state_str}"
        ]

        # 2. 拦截模式 vs 正常模式
        if is_intercepted:
            sections.append(f"【⚠️ 拦截报告】:\n{intercept_reason}")
            # 拦截时也需要知道标准答案，以便引导
            sections.append(f"【标准参考】:\n{xdl_context}")
        else:
            # 正常执行模式
            sections.append(f"【最新操作反馈】:\n{execution_log}")
            sections.append(f"【当前任务参考】:\n{xdl_context}")
            
            if validation_hints:
                sections.append(f"{validation_hints}")

        return "\n\n".join(sections)

    def _constraints_to_natural_language(self, constraints: dict) -> str:
        """
        [NEW] 将物理约束 JSON 转换为自然语言，供 Agent 理解验收标准。
        """
        if not constraints:
            return "（当前任务无特定物理约束，或属于自由探索）"
            
        lines = []
        
        # 1. 状态约束
        if "requirements" in constraints:
            for req in constraints["requirements"]:
                role = req.get("role", "Object")
                criteria_desc = []
                for c in req.get("criteria", []):
                    check = c.get("check")
                    args = c.get("args", [])
                    
                    if check == "is_sealed":
                        criteria_desc.append("必须密封")
                    elif check == "is_covered":  # <--- [新增]
                        criteria_desc.append("瓶口需盖上 (Glass Plate/Lid)")
                    elif check == "has_chemical":
                        criteria_desc.append(f"含有 {args[0]}")
                    elif check == "is_heated":
                        criteria_desc.append(f"温度达到 {args[0]}度")
                    elif check == "is_type":
                        criteria_desc.append(f"必须是 {args[0]} 类型")
                        
                lines.append(f"- 【{role}】: {', '.join(criteria_desc)}")
                
        # 2. 拓扑约束
        if "topology_requirements" in constraints:
            for link in constraints["topology_requirements"]:
                args = link.get("args", [])
                if len(args) >= 2:
                    lines.append(f"- 【连接】: {args[0]} 必须连接到 {args[1]} ({link.get('type')}方式)")
                    
        return "\n".join(lines)

    def _generate_equipment_list_str(self) -> str:
        """
        [NEW] 生成桌面上器材的自然语言清单
        格式：中文标准名 (ID)
        例如：直流电源 (DC_power_supply), 烧杯 (beaker_1), 烧杯 (beaker_2)
        """
        items = []
        
        # 获取硬件管理器引用
        hw_mgr = self.sim.hw_manager
        
        for vid, obj in self.sim.containers.items():
            # 1. 基础过滤
            if not hasattr(obj, 'name'): continue
            
            # 2. 获取类型和标准名称
            obj_type = hw_mgr._get_type(vid)
            standard_name = None
            
            if obj_type and obj_type in hw_mgr.manifest:
                cfg = hw_mgr.manifest[obj_type]
                standard_name = cfg.get("name") # 从 YAML 获取标准中文名
            
            # 3. 确定显示名称
            # 如果 Manifest 有中文名，用中文名；否则用 obj.name；最后用 vid 兜底
            human_name = standard_name if standard_name else (obj.name if obj.name else vid)
            
            # 4. [核心修改] 拼接名称与 ID
            # 格式示例: "直流电源 (DC_power_supply)" 或 "烧杯 (beaker_1)"
            display_str = f"{human_name} ({vid})"
            
            items.append(display_str)
        
        # 5. 排序并返回
        # 注意：这里 set() 不会再合并同类器材了，因为它们的 ID 不同
        # 例如 "烧杯 (b1)" 和 "烧杯 (b2)" 是不同的字符串
        unique_items = sorted(list(set(items)))
        
        return "、".join(unique_items)

    def _format_actions_for_teacher(self, actions: List[Dict]) -> str:
        """
        将 XDL 动作列表转换为自然语言/伪代码，供 Teacher 阅读
        """
        if not actions:
            return "（无物理动作）"
        
        lines = []
        for act in actions:
            # 提取动作类型
            verb = act.get("action", "Unknown")
            # 提取关键参数 (忽略 id 等技术字段)
            details = []
            
            # 针对不同动作提取参数
            if "vessel" in act: details.append(f"容器={act['vessel']}")
            if "reagent" in act: details.append(f"试剂={act['reagent']}")
            if "volume" in act: details.append(f"量={act['volume']}")
            if "mass" in act: details.append(f"量={act['mass']}")
            if "from_vessel" in act: details.append(f"从={act['from_vessel']}")
            if "to_vessel" in act: details.append(f"到={act['to_vessel']}")
            if "tool" in act: details.append(f"工具={act['tool']}")
            
            lines.append(f"- {verb}: {', '.join(details)}")
            
        return "\n".join(lines)

    def _generate_grounded_constraints(self, target_node):
        """
        [Hybrid Core v3.0] 混合生成：Ghost Sim + Procedural Intent (LLM) -> JSON
        利用 LLM 的语义理解能力，结合动作历史和物理快照来生成规则。
        """
        task_desc = target_node.desc
        
        # 1. 缓存检查
        if task_desc == self.last_task_desc and self.current_constraints:
            return

        print(Fore.BLUE + f"⚙️ [Hybrid Gen] Calculating Ground Truth for: '{task_desc}'...")

        # 2. 获取标准 XDL 动作序列
        actions = self.validator._get_cumulative_actions(self.sim, target_node)
        
        if not actions:
            print(Fore.RED + f"  -> ⚠️ No standard actions found for node '{target_node.id}'. Skipping constraints.")
            self.current_constraints = {} 
            return

        # =========================================================
        # [NEW] Step 2.1: 动作历史格式化 (Procedural History Formatting)
        # =========================================================
        procedural_lines = []
        for i, act in enumerate(actions):
            verb = act.get('action', 'Unknown')
            details = []
            
            # 提取关键参数，帮助 LLM 理解意图
            # 1. 温度意图
            if 'temperature' in act: 
                details.append(f"TargetTemp={act['temperature']}")
            elif 'target_temperature' in act:
                details.append(f"TargetTemp={act['target_temperature']}")
            
            # 2. 物质与工具
            if 'reagent' in act: details.append(f"Reagent={act['reagent']}")
            if 'tool' in act: details.append(f"Tool={act['tool']}")
            if 'vessel' in act: details.append(f"Target={act['vessel']}")
            
            # 3. 量
            if 'volume' in act: details.append(f"Vol={act['volume']}")
            if 'mass' in act: details.append(f"Mass={act['mass']}")

            procedural_lines.append(f"Step {i+1}: {verb} ({', '.join(details)})")
            
        procedural_history_str = "\n".join(procedural_lines)
        # 调试打印，确保格式正确
        # print(Fore.CYAN + f"   -> History context:\n{procedural_history_str}")

        # =========================================================
        # Step 3: 幽灵推演 (Ghost Simulation) 获取 Ground Truth
        # =========================================================
        # 在幽灵沙箱中执行动作并快进，以获取“完成后的物理状态”
        report = self.sim.fork_and_predict(actions, steps=15)
        ideal_snap = report['snapshot']
        
        # 提取 Grounding Info (物理快照文本化)
        grounding_lines = []
        for vid, data in ideal_snap['hardware'].items():
            obj_type = str(data.get('type', '')).lower()
            if obj_type == 'vessel':
                contents = data.get('contents', {})
                # 过滤微量物质
                chemicals = [f"{k}={v:.3f}mol" for k,v in contents.items() if v > 1e-4 and k != 'H2O']
                chem_str = ", ".join(chemicals) if chemicals else "Empty"
                
                # 提取状态
                temp = data.get('temperature', 25.0)
                status_tags = []
                if data.get('is_sealed'): status_tags.append("🔒SEALED")
                # 标记当前温度 (LLM 会对比 History 和这个温度)
                status_tags.append(f"Temp={temp:.1f}C")
                
                status_str = " ".join(status_tags)
                
                # 只有有意义的信息才加入
                if chemicals or status_str or "SEALED" in status_str:
                    grounding_lines.append(f"- 容器({data.get('name', vid)}): {status_str} [{chem_str}]")
            
        # 拓扑连接
        topology = ideal_snap.get('topology', [])
        if topology:
            topo_descs = [f"{link['child']}--({link['type']})-->{link['parent']}" for link in topology]
            grounding_lines.append(f"- 装置连接状态: {', '.join(topo_descs)}")
        
        grounding_info_str = "\n".join(grounding_lines)

        # =========================================================
        # [核心修改] 动态提取 XDL 中的官方 Type
        # =========================================================
        # 我们遍历 self.sim.containers，获取每个物体在初始化时记录的 type
        # 这些 type 正是你 XDL 里写的 "rubber_stopper_with_tube", "stand", "bottle" 等
        
        xdl_types_map = {} # 格式: {id: type} 用于调试
        valid_types_set = set()
        
        for vid, obj in self.sim.containers.items():
            # 获取 XDL 定义的原始类型
            # 假设 HardwareManager 或 Container 记住了这个值
            # 如果没记住，我们需要去 hw_manager 查
            raw_type = self.sim.hw_manager._get_type(vid) 
            
            if raw_type and raw_type != "unknown":
                valid_types_set.add(raw_type)
                xdl_types_map[vid] = raw_type
        
        # 排序并转为字符串，供 Prompt 使用
        valid_types_list = sorted(list(valid_types_set))
        valid_types_str = ", ".join([f'"{t}"' for t in valid_types_list])

        # 打印调试信息，确保提取到了正确的词
        # print(Fore.CYAN + f"   -> XDL Valid Types Extracted: {valid_types_str}")

        # =========================================================
        # Step 4: 注入到 Prompt 中
        # =========================================================
        
        base_prompt = PromptManager.get_grounded_constraint_prompt(
            task_desc, 
            grounding_info=grounding_info_str,
            procedural_history=procedural_history_str,
            valid_types_str=valid_types_str  # <--- Critical Fix
        )
        
        # [关键] 告诉 LLM：只能从 XDL 定义的词表里选！
        type_constraint_instruction = f"""
        \n## 4. 🚨 STRICT TYPE CONSTRAINT (从 XDL 定义中选择)
        When generating `is_type` checks, you MUST ONLY use the exact `type` values defined in the XDL file.
        Do NOT use the Object ID. Do NOT invent new words.
        
        **✅ Valid XDL Types (Whitelist)**: 
        [{valid_types_str}]
        
        **Example Rules**:
        - If the object ID is "rubber_stop_tube" but its XDL type is "rubber_stopper_with_tube":
          - ✅ Correct: {{"check": "is_type", "args": ["rubber_stopper_with_tube"]}}
          - ❌ Wrong (Using ID): {{"check": "is_type", "args": ["rubber_stop_tube"]}}
          - ❌ Wrong (Partial): {{"check": "is_type", "args": ["stopper"]}}
        """
        
        final_prompt = base_prompt + type_constraint_instruction
        
        max_retries = 3
        last_error_msg = ""

        for attempt in range(max_retries):
            try:
                # 构造当前轮次的 Prompt
                current_system_prompt = final_prompt
                
                if last_error_msg:
                    current_system_prompt += (
                        f"\n\n🚨 [SYSTEM ERROR FIX]: Your previous JSON output was invalid.\n"
                        f"Error Details: {last_error_msg}\n"
                        f"Please fix the JSON format. Output RAW JSON only."
                    )

                # 调用 LLM
                resp = self.client.chat.completions.create(
                    model=Config.MODEL_TEACHER,
                    messages=[{"role": "system", "content": current_system_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2 
                )
                
                raw_content = resp.choices[0].message.content
                raw_data = safe_parse_json(raw_content)

                # --- 深度校验 (Deep Validation) ---
                if not raw_data:
                    raise ValueError("Parsed JSON is empty.")
                
                if "requirements" not in raw_data and "topology_requirements" not in raw_data:
                    raise ValueError("Missing 'requirements' or 'topology_requirements' keys.")

                # 归一化处理
                if "requirements" in raw_data:
                    VALID_PREDICATES = {"has_chemical", "is_heated", "is_sealed", "is_type"}
                    for req in raw_data["requirements"]:
                        if not req.get("role"): 
                             raise ValueError("Missing 'role' in requirements.")
                        
                        if "criteria" in req and isinstance(req["criteria"], list):
                            normalized = []
                            for c in req["criteria"]:
                                if c.get("check") not in VALID_PREDICATES:
                                    raise ValueError(f"Invalid predicate '{c.get('check')}'.")
                                if isinstance(c, str): normalized.append({"check": c, "args": []})
                                elif isinstance(c, dict): normalized.append(c)
                            req["criteria"] = normalized

                # === 成功出口 ===
                self.current_constraints = raw_data
                self.last_task_desc = task_desc
                if attempt > 0:
                    print(Fore.GREEN + f"   -> ✅ Fixed and generated successfully on attempt {attempt+1}.")
                return 

            except Exception as e:
                last_error_msg = str(e)
                if attempt == max_retries - 1:
                    self.logger.error(f"❌ Hybrid Gen Failed. Final Error: {e}")
                    self.current_constraints = {}

    def _init_dataset_file(self):
        # 确保文件存在
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                pass
        except Exception as e:
            self.logger.error(f"无法创建数据集文件: {e}")

    def _translate_xdl_to_pedagogy(self, actions: List[Dict]) -> str:
        """
        [Refined] 将 XDL 动作序列翻译为纯自然语言，移除所有 JSON 痕迹
        """
        lines = []
        for i, act in enumerate(actions):
            verb = act.get("action")
            desc = ""
            
            # 提取对象名称（优先使用中文名，如果没有则用ID）
            def get_name(key):
                vid = act.get(key)
                if not vid: return "未知物体"
                # 尝试从 sim 的容器中获取 name
                obj = self.sim.containers.get(vid)
                if obj and hasattr(obj, 'name'): return obj.name
                return vid # 兜底

            # === 规则优化 ===
            if verb == "Attach":
                child = get_name("vessel") or get_name("object")
                parent = get_name("support") or get_name("target")
                desc = f"将 {child} 固定/连接到 {parent} 上"
                
            elif verb == "Add":
                reagent = act.get("reagent")
                target = get_name("vessel")
                amt = act.get("volume") or act.get("mass") or "适量"
                desc = f"向 {target} 中加入 {amt} 的 {reagent}"
                
            elif verb == "Insert":
                tool = get_name("tool") or get_name("object")
                vessel = get_name("vessel")
                desc = f"将 {tool} 插入 {vessel} 中"
                
            elif verb == "Transfer":
                src = get_name("from_vessel")
                dst = get_name("to_vessel")
                desc = f"将 {src} 中的物质倒入 {dst}"
                
            elif verb == "Heat":
                target = get_name("vessel")
                temp = float(act.get("target_temperature", 999))
                if temp < 50:
                    desc = f"用手捂热 {target} (微热)"
                else:
                    desc = f"加热 {target}"

            else:
                # 兜底：虽然还有点机器味，但去掉了括号和引号
                args = [f"{k}={v}" for k,v in act.items() if k not in ["action", "id", "agent"]]
                desc = f"执行 {verb} ({', '.join(args)})"
                
            lines.append(f"{i+1}. {desc}")
            
        return "\n".join(lines)

    def _get_node_reference_context(self, task_node) -> str:
        """
        [Helper] 获取某个任务节点对应的标准 XDL 动作参考上下文
        """
        if not task_node:
            return ""

        task_desc = task_node.desc
        target_step_ids = getattr(task_node, "required_steps", [])
        
        # 提取并清洗数据
        reference_actions = []
        for sid in target_step_ids:
            if sid in self.sim.initial_xdl_procedure:
                # 浅拷贝并过滤掉系统字段 (省 Token)
                raw_act = self.sim.initial_xdl_procedure[sid]
                clean_act = {k: v for k, v in raw_act.items() if k not in ['id', 'agent', 'step_id']}
                reference_actions.append(clean_act)
        
        # 序列化
        if reference_actions:
            # [修改] 使用新的翻译器
            readable_steps = self._translate_xdl_to_pedagogy(reference_actions)
            
            return (
                f"\n【实验参考步骤】:\n"
                f"{readable_steps}\n"
                f"(注意：如果步骤中包含‘微热/30度’，意味着严禁使用酒精灯。\n"
                f" **重要：参考步骤仅供参考。如果学生的操作顺序（如先加药后塞棉花）符合物理常识且不造成危险，请允许其通过，不要死板地照搬步骤顺序。**)" 
            )
        return ""
    
    # 修改签名，增加 student_actions
    def _resolve_task_info(self, selection_result, task_selector=None):
        """
        [Helper] 解析任务选择结果 (返回 4 个值)
        Returns:
            task_desc (str): 任务描述文本
            node_or_list (Obj): 节点对象或列表 (用于逻辑判断)
            xdl_info (str): 节点的 XDL 上下文信息 (用于 Teacher Prompt)
            constraint_str (str): 物理验证规则
        """
        task_desc = "自由探索"
        xdl_info = ""
        constraint_str = ""
        
        # Case 1: 分支列表
        if isinstance(selection_result, list):
            candidates = selection_result
            options_text = "、".join([f"【{n.desc}】" for n in candidates])
            task_desc = (
                f"【决策阶段】当前出现分支，可选任务有：{options_text}。"
                f"请询问学生想进行哪一个实验？暂时不要进行具体步骤指导，仅确认意图。"
            )
            self.current_constraints = {}
            constraint_str = "等待学生选择任务分支..."
            
            # 返回: 描述, 列表对象, 空XDL信息, 空规则
            return task_desc, candidates, "", constraint_str 

        # Case 2: 单个节点
        if selection_result:
            current_node = selection_result
            task_desc = current_node.desc
            
            if current_node.id != getattr(self, "last_focused_node_id", None) or not self.current_constraints:
                self._generate_grounded_constraints(current_node)
                self.last_focused_node_id = current_node.id
                
            xdl_info = self._get_node_reference_context(current_node)
            constraint_str = self._constraints_to_natural_language(self.current_constraints)
            
            # 返回: 描述, 单个节点对象, XDL文本, 规则文本
            return task_desc, current_node, xdl_info, constraint_str
            
        # Case 3: 空
        return task_desc, None, "", "无特定约束"
    
    def _handle_experiment_end(self):
        print(Fore.MAGENTA + "🎉 恭喜！所有实验步骤已完成！")
        hist_str = self.memory.get_context_for_prompt()
        
        # === FIXED: Align arguments with TeacherAgent.respond definition ===
        t_final = self.teacher.respond(
            history_str=hist_str,                # Pass memory context here
            policy_decision={},                  # Empty policy
            focus_goal="实验总结",                # Focus goal
            cognitive_state="实验结束状态",       # Simple state description
            scaffold_ctx={},                     # Empty scaffold context
            policy_instruction="实验已结束，请对学生的整体表现进行总结。", # Explicit instruction
            env_state="(实验结束)",               # Placeholder environment state
            reference_info=""                    # No reference needed
        )
        # =================================================================
        
        print(Fore.GREEN + f"[Teacher Final]: {t_final.get('response')}")

    def run_episode(self, max_turns=15):
        """
        全量记录版主循环：包含 Turn 0 开场与 Turn 1-N 主交互。
        """
        # 1. 初始化全量记录器
        recorder = UniversalRecorder(self.output_file)
        
        print(Fore.CYAN + f"🚀 [System] Session Started: {self.session_id}")
        
        # === Phase 0: 静态场景与开场初始化 ===
        
        # 获取初始物理快照
        obs = self.sim.get_snapshot() # 这是 Turn 0 的初始状态
        scenario_intro = self._build_scenario_intro()
        
        # 初始任务分析 (为了 Turn 0 的 Teacher 知道该引导什么)
        task_selector = SmartTaskSelector() 
        dag_status = self.oracle.analyze(obs, set())
        first_selection = task_selector.select_next_task(dag_status, cognitive_model=self.cognitive_model)
        first_task_desc, _, xdl_context_info, constraint_str = self._resolve_task_info(first_selection, task_selector)
        
        # 记录系统开场白到记忆
        self.memory.add_turn("system", scenario_intro)

        # ---------------------------------------------------------
        # Turn 0: 开场交互 (完全展开，不省略)
        # ---------------------------------------------------------
        print(Fore.YELLOW + f"\n{'='*15} Turn 0 (Opening) {'='*15}")
        
        # 0.1 更新学生状态
        current_clumsiness = self.cognitive_model.get_dynamic_clumsiness()
        self.student.update_status(current_clumsiness)
        
        # 0.2 学生开场 Input
        s0_instruction = f"【场景背景】: {scenario_intro}\n【指令】: 请向老师打招呼，简述你知道的实验目标，并表现出你的人设。"
        s0_input_packet = {
            "history": [],
            "last_teacher_msg": s0_instruction,
            "perception": ObservationEngine.describe_full_state(obs, "student"),
            "traits": self.student_profile
        }
        
        # 0.3 学生生成 Output
        s0_resp = self.student.respond(
            history=[], 
            last_teacher_msg=s0_instruction, 
            snapshot=obs, 
            reagent_map=self.exp_config['reagent_map'],
            instruction="你现在处于实验准备阶段。"
        )
        s0_speak = s0_resp.get("speak", "老师好。")
        print(Fore.WHITE + f"[Student]: {s0_speak}")
        self.memory.add_turn("student", s0_speak)
        
        # 0.4 老师开场策略
        t0_policy = {
            "strategy_type": "RAPPORT_BUILDING",
            "thought_trace": "实验开始。建立关系，确认状态，引导进入第一个任务。",
            "instruction_to_teacher": "热情回应，引导学生思考第一步。"
        }
        
        # 0.5 老师生成 Output
        # 开场时物理环境是初始状态
        god_view_str_0 = ObservationEngine.describe_full_state(obs, "god")
        
        t0_resp = self.teacher.respond(
            history_str=json.dumps([{"role": "student", "content": s0_speak}], ensure_ascii=False),
            policy_decision=t0_policy,
            focus_goal=f"引导进入任务：{first_task_desc}",
            cognitive_state=self.cognitive_model.get_diagnosis(),
            scaffold_ctx=self.cognitive_model.get_scaffold_context(),
            env_state=god_view_str_0,
            reference_info=xdl_context_info,
            policy_instruction=t0_policy["instruction_to_teacher"],
            causal_insight="（实验刚开始，物理状态平稳）"
        )
        t0_speak = t0_resp.get("response", "你好，开始吧。")
        print(Fore.GREEN + f"[Teacher]: {t0_speak}")
        self.memory.add_turn("teacher", t0_speak)

        # 0.6 [Record Turn 0]
        recorder.record_turn(self.session_id, 0, {
            "meta": {"phase": "opening"},
            # ================= [新增] Turn 0 也记录 BKT =================
            "bkt_status": {
                "knowledge_state": self.cognitive_model.knowledge_state.copy(),
                "scaffold_level": self.cognitive_model.scaffold_level
            },
            # ==========================================================
            "physics_context": {"snapshot_initial": obs, "god_view": god_view_str_0},
            "agents": {
                "student": {"input": s0_input_packet, "output": s0_resp},
                "policy": {"input": "fixed_opening_rule", "output": t0_policy},
                "teacher": {"input": "opening_context", "output": t0_resp}
            }
        })

        # ---------------------------------------------------------
        # Turn 1 ~ N: 主循环 (Main Loop) - 修复版
        # ---------------------------------------------------------
        last_turn_fail_count = 999 

        for turn in range(1, max_turns + 1):
            print(Fore.YELLOW + f"\n{'='*15} Turn {turn} {'='*15}")
            
            # === 数据包容器 (Super Packet) ===
            # [修正] 将 policy 拆分为 intent (拦截) 和 feedback (决策)
            turn_record = {
                "meta": {"current_task_desc": "", "constraints": {}},
                "physics_context": {},
                "agents": {
                    "student": {}, 
                    "policy_intent": {},   # 1. 拦截策略
                    "policy_feedback": {}, # 2. 教学策略 (新增)
                    "teacher": {}
                },
                "evaluation": {}
            }

            # --- 1. 环境感知与任务更新 ---
            dag_status = self.oracle.analyze(obs, set())
            selection_result = task_selector.select_next_task(dag_status, cognitive_model=self.cognitive_model)
            
            if not selection_result and dag_status.get('completed_nodes'):
                self._handle_experiment_end()
                break

            current_task_desc, node_or_list, xdl_context_info, constraint_str = self._resolve_task_info(
                selection_result, task_selector
            )
            is_branching = isinstance(node_or_list, list)
            candidate_nodes = node_or_list if is_branching else []
            
            # [CAPTURE META]
            turn_record["meta"]["current_task_desc"] = current_task_desc
            turn_record["meta"]["constraints"] = self.current_constraints
            turn_record["meta"]["constraints_text"] = constraint_str

            # 准备物理视图 (Before Action)
            obs_before = obs 
            god_view_str = ObservationEngine.describe_full_state(obs_before, "god")
            student_view_str = ObservationEngine.describe_full_state(obs_before, "student")
            short_env_summary = describe_snapshot_briefly(obs_before)

            # [CAPTURE PHYSICS BEFORE]
            turn_record["physics_context"]["snapshot_before"] = obs_before
            turn_record["physics_context"]["god_view_text"] = god_view_str

            # --- 2. 学生代理 (Student Agent) ---
            s_internal_instruction = self.cognitive_model.generate_behavior_instruction()
            
            # [CAPTURE STUDENT INPUT]
            turn_record["agents"]["student"]["input"] = {
                "history_tail": list(self.memory.history)[-5:], 
                "last_teacher_msg": self.memory.history[-1]['content'],
                "perception": student_view_str,
                "internal_instruction": s_internal_instruction,
                "cognitive_state": self.cognitive_model.get_diagnosis()
            }

            s_resp = self.student.respond(
                history=self.memory.history, 
                last_teacher_msg=self.memory.history[-1]['content'], 
                snapshot=obs_before, 
                reagent_map=self.exp_config['reagent_map'],
                instruction=f"【行为指令】: {s_internal_instruction}" 
            )

            # [CAPTURE STUDENT OUTPUT]
            turn_record["agents"]["student"]["output"] = s_resp

            s_speak = s_resp.get("speak")
            pending_actions = s_resp.get("actions_ready_to_run", [])
            
            if s_speak: print(Fore.WHITE + f"[Student]: {s_speak}")
            if pending_actions: print(Fore.WHITE + f"          (Action): {json.dumps(pending_actions, ensure_ascii=False)}")
            
            self.memory.add_turn("student", s_speak if s_speak else "（默默操作）")
            intent_log_data = None # 用于记录到 JSONL
            # --- 2.1 意图锁定 ---
            if is_branching and candidate_nodes:
                # [Mod] 接收 (target_node, thought_trace)
                target_node, intent_thought = self._match_student_intent(s_speak, pending_actions, candidate_nodes)
                
                # 记录意图分析数据
                intent_log_data = {
                    "thought": intent_thought,
                    "choice": target_node.desc,
                    "candidates": [n.desc for n in candidate_nodes]
                }
                
                if target_node:
                    print(f"🔒 Locking focus: {target_node.desc}")
                    task_selector.current_focus_id = target_node.id
                    current_task_desc, _, xdl_context_info, constraint_str = self._resolve_task_info(target_node, task_selector)
                    turn_record["meta"]["current_task_desc"] = current_task_desc 
            
            # [新增] 将意图分析保存到本轮记录中
            if intent_log_data:
                turn_record["agents"]["intent_analysis"] = intent_log_data

            # --- 3. 拦截策略 (Policy Agent - Intent Phase) ---
            ghost_report = self.sim.fork_and_predict(pending_actions, steps=20)
            
            # [CAPTURE GHOST]
            turn_record["physics_context"]["ghost_report"] = {
                "status": ghost_report.get("status"),
                "alerts": ghost_report.get("safety_alerts"),
                "causal_chains": [c.to_dict() for c in ghost_report.get("causal_chains", [])]
            }

            policy_ctx_text = f"{current_task_desc}\n【环境】:{short_env_summary}\n【标准】:{constraint_str}"
            
            # [CAPTURE POLICY INTENT INPUT]
            turn_record["agents"]["policy_intent"]["input"] = {
                "student_act": pending_actions,
                "student_say": s_speak,
                "ghost_status": ghost_report.get("status"),
                "context": policy_ctx_text
            }

            # 决策：是否拦截
            policy_decision = self.policy_agent.decide_on_intent(
                s_speak, pending_actions, ghost_report, policy_ctx_text
            )

            # [CAPTURE POLICY INTENT OUTPUT]
            turn_record["agents"]["policy_intent"]["output"] = policy_decision

            is_intercepted = False
            intercept_reason = ""
            if policy_decision.get('decision') in ['INTERCEPT', 'EMERGENCY_STOP']:
                is_intercepted = True
                intercept_reason = policy_decision.get('reasoning', '风险操作')
                print(Fore.RED + f"🛑 [INTERCEPTED]: {intercept_reason}")

            # --- 4. 物理执行 (Execution) ---
            real_exec_info = ""
            obs_after = obs_before 
            
            if is_intercepted:
                turn_record["evaluation"]["execution_status"] = "INTERCEPTED"
                self.memory.add_turn("system", f"【系统警告】动作被拦截！原因: {intercept_reason}")
                
                # 强制覆盖 Policy 为拦截模式
                policy_decision = {"strategy_type": "PREDICTIVE_QUESTIONING", "instruction_to_teacher": f"拦截！指出 {intercept_reason}"}
                final_focus_goal = "[最高优先级] 阻止危险行为"
                final_policy_instr = f"【拦截模式】{intercept_reason}"
                final_ref_info = xdl_context_info 
                
            else:
                turn_record["evaluation"]["execution_status"] = "EXECUTED"
                
                # 执行动作
                if pending_actions:
                    real_exec_info, obs_after = self.sim.execute_batch(pending_actions)
                    print(Fore.BLUE + f"⚡ [Physics]: {real_exec_info}")
                    self.memory.add_turn("system", f"【系统反馈】观察结果: {real_exec_info}")
                else:
                    wait_log, obs_after = self.sim.execute_batch([], duration=3.0)
                    real_exec_info = wait_log if "观测到" in wait_log else "(环境静止)"
                    self.memory.add_turn("system", f"【系统反馈】{wait_log}")

                # [CAPTURE PHYSICS AFTER]
                turn_record["physics_context"]["snapshot_after"] = obs_after
                turn_record["physics_context"]["exec_log"] = real_exec_info

                # --- 验证 (Validation) ---
                search_scope = list({act['vessel'] for act in pending_actions if act.get('vessel')}) if pending_actions else None
                is_goal_met, fail_reasons = self.validator.validate_state(obs_after, self.current_constraints, focus_ids=search_scope)
                
                # [CAPTURE EVALUATION]
                turn_record["evaluation"]["is_goal_met"] = is_goal_met
                turn_record["evaluation"]["fail_reasons"] = fail_reasons

                # 进度计算
                current_fail_count = len(fail_reasons)
                is_making_progress = (current_fail_count < last_turn_fail_count) and (current_fail_count > 0)
                if current_fail_count != last_turn_fail_count: last_turn_fail_count = current_fail_count

                # 处理节点完成
                if is_goal_met and selection_result and not isinstance(selection_result, list):
                    print(Fore.MAGENTA + f"✅ 任务完成: {selection_result.desc}")
                    self.oracle.mark_node_complete(selection_result.id)
                    self.cognitive_model.update_state(True, [])
                    task_selector.current_focus_id = None
                elif pending_actions:
                    self.cognitive_model.update_state(False, [])

                # === [补全部分] 教学策略 (Policy Agent - Feedback Phase) ===
                
                # 准备 Feedback Policy 的输入
                feedback_policy_input = {
                    "student_text": s_speak,
                    "dag_status": self.oracle.analyze(obs_after, set()), # 简单的序列化可能不行，这里可以只存关键信息
                    "cognitive_diagnosis": self.cognitive_model.get_diagnosis(),
                    "sim_engine": "ref(sim_engine)", # 不存引擎对象
                    "pending_actions": pending_actions,
                    "is_goal_met": is_goal_met,
                    "fail_reasons": fail_reasons,
                    "physics_observation": real_exec_info,
                    "current_task_desc": current_task_desc,
                    "is_making_progress": is_making_progress
                }

                # [CAPTURE POLICY FEEDBACK INPUT]
                turn_record["agents"]["policy_feedback"]["input"] = feedback_policy_input

                # 调用 Feedback Policy
                policy_decision = self.policy_agent.decide(
                    student_text=s_speak,
                    dag_status=self.oracle.analyze(obs_after, set()),
                    cognitive_diagnosis=self.cognitive_model.get_diagnosis(),
                    intent_analysis={"is_consistent": True},
                    sim_engine=self.sim,
                    pending_actions=pending_actions,
                    student_profile=self.student_profile,
                    current_snapshot=obs_after,
                    is_goal_met=is_goal_met,
                    fail_reasons=fail_reasons,
                    physics_observation=real_exec_info,
                    current_task_desc=current_task_desc,
                    is_making_progress=is_making_progress
                )
                
                # [CAPTURE POLICY FEEDBACK OUTPUT]
                turn_record["agents"]["policy_feedback"]["output"] = policy_decision

                # 准备 Teacher 参数
                final_focus_goal = self.dialogue_manager.synthesize_prompt(current_task_desc, policy_decision)
                final_policy_instr = policy_decision.get("instruction_to_teacher", "")
                validation_hint = "\n".join([f"- 待解决: {r}" for r in fail_reasons]) if not is_goal_met else ""
                final_ref_info = xdl_context_info + "\n" + validation_hint

            # --- 5. 教师代理 (Teacher Agent) ---
            
            # 准备上下文
            full_env_state_str = ObservationEngine.describe_full_state(obs_after, perspective="god")
            hist_context_str = self.memory.get_context_for_prompt()
            causal_insight = "\n".join([c['desc'] for c in turn_record["physics_context"].get("ghost_report", {}).get("causal_chains", [])])

            # [CAPTURE TEACHER INPUT]
            turn_record["agents"]["teacher"]["input"] = {
                "history_context": hist_context_str,
                "policy_decision": policy_decision, # 这里已经是最终决策（拦截 或 教学）
                "focus_goal": final_focus_goal,
                "cognitive_state": self.cognitive_model.get_diagnosis(),
                "scaffold": self.cognitive_model.get_scaffold_context(),
                "env_state": full_env_state_str,
                "reference": final_ref_info,
                "instruction": final_policy_instr,
                "insight": causal_insight
            }

            # 调用 Teacher
            t_resp = self.teacher.respond(
                history_str=hist_context_str,
                policy_decision=policy_decision,
                focus_goal=final_focus_goal,
                cognitive_state=self.cognitive_model.get_diagnosis(),
                scaffold_ctx=self.cognitive_model.get_scaffold_context(),
                env_state=full_env_state_str,
                reference_info=final_ref_info,
                policy_instruction=final_policy_instr,
                causal_insight=causal_insight
            )

            # [CAPTURE TEACHER OUTPUT]
            turn_record["agents"]["teacher"]["output"] = t_resp

            t_content = t_resp.get("response", "...")
            print(Fore.GREEN + f"[Teacher]: {t_content}")
            self.memory.add_turn("teacher", t_content)

            # --- 6. 结束本轮与状态更新 ---

            # =========================================================
            # [User Requirement] 新增字段：保存本轮 BKT 和 Scaffold 数值
            # =========================================================
            turn_record["bkt_status"] = {
                "knowledge_state": self.cognitive_model.knowledge_state.copy(), # 记录当前的 KC 概率值
                "scaffold_level": self.cognitive_model.scaffold_level           # 记录当前的支架等级
            }
            # =========================================================
            
            # 写入文件
            recorder.record_turn(self.session_id, turn, turn_record)
            
            # 更新 obs 为最新状态，供下一轮使用
            obs = obs_after

    def _save_data_point(self, turn, inputs, outputs):
        """
        构造符合 Alpaca/ShareGPT 格式的微调数据
        """
        data_point = {
            "id": f"{self.session_id}_t{turn}",
            "timestamp": datetime.now().isoformat(),
            "instruction": (
                "你是一个苏格拉底式化学教学助手。请根据学生的行为、当前的物理环境状态、"
                "以及潜在的安全风险，判断教学策略并给出回复。"
                "如果存在安全风险（如ghost_simulation_result不为空），请采用 PREDICTIVE_QUESTIONING 策略。"
            ),
            "input": json.dumps(inputs, ensure_ascii=False),
            "output": json.dumps(outputs, ensure_ascii=False)
        }
        
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data_point, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"写入数据失败: {e}")


# ==========================================
# [新增模块] 真实学生交互平台 (Interactive Platform)
# ==========================================
class InteractivePlatform(SocraticDataGenerator):
    """
    继承自 SocraticDataGenerator，将 LLM 学生替换为真实的人类命令行输入。
    """
    def __init__(self, client, yaml_path: str):
        # 初始化真实的 "人类学生" 画像
        real_student_profile = {
            "name": "真实学生",
            "traits": ["真实人类"],
            "clumsiness": "low", # 真实操作不由系统模拟手抖
            "knowledge_level": "average"
        }
        super().__init__(client, yaml_path, real_student_profile, "real_student_session.jsonl")

    def run_interactive(self):
        """
        交互式主循环，完全遵循真实学生的输入与反馈节奏。
        """
        print(Fore.CYAN + f"🚀 [System] 真实交互模式启动 Session: {self.session_id}")
        
        # 1. 开场初始化
        obs = self.sim.get_snapshot()
        scenario_intro = self._build_scenario_intro()
        task_selector = SmartTaskSelector() 
        
        dag_status = self.oracle.analyze(obs, set())
        selection_result = task_selector.select_next_task(dag_status, cognitive_model=self.cognitive_model)
        current_task_desc, node_or_list, xdl_context_info, constraint_str = self._resolve_task_info(selection_result, task_selector)
        
        self.memory.add_turn("system", scenario_intro)
        last_fail_reasons = [] # 用于记录上一步未完成的原因

        # 2. 开场老师引导
        god_view_str_0 = ObservationEngine.describe_full_state(obs, "god")
        t0_resp = self.teacher.respond(
            history_str=json.dumps([], ensure_ascii=False),
            policy_decision={"strategy_type": "RAPPORT_BUILDING", "instruction_to_teacher": "热情欢迎学生，引导其开始第一步。"},
            focus_goal=f"引导进入任务：{current_task_desc}",
            cognitive_state="未知（真实学生）",
            scaffold_ctx={"level": 1, "frustration": "Low", "errors": 0},
            env_state=god_view_str_0,
            reference_info=xdl_context_info,
            policy_instruction="热情欢迎学生，引导其开始第一步。",
            causal_insight="（实验刚开始，物理状态平稳）"
        )
        t0_speak = t0_resp.get("response", "你好，请开始实验。")
        print(Fore.GREEN + f"\n[Teacher]: {t0_speak}")
        self.memory.add_turn("teacher", t0_speak)

        # 3. 交互主循环
        turn = 1
        while True:
            print(Fore.YELLOW + f"\n{'='*20} 第 {turn} 轮操作 {'='*20}")
            
            # === (1) 学生执行前：输出器材状态、连接、当前节点、失败原因 ===
            student_view_str = ObservationEngine.describe_full_state(obs, "student")
            print(Fore.CYAN + f"【当前器材状态与连接】:\n{student_view_str}")
            print(Fore.MAGENTA + f"【当前实验节点】: {current_task_desc}")
            if last_fail_reasons:
                print(Fore.RED + f"【当前节点未完成原因】:\n" + "\n".join([f"- {r}" for r in last_fail_reasons]))
            print(Fore.WHITE + "-" * 50)

            # === (2) 学生两次输入：语言 + 动作 ===
            s_speak = input(Fore.WHITE + "👨‍🎓 请输入你的回答 (Language): ").strip()
            s_action_str = input(Fore.WHITE + "👨‍🎓 请输入你的动作 (JSON格式，如无动作输入 []): ").strip()

            # 解析 JSON 动作
            pending_actions = []
            if s_action_str:
                try:
                    pending_actions = json.loads(s_action_str)
                    if not isinstance(pending_actions, list):
                        pending_actions = [pending_actions]
                except Exception as e:
                    print(Fore.RED + f"⚠️ JSON解析失败 ({e})，系统将其视为无动作执行。")

            self.memory.add_turn("student", s_speak if s_speak else "（静默操作）")

            # === (3) 物理推演与意图分析 ===
            obs_before = obs
            short_env_summary = describe_snapshot_briefly(obs_before)
            
            # 意图锁定 (如果有分支)
            is_branching = isinstance(node_or_list, list)
            if is_branching and node_or_list:
                target_node, _ = self._match_student_intent(s_speak, pending_actions, node_or_list)
                if target_node:
                    task_selector.current_focus_id = target_node.id
                    current_task_desc, _, xdl_context_info, constraint_str = self._resolve_task_info(target_node, task_selector)

            # Policy 拦截诊断 (Ghost Sim)
            ghost_report = self.sim.fork_and_predict(pending_actions, steps=20)
            policy_ctx_text = f"{current_task_desc}\n【环境】:{short_env_summary}\n【标准】:{constraint_str}"
            policy_decision = self.policy_agent.decide_on_intent(s_speak, pending_actions, ghost_report, policy_ctx_text)

            is_intercepted = policy_decision.get('decision') in ['INTERCEPT', 'EMERGENCY_STOP']
            intercept_reason = policy_decision.get('reasoning', '风险操作')
            
            obs_after = obs_before
            real_exec_info = ""

            # === (4) 执行与验证 ===
            if is_intercepted:
                print(Fore.RED + f"🛑 [系统拦截]: {intercept_reason}")
                self.memory.add_turn("system", f"【系统警告】动作被拦截！原因: {intercept_reason}")
                
                # 强制拦截策略
                policy_decision = {"strategy_type": "PREDICTIVE_QUESTIONING", "instruction_to_teacher": f"拦截！指出 {intercept_reason}"}
                final_focus_goal = "[最高优先级] 阻止危险行为"
                final_policy_instr = f"【拦截模式】{intercept_reason}"
                final_ref_info = xdl_context_info 
            else:
                # 真实执行动作
                if pending_actions:
                    real_exec_info, obs_after = self.sim.execute_batch(pending_actions)
                    print(Fore.BLUE + f"⚡ [系统执行]: {real_exec_info}")
                    self.memory.add_turn("system", f"【系统反馈】观察结果: {real_exec_info}")
                else:
                    wait_log, obs_after = self.sim.execute_batch([], duration=3.0)
                    real_exec_info = wait_log if "观测到" in wait_log else "(环境静止)"
                    self.memory.add_turn("system", f"【系统反馈】{wait_log}")

                # 校验状态
                search_scope = list({act['vessel'] for act in pending_actions if act.get('vessel')}) if pending_actions else None
                is_goal_met, fail_reasons = self.validator.validate_state(obs_after, self.current_constraints, focus_ids=search_scope)
                last_fail_reasons = fail_reasons if not is_goal_met else []

                # 如果任务完成，清除聚焦
                if is_goal_met and selection_result and not isinstance(selection_result, list):
                    print(Fore.MAGENTA + f"✅ [系统提示] 目标完成: {selection_result.desc}")
                    self.oracle.mark_node_complete(selection_result.id)
                    task_selector.current_focus_id = None

                # 获取教学反馈策略
                policy_decision = self.policy_agent.decide(
                    student_text=s_speak,
                    dag_status=self.oracle.analyze(obs_after, set()),
                    cognitive_diagnosis="真实学生",
                    intent_analysis={"is_consistent": True},
                    sim_engine=self.sim,
                    pending_actions=pending_actions,
                    student_profile={"knowledge_level": "average"},
                    current_snapshot=obs_after,
                    is_goal_met=is_goal_met,
                    fail_reasons=fail_reasons,
                    physics_observation=real_exec_info,
                    current_task_desc=current_task_desc,
                    is_making_progress=False
                )

                final_focus_goal = self.dialogue_manager.synthesize_prompt(current_task_desc, policy_decision)
                final_policy_instr = policy_decision.get("instruction_to_teacher", "")
                validation_hint = "\n".join([f"- 待解决: {r}" for r in fail_reasons]) if not is_goal_met else ""
                final_ref_info = xdl_context_info + "\n" + validation_hint

            # === (5) 执行后：给出新器材状态，并触发 Teacher 反馈 ===
            print(Fore.CYAN + f"\n【执行后器材状态与连接】:\n{ObservationEngine.describe_full_state(obs_after, 'student')}")

            full_env_state_str = ObservationEngine.describe_full_state(obs_after, perspective="god")
            hist_context_str = self.memory.get_context_for_prompt()
            
            t_resp = self.teacher.respond(
                history_str=hist_context_str,
                policy_decision=policy_decision,
                focus_goal=final_focus_goal,
                cognitive_state="未知（真实学生）",
                scaffold_ctx={"level": 1, "frustration": "Low", "errors": 0},
                env_state=full_env_state_str,
                reference_info=final_ref_info,
                policy_instruction=final_policy_instr,
                causal_insight=""
            )

            t_content = t_resp.get("response", "系统老师暂时无响应。")
            print(Fore.GREEN + f"\n[Teacher]: {t_content}")
            self.memory.add_turn("teacher", t_content)

            # 刷新下一轮任务状态
            obs = obs_after
            dag_status = self.oracle.analyze(obs, set())
            selection_result = task_selector.select_next_task(dag_status, cognitive_model=self.cognitive_model)
            current_task_desc, node_or_list, xdl_context_info, constraint_str = self._resolve_task_info(selection_result, task_selector)
            
            if not selection_result and dag_status.get('completed_nodes'):
                self._handle_experiment_end()
                break
                
            turn += 1


def main():
    init(autoreset=True)

    # 启动日志与控制台录制
    start_console_recording("session_records") 
    setup_logging()
    
    # 实例化大模型客户端
    client = OpenAI(api_key=Config.OPENAI_API_KEY, base_url=Config.OPENAI_BASE_URL)
    
    # ==========================================
    # 实验选择菜单
    # ==========================================
    print(Fore.CYAN + "\n" + "="*40)
    print(Fore.CYAN + "欢迎来到 Socratic ChemLab (真实交互平台)")
    print(Fore.CYAN + "="*40)
    print("请选择你要进行的实验：")
    print("1. 高锰酸钾制取氧气")
    print("2. 大理石与稀盐酸反应")
    print("3. 水的沸腾")
    print("4. 光束通过溶液和胶体")
    print("5. 蒸馏水和无水乙醇与钠的反应")
    print("6. 氢氧化钠滴加硫酸铜溶液")
    print("7. 重结晶方法提纯苯甲酸")
    print("8. 木炭分别在空气和 氧气中燃烧")
    print("9. 钠的燃烧")
    print("10. 石蜡的融化")


    
    # 定义实验路径字典（请根据你本地服务器上的实际路径进行调整）
    experiment_paths = {
        "1": "/home/yjh/socChem_final/experiments/2-5 高锰酸钾制取氧气_sample_chem_train_14.xdl",
        "2": "/home/yjh/socChem_final/experiments/1 大理石与稀盐酸反应_sample_chem_train_0.xdl",  # 示例路径，需替换
        "3": "/home/yjh/socChem_final/experiments/1 水的沸腾_sample_chem_train_2.xdl",  # 示例路径，需替换
        "4": "/home/yjh/socChem_final/experiments/1-1 光束通过溶液和胶体_sample_chem_train_69.xdl",  # 示例路径，需替换
        "5": "/home/yjh/socChem_final/experiments/1-1 蒸馏水和无水乙醇与钠的反应_sample_chem_train_149.xdl",  # 示例路径，需替换
        "6": "/home/yjh/socChem_final/experiments/1-5 氢氧化钠滴加硫酸铜溶液_sample_chem_train_8.xdl",  # 示例路径，需替换
        "7": "/home/yjh/socChem_final/experiments/1-探究1 重结晶方法提纯苯甲酸_sample_chem_train_150.xdl",  # 示例路径，需替换
        "8": "/home/yjh/socChem_final/experiments/2-2 木炭分别在空气和 氧气中燃烧_sample_chem_train_11.xdl",  # 示例路径，需替换
        "9": "/home/yjh/socChem_final/experiments/2-2 钠的燃烧_sample_chem_train_74.xdl",  # 示例路径，需替换
        "10": "/home/yjh/socChem_final/experiments/1 石蜡的融化_sample_chem_train_3.xdl"       # 示例路径，需替换
    }

    choice = input("\n请输入实验编号 (1-10) [默认选1]: ").strip()
    xdl_path = experiment_paths.get(choice, experiment_paths["1"])
    
    print(Fore.GREEN + f"\n>> 已选择实验，正在加载实验环境: {xdl_path}")
    print(Fore.GREEN + ">> 提示：你可以开始实验了！\n")

    # 启动真实的交互平台
    try:
        platform = InteractivePlatform(client=client, yaml_path=xdl_path)
        platform.run_interactive()
    except Exception as e:
        print(Fore.RED + f"\n系统运行异常退出: {e}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n运行耗时: {end_time - start_time:.2f} 秒")