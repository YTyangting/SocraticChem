import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from openai import OpenAI
from colorama import Fore, Style, init
from config import Config
import uuid
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import re
import math
from xdl_utils import XDLParser
import copy
import difflib
import argparse # [新增]
import networkx as nx
# === [核心依赖] ===
try:
    from chemlib import Reaction as ChemReaction
    from chemlib import Compound
except ImportError:
    # print(Fore.RED + "Error: 'chemlib' not found. Please run: pip install chemlib")
    exit(1)

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Tuple

import logging
from transformers import pipeline
import torch # 需要导入 torch 来检查 cuda

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

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

import sys

class BertIntentClassifier:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BertIntentClassifier, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        self.logger = logging.getLogger("IntentClassifier")
        self.logger.info("正在加载 BERT 意图分类模型 (首次运行会自动下载，约 400MB)...")
        
        # === 核心模型选择 ===
        # 推荐模型：MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
        # 理由：支持多语言(含中文)，在 NLI 任务上表现极佳，且体积适中
        model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

        device_id = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device_id == 0 else "CPU"
        self.logger.info(f"Using device: {device_name}")
        
        try:
            # device=-1 表示使用 CPU，如果有显卡可改为 0
            self.classifier = pipeline("zero-shot-classification", model=model_name, device=-1)
            self.logger.info("BERT 模型加载完成。")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            self.classifier = None

    def predict(self, text: str) -> str:
        """
        输入文本，返回意图： "QUESTION" 或 "STATEMENT"
        """
        if not text or not self.classifier:
            return "STATEMENT" # 兜底

        # 定义候选标签（可以使用中文标签让模型理解语义，最后再映射回英文代码）
        candidate_labels = ["询问原理或请求帮助的疑问句", "陈述事实或闲聊的陈述句"]
        
        try:
            # 核心推理
            result = self.classifier(text, candidate_labels)
            
            # result['labels'] 是按概率排序的标签列表，取第一个
            top_label = result['labels'][0]
            score = result['scores'][0]

            # 调试日志：看看模型到底有多确信
            # # print(f"Text: {text} -> {top_label} (Conf: {score:.4f})")

            # 映射回系统需要的代码
            if top_label == "询问原理或请求帮助的疑问句":
                return "QUESTION"
            else:
                return "STATEMENT"

        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            return "STATEMENT"

# 全局单例
INTENT_CLASSIFIER = BertIntentClassifier()

# === 定义容差级别 ===
class ToleranceLevel(Enum):
    STRICT = "STRICT"       # 滴定、精密量取 (误差 < 2%)
    NORMAL = "NORMAL"       # 一般添加、配液 (误差 < 10%)
    LOOSE = "LOOSE"         # 清洗、废液处理 (误差 < 30% 或 仅定性)

# === 定义评估状态 ===
class ReviewStatus(Enum):
    PASS = "PASS"
    IN_PROGRESS = "INFO"    # 保持原样
    FAIL = "FAIL"
    STUCK = "STUCK"
    WAIT = "WAIT"           # <--- [新增] 补上这个缺失的状态，防止后面报错

# === [新增] 理想状态快照 (Golden State) ===
@dataclass
class GoalState:
    step_index: int
    step_desc: str
    target_vessel_id: str
    expected_volume: float
    expected_ph: float
    expected_species: Set[str]
    tolerance: ToleranceLevel
    # === 新增字段 ===
    is_delta_check: bool = False   # 标记是否仅检查本次操作带来的变化量
    expected_delta: float = 0.0     # 预期增量（例如补加 20ml，则 delta 为 20）
    # === 既有字段 ===
    expected_moles: Dict[str, float] = field(default_factory=dict) 
    expected_topology: List[Tuple[str, str]] = field(default_factory=list)

@dataclass
class InspectionResult:
    status: ReviewStatus
    reason: str
    diagnosis: Optional[Dict[str, Any]] = None # 结构化诊断数据
    metric_delta: float = 0.0  # 记录核心物理量的变化值(ml/°C)


# ==========================================
# [新增] 全局工具函数: 历史记录格式化
# ==========================================
def format_history_to_string(history: List[Dict[str, str]], max_turns: int = 20) -> str:
    """
    将结构化的 history 列表转换为带角色标签的纯文本字符串。
    解决 API 不支持 'student'/'teacher' 角色类型的问题。
    """
    if not history:
        return "（暂无对话历史）"

    # 截取最近的 N 轮，防止 Prompt 溢出
    recent_history = history[-max_turns:] if len(history) > max_turns else history
    
    formatted_lines = []
    
    # 定义角色映射表 (Role Map)
    role_map = {
        "student": "👨‍🎓 学生 (Student)",
        "user": "👨‍🎓 学生 (Student)",
        "teacher": "👩‍🏫 老师 (Teacher)",
        "assistant": "👩‍🏫 老师 (Teacher)",
        "system": "🖥️ 系统 (System)",
        "opening": "🎬 开场白 (Opening)"
    }

    for msg in recent_history:
        raw_role = msg.get("role", "unknown").lower()
        content = msg.get("content", "").strip()
        display_name = role_map.get(raw_role, raw_role.capitalize())
        formatted_lines.append(f"{display_name}: {content}")

    return "\n\n".join(formatted_lines)


# === [新增] 工具函数：解析带单位的数值 ===
def parse_quantity(value: Any, default: float = 0.0) -> Tuple[float, str]:
    """
    解析 "2.5 mL", "10g", "5.0" 这种格式。
    返回: (数值, 单位字符串小写)
    例如: "2 mL" -> (2.0, "ml"); "5g" -> (5.0, "g"); "10" -> (10.0, "")
    """
    if value is None:
        return default, ""
    
    s_val = str(value).strip().lower()
    
    # 1. 尝试直接转换
    try:
        return float(s_val), ""
    except ValueError:
        pass

    # 2. 正则提取数字部分
    match = re.match(r"([-+]?\d*\.?\d+)\s*([a-z]+)?", s_val)
    if match:
        num_str = match.group(1)
        unit_str = match.group(2) or ""
        try:
            return float(num_str), unit_str
        except ValueError:
            return default, unit_str
            
    return default, ""

class JSONFormatter(logging.Formatter):
    """
    [修复版] 将日志输出为 JSON 格式。
    增加了 default=str 参数，防止因对象无法序列化导致程序崩溃。
    """
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, 'data'):
            log_entry['data'] = record.data
        
        # 核心修复：添加 default=str
        # 遇到无法识别的类型（如 CompletionTokensDetails），强制转为字符串描述
        return json.dumps(log_entry, ensure_ascii=False, default=str)

# [新增] 全局日志队列，用于缓冲日志
log_queue = queue.Queue(-1)

def setup_advanced_logging(is_batch=False):
    """
    [高性能版] 异步日志系统
    批量模式下：关闭控制台打印，仅记录警告以上级别的日志，大幅减少 I/O。
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_logger = logging.getLogger()
    
    # 批量模式下提升根日志级别，减少处理开销
    root_logger.setLevel(logging.WARNING if is_batch else logging.INFO)
    root_logger.handlers = [] # 清理旧 Handler

    # 1. 准备后台写入 Handler
    active_handlers = []
    
    # 系统日志处理器 (始终保留以便排查严重崩溃)
    system_handler = logging.FileHandler(f"logs/system_{timestamp}.log", encoding='utf-8')
    system_handler.setFormatter(logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'))
    active_handlers.append(system_handler)

    if not is_batch:
        # 调试模式：开启对话细节日志和控制台打印
        dialogue_handler = logging.FileHandler(f"logs/dialogue_{timestamp}.jsonl", encoding='utf-8')
        dialogue_handler.setFormatter(JSONFormatter())
        dialogue_handler.setLevel(logging.DEBUG)
        active_handlers.append(dialogue_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        active_handlers.append(console_handler)
    
    # 2. 启动异步监听器 (QueueListener)
    # 它在后台线程中运行，主逻辑只需将日志丢进队列，毫秒级返回
    listener = logging.handlers.QueueListener(log_queue, *active_handlers, respect_handler_level=True)
    listener.start()

    # 3. 主进程挂载 QueueHandler
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)

    return logging.getLogger("Dialogue")

# ==========================================
# [新增] 全局日志上下文管理器
# ==========================================
class LogContext:
    """用于在不同模块间共享当前的轮次和步骤信息"""
    current_turn = 0
    current_step_index = 0
    current_step_desc = "Initializing"

def log_turn_event(logger_name: str, event_type: str, data: Dict[str, Any]):
    """
    全局通用的结构化日志记录函数。
    自动注入当前的 turn 和 step 信息，无需手动传递。
    """
    logger = logging.getLogger(logger_name)
    
    # 自动获取当前上下文
    payload = {
        'turn': LogContext.current_turn,
        'step_index': LogContext.current_step_index,
        'step_desc': LogContext.current_step_desc,
        'event_type': event_type, # 方便后续过滤
        **data # 合并传入的数据
    }
    
    # 统一使用 info 级别记录
    logger.info(event_type, extra={'data': payload})

# ==========================================
# 新模块: 实验配置注册表 (Configuration Registry)
# ==========================================
class ExperimentRegistry:
    def __init__(self, directory='experiments/'):
        self.directory = directory
        self.experiments = {}

    def load_xdl_experiment(self, filename: str):
        filepath = os.path.join(self.directory, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到实验文件: {filepath}")
        
        # 调用你的 XDLParser
        config = XDLParser.parse_xdl(filepath)
        
        # 将解析结果存入缓存，以 filename 为 ID
        exp_id = filename
        self.experiments[exp_id] = config
        return exp_id

    def get_experiment(self, exp_id: str):
        return self.experiments.get(exp_id)

# ==========================================
# 模块 B (简化版): 场景导演 (Scenario Director)
# ==========================================
class ScenarioDirector:
    """
    [简化版] 确定性场景导演
    职责：
    1. 不再调用 LLM，直接通过字符串拼接生成开场白。
    2. 确保环境状态归零（Clean Slate）。
    3. 设定标准的初学者人设。
    """
    def __init__(self, client, registry):
        # 这里的 client 目前不需要用到，但保留以维持接口一致性
        self.client = client
        self.registry = registry
        self.logger = logging.getLogger(self.__class__.__name__)

    def initialize_scenario(self, exp_id: str) -> Dict[str, Any]:
        """
        基于“器材 + 目标”的模板拼接，生成标准开场。
        """
        self.logger.info(f"Initializing Deterministic Scenario for {exp_id}")
        
        # 1. 获取实验配置
        exp_config = self.registry.get_experiment(exp_id)
        if not exp_config:
            self.logger.error(f"Experiment {exp_id} not found.")
            raise ValueError(f"实验 {exp_id} 不存在")
            
        # 2. 提取关键字段
        # 假设 equipment 是一个列表 ["试管", "烧杯", "酒精灯"]
        equipment_list = exp_config.get('equipment', [])
        equipment_str = "、".join(equipment_list) # 拼接成字符串
        
        title = exp_config.get('title', '该实验')
        goal = exp_config.get('goal', '完成实验')

        # 3. 【核心逻辑】字符串拼接生成开场白
        # 模板："老师，我看到了[器材清单]。我们要怎么[实验目标]？"
        opening_line = (
            f"老师，实验台上摆放了{equipment_str}。"
            f"请问我们要怎么利用这些器材来完成{title}（{goal}）呢？"
        )

        # 4. 设定固定的人设 (Standard Beginner)
        student_persona = (
            "你是一个初学化学的高中生。你非常听话，"
            "面对实验台上的器材，在没有得到老师明确指令前，你不敢随意乱动。"
            "你会先询问步骤，然后再执行。"
        )

        # 5. 构建返回包
        scenario_data = {
            "opening_line": opening_line,
            "state_patch": {}, # 空字典，代表环境无任何预设修改
            "student_specific_prompt": student_persona
        }
        
        self.logger.info(f"Generated Opening: {opening_line}")
        return scenario_data, exp_config

# ==========================================
# 自定义兼容 OpenAI v1.0 的 Embedding 函数
# ==========================================
class CustomOpenAIEmbeddingFunction(EmbeddingFunction):
    """
    自定义包装器，确保 ChromaDB 使用 OpenAI v1.x 的客户端语法
    """
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "text-embedding-3-small"):
        # 独立初始化一个 OpenAI 客户端，专门用于 Embeddings
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    @RETRY_RULE
    def __call__(self, input: Documents) -> Embeddings:
        input = [text.replace("\n", " ") for text in input]
        response = self.client.embeddings.create(
            input=input,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

# ==========================================
# 模块 E: 记忆端 (Chroma Vector Memory)
# ==========================================
class ExperienceMemory:
    """
    基于 ChromaDB 的向量记忆库 (RAG)。
    """
    def __init__(self, api_key: str, base_url: str = None, persist_path: str = "./chroma_db"):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 1. 初始化持久化客户端
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # 2. 使用我们自定义的 Embedding 函数 (修复报错的关键)
        self.embedding_fn = CustomOpenAIEmbeddingFunction(
            api_key=api_key,
            base_url=base_url, # 支持自定义 API 地址
            model_name="text-embedding-3-small"
        )
        
        # 3. 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="teaching_strategies",
            embedding_function=self.embedding_fn
        )
        
        # 4. 冷启动数据预埋
        if self.collection.count() == 0:
            self._seed_initial_knowledge()

    def _seed_initial_knowledge(self):
        """预埋初始专家知识库"""
        self.logger.info("Seeding initial knowledge into Vector DB...")
        
        initial_data = [
            {
                "situation": "学生忘记加指示剂，一直在滴定，溶液无变化。",
                "strategy": "不要直接提醒。问学生：'如果不加指示剂，肉眼能看到酸碱反应的终点吗？'"
            },
            {
                "situation": "学生滴定过快，溶液瞬间变深红（过量）。",
                "strategy": "引导反思。问：'现在的颜色说明溶液pH值大概是多少？这符合我们的预期吗？'"
            },
            {
                "situation": "学生不知道如何读数，视线没有平视。",
                "strategy": "示范正确的读数姿势，强调'凹液面最低点'与视线水平。"
            },
             {
                "situation": "铁钉没有打磨直接放入硫酸铜溶液，反应缓慢。",
                "strategy": "提示观察铁钉表面：'铁锈（氧化铁）会和硫酸铜反应吗？我们需要先做什么处理？'"
            }
        ]
        
        self.collection.add(
            documents=[item["situation"] for item in initial_data],
            metadatas=[{"strategy": item["strategy"]} for item in initial_data],
            ids=[str(uuid.uuid4()) for _ in range(len(initial_data))]
        )

    def retrieve_strategy(self, current_situation: str, n_results: int = 1) -> str:
        try:
            results = self.collection.query(
                query_texts=[current_situation],
                n_results=n_results
            )
            
            if results["documents"] and len(results["documents"][0]) > 0:
                best_strategy = results["metadatas"][0][0]["strategy"]
                matched_doc = results["documents"][0][0]
                dist = results["distances"][0][0]
                
                # 注意：Chroma 默认使用 L2 距离 (欧氏距离) 或 Cosine 距离
                # 距离越小越相似。对于 text-embedding-3-small，0.8-1.0 左右通常是相关性分界线
                if dist < 1.0: 
                    self.logger.info(f"RAG Hit! Query: '{current_situation}' matched: '{matched_doc}' (Dist: {dist:.4f})")
                    return f"【历史教学经验 (RAG)】: 针对类似情境（{matched_doc}），建议策略：{best_strategy}"
                else:
                    self.logger.info(f"RAG Miss. Distance too high ({dist:.4f})")
            
            return ""
        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {e}")
            return ""

    def add_experience(self, tag: str, strategy: str):
        self.logger.info(f"Learning new experience: {tag} -> {strategy}")
        self.collection.add(
            documents=[tag],
            metadatas=[{"strategy": strategy}],
            ids=[str(uuid.uuid4())]
        )


class Oracle:
    """
    [Dynamic Oracle] 动态真理持有者
    维护一个“理想世界”的物理引擎。
    能够基于输入的任意步骤（标准或补救），实时推演该步骤完成后的理想物理状态。
    """
    def __init__(self, exp_config: Dict[str, Any], sim_engine_class):
        self.config = exp_config
        self.procedure = exp_config.get("procedure", [])
        self.reagent_map = exp_config.get("reagent_map", {})
        
        # 保存类引用，用于创建沙盒
        self.SimClass = sim_engine_class
        self.hardware_config = exp_config["hardware"]

        # === 核心：理想世界引擎 (The Golden World) ===
        # 这个引擎代表了“当前实验进度下，完美的物理状态”
        # 它只在学生通过标准步骤时才更新 (commit)
        self.ideal_engine = self.SimClass(self.hardware_config, reagent_map=self.reagent_map)
        
        # [NEW] 历史存档：{step_index: snapshot_data}
        self.history: Dict[int, Dict] = {}
        
        # 初始状态快照 (作为 Step -1，即实验开始前的状态)
        self._sync_snapshot()
        self.history[-1] = {
            "containers": copy.deepcopy(self.chk_containers),
            "topology": copy.deepcopy(self.chk_topology)
        }

    def get_step_ideal_snapshot(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        [State Trap] 预演：返回该步骤如果正确执行后，世界应有的完整理想快照。
        """
        # 1. 创建沙盒环境，并从当前的“理想基准”开始
        sandbox = self.SimClass(self.hardware_config, reagent_map=self.reagent_map)
        sandbox.containers = copy.deepcopy(self.chk_containers)
        
        # 2. 同步拓扑结构
        sandbox.hw_manager.graph.clear()
        sandbox.hw_manager.add_hardware(list(sandbox.containers.keys()))
        for link in self.chk_topology:
            sandbox.hw_manager.attach(link['child'], link['parent'], link['type'])
            
        # 3. 在沙盒中执行该动作
        action = {"action": step_data["action"], **step_data.get("params", {})}
        sandbox.execute(action)
        
        # 4. 导出执行后的完整物理快照
        return {
            "containers": {k: v.get_snapshot() for k, v in sandbox.containers.items()},
            "topology": sandbox.hw_manager.get_topology_snapshot()
        }

    def _sync_snapshot(self):
        """保存理想引擎的当前状态快照，用于创建沙盒"""
        # 注意：这里假设 SimEngine 有 create_checkpoint 机制，或者我们直接深拷贝对象
        # 为了通用性，这里使用 deepcopy 整个容器字典和拓扑
        self.chk_containers = copy.deepcopy(self.ideal_engine.containers)
        # 拓扑结构比较复杂，假设 hw_manager 可以导出和导入
        self.chk_topology = self.ideal_engine.hw_manager.get_topology_snapshot()

    # [修改] Oracle 类内部
    def get_dynamic_goal(self, step_data: Dict[str, Any], current_actual_snapshot: Dict[str, Any] = None, base_snapshot: Dict[str, Any] = None) -> Optional[GoalState]:
        """
        [修正版] 动态目标推演：支持从指定基准快照推演
        """
        if not step_data: 
            return None

        # 1. 初始化沙盒
        sandbox = self.SimClass(self.hardware_config, reagent_map=self.reagent_map)
        
        # 2. 状态恢复核心逻辑
        # 优先使用显式传入的 base_snapshot (用于回溯推演)
        snapshot_source = base_snapshot
        
        # 如果没有 base_snapshot，且是补救模式，则使用 current_actual_snapshot
        if not snapshot_source and step_data.get("is_remedial") and current_actual_snapshot:
             snapshot_source = current_actual_snapshot
             
        if snapshot_source:
            # === [核心修复：从快照还原对象属性] ===
            snap_containers = snapshot_source.get("containers", {})
            for vid, snap_data in snap_containers.items():
                if vid in sandbox.containers:
                    target_obj = sandbox.containers[vid]
                    
                    # === [🔥核心修复] 类型兼容处理 ===
                    if isinstance(snap_data, dict):
                        # 情况A: 传入的是 get_snapshot() 返回的字典 (来自 current_actual_snapshot)
                        target_obj.volume = snap_data.get("volume_ml", 0.0)
                        target_obj.temperature = snap_data.get("temperature", 25.0)
                        target_obj.contents = copy.deepcopy(snap_data.get("contents", {}))
                    else:
                        # 情况B: 传入的是 Oracle.history 中的 Container 对象实例 (来自 clean_base)
                        # 直接读取对象属性
                        target_obj.volume = getattr(snap_data, "volume", 0.0)
                        target_obj.temperature = getattr(snap_data, "temperature", 25.0)
                        # 注意：对象中的 contents 也是字典，需深拷贝
                        target_obj.contents = copy.deepcopy(getattr(snap_data, "contents", {}))
            
            active_topo = snapshot_source.get("topology", [])
        else:
            # 主线模式：使用 Oracle 维护的理想状态
            sandbox.containers = copy.deepcopy(self.chk_containers)
            active_topo = self.chk_topology

        # 3. 同步拓扑
        sandbox.hw_manager.graph.clear()
        sandbox.hw_manager.add_hardware(list(sandbox.containers.keys()))
        for link in active_topo:
            sandbox.hw_manager.attach(link['child'], link['parent'], link['type'])

        # 4. 执行物理推演
        action = {"action": step_data["action"], **step_data.get("params", {})}
        target_vessel = action.get("vessel") or action.get("to_vessel") or action.get("from_vessel")
        
        pre_vol = 0.0
        if target_vessel and target_vessel in sandbox.containers:
            pre_vol = sandbox.containers[target_vessel].volume

        sandbox.execute(action)

        # 5. 生成 GoalState
        ideal_c = sandbox.containers.get(target_vessel)
        if not ideal_c:
            return None

        exp_vol = ideal_c.volume if ideal_c else 0.0
        ideal_delta = exp_vol - pre_vol 
        is_remedial = step_data.get("is_remedial", False)

        return GoalState(
            step_index=999,
            step_desc=step_data.get("desc", f"Action: {step_data.get('action')}"),
            target_vessel_id=target_vessel,
            expected_volume=exp_vol,
            expected_delta=ideal_delta,
            is_delta_check=is_remedial,
            expected_ph=ideal_c.ph if ideal_c else 7.0,
            expected_species={k for k, v in ideal_c.contents.items() if v > 1e-4} if ideal_c else set(),
            expected_moles={k: v for k, v in ideal_c.contents.items() if k not in ["H2O", "H+", "OH-"] and v > 1e-9} if ideal_c else {},
            expected_topology=[(item['child'], item['parent']) for item in sandbox.hw_manager.get_topology_snapshot()],
            tolerance=ToleranceLevel.LOOSE if action["action"] == "Wash" else ToleranceLevel.NORMAL
        )
    # [修改] commit_step：增加 step_index 参数并存档
    def commit_step(self, step_data: Dict[str, Any], step_index: int, actual_snapshot: Dict = None):
        """
        [状态锁定] 
        增加 actual_snapshot 参数，如果提供了实测快照，则直接同步，
        而不是在理想引擎中再次模拟执行（防止误差累积）。
        """
        # 如果是纯虚拟辅导（无物理动作），则不处理
        if step_data.get("is_virtual"):
            return

        # print(f"\n[Oracle] 🔒 Committing Step {step_index}...")

        # 如果提供了当前的物理快照，直接“承认”现实作为新的理想基准
        if actual_snapshot:
            # 从 snapshot 中还原 ideal_engine 的容器状态
            for vid, snap_data in actual_snapshot.get("containers", {}).items():
                if vid in self.ideal_engine.containers:
                    target_obj = self.ideal_engine.containers[vid]
                    target_obj.volume = snap_data.get("volume_ml", 0.0)
                    target_obj.contents = copy.deepcopy(snap_data.get("contents", {}))
                    target_obj.temperature = snap_data.get("temperature", 25.0)
        else:
            # 否则按原逻辑执行动作
            action = {"action": step_data["action"], **step_data.get("params", {})}
            self.ideal_engine.execute(action)
        
        # 同步快照供下一步预演使用
        self._sync_snapshot()
        self.history[step_index] = {
            "containers": copy.deepcopy(self.chk_containers),
            "topology": copy.deepcopy(self.chk_topology)
        }

    # [新增] 回滚方法
    def rollback_to(self, step_index: int):
        """
        [时光倒流] 将理想世界重置到 Step {step_index} *完成时* 的状态
        如果要重做 Step N，我们需要调用 rollback_to(N-1)
        """
        target_snapshot = self.history.get(step_index)
        if not target_snapshot:
            # print(f"[Oracle] ❌ Critical: Cannot rollback to step {step_index}. Snapshot missing.")
            return False

        # print(f"[Oracle] ⏪ Rolling back ideal world to end of Step {step_index}...")
        
        # 1. 恢复容器
        self.ideal_engine.containers = copy.deepcopy(target_snapshot["containers"])
        self.chk_containers = copy.deepcopy(target_snapshot["containers"])
        
        # 2. 恢复拓扑
        self.ideal_engine.hw_manager.graph.clear()
        self.ideal_engine.hw_manager.add_hardware(list(self.ideal_engine.containers.keys()))
        saved_topo = target_snapshot.get("topology", [])
        for link in saved_topo:
            self.ideal_engine.hw_manager.attach(link['child'], link['parent'], link['type'])
        
        self.chk_topology = saved_topo
        return True

    # === 兼容性接口 ===
    def get_total_steps(self) -> int:
        return len(self.procedure)

    def get_initial_steps(self) -> List[Dict[str, Any]]:
        return self.procedure
    
    def get_ground_truth_prompt(self) -> str:
        # (保持原样) 返回标准流程文本供 Teacher 参考
        steps_text = []
        for idx, step in enumerate(self.procedure):
            desc = step.get("desc", f"Step {idx+1}")
            steps_text.append(f"{idx + 1}. {desc}")
        full_text = "\n".join(steps_text)
        return f"【标准流程】:\n{full_text}"
# ==========================================
# 模块 A: 教学端 - DynamicPlanner (动态规划师)_scan_all_vessel_changes
# ==========================================
class DynamicPlanner:
    """
    动态规划师 (XDL版)：管理基于 XDL 的步骤队列。
    """
    # [新增] 获取当前阶段类型
    def get_phase_type(self) -> str:
        """
        判断当前处于主线还是支线。
        返回: "MAINLINE" | "REMEDIAL"
        """
        if self.current_step_index < len(self.steps):
            step = self.steps[self.current_step_index]
            # 检查标记位 (inject_remedial_plan 时插入的)
            if step.get("is_remedial") or step.get("is_virtual"):
                return "REMEDIAL"
        return "MAINLINE"

    # [新增] 跳转指针方法
    def jump_to(self, step_index: int):
        """强制跳转到指定步骤索引"""
        if 0 <= step_index < len(self.steps):
            self.current_step_index = step_index
            self.logger.warning(f"Planner jumped to step {step_index + 1}")

    def __init__(self, oracle: Oracle):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.oracle = oracle
        self.is_remedial_active = False
        self.remedy_target_snapshot = None # 存储补救的目标：主线理想态

        # 加载步骤并追加一个显式的结束步骤
        self.steps = oracle.get_initial_steps()
        self.steps.append({
            "action": "Finish",
            "desc": "实验结束：请老师进行总结评价。",
            "is_virtual": True
        })
        # 加载步骤对象列表
        self.steps = oracle.get_initial_steps() # List[Dict]
        self.current_step_index = 0
        self.logger.info(f"Planner initialized with {len(self.steps)} XDL steps.")

    def peek_next_step_data(self) -> Optional[Dict[str, Any]]:
        """
        [新增] 偷看下一步的数据，用于前瞻检查。
        如果不越界，返回下一步对象；否则返回 None。
        """
        next_idx = self.current_step_index + 1
        if next_idx < len(self.steps):
            return self.steps[next_idx]
        return None

    def get_current_plan(self) -> str:
        """返回当前步骤的自然语言描述 (给 Teacher/Student 看)"""
        if self.current_step_index < len(self.steps):
            step = self.steps[self.current_step_index]
            # 如果是虚拟步骤，直接返回内容
            if step.get("is_virtual"):
                return f"[辅导环节] {step.get('desc')}"
            
            # 如果是 XDL 步骤，返回标准化描述
            return f"Step {self.current_step_index + 1}: {step.get('desc')}"
        
        return "实验总结与反思阶段"

    def get_current_step_data(self) -> Dict[str, Any]:
        """[新增] 返回当前步骤的完整数据对象 (给 Reviewer 用)"""
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return {"action": "Finish", "desc": "实验结束", "params": {}}

    def advance(self):
        """推进到下一步"""
        if self.current_step_index < len(self.steps):
            # 获取刚完成的步骤
            finished_step = self.steps[self.current_step_index]
            self.logger.info(f"Step completed: {finished_step.get('desc')}")
            
            # 如果刚完成的是虚拟辅导步骤，移除它 (可选逻辑，或者保留在历史中)
            # 这里简单的逻辑是索引+1
            self.current_step_index += 1
            # print(Fore.BLUE + f"[Planner] Advancing to step {self.current_step_index + 1}")

    def replan(self, issue_description: str):
        """
        当陷入僵局 (STUCK) 时，动态插入一个虚拟步骤。
        """
        self.logger.warning(f"Replanning triggered: {issue_description}")
        # print(Fore.MAGENTA + f"[Planner] ⚠️ 检测到教学受阻，正在插入辅导步骤...")
        
        # 构造一个符合 XDL 列表结构的“虚拟步骤”
        remedial_step = {
            "action": "Guidance",
            "is_virtual": True, # 标记位，告诉 Oracle 这是生成的
            "desc": f"针对'{issue_description}'问题进行专项辅导与概念澄清。",
            "params": {}
        }
        
        # 将辅导步骤插入到当前索引位置 (让它成为下一个立刻执行的步骤)
        # 注意：不改变 index，而是改变列表内容
        self.steps.insert(self.current_step_index, remedial_step)
        
        return remedial_step.get("desc")
    
    # === [DynamicPlanner 类内部] ===

    # [修改] 注入补救计划
    def inject_remedial_plan(self, diagnosis: Dict[str, Any], ideal_snapshot: Dict[str, Any]) -> Tuple[bool, str]:
        """
        [v6.0 Restoration Edition] 补救计划注入器
        实现了自动化的“清洗-恢复-重做”链路。
        """

        self.remedy_target_snapshot = ideal_snapshot
        self.is_remedial_active = True
        remedy_type = diagnosis.get("type")
        vessel = diagnosis.get("target_vessel") or diagnosis.get("vessel")
        severity = diagnosis.get("severity", "LOW")
        
        if not vessel:
            curr_step = self.get_current_step_data()
            vessel = curr_step.get("params", {}).get("vessel") or curr_step.get("params", {}).get("to_vessel")
        
        new_steps = []

        
        
        # === 场景 2: 严重过量或试剂错误 -> 清洗并自动恢复现场 ===
        if remedy_type in ["WRONG_REAGENT", "OVERSHOOT", "WRONG_VESSEL", "IRREVERSIBLE_POLLUTION"]:
            # 1. 插入清洗步骤
            new_steps.append({
                "action": "Wash",
                "desc": f"【补救-第1步】由于操作失误，请先清洗 {vessel} 以防止污染实验。",
                "params": {"vessel": vessel},
                "is_remedial": True
            })

            # 2. 【核心功能】：寻找历史依赖并自动恢复
            # 扫描当前步骤之前的所有主线步骤，找到所有曾向此 vessel 添加过试剂的操作
            for i in range(self.current_step_index):
                hist_step = self.steps[i]
                # 排除掉之前的虚拟/补救步骤
                if hist_step.get("is_virtual") or hist_step.get("is_remedial"):
                    continue

                # === [FIX START] 动作类型白名单过滤 ===
                # 我们只恢复“加液”操作，不恢复“组装(Attach)”或“等待(Wait)”
                # 只有 Add 和 Transfer 会改变容器内的化学状态
                if hist_step.get("action") not in ["Add", "Transfer"]:
                    continue
                # === [FIX END] ===
                
                h_params = hist_step.get("params", {})
                if h_params.get("vessel") == vessel or h_params.get("to_vessel") == vessel:
                    # 创建恢复步骤：克隆原步骤但标记为补救模式
                    restoration = copy.deepcopy(hist_step)
                    restoration["is_remedial"] = True
                    restoration["desc"] = f"【补救-第2步:恢复】重新执行第 {i+1} 步：{hist_step.get('desc')}"
                    new_steps.append(restoration)

            # 倒序插入，保证执行顺序：Wash -> Restore 1 -> Restore 2 -> ... -> (回到当前失败步)
            for step in reversed(new_steps):
                self.steps.insert(self.current_step_index, step)
            
            return True, f"已生成‘清洗-恢复试剂’的完整补救链路（共 {len(new_steps)} 步）。"

        # === 场景 1: 量不足 (INSUFFICIENT) ===
        elif remedy_type == "INSUFFICIENT":
            diff = diagnosis.get("diff", 0.0)
            # 确保补加量精确且有效
            add_vol = round(abs(float(diff)), 1)
            if add_vol < 0.1: add_vol = 0.5 # 最小有效操作量

            reagent = diagnosis.get("target_reagent")
            curr_step = self.get_current_step_data()
            if not reagent:
                reagent = curr_step.get("params", {}).get("reagent", "所需试剂")

            # === [🔥 核心修复：防止步骤堆叠] ===
            # 如果当前已经在执行补救步骤，且试剂/动作一致，则直接更新它，而不是插入新的
            if curr_step.get("is_remedial") and curr_step.get("action") in ["Add", "Transfer"]:
                curr_step["params"]["volume"] = add_vol # 更新物理参数
                curr_step["desc"] = f"【补救】{vessel} 中 {reagent} 量仍不足，请继续补加约 {add_vol}ml。"
                return True, f"检测到补加仍未达标，已更新当前补救指令（需再加 {add_vol}ml）。"

            # 生成精准补加步骤
            act_type = curr_step.get("action")
            if act_type == "Transfer":
                src = curr_step.get("params", {}).get("from_vessel", "源容器")
                new_steps.append({
                    "action": "Transfer",
                    "desc": f"【补救】量不足，请从 {src} 继续滴加约 {add_vol}ml {reagent} 到 {vessel}。",
                    "params": {"from_vessel": src, "to_vessel": vessel, "volume": add_vol},
                    "is_remedial": True
                })
            else:
                new_steps.append({
                    "action": "Add",
                    "desc": f"【补救】{vessel} 中 {reagent} 量不足，请补加约 {add_vol}ml。",
                    "params": {"vessel": vessel, "reagent": reagent, "volume": add_vol},
                    "is_remedial": True
                })
            
            for step in reversed(new_steps):
                self.steps.insert(self.current_step_index, step)
            return True, f"已生成补加约 {add_vol}ml {reagent} 的指令。"
        
        # === 场景 3: 连接/位置错误 (TOPOLOGY_ERROR) ===
        elif remedy_type == "TOPOLOGY_ERROR":
            # 1. 获取需要纠正的部件 (Child)
            # 假设 Reviewer 在 diagnosis 中返回了涉及的 child (如 conductivity_meter)
            # 如果没返回，尝试从当前步骤参数中获取
            curr_step = self.get_current_step_data()
            child = curr_step.get("params", {}).get("tool") or curr_step.get("params", {}).get("vessel")
            
            target_parent = curr_step.get("params", {}).get("vessel") or curr_step.get("params", {}).get("support")

            if child:
                # 插入拆卸步骤
                new_steps.append({
                    "action": "Detach", # 假设 Engine 支持 Detach，或者用 Insert 到 'None' 模拟
                    "desc": f"【补救】检测到连接位置错误。请先将 {child} 取下。",
                    "params": {"object": child}, # 确保参数名与 Engine 兼容
                    "is_remedial": True
                })
                
                # 插入正确的重连步骤
                new_steps.append({
                    "action": "Insert", # 或 Attach
                    "desc": f"【补救】请重新将 {child} 插入到正确的目标容器 {target_parent} 中。",
                    "params": {"tool": child, "vessel": target_parent},
                    "is_remedial": True
                })

                # 倒序插入
                for step in reversed(new_steps):
                    self.steps.insert(self.current_step_index, step)
                
                return True, f"已生成‘拆卸-重装’的补救指令。"


        return False, f"未知的诊断类型: {remedy_type}"
    
    def try_exit_remedy(self):
        """
        [State Trap Exit] 强行清除所有补救步骤，回归主线。
        [Final Fix] 向后搜索最近的主线步骤作为锚点。
        """
        self.logger.info("✨ State Trap Triggered: Exiting remedial mode.")
        
        target_mainline_step = None
        
        # 1. 【核心逻辑】在脏列表中，向后寻找最近的一个主线步骤
        # 因为补救步骤是插入在主线步骤之前的，所以主线步骤一定在当前指针或其后面
        for i in range(self.current_step_index, len(self.steps)):
            step = self.steps[i]
            # 找到第一个不是补救步骤，且不是虚拟步骤的步骤（即原始 XDL 步骤）
            # 或者如果是 finish 步骤也算
            if not step.get("is_remedial"):
                target_mainline_step = step
                break
        
        # 2. 移除所有补救步骤（清洗列表）
        self.steps = [s for s in self.steps if not s.get("is_remedial")]
        
        # 3. 【索引重校准】在干净列表中找到该主线步骤
        if target_mainline_step and target_mainline_step in self.steps:
            new_index = self.steps.index(target_mainline_step)
            self.logger.info(f"Index recalibrated to Mainline Step: {new_index} ({target_mainline_step.get('desc')})")
            
            # 这里的逻辑是： Reviewer 认为物理状态已经满足了 target_mainline_step 的目标
            # 所以我们需要把指针指在它身上，让随后的 advance() 将其跳过，直接进入下一步
            self.current_step_index = new_index
            
        else:
            # 只有当找不到任何主线步骤时（极罕见），才保持原样或报错
            self.logger.error("Critical: Could not locate original mainline step after remedy exit.")
            # 此时不要重置为 0，尽量保持在当前长度的末尾防止崩溃
            self.current_step_index = min(self.current_step_index, len(self.steps) - 1)

        # 4. 重置状态位
        self.remedy_target_snapshot = None
        self.is_remedial_active = False
# ==========================================
# 1. 动态知识库加载器 (Knowledge Loader)
# ==========================================
class ChemistryDatabase:
    """
    [修正版] 动态知识库加载器
    修复了数据库文件缺失时导致系统崩溃或产生幻觉数据的风险。
    现在包含一个硬编码的“最小可用数据集”作为兜底。
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChemistryDatabase, cls).__new__(cls)
            cls._instance.substances = {}
            cls._instance.reactions = []
            cls._instance.load_data()
        return cls._instance
    
    # [修改] ChemistryDatabase 类内部
    def get_components(self, chem_name: str, amount: float, is_solid: bool = False, molarity: float = None) -> Dict[str, float]:
        """
        [通用版] 计算摩尔数
        :param amount: 如果是液体为 mL，如果是固体为 g
        :param is_solid: 是否为固体添加
        """
        info = self.substances.get(chem_name, {})
        molar_mass = info.get("molar_mass", 1.0) # 防止除零
        # [FIX] 强制兜底：如果数据库里写了 null，强行设为 1.0，防止除以 None
        if molar_mass is None or molar_mass <= 0:
            molar_mass = 1.0
        real_formula = chem_name
        # 简单映射处理
        if "HCl" in chem_name: real_formula = "HCl"
        if "NaOH" in chem_name: real_formula = "NaOH"

        total_moles = 0.0

        if is_solid:
            # === 固体逻辑: n = m / M ===
            # amount 单位是 g
            total_moles = amount / molar_mass
        else:
            # === 液体逻辑: n = C * V ===
            # amount 单位是 mL
            if molarity is None:
                molarity = info.get("default_molarity", 0.1)
                if chem_name == "H2O": molarity = 55.5
            total_moles = molarity * (amount / 1000.0)

        return {real_formula: total_moles}

    def load_data(self):
        """
        尝试从文件加载，如果失败则使用硬编码的兜底数据。
        """
        try:
            # 1. 检查文件是否存在
            if not os.path.exists('database/substances.json') or not os.path.exists('database/reactions.json'):
                raise FileNotFoundError("Database files missing.")

            # 2. 加载物质库
            with open('database/substances.json', 'r', encoding='utf-8') as f:
                self.substances = json.load(f)
                # RGB 格式预处理
                for k, v in self.substances.items():
                    if "color_rgb" in v and v["color_rgb"]:
                        v["color_rgb"] = tuple(v["color_rgb"])

            # 3. 加载反应库
            with open('database/reactions.json', 'r', encoding='utf-8') as f:
                self.reactions = json.load(f)
                
            # print(Fore.GREEN + f"[Database] Successfully loaded {len(self.substances)} substances and {len(self.reactions)} reactions.")

        except Exception as e:
            # [Fix] 捕获所有异常，切换到兜底模式
            # print(Fore.RED + f"[Database] Critical Error loading files: {e}")
            # print(Fore.YELLOW + "[Database] Switching to Fallback Mode (Hardcoded Data)...")
            self._load_fallback_data()

    def _load_fallback_data(self):
        """
        [纯分子版] 数据库
        """
        # 1. 物质定义：不再拆分离子，增加 acidity/basicity 标记
        self.substances = {
            # === 试剂 ===
            "HCl": {
                "molar_mass": 36.46,
                "state": "aq",
                "type": "acid",      # 标记：酸
                "strength": 1,       # 强酸 (用于 pH 计算)
                "valence": 1         # 一元酸
            },
            "NaOH": {
                "molar_mass": 39.99,
                "state": "aq",
                "type": "base",      # 标记：碱
                "strength": 1,       # 强碱
                "valence": 1
            },
            "Unknown_HCl": {         # 兼容 XDL
                "molar_mass": 36.46,
                "state": "aq",
                "type": "acid",
                "strength": 1,
                "valence": 1
            },
            "Standard_NaOH": {       # 兼容 XDL
                "molar_mass": 39.99,
                "state": "aq",
                "type": "base",
                "strength": 1,
                "valence": 1
            },
            # === 指示剂 ===
            "Phenolphthalein": {
                "molar_mass": 318.32,
                "state": "aq",
                "is_indicator": True, # 标记为指示剂
                "pKa": 9.3,           # 变色点
                "acid_color": [255, 255, 255, 0.0], # 无色
                "base_color": [255, 20, 147, 0.8]   # 粉红
            },
            # === 产物 ===
            "NaCl": {
                "molar_mass": 58.44,
                "state": "aq",
                "type": "salt"
            },
            "H2O": {
                "molar_mass": 18.01,
                "state": "l",
                "type": "solvent"
            }
        }

        # 2. 反应定义：全部使用 chemlib 字符串方程式
        self.reactions = [
            {
                # 只有当容器里存在 HCl 和 NaOH 分子时才触发
                "equation": "HCl + NaOH -> NaCl + H2O",
                "exothermic": 57.3,
                "phenomena": "放热反应"
            },
            # 兼容 Unknown_HCl 的写法 (如果 chemlib 识别不了自定义名字，见下文 ChemSolver 的处理)
            # 或者我们在 Add 的时候就做好了映射，这里只写标准式
        ]
        # print(Fore.YELLOW + f"[Database] Molecular Mode initialized.")

class ChemSolver:
    @staticmethod
    def parse_and_solve(container_contents: Dict[str, float], equation_str: str) -> Dict[str, float]:
        """
        利用 chemlib 处理分子反应
        """
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

# 初始化单例
CHEM_DB = ChemistryDatabase()

# 兼容旧代码的别名 (这一步很重要，让你不用改下面大量的代码)
SUBSTANCES_DB = CHEM_DB.substances
REACTIONS_DB = CHEM_DB.reactions
# ==========================================
# 2. 容器类 (Container v3.1 - Enhanced Rendering)
# ==========================================
class Container:
    def __init__(self, name: str, capacity: float = 500.0):
        self.name = name
        self.capacity = capacity
        
        # 核心状态
        self.contents: Dict[str, float] = {} 
        self.volume: float = 0.0      
        self.temperature: float = 25.0 
        self.pressure: float = 1.0    
        self.is_sealed: bool = False 
        # [删除] self.is_source = False  <-- 这一行删掉

    def get_brief_desc(self) -> str:
        """返回容器的简要状态，用于上下文消歧义。"""
        if self.volume <= 0.1:
            return "Empty"
        
        if not self.contents:
            return f"{self.volume:.1f}ml solvent"
            
        major_chem = max(self.contents, key=self.contents.get)
        return f"{self.volume:.1f}ml, contains {major_chem}..."

    def add_chemical(self, chem_name: str, moles: float, vol_ml: float, fluid_temp: float = 25.0):
        """
        [修正版] 物理添加物质
        :param fluid_temp: 传入液体的温度 (默认为室温 25.0)
        """
        if vol_ml + self.volume > self.capacity:
            logging.warning(f"Vessel {self.name} overflowed!")
            
        # 1. 更新化学成分
        self.contents[chem_name] = self.contents.get(chem_name, 0) + moles
        
        # 2. 热力学混合计算 (Q = cmΔt 近似模型)
        # 在更新体积之前，先计算混合后的温度
        if self.volume > 0 or vol_ml > 0:
            current_mass = self.volume  # 简化假设密度为 1
            added_mass = vol_ml
            total_mass = current_mass + added_mass
            
            if total_mass > 0:
                # 加权平均温度公式: (m1*t1 + m2*t2) / (m1+m2)
                # [Fix]: 使用传入的 fluid_temp，而不是硬编码的 25.0
                total_heat = (current_mass * self.temperature) + (added_mass * fluid_temp)
                self.temperature = total_heat / total_mass

        # 3. 更新体积
        self.volume += vol_ml

    # 在 Container 类中修改 transfer_to 方法

    def transfer_to(self, target: 'Container', transfer_vol_ml: float):
        """物理转移溶液 (增强版：支持气体/固体倾倒)"""
        
        # === 场景 A: 液体转移 (原逻辑) ===
        if self.volume > 1e-6:
            # 计算转移比例
            real_vol = min(self.volume, transfer_vol_ml)
            ratio = real_vol / self.volume
            
            # 1. 转移溶质
            chemicals_to_remove = []
            for chem, amount in self.contents.items():
                transferred_moles = amount * ratio
                target.add_chemical(chem, transferred_moles, 0) 
                self.contents[chem] -= transferred_moles
                if self.contents[chem] <= 1e-9:
                    chemicals_to_remove.append(chem)
            for c in chemicals_to_remove:
                del self.contents[c]
                
            # 2. 转移体积与热量
            target_old_vol = target.volume
            target.volume += real_vol
            total_vol = target.volume
            if total_vol > 0:
                target.temperature = (target_old_vol * target.temperature + real_vol * self.temperature) / total_vol
            self.volume -= real_vol

        # === [新增] 场景 B: 干转移 (气体/固体) ===
        # 如果没有液体体积，但有内容物 (contents不为空)，说明是气体或固体
        elif self.contents:
            # 模拟“全部倒出” (因为气体/固体很难像液体一样按ml精确分割倒出，这里简化为全量转移)
            # 这符合 "Add 100ml -> Transfer" 的操作流
            for chem, moles in self.contents.items():
                target.add_chemical(chem, moles, 0.0, fluid_temp=self.temperature)
            
            # 清空源容器
            self.contents.clear()

    @property
    def ph(self) -> float:
        """
        [分子版] pH 估算逻辑
        遍历所有内容物，根据 type="acid"/"base" 累加 H+/OH- 浓度
        """
        if self.volume <= 1e-6: return 7.0
        
        vol_L = self.volume / 1000.0
        
        total_h_moles = 0.0
        total_oh_moles = 0.0
        
        # 遍历容器内的所有分子
        for chem, moles in self.contents.items():
            info = SUBSTANCES_DB.get(chem, {})
            c_type = info.get("type")
            # === [修复] 安全获取化合价 ===
            valence = info.get("valence")
            if valence is None: 
                valence = 1 # 默认一元
            
            strength = info.get("strength", 0)
            
            # 简单模型：只考虑强酸强碱完全电离
            # 如果要支持弱酸，需要解方程，这里简化处理
            if c_type == "acid" and strength == 1:
                total_h_moles += moles * valence
            elif c_type == "base" and strength == 1:
                total_oh_moles += moles * valence
        
        # 计算净浓度
        net_h = (total_h_moles - total_oh_moles) / vol_L
        
        # 中性
        if abs(net_h) < 1e-9:
            return 7.0
            
        # 酸性
        if net_h > 0:
            return -math.log10(net_h)
        # 碱性
        else:
            poh = -math.log10(abs(net_h))
            return 14.0 - poh

    @property
    def precipitate_mass(self) -> float:
        """计算固体总质量"""
        total = 0.0
        for name, moles in self.contents.items():
            info = SUBSTANCES_DB.get(name, {})
            
            # 检查是否为固体或沉淀
            is_solid = (info.get("state") == "s" or name.endswith("_ppt"))
            
            if is_solid:
                # === [修复] 安全获取摩尔质量 ===
                mm = info.get("molar_mass")
                # 如果数据库里是 null 或未定义，默认为 0，防止报错
                if mm is None: 
                    mm = 0.0
                
                total += moles * mm
        return total

    # ========================================================
    # 新增核心代码：高级颜色渲染 (Phase 1.3)
    # ========================================================
    
    def _sigmoid(self, x: float) -> float:
        """
        S型函数：将输入映射到 0.0 ~ 1.0
        用于模拟从酸式态到碱式态的平滑突变
        """
        if x < -10: return 0.0
        if x > 10: return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def _get_indicator_color(self, name: str, moles: float, current_ph: float) -> tuple:
        """
        根据 Henderson-Hasselbalch 原理计算指示剂当前颜色
        返回: (r, g, b, alpha_intensity)
        """
        info = SUBSTANCES_DB.get(name, {})
        
        # 1. 获取基本参数 (带默认值兜底)
        pKa = info.get("pKa", 7.0)
        acid_rgb = info.get("acid_color", (255, 255, 255, 0.0))
        base_rgb = info.get("base_color", (255, 255, 255, 0.0))
        width = info.get("transition_width", 2.0)

        # 2. 计算碱式组分比例 (Alpha Ratio)
        # steepness 控制 S 曲线的陡峭程度
        steepness = 4.0 / max(width, 0.1) 
        x = (current_ph - pKa) * steepness
        ratio = self._sigmoid(x)

        # 3. 颜色插值 (Linear Interpolation)
        r = acid_rgb[0] * (1 - ratio) + base_rgb[0] * ratio
        g = acid_rgb[1] * (1 - ratio) + base_rgb[1] * ratio
        b = acid_rgb[2] * (1 - ratio) + base_rgb[2] * ratio
        
        # 4. 计算强度 (Intensity)
        # 基础透明度插值
        base_alpha = acid_rgb[3] * (1 - ratio) + base_rgb[3] * ratio
        
        # 结合浓度 (Beer-Lambert Law 近似)
        vol_L = self.volume / 1000.0 if self.volume > 0 else 1.0
        concentration = moles / vol_L
        # 使用 1 - exp(-k * c) 模拟颜色饱和，防止浓度越高颜色越深到离谱
        intensity = 1.0 - math.exp(-50.0 * concentration) 
        final_alpha = base_alpha * intensity
        
        return (r, g, b, final_alpha)

    @property
    def color_description(self) -> str:
        """
        [v3.1 Upgrade] 混合渲染引擎：支持指示剂突变 + 普通离子显色
        """
        if self.volume <= 0.1 and self.precipitate_mass <= 0: return "空"

        mixed_r, mixed_g, mixed_b = 0.0, 0.0, 0.0
        total_weight = 0.0
        max_alpha = 0.0 # 记录最强的一股颜色的强度，用于判断深浅

        # --- 1. 遍历计算混合色 ---
        for name, moles in self.contents.items():
            info = SUBSTANCES_DB.get(name, {})
            current_rgba = None
            
            # 分支 A: 指示剂 (Dynamic Color)
            if info.get("is_indicator"):
                current_rgba = self._get_indicator_color(name, moles, self.ph)
            
            # 分支 B: 普通有色物质 (Static Color)
            elif info.get("color_rgb"):
                base_rgb = info.get("color_rgb") # (r, g, b, alpha)
                vol_L = self.volume / 1000.0 if self.volume > 0 else 1.0
                conc = moles / vol_L
                # 简单估算强度
                alpha = base_rgb[3] * conc * 100.0
                current_rgba = (base_rgb[0], base_rgb[1], base_rgb[2], alpha)

            # 累加颜色向量
            if current_rgba:
                r, g, b, alpha = current_rgba
                if alpha > 0.01: # 忽略太淡的，减少噪音
                    mixed_r += r * alpha
                    mixed_g += g * alpha
                    mixed_b += b * alpha
                    total_weight += alpha
                    max_alpha = max(max_alpha, alpha)

        # --- 2. 沉淀影响 ---
        has_ppt = self.precipitate_mass > 0.01
        ppt_desc = "浑浊液" if has_ppt else "液体"

        # --- 3. 最终判定 ---
        # 如果所有色素加起来都很淡
        # [修复] 提高颜色判定阈值，防止微量试剂渲染出明显的“白色”
        if total_weight < 0.5: 
            return f"白色{ppt_desc}" if has_ppt else "无色透明液体"

        # 归一化 (Normalization)
        final_r = mixed_r / total_weight
        final_g = mixed_g / total_weight
        final_b = mixed_b / total_weight

        color_name = self._rgb_to_name(final_r, final_g, final_b)
        
        # 强度修饰语
        depth = ""
        if max_alpha < 0.5: depth = "微"
        elif max_alpha < 2.0: depth = "浅"
        elif max_alpha > 12.0: depth = "深" # 提高“深色”判定门槛
        
        # [Visual Hack] 特殊逻辑：针对滴定终点的“微粉色”体验
        # 如果计算出是粉色，且强度很低，这正是滴定终点的特征
        if "粉" in color_name and 0.1 < max_alpha < 0.8:
            return f"微粉色{ppt_desc} (30秒不褪色)"

        return f"{depth}{color_name}{ppt_desc}"

    def _rgb_to_name(self, r, g, b):
        """简单的最近邻颜色分类"""
        colors = {
            "红": (255, 0, 0), "蓝": (0, 0, 255), "黄": (255, 255, 0),
            "绿": (0, 128, 0), "紫": (128, 0, 128), "粉": (255, 192, 203),
            "浅粉": (255, 220, 230), # <--- [新增] 专门捕捉滴定终点
            "白": (255, 255, 255), "黑": (0, 0, 0)
        }
        best_name = "无色"
        min_dist = float('inf')
        
        for name, (cr, cg, cb) in colors.items():
            dist = (r-cr)**2 + (g-cg)**2 + (b-cb)**2
            if dist < min_dist:
                min_dist = dist
                best_name = name

        if best_name == "浅粉":
            return "微粉色"
        return best_name

    # [修改] Container.get_snapshot
    def get_snapshot(self) -> Dict[str, Any]:
        """返回给 Reviewer 的结构化数据"""
        return {
            "ph": round(self.ph, 2),
            "temperature": round(self.temperature, 1),
            "volume_ml": round(self.volume, 1),
            "color_desc": self.color_description,
            "precipitate_g": round(self.precipitate_mass, 3),
            # [关键修复] 将阈值从 1e-3 降低到 1e-6，确保微量试剂可见
            "major_species": [k for k, v in self.contents.items() if v >= 1e-6],
            "contents": copy.deepcopy(self.contents) # [新增] 暴露完整组分，供 Inspector 做定量检查
        }
# ==========================================
# 3. 反应管理器 (Reaction Solver)
# ==========================================
class ReactionManager:
    @staticmethod
    def solve(container: Container) -> List[str]:
        phenomena_report = []
        
        # 简单的迭代，处理连锁反应
        for _ in range(5): 
            reaction_happened = False
            
            for reaction in REACTIONS_DB:
                # 只处理带 equation 字段的反应
                if "equation" not in reaction: continue
                
                eq_str = reaction["equation"]
                
                # 调用 chemlib 求解
                changes, runs = ChemSolver.parse_and_solve(container.contents, eq_str)
                
                if runs > 0:
                    # 应用变化
                    for chem, delta in changes.items():
                        container.contents[chem] = container.contents.get(chem, 0) + delta
                        if container.contents[chem] <= 1e-9:
                            del container.contents[chem]
                    
                    # 热力学 (简单估算)
                    if "exothermic" in reaction:
                        # === [修复] 安全获取放热值 ===
                        exothermic_val = reaction.get("exothermic")
                        if exothermic_val is not None:
                            heat_kj = runs * exothermic_val
                            # 假设主要是水，比热容 4.18
                            total_vol = container.volume 
                            if total_vol > 0:
                                container.temperature += (heat_kj * 1000) / (total_vol * 4.18)
                            
                    if "phenomena" in reaction:
                        phenomena_report.append(reaction["phenomena"])
                        
                    reaction_happened = True
                    # 如果发生反应，跳出内层循环，重新扫描所有可能的新反应
                    break 
            
            if not reaction_happened:
                break
                
        return list(set(phenomena_report))

# ==========================================
# 4. 硬件管理器 (Hardware Topology)
# ==========================================
# ==========================================
# 4. 硬件管理器 (Hardware Topology - Graph Version)
# ==========================================
class HardwareManager:
    def __init__(self):
        # 使用有向图：节点是硬件ID，边是 "Child -> Parent" (依赖关系)
        # 允许一个节点指向多个 Parent (一子多父)，边属性区分连接类型
        self.graph = nx.DiGraph()

    def add_hardware(self, hardware_ids: List[str]):
        """初始化时注册所有硬件节点"""
        self.graph.add_nodes_from(hardware_ids)

    def attach(self, child: str, parent: str, connection_type: str = "fixed") -> Tuple[bool, str]:
        # 1. 基础检查
        if not self.graph.has_node(child): self.graph.add_node(child)
        if not self.graph.has_node(parent): self.graph.add_node(parent)

        if child == parent:
            return False, "无法将物体安装在自己身上。"

        # === [🔥 核心修改：完全放开连接限制] ===
        # 移除所有“自动断开”逻辑。
        # 移除所有“双端设备白名单”检查。
        # 理由：信任 Agent 的常识。如果学生做出了违背物理（如单口连双管）的操作，
        # 应由后续的 Reviewer 或 Teacher 指出，而不是引擎层报错。

        # 唯一保留：幂等性检查 (防止重复添加完全一样的边)
        if self.graph.has_edge(child, parent):
             if self.graph[child][parent].get('type') == connection_type:
                 return False, f"{child} 已经在 {parent} 上了。"

        # 3. 直接建立连接
        self.graph.add_edge(child, parent, type=connection_type)

        # 4. 环路检测 (物理死锁依然是底线，必须防止 A->B->A)
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(child, parent) # 回滚
            return False, "操作失败：该连接会导致物理死锁（循环依赖）。"

        return True, f"成功将 {child} 连接到 {parent}。"

    def detach(self, child: str, parent: str = None) -> Tuple[bool, str]:
        """
        拆卸设备。
        如果指定 parent，只断开那条边；否则断开该物体所有的连接。
        """
        if not self.graph.has_node(child):
            return False, f"找不到设备 {child}。"

        out_edges = list(self.graph.out_edges(child)) # List of (u, v)
        
        if not out_edges:
            return False, f"{child} 目前是自由状态，没有被固定。"

        removed_parents = []
        
        if parent:
            # 移除指定连接
            if self.graph.has_edge(child, parent):
                self.graph.remove_edge(child, parent)
                return True, f"已将 {child} 从 {parent} 上拆下。"
            else:
                return False, f"{child} 并没有连接在 {parent} 上。"
        else:
            # 移除所有连接 (默认行为)
            for u, v in out_edges:
                self.graph.remove_edge(u, v)
                removed_parents.append(v)
            return True, f"已解除 {child} 的所有连接 (曾连接: {', '.join(removed_parents)})。"

    def get_topology_snapshot(self) -> List[Dict[str, str]]:
        """
        生成用于日志和 Reviewer 的拓扑快照。
        格式: List of {child, parent, type}
        """
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "child": u,
                "parent": v,
                "type": data.get("type", "fixed")
            })
        return edges

    def get_connected_parents(self, child: str) -> List[str]:
        """获取某物体的所有父级（用于液体流向计算）"""
        if self.graph.has_node(child):
            return [v for u, v in self.graph.out_edges(child)]
        return []

# ==========================================
# 5. 主引擎 (ChemSimEngine)
# ==========================================
class ChemSimEngine:
    # 1. 修改初始化函数，接收 reagent_map
    def __init__(self, hardware_config: List[Dict], reagent_map: Dict[str, str] = None):
        self.containers: Dict[str, Container] = {}
        self.hw_manager = HardwareManager()
        self.logger = logging.getLogger("ChemSimEngine")
        self.checkpoints: Dict[str, Any] = {}
        
        # [新增] 存储名称映射表 {"Unknown_HCl": "HCl"}
        self.reagent_map = reagent_map or {} 
        
        # 初始化硬件
        self._init_hardware(hardware_config)

    # [新增] 辅助函数：解析化学式
    def _resolve_chem_name(self, name: str) -> str:
        """将实验变量名(Unknown_HCl) 转换为 数据库键名(HCl)"""
        if not name: return "H2O" # 默认防爆
        # 优先查表，查不到就用原名
        return self.reagent_map.get(name, name)

    def _init_hardware(self, hardware_config: List[Dict]):
        """
        [简化版] 硬件初始化
        不再实体化试剂瓶，只初始化实验器皿。
        """
        self.logger.info("=== Initializing Hardware ===")
        
        for item in hardware_config:
            vid = item['id']
            
            # 1. 解析容量 (Capacity)
            raw_cap = str(item.get('capacity', '500')).lower().replace('ml', '').strip()
            try:
                capacity = float(raw_cap)
            except ValueError:
                capacity = 500.0 
            
            # 2. 直接创建空容器
            self.containers[vid] = Container(vid, capacity=capacity)
            # [删除] self.containers[vid].is_source = True 以及相关的 10000ml 初始化代码
            
            self.logger.info(f" -> Initialized container [{vid}] (Capacity: {capacity}ml)")

        self.logger.info("=== Hardware Initialization Complete ===\n")
    def create_checkpoint(self, name: str = "latest"):
        """
        保存当前物理世界的完整状态（深拷贝）
        """
        self.logger.info(f"Saving checkpoint: {name}")
        try:
            # Deepcopy 是必须的，否则只会复制引用，由于 Container 是可变对象，
            # 后续修改会污染存档。
            snapshot = {
                "containers": copy.deepcopy(self.containers),
                "attachments": copy.deepcopy(self.hw_manager.get_topology_snapshot())
            }
            self.checkpoints[name] = snapshot
            # # print(Fore.MAGENTA + f"[System] 自动存档完成: {name}") # 调试用，可注释
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """[FIX] 安全转换浮点数，处理 NoneType 报错"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def execute(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        [修复版] 执行操作并返回结果。
        增加了 Refill 的安全校验，防止死循环。
        """
        act_type = action.get("action")
        obs_text = []
        
        # 获取当前快照用于返回（防止执行一半出错没返回值）
        current_snapshot = {
            "containers": {k: v.get_snapshot() for k, v in self.containers.items()},
            "topology": self.hw_manager.get_topology_snapshot()
        }

        try:
            # === [Level 1] 全局参数防御 ===
            if act_type in ["Add", "Heat", "Attach", "Refill"] and not action.get("vessel"):
                return f"操作失败：动作 '{act_type}' 缺少指定容器。", current_snapshot
            
            if act_type == "Transfer" and (not action.get("from_vessel") or not action.get("to_vessel")):
                return "操作失败：转移操作必须明确‘从哪里’转移‘到哪里’。", current_snapshot

            # === [Level 2] 具体动作执行 ===

            if act_type == "Wash":
                vessel_id = action.get("vessel")
                target = self.containers.get(vessel_id)
                if not target:
                    return f"操作失败：找不到器材 {vessel_id}", current_snapshot
                
                # 1. 物理清洗
                target.contents = {} 
                target.volume = 0.0
                target.temperature = 25.0
                
                # 2. 模拟挂壁残留
                # target.add_chemical("H2O", 0.001, 0.5) 
                obs_text.append(f"学生将 {vessel_id} 拿到水槽进行了彻底清洗。现在它是干净的（内壁微湿）。")

            elif act_type == "Refill":
                vessel_id = action.get("vessel")
                reagent_name = action.get("reagent")
                
                target = self.containers.get(vessel_id)
                if not target: 
                    return f"操作失败：找不到容器 {vessel_id}", current_snapshot

                # [修改] 不再检查 target.is_source，而是检查是否指定了药品
                if not reagent_name: 
                    return f"操作失败：Refill 必须指定试剂名称。", current_snapshot
                
                real_formula = self._resolve_chem_name(reagent_name)
                
                # [可选] 这里可以加一个逻辑：只允许 Refill 滴定管 (burette)
                # if "burette" not in vessel_id.lower():
                #    return "只有滴定管可以使用 Refill 操作，普通容器请使用 Add。", current_snapshot

                # 使用 real_formula 查库
                if real_formula not in CHEM_DB.substances:
                    return f"系统错误：数据库缺失物质 '{real_formula}'。", current_snapshot

                # 默认填满或者填 50ml
                refill_vol = target.capacity if target.capacity < 1000 else 50.0
                
                # 清空并重填
                target.contents = {}
                target.volume = 0.0
                target.temperature = 25.0
                
                components = CHEM_DB.get_components(real_formula, refill_vol)
                for ion, moles in components.items():
                    target.add_chemical(ion, moles, refill_vol if ion == list(components.keys())[0] else 0)
                
                obs_text.append(f"【系统】已将 {vessel_id} 装满 {reagent_name} ({refill_vol}ml)。")

            # -------------------------------------------------------
            # [修复] 针对 Attach 和 Insert 的空值防御
            # -------------------------------------------------------
            # === [Modified] Attach 动作 (回归纯粹) ===
            elif act_type == "Attach":
                child = action.get("vessel")
                support = action.get("support")
                
                # 不再需要处理 tool 参数了，逻辑简化
                if not child or not support:
                    return "操作失败...", current_snapshot
                
                # 默认就是夹持 (clamped_on)
                success, msg = self.hw_manager.attach(child, support, connection_type="clamped_on")
                obs_text.append(msg)

            # === [NEW] 独立的 Insert 动作 ===
            elif act_type == "Insert":
                # 获取参数 (兼容 XDL 里的 object/tool 和 target/vessel 写法)
                child = action.get("object") or action.get("tool")
                parent = action.get("target") or action.get("vessel")

                if not child or not parent:
                    return f"操作失败：Insert 动作缺少对象或目标。", current_snapshot
                
                # 执行物理连接
                # connection_type="sealed_in" 表示塞紧密封
                success, msg = self.hw_manager.attach(child, parent, connection_type="sealed_in")
                
                if success:
                    obs_text.append(f"成功将 {child} 塞入 {parent}。")
                else:
                    obs_text.append(f"操作失败：{msg}")
            elif act_type == "Detach":
                # 兼容 XDL 各种叫法
                child = action.get("object") or action.get("tool") or action.get("vessel")
                # 支持指定从哪里拆（可选）
                parent = action.get("support") or action.get("from_object")

                if not child:
                    return "操作失败：Detach 动作必须指定要拆卸的对象 (object)。", current_snapshot
                
                # 调用 HardwareManager 的新版 detach
                # 注意：此时 HardwareManager.detach 应该已经包含了我们之前讨论的“歧义保护”逻辑
                success, msg = self.hw_manager.detach(child, parent)
                obs_text.append(msg)

            # --- 试剂操作 ---
            # === [修正点 3] Add 逻辑 ===
            # === [修正点 3] Add 逻辑 (支持固体) ===
            elif act_type == "Add":
                vessel_id = action.get("vessel")
                reagent = action.get("reagent")
                
                # 1. 【优先】解析化学式并查库
                real_formula = self._resolve_chem_name(reagent)
                # 使用你提供的 substances.json 数据结构
                db_info = CHEM_DB.substances.get(real_formula, {})

                # === [修改] 增加气体判定 ===
                # 检查数据库 state 字段，或者根据名字猜测
                db_state = db_info.get("state", "l")
                is_gas = (db_state == "g" or db_state == "gas")
                
                # 2. 【核心修复】智能判断是否为固体
                # 依据 provided data: "state": "s"
                db_state_is_solid = (db_info.get("state") == "s")
                
                # 增加关键词兜底，防止 "Marble Chips" 这种名字查不到库
                name_indicates_solid = any(k in str(reagent).lower() for k in ["marble", "solid", "chunk", "powder", "大理石", "块", "粉", "粒"])
                
                should_be_solid = db_state_is_solid or name_indicates_solid

                # 3. 解析数量
                raw_vol = action.get("volume")
                raw_mass = action.get("mass") or action.get("amount")
                
                val, unit = parse_quantity(raw_mass) if raw_mass else parse_quantity(raw_vol)

                # 4. 【核心修复】自动填充与单位纠正
                if val <= 1e-6:
                    val = 5.0 # 默认给 5.0
                    self.logger.warning(f"Action 'Add {reagent}' had 0 amount. Auto-filled to {val}.")
                    
                    # 如果判定应该是固体，但没有单位，强制设为 'g'
                    if not unit and should_be_solid:
                        unit = 'g'
                
                # [🔥新增] 冲突检测：如果单位是 g 但数据库说是液体
                if unit == 'g' and not db_state_is_solid:
                    # 策略：相信物理单位，视为通过质量添加液体（需要密度，这里简化为 1g=1ml）
                    # 或者提示错误。这里为了鲁棒性，将其转换为体积
                    obs_text.append(f"警告：尝试以质量(g)添加液体 {reagent}，按密度1.0换算为体积。")
                    unit = 'ml'
                    is_solid_op = False # 强制纠正为液体操作

                # 5. 最终判定逻辑
                is_solid_op = (unit == 'g') or should_be_solid

                target = self.containers.get(vessel_id)
                if target:
                    # 获取温度
                    raw_temp = action.get("temp")
                    reagent_temp, _ = parse_quantity(raw_temp, default=25.0)

                    if is_solid_op:
                        # === 固体添加路径 ===
                        # 固体加入不增加容器的液体体积 (vol=0.0)
                        components = CHEM_DB.get_components(real_formula, val, is_solid=True)
                        for ion, moles in components.items():
                            target.add_chemical(ion, moles, 0.0, fluid_temp=reagent_temp)
                        obs_text.append(f"向 {vessel_id} 加入了 {val}g {reagent} (固体)。")

                    elif is_gas:
                        # === [新增] 气体添加路径 ===
                        # 气体作为溶质加入，但不增加液体体积 (vol_to_add = 0)
                        # 注意：这里简化处理，假设气体溶解或填充空间，不计算压强
                        components = CHEM_DB.get_components(real_formula, val, is_solid=False) # 计算摩尔数
                        for ion, moles in components.items():
                            # 关键：volume 传 0.0
                            target.add_chemical(ion, moles, 0.0, fluid_temp=reagent_temp)
                        obs_text.append(f"向 {vessel_id} 通入了 {val}ml {reagent} (气体)。")
                    else:
                        # === 液体添加路径 ===
                        components = CHEM_DB.get_components(real_formula, val, is_solid=False)
                        for ion, moles in components.items():
                            vol_to_add = val if ion == next(iter(components)) else 0
                            target.add_chemical(ion, moles, vol_to_add, fluid_temp=reagent_temp)
                        obs_text.append(f"向 {vessel_id} 加入了 {val}ml {reagent}。")
                    
                    # 触发反应计算
                    phenomena = ReactionManager.solve(target)
                    if phenomena: obs_text.append(f"现象：{'；'.join(phenomena)}")
            
            elif act_type == "Transfer":
                src_id = action.get("from_vessel")
                dst_id = action.get("to_vessel")
                
                src = self.containers.get(src_id)
                dst = self.containers.get(dst_id)
                
                # 1. 获取原始值
                raw_vol = action.get("volume")
                # 2. 使用智能解析
                vol, _ = parse_quantity(raw_vol)

                # === [修改点 1] 零值防御逻辑 (保持你现有的，防止 LLM 给 0) ===
                if vol is None or vol <= 1e-6:
                    if "waste" in dst_id.lower() or vol is None:
                        # 如果是倒废液，或者没写体积，默认全部倒出
                        # 注意：如果是气体，src.volume是0，这里vol会变0，
                        # 但没关系，后面的 transfer_to 会识别 contents 并全量转移
                        vol = src.volume if src else 0
                    else:
                        # 显式写了 0 且不是倒废液，给个默认操作量 (如 10ml)
                        vol = 10.0
                        obs_text.append(f"(系统提示: 转移量过小，自动调整为 {vol}ml)")

                if not src: return f"操作失败：源容器 '{src_id}' 不存在。", current_snapshot
                if not dst: return f"操作失败：目标容器 '{dst_id}' 不存在。", current_snapshot
                
                # === [修改点 2] 智能警告与截断 (核心修复) ===
                # 判断源容器里是否有液体（体积大于微量）
                is_liquid_transfer = src.volume > 1e-6
                
                if is_liquid_transfer:
                    # 只有是液体时，才检查体积够不够。如果不够，截断为剩余体积。
                    if src.volume < vol:
                        obs_text.append(f"警告：{src_id} 只有 {src.volume:.1f}ml，全部倒出了。")
                        vol = src.volume
                # else: 
                #   如果是气体/固体 (is_liquid_transfer 为 False)，
                #   src.volume 是 0，肯定小于 vol (比如 100)，
                #   绝对不能执行 vol = src.volume，否则转移量就变 0 了！
                #   所以这里直接跳过截断逻辑，保留 vol = 100 传给底层。

                # 调用底层转移函数 (需配合 Container.transfer_to 的修改)
                src.transfer_to(dst, vol)
                
                # === [修改点 3] 差异化描述 ===
                if is_liquid_transfer:
                    obs_text.append(f"将 {vol:.1f}ml 液体从 {src_id} 转移到了 {dst_id}。")
                else:
                    # 气体或固体转移的特殊描述
                    obs_text.append(f"将 {src_id} 中的内容物（气体/固体）转移到了 {dst_id}。")
                
                phenomena = ReactionManager.solve(dst)
                if phenomena:
                    obs_text.append(f"混合现象：{'；'.join(phenomena)}")
                obs_text.append(f"目标容器状态：{dst.color_description}。")

            elif act_type == "Stir":
                vessel_id = action.get("vessel")
                obs_text.append(f"正在搅拌容器 {vessel_id}...")
                target = self.containers.get(vessel_id)
                if target:
                    phenomena = ReactionManager.solve(target)
                    if phenomena: obs_text.append(f"搅拌加速了反应：{'；'.join(phenomena)}")

            elif act_type == "Heat":
                vessel_id = action.get("vessel")
                target = self.containers.get(vessel_id)

                # --- [FIX START] 修复类型比较错误 ---
                # 1. 获取原始值
                raw_temp = action.get("target_temperature")
                
                # 修复：使用 parse_quantity 处理单位
                raw_temp = action.get("target_temperature")
                assigned_temp, _ = parse_quantity(raw_temp) # 即使解析失败也会返回 0.0，但能处理 "100C"
                if raw_temp is not None and assigned_temp == 0.0 and "0" not in str(raw_temp):
                    # 简单的兜底：如果没解析出数字且原字符串不是"0"，可能默认设为100
                    assigned_temp = 100.0
                # --- [FIX END] ---
                
                
                if target:
                    current_temp = target.temperature
                    
                    if assigned_temp is not None:
                        # 模式 A: 目标导向加热 (LLM 说要加热到 100度)
                        # 模拟逐渐升温的过程，或者直接到位(取决于你的仿真粒度)
                        if current_temp < assigned_temp:
                            # 这里做一个简单的物理限制：不能瞬间升温太快
                            real_increase = min(assigned_temp - current_temp, 30.0) # 这是一个时间步长的最大升温
                            target.temperature += real_increase
                            obs_text.append(f"加热 {vessel_id}，温度上升至 {target.temperature:.1f}°C (目标: {assigned_temp}°C)。")
                        else:
                            obs_text.append(f"{vessel_id} 温度维持在 {target.temperature:.1f}°C。")
                    else:
                        # 模式 B: 傻瓜式加热 (默认点火)
                        target.temperature += 20.0
                        obs_text.append(f"对 {vessel_id} 进行了持续加热，当前温度 {target.temperature:.1f}°C。")

                    # 沸腾检查 (通用物理法则)
                    if target.temperature >= 100.0:
                        target.temperature = 100.0 # 简单的水沸腾模型
                        obs_text.append("液体正在沸腾！")
            
            elif act_type == "Cool":
                vessel_id = action.get("vessel")
                target = self.containers.get(vessel_id)
                if target:
                    target.temperature = max(0.0, target.temperature - 15.0)
                    obs_text.append(f"{vessel_id} 正在冷却，温度降至 {target.temperature:.1f}°C。")


            elif act_type == "MeasureTemperature":
                vessel_id = action.get("vessel")
                target = self.containers.get(vessel_id)
                if target:
                    obs_text.append(f"【读数】{vessel_id} 当前温度为: {target.temperature:.1f} °C")
            
            elif act_type == "MeasureMass":
                vessel_id = action.get("vessel") # 通常是称量某个容器里的东西
                target = self.containers.get(vessel_id)
                if target:
                    # 计算总质量 = 液体质量(近似体积) + 沉淀质量
                    # 简化：假设密度为1
                    total_mass = target.volume + target.precipitate_mass
                    obs_text.append(f"【读数】{vessel_id} 当前总质量约为: {total_mass:.2f} g")

            elif act_type == "Filter":
                src_id, dst_id = action.get("from_vessel"), action.get("to_vessel")
                src = self.containers.get(src_id)
                dst = self.containers.get(dst_id)

                if src and dst:
                    # 1. 模拟物理转移：将液体部分转移到 dst
                    # 简单模型：转移所有液体，留下所有沉淀
                    liquid_vol = src.volume

                    # 将 src 的所有溶质（非沉淀）转移给 dst
                    # 注意：这里需要更复杂的化学逻辑判断谁是沉淀，这里做个简化版
                    dst.volume += liquid_vol
                    dst.temperature = src.temperature # 传递温度

                    # 转移溶解的离子
                    for chem, moles in src.contents.items():
                        # 假设 _ppt 后缀或数据库 state='s' 的是沉淀
                        info = CHEM_DB.substances.get(chem, {})
                        if not (chem.endswith("_ppt") or info.get("state") == "s"):
                            dst.contents[chem] = dst.contents.get(chem, 0) + moles

                    # src 只保留沉淀和少量润湿液体
                    src.volume = 0.5 # 滤纸吸附少量水
                    # 清除 src 中的溶解离子，只保留沉淀（此处代码略，需遍历删除）

                    obs_text.append(f"过滤完成：{dst_id} 获得 {liquid_vol:.1f}ml 滤液。")
                
            elif act_type == "CollectGas":
                src, col = action.get("source_vessel"), action.get("collector")
                obs_text.append(f"正在收集 {src} 产生的气体到 {col} 中。")

            else:
                return f"不支持的 XDL 动作: {act_type}", current_snapshot
        
        except Exception as e:
            error_msg = f"模拟引擎内部错误: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return error_msg, current_snapshot

        final_snapshot = {
            "containers": {k: v.get_snapshot() for k, v in self.containers.items()},
            "topology": self.hw_manager.get_topology_snapshot() # <--- 这里变了
        }
        
        # === [补全日志] 记录物理数值快照 ===
        # 提取有液体的容器状态，存入日志
        active_containers = {}
        for vid, c in self.containers.items():
            if c.volume > 0:
                active_containers[vid] = {
                    "vol": c.volume,
                    "ph": round(c.ph, 2),
                    "color": c.color_description
                }
        
        # 这里记录到 System 日志或 Dialogue 日志都行，建议 Dialogue 以便关联查看
        log_turn_event("Dialogue", "Physics_State_Snapshot", {
            "action": action.get("action"),
            "observation": "".join(obs_text),
            "containers": active_containers # <--- 这里就是缺失的物理数值
        })
        
        return "".join(obs_text), final_snapshot

    def execute_batch(self, actions: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        顺序执行多个动作
        """
        total_observation = []
        last_snapshot = {
            "containers": {k: v.get_snapshot() for k, v in self.containers.items()},
            "topology": self.hw_manager.get_topology_snapshot()
        }
        
        # print(f"[Engine] Batch executing {len(actions)} actions...")

        for idx, action in enumerate(actions):
            # 执行单步
            obs_text, snapshot = self.execute(action)
            last_snapshot = snapshot 
            
            step_obs = f"Step {idx+1}: {obs_text}"
            total_observation.append(step_obs)
            
            # [新增] 熔断机制：如果发现“操作失败”或“内部错误”，停止后续步骤
            if "操作失败" in obs_text or "内部错误" in obs_text:
                total_observation.append("(由于上一步出错，后续动作已自动取消)")
                break
        
        full_report = "\n".join(total_observation)
        return full_report, last_snapshot
    
# ==========================================
# 6. 中间件: ActionTranslator (增强鲁棒性版)
# ==========================================
class ActionTranslator:
    """
    [XDL Strict Version] 动作翻译器
    严格遵循 XDL 3.0 规范，确保输出指令与物理引擎兼容。
    """
    def __init__(self, client: OpenAI):
        self.client = client
        self.logger = logging.getLogger("ActionTranslator")
        
        # 定义 XDL 允许的动作白名单 (基于 XDL_description_build.txt)
        self.ALLOWED_ACTIONS = {
            "Add", "Transfer",                  # Group 1: Fluid [cite: 9]
            "Attach", "Insert",                 # Group 2: Topology [cite: 14, 15]
            "Heat", "Cool", "Stir",             # Group 3: Environment [cite: 16, 17]
            "MeasureTemperature", "MeasureMass", "Wash",
            "Filter", "CollectGas"              # Group 4: Separation [cite: 18]
        }
    @RETRY_RULE
    def _call_translator_llm(self, messages):
        return self.client.chat.completions.create(
            model=Config.MODEL_TRANSLATOR, 
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )

    def translate(self, raw_text_intent: str, 
              snapshot: Dict[str, Any], 
              reagent_state_map: Dict[str, str],
              history: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[str]]: # 返回 (动作, 错误)
        
        if not raw_text_intent: 
            return [], []  # <--- 修改这里，返回两个空列表

        # 1. 上下文构建
        all_hardware = snapshot.get("containers", {})
        valid_ids = list(all_hardware.keys())

        # [新增] 硬件列表描述，强化 LLM 认知
        hw_context = "\n".join([f"- {vid}" for vid in valid_ids])
        
        # 提取最近对话辅助消歧义
        recent_dialogue = history[-2:] if len(history) >= 2 else history
        dialogue_str = json.dumps(recent_dialogue, ensure_ascii=False)

        # 2. 构建符合 XDL 规范的 Prompt
        prompt = f"""
        You are a Strict XDL Compiler for a Chemistry Lab.
        Translate Student Intent into executable JSON actions.

        ### 1. Lab Context(STRICT BOUNDARY)
        - Available Hardware IDs: 
        {hw_context}
        * CRITICAL: Do NOT use any IDs not listed above. If student asks for 'cylinder' but it's not here, do not invent one.
        - Reagents (Name -> State): {json.dumps(reagent_state_map, ensure_ascii=False)} 
          * Note: 'solid' requires 'mass', 'liquid' requires 'volume'.

        ### 2. Action Logic - Add vs. Transfer (CRITICAL)
        - **Use "Add"**: When introducing a NEW reagent into a vessel. The source is always an external reagent bottle (not in the hardware list).
        * Example: "Add 5ml HCl to the beaker" -> {{"action": "Add", "vessel": "beaker", "reagent": "HCl", "volume": 5.0}}
        - **Use "Transfer"**: ONLY when moving existing liquid between two hardware vessels already on the table.
        * Example: "Pour the liquid from the test_tube into the flask" -> {{"action": "Transfer", "from_vessel": "test_tube", "to_vessel": "flask", "volume": 10.0}}

        ### 3. Recent Dialogue (For resolving 'it', 'that')
        {dialogue_str}
        
        ### 4. Student Intent
        "{raw_text_intent}"

        ### 5. STRICT ACTION SCHEMA (Must use ONLY these)
        
        [GROUP 1] FLUID & REAGENT
        - Add: {{ "action": "Add", "vessel": "ID", "reagent": "Name", "volume": "if liquid", "mass": "if solid" }}
        - Transfer: {{ "action": "Transfer", "from_vessel": "ID", "to_vessel": "ID", "volume": float }}
        
        [GROUP 2] TOPOLOGY
        - Attach: {{ "action": "Attach", "vessel": "child_ID", "support": "parent_ID" }}  (e.g. fix burette to stand)
        - Insert: {{ "action": "Insert", "tool": "object_ID", "vessel": "container_ID" }} (e.g. thermometer in beaker)
        - Detach: {{ "action": "Detach", "object": "child_ID", "support": "optional_parent_ID" }} 
          * Use this when student wants to remove/disconnect/take off an object.
          * Example: "Remove the thermometer" -> {{ "action": "Detach", "object": "thermometer" }}
          * Example: "Take the tube out of the flask" -> {{ "action": "Detach", "object": "tube", "support": "flask" }}

        [GROUP 3] OPERATION
        - Heat: {{ "action": "Heat", "vessel": "ID", "target_temperature": float }}
        - Stir: {{ "action": "Stir", "vessel": "ID" }}
        - MeasureMass: {{ "action": "MeasureMass", "vessel": "ID" }} (or reagent)
        - MeasureTemperature: {{ "action": "MeasureTemperature", "vessel": "ID" }}

        [GROUP 4] SEPARATION
        - Filter: {{ "action": "Filter", "from_vessel": "ID", "to_vessel": "ID" }}
        - CollectGas: {{ "action": "CollectGas", "source_vessel": "ID", "collector": "ID" }}
        - Wash: {{ "action": "Wash", "vessel": "ID" }}

        ### 5. MAPPING RULES
        - "Titrate", "Drop", "Flow" -> Transfer (burette -> flask)
        - "Dissolve", "Mix" -> Stir
        - "Weigh" -> MeasureMass
        - "Check temp" -> MeasureTemperature
        - "Pour A into B" -> Transfer (from=A, to=B)
        - "Check volume/amount" -> Read volume visually (Do NOT use MeasureMass unless weighing a solid)
        - "Remove", "Disconnect", "Take off", "Unplug" -> Detach
        - "Move A to B" -> If A is already attached, generate TWO actions: [Detach A, Insert A->B]
        - "Prepare solution" -> If adding liquid, use "Add"; do NOT infer "MeasureMass" unless explicitly stated.
        - 当老师或学生提到“清洗（Wash）”、“洗干净”时，必须优先生成单个 {{"action": "Wash", "vessel": "ID"}} 动作，严禁将其拆解为多个 Add 和 Transfer 的循环。

        Output JSON: {{ "thought": "...", "actions": [...] }}
        """
        
        try:
            resp = self._call_translator_llm([{"role": "system", "content": prompt}])

            content = resp.choices[0].message.content
            
            # [Fix] 增加解析容错：处理可能存在的 markdown 包裹
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            result = json.loads(content.strip())
            # === [核心修复 Start] ===
            # 判断 LLM 返回的是 字典 还是 列表
            if isinstance(result, list):
                # 情况 A: LLM 直接返回了动作列表 [ ... ]
                self.logger.warning("LLM output a direct list, skipping 'thought'.")
                raw_actions = result
            elif isinstance(result, dict):
                # 情况 B: LLM 按要求返回了 { "thought": ..., "actions": ... }
                self.logger.info(f"[Translator Thought]: {result.get('thought', 'No thought')}")
                raw_actions = result.get("actions", [])
            else:
                # 情况 C: 未知格式
                return [], ["系统错误: LLM 返回了非 JSON 对象/数组格式"]
            # === [核心修复 End] ===
            
            valid_actions = []
            hallucination_errors = [] # 记录幻觉错误

            for act in raw_actions:
                # 防御 1: Add 动作不能有 from_vessel
                if act.get("action") == "Add":
                    if "from_vessel" in act:
                        self.logger.warning("Corrected: Removed from_vessel from Add action.")
                        del act["from_vessel"]
                        
                # 防御 2: Transfer 动作的源必须是有效的硬件 ID，不能是试剂名
                if act.get("action") == "Transfer":
                    from_id = act.get("from_vessel")
                    if from_id and from_id not in valid_ids:
                        # 如果 from_vessel 是个试剂名，将其转换为 Add 动作
                        self.logger.warning(f"Corrected: Converted Transfer from reagent '{from_id}' to Add.")
                        act["action"] = "Add"
                        act["reagent"] = from_id
                        act["vessel"] = act.get("to_vessel")
                        del act["from_vessel"]
                        del act["to_vessel"]
                # 修改 _sanitize_action_ids 使其在 ID 无效时返回具体信息
                sanitized, err_msg = self._sanitize_with_feedback(act, valid_ids)
                if sanitized:
                    valid_actions.append(sanitized)
                elif err_msg:
                    hallucination_errors.append(err_msg)

            return valid_actions, hallucination_errors

        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return [], [f"系统翻译错误: {str(e)}"]
        
    def _sanitize_with_feedback(self, action: Dict[str, Any], valid_ids: List[str]) -> Tuple[Optional[Dict], Optional[str]]:
        """检查 ID 是否合法，如果不合法，返回错误描述"""
        new_action = action.copy()
        id_fields = ["vessel", "from_vessel", "to_vessel", "support", "tool", "collector"]
        
        for field in id_fields:
            if field in new_action:
                raw_id = new_action[field]
                if not raw_id: continue
                
                # 模糊匹配
                matches = difflib.get_close_matches(raw_id, valid_ids, n=1, cutoff=0.6)
                if matches:
                    new_action[field] = matches[0]
                else:
                    # 核心拦截：如果匹配不到任何现有硬件，则判定为幻觉
                    return None, f"实验台上找不到您提到的器材 '{raw_id}'。"
        return new_action, None
        
    def _fill_defaults(self, action: Dict[str, Any], state_map: Dict[str, str] = None) -> Dict[str, Any]:
        """根据 XDL 规范填充默认值"""
        act_type = action.get("action")

        # 辅助函数：解析任意格式并判断是否为有效正数
        # 依赖外部定义的 parse_quantity 函数 (在 soc_chem_dia.py 全局作用域中)
        def is_effectively_zero(val):
            if val is None: return True
            # 使用 parse_quantity 提取数值 (它能处理 "0 mL" -> 0.0)
            num, _ = parse_quantity(val)
            return num <= 1e-6
        
        if act_type == "Add":
            raw_vol = action.get("volume")
            raw_mass = action.get("mass")
            
            # 如果两个都为空或者都为 0 (包括 "0 mL")
            if is_effectively_zero(raw_vol) and is_effectively_zero(raw_mass):
                reagent = action.get("reagent")
                is_solid = False
                if state_map and reagent:
                    state = state_map.get(reagent, "liquid")
                    if state == "solid" or state == "s":
                        is_solid = True
                
                if is_solid:
                    action["mass"] = 5.0
                else:
                    action["volume"] = 5.0

        elif act_type == "Transfer":
            raw_vol = action.get("volume")
            # 如果是 0 或 "0 mL" 或 None
            if is_effectively_zero(raw_vol):
                action["volume"] = 10.0 # 强制修正为 10ml
                action["volume"] = 10.0 # 强制修正为 10ml
        
        # Heat: 默认 100度 [cite: 16]
        elif act_type == "Heat":
            if "target_temperature" not in action:
                action["target_temperature"] = 100.0
                
                
        return action

    def _format_hardware_context(self, all_hardware: Dict[str, Any]) -> str:
        """
        辅助函数：将硬件分为 [Containers] 和 [Tools] 两类，方便 LLM 理解。
        """
        vessels = []
        tools = []
        
        # 简单的启发式关键词分类
        # 建议补充 cylinder, match, paper 等
        tool_keywords = ["pipette", "dropper", "rod", "spoon", "bulb", "tool", 
                         "cylinder", "match", "paper", "stand", "lamp", "mesh", "thermometer"]
        
        for vid, data in all_hardware.items():
            # 这里虽然 Simulation 传来的 snapshot 主要是物理数据
            # 但我们可以根据 ID 名字来猜测它是工具还是容器
            # 如果你在 XDL 里定义了 type="tool"，这里可以通过 ID 命名规范来区分
            
            is_tool = any(k in vid.lower() for k in tool_keywords)
            
            # 构造描述
            vol = data.get('volume_ml', 0)
            desc = f"- {vid}"
            if vol > 0:
                desc += f" (Vol: {vol}ml)"
            
            if is_tool:
                tools.append(desc)
            else:
                vessels.append(desc)
                
        # 组装文本
        output = []
        if vessels:
            output.append("[Containers/Vessels]")
            output.extend(vessels)
        if tools:
            output.append("\n[Tools/Instruments]")
            output.extend(tools)
        
        if not output:
            return "(No hardware found)"
            
        return "\n".join(output)

    def _validate_structure(self, action: Dict[str, Any]) -> bool:
        """校验 XDL 必填字段 [cite: 10, 13, 14, 15]"""
        act_type = action.get("action")
        
        required_map = {
            "Add": ["vessel", "reagent"], # volume/mass 也是必填，但在 _fill_defaults 处理了
            "Transfer": ["from_vessel", "to_vessel"],
            "Attach": ["vessel", "support"],
            "Insert": ["tool", "vessel"],
            "Heat": ["vessel"],
            "Filter": ["from_vessel", "to_vessel"],
            "CollectGas": ["source_vessel", "collector"]
        }
        
        if act_type in required_map:
            for field in required_map[act_type]:
                if not action.get(field):
                    return False
        return True

    def _sanitize_action_ids(self, action: Dict[str, Any], valid_ids: List[str]) -> Optional[Dict[str, Any]]:
        """
        使用 difflib 修复所有 ID 字段 (包括 tool)
        """

        new_action = action.copy()
        act_type = new_action.get("action")

        # [新增] 核心参数完整性检查
        # 如果是 Insert/Attach，且缺少核心目标 ID，直接丢弃该动作，防止引擎崩溃
        if act_type == "Insert" and not new_action.get("vessel"):
            self.logger.warning(f"Dropping invalid Insert action (no vessel): {new_action}")
            return None # 返回 None 表示丢弃此动作
        
        if act_type == "Attach" and (not new_action.get("vessel") or not new_action.get("support")):
            self.logger.warning(f"Dropping invalid Attach action: {new_action}")
            return None
        
        # 需要检查 ID 的所有潜在字段名
        id_fields = ["vessel", "from_vessel", "to_vessel", "support", "tool", "collector"]
        new_action = action.copy()
        
        for field in id_fields:
            if field in new_action:
                raw_id = new_action[field]
                if not raw_id: continue # 跳过空值
                
                # 1. 完全匹配
                if raw_id in valid_ids:
                    continue 
                
                # 2. 模糊匹配
                matches = difflib.get_close_matches(raw_id, valid_ids, n=1, cutoff=0.5)
                
                if matches:
                    best_match = matches[0]
                    self.logger.warning(f"Fuzzy Match ({field}): '{raw_id}' -> '{best_match}'")
                    new_action[field] = best_match
                else:
                    # 特殊处理：如果 tool 找不到 ID，且看起来像是一个通用名称（如 'pipette'），
                    # 我们可以选择保留它（前提是 Engine 那边做了兼容），或者直接丢弃该字段
                    if field == "tool":
                        self.logger.warning(f"Tool ID '{raw_id}' not found in hardware list. Dropping tool parameter.")
                        del new_action[field] # 删除无效的工具参数，降级为普通操作
                    else:
                        self.logger.error(f"Invalid ID '{raw_id}' in field '{field}'. Action dropped.")
                        return None # 核心容器找不到，动作无效
                    
        return new_action

# ==========================================
# 模块 D: 评估端 - Reviewer
# ==========================================

class GoalOrientedInspector:
    """
    [State Verification Engine]
    目标导向检查器：对比 [学生当前快照] 与 [Oracle理想快照]。
    """
    def __init__(self):
        # 基础容差定义
        self.VOL_TOLERANCE_MAP = {
            ToleranceLevel.STRICT: 0.5,   # 精密操作 (如滴定终点前)
            ToleranceLevel.NORMAL: 2.0,   # 标准添加
            ToleranceLevel.LOOSE: 5.0     # 粗略操作 (如清洗)
        }
        self.PH_TOLERANCE = 0.5

    def _is_reactant_product_pair(self, reactant_name: str, product_name: str, reaction_db: List[Dict]) -> bool:
        """
        [Chemlib 通用版] 检查是否存在 Reactant -> Product 的转化路径
        利用 chemlib 自动解析化学式，忽略系数影响 (如 2NaOH -> NaOH)
        """
        if not ChemReaction:
            return False # 防御性编程

        for rxn_entry in reaction_db:
            eq_str = rxn_entry.get("equation")
            if not eq_str: continue

            try:
                # 1. 使用 chemlib 解析方程式
                # 例如: "HCl + NaOH -> NaCl + H2O"
                rxn = ChemReaction.by_formula(eq_str)

                # 2. 提取反应物和生成物的化学式集合
                # rxn.reactants 是 Compound 对象列表，取 .formula 属性
                r_formulas = {r.formula for r in rxn.reactants}
                p_formulas = {p.formula for p in rxn.products}

                # 3. 判定：反应物里有 A，且生成物里有 B
                if reactant_name in r_formulas and product_name in p_formulas:
                    return True

            except Exception:
                # 捕获解析错误，跳过格式不对的方程式
                continue
                
        return False

    def inspect(self, current_snapshot: Dict[str, Any], 
                pre_snapshot: Dict[str, Any], 
                goal: GoalState,
                last_actions: List[Dict[str, Any]] = None) -> InspectionResult:
        """
        [修正版] 核心评估函数：智能识别增量与全量偏差
        """
        if not goal or not goal.target_vessel_id:
            return InspectionResult(ReviewStatus.PASS, "无目标步骤")

        target_c = current_snapshot.get("containers", {}).get(goal.target_vessel_id)
        pre_c = pre_snapshot.get("containers", {}).get(goal.target_vessel_id, {})
        
        if not target_c:
            return InspectionResult(ReviewStatus.FAIL, "找不到目标容器", diagnosis={"type": "MISSING_VESSEL"})
        
        # --- 1. [核心修复] 化学成分指纹校验 (Chemical Fingerprint) ---
        # 解决 0.2ml 酚酞被 3.0ml 容差覆盖的问题
        if goal.expected_moles:
            actual_species = set(target_c.get("major_species", []))
            
            for chem, exp_mol in goal.expected_moles.items():
                if exp_mol > 1e-8:  # 只有预期存在的物质才检查
                    # === [修改] 增强模糊匹配 (去空格 + 忽略大小写) ===
                    # 目标: CarbonDioxide -> carbondioxide
                    # 实际: Carbon Dioxide -> carbondioxide
                    
                    target_key = chem.replace(" ", "").lower()
                    
                    found = False

                    for s in actual_species:
                        actual_key = s.replace(" ", "").lower()
                        # 1. 完全包含 (如 "dilute hcl" vs "hcl")
                        # 2. 去空格后相等 (如 "Carbon Dioxide" vs "CarbonDioxide")
                        if target_key in actual_key or actual_key in target_key:
                            found = True
                            break
                    
                    # 修改 GoalOrientedInspector.inspect 中成分缺失的部分
                    # 估算缺失的试剂对应的体积 (V = n / C)
                    # [修正版] 智能诊断逻辑
                    if not found:
                        current_vol = target_c.get("volume_ml", 0.0)
                        
                        # 获取上一刻的体积 (Pre-Snapshot)
                        pre_vol = pre_c.get("volume_ml", 0.0)
                        vol_diff = current_vol - pre_vol
                        
                        # === [🔥核心修复] ===
                        # 如果体积几乎没变 (变化量 < 1.0ml)，说明是单纯的“没加进去” (INSUFFICIENT)
                        # 即使杯子里有 20ml 液体，那也是之前的底液，不是污染！
                        if abs(vol_diff) < 1.0:
                            missing_vol = (exp_mol / 0.1) * 1000 
                            return InspectionResult(
                                ReviewStatus.FAIL,
                                f"关键成分 {chem} 缺失 (未检测到添加操作)",
                                diagnosis={"type": "INSUFFICIENT", "diff": missing_vol}
                            )

                        # 只有当体积显著增加 (> 1.0ml) 或者原本是空的现在有了液体
                        # 但依然缺关键成分时，才判定为“加错了” (WRONG_REAGENT)
                        if current_vol > 2.0:
                             return InspectionResult(
                                ReviewStatus.FAIL,
                                f"容器内新增了 {vol_diff:.1f}ml 液体，但关键成分 {chem} 缺失 (判定为错加试剂)。",
                                diagnosis={"type": "WRONG_REAGENT", "vessel": goal.target_vessel_id}
                            )

                        # 兜底：如果是微量残留
                        missing_vol = (exp_mol / 0.1) * 1000 
                        return InspectionResult(
                            ReviewStatus.FAIL,
                            f"关键成分 {chem} 缺失",
                            diagnosis={"type": "INSUFFICIENT", "diff": missing_vol}
                        )

        # 1. 物理连接检查
        curr_topo = {(l['child'], l['parent']) for l in current_snapshot.get("topology", [])}
        for req in goal.expected_topology:
            if req not in curr_topo:
                return InspectionResult(ReviewStatus.FAIL, f"设备连接未完成: {req[0]} -> {req[1]}", diagnosis={"type": "TOPOLOGY_ERROR"})

        
        # 2. 体积审计：模式切换
        curr_vol = target_c.get("volume_ml", 0.0)
        pre_vol = pre_c.get("volume_ml", 0.0)
        actual_delta = curr_vol - pre_vol
        
        # 核心逻辑：根据 Goal 类型决定比对基准
        if goal.is_delta_check:
            # 补救模式：检查“这次加了多少”
            diff = actual_delta - goal.expected_delta
            check_type = "增量"
            target_val = goal.expected_delta
            current_val = actual_delta
        else:
            # 标准模式：检查“容器里最终有多少”
            diff = curr_vol - goal.expected_volume
            check_type = "总量"
            target_val = goal.expected_volume
            current_val = curr_vol

        vol_tol = self.VOL_TOLERANCE_MAP[goal.tolerance]

        # 3. 判定偏差
        if diff > vol_tol:
            # 容错小技巧：如果是补救模式且超出不多（15%以内），放行但警告
            if goal.is_delta_check and diff < (target_val * 0.15):
                return InspectionResult(ReviewStatus.PASS, f"PASS_WITH_WARN: 补加量略多({diff:.1f}ml)，但不影响实验。")
            
            return InspectionResult(ReviewStatus.FAIL, f"液体{check_type}过量 (当前 {current_val:.1f}ml, 预期 {target_val:.1f}ml)", 
                                 diagnosis={"type": "OVERSHOOT", "vessel": goal.target_vessel_id, "diff": diff})
        
        if diff < -vol_tol:
             return InspectionResult(ReviewStatus.FAIL, f"液体{check_type}不足 (当前 {current_val:.1f}ml)", 
                                 diagnosis={"type": "INSUFFICIENT", "vessel": goal.target_vessel_id, "diff": abs(diff)})

        return InspectionResult(ReviewStatus.PASS, "状态验收通过。")

    def scan_side_effects(self, current_snapshot: Dict[str, Any], 
                          pre_snapshot: Dict[str, Any], 
                          goal: GoalState) -> Optional[str]:
        """
        [安全扫描] 检查是否有非目标容器发生了意外变化
        """
        if not goal or not goal.target_vessel_id: return None
        
        warnings = []
        for vid, post_c in current_snapshot.get("containers", {}).items():
            # 跳过本次任务的目标容器
            if vid == goal.target_vessel_id: continue
            # 跳过源试剂瓶 (因为取液肯定会减少体积)
            if post_c.get("major_species") and "is_source" in post_c: continue # 需在 Container 加标志位，或者简单通过命名判断

            pre_c = pre_snapshot.get("containers", {}).get(vid, {})
            
            vol_diff = post_c.get("volume_ml", 0) - pre_c.get("volume_ml", 0)
            
            # 如果非目标容器体积发生了显著变化 (比如倒废液倒错了地方)
            if abs(vol_diff) > 5.0:
                warnings.append(f"注意：{vid} 的体积发生了意外变化 ({vol_diff:+.1f}ml)。")
        
        return " ".join(warnings) if warnings else None

class Reviewer:
    """
    [v5.0 Optimized] 评估控制器
    职责：调度 Inspector，并在需要时调用 LLM 进行深度归因。
    """
    def __init__(self, client, oracle):
        self.client = client
        self.oracle = oracle
        self.logger = logging.getLogger(self.__class__.__name__)
        # 初始化规则检查器
        self.inspector = GoalOrientedInspector() # 使用新 Inspector

    # [修复] 历史追溯方法
    def _is_history_lost(self, target_vessel: str, missing_species: List[str]) -> int:
        """
        检查缺失的物质是否属于“历史遗留资产”。
        返回：该物质最早出现的步骤索引 (origin_step)。
        """
        earliest_origin = 9999
        found_in_history = False

        # 遍历 Oracle 历史
        sorted_steps = sorted([k for k in self.oracle.history.keys() if k >= 0])
        
        for step_idx in sorted_steps:
            snapshot = self.oracle.history[step_idx]
            
            # 这里获取到的是 Container 对象实例
            container_obj = snapshot["containers"].get(target_vessel)
            
            if not container_obj:
                continue

            # === [FIX START] ===
            # 将 Container 对象转换为数据字典
            if hasattr(container_obj, "get_snapshot"):
                container_data = container_obj.get_snapshot()
            elif isinstance(container_obj, dict):
                # 防御性编程：万一 history 存的是 dict
                container_data = container_obj
            else:
                continue
            # === [FIX END] ===

            # 现在可以安全使用 .get() 了
            hist_species = set(container_data.get("major_species", []))
            
            # 检查是否有交集 (缺失物 是否曾经存在过)
            if set(missing_species).intersection(hist_species):
                if step_idx < earliest_origin:
                    earliest_origin = step_idx
                found_in_history = True
        
        return earliest_origin if found_in_history else -1

    def _analyze_topology_change(self, pre: Dict[str, Any], post: Dict[str, Any]) -> str:
        """
        [Update] 计算拓扑结构的增量变化 (支持 Graph 结构)
        """
        # 数据格式: List[{"child": "A", "parent": "B", "type": "sealed"}]
        pre_topo_list = pre.get("topology", [])
        post_topo_list = post.get("topology", [])

        # 转换为 Set of tuples 以便比较: set((child, parent, type))
        def to_set(topo_list):
            s = set()
            for item in topo_list:
                s.add((item['child'], item['parent'], item['type']))
            return s

        pre_links = to_set(pre_topo_list)
        post_links = to_set(post_topo_list)

        added = post_links - pre_links
        removed = pre_links - post_links

        if not added and not removed:
            return "【连接状态】: 无变化"

        report = []
        if added:
            desc_list = []
            for child, parent, ctype in added:
                relation_str = f"{child} --({ctype})--> {parent}"
                desc_list.append(relation_str)
            report.append(f"【新连接】: {', '.join(desc_list)}")
            
        if removed:
            desc_list = []
            for child, parent, ctype in removed:
                relation_str = f"{child} --({ctype})--> {parent}"
                desc_list.append(relation_str)
            report.append(f"【已断开】: {', '.join(desc_list)}")

        return "\n".join(report)

    def _get_vessel_diff(self, vid: str, pre_containers: dict, post_containers: dict) -> str:
        """
        [辅助] 计算单个容器的详细变化
        """
        if vid not in pre_containers and vid not in post_containers:
            return ""

        pre_c = pre_containers.get(vid, {})
        post_c = post_containers.get(vid, {})

        # 1. 核心物理量
        v_pre = pre_c.get("volume_ml", 0)
        v_post = post_c.get("volume_ml", 0)
        v_delta = v_post - v_pre
        
        # 忽略微小的浮点误差
        if abs(v_delta) < 0.05: v_delta = 0.0

        # 2. 颜色与pH
        c_pre = pre_c.get("color_desc", "N/A")
        c_post = post_c.get("color_desc", "N/A")
        ph_pre = pre_c.get("ph", 7.0)
        ph_post = post_c.get("ph", 7.0)

        # === [新增] 3. 成分指纹变化 (用于捕捉固体添加/反应) ===
        s_pre = set(pre_c.get("major_species", []))
        s_post = set(post_c.get("major_species", []))
        new_species = s_post - s_pre

        # 3. 组装文本
        change_tags = []
        if abs(v_delta) > 0.1: change_tags.append(f"体积({v_delta:+.1f}ml)")
        if c_pre != c_post: change_tags.append(f"颜色变({c_post})")
        if abs(ph_pre - ph_post) > 0.5: change_tags.append(f"pH变({ph_post:.1f})")

        # [关键] 如果体积没大变，但出现了新物质，必须报告！
        if new_species:
            change_tags.append(f"新成分({', '.join(new_species)})")

        summary = f"无明显变化" if not change_tags else ", ".join(change_tags)

        return (
            f"容器 [{vid}]:\n"
            f"  - Vol: {v_pre:.1f} -> {v_post:.1f} ml (Delta: {v_delta:+.1f})\n"
            f"  - Color: {c_pre} -> {c_post}\n"
            f"  - pH: {ph_pre:.1f} -> {ph_post:.1f}\n"
            f"  > 总结: {summary}"
        )
    
    def _analyze_chemical_changes(self, pre: Dict[str, Any], post: Dict[str, Any], step_data: Dict[str, Any]) -> str:
        """
        [NEW] 化学变化深度审计
        对比前后快照，提取颜色突变、pH突跃、沉淀生成、主要离子变化。
        """
        target_vessel = step_data.get("params", {}).get("vessel") or step_data.get("params", {}).get("to_vessel")
        if not target_vessel: return "无主要化学反应容器。"

        pre_c = pre.get("containers", {}).get(target_vessel, {})
        post_c = post.get("containers", {}).get(target_vessel, {})

        if not pre_c or not post_c: return "容器数据缺失。"

        changes = []

        # 1. 颜色突变 (Macroscopic)
        if pre_c.get("color_desc") != post_c.get("color_desc"):
            changes.append(f"🎨 颜色改变: [{pre_c.get('color_desc')}] -> [{post_c.get('color_desc')}]")

        # 2. pH 突跃 (Chemical Property)
        ph_diff = post_c.get("ph", 7) - pre_c.get("ph", 7)
        if abs(ph_diff) > 0.5:
            trend = "上升" if ph_diff > 0 else "下降"
            changes.append(f"📉 pH剧烈变化: {pre_c.get('ph', 7):.1f} -> {post_c.get('ph', 7):.1f} ({trend} {abs(ph_diff):.1f})")

        # 3. 沉淀生成 (Precipitation)
        ppt_diff = post_c.get("precipitate_g", 0) - pre_c.get("precipitate_g", 0)
        if ppt_diff > 0.01:
            changes.append(f"☁️ 产生沉淀: +{ppt_diff:.2f}g")
        
        # 4. 温度变化 (Exothermic/Endothermic)
        temp_diff = post_c.get("temperature", 25) - pre_c.get("temperature", 25)
        if abs(temp_diff) > 2.0:
            changes.append(f"🌡️ 温度波动: {temp_diff:+.1f}°C")

        # 5. 物质成分追踪 (Microscopic - Trace Major Species)
        pre_m = set(pre_c.get("major_species", []))
        post_m = set(post_c.get("major_species", []))
        new_species = post_m - pre_m
        gone_species = pre_m - post_m
        
        if new_species:
            changes.append(f"✨ 新生成物质: {', '.join(new_species)}")
        if gone_species:
            changes.append(f"💀 消耗/消失物质: {', '.join(gone_species)}")

        if not changes:
            return "⚗️ 化学状态稳定 (无明显反应现象)"
        
        return "\n".join(changes)

    def _generate_comprehensive_report(self, pre: Dict[str, Any], post: Dict[str, Any], step_data: Dict[str, Any]) -> str:
        """
        [修复版] 全局扫描：不仅检查计划内的容器，还扫描所有发生变化的容器。
        """
        if not pre or not post: return "无数据对比。"

        # 1. 拓扑审计
        topo_report = self._analyze_topology_change(pre, post)

        # 2. [关键修复] 全局容器审计
        # 不再局限于 step_data 中的 ID，而是对比前后快照中所有容器
        pre_conts = pre.get("containers", {})
        post_conts = post.get("containers", {})
        
        # 获取当前步骤的目标容器 (用于在报告中标注)
        params = step_data.get("params", {})
        target_ids = [v for k, v in params.items() if k in ["vessel", "to_vessel", "from_vessel"]]

        all_ids = set(pre_conts.keys()) | set(post_conts.keys())
        changed_reports = []
        
        for vid in all_ids:
            diff = self._get_vessel_diff(vid, pre_conts, post_conts)
            if "无明显变化" not in diff:
                # [新增] 在报告里标记这是“目标”还是“意外”
                prefix = "【目标】" if vid in target_ids else "【⚠️意外变动】"
                changed_reports.append(f"{prefix} {diff}")

        # 如果没有检测到任何变化，才尝试去强制输出目标容器的状态（为了让 LLM 看到它还是空的）
        if not changed_reports:
            target_id = step_data.get("params", {}).get("vessel") or step_data.get("params", {}).get("to_vessel")
            if target_id:
                changed_reports.append(self._get_vessel_diff(target_id, pre_conts, post_conts))

        # [NEW] 调用化学分析
        chem_report = self._analyze_chemical_changes(pre, post, step_data)

        # 3. 合并报告
        return f"""
        === 🛠️ 拓扑连接审计 ===
        {self._analyze_topology_change(pre, post)}

        === ⚗️ 核心化学反应审计 (Target Vessel) ===
        {chem_report}

        === 🧪 容器物理状态审计 (All Changes) ===
        {self._scan_all_vessel_changes(pre, post, step_data)} 
        """
        # 注: _scan_all_vessel_changes 是你原代码里 _get_vessel_diff 的循环逻辑封装

    def _scan_all_vessel_changes(self, pre: Dict[str, Any], post: Dict[str, Any], step_data: Dict[str, Any]) -> str:
        """
        [NEW] 扫描所有容器的物理变化 (体积、颜色描述字符串、pH数值)
        这是对原代码中 for 循环逻辑的封装。
        """
        pre_conts = pre.get("containers", {})
        post_conts = post.get("containers", {})
        
        # 1. 识别谁是当前步骤的“主角” (Target Vessel)
        params = step_data.get("params", {})
        # 提取涉及的所有关键容器ID
        target_ids = []
        for key in ["vessel", "to_vessel", "from_vessel", "source_vessel"]:
            if params.get(key):
                target_ids.append(params[key])

        all_ids = set(pre_conts.keys()) | set(post_conts.keys())
        changed_reports = []
        
        # 2. 遍历所有容器，寻找变化
        for vid in all_ids:
            # 调用你原有的 _get_vessel_diff 函数
            diff = self._get_vessel_diff(vid, pre_conts, post_conts)
            
            if "无明显变化" not in diff:
                # 区分是“目标容器”还是“意外容器”
                prefix = "【🎯目标变动】" if vid in target_ids else "【⚠️意外副作用】"
                changed_reports.append(f"{prefix} {diff}")

        # 3. 兜底逻辑：如果什么变化都没检测到，强制显示目标容器的状态
        # (这是为了让 LLM 知道“虽然没变化，但容器现在是什么样”，防止它瞎猜)
        if not changed_reports:
            for tid in target_ids:
                if tid in pre_conts or tid in post_conts:
                    # 强制获取一次状态，哪怕是“无变化”
                    static_status = self._get_vessel_diff(tid, pre_conts, post_conts)
                    changed_reports.append(f"【目标静止】{static_status}")

        if not changed_reports:
            return "未检测到任何物理状态变化。"
            
        return "\n".join(changed_reports)
    
    def check_goal_met_statically(self, step_data: Dict[str, Any], 
                                  current_snapshot: Dict[str, Any], 
                                  last_actions: List[Dict[str, Any]] = None) -> bool:
        """
        [v5.3 Fixed] 智能前瞻检查 (Fast-Forward Check)
        修复逻辑：如果存在“动作证据”(last_actions)，则允许成分检测失败（视为反应消耗）。
        """
        action = step_data.get("action")
        params = step_data.get("params", {})
        
        target_id = params.get("vessel") or params.get("to_vessel")
        # 如果是 Transfer，目标容器在 to_vessel
        if not target_id: return False

        target_container = current_snapshot["containers"].get(target_id)
        if not target_container: return False

        # === Case A: 增量操作 (Add, Transfer) ===
        if action in ["Add", "Transfer"]:
            # 1. 智能解析目标量
            raw_vol = params.get("volume")
            raw_mass = params.get("mass") or params.get("amount")
            
            # 使用之前的 parse_quantity 工具函数
            val_vol, unit_vol = parse_quantity(raw_vol)
            val_mass, unit_mass = parse_quantity(raw_mass)
            
            # 判断是否为固体操作 (单位是 g 或 明确有 mass 参数)
            is_solid_op = (unit_vol == 'g') or (unit_mass == 'g') or (raw_mass is not None)
            
            # 2. 获取当前状态
            curr_vol = target_container.get("volume_ml", 0)
            
            # === 分支逻辑 ===
            vol_met = False
            
            if is_solid_op:
                # [固体逻辑] 不查体积，只查成分是否存在
                # 因为计算固体的预期质量比较复杂(涉及溶解)，这里做简化：只要有这个成分就算过
                req_chem = params.get("reagent", "")
                curr_species = target_container.get("major_species", [])
                # 模糊匹配
                chem_found = any(req_chem in s or s in req_chem for s in curr_species)
                
                # 如果成分在，算过
                if chem_found: vol_met = True
                
            else:
                # [液体逻辑] 查体积 (容差 95%)
                # 注意：这里假设主要是加液体。如果是微量添加，体积变化可能很小，需要结合 evidence
                if curr_vol >= val_vol * 0.95:
                    vol_met = True

            # 3. [关键修复] 检查【动作证据】(Action Evidence)
            # 这一步是为了确认：这真的是刚才学生倒进去的，而不是杯子里原本就有的
            found_evidence = False
            if last_actions:
                for act in last_actions:
                    # 动作类型匹配
                    if act.get("action") != action: continue
                    
                    # 对象匹配
                    act_target = act.get("vessel") or act.get("to_vessel")
                    if act_target != target_id: continue
                    
                    # (Add特有) 试剂名匹配
                    if action == "Add":
                        act_reagent = act.get("reagent", "")
                        target_reagent = params.get("reagent", "")
                        # 宽松匹配 (如 "HCl" vs "Unknown_HCl")
                        if target_reagent not in act_reagent and act_reagent not in target_reagent:
                            continue

                    # 找到匹配动作！
                    found_evidence = True
                    break
            
            # 4. 成分检查 (Chemical Check)
            chem_met = True
            if action == "Add" and "reagent" in params:
                req_chem = params["reagent"]
                curr_species = target_container.get("major_species", [])
                # 只有直接找到才算 True
                chem_met = any(req_chem in s or s in req_chem for s in curr_species)

            # === [最终判定逻辑] ===
            
            # 情况 A: 有动作证据 (Best Case)
            # 只要学生做了这个动作，且没有严重报错（引擎层面），我们就认为通过
            # 即使 chem_met 为 False (反应掉了)，也给过！
            if found_evidence:
                return True
            
            # 情况 B: 无动作证据 (纯静态扫描，例如跳过了几步)
            # 这种情况下，必须【体积达标】且【成分存在】
            # 如果反应掉了，静态扫描确实会失效，但这属于“非正常流程跳跃”，判 False 也是合理的
            if not chem_met and not is_solid_op:
                return False

            # 如果是固体，且没找到证据，也没找到成分 -> False
            if is_solid_op and not chem_met:
                return False
                
            # 如果是液体，且没证据，但体积够了 -> 可能是之前的步骤加的，或者是水
            # 这里稍微放宽：如果体积够了且没有显式要求特定成分(或者成分也在)，给过
            if not is_solid_op and vol_met:
                return True

            return False

        # =========================================================
        # Case B: 状态操作 (Refill, Heat, Attach) -> 仅查状态
        # =========================================================
        # Refill: 只要满了就算过 (比如 > 450ml)
        if action == "Refill":
            return target_container and target_container.get("volume_ml", 0) > 450.0

        # Heat: 温度达标即可 (比如达到目标的 90%)
        elif action == "Heat":
            target_temp = float(params.get("target_temperature", 100.0))
            return target_container and target_container.get("temperature", 25.0) >= target_temp * 0.9

        # Attach/Insert: 检查拓扑连接
        elif action in ["Attach", "Insert"]:
            child = params.get("vessel") or params.get("tool")
            parent = params.get("support") or params.get("vessel")
            topology = current_snapshot.get("topology", [])
            return any(l['child'] == child and l['parent'] == parent for l in topology)

        return False
    
    def _generate_diff_report(self, pre, post) -> str:
        """生成精简的 Markdown 物理变化报告"""
        lines = ["### 物理状态变化报告"]
        
        all_ids = set(pre['containers'].keys()) | set(post['containers'].keys())
        has_change = False
        for vid in all_ids:
            v_pre = pre['containers'].get(vid, {}).get('volume_ml', 0)
            v_post = post['containers'].get(vid, {}).get('volume_ml', 0)
            
            # 只记录有变化的容器
            if abs(v_post - v_pre) > 0.1:
                has_change = True
                lines.append(f"- **{vid}**: 体积 {v_pre:.1f}ml -> {v_post:.1f}ml")
                
                # 检查颜色变化
                c_pre = pre['containers'].get(vid, {}).get('color_desc', '')
                c_post = post['containers'].get(vid, {}).get('color_desc', '')
                if c_pre != c_post:
                    lines.append(f"  - 颜色: {c_pre} -> {c_post}")

        if not has_change:
            return "无明显的物理变化。"
        return "\n".join(lines)
    
    # [新增] 教育学影响评估：对物理结果进行“分级”
    def assess_pedagogical_impact(self, result: InspectionResult, goal: GoalState, planner=None, last_actions=None) -> InspectionResult:
        """
        根据误差程度，调整 InspectionResult 的状态。
        将冰冷的物理 FAIL 转换为 Trivial(通过), Teachable(教学), Critical(阻断)。
        """
        # 如果已经是 PASS，直接放行
        if result.status == ReviewStatus.PASS:
            return result
        
        # 获取诊断数据
        diag = result.diagnosis or {}
        dtype = diag.get("type")
        
        # === 场景 1: Trivial Mistake (微小误差 < 2%) ===
        # 只有 "OVERSHOOT" (倒多了) 或 "INSUFFICIENT" (倒少了) 且有目标值时才计算
        if dtype in ["OVERSHOOT", "INSUFFICIENT"] and goal.expected_volume > 0:
            diff = diag.get("diff", 0.0)
            target = goal.expected_volume
            
            # 计算误差百分比
            error_ratio = diff / target
            
            # 判定阈值：2% (0.02)
            if error_ratio <= 0.02:
                # [核心逻辑] 偷偷放行：改判 PASS
                # 但在 reason 里留下“暗号”，让 Teacher 看到后表扬学生
                if "reagent" in diag or diag.get("type") == "WRONG_REAGENT":
                    return result
                new_reason = (
                    f"PASS_WITH_WARN: 误差仅 {error_ratio*100:.1f}% ({diff:.2f}ml)。"
                    "不影响实验结果。建议老师表扬学生手稳，稍微提醒视线平齐即可。"
                )
                return InspectionResult(
                    status=ReviewStatus.PASS, # <--- 强行改判
                    reason=new_reason,
                    diagnosis=None # 清空诊断，防止触发补救流程
                )

        # === 场景 2: 严重性修正 (防止 AI 滥用 CRITICAL) ===
        # 只要是可以通过“清洗容器”解决的错误，强制降级为 HIGH，严禁判定为 CRITICAL
        recoverable_errors = ["OVERSHOOT", "WRONG_VESSEL", "WRONG_REAGENT"]
        if dtype in recoverable_errors:
            if result.diagnosis:
                result.diagnosis["severity"] = "HIGH" # 标记为 HIGH 触发 Wash 链路
            else:
                result.diagnosis = {"severity": "HIGH", "type": dtype}
            return result

        # === 场景 3: Teachable Moment (普通错误) ===
        # 既不是微小误差，也不是严重事故 -> 保持原样 (FAIL)
        # 这就是苏格拉底教学的最佳时机
        return result

    # [修改] Reviewer 类内部方法
    # === Reviewer 类内部修改 ===

    def evaluate(self, history, step_index, current_snapshot, pre_snapshot, 
             goal=None, last_actions=None, planner=None, 
             translation_errors=None):
        """
        [v6.0 Deterministic] 核心评估函数
        已移除不稳定 LLM 诊断，回归纯物理引擎与规则判定。
        """
        
        # === 1. 优先处理翻译阶段的幻觉错误 ===
        if translation_errors:
            err_msg = "; ".join(translation_errors)
            return {
                "status": ReviewStatus.FAIL.value,
                "reason": f"操作指令无效：{err_msg}",
                "diagnosis": {
                    "type": "WRONG_VESSEL", 
                    "severity": "LOW",
                    "analysis": "学生尝试使用不存在的器材。"
                }
            }

        # === 2. 补救流程的“逃逸出口”检查 ===
        if planner and planner.is_remedial_active and planner.remedy_target_snapshot:
            if self._is_world_state_matched(current_snapshot, planner.remedy_target_snapshot):
                planner.try_exit_remedy()
                return {
                    "status": ReviewStatus.PASS.value,
                    "reason": "检测到物理状态已提前达到主线目标，补救流程自动结束。",
                    "diagnosis": {}
                }

        # === 3. 物理引擎检查 (Inspector) ===
        # 获取精确的体积、pH、成分偏差，这是唯一的“事实来源”
        raw_result = self.inspector.inspect(current_snapshot, pre_snapshot, goal, last_actions=last_actions)
        
        # === 4. 教育学影响评估 ===
        # 基于配置的容差（tolerance）判断微小误差是否算通过
        result = self.assess_pedagogical_impact(raw_result, goal, planner=planner, last_actions=last_actions)
        
        # === 5. 纯规则逻辑增强 (仅在失败时触发) ===
        if result.status == ReviewStatus.FAIL:
            
            # [Python逻辑] 容器错位检测
            # 这是一个纯代码检查，确定性高，保留下来用于丰富错误提示
            vessel_mismatch_warning = ""
            goal_vessel = goal.target_vessel_id if goal else None
            
            if last_actions and goal_vessel:
                # === [修改] 智能过滤中间步骤 ===
                # 只有当每一个动作都跟目标无关时，才报错。
                # 或者：只检查是否有任何一个动作成功作用于目标容器。
                
                acted_on_target = False
                mismatch_vessels = set()
                
                for act in last_actions:
                    act_vessel = act.get("vessel") or act.get("to_vessel")
                    if act_vessel == goal_vessel:
                        acted_on_target = True
                    elif act_vessel:
                        mismatch_vessels.add(act_vessel)
                
                # 只有当完全没有触及目标容器，且操作了其他容器时，才报警
                if not acted_on_target and mismatch_vessels:
                    vessel_str = ", ".join(list(mismatch_vessels))
                    vessel_mismatch_warning = f" (检测到操作容器为 [{vessel_str}]，但目标容器应为 [{goal_vessel}])"
                    
                    if not result.diagnosis.get("type"):
                        result.diagnosis["type"] = "WRONG_VESSEL"
                        result.diagnosis["severity"] = "HIGH"
            
            # 将错位信息追加到原因中，供前端或 System Prompt 使用
            if vessel_mismatch_warning:
                result.reason += vessel_mismatch_warning

            # [Python逻辑] 关键成分缺失强制判定
            # 防止物理引擎只报 "Volume Mismatch" 而忽略了化学本质
            if "关键成分" in result.reason and "缺失" in result.reason:
                if goal and goal.target_vessel_id:
                    self.logger.warning(f"Critical Component Missing in {goal.target_vessel_id}.")
                    result.diagnosis["type"] = "WRONG_REAGENT" # 明确标记为试剂错误
                    result.diagnosis["target_vessel"] = goal.target_vessel_id
                    result.diagnosis["severity"] = "HIGH"

        # === 6. 直接返回结果 ===
        return {
            "status": result.status.value,
            "reason": result.reason,
            "diagnosis": result.diagnosis
        }

    def _is_world_state_matched(self, current: Dict[str, Any], target: Dict[str, Any]) -> bool:
        """
        深度状态比对逻辑。
        """
        cur_conts = current.get("containers", {})
        tar_conts = target.get("containers", {})
        
        # 检查每个在理想态中定义过的容器
        for vid, tar_data in tar_conts.items():
            cur_data = cur_conts.get(vid)
            if not cur_data: return False
            
            # 1. 体积比对 (允许 3% 的绝对/相对误差)
            vol_err = abs(cur_data['volume_ml'] - tar_data['volume_ml'])
            if vol_err > (tar_data['volume_ml'] * 0.03 + 0.1):
                return False
                
            # 2. 化学成分指纹比对 (主要离子/分子种类必须完全一致)
            if set(cur_data['major_species']) != set(tar_data['major_species']):
                return False
                
            # 3. pH 比对 (容差 0.5)
            if abs(cur_data['ph'] - tar_data['ph']) > 0.5:
                return False
        
        return True
    # =========================================================
    #  [Upgrade] 深度诊断模块 v2.0
    # =========================================================

    def _get_student_intent(self, history: List[Dict[str, str]]) -> str:
        """从历史记录中提取学生最近一次表达的意图"""
        if not history: return "无记录"
        
        # 倒序查找最近一条 Role 为 student 的消息
        for msg in reversed(history):
            if msg.get("role").lower() == "student":
                return msg.get("content", "")
        return "未表达明确意图"
    
    @RETRY_RULE
    def _call_diagnosis_llm(self, msgs):
        return self.client.chat.completions.create(
            model=Config.MODEL_REVIEWER,
            messages=msgs,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=800
        )

    def _diagnose_with_llm(self, history, step_data, rule_reason, diff_report, current_snapshot, last_actions=None, known_error=None):
    
        # === 1. 数据清洗与上下文构建 ===
        
        # A. 获取标准答案 (Ground Truth)
        # 假设 step_data 包含 target_vessel, target_reagent, target_volume 等字段
        target_info = step_data.get("desc", "无详细目标描述")

        # B. 获取学生实际行为 (Reality)
        spoken_intent = self._get_student_intent(history)
        
        action_str = "【无物理动作】(学生仅进行了口头表达)"
        if last_actions:
            # 将复杂的动作对象简化为关键信息
            simple_acts = []
            for act in last_actions:
                # 简化显示，只取关键字段
                params = {k:v for k,v in act.items()}
                simple_acts.append(f"{act.get('action')}{json.dumps(params, ensure_ascii=False)}")
            action_str = " -> ".join(simple_acts)

        # === 2. 构建诊断 Prompt ===
        
        prompt = f"""
        你是一位化学实验诊断专家。
        系统检测到学生当前步骤未通过 (Status: FAIL/WAIT)。你需要对比【标准目标】与【实际行为】，输出精准的诊断报告。

        ### 🔍 第一部分：案发现场数据
        
        【1. 教学目标 (Target)】
        {target_info}

        【2. 学生实际表现 (Actual)】
        - 口头意图: "{spoken_intent}"
        - 物理动作流: {action_str}
        - 物理状态变化报告: 
        {diff_report}
        
        【3. 系统报错/状态描述】
        - 底层检测结果: "{rule_reason}"
        - 已知程序异常: "{known_error if known_error else '无'}"

        ---------------------------------------------------

        ### 🧠 第二部分：诊断逻辑链 (推理必读)
        
        请按以下优先级顺序进行排查，**一旦命中即停止**：

        1. **检查动作类型 (INVALID_ACTION)**:
        - 学生是否执行了根本不存在的操作？(如：试图用烧杯去加热滴定管)
        
        2. **检查容器对象 (WRONG_VESSEL)**: [高优先级]
        - **判断标准**: 学生操作的 `容器ID` 是否与 Target 中的 `目标容器` 一致？
        - **陷阱警示**: 如果学生在错误的容器里加了正确的试剂，这是**严重的容器错误**，绝不是“量不够”！
        
        3. **检查试剂/对象 (WRONG_REAGENT)**:
        - 容器对了，但是加的东西对吗？(如：本该加 NaOH，却加了 HCl)
        
        4. **检查数值/程度 (INSUFFICIENT / OVERSHOOT)**:
        - 容器、试剂都对，唯独量不对。
        - 比目标少 -> INSUFFICIENT
        - 比目标多 -> OVERSHOOT
        
        5. **检查不可逆后果 (IRREVERSIBLE_POLLUTION)**:
        - 之前的操作是否导致试剂被污染，无法继续使用？

        ---------------------------------------------------

        ### 📝 第三部分：输出要求
        
        请输出严格的 JSON 格式 (不要包含 markdown 代码块标记)：
        {{
            "analysis": "简述发生了什么（客观事实）。例如：学生试图向烧杯A中加入盐酸，但实际操作对象却是锥形瓶B。",
            "root_cause_explanation": "推测原因。例如：学生可能混淆了反应容器和废液缸。",
            "diagnosis": {{
                "type": "必须从 [WRONG_VESSEL, WRONG_REAGENT, OVERSHOOT, INSUFFICIENT, IRREVERSIBLE_POLLUTION, NO_ACTION, SYSTEM_ERROR] 中选择一个",
                "severity": "LOW (可微调) / MEDIUM (需重试) / HIGH (需清洗或重置)",
                "key_discrepancy": "简短指出哪里不一致 (如: Vessel Mismatch)"
            }},
            "suggested_remedy": "给老师的指导建议。如果是操作错误，请建议老师指出具体差异；如果是数值错误，请建议提供具体差值。"
        }}
        """
    

        try:
            # 扩大历史窗口，获取更多上下文 (最近3轮: 老师->学生->系统反馈)
            recent_history = history[-3:] 
            
            # 构造消息
            msgs = [{"role": "system", "content": prompt}] + recent_history

            resp = self._call_diagnosis_llm(msgs)
            
            result = json.loads(resp.choices[0].message.content)
            
            # [关键]：直接把 LLM 的结果原封不动地返回，并补上 status
            # 这里我们把所有 AI 生成的字段都放在顶层，方便 Teacher 读取
            return {
                "status": "FAIL",
                "reason": rule_reason, # 原始物理报错
                "ai_diagnosis": result # <--- 将整个 JSON 包在这里传出去
            }

        except Exception as e:
            self.logger.error(f"LLM Diagnosis failed: {e}", exc_info=True)
            # 兜底返回
            return {
                "status": "FAIL", 
                "reason": rule_reason, 
                "diagnosis": {"type": "UNKNOWN"}
            }
        
    def _format_state_for_prompt(self, snapshot: Dict[str, Any], step_data: Dict[str, Any]) -> str:
        """
        辅助函数：将复杂的 snapshot 过滤并格式化为自然语言。
        只提取与当前步骤相关的容器信息，节省 Token。
        """
        if not snapshot:
            return "【传感器无数据】(可能是纯对话环节)"

        text = []
        
        # 1. 提取拓扑关系 (针对 Attach 操作)
        topology = snapshot.get("topology", {})
        if topology:
            # 将字典转换为自然语言描述
            connections = [f"[{child}] 已固定在 [{parent}] 上" for child, parent in topology.items()]
            text.append(f"★ 器材连接状态 (Topology): {', '.join(connections)}")
        else:
            text.append("★ 器材连接状态 (Topology): 无任何连接")

        # 2. 提取容器状态
        # 为了避免 Prompt 过长，我们优先提取 step_data 里提到的容器
        target_vessel = step_data.get("params", {}).get("vessel")
        all_containers = snapshot.get("containers", {})

        # 如果是 Attach 操作，还需要关注 support
        support_id = step_data.get("params", {}).get("support")
        
        relevant_vessels = []
        if target_vessel: relevant_vessels.append(target_vessel)
        if support_id and support_id in all_containers: relevant_vessels.append(support_id)
        
        # 如果列表为空，则显示所有（防止漏看）
        if not relevant_vessels:
            relevant_vessels = list(all_containers.keys())

        for vid in relevant_vessels:
            data = all_containers.get(vid)
            if data:
                # 格式化单个容器数据
                v_info = (
                    f"- 容器 [{vid}]: "
                    f"体积={data.get('volume_ml')}ml, "
                    f"pH={data.get('ph')}, "
                    f"温度={data.get('temperature')}°C, "
                    f"外观='{data.get('color_desc')}'"
                )
                text.append(v_info)
        
        return "\n".join(text)

# ==========================================
# 模块: Teacher & Student (Agents)
# ==========================================
class TeacherAgent:
    def __init__(self, client, memory: ExperienceMemory, oracle: Oracle):
        self.client = client
        self.memory = memory
        self.oracle = oracle
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 内部状态：记录当前步骤的尝试次数
        self.current_step_attempts = 0
        self.last_step_name = ""
        self.step_attempts_map = {}
        self.current_strategy = "SOCRATIC_STANDARD"

    # =========================================================================
    # [修改] 内部状态更新 (增加 intent 参数)
    # =========================================================================
    def _update_internal_state(self, plan_step: str, review_status: str, student_intent: str):
        """
        [v2.0 Fixed] 使用字典记录每个步骤的历史尝试次数，防止因补救步骤插入导致主线计数丢失。
        """
        # 初始化该步骤的计数（如果不存在）
        if plan_step not in self.step_attempts_map:
            self.step_attempts_map[plan_step] = 0

        if review_status == "PASS":
            # 只有真正通过了，才重置该步骤的计数 (或者直接从字典删除以节省内存)
            self.step_attempts_map[plan_step] = 0
            self.logger.info(f"Step '{plan_step}' Passed. Counter reset.")
        
        elif review_status == "WAIT":
            # WAIT 状态下：如果是提问，不消耗耐心；如果是发呆/瞎操作，消耗耐心
            if student_intent == "QUESTION":
                self.logger.info(f"Student is asking a question. Patience preserved for '{plan_step}'.")
            else:
                self.step_attempts_map[plan_step] += 1
                self.logger.info(f"Patience consumed for '{plan_step}': {self.step_attempts_map[plan_step]}")
                
        elif review_status == "FAIL":
            # FAIL 状态：直接增加计数
            self.step_attempts_map[plan_step] += 1
            self.logger.info(f"Step '{plan_step}' Failed. Attempts: {self.step_attempts_map[plan_step]}")


    def reset_patience(self):
        """
        [v2.0 Update] 重置所有步骤的耐心值记录。
        通常在系统回滚(Rollback)或致命错误重置时调用，给学生一个"重新做人"的机会。
        """
        self.logger.info("Resetting teacher patience (Clearing Step Attempts Map) due to World Reset.")
        
        # 1. 清空字典记录 (核心修改)
        # 这确保了之前所有步骤的错误计数都被归零
        self.step_attempts_map.clear()
        
        # 2. 重置旧变量 (兼容性兜底)
        # 防止代码中还有遗漏的地方引用了这个旧变量
        self.current_step_attempts = 0

    # =========================================================================
    # [修改] 策略选择 (增加 intent 参数)
    # =========================================================================
    def _determine_strategy(self, review_status: str, observation: str, 
                        student_intent: str, plan_step: str, phase_type: str = "MAINLINE") -> str:
         # 新增判断：如果是总结阶段，直接进入总结模式
        if plan_step == "实验总结与反思":
            return "FINAL_SUMMARY"
        # # 1. 安全熔断 (最高优先级)
        # danger_keywords = ["爆炸", "碎裂", "燃烧", "腐蚀", "有毒", "危险", "破裂"]
        # if any(k in observation for k in danger_keywords):
        #     return "SAFETY_INTERVENTION"

       # 2. [新增] 补救阶段专用策略
        # 如果当前是补救步骤（如“清洗烧杯”），不管尝试了几次，直接用补救指导模式
        if phase_type == "REMEDIAL":
            return "REMEDIAL_GUIDANCE"
        
        # 获取当前步骤的累积尝试次数 (从字典取值)
        current_attempts = self.step_attempts_map.get(plan_step, 0)
        # 3. 耐心耗尽 -> 强制脚手架
        # 现在即使中间插了10个补救步骤，回到主线时，current_attempts 依然保留着之前的值
        if current_attempts >= 3:
            return "SCAFFOLDING"

        # 4. 状态分支逻辑
        if review_status == "STUCK":
            return "SCAFFOLDING"
            
        elif review_status == "FAIL":
            if current_attempts < 2:
                return "SOCRATIC_DEBUG"  # 第一次错：引导反思
            else:
                return "GUIDANCE"        # 第二次错：给线索 (其实上面 >=2 已经拦截了，这里作为兜底)
                
        elif review_status == "WAIT":
            if student_intent == "QUESTION":
                return "ANSWER_AND_GUIDE" 
            
            # 第一次只说不做 -> 搭桥
            if current_attempts < 1:
                return "SOCRATIC_BRIDGE" 
            # 多次只说不做 -> 催促
            else:
                return "URGE_ACTION"
                
        else: # PASS / INFO
            return "SOCRATIC_STANDARD"
        
    # =========================================================================
    # [新增方法] 将 Reviewer 的结构化诊断转化为具体的教学指令 (Micro-Prompting)
    # =========================================================================
    def _generate_pedagogical_insight(self, diagnosis: Dict[str, Any], status: str, reason: str = "") -> str:
        """
        核心连接器：解析 diagnosis 字典，生成针对性的教学话术指导。
        [修改] 增加了 reason 参数，用于捕获 Trivial Mistake 的提示
        """
        # === 场景 A: Trivial Mistake (微小误差) ===
        # 识别暗号：Status 是 PASS，但 Reason 里有 "WARN"
        if status == "PASS":
            if "PASS_WITH_WARN" in reason:
                return (
                    "【教学指令 - 正向强化】\n"
                    "学生虽然通过了，但有一点微小误差（<2%）。\n"
                    "1. 请肯定他的操作：'做得很好，量取非常接近。'\n"
                    "2. 顺带提醒优化点：'下次注意视线完全平齐，就能做到完美。'\n"
                    "3. 允许继续下一步。"
                )
            else:
                return "当前操作符合预期。请根据苏格拉底策略引导学生总结或思考下一步。"

        # === 场景 B: Critical Failure (严重事故) ===
        severity = diagnosis.get("severity", "LOW")
        if severity == "CRITICAL":
            return (
                "【🛑 紧急安全阻断】\n"
                "检测到严重实验事故（不可逆损失或严重过量）。\n"
                "1. **立即停止**苏格拉底式提问。\n"
                "2. 严肃指出错误后果（如废液溢出、产物报废）。\n"
                "3. 发出明确指令进行补救（如：'请立刻停止操作，我们需要重新清洗容器'）。"
            )

        dtype = diagnosis.get("type", "UNKNOWN")
        vessel = diagnosis.get("vessel") or diagnosis.get("target_vessel", "容器")
        
        # 1. 液体倒多了 (Overshoot)
        if dtype == "OVERSHOOT":
            diff = diagnosis.get("diff", 0)
            return (
                f"【教学指令】学生在 {vessel} 中多加了约 {diff:.1f}ml 液体。\n"
                f"1. 严禁直接叫学生“重做”。\n"
                f"2. 请提问引导后果分析：'现在的液面高度似乎超过了预期，你觉得这会让最终的浓度计算值偏大还是偏小？'\n"
                f"3. 只有当学生意识到错误后，再讨论如何处理多余液体。"
            )

        # 2. 液体倒少了 (Insufficient)
        elif dtype == "INSUFFICIENT":
            diff = diagnosis.get("diff", 0)
            return (
                f"【教学指令】学生在 {vessel} 中还差约 {diff:.1f}ml 液体。\n"
                f"1. 肯定他的操作方向是对应的。\n"
                f"2. 温和提示：'看起来还没有达到刻度线/目标量，我们需要继续添加吗？要加多少？'"
            )

        # 3. 加错试剂 (Wrong Reagent)
        elif dtype == "WRONG_REAGENT":
            missing = diagnosis.get("missing", [])
            missing_str = ", ".join(missing) if isinstance(missing, list) else str(missing)
            return (
                f"【教学指令】容器 {vessel} 中检测不到 {missing_str}，可能加错了试剂。\n"
                f"1. 不要直接说'你加错了'。\n"
                f"2. 引导观察：'请仔细看看你刚才用的试剂瓶标签，或者观察一下现在液体的颜色/反应现象，这和书本上描述的一样吗？'"
            )

        # 4. 用错容器 (Wrong Vessel)
        elif dtype == "WRONG_VESSEL":
            return (
                f"【教学指令】学生把试剂加到了错误的容器 {vessel} 里。\n"
                f"1. 引导反查步骤：'请停下来回顾一下实验步骤，我们这一步的目标容器真的是这个吗？'"
            )
            
        # 5. 简单缺失 (Simple Missing - 比如洗得太干净了)
        elif dtype == "SIMPLE_MISSING":
             return "【教学指令】检测到容器中缺少必要成分（可能是清洗过或未添加）。请提示学生重新添加所需试剂。"

        return f"【教学指令】检测到操作异常 ({dtype})。请引导学生观察当前现象与预期的差异。"
    

    def _detect_student_intent(self, history: List[Dict[str, str]]) -> str:
        """
        [BERT版] 判断学生上一句话的意图。
        """
        if not history: return "STATEMENT"
        
        # 1. 获取学生最后一条消息
        last_msg = history[-1]
        if last_msg.get("role") != "user":
            return "STATEMENT"
            
        content = last_msg.get("content", "").strip()
        
        # 2. 调用 BERT 分类器
        # 这里使用全局单例，避免每次调用都重新加载模型
        # 如果是同文件内定义的类：
        intent = INTENT_CLASSIFIER.predict(content)
        
        self.logger.info(f"Intent Detection: '{content}' -> {intent}")
        return intent
    
    # [新增] 将物理快照转化为自然语言描述
    def _format_lab_state(self, snapshot: Dict[str, Any]) -> str:
        if not snapshot:
            return "（实验台数据缺失）"

        containers = snapshot.get("containers", {})
        lines = []
        
        # 1. 容器状态
        has_content = False
        for vid, data in containers.items():
            vol = data.get("volume_ml", 0)
            if vol <= 0.01: continue 
            
            has_content = True
            species_str = json.dumps(data.get('major_species', []), ensure_ascii=False)
            # === 修改点：格式与学生保持一致，甚至更详细 ===
            info = (
                f"- 【{vid}】: "
                f"Vol={vol:.1f}ml | "
                f"pH={data.get('ph', 7):.1f} | "
                f"Color='{data.get('color_desc', '无色')}' | "
                f"Species={species_str}" # <--- 这里原来是直接 str()，会带单引号
            )
            lines.append(info)
            
        if not has_content:
            lines.append("（所有容器均为空）")

        # 2. 连接状态
        topology = snapshot.get("topology", [])
        if topology:
            lines.append("【连接状态】:")
            for link in topology:
                lines.append(f"- {link['child']} -> {link['parent']} ({link['type']})")

        return "\n".join(lines)
    
    @RETRY_RULE
    def _call_teacher_llm(self, messages):
        return self.client.chat.completions.create(
            model=Config.MODEL_TEACHER,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1, # [修改] 稍微提高一点温度，避免死板地陷入死循环
            max_tokens=1000
        )
    
    def _safe_parse_json(self, llm_output: str, fallback_thought: str, fallback_response: str):
        """
        [v2.0 Fixed] 健壮的 JSON 解析器
        1. 尝试去除 Markdown 代码块标记 ```json ... ```
        2. 使用正则寻找最外层的 {}
        3. 如果解析失败，返回预设的 Fallback 结构，绝不返回原始乱码
        """
        try:
            # 1. 预处理：去除可能的 markdown 标记
            text = llm_output.strip()
            #有些模型喜欢在json前面加一些废话，或者用 ```json 包裹
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            
            # 2. 正则提取：寻找第一个 { 和最后一个 }
            # re.DOTALL 让 . 可以匹配换行符，防止多行 JSON 匹配失败
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                # 尝试直接解析（应对某些不带大括号的极端情况，虽少见）
                return json.loads(text)
                
        except Exception as e:
            # self.logger.warning(f"JSON Parse Failed: {e}. Output was: {llm_output[:50]}...")
            # 解析失败时，返回一个构造好的安全字典，而不是让 garbage text 泄露出去
            return {
                "thought": f"系统解析错误 (Fallback): {fallback_thought}",
                "response": fallback_response
            }
    
    # =========================================================================
    # [修改方法] Respond: 接收完整 review_data 并传递给 Prompt 构建器
    # =========================================================================
    # [修改] 增加 current_snapshot 参数
    def respond(self, history, plan_step, observation, review_data, 
                phase_type, current_snapshot, available_hardware=None, 
                last_plan_step=None, last_phase_type=None): # [新增参数]
        
        # 1. 解包
        status = review_data.get("status", "FAIL")
        reason = review_data.get("reason", "")
        # [修复] 使用 or {} 确保即使值为 None 也能转为空字典
        ai_diagnosis_data = review_data.get("ai_diagnosis") or {}

        # 1. [新增] 检测学生意图
        student_intent = self._detect_student_intent(history)

        # 兼容旧逻辑：如果没有 AI 诊断，尝试用 review_data 里的 diagnosis
        simple_diagnosis = review_data.get("diagnosis") or {}
        
        # 2. [修改] 更新状态 (传入 intent)
        self._update_internal_state(plan_step, status, student_intent)
        # 3. [修改] 决定策略 (传入 intent)
        strategy_mode = self._determine_strategy(status, observation, student_intent, plan_step)

        # [核心优化] 如果处于补救阶段，强制切换策略为 "REMEDIAL_GUIDANCE"
        # 除非涉及安全熔断 (SAFETY_INTERVENTION 优先级最高)
        if phase_type == "REMEDIAL" and strategy_mode not in ["SAFETY_INTERVENTION", "SCAFFOLDING"]:
            strategy_mode = "REMEDIAL_GUIDANCE"

        self.current_strategy = strategy_mode

        # # 3. RAG 检索
        # memory_context = ""
        # if self.memory and strategy_mode in ["GUIDANCE", "SCAFFOLDING"]:
        #     query = f"教学策略检索: 如何解决学生在 '{plan_step}' 环节遇到的 '{reason}' 问题？"
        #     try:
        #         memory_context = self.memory.retrieve_strategy(query)
        #     except Exception:
        #         memory_context = ""

        ground_truth = self.oracle.get_ground_truth_prompt() 

        # 1. 生成状态描述
        lab_state_str = self._format_lab_state(current_snapshot)
        attempts = self.step_attempts_map.get(plan_step, 0)

        history_str = format_history_to_string(history)

        # 2. 传入 _build_system_prompt
        system_prompt = self._build_system_prompt(
            strategy_mode=strategy_mode,
            plan_step=plan_step,
            last_plan_step=last_plan_step,  # 上一轮目标 [新增]
            ground_truth=ground_truth,
            dialogue_history_str=history_str, # <--- 传入字符串
            # memory_context=memory_context, # 确保使用了 RAG 检索结果
            review_reason=review_data.get("reason", ""),
            status=review_data.get("status", "FAIL"),
            phase_type=phase_type,
            ai_data=review_data.get("ai_diagnosis", {}), # 传入 AI 诊断
            current_lab_state=lab_state_str,
            available_hardware=available_hardware,
            attempts=attempts # <--- [修复] 必须显式传入这个内部计数器
        )

        messages = [{"role": "system", "content": system_prompt}]

        # 日志记录 (保持不变)
        log_turn_event("Dialogue", "Teacher_LLM_Request", {
            "strategy_mode": strategy_mode,
            "review_status": status,
            "diagnosis_type": ai_diagnosis_data.get("type", "N/A")
        })
        

        # === [准备兜底数据] (提前定义，供最后失败时使用) ===
        fallback_map = {
            "URGE_ACTION": f"请立刻执行：{plan_step}。",
            "SCAFFOLDING": f"请直接对 {plan_step} 进行操作。",
            "SAFETY_INTERVENTION": "请注意安全，停止操作。",
            "default": f"我们要继续进行：{plan_step}。"
        }
        safe_fallback = fallback_map.get(strategy_mode, fallback_map["default"])
        
        # === [新增] JSON 错误重试循环 ===
        max_retries = 3
        content_str = ""  # 用于存储最后一次原本的模型输出

        for attempt in range(max_retries):
            try:
                # 1. 调用 LLM
                resp = self._call_teacher_llm(messages)
                content_str = resp.choices[0].message.content

                # 2. 尝试验证 JSON 格式 (利用 json.loads 进行快速检查)
                #    注意：这里做一个简单的清洗，防止 ```json 包裹导致直接 loads 失败
                check_str = content_str.strip()
                if check_str.startswith("```"):
                    # 简单去除 markdown 标记
                    check_str = check_str.replace("```json", "").replace("```", "").strip()
                
                # 如果这一步报错，说明格式非法，直接跳到 except
                json.loads(check_str) 

                # 3. 如果验证通过，直接使用原有逻辑解析并返回
                content = self._safe_parse_json(
                    llm_output=content_str,
                    fallback_thought="解析成功", # 这里其实不会用到了
                    fallback_response=safe_fallback
                )
                self.logger.info(f"[Teacher Thought]: {content.get('thought')}")
                return content 

            except (json.JSONDecodeError, Exception) as e:
                # 记录警告
                self.logger.warning(f"[Retry {attempt+1}/{max_retries}] JSON Parse Failed: {str(e)}")
                
                # 如果还有重试机会，将错误反馈给模型
                if attempt < max_retries - 1:
                    # 将错误的回复加入历史作为 Context
                    messages.append({"role": "assistant", "content": content_str})
                    # 追加明确的纠错指令
                    messages.append({
                        "role": "user", 
                        "content": f"系统提示：检测到 JSON 格式错误（{str(e)}）。请修正格式，仅输出标准的 JSON，不要输出其他文字。"
                    })
                else:
                    # 次数用尽，不做操作，代码自然向下执行到兜底逻辑
                    self.logger.error("Max retries reached. Falling back.")

        # === [原有兜底逻辑] (只有循环跑完没 return 才会走到这里) ===
        try:
            # 这里的 content_str 是最后一次失败的输出
            content = self._safe_parse_json(
                llm_output=content_str,
                fallback_thought="模型输出解析失败或包含非转义字符 (重试后仍失败)",
                fallback_response=safe_fallback
            )
            return content 
        except Exception:
            self.logger.error("Teacher agent failed completely", exc_info=True)
            return {"thought": "系统严重错误", "response": safe_fallback}

    # 建议修改定义，将所有非核心参数设为默认 None
    def _build_system_prompt(self, strategy_mode, plan_step, 
                         last_plan_step=None, 
                         memory_context=None, 
                         ground_truth=None, 
                         review_reason=None, 
                         status="WAIT", 
                         phase_type="MAINLINE", 
                         ai_data=None, 
                         current_lab_state="", 
                         attempts=0, 
                         dialogue_history_str="",  # <--- [🔥 新增参数]
                         available_hardware=None):

        # === 1. 数据清洗与基础信息构建 ===
        # 简化状态描述，供 Context 使用
        status_desc = {
            "PASS": "SUCCESS (上一步已完成)",
            "FAIL": "FAILURE (上一步操作错误)",
            "WAIT": "IN_PROGRESS (正在进行)",
            "INFO": "IDLE (等待指令)"
        }.get(status, "UNKNOWN")
        
        safe_review_reason = review_reason if review_reason else "无"
        
        # 硬件限制文本
        hw_text = f"[{', '.join(available_hardware)}]" if available_hardware else "无特殊限制"

        # === 2. 动态场景构建 (Context Builder) ===
        # 将复杂的 if-else 逻辑收敛为明确的“当前情境描述”
        situation_desc = ""
        
        if last_plan_step and plan_step and last_plan_step != plan_step and status == "PASS":
            situation_desc = f"【阶段切换】学生刚完成了步骤 '{last_plan_step}'，现在系统自动推进到了新步骤 '{plan_step}'。请先简短肯定上一步，然后立刻发布新指令。"
        elif status == "FAIL":
            situation_desc = f"【操作报错】学生在执行 '{last_plan_step}' 时出错。原因：{safe_review_reason}。当前任务是指出错误并引导学执行 '{plan_step}'。"
        else:
            situation_desc = f"【正常进行】学生正在尝试步骤 '{plan_step}'。请根据观察到的实验台状态进行引导。"

        # 1. 提取错误类型
        error_type = ai_data.get("type", "UNKNOWN") if ai_data else "UNKNOWN"

        if error_type == "TOPOLOGY_ERROR":
            debug_instruction = """
            - **当前错误是【连接/位置错误】**。
            - ⛔ **严禁**询问“溶液颜色”、“反应现象”等化学问题（因为根本没反应，问了会导致学生幻觉）。
            - ✅ **请引导**：让学生顺着导管检查连接端点，或对比图纸确认器材位置。
            - 话术示例：“请仔细看看，这根导管的末端真正接到了哪里？这符合装置图的要求吗？”
            """
        elif error_type in ["OVERSHOOT", "INSUFFICIENT"]:
            debug_instruction = """
            - **当前错误是【数值偏差】**。
            - ⛔ **不要**直接告诉学生“你倒多了/少了”。
            - ✅ **请引导**：让学生重新读数，注意视线平齐。
            - 话术示例：“请再次平视刻度线，现在的液面高度读数是多少？这和目标值一致吗？”
            """
        elif error_type == "WRONG_REAGENT":
            debug_instruction = """
            - **当前错误是【加错试剂】**。
            - ⛔ **不要**直接说“你加错了”。
            - ✅ **请引导**：让学生检查试剂瓶标签，或观察加入后的异常现象（如颜色不对）。
            - 话术示例：“请确认一下你刚才拿起的试剂瓶标签写着什么？现在的反应现象符合预期吗？”
            """
        else:
            # 默认兜底
            debug_instruction = f"""
            - 引导学生自己发现 '{safe_review_reason}' 这个问题。
            - 既然学生没发现，说明他忽略了某个细节，请用反问句提示该细节。
            """
        # === 3. 策略库 (Strategy Registry) ===
        # 使用字典管理策略，提高可读性和扩展性
        strategies = {
            "REMEDIAL_GUIDANCE": """
            [策略：🔧 严格纠错]
            1. 语气严肃，停止表扬。
            2. 必须明确指出具体的物理错误事实（如“瓶子没洗”）。
            3. 给出直接的纠正指令，不要反问,让学生执行{plan_step}。
            """,
            "SCAFFOLDING": f"""
            [策略：🪜 脚手架兜底]
            1. 既然已经尝试了 {attempts + 1} 次，停止启发式提问。
            2. 直接给出答案级别的“手把手”指令,让学生执行{plan_step}。
            """,
            "SAFETY_INTERVENTION": """
            [策略：🛑 安全熔断]
            1. 立即叫停实验，语气严厉。
            2. 解释该操作在真实世界的危险后果。
            3. 要求学生复位或清洗。
            """,
            "SOCRATIC_DEBUG": f"""
            [策略：🕵️ 引导反思 (针对性)]
            {debug_instruction}
            """,
            "ANSWER_AND_GUIDE": """
            [策略：🧠 解答并回归]
            1. 专业解答学生的理论问题。
            2. 话锋一转：“这个原理正好对应我们现在的操作...”，拉回实验步骤。
            """,
            "URGE_ACTION": f"""
            [策略：👋 强力催促]
            1. 禁止废话和寒暄。
            2. 强制输出简短指令：“请立刻执行：{plan_step}”。
            """,
            "FINAL_SUMMARY": """
            [策略：🎓 结课总结]
            1. 肯定探究精神。
            2. 总结核心原理。
            3. 抛出一个延伸思考题。
            """,
            "SOCRATIC_STANDARD": f"""
            [策略：🎓 标准苏格拉底]
            1. 针对 '{plan_step}' 进行启发式引导。
            2. 如果学生做对了但没做全，用问题提示漏掉的细节。
            3. 严禁直接给出操作答案。
            """
        }
        
        current_strategy = strategies.get(strategy_mode, strategies["SOCRATIC_STANDARD"])
        # [新增] 动态调整对话限制
        conversational_constraint = """
        【对话风格指南】
        - 像真正的师生聊天一样。如果学生表现得很自信，你可以稍微调皮一点；如果学生很困惑，请表现得更温柔。
        - 避免使用：'接下来我们需要...'、'本步骤的目标是...' 等过于书面语的开场白。
        - 如果当前是补救模式，表现得更像是在陪学生解决难题的伙伴，而不是发布命令的机器人。
        """

        # === 4. 最终 Prompt 组装 (Structured Format) ===
        # 采用结构化标记，逻辑分层
        final_prompt = f"""
        # Role
        你是一位拥有“上帝视角”的高中化学老师 (SocraticChem Teacher)。
        你必须基于【实验台实时快照】的物理事实进行指导，严禁依赖学生的口头描述。

        # 📜 Dialogue History (近期对话记录)
        {dialogue_history_str}

        # Current Context (当前上下文)
        <status>
        - 当前目标步骤: "{plan_step}"
        - 尝试次数: 第 {attempts + 1} 次尝试当前目标步骤
        - 场景判定: {situation_desc}
        </status>

        <lab_snapshot>
        {current_lab_state}
        </lab_snapshot>

        <hardware_constraints>
        可用器材: {hw_text} (若学生要求使用列表外器材，明确告知不存在)
        </hardware_constraints>

        # Operational Rules (行为准则)
        1. **事实优先**: 如果 <lab_snapshot> 显示容器脏或量不对，即使学生说“我洗了”，也必须判定为未清洗。
        2. **数值严谨**: 涉及体积/质量时，严格遵守目标步骤数值，禁止模糊处理。
        3. **成功判定**: 若 status 为 PASS，说明物理状态已达标，**必须**无视学生口误，直接推进{plan_step}。
        4. **错误处理**: 若 status 为 FAIL，必须遵循：指出客观现象 -> 解释危害 -> 给出纠正指令,推进{plan_step}。
        5. **策略保持**: 请严格遵守当前策略。

        # Instructional Strategy (当前策略)
        {current_strategy}

        # Dialogue Style
        {conversational_constraint}
        # Output Format
        请严格输出以下 JSON 格式，不要包含 Markdown 代码块标记：
        {{
            "thought": "简短分析：观察到的物理事实 -> 学生的意图 -> 决定采用的话术策略",
            "response": "对学生说的话 (口语化，不要带格式，不要带'老师说'的前缀)"
        }}
        """
        return final_prompt

class StudentAgent:
    def __init__(self, client, persona_config: Dict[str, Any] = None):
        self.client = client
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 1. 更加丰富的配置化人设
        # 默认配置
        default_config = {
            "name": "新手小明",
            "traits": ["急躁", "不喜欢看说明书", "过度自信"],
            "misconceptions": [
                "认为指示剂加得越多现象越明显",
                "认为滴定管读数是从下往上读的"
            ],
            "knowledge_level": "low"
        }
        self.config = persona_config or default_config
        self.logger.info(f"Student initialized: {self.config['name']}")

        # [新增方法] 模拟视觉感知：过滤掉微观数据（如离子浓度），只保留宏观现象
    def _format_full_state_perception(self, snapshot: Dict[str, Any]) -> str:
        if not snapshot:
            return "（实验台数据缺失）"

        containers = snapshot.get("containers", {})
        topology = snapshot.get("topology", [])
        
        state_lines = []

        # 1. 容器全状态 (不再过滤 pH 和 微观成分)
        has_content = False
        for vid, data in containers.items():
            vol = data.get("volume_ml", 0)
            
            # 依然忽略空容器以节省 Token，除非它是关键设备
            # 但如果只有很少残留(>0)也显示，方便调试
            if vol <= 0.01 and data.get("precipitate_g", 0) <= 0:
                continue

            has_content = True
            
            # === 核心修改：暴露所有物理参数 ===
            # 将 pH, 温度, 沉淀, 主要成分全部格式化出来
            major_species = data.get("major_species", [])
            species_str = ", ".join(major_species) if major_species else "无"
            
            info = (
                f"- 【{vid}】: "
                f"Vol={vol:.1f}ml | "
                f"pH={data.get('ph', 7):.1f} | "
                f"Temp={data.get('temperature', 25):.1f}°C | "
                f"Color='{data.get('color_desc', '无色')}' | "
                f"Contains=[{species_str}]"
            )
            
            state_lines.append(info)

        if not has_content:
            state_lines.append("（所有容器均为空）")

        if state_lines:
            state_lines.insert(0, "【当前容器详细物理状态】:")

        # 2. 连接状态 (保持不变)
        if topology:
            state_lines.append("\n【装置连接情况】:")
            for link in topology:
                state_lines.append(f"- {link['child']} 连接在 {link['parent']} 上 ({link['type']})")

        return "\n".join(state_lines)
    
    @RETRY_RULE
    def _call_student_llm(self, messages):
        return self.client.chat.completions.create(
            model=Config.MODEL_STUDENT,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )

    def respond(self, history: List[Dict[str, str]], last_observation: str, current_snapshot: Dict[str, Any] = None) -> Dict[str, Any]:
        d_logger = logging.getLogger("Dialogue")
        try:
            # === [新增 1] 提取老师的上一句话 ===
            # history 里的格式通常是 [{"role": "user", ...}, {"role": "assistant", ...}]
            # 在这里，Teacher 是 "assistant" (对于学生来说，外界输入是 user 还是 assistant 取决于你的 history 组装方式，
            # 但看你的 main 函数，Teacher 的话存为了 "assistant")。
            last_teacher_saying = "（老师还没说话，这是实验开始）"
            for msg in reversed(history):
                if msg['role'].lower() == 'teacher':
                    last_teacher_saying = msg['content']
                    break
            # === 修改点：调用全状态感知函数 ===
            state_context = self._format_full_state_perception(current_snapshot)

            # === [🔥 核心修改] 生成历史字符串 ===
            history_str = format_history_to_string(history)
                
            # 传入 _build_prompt
            system_prompt = self._build_prompt(last_observation, last_teacher_saying, state_context,dialogue_history_str=history_str)

            messages = [{"role": "system", "content": system_prompt}]

            # === [补全日志 1] ===
            log_turn_event("Dialogue", "Student_LLM_Request", {
                "persona": self.config['name'],
                "prompt_preview": system_prompt[:100] + "..."
            })
            start_time = time.time()
            resp = self._call_student_llm(messages)

            raw_content = resp.choices[0].message.content
            
            if not raw_content:
                self.logger.warning("Student LLM returned None content. Using fallback.")
                return {
                    "thought": "（大脑一片空白）",
                    "speak": "老师，我刚才走神了，没听清您说什么。",
                    "action_intents": []
                }
            
            content = json.loads(raw_content)
            
           # === [补全日志 2] ===
            usage = {}
            if resp.usage:
                usage = {"total_tokens": resp.usage.total_tokens}

            content = json.loads(resp.choices[0].message.content)
            
            log_turn_event("Dialogue", "Student_LLM_Response", {
                "thought": content.get("thought"),
                "speak": content.get("speak"),
                "usage": usage,
                "duration": round(time.time() - start_time, 2)
            })
            # === [插入代码 End] ===
            
            # 3. 结构化输出清洗
            return {
                "thought": content.get("thought", "..."),
                "speak": content.get("speak", "..."),
                # 兼容旧代码：如果 LLM 还是返回了单数 action_intent，强转为列表
                "action_intents": content.get("action_intents") or ([content.get("action_intent")] if content.get("action_intent") else [])
            }

        except Exception as e:
            self.logger.error("Student agent failed", exc_info=True)
            d_logger.error(f"Student Error: {str(e)}")
            return {"thought": "我有点晕...", "speak": "老师，我没听懂。", "action_intent": None}

    def _build_prompt(self, last_observation: str, last_teacher_saying: str, state_context: str, 
                      dialogue_history_str: str, # <--- [🔥 新增参数]
                      available_hardware=None) -> str:
        name = self.config.get('name', '实验新手')
        traits = "、".join(self.config.get('traits', ["积极尝试"]))

        # [新增] 提取深度人设属性
        misconceptions = "、".join(self.config.get('misconceptions', []))
        bias = self.config.get('behavioral_bias', '按照老师的要求进行操作')
        k_level = self.config.get('knowledge_level', 'medium')

       # 硬件约束 (强化版)
        hw_constraint = ""
        if available_hardware:
            hw_list = ", ".join(available_hardware)
            hw_constraint = f"""
            【🚨 物理边界限制 (HARDWARE)】
            当前实验台上**仅有**以下设备 ID：[{hw_list}]。
            1. 你生成的 action_intents 中的 `vessel_id` 必须严格来自上述列表。
            2. 严禁臆造不存在的设备（如 'test_tube_999'）。
            """
        
        return f"""
        # Role: 化学实验学生 - {name}
    
        ## 👤 个人画像 (Persona)
        - **性格特征**: {traits} (请在 'speak' 和 'thought' 中体现这些性格，比如急躁的学生可能说话简短，操作鲁莽)
        - **知识水平**: {k_level}
        - **行为偏见**: {bias} 
        - **潜在认知误区**: {misconceptions} (如果当前场景触发了误区，请大胆地犯错，除非老师刚刚明确纠正过)

        {hw_constraint}

       ## ⚡ 行为准则 (Behavioral Rules)
        1. **指令即动作**：只要老师提到了某个器材或试剂，你必须尝试生成 `action_intents`。指出动作类型，以及具体参数（如器材、药品等）。
        2. **操作逻辑区分 (Mental Model)**:
           - **“加药” (Add Reagent)**: 向容器中加入新的化学药品时，使用 `Add`。
           - **“倒液/转移” (Transfer Liquid)**: 容器间倒液时，使用 `Transfer`。
           - **“清洗” (Wash)**: 老师要求清洗容器时，使用 `Wash`。
           - **“连接/固定” (Attach)**: 老师要求安装器材时，使用 `Attach`。
           - **“加热/冷却” (Heat/Cool)**: 老师要求调节温度时，使用'Heat'或'Cool'。
           - **“插入” (Insert)**: 老师要求插入滴定管、温度计等时，使用'Insert'。
           - **“拆卸/断开” (Detach)**: 当你需要移动一个已经连接在某处的设备（如把电极从盐酸杯移到醋酸杯）时，物理上不能直接穿墙！你必须**分两步走**：
             1. 先生成 `Detach` 动作把它取下来。
             2. 再生成 `Insert` 或 `Attach` 动作把它装到新地方。
             * 意图示例: `[{{"action": "Detach", "object": "meter"}}, {{"action": "Insert", "tool": "meter", "vessel": "beaker_2"}}]`
        3. **模糊指令处理**：老师没给具体数值时，尝试一个安全小剂量。
        

       ## 🔍 当前情境 (Context)
        ### 1. 眼前的景象 (Visual)
        {state_context}

        ### 2. 📜 对话记忆 (Memory)
        {dialogue_history_str}

        ### 3. 老师的指令 (Audio)
        "{last_teacher_saying}"

         ## 📝 决策思维链 (Chain of Thought)
        1. **提取动词**：老师的话里包含了哪些暗示？请你思考老师想让你执行的操作并尝试去做。
        2. **锁定目标**：根据【物理状态快照】，哪个器材是空的？哪个是老师提到的？锁定需要操作的器材，试剂。
        3. **量化尝试**：老师给了数字就按数字做；没给数字就自己定一个合适的初试量。

        ## ⚠️ 输出要求 (JSON 格式)
        必须返回 JSON 格式，必须给出具体的 action_intents。
        {{
            "thought": "内心独白：老师让我... 我看到桌上有... 我决定...",
            "speak": "对老师的回应 (口语化)",
            "action_intents": [动作类型及其参数列表;...]
        }}
        """
    
class PersonaRegistry:
    def __init__(self, directory='profiles/'):
        self.directory = directory
        self.profiles = {}
        self.logger = logging.getLogger("PersonaRegistry")

    def load_profiles(self):
        """扫描并加载目录下所有的 .json 配置文件"""
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.logger.warning(f"Profile directory created at {self.directory}")
            return

        for filename in os.listdir(self.directory):
            if filename.endswith(".json"):
                filepath = os.path.join(self.directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 以文件名(不含后缀)或配置中的 id 作为索引键
                        p_id = data.get("id", filename.replace(".json", ""))
                        self.profiles[p_id] = data
                        self.logger.info(f"Loaded student profile: {p_id} ({data.get('name')})")
                except Exception as e:
                    self.logger.error(f"Failed to load profile {filename}: {e}")

    def get_profile(self, profile_id: str) -> dict:
        """获取指定 ID 的人设配置"""
        return self.profiles.get(profile_id)

    def list_profiles(self):
        """返回所有可用的人设 ID"""
        return list(self.profiles.keys())
    

def format_executed_actions(actions: List[Dict[str, Any]]) -> str:
    """将物理引擎执行的动作列表转化为自然语言日志"""
    lines = []
    for idx, act in enumerate(actions):
        atype = act.get("action")
        desc = f"{idx + 1}. {atype}"
        
        # 提取关键参数以增强可读性
        if atype == "Add":
            reagent = act.get("reagent", "未知试剂")
            amount = f"{act.get('mass')}g" if "mass" in act else f"{act.get('volume')}ml"
            vessel = act.get("vessel", "")
            desc += f": {reagent} ({amount}) -> {vessel}"
            
        elif atype == "Transfer":
            vol = f"{act.get('volume')}ml"
            src = act.get("from_vessel", "")
            dst = act.get("to_vessel", "")
            desc += f": {src} --({vol})--> {dst}"
            
        elif atype in ["Attach", "Insert"]:
            child = act.get("vessel") or act.get("tool") or act.get("object")
            parent = act.get("support") or act.get("vessel") or act.get("target")
            desc += f": {child} -> {parent}"
            
        elif atype in ["Heat", "Cool", "Stir", "Wash"]:
            vessel = act.get("vessel", "")
            desc += f": {vessel}"
            
        lines.append(desc)
    
    return "\n".join(lines)

def main(xdl_filename: str = "titration.xdl", persona_id: str = "default", output_dir: str = "generated_datasets", is_batch: bool = False):

    # ==========================================
    # 1. 初始化数据记录结构
    # ==========================================
    session_id = f"{xdl_filename.split('.')[0]}_{persona_id}_{datetime.now().strftime('%m%d_%H%M%S')}"
    session_data = {
        "metadata": {
            "session_id": session_id,
            "xdl_file": xdl_filename,
            "persona_id": persona_id,
            "timestamp": datetime.now().isoformat()
        },
        "turns": []
    }
    # ==========================================
    # 1. 基础环境配置
    # ==========================================

    setup_advanced_logging(is_batch=is_batch)
    
    # 批量模式静默输出，防止控制台渲染拖慢速度
    if is_batch:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')

    init(autoreset=True)
    # print(Fore.CYAN + Style.BRIGHT + "=== Socratic ChemLab Pro (Goal-Oriented Final) ===")
    
    client = OpenAI(
        api_key=Config.OPENAI_API_KEY, 
        base_url=Config.OPENAI_BASE_URL
    )

    # ==========================================
    # 2. 加载实验与人设
    # ==========================================
    ensure_xdl_file_exists(xdl_filename)
    registry = ExperimentRegistry()
    
    try:
        # print(Fore.WHITE + f"Loading XDL: {xdl_filename} ...")
        exp_id = registry.load_xdl_experiment(xdl_filename)
        exp_config = registry.get_experiment(exp_id)
        
        # =====================================================
        # [核心修正] 构建两个独立的映射表
        # =====================================================
        reagent_formula_map = {}  # 给 Engine/Oracle 用: Name -> Formula (e.g., "Unknown_HCl" -> "HCl")
        reagent_state_map = {}    # 给 Translator 用: Name -> State (e.g., "Unknown_HCl" -> "liquid")
        
        reagents_list = exp_config.get('reagents', [])
        
        for r in reagents_list:
            r_name = r.get('name')
            if not r_name: continue
            
            # 1. 提取 Formula (如果 XDL 里没有显式写 formula 字段，可能就是 name 本身或者通过其他映射)
            # 这里假设 XDL 中可能有一个 'formula' 字段，或者 'real_name' 字段
            # 如果没有，默认就用 name (即保持你原有的逻辑)
            r_formula = r.get('formula', r_name) 
            reagent_formula_map[r_name] = r_formula
            
            # 2. 提取 State (默认为 liquid)
            r_state = r.get('state', 'liquid')
            reagent_state_map[r_name] = r_state
            
        # print(Fore.GREEN + f"Loaded Reagents: {len(reagents_list)}")
        
    except Exception as e:
        # print(Fore.RED + f"Error loading XDL: {e}")
        return

    # 加载学生人设
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="default", help="指定学生人设ID")
    args, unknown = parser.parse_known_args()
    
    persona_reg = PersonaRegistry()
    persona_reg.load_profiles()
    student_config = persona_reg.get_profile(persona_id) or {"name": "默认学生", "traits": ["普通"]}
    # print(Fore.CYAN + f"Loaded Student Profile: {student_config['name']}")

    # ==========================================
    # 3. 初始化核心组件 (Core Init)
    # ==========================================
    
    # [关键] Oracle 初始化：传入 ChemSimEngine 类，让 Oracle 能在后台预演生成 Golden States
    oracle = Oracle(exp_config, ChemSimEngine)
    
    # 初始化真实物理引擎
    sim = ChemSimEngine(exp_config["hardware"], reagent_map=reagent_formula_map)

    oracle.reagent_map = reagent_formula_map
    
    # 初始化 Agents 和控制器
    planner = DynamicPlanner(oracle)
    reviewer = Reviewer(client, oracle) # Reviewer 现在是目标导向的
    translator = ActionTranslator(client)
    teacher = TeacherAgent(client, None, oracle)
    student = StudentAgent(client, persona_config=student_config)

    # ==========================================
    # 4. 建立初始状态快照 (Baseline)
    # ==========================================
    # last_snapshot 用于存储"上一个稳定状态"，用于计算副作用
    last_snapshot = {
        "containers": {k: v.get_snapshot() for k, v in sim.containers.items()},
        "topology": sim.hw_manager.get_topology_snapshot()
    }
    
    # [新增] 提取 Metadata 信息
    title = exp_config.get('title', '未命名实验')
    goal = exp_config.get('goal', '完成实验任务')
    difficulty = exp_config.get('difficulty', '未知')
    # print(Fore.MAGENTA + Style.BRIGHT + f"Experiment Loaded: {title} | Goal: {goal} | Difficulty: {difficulty}")
    # 初始化全局日志上下文
    LogContext.current_turn = 0
    LogContext.current_step_index = 0
    LogContext.current_step_desc = "System Init"
    
    log_turn_event("Dialogue", "System_Initialization", {
        "experiment": exp_config['title'],
        "total_steps": oracle.get_total_steps(),
        "initial_state": last_snapshot
    })

    # 生成开场白
    hw_list = [h['id'] for h in exp_config['hardware']]
    hw_str = ", ".join(hw_list)
    
    history = []
    last_obs = f"实验开始。台上摆放着: {hw_str}。所有容器初始为空。"
    # 初始 Review 状态，避免第一轮 Teacher 报错
    last_review = {"status": "PASS", "reason": "实验顺利开始"}
    
    student_opening_line = (
        f"老师，实验台上已经摆放好了 {hw_str}。"
        f"我们要如何“{title}”实验？"
        f"我记得我们的目标是“{goal}”，请您指导我完成它！"
    )
    # print(Fore.YELLOW + f"[Student]: {student_opening_line}")
    history.append({"role": "Student", "content": student_opening_line})
    last_step_desc = "实验引入与目标确认"
    pre_phase = "MAINLINE"

    # 记录初始 Turn
    session_data["turns"].append({
        "turn": -1,
        "role": "opening",
        "phase": pre_phase,
        "target_step": last_step_desc,
        "teaching_strategy": "SOCRATIC_STANDARD",
        "teacher": None,
        "student": {"speak": student_opening_line, "thought": "准备开始实验"},
        "simulation": {"observation": "实验初始化", "snapshot": copy.deepcopy(last_snapshot)},
        "evaluation": {"status": "PASS", "reason": "实验顺利开始"}
    })

    CONTEXT_WINDOW_SIZE = 12
    max_turns = oracle.get_total_steps() * 5
    last_action_turn = 0  # 记录最后一次物理动作发生的轮次
    max_reflection_turns = 2 # 实验结束后允许总结的最大对谈轮次
    # ==========================================
    # 5. 主循环 (Main Loop)
    # ==========================================
    for turn in range(max_turns):
        # print(Fore.BLACK + Style.BRIGHT + f"\n--- Turn {turn + 1}/{max_turns} ---")

        # 更新日志上下文
        LogContext.current_turn = turn + 1
        current_step_data = planner.get_current_step_data()
        curr_idx = planner.current_step_index
        LogContext.current_step_index = curr_idx

        # ==========================================
        # 这里就是放置该代码的最佳位置！
        # 它取代了原本那句：if curr_idx >= oracle.get_total_steps():
        # ==========================================
        if current_step_data.get("action") == "Finish":
            # print(Fore.GREEN + Style.BRIGHT + "=== 🎉 实验目标已达成，进入总结阶段 ===")
            
            # 准备硬件列表（从 sim 引擎中获取所有容器 ID）
            valid_hws = list(sim.containers.keys())
            
            # [核心语句]：参数完全匹配 teacher.respond 的定义
            summary_msg = teacher.respond(
                history=history,                   # 对话历史
                plan_step="实验总结与反思",         # 当前阶段描述
                # last_plan_step=last_step_desc,
                observation="实验已圆满完成，请对学生的表现进行深度点评。", # 观察到的现象
                review_data={"status": "PASS", "reason": "实验成功结束"}, # 评估数据
                phase_type="MAINLINE",             # 阶段类型（主线）
                current_snapshot=current_snapshot, # 传入最终的物理快照，供老师点评实验结果
                available_hardware=valid_hws       # 传入可用硬件列表
            )
            session_data["turns"].append({
                "turn": turn,
                "role": "summary",
                "phase": "MAINLINE",
                "target_step": "实验总结与反思",
                "teaching_strategy": "FINAL_SUMMARY",
                "teacher": summary_msg,
                "simulation": {"snapshot": last_snapshot},
                "evaluation": {"status": "PASS"}
            })
            
            # 打印老师的总结陈词
            # print(Fore.GREEN + f"\n[Teacher]: {summary_msg.get('response', '太棒了！这次实验非常成功。')}")
            
            # 真正结束主循环
            break

        # 1. 获取 Planner 当前想要学生做的步骤 (可能是原定的，也可能是刚插入的洗杯子)
        current_step_data = planner.get_current_step_data()
        
        # 获取描述用于显示
        current_step_desc = current_step_data.get("desc")

        # [新增] 获取当前是主线还是支线
        current_phase = planner.get_phase_type()  # <--- 获取状态

        # 在控制台打印，方便调试
        phase_color = Fore.CYAN if current_phase == "MAINLINE" else Fore.MAGENTA
        print(phase_color + f"[Phase]: {current_phase}")

        LogContext.current_step_desc = current_step_desc
        print(Fore.BLUE + f"[Target]: {current_step_desc}")

        # ----------------------------------
        # A. Teacher Agent (老师发言)
        # ----------------------------------
        active_context = get_context_window(history, window_size=CONTEXT_WINDOW_SIZE)
        # [修改] 传入 snapshot
        # 注意：如果是第一轮，current_snapshot 可能还不存在，用 last_snapshot 兜底
        snapshot_for_teacher = current_snapshot if 'current_snapshot' in locals() else last_snapshot
        valid_hws = list(sim.containers.keys())
        teacher_msg = teacher.respond(
            history=active_context, 
            plan_step=current_step_desc, 
            last_plan_step=last_step_desc,
            observation=last_obs, 
            review_data=last_review,
            phase_type=current_phase,
            last_phase_type=pre_phase,
            available_hardware=valid_hws, # [关键传参]
            current_snapshot=snapshot_for_teacher # <--- 新增
        )
        t_speak = teacher_msg.get("response", "...") 
        # print(Fore.GREEN + f"[Teacher]: {t_speak}")
        history.append({"role": "Teacher", "content": t_speak})
        
        # 更新上下文供学生使用
        active_context = get_context_window(history, window_size=CONTEXT_WINDOW_SIZE)

        # ----------------------------------
        # B. Student Agent (学生发言)
        # ----------------------------------
        # [修改] 传入 current_snapshot (注意：如果是第一轮，用 last_snapshot)
        snapshot_to_show = current_snapshot if 'current_snapshot' in locals() else last_snapshot
        
        student_output = student.respond(
            history=active_context, 
            last_observation=last_obs, 
            current_snapshot=snapshot_to_show # <--- 新增传参
        )
        raw_actions = student_output.get("action_intents", [])
        s_speak = student_output.get("speak")
        # print(Fore.YELLOW + f"[Student]: {s_speak}")
        history.append({"role": "Student", "content": s_speak})

        # ----------------------------------
        # C. Simulation & Execution (物理模拟)
        # ----------------------------------
        sim_text = ""
        engine_actions = []
        translation_errors = [] # [新增]
        trans_errors = []

        review_result = {"status": "WAIT", "reason": "No actions performed this turn."}
        # 默认当前快照等于上一轮 (假设学生没做动作)
        current_snapshot = copy.deepcopy(last_snapshot)

        if raw_actions:
            # 翻译动作 (Translator 需要看当前的 snapshot 来辅助消歧义)
            engine_actions, trans_errors = translator.translate(
                raw_text_intent=raw_actions,
                snapshot=last_snapshot, 
                reagent_state_map=reagent_state_map,
                history=history
            )
            
            if engine_actions:
                # [核心] 执行动作，物理世界发生改变
                sim_text, current_snapshot = sim.execute_batch(engine_actions)

                # === [修改 Start] 生成详细的动作执行回执 ===
                
                # 1. 格式化动作列表
                action_log_str = format_executed_actions(engine_actions)
                
                # 2. 组合 System Content
                # 明确告诉学生：这些动作系统已经收到了，并且执行完了，不要再做了！
                system_feedback = (
                    f"【系统确认：以下动作学生已成功执行】\n"
                    f"{action_log_str}\n\n"
                )
                
                # 3. 存入历史
                history.append({"role": "system", "content": system_feedback})
                
                # === [修改 End] ===
                
                
                last_obs = sim_text 
                
                log_turn_event("Dialogue", "Simulation_Action", {
                    "actions": engine_actions,
                    "result_text": sim_text
                })
            else:
                sim_text = "操作无效或不清晰，系统未执行。"
                # print(Fore.RED + f"[System]: {sim_text}")
                log_turn_event("Dialogue", "Action_Ignored", {"raw": raw_actions})
        else:
            sim_text = "Student is just talking..."
            # current_snapshot 保持不变

        # ----------------------------------
        # D. Reviewer (Goal-Oriented Evaluation)
        # ----------------------------------
        # 2. [关键修改] 动态获取 Goal
        # 不再使用 index 查表，而是把当前的 step_data 扔给 Oracle 去推演

        # 【核心修复】只有当确实执行了物理引擎动作时，才进行物理审计
        if engine_actions:
           # 2. 动态获取 Goal
            goal = oracle.get_dynamic_goal(current_step_data, current_actual_snapshot=last_snapshot)

            review_result = reviewer.evaluate(
                history=history,
                step_index=curr_idx,
                current_snapshot=current_snapshot,
                pre_snapshot=last_snapshot,
                goal=goal,
                last_actions=engine_actions,  # 这一行之前加了
                planner=planner,               # <--- [新增] 务必加上这一行！
                translation_errors=trans_errors
            )
            status = review_result['status']
            reason = review_result.get('reason', '')
            diagnosis = review_result.get('diagnosis') or {}

            # print(Fore.MAGENTA + f"[Reviewer]: {status} | {reason}")
            
            # 保存 Review 结果供下一轮 Teacher 参考
            last_review = review_result

            # ==========================================
            # [🔥 核心修改] 将系统判决注入历史记录
            # ==========================================
            feedback_content = ""
            if status == "FAIL":
                err_type = diagnosis.get("type", "Error")
                feedback_content = f"【🚫 系统判定: 失败】\n类型: {err_type}\n原因: {reason}\n>> 请老师指导学生修正。"
            elif status == "PASS":
                feedback_content = f"【✅ 系统判定: 通过】\n操作符合预期。请继续下一步。"
            
            if feedback_content:
                history.append({"role": "system", "content": feedback_content})
                # print(Fore.MAGENTA + feedback_content)
        
        else:
            # 如果学生只是在说话，没有产生 engine_actions
            # print(Fore.MAGENTA + f"[Reviewer]: SKIP (Dialogue Phase)")

            # 【修复点】显式重置 status，防止沿用上一轮的 "PASS"
            status = "WAIT"
            # 保持 last_review 不变，或者设为 WAIT，防止 Teacher 下一轮以为出错了
            last_review = {"status": "WAIT", "reason": "学生正在进行理论回答，未执行操作。"}

        # --- F. 记录当前 Turn 数据 ---
        session_data["turns"].append({
            "turn": turn,
            "phase": current_phase,
            "target_step": current_step_data.get("desc"),
            "teaching_strategy": teacher.current_strategy,
            "teacher": teacher_msg,
            "student": student_output,
            "translation": {"actions": engine_actions, "errors": trans_errors},
            "simulation": {"observation": sim_text, "snapshot": copy.deepcopy(current_snapshot)},
            "evaluation": review_result
        })

        # 在下一轮 Teacher 发言时，"上一步" 就是现在的 "current_step_desc"
        if status == "PASS":
            last_step_desc = current_step_desc
        else:
            # 如果没过 (FAIL/WAIT)，说明学生还卡在当前步骤
            # 下一轮的 "上一步" (刚才尝试的) 依然是当前步骤
            last_step_desc = current_step_desc
        
        # ----------------------------------
        # E. Flow Control (状态机流转)
        # ----------------------------------
        if status == "PASS":
            # print(Fore.GREEN + f">> ✅ Step {curr_idx+1} Completed!")

            is_remedial = current_step_data.get("is_remedial", False)
            
            if not current_step_data.get("is_virtual"):
                # 如果是补救步骤，索引可以传入 999 或特殊标记，
                # 或者直接承认它为当前最新理想状态，但不计入主线历史
                oracle.commit_step(
                    current_step_data, 
                    step_index=curr_idx if not is_remedial else -99, # 区分主线与补救
                    actual_snapshot=current_snapshot
                )
            
            # 2. 标记刚才通过的是否是补救步骤
            just_finished_remedial = current_step_data.get("is_remedial", False)
            
            # 3. 推进指针
            planner.advance()
            
            # === [新增修复] 冗余步骤自动跳过 (Auto-Skip Redundant Step) ===
            # 如果刚做完补救步骤，且下一步骤就是那个“原步骤”，
            # 我们需要检查现在的物理状态是否已经满足了“原步骤”的要求。
            if just_finished_remedial:
                # 获取新的当前步骤 (即原先失败的那个步骤)
                next_step_data = planner.get_current_step_data()
                
                if next_step_data and not next_step_data.get("is_virtual") and next_step_data.get("action") != "Finish":
                    
                    # === [Fix] 回溯寻找干净的基准 (Clean Base) ===
                    clean_base = None
                    # 从当前位置的前一步开始倒序回溯
                    # 寻找最近的一个非补救步骤 (Mainline) 的结束状态
                    for idx in range(planner.current_step_index - 1, -2, -1):
                        if idx == -1:
                            clean_base = oracle.history.get(-1) # 初始状态
                            break
                        
                        if idx < len(planner.steps):
                            step = planner.steps[idx]
                            # 只要找到一个不是补救步骤的，就是我们要的“干净历史”
                            if not step.get("is_remedial"):
                                clean_base = oracle.history.get(idx)
                                break
                    
                    # 计算目标：基于干净的历史 + 下一步动作
                    # 这样计算出来的目标就是 "0 + 20 = 20ml"，而不是 "19.9 + 20 = 39.9ml"
                    if clean_base:
                        next_goal = oracle.get_dynamic_goal(next_step_data, base_snapshot=clean_base)
                    else:
                        next_goal = oracle.get_dynamic_goal(next_step_data)

                    # 静态检查：现在的状态(19.9ml) vs 干净目标(20.0ml) -> PASS
                    check_result = reviewer.inspector.inspect(current_snapshot, last_snapshot, next_goal)
                    
                    # 1. 完美通过 -> 跳过
                    if check_result.status == ReviewStatus.PASS:
                        # print(Fore.CYAN + f">> ⏩ 补救成功，原步骤 '{next_step_data.get('desc')}' 目标已自然达成，自动跳过。")
                        oracle.commit_step(next_step_data, planner.current_step_index, current_snapshot)
                        planner.advance()

                    # 2. [🔥核心修复] 豁免“过量”误判
                    # 如果是因为 OVERSHOOT 导致的失败，且误差正好等于我们刚才补加的量，说明是系统误判
                    elif check_result.status == ReviewStatus.FAIL and "过量" in check_result.reason:
                        # print(Fore.YELLOW + f">> ⚠️ 检测到补救后的'伪过量' ({check_result.reason})。鉴于这是刚补加的试剂，强制判定为通过。")
                        
                        # 强制提交并跳过
                        oracle.commit_step(next_step_data, planner.current_step_index, current_snapshot)
                        planner.advance()
                    
                    # else:
                         # print(Fore.YELLOW + f"[Auto-Skip Failed] Status: {check_result.status} | Reason: {check_result.reason}")
            # === [🔥核心修复] 状态重置 ===
            # 切换到新步骤后，强制重置 last_review 为 WAIT
            # 防止 Teacher Agent 看到上一轮的 PASS 而误判当前步骤已完成
            # last_review = {
            #     "status": "WAIT", 
            #     "reason": "New step initialized. Waiting for student action."
            # }
            
            # 更新基准快照
        # elif status == "INFO": 
            # 对应 ReviewStatus.IN_PROGRESS (例如: 加水加了一半)
            # print(Fore.CYAN + ">> 🔄 In Progress. Teacher will guide student to continue.")
            # 策略：不推进，也不更新 last_snapshot，继续在当前步骤循环

        elif status == "FAIL":
            # 补救与回滚逻辑 (保持您原有的处理)
            diagnosis = review_result.get('diagnosis', {})
            ideal_snap = oracle.get_step_ideal_snapshot(planner.get_current_step_data())
            handled, msg = planner.inject_remedial_plan(diagnosis, ideal_snap)
            # print(Fore.RED + f">> ❌ Step Failed: {diagnosis.get('type')}")
            
            if handled and msg.startswith("ROLLBACK:"):
                # === [🔥核心修复 Start] 处理回滚信号 ===
                if msg.startswith("ROLLBACK:"):
                    # 解析目标步骤 (我们要去重做的那一步)
                    try:
                        target_step_idx = int(msg.split(":")[1])
                    except:
                        target_step_idx = 0
                    
                    # print(Fore.RED + Style.BRIGHT + f">> 🔄 IRREVERSIBLE ERROR. Rolling back to Step {target_step_idx + 1}...")
                    
                    # A. Oracle 回滚：回到 target_step 开始之前的状态 (即 target_step - 1 结束时的状态)
                    rollback_success = oracle.rollback_to(target_step_idx - 1)
                    if not rollback_success:
                        # print(Fore.RED + "Critical System Error: Oracle rollback failed.")
                        break

                    # B. Planner 回跳：指针指向 target_step
                    planner.jump_to(target_step_idx)
                    
                    # C. [🔥关键修复] 物理引擎 (Student World) 强制同步为 Oracle 的回滚状态
                    # 必须使用 deepcopy，防止 Sim 和 Oracle 共享引用
                    sim.containers = copy.deepcopy(oracle.chk_containers)
                    
                    # D. [🔥关键修复] 重建硬件拓扑 (Hardware Topology)
                    # 先清空当前错误的连接关系
                    sim.hw_manager.graph.clear()
                    # 重新注册所有硬件节点
                    sim.hw_manager.add_hardware(list(sim.containers.keys()))
                    # 根据存档恢复连接 (Edges)
                    for link in oracle.chk_topology:
                        sim.hw_manager.attach(link['child'], link['parent'], link['type'])
                    
                    # print(Fore.GREEN + f">> System State (Physics & Topology) restored to Step {target_step_idx + 1}.")

                    # E. [🔥关键修复] 重置基准快照 (last_snapshot)
                    # 这步至关重要！Reviewer 需要用这个干净的状态作为 pre_snapshot
                    # 否则它会拿"刚洗干净的杯子"去对比"刚才脏的杯子"，导致计算出错误的 diff
                    last_snapshot = {
                        "containers": {k: v.get_snapshot() for k, v in sim.containers.items()},
                        "topology": sim.hw_manager.get_topology_snapshot()
                    }
                    # 同时更新当前的 current_snapshot，防止本轮后续逻辑读取旧数据
                    current_snapshot = copy.deepcopy(last_snapshot)

                    # F. 重置老师的耐心值
                    # 既然世界已经重启，老师应该重新把学生当作第一次尝试
                    teacher.reset_patience()
                    
                    # G. 通知老师和历史记录
                    sys_msg = f"【系统干预】检测到不可逆错误（如前置产物丢失）。实验已自动回滚至第 {target_step_idx + 1} 步。物理环境已重置，请指导学生重新开始。"
                    history.append({"role": "system", "content": sys_msg})
                    
                    # H. 欺骗 Reviewer 状态，这轮不算 FAIL，而是 INFO，避免 TeacherAgent 再次批评
                    last_review = {"status": "INFO", "reason": "系统正在执行回滚操作，请忽略本次报错。"}
                
                # === [🔥核心修复 End] ===

                else:
                    # 普通补救 (Inject) - 例如只是需要加点水
                    # print(Fore.YELLOW + f">> 🛠️ Remedial steps injected: {msg}")
                    sys_msg = f"【系统提示】检测到操作偏差。{msg}"
                    history.append({"role": "system", "content": sys_msg})
                    # 普通补救不需要修改 last_snapshot，因为物理世界是连续的
            
        # 【核心修正】：每一轮结束，无论什么状态，统一更新基准
        # 这样确保下一轮的 pre_snapshot 永远等于本轮结束时的 current_snapshot
        last_snapshot = copy.deepcopy(current_snapshot)
        pre_phase = current_phase

    # ==========================================
    # 5. 保存数据到文件
    # ==========================================
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"{session_id}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    # print(Fore.GREEN + f"Session Data Recorded: {save_path}")
    # 恢复标准输出以便 Generator 看到进度
    if is_batch: sys.stdout = sys.__stdout__
    return session_data

def get_context_window(history: List[Dict[str, str]], window_size: int = 12) -> List[Dict[str, str]]:
    """
    [修正版] 智能上下文窗口
    策略：
    1. 总是保留 history[0] (对话的“锚点”，通常是开场白或初始设定)。
    2. 保留最近的 window_size - 1 条消息。
    3. 确保中间被切断的地方逻辑不会太突兀（虽然 LLM 容忍度很高）。
    """
    # 1. 历史记录很少，不需要截断
    if len(history) <= window_size:
        return history
    
    # 2. 提取首条消息 (Anchor)
    first_msg = history[0]
    
    # 3. 提取最近的消息 (Sliding Window)
    # 我们保留最近的 (window_size - 1) 条，给 first_msg 腾出一个位置
    recent_msgs = history[-(window_size - 1):]
    
    # [进阶优化] 防止截断导致 Role 错位 (例如连续两条 User)
    # 这一步不是必须的，现代模型能处理，但为了严谨可以检查
    # 如果 first_msg 是 User，recent_msgs[0] 最好是 Assistant，反之亦然
    
    # 4. 拼接
    return [first_msg] + recent_msgs

def ensure_xdl_file_exists(filename):
    """辅助函数：如果文件不存在，则创建（用于演示）"""
    path = os.path.join("experiments", filename)
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
        
    if not os.path.exists(path):
        content = """<?xml version="1.0" encoding="UTF-8"?>
<XDL>
  <Metadata 
      title="酸碱中和滴定" 
      goal="利用已知浓度的 NaOH 标准液，测定未知盐酸的浓度。" 
      difficulty="Medium"
  />
  <Synthesis>
    <Hardware>
      <Component id="conical_flask_1" type="flask" capacity="250ml" description="锥形瓶"/>
      <Component id="burette_1" type="burette" capacity="50ml" description="酸式滴定管"/>
      <Component id="iron_stand_1" type="stand" description="铁架台"/>
      <Component id="beaker_waste" type="beaker" description="废液缸"/>
    </Hardware>
    <Reagents>
      <Reagent name="Unknown_HCl" state="liquid" description="待测盐酸溶液"/>
      <Reagent name="Standard_NaOH" state="liquid" description="0.1mol/L氢氧化钠标准液"/>
      <Reagent name="Phenolphthalein" state="liquid" description="酚酞指示剂"/>
    </Reagents>
    <Procedure>
      <Attach vessel="burette_1" support="iron_stand_1" />
      <Add vessel="conical_flask_1" reagent="Unknown_HCl" volume="20.0" tool="pipette"/>
      <Add vessel="conical_flask_1" reagent="Phenolphthalein" volume="0.2" tool="dropper"/>
      <Add vessel="burette_1" reagent="Standard_NaOH" volume="50.0" tool="beaker"/>
      <Transfer from_vessel="burette_1" to_vessel="conical_flask_1" volume="15.0" tool="burette_stopcock"/>
    </Procedure>
  </Synthesis>
</XDL>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        # print(f"Created demo file: {path}")

if __name__ == "__main__":
    main("auto_generated.xdl")