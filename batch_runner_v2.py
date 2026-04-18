import os
import json
import uuid
import time
import logging
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import Fore, init
from tqdm import tqdm
from openai import OpenAI
import sys

# === 导入核心系统 ===
# 确保 soc_chem_dia_refactored.py 在同一目录下
from soc_chem_dia_refactored import SocraticDataGenerator, Config, setup_logging

# 初始化环境
init(autoreset=True)
# 建立专门的 Batch Logger
logger = logging.getLogger("BatchRunner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ==========================================
# 0. 配置：选定探针角色
# ==========================================
# 选定一个操作最规范的角色作为“探路者”
STABLE_PROFILE_NAME = "标准-普通"

# 【新增开关】 严格模式 / 熔断机制
# True: 只要有一个探针失败，立即终止整个程序（适合调试代码 Bug）
# False: 探针失败只跳过当前实验，继续尝试下一个实验（适合大规模生产）
STOP_ON_FIRST_FAILURE = True

# ==========================================
# 1. 全谱系学生画像库 (12+X Coverage)
# ==========================================
STUDENT_PROFILES = [
    # --- Group A: 标准对照组 ---
    {"name": "标准-新手", "traits": ["紧张", "犹豫"], "clumsiness": "high", "knowledge_level": "novice"},
    {"name": "标准-普通", "traits": ["按部就班", "听话"], "clumsiness": "average", "knowledge_level": "average"},
    {"name": "标准-学霸", "traits": ["自信", "专业"], "clumsiness": "low", "knowledge_level": "expert"},

    # --- Group B: 认知能力错位 ---
    {"name": "眼高手低", "traits": ["理论满分", "动手能力差", "害怕失败"], "clumsiness": "high", "knowledge_level": "expert"},
    {"name": "盲目自信", "traits": ["鲁莽", "迷之自信", "不看说明书"], "clumsiness": "average", "knowledge_level": "novice"},
    
    # --- Group C: 沟通障碍组 ---
    {"name": "沉默寡言", "traits": ["极度内向", "不爱说话", "被动执行"], "clumsiness": "average", "knowledge_level": "average"},
    {"name": "好奇宝宝", "traits": ["好奇心过剩", "喜欢打岔", "注意力不集中"], "clumsiness": "high", "knowledge_level": "novice"},
    {"name": "叛逆挑战", "traits": ["固执", "质疑权威", "喜欢反着来"], "clumsiness": "average", "knowledge_level": "average"},

    # --- Group D: 特定风险组 ---
    {"name": "粗心大意", "traits": ["健忘", "忽略细节", "急躁"], "clumsiness": "average", "knowledge_level": "average"},
    {"name": "极度恐慌", "traits": ["过度谨慎", "手抖严重", "甚至不敢开始"], "clumsiness": "high", "knowledge_level": "novice"}
]

# ==========================================
# [已移除] RichSessionCollector
# 原因：SocraticDataGenerator 现已原生支持元数据头写入和流式记录。
# 直接使用基类即可。
# ==========================================

# ==========================================
# 3. 线程工作函数 (Worker - Upgraded)
# ==========================================
def worker_run_session_struct(args):
    """
    修改后的 Worker，直接使用 SocraticDataGenerator
    """
    client, xdl_path, profile, output_dir, max_turns = args
    
    # 构造文件名相关信息
    clean_exp = os.path.basename(xdl_path).replace(".xdl", "").replace(" ", "_")
    p_name = profile['name'].replace("-", "_")
    
    # 生成唯一文件名: 实验名_角色名_UUID前6位.jsonl
    session_uuid = uuid.uuid4().hex[:6]
    file_name = f"{clean_exp}_{p_name}_{session_uuid}.jsonl"
    full_path = os.path.join(output_dir, file_name)
    
    try:
        # === 核心逻辑修改 ===
        # 直接实例化生成器，传入指定的文件路径
        # 注意：__init__ 内部会自动写入 "record_type": "experiment_metadata" (XDL/DAG)
        generator = SocraticDataGenerator(
            client=client,
            yaml_path=xdl_path,
            student_profile=profile,
            output_file=full_path
        )
        
        # 开启静默模式，防止控制台被 20 个线程的日志刷屏
        if hasattr(generator, 'sim'):
            generator.sim.silent_mode = True
        
        # 运行实验
        # 注意：run_episode 内部会调用 UniversalRecorder 逐行追加数据
        generator.run_episode(max_turns=max_turns)
        
        # 验证文件是否生成且有内容
        if os.path.exists(full_path) and os.path.getsize(full_path) > 100:
            return True, f"✅ Pass: {clean_exp}... | {p_name}", 0
        else:
            return False, f"❌ Fail: {clean_exp}... | {p_name} (Empty file)", 0
            
    except Exception as e:
        # 捕获异常，防止单个线程崩溃导致主程序退出
        return False, f"🔥 Error: {clean_exp}... | {p_name} -> {str(e)[:100]}", 0

# ==========================================
# 4. 主控逻辑 (Smart Batch Runner)
# ==========================================
def run_batch_job(
    xdl_folder="experiments", 
    output_folder="raw_data_v2", 
    max_turns=60, 
    max_workers=5
):
    # 1. 准备目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Output dir: {output_folder}")

    # 获取 XDL 列表
    xdl_files = glob.glob(os.path.join(xdl_folder, "*.xdl"))
    if not xdl_files:
        logger.error("No .xdl files found.")
        return

    # 初始化 Client
    client = OpenAI(api_key=Config.OPENAI_API_KEY, base_url=Config.OPENAI_BASE_URL)
    
    # 找到探针画像配置
    try:
        PROBE_PROFILE = next(p for p in STUDENT_PROFILES if p["name"] == STABLE_PROFILE_NAME)
    except StopIteration:
        logger.error(f"Probe profile '{STABLE_PROFILE_NAME}' not found in configuration!")
        return

    # ==========================================
    # Phase 1: 探针测试 (Probe)
    # ==========================================
    print(Fore.CYAN + f"\n🕵️  Phase 1: Probing {len(xdl_files)} experiments with '{STABLE_PROFILE_NAME}'...")
    print(Fore.CYAN + "   (如果学霸都做不完/卡死，则跳过该实验的所有其他角色)")

    passed_experiments = [] # 存储通过测试的 XDL 路径
    probe_tasks = []

    # 1.1 构建探针任务 & 检查已存在文件
    for xdl in xdl_files:
        clean_exp_name = os.path.basename(xdl).replace(".xdl", "").replace(" ", "_")
        clean_probe_name = STABLE_PROFILE_NAME.replace("-", "_")
        
        # 检查是否已经跑过探针
        pattern = os.path.join(output_folder, f"{clean_exp_name}_{clean_probe_name}_*.jsonl")
        if glob.glob(pattern):
            passed_experiments.append(xdl) # 假设已存在的算通过
        else:
            probe_tasks.append((client, xdl, PROBE_PROFILE, output_folder, max_turns))

    # 1.2 运行探针 (如果有没跑过的)
    if probe_tasks:
        # 探针阶段使用较少线程以便及时熔断
        probe_workers = 1 if STOP_ON_FIRST_FAILURE else max_workers
        
        print(Fore.YELLOW + f"   (Probe Mode: {'Strict/Stop-on-Fail' if STOP_ON_FIRST_FAILURE else 'Lenient/Skip-Only'})")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(worker_run_session_struct, t): t for t in probe_tasks}
            
            with tqdm(total=len(probe_tasks), unit="probe", desc="Probing") as pbar:
                for future in as_completed(future_map):
                    # 获取原始参数中的 xdl 路径
                    _, original_xdl, _, _, _ = future_map[future]
                    
                    is_success, msg, turns = future.result()
                    
                    if is_success:
                        passed_experiments.append(original_xdl)
                        pbar.write(msg)
                    else:
                        # === 熔断逻辑 ===
                        pbar.write(Fore.RED + f"🚨 PROBE FAILED: {msg}")
                        
                        if STOP_ON_FIRST_FAILURE:
                            print(Fore.RED + "\n" + "="*40)
                            print(Fore.RED + "🛑 触发熔断机制 (Circuit Breaker Triggered)")
                            print(Fore.RED + f"❌ 实验失败: {os.path.basename(original_xdl)}")
                            print(Fore.RED + "⚠️  程序已立即终止，请检查代码或 Prompt 后重试。")
                            print(Fore.RED + "="*40)
                            
                            executor.shutdown(wait=False)
                            sys.exit(1) # 直接退出 Python 进程
                        
                    pbar.update(1)
    
    if not passed_experiments:
        print(Fore.RED + "❌ No experiments passed the probe test. Stopping.")
        return

    # ==========================================
    # Phase 2: 批量生成 (Batch Production)
    # ==========================================
    print(Fore.CYAN + f"\n🏭 Phase 2: Generating diverse data for {len(passed_experiments)} verified experiments...")
    
    batch_tasks = []
    
    for xdl in passed_experiments:
        for profile in STUDENT_PROFILES:
            # 跳过探针画像 (因为Phase 1已经跑过了)
            if profile["name"] == STABLE_PROFILE_NAME:
                continue
            
            # 断点续传检查
            clean_exp = os.path.basename(xdl).replace(".xdl", "").replace(" ", "_")
            clean_stu = profile['name'].replace("-", "_")
            if glob.glob(os.path.join(output_folder, f"{clean_exp}_{clean_stu}_*.jsonl")):
                continue
                
            batch_tasks.append((client, xdl, profile, output_folder, max_turns))

    if not batch_tasks:
        print(Fore.GREEN + "🎉 All batch tasks completed (or skipped).")
        return

    # 运行 Phase 2
    results_summary = {"success": 0, "failed": 0}
    
    print(Fore.CYAN + f"🚀 Launching {len(batch_tasks)} batch tasks | {max_workers} threads")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker_run_session_struct, t): t for t in batch_tasks}
        
        with tqdm(total=len(batch_tasks), unit="session", desc="Generating", dynamic_ncols=True) as pbar:
            for future in as_completed(future_map):
                is_success, msg, _ = future.result()
                
                if is_success:
                    results_summary["success"] += 1
                else:
                    results_summary["failed"] += 1
                    pbar.write(msg) # 只打印问题日志
                
                pbar.update(1)
                pbar.set_postfix({"✅": results_summary["success"], "❌": results_summary["failed"]})

    # 结束汇总
    logger.info("="*30)
    logger.info(f"Job Finished. Total Success: {results_summary['success']}, Failed: {results_summary['failed']}")
    logger.info(f"Data directory: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    # run_batch_job(
    #     xdl_folder="experiments",
    #     output_folder="finetune_data_dataset", # 修改输出目录
    #     max_turns=40, # 用户指定的上限
    #     max_workers=10 # 建议不要开太高，防止 OOM
    # )
    run_batch_job(
        xdl_folder="/home/yjh/socChem_final/distribution_shift_test",
        output_folder="/home/yjh/socChem_final/distribution_shift_test_result", # 修改输出目录
        max_turns=40, # 用户指定的上限
        max_workers=10 # 建议不要开太高，防止 OOM
    )