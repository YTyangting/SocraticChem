import json
import os
import glob
import re
from typing import List, Dict, Any

# ================= 配置区域 =================
INPUT_FOLDER = "./raw_data_v1"
OUTPUT_FILE = "./chemistry_lab_dataset_v10_xml.json"
MAX_HISTORY_TURNS = 3
STUCK_THRESHOLD = 3
# ===========================================

def generate_god_view_summary(snapshot: Dict) -> str:
    """[God View] 生成环境描述"""
    if not snapshot: return "实验台数据缺失"
    lines = []
    hardware = snapshot.get("hardware", {}) or snapshot.get("containers", {})
    has_interesting_item = False
    
    for vid, data in hardware.items():
        details = []
        is_default = True
        
        contents = data.get("contents", {})
        major_chems = [f"{k}={v:.3f}" for k, v in contents.items() if v > 1e-4]
        if major_chems:
            is_default = False; details.append(f"含: [{', '.join(major_chems)}]")
        else:
            vol = data.get("occupied_volume_ml", data.get("volume_ml", 0))
            if vol > 0.1:
                is_default = False
                species = data.get("major_species", [])
                s_str = ", ".join(species) if species else "未知液体"
                details.append(f"含: {vol}ml {s_str}")
        
        temp = data.get("temperature", 25.0)
        if abs(temp - 25.0) > 2.0: is_default = False; details.append(f"Temp={temp:.0f}C")
        if data.get("is_sealed"): is_default = False; details.append("已密封")
        if data.get("is_covered"): is_default = False; details.append("已盖上")
        if data.get("is_on"): is_default = False; details.append("🔥开启")
        
        if is_default: continue
        lines.append(f"- {vid}: {'; '.join(details)}")
        has_interesting_item = True
        
    if not has_interesting_item: lines.append("（所有器材均处于初始空置状态）")
    
    topo = snapshot.get("topology", [])
    if topo:
        links = [f"{l['child']}连在{l['parent']}" for l in topo]
        lines.append("连接关系: " + ", ".join(links))
        
    return " | ".join(lines)

def analyze_progress_trend(inputs: Dict, hidden: Dict, stuck_count: int) -> str:
    """分析进展趋势"""
    actions = inputs.get('student_actions', [])
    is_intercepted = hidden.get('intercepted', False)
    fail_reasons = hidden.get('fail_reasons', [])
    real_obstacles = [r for r in fail_reasons if "验证通过" not in r and "Success" not in r]

    if is_intercepted: return "📉 停滞 (Stagnant - 操作被拦截)"
    if stuck_count > 0: return f"📉 停滞 (Stagnant - 已卡顿 {stuck_count} 轮)"
    if actions and not real_obstacles: return "📈 正在进步 (Valid Progress)"
    if actions and real_obstacles: return "📈 物理操作有效 (Valid Action) - 但存在未满足条件"
    return "📉 停滞 (No Action)"

def format_student_action(actions: List[Dict], is_intercepted: bool = False) -> str:
    if not actions: return ""
    desc_list = []
    for act in actions:
        if isinstance(act, dict):
            act_type = act.get('action', 'Unknown')
            params = [f"{k}={v}" for k, v in act.items() if k not in ['action', 'id', 'agent']]
            desc_list.append(f"[{act_type}: {', '.join(params)}]")
        else:
            desc_list.append(str(act))
    action_str = ", ".join(desc_list)
    return f" (试图执行: {action_str} -> 被系统拦截)" if is_intercepted else f" (执行操作: {action_str})"

def convert_raw_turn_to_sample(record: Dict, history_str: str, meta_info: Dict, stuck_count: int, experiment_name: str) -> Dict:
    inputs = record.get('inputs', {})
    outputs = record.get('outputs', {})
    hidden = record.get('hidden_states', {}) 

    # 1. 环境
    snapshot = inputs.get('full_physics_snapshot', {})
    env_summary = generate_god_view_summary(snapshot)
    
    # === 2. [关键修改] 极简版学生档案 ===
    # 去掉 Name，只保留 Traits
    traits_list = []
    if meta_info and 'student_profile' in meta_info:
        traits_list = meta_info['student_profile'].get('traits', [])
    
    traits_str = ", ".join(traits_list) if traits_list else "普通"

    k_state = hidden.get('student_knowledge_state', {})
    weak_points = [k.replace("KC_", "") for k, v in k_state.items() if v < 0.5]
    # 如果没有短板，就不显示 Cognitive 标签，进一步节省 Token
    if weak_points:
        cognitive_str = f"\n  <CognitiveGaps>{', '.join(weak_points)}</CognitiveGaps>"
    else:
        cognitive_str = ""

    frustration = hidden.get('student_frustration', 0.0)
    
    # 3. 逻辑判定
    is_goal_met = hidden.get('is_goal_met', False)
    raw_fails = hidden.get('fail_reasons', [])
    real_obstacles = [r for r in raw_fails if "验证通过" not in r and "Success" not in r]
    trend_str = analyze_progress_trend(inputs, hidden, stuck_count)

    logic_warning = ""
    if is_goal_met:
        status_line = "✅ 目标已达成"
        obstacle_line = "无"
    else:
        if stuck_count >= STUCK_THRESHOLD:
            status_line = f"⚠️ 严重受阻 (已卡顿 {stuck_count} 轮)"
            obstacle_line = f"{'; '.join(real_obstacles)}"
            logic_warning = "严重警告: 学生反复犯错，陷入死循环。请立即停止苏格拉底式引导，直接指出错误并要求其纠正！"
        else:
            status_line = "❌ 目标未达成"
            obstacle_line = f"{'; '.join(real_obstacles)}" if real_obstacles else "无明显阻碍"

    # 4. 清洗实验名称
    clean_exp_name = experiment_name.split('_sample_')[0]

    # === [核心修改] 构建 XML 格式的 Input ===
    
    # <LogicAnalysis> 内部结构
    logic_inner = f"  <Status>{status_line}</Status>\n  <Trend>{trend_str}</Trend>\n  <Obstacles>{obstacle_line}</Obstacles>"
    if logic_warning:
        logic_inner += f"\n  <CriticalWarning>{logic_warning}</CriticalWarning>"

    # <StudentState> (重构：更聚焦)
    # 只有当挫败感 > 0.3 时才显示，否则认为是正常，不占用注意力
    frust_str = ""
    if frustration > 0.3:
        frust_str = f"\n  <FrustrationLevel>{frustration:.2f} (High)</FrustrationLevel>"

    student_inner = f"  <Traits>{traits_str}</Traits>{cognitive_str}{frust_str}"

    # 组装 XML Context
    xml_context = f"""
<Context>
<ExperimentTask>{clean_exp_name}</ExperimentTask>
<GodViewEnvironment>
{env_summary}
</GodViewEnvironment>
<StudentState>
{student_inner}
</StudentState>
<LogicAnalysis>
{logic_inner}
</LogicAnalysis>
</Context>
""".strip()

    # 组装 History
    # 如果没有历史，可以用 <DialogueHistory>None</DialogueHistory>
    xml_history = f"<DialogueHistory>\n{history_str}\n</DialogueHistory>" if history_str else "<DialogueHistory>无</DialogueHistory>"

    # 组装 Current Input
    s_speak = inputs.get('student_speak', '')
    s_actions = inputs.get('student_actions', [])
    is_intercepted = hidden.get('intercepted', False)
    s_act_str = format_student_action(s_actions, is_intercepted)
    
    xml_current = f"""
<CurrentInput>
<Student>
{s_speak}{s_act_str}
</Student>
</CurrentInput>
""".strip()

    # 最终 Input 组合
    full_input = f"{xml_context}\n\n{xml_history}\n\n{xml_current}"

   # === [核心修改] 构建 XML Output ===
    raw_trace = outputs.get('teacher_thought_trace', 'None')
    clean_trace = re.sub(r"^(分析|Analysis)\s*[:：]\s*", "", raw_trace)
    
    full_output = f"""<Thought>
<Strategy>{outputs.get('teacher_strategy', 'NORMAL')}</Strategy>
<Trace>{clean_trace}</Trace>
<Instruction>{outputs.get('teacher_instruction', 'None')}</Instruction>
</Thought>
<Response>
{outputs.get('teacher_speak', '')}
</Response>"""

    # Instruction 明确要求 XML 格式
    instruction_text = "你是一名苏格拉底式化学老师。请分析提供的 XML 上下文，构思教学策略，并输出严格遵循 XML 格式的思考过程(<Thought>)和回复(<Response>)。"

    return {
        "instruction": instruction_text,
        "input": full_input,
        "output": full_output
    }

def process_folder():
    sft_data = []
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ 错误: 文件夹 {INPUT_FOLDER} 不存在。")
        return

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.jsonl"))
    print(f"📂 找到 {len(files)} 个文件，开始处理...")

    for file_path in files:
        history_buffer = []
        current_meta = {}
        last_obstacles_hash = "" 
        stuck_counter = 0        
        experiment_name = "未知化学实验"

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    if record.get('meta_type') == 'session_info':
                        current_meta = record
                        experiment_name = record.get('experiment_name', "未知化学实验")
                        continue
                    
                    if 'inputs' not in record or 'outputs' not in record: continue
                    if not record['outputs'].get('teacher_speak'): continue

                    # 卡顿逻辑
                    hidden = record.get('hidden_states', {})
                    raw_fails = hidden.get('fail_reasons', [])
                    current_obs = sorted([r for r in raw_fails if "验证通过" not in r and "Success" not in r])
                    current_obs_hash = str(current_obs)
                    
                    if current_obs and current_obs_hash == last_obstacles_hash: stuck_counter += 1
                    else: stuck_counter = 0; last_obstacles_hash = current_obs_hash

                    history_str = "\n".join(history_buffer[-MAX_HISTORY_TURNS:])
                    sample = convert_raw_turn_to_sample(record, history_str, current_meta, stuck_counter, experiment_name)
                    sft_data.append(sample)

                    s_speak = record['inputs'].get('student_speak', '')
                    s_acts = record['inputs'].get('student_actions', [])
                    is_int = record.get('hidden_states', {}).get('intercepted', False)
                    act_summary = ""
                    if s_acts: act_summary = " (试图操作被拦截)" if is_int else " (执行操作)"
                    
                    t_speak = record['outputs'].get('teacher_speak', '')
                    if s_speak or act_summary: history_buffer.append(f"Student: {s_speak}{act_summary}")
                    if t_speak: history_buffer.append(f"Teacher: {t_speak}")

                except json.JSONDecodeError: continue

    print(f"💾 正在保存 {len(sft_data)} 条数据至 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"✅ XML 格式转换完成！结构更加清晰，抗噪性更强。")

if __name__ == "__main__":
    process_folder()