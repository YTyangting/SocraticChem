import json
import os

# --- 全局配置: 学生画像库 ---
STUDENT_PROFILES = [
    {"name": "标准-新手", "traits": ["紧张", "犹豫"], "clumsiness": "high", "knowledge_level": "novice"},
    {"name": "标准-普通", "traits": ["按部就班", "听话"], "clumsiness": "average", "knowledge_level": "average"},
    {"name": "标准-学霸", "traits": ["自信", "专业"], "clumsiness": "low", "knowledge_level": "expert"},
    {"name": "眼高手低", "traits": ["理论满分", "动手能力差", "害怕失败"], "clumsiness": "high", "knowledge_level": "expert"},
    {"name": "盲目自信", "traits": ["鲁莽", "迷之自信", "不看说明书"], "clumsiness": "average", "knowledge_level": "novice"},
    {"name": "沉默寡言", "traits": ["极度内向", "不爱说话", "被动执行"], "clumsiness": "average", "knowledge_level": "average"},
    {"name": "好奇宝宝", "traits": ["好奇心过剩", "喜欢打岔", "注意力不集中"], "clumsiness": "high", "knowledge_level": "novice"},
    {"name": "叛逆挑战", "traits": ["固执", "质疑权威", "喜欢反着来"], "clumsiness": "average", "knowledge_level": "average"},
    {"name": "粗心大意", "traits": ["健忘", "忽略细节", "急躁"], "clumsiness": "average", "knowledge_level": "average"},
    {"name": "极度恐慌", "traits": ["过度谨慎", "手抖严重", "甚至不敢开始"], "clumsiness": "high", "knowledge_level": "novice"}
]

# --- 全局辅助变量 ---
DESC_TO_ID_MAP = {}        

# --- 辅助函数 ---

def build_desc_map(filepath):
    global DESC_TO_ID_MAP
    DESC_TO_ID_MAP = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    rec = json.loads(line)
                    if rec.get('record_type') == 'experiment_metadata':
                        nodes = rec.get('data', {}).get('dag_structure', {}).get('graph_nodes', [])
                        for n in nodes:
                            DESC_TO_ID_MAP[n['description'].strip()] = n['id']
                except: pass
    except Exception:
        pass

def get_profile_by_filename(filename):
    normalized_name = filename.replace("_", "-")
    for profile in STUDENT_PROFILES:
        p_name = profile['name']
        if p_name in filename or p_name in normalized_name:
            return profile
        keywords = p_name.split("-")[-1]
        if keywords in filename:
            return profile
    return STUDENT_PROFILES[1] 

def format_list(lst):
    return json.dumps(lst, ensure_ascii=False)

def format_actions(actions):
    res = []
    for a in actions:
        if isinstance(a, str):
            res.append(a)
            continue
        name = a.get('action', 'Unknown')
        params = [f"{k}={v}" for k, v in a.items() if k != 'action']
        if params:
            res.append(f"{name}({', '.join(params)})")
        else:
            res.append(name)
    return format_list(res)

def parse_snapshot(snapshot):
    if not snapshot: return "环境数据为空"
    target_data = snapshot.get('hardware', snapshot) if isinstance(snapshot, dict) else {}
    lines = ["【器材清单 (god view)】:"]
    for k, v in target_data.items():
        if not isinstance(v, dict): continue
        type_name = v.get('type', 'unknown')
        desc = f"- {k} ({type_name}): "
        props = []
        contents = v.get('contents', {})
        if contents and isinstance(contents, dict):
            subs_str = []
            for sub_name, amount in contents.items():
                if amount > 1e-4: 
                    subs_str.append(f"{sub_name}={amount:.3f}")
            if subs_str:
                props.append(f"含: [{', '.join(subs_str)}]")
        elif v.get('substances'):
             props.append(f"含: {[s['name'] for s in v['substances']]}")
        if v.get('is_stoppered') or v.get('is_sealed'): 
            props.append("已密封")
        temp = v.get('temperature', 25)
        if temp > 40:
            props.append(f"Temp={int(temp)}C")
        lines.append(desc + ", ".join(props) if props else desc + "空")
    return "\n    ".join(lines)

# --- 核心逻辑 ---

def calculate_decision_logic(dag_status, intent_analysis, current_desc_meta, initial_roots, last_decision):
    """
    计算逻辑 V20.0: 融合 Meta 对齐与 DAG 推理，优化文案
    """
    completed = [n['id'] for n in dag_status.get('completed_nodes', [])]
    unfinished_candidates = [t['id'] for t in dag_status.get('next_tasks', [])]
    
    intent_thought = "未检测到显著意图"
    intent_choice_text = ""
    if intent_analysis:
        intent_thought = intent_analysis.get('thought', '无详细分析')
        intent_choice_text = intent_analysis.get('choice', '')

    meta_target_id = DESC_TO_ID_MAP.get(current_desc_meta.strip(), None)

    trace_lines = []
    decision_node = "None"
    state_enum = "Thinking..."

    # 1. 优先检查 Meta 是否指向了一个“已完成”的节点（回溯/结算场景）
    if meta_target_id and meta_target_id in completed:
        decision_node = meta_target_id
        trace_lines.append(f"上下文分析：当前 Meta 描述仍聚焦于节点 \"{meta_target_id}\"。")
        trace_lines.append(f"状态确认：DAG 显示该节点刚刚完成 (Success)。")
        trace_lines.append(f"判定：锁定该节点以进行结果确认 (COMPLETED)。")
        state_enum = "COMPLETED"
    
    # 2. 否则，走正常的 DAG 推进逻辑
    else:
        if not unfinished_candidates and not completed:
            unfinished_candidates = initial_roots 
            trace_lines.append("状态分析：实验处于开场阶段 (Completed/NextTasks 为空)。")
            trace_lines.append(f"全局规划：初始化识别出 {len(initial_roots)} 个起始任务。")
            decision_node = "WAITING"
            state_enum = "IDLE"
        elif unfinished_candidates:
            if len(unfinished_candidates) == 1:
                target = unfinished_candidates[0]
                if target == last_decision:
                    trace_lines.append(f"状态延续：上一轮锁定任务 \"{target}\" 尚未完成，继续保持锁定。")
                else:
                    trace_lines.append(f"DAG 状态：当前唯一活跃候选为 \"{target}\"。")
                    trace_lines.append(f"判定：锁定节点 \"{target}\"。")
                decision_node = target
                state_enum = "LOCKED_SINGLE"
            else:
                matched_id = "None"
                for uid in unfinished_candidates:
                    if uid in intent_choice_text: matched_id = uid; break
                if matched_id == "None": matched_id = unfinished_candidates[0]
                
                trace_lines.append(f"DAG 状态：存在多个分支 {unfinished_candidates}。")
                if intent_thought != "未检测到显著意图":
                    trace_lines.append(f"意图分析：{intent_thought}")
                trace_lines.append(f"判定：锁定分支节点 \"{matched_id}\"。")
                decision_node = matched_id
                state_enum = "LOCKED_BRANCH"
        else:
            decision_node = "WAITING"
            state_enum = "IDLE"
    
    return trace_lines, decision_node, state_enum, completed, unfinished_candidates

def build_find_node_xml(trace_lines, completed, unfinished, decision_node, state_enum):
    return f"""<FindcurrentNode>
  <LogicTrace>
    {chr(10).join([f"    {line}" for line in trace_lines])}
  </LogicTrace>
  <StatusCheck>
    <Completed>{format_list(completed)}</Completed>
    <UnfinishedCandidates>{format_list(unfinished)}</UnfinishedCandidates>
  </StatusCheck>
  <Decision>
    <CurrentNode>{decision_node}</CurrentNode>
    <State>{state_enum}</State>
  </Decision>
</FindcurrentNode>"""


# --- 主处理流程 ---
def process_file_v3(input_path, output_path):
    build_desc_map(input_path)
    filename = os.path.basename(input_path)
    profile = get_profile_by_filename(filename)
    
    session_ctx_cache = {}
    session_last_info = {} 

    print(f"开始处理: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line: continue
            try:
                record = json.loads(line)
                sid = record.get('session_id', 'default')

                if record.get('record_type') == 'experiment_metadata':
                    data = record.get('data', {})
                    graph_nodes = data.get('dag_structure', {}).get('graph_nodes', [])
                    initial_roots = [n['id'] for n in graph_nodes if not n.get('dependencies')]
                    session_ctx_cache[sid] = {
                        "xdl": data.get('xdl_enriched', ''),
                        "dag": json.dumps(data.get('dag_structure', {}), ensure_ascii=False, indent=2),
                        "initial_roots": initial_roots
                    }
                    continue

                data = record.get('data', {})
                if 'agents' not in data: continue 
                
                meta = data.get('meta', {})
                agents = data.get('agents', {})
                student_agent = agents.get('student', {})
                teacher_agent = agents.get('teacher', {}) 
                
                policy_in = {}
                policy_out = {}
                if 'policy_feedback' in agents: 
                    policy_in = agents['policy_feedback'].get('input', {})
                    policy_out = agents['policy_feedback'].get('output', {})
                elif 'policy_feedback' in data:
                    policy_in = data['policy_feedback'].get('input', {})
                    policy_out = data['policy_feedback'].get('output', {})
                elif 'policy' in agents and isinstance(agents['policy'].get('input'), dict):
                    policy_in = agents['policy'].get('input', {})
                    policy_out = agents['policy'].get('output', {})

                dag_status = policy_in.get('dag_status', {})
                intent_analysis = policy_in.get('intent_analysis', {})
                is_node_success = dag_status.get('is_success', False)
                
                ctx = session_ctx_cache.get(sid, {"xdl":"", "dag":"", "initial_roots": []})
                
                last_decision_node = "None"
                last_state_msg = ""
                if sid in session_last_info:
                    last_info = session_last_info[sid]
                    last_dag = last_info['dag']         
                    last_decision_node = last_info['decision']
                    last_comp = [n['id'] for n in last_dag.get('completed_nodes', [])]
                    last_next = [t['id'] for t in last_dag.get('next_tasks', [])]
                    if not last_comp and not last_next: 
                        last_next = ctx['initial_roots']
                    if last_decision_node not in ["WAITING", "None", "IDLE"]:
                        status_str = "未知"
                        if last_decision_node in last_comp:
                            status_str = "已完成"
                        else:
                            status_str = "进行中"
                        last_state_msg = f"【系统隐含状态】: 已完成节点={format_list(last_comp)}, 上轮锁定节点=\"{last_decision_node}\" (状态: {status_str}), 当前待办候选={format_list(last_next)}"
                    else:
                        last_state_msg = f"【系统隐含状态】: 已完成节点={format_list(last_comp)}, 当前待办候选={format_list(last_next)}"
                else:
                    last_state_msg = f"【系统隐含状态】: 实验初始待办={format_list(ctx['initial_roots'])}"

                student_out = student_agent.get('output', {})
                teacher_out = teacher_agent.get('output', {})
                
                before_env = parse_snapshot(data.get('physics_context', {}).get('snapshot_before', {}))
                
                teacher_input_raw = teacher_agent.get('input', {})
                history_list = []
                if isinstance(teacher_input_raw, str):
                    student_in = student_agent.get('input', {})
                    if isinstance(student_in, dict):
                        sys_msg = student_in.get('last_teacher_msg', '')
                        if sys_msg:
                            history_list.append({"role": "system", "content": sys_msg})
                else:
                    history_raw = teacher_input_raw.get('history_context', [])
                    if isinstance(history_raw, str):
                        try: history_list = json.loads(history_raw)
                        except: history_list = []
                    else:
                        history_list = history_raw if isinstance(history_raw, list) else []
                
                current_speak = student_out.get('speak', '').strip()
                if history_list and current_speak:
                    cut_index = -1
                    for i in range(len(history_list) - 1, max(-1, len(history_list)-5), -1):
                        item = history_list[i]
                        if item.get('role') == 'student' and item.get('content', '').strip() == current_speak:
                            cut_index = i
                            break
                    if cut_index != -1:
                        history_list = history_list[:cut_index]

                history_list.append({"role": "system", "content": last_state_msg})
                history_str = json.dumps(history_list, ensure_ascii=False, indent=2)

                student_actions = student_out.get('actions_ready_to_run', [])
                student_act_str = format_actions(student_actions)
                
                input_xml = f"""<Input>
  <Profile>
    <Name>{profile['name']}</Name>
    <Traits>{format_list(profile['traits'])}</Traits>
  </Profile>

  {ctx['xdl']}

  <DAG>
    {ctx['dag']}
  </DAG>

  <BeforeEnvironment>
    {before_env}
  </BeforeEnvironment>

  <History>
{history_str}
  </History>

  <Student>
    <Speak>"{student_out.get('speak', '')}"</Speak>
    <Action>{student_act_str}</Action>
  </Student>
</Input>"""

                current_desc_meta = meta.get('current_task_desc', '')
                trace_lines, decision_node, state_enum, comp_list, unfin_list = calculate_decision_logic(
                    dag_status, intent_analysis, current_desc_meta, ctx['initial_roots'], last_decision_node
                )
                find_node_xml = build_find_node_xml(trace_lines, comp_list, unfin_list, decision_node, state_enum)
                
                # --- Fail Reasons 提取 ---
                raw_fail = policy_in.get('fail_reasons', [])
                hard_errors = []
                progress_hints = []
                for r in raw_fail:
                    r_txt = str(r).strip()
                    if not r_txt: continue
                    if any(token in r_txt for token in ["验证通过", "Success", "None", "Pass"]): continue
                    
                    if "进度提示" in r_txt:
                        progress_hints.append(r_txt)
                    else:
                        hard_errors.append(r_txt)
                
                constraints_text = meta.get('constraints_text', '')
                checklist_lines = []
                if constraints_text:
                    c_lines = [l.strip() for l in constraints_text.split('\n') if l.strip()]
                    for cl in c_lines:
                        cl_clean = cl.replace('- ', '').replace('* ', '')
                        checklist_lines.append(f"- 预期目标: {cl_clean}")
                
                # --- 🔥 V20.0 核心修正: 完成即通过 ---
                final_status = "PASS"
                logic_status_txt = "正常"
                display_reasons = []

                # 如果当前决策的节点已经处于 Completed 列表中，强制 PASS
                if decision_node in comp_list:
                    final_status = "PASS"
                    checklist_lines.append("- 实际结果: PASS [该任务已完成]")
                    logic_status_txt = "正常"
                # 否则继续使用原有逻辑
                elif is_node_success:
                    final_status = "PASS"
                    checklist_lines.append("- 实际结果: PASS [系统自检无异常]")
                    logic_status_txt = "正常"
                else:
                    final_status = "FAIL"
                    if hard_errors:
                        display_reasons = hard_errors
                        logic_status_txt = "阻碍"
                    elif progress_hints:
                        display_reasons = progress_hints
                        logic_status_txt = "正常" 
                    else:
                        display_reasons = ["当前任务尚未完成"]
                        logic_status_txt = "正常"
                    
                    checklist_lines.append(f"- 实际结果: FAIL [{'; '.join(display_reasons)}]")
                
                check_desc = "\n    ".join(checklist_lines)

                vertify_xml = f"""<vertify>
  <CheckList>
    {check_desc}
  </CheckList>
  <Result> {'❌' if final_status == 'FAIL' else '✅'} </Result>
</vertify>"""
                
                logic_fail_txt = '; '.join(display_reasons)
                
#                 logic_xml = f"""<LogicAnalysis>
#   <Status>{logic_status_txt}</Status>
#   <FailReason>{logic_fail_txt}</FailReason>
# </LogicAnalysis>"""

                teacher_analysis = teacher_out.get('analysis', '')
                teacher_process = teacher_out.get('thought_process', '')
                policy_trace = policy_out.get('thought_trace', '')
                strategy_type = policy_out.get('strategy_type', teacher_out.get('strategy', 'GUIDANCE'))

                combined_trace = ""
                if policy_trace:
                     combined_trace += f"【策略层评估】{policy_trace}\n    "
                if teacher_process:
                     combined_trace += f"【执行层思考】{teacher_process}"
                if not combined_trace:
                     combined_trace = "【执行层思考】(无详细思考记录)"

                thought_xml = f"""<Thought>
  <CognitiveDiagnosis>
    {teacher_analysis}
  </CognitiveDiagnosis>
  <Strategy>{strategy_type}</Strategy>
  <Trace>
    {combined_trace}
  </Trace>
  <Instruction>{policy_out.get('instruction_to_teacher', '参考上述策略回复。')}</Instruction>
</Thought>"""

                response_xml = f"<Response>\n{teacher_out.get('response', '')}\n</Response>"

                output_block = f"output:\n{find_node_xml}\n\n{vertify_xml}\n\n{thought_xml}\n\n{response_xml}"
                f_out.write(input_xml + "\n\n" + output_block + "\n" + "-"*80 + "\n")
                
                session_last_info[sid] = {
                    'dag': dag_status,
                    'decision': decision_node
                }

            except json.JSONDecodeError: continue
            except Exception as e:
                continue
            
    print(f"转换完成: {output_path}")

# 执行
input_file = 'finetune_data_dataset/2-5_高锰酸钾制取氧气_sample_chem_train_14_标准_普通_ed4f1e.jsonl'
output_file = 'fine_tuning_v20_final.txt'

if os.path.exists(input_file):
    process_file_v3(input_file, output_file)
else:
    print("文件不存在，请检查路径")