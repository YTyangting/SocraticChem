import io
import json
import base64
import argparse
import uvicorn
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 导入你原有的核心模块库 (确保文件名正确)
from soc_chem_dia_Interactive_Platform import (
    InteractivePlatform, Config, ObservationEngine, 
    SmartTaskSelector, describe_snapshot_briefly, OpenAI
)

app = FastAPI()

# 建立一个全局对象来保存“单例”状态
class SessionState:
    def __init__(self):
        self.platform: InteractivePlatform = None
        self.obs = None
        self.task_selector = None
        self.current_task_desc = ""
        self.node_or_list = None
        self.xdl_context_info = ""
        self.constraint_str = ""
        self.last_fail_reasons = []
        self.language: str = "zh"  # "zh" 或 "en"
        self.img_client: OpenAI = None  # 图片生成专用客户端

session = SessionState()

# 定义 Unity 传过来的请求格式
class ChatRequest(BaseModel):
    speak: str          # 学生说的话
    action_str: str     # 学生执行的动作 (JSON字符串，没有动作则传 "[]")

# 定义返回给 Unity 的格式
class ChatResponse(BaseModel):
    teacher_response: str
    env_state_desc: str # 物理环境的文字描述（可选，方便在 Unity 显示状态）
    is_crashed: bool    # 实验室是否炸了

def _build_dag_prompt() -> str:
    """从当前 DAG 状态构建图片生成 prompt（全英文）"""
    p = session.platform
    G = p.oracle.graph
    nodes = p.oracle.nodes
    completed = p.oracle.force_completed_nodes
    focus_id = session.task_selector.current_focus_id if session.task_selector else None

    node_lines = []
    for nid, node in nodes.items():
        if nid in completed:
            status = "Completed"
        elif nid == focus_id:
            status = "In Progress"
        else:
            status = "Pending"
        deps = ", ".join(node.dependencies) if node.dependencies else "None"
        node_lines.append(f"- [{nid}] {node.desc} (status: {status}, depends on: {deps})")

    edge_lines = [f"  {u} -> {v}" for u, v in G.edges()]

    instruction = (
        "Generate a clean, professional directed flowchart diagram for a chemistry experiment. "
        "Use GREEN color for completed nodes, ORANGE for in-progress nodes, GRAY for pending nodes. "
        "Show directed arrows for dependency relationships. White background. "
        "ALL TEXT in the image MUST be in English. "
        "If any node description below is in Chinese, translate it to English in the diagram. "
        "All text must be clearly readable at a large font size."
    )

    prompt = f"""{instruction}

Title: Experiment Task Flow Chart (DAG)
Legend: Completed (green), In Progress (orange), Pending (gray)

Nodes:
{chr(10).join(node_lines)}

Dependency edges:
{chr(10).join(edge_lines) if edge_lines else "  (no dependencies, all nodes are independent)"}
"""
    return prompt


def render_dag_image() -> bytes:
    """将当前 DAG 状态渲染为 PNG 图片，返回字节流"""
    # 优先使用 gpt-image-2 生成
    if session.img_client:
        try:
            prompt = _build_dag_prompt()
            response = session.img_client.images.generate(
                model="gpt-image-2",
                prompt=prompt,
                size="1536x1024",
                n=1,
            )
            img_data = base64.b64decode(response.data[0].b64_json)
            return img_data
        except Exception as e:
            print(f"⚠️ [DAG图片生成] gpt-image-2 调用失败，回退到 matplotlib: {e}")

    # 回退：matplotlib 渲染
    p = session.platform
    G = p.oracle.graph
    nodes = p.oracle.nodes
    completed = p.oracle.force_completed_nodes
    focus_id = session.task_selector.current_focus_id if session.task_selector else None

    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK JP', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(max(16, len(nodes) * 2.5), 10))

    try:
        layers = list(nx.topological_generations(G))
    except nx.NetworkXError:
        layers = [list(G.nodes)]

    pos = {}
    for layer_idx, layer in enumerate(layers):
        for node_idx, node_id in enumerate(layer):
            pos[node_id] = (layer_idx, -(node_idx - (len(layer) - 1) / 2))

    node_ids = list(nodes.keys())
    completed_ids = [nid for nid in node_ids if nid in completed]
    focus_ids = [nid for nid in node_ids if nid == focus_id and nid not in completed]
    pending_ids = [nid for nid in node_ids if nid not in completed and nid != focus_id]

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=20,
                           edge_color='#888888', width=1.5, alpha=0.7,
                           connectionstyle="arc3,rad=0.05")

    node_kwargs = dict(node_size=5000, ax=ax)
    if completed_ids:
        nx.draw_networkx_nodes(G, pos, nodelist=completed_ids, node_color='#4CAF50', **node_kwargs)
    if focus_ids:
        nx.draw_networkx_nodes(G, pos, nodelist=focus_ids, node_color='#FF9800', **node_kwargs)
    if pending_ids:
        nx.draw_networkx_nodes(G, pos, nodelist=pending_ids, node_color='#BDBDBD', **node_kwargs)

    labels = {nid: node.desc for nid, node in nodes.items()}
    font_kwargs = dict(font_family='WenQuanYi Zen Hei', font_size=13, ax=ax)
    if completed_ids:
        nx.draw_networkx_labels(G, pos, labels={n: labels[n] for n in completed_ids},
                                font_color='white', **font_kwargs)
    if focus_ids:
        nx.draw_networkx_labels(G, pos, labels={n: labels[n] for n in focus_ids},
                                font_color='black', **font_kwargs)
    if pending_ids:
        nx.draw_networkx_labels(G, pos, labels={n: labels[n] for n in pending_ids},
                                font_color='#555555', **font_kwargs)

    legend_elements = [
        Patch(facecolor='#4CAF50', label='已完成'),
        Patch(facecolor='#FF9800', label='进行中'),
        Patch(facecolor='#BDBDBD', label='待完成'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=13)
    ax.set_title("实验任务流程图 (DAG)", fontsize=18, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@app.get("/api/dag")
async def get_dag():
    """
    返回完整的 DAG 任务图图片：绿色=已完成，橙色=进行中，灰色=待完成
    """
    p = session.platform
    if not p:
        return StreamingResponse(io.BytesIO(b""), status_code=404, media_type="text/plain")

    image_bytes = render_dag_image()
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

@app.post("/api/init_experiment")
async def init_experiment(exp_id: str = "1"):
    """
    初始化实验环境（对应原代码 run_interactive 的前置准备）
    """
    # 你的实验路径配置
    experiment_paths = {
        "1": "/home/yjh/socChem_final/experiments/2-5 高锰酸钾制取氧气_sample_chem_train_14.xdl",
        "2": "/home/yjh/socChem_final/experiments/1 大理石与稀盐酸反应_sample_chem_train_0.xdl",
        "3": "/home/yjh/socChem_final/experiments/1 水的沸腾_sample_chem_train_2.xdl",
        "4": "/home/yjh/socChem_final/experiments/1-1 光束通过溶液和胶体_sample_chem_train_69.xdl",
        "5": "/home/yjh/socChem_final/experiments/1-1 蒸馏水和无水乙醇与钠的反应_sample_chem_train_149.xdl",
        "6": "/home/yjh/socChem_final/experiments/1-5 氢氧化钠滴加硫酸铜溶液_sample_chem_train_8.xdl",
        "7": "/home/yjh/socChem_final/experiments/1-探究1 重结晶方法提纯苯甲酸_sample_chem_train_150.xdl",
        "8": "/home/yjh/socChem_final/experiments/2-2 木炭分别在空气和 氧气中燃烧_sample_chem_train_11.xdl",
        "9": "/home/yjh/socChem_final/experiments/2-2 钠的燃烧_sample_chem_train_74.xdl",
        "10": "/home/yjh/socChem_final/experiments/1 石蜡的融化_sample_chem_train_3.xdl"
    }

    xdl_path = experiment_paths.get(exp_id, experiment_paths["1"])

    # 实例化大模型客户端与平台
    client = OpenAI(api_key=Config.OPENAI_API_KEY, base_url=Config.OPENAI_BASE_URL)
    session.platform = InteractivePlatform(client=client, yaml_path=xdl_path)

    # 初始化图片生成专用客户端
    try:
        session.img_client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url="https://api.vveai.com/v1"
        )
    except Exception as e:
        print(f"⚠️ [图片生成] img_client 初始化失败，DAG 图片将使用 matplotlib 渲染: {e}")
        session.img_client = None
    
    # 初始化场景与状态
    session.obs = session.platform.sim.get_snapshot()
    scenario_intro = session.platform._build_scenario_intro()
    session.task_selector = SmartTaskSelector()
    
    dag_status = session.platform.oracle.analyze(session.obs, set())
    selection_result = session.task_selector.select_next_task(dag_status, cognitive_model=session.platform.cognitive_model)
    (session.current_task_desc, session.node_or_list, 
     session.xdl_context_info, session.constraint_str) = session.platform._resolve_task_info(selection_result, session.task_selector)
    
    session.platform.memory.add_turn("system", scenario_intro)
    session.last_fail_reasons = []

    # [新增日志] 打印初始化状态与当前目标
    print(f"\n============== [初始化实验 {exp_id}] ==============")
    print(f"🥼 加载XDL文件: {xdl_path}")
    print(f"🎯 初始目标 (Current Task): {session.current_task_desc}")
    print(f"📋 任务约束 (Constraints): {session.constraint_str}")
    print(f"=================================================\n")
    
    # 获取老师的开场白
    god_view_str_0 = ObservationEngine.describe_full_state(session.obs, "god")
    t0_resp = session.platform.teacher.respond(
        history_str=json.dumps([], ensure_ascii=False),
        policy_decision={"strategy_type": "RAPPORT_BUILDING", "instruction_to_teacher": "热情欢迎学生，引导其开始第一步。"},
        focus_goal=f"引导进入任务：{session.current_task_desc}",
        cognitive_state="未知（真实学生）",
        scaffold_ctx={"level": 1, "frustration": "Low", "errors": 0},
        env_state=god_view_str_0,
        reference_info=session.xdl_context_info,
        policy_instruction="热情欢迎学生，引导其开始第一步。",
        causal_insight="（实验刚开始，物理状态平稳）",
        language=session.language
    )
    
    teacher_msg = t0_resp.get("response", "你好，请开始实验。")
    session.platform.memory.add_turn("teacher", teacher_msg)
    
    return {"message": "Init Success", "teacher_response": teacher_msg}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_step(request: ChatRequest):
    """
    处理玩家每一次发言/动作的执行闭环
    """
    p = session.platform # 简写
    if not p:
        return ChatResponse(teacher_response="实验尚未初始化，请重启系统。", env_state_desc="", is_crashed=True)

    # 1. 接收 Unity 传来的输入
    s_speak = request.speak.strip()
    pending_actions = []
    if request.action_str and request.action_str != "[]":
        try:
            pending_actions = json.loads(request.action_str)
            if not isinstance(pending_actions, list):
                pending_actions = [pending_actions]
        except Exception:
            pass # 解析失败则视为无动作
            
    p.memory.add_turn("student", s_speak if s_speak else "（静默操作）")

    # [新增日志] 打印玩家输入
    print(f"\n============== [玩家回合开始] ==============")
    print(f"🗣️ 玩家发言: {s_speak if s_speak else '无'}")
    print(f"🎮 玩家动作: {json.dumps(pending_actions, ensure_ascii=False) if pending_actions else '无动作'}")
    print(f"🎯 当前执行目标: {session.current_task_desc}")

    # 2. 意图分析与 Policy 拦截 (Ghost Sim)
    obs_before = session.obs
    short_env_summary = describe_snapshot_briefly(obs_before)
    
    is_branching = isinstance(session.node_or_list, list)
    if is_branching and session.node_or_list:
        target_node, _ = p._match_student_intent(s_speak, pending_actions, session.node_or_list)
        if target_node:
            session.task_selector.current_focus_id = target_node.id
            (session.current_task_desc, _, 
             session.xdl_context_info, session.constraint_str) = p._resolve_task_info(target_node, session.task_selector)
            # [新增日志] 意图分支更新
            print(f"🔄 [意图匹配] 识别到分支意图，更新目标为: {session.current_task_desc}")

    ghost_report = p.sim.fork_and_predict(pending_actions, steps=20)
    policy_ctx_text = f"{session.current_task_desc}\n【环境】:{short_env_summary}\n【标准】:{session.constraint_str}"
    policy_decision = p.policy_agent.decide_on_intent(s_speak, pending_actions, ghost_report, policy_ctx_text)

    is_intercepted = policy_decision.get('decision') in ['INTERCEPT', 'EMERGENCY_STOP']
    intercept_reason = policy_decision.get('reasoning', '风险操作')
    
    obs_after = obs_before
    real_exec_info = ""

    # 3. 物理执行与验证
    if is_intercepted:
        p.memory.add_turn("system", f"【系统警告】动作被拦截！原因: {intercept_reason}")
        policy_decision = {"strategy_type": "PREDICTIVE_QUESTIONING", "instruction_to_teacher": f"拦截！指出 {intercept_reason}"}
        final_focus_goal = "[最高优先级] 阻止危险行为"
        final_policy_instr = f"【拦截模式】{intercept_reason}"
        final_ref_info = session.xdl_context_info 
        
        # [新增日志]
        print(f"⛔ [动作拦截] 被 Policy 拦截！原因: {intercept_reason}")
    else:
        # 执行物理动作
        if pending_actions:
            real_exec_info, obs_after = p.sim.execute_batch(pending_actions)
            p.memory.add_turn("system", f"【系统反馈】观察结果: {real_exec_info}")
            # [新增日志]
            print(f"⚙️ [物理执行] 成功应用动作，环境反馈: {real_exec_info}")
        else:
            wait_log, obs_after = p.sim.execute_batch([], duration=3.0)
            real_exec_info = wait_log if "观测到" in wait_log else "(环境静止)"
            p.memory.add_turn("system", f"【系统反馈】{wait_log}")
            # [新增日志]
            print(f"⏳ [物理静置] 环境反馈: {real_exec_info}")

        search_scope = list({act['vessel'] for act in pending_actions if act.get('vessel')}) if pending_actions else None
        is_goal_met, fail_reasons = p.validator.validate_state(obs_after, p.current_constraints, focus_ids=search_scope)
        session.last_fail_reasons = fail_reasons if not is_goal_met else []

        # [新增日志] 打印目标校验结果与未完成原因
        print(f"🔎 [状态校验] 目标是否达成: {is_goal_met}")
        if not is_goal_met and fail_reasons:
            print(f"❌ [当前未完成的原因]:")
            for fr in fail_reasons:
                print(f"   - {fr}")
        elif is_goal_met:
            print(f"✅ [校验通过] 玩家成功完成了当前任务节点。")

        if is_goal_met and session.node_or_list and not isinstance(session.node_or_list, list):
            p.oracle.mark_node_complete(session.node_or_list.id)
            session.task_selector.current_focus_id = None

        # 教学反馈策略
        policy_decision = p.policy_agent.decide(
            student_text=s_speak,
            dag_status=p.oracle.analyze(obs_after, set()),
            cognitive_diagnosis="真实学生",
            intent_analysis={"is_consistent": True},
            sim_engine=p.sim,
            pending_actions=pending_actions,
            student_profile={"knowledge_level": "average"},
            current_snapshot=obs_after,
            is_goal_met=is_goal_met,
            fail_reasons=fail_reasons,
            physics_observation=real_exec_info,
            current_task_desc=session.current_task_desc,
            is_making_progress=False
        )

        final_focus_goal = p.dialogue_manager.synthesize_prompt(session.current_task_desc, policy_decision)
        final_policy_instr = policy_decision.get("instruction_to_teacher", "")
        validation_hint = "\n".join([f"- 待解决: {r}" for r in fail_reasons]) if not is_goal_met else ""
        final_ref_info = session.xdl_context_info + "\n" + validation_hint

    # 4. Teacher Agent 回复
    full_env_state_str = ObservationEngine.describe_full_state(obs_after, perspective="god")
    hist_context_str = p.memory.get_context_for_prompt()

    # === (5) 执行后：给出新器材状态，并触发 Teacher 反馈 ===
    # [强化日志] 使用你原有的调用，但增加了清晰的装饰，突出“连接状态”
    print(f"\n🔌 [操作后的物体的连接状态与物理环境]:\n{ObservationEngine.describe_full_state(obs_after, 'student')}")
    
    t_resp = p.teacher.respond(
        history_str=hist_context_str,
        policy_decision=policy_decision,
        focus_goal=final_focus_goal,
        cognitive_state="未知（真实学生）",
        scaffold_ctx={"level": 1, "frustration": "Low", "errors": 0},
        env_state=full_env_state_str,
        reference_info=final_ref_info,
        policy_instruction=final_policy_instr,
        causal_insight="",
        language=session.language
    )

    t_content = t_resp.get("response", "系统老师暂时无响应。")
    p.memory.add_turn("teacher", t_content)
    # [新增日志] 打印系统老师的回复
    print(f"👩‍🏫 [教师回复]: {t_content}")

    # 5. 更新状态留给下一轮
    session.obs = obs_after
    dag_status = p.oracle.analyze(session.obs, set())
    selection_result = session.task_selector.select_next_task(dag_status, cognitive_model=p.cognitive_model)
    (session.current_task_desc, session.node_or_list, 
     session.xdl_context_info, session.constraint_str) = p._resolve_task_info(selection_result, session.task_selector)

    # 检查实验是否全剧终
    if not selection_result and dag_status.get('completed_nodes'):
         t_content += "\n\n🎉 恭喜！所有实验步骤已完成！"
         print("\n🎉 [实验进度]: 所有实验步骤已完成！")
    else:
         # [新增日志] 打印下一轮分配的任务
         print(f"\n➡️ [下一轮当前目标] (Next Task): {session.current_task_desc}")

    print(f"============== [玩家回合结束] ==============\n")

    # === 修改后的代码 ===
    student_view_str = ObservationEngine.describe_full_state(obs_after, 'student')

    return ChatResponse(
        teacher_response=t_content,
        env_state_desc=student_view_str,
        is_crashed=is_intercepted  # 修改点：传入已有的 is_intercepted 变量
        # 如果你上面模型里没改名，这里就写: is_crashed=is_intercepted
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="化学实验教学平台 API 服务器")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh",
                        help="教师回复语言: zh=中文(默认), en=英文")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认 8000)")
    args = parser.parse_args()

    session.language = args.lang
    print(f"启动服务器... 监听 http://{args.host}:{args.port} [语言: {args.lang}]")
    uvicorn.run(app, host=args.host, port=args.port)