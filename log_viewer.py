import json
import os
import glob
import webbrowser
import argparse
from datetime import datetime

# ================= Configuration =================
LOG_DIR = "logs"
# ===============================================

def read_jsonl_file(file_path):
    """
    读取并解析 .jsonl 文件
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return None

    print(f"📂 Loading log: {file_path}")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except Exception as e:
                    print(f"⚠️ Warning: Error parsing line {line_num}: {e}")
                    continue
    except Exception as e:
        print(f"❌ Error opening file: {e}")
        return None
        
    return data

def load_latest_log():
    """
    自动加载最新的日志文件
    """
    if not os.path.exists(LOG_DIR):
        print(f"Directory '{LOG_DIR}' does not exist.")
        return None, None

    # 查找 dialogue_*.jsonl 或 trace_*.jsonl
    list_of_files = glob.glob(f'{LOG_DIR}/*.jsonl')
    
    if not list_of_files:
        print(f"No log files found in {LOG_DIR}")
        return None, None
        
    latest_file = max(list_of_files, key=os.path.getctime)
    return read_jsonl_file(latest_file), latest_file

def safe_parse_json(content):
    """尝试解析可能是 JSON 字符串的内容"""
    if isinstance(content, dict):
        return content
    try:
        return json.loads(content)
    except:
        return {}

def render_html(logs, source_filename):
    """
    核心渲染函数：生成 HTML 报告
    """
    if not logs:
        print("No logs to render.")
        return None

    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(source_filename))[0]
    output_html = f"report_{base_name}_{datetime.now().strftime('%H%M%S')}.html"

    # 1. 预处理：按 Turn 分组
    turns = {}
    current_turn_id = 0
    
    # 获取实验元数据
    experiment_title = "Unknown Experiment"
    
    for entry in logs:
        payload = entry.get('data', {})
        message_type = entry.get('message', '')
        
        # 提取实验标题
        if message_type == "System_Initialization":
            experiment_title = payload.get('experiment', experiment_title)

        # 更新 Turn ID
        if 'turn' in payload:
            current_turn_id = payload['turn']
        
        # 获取步骤描述
        step_desc = payload.get('step_desc', payload.get('step', 'General Context'))
        
        if current_turn_id not in turns:
            turns[current_turn_id] = {
                "id": current_turn_id, 
                "step": step_desc,
                "events": []
            }
        
        # 更新步骤描述（如果是 Teacher Request，通常包含最新的 step_desc）
        if message_type == 'Teacher_LLM_Request' and payload.get('step_desc'):
             turns[current_turn_id]["step"] = payload.get('step_desc')

        turns[current_turn_id]["events"].append({
            "type": message_type, 
            "level": entry.get("level", "INFO"),
            "time": entry.get("timestamp"),
            "data": payload
        })

    # 2. 生成 HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>Socratic ChemLab Report</title>
        <style>
            :root {{
                --bg-color: #f4f7f6;
                --card-bg: #ffffff;
                --primary: #4a90e2;
                --teacher-bg: #e3f2fd;
                --student-bg: #fff3e0;
                --world-bg: #e0f7fa;
                --physics-bg: #e8eaf6;
                --reviewer-bg: #ffebee; /* Reddish for errors */
                --text-main: #2c3e50;
                --text-secondary: #7f8c8d;
            }}
            body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: var(--bg-color); color: var(--text-main); margin: 0; padding: 40px 20px; line-height: 1.6; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            
            /* Header */
            .report-header {{ text-align: center; margin-bottom: 40px; }}
            .report-header h1 {{ margin: 0; font-size: 2.2em; color: #2c3e50; }}
            .report-header p {{ margin: 5px 0 0; color: var(--text-secondary); font-size: 0.9em; }}
            .source-file {{ font-family: monospace; background: #e0e0e0; padding: 2px 6px; border-radius: 4px; }}

            /* Turn Card */
            .turn-section {{ margin-bottom: 40px; position: relative; }}
            .turn-header-row {{ margin-bottom: 20px; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }}
            .turn-marker {{ 
                background: #2c3e50; color: #fff; padding: 5px 15px; border-radius: 20px; 
                display: inline-block; font-weight: bold; font-size: 0.9em; margin-right: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .step-desc {{ color: var(--text-secondary); font-weight: 500; font-size: 1.1em; }}

            .timeline-item {{
                display: flex;
                margin-bottom: 20px;
                position: relative;
                align-items: flex-start;
            }}
            
            /* Avatars */
            .avatar {{
                width: 50px; height: 50px; border-radius: 50%; 
                display: flex; align-items: center; justify-content: center;
                font-size: 24px; flex-shrink: 0; margin-right: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); border: 2px solid #fff;
            }}
            .avatar.teacher {{ background: var(--teacher-bg); color: #1565c0; }}
            .avatar.student {{ background: var(--student-bg); color: #c62828; }}
            .avatar.world {{ background: var(--world-bg); color: #00838f; }}
            .avatar.physics {{ background: var(--physics-bg); color: #3f51b5; }}
            .avatar.reviewer {{ background: var(--reviewer-bg); color: #c62828; }}
            .avatar.init {{ background: #fff8e1; color: #ff8f00; }}

            /* Content Bubble */
            .bubble-content {{
                background: var(--card-bg);
                border-radius: 12px;
                padding: 15px 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                flex-grow: 1;
                border-left: 5px solid transparent;
                min-width: 0;
            }}

            /* Role Specific Borders */
            .timeline-item.teacher .bubble-content {{ border-left-color: #2196f3; }}
            .timeline-item.student .bubble-content {{ border-left-color: #ff9800; }}
            .timeline-item.world .bubble-content {{ border-left-color: #00bcd4; background: #fafafa; }}
            .timeline-item.physics .bubble-content {{ border-left-color: #3f51b5; background: #fdfdff; }}
            .timeline-item.reviewer .bubble-content {{ border-left-color: #f44336; }}

            /* Typography */
            .meta-header {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85em; color: var(--text-secondary); border-bottom: 1px solid #eee; padding-bottom: 5px;}}
            .role-name {{ font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; }}
            .timestamp {{ font-family: monospace; }}
            
            .thought-block {{
                background: #f8f9fa; border-left: 3px solid #ced4da;
                padding: 8px 12px; margin-bottom: 10px; color: #555;
                font-style: italic; font-size: 0.95em; border-radius: 0 4px 4px 0;
            }}
            .speak-block {{
                font-size: 1.05em; color: #2c3e50; font-weight: 500; margin-bottom: 5px;
            }}
            
            /* Setup Box */
            .setup-box {{
                background: linear-gradient(135deg, #fff 0%, #fefcf5 100%);
                border: 1px solid #fae588;
                padding: 20px; border-radius: 12px;
                box-shadow: 0 4px 15px rgba(255, 193, 7, 0.1);
            }}
            .setup-title {{ font-size: 1.4em; color: #f57c00; font-weight: bold; margin-bottom: 10px; }}
            .setup-hardware {{ margin-top: 15px; display: flex; flex-wrap: wrap; gap: 8px; }}
            .hw-tag {{ display: inline-block; background: #fff3e0; color: #e65100; padding: 4px 10px; border-radius: 15px; font-size: 0.85em; border: 1px solid #ffe0b2; }}
            .hw-detail {{ font-size: 0.8em; color: #8d6e63; margin-left: 5px; }}

            /* JSON & Code */
            .json-toggle {{ color: #2196f3; cursor: pointer; font-size: 0.85em; display: inline-block; margin-top: 5px; text-decoration: underline; }}
            .code-box {{ background: #2d2d2d; color: #f8f8f2; padding: 10px; border-radius: 6px; font-family: Consolas, monospace; font-size: 0.85em; overflow-x: auto; margin-top: 5px; white-space: pre-wrap; display: none; }}
            
            /* Status Badges */
            .status-badge {{
                display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; margin-left: 8px;
            }}
            .status-PASS {{ background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }}
            .status-FAIL {{ background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }}
            .status-WAIT {{ background: #fff3e0; color: #ef6c00; border: 1px solid #ffcc80; }}
            .status-INFO {{ background: #e1f5fe; color: #0277bd; border: 1px solid #81d4fa; }}

            /* Failure / Diagnosis */
            .error-box {{
                background: #ffebee; border: 1px solid #ffcdd2; color: #b71c1c;
                padding: 12px; border-radius: 6px; margin-top: 8px;
            }}
            .error-title {{ font-weight: bold; display: flex; align-items: center; gap: 5px; }}
            .error-detail {{ margin-top: 5px; font-size: 0.95em; }}
            .warning-box {{
                background: #fff8e1; border: 1px solid #ffecb3; color: #f57f17;
                padding: 10px; border-radius: 6px; margin-top: 8px; font-size: 0.9em;
            }}

            /* Actions */
            .action-line {{
                display: flex; align-items: center; gap: 8px; margin-bottom: 5px;
                background: #e1f5fe; padding: 6px 10px; border-radius: 6px; color: #0277bd; font-size: 0.9em; font-family: monospace;
            }}
            .physics-action {{
                 background: #e8eaf6; color: #303f9f; border-left: 3px solid #3f51b5;
            }}
        </style>
        <script>
            function toggleJson(id) {{
                var x = document.getElementById(id);
                if (x.style.display === "none" || x.style.display === "") {{
                    x.style.display = "block";
                }} else {{
                    x.style.display = "none";
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <div class="report-header">
                <h1>🧪 {experiment_title} Report</h1>
                <p>Source: <span class="source-file">{os.path.basename(source_filename)}</span></p>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
    """

    for t_id in sorted(turns.keys()):
        turn = turns[t_id]
        
        html_content += f"""
        <div class="turn-section">
            <div class="turn-header-row">
                <span class="turn-marker">TURN {turn['id']}</span>
                <span class="step-desc">{turn['step']}</span>
            </div>
        """
        
        event_idx = 0
        for event in turn['events']:
            event_idx += 1
            e_type = event['type']
            data = event['data']
            unique_id = f"t{t_id}_e{event_idx}"
            
            # --- Role Classification ---
            if "System_Initialization" in e_type:
                role_class = "init"
                avatar_icon = "🚩"
                role_name = "Setup"
            elif "Teacher" in e_type:
                role_class = "teacher"
                avatar_icon = "👩‍🏫"
                role_name = "Teacher"
            elif "Student" in e_type:
                role_class = "student"
                avatar_icon = "🧑‍🎓"
                role_name = "Student"
            elif "Physics" in e_type:
                role_class = "physics"
                avatar_icon = "⚗️"
                role_name = "Physics Engine"
            elif "Simulation" in e_type or "World" in e_type:
                role_class = "world"
                avatar_icon = "🌍"
                role_name = "Simulation"
            elif "Step_Failure" in e_type or "Action_Ignored" in e_type:
                role_class = "reviewer"
                avatar_icon = "⚠️"
                role_name = "System Monitor"
            else:
                role_class = "world"
                avatar_icon = "🤖"
                role_name = e_type

            # --- Content Rendering ---
            content_html = ""
            
            # 1. SETUP
            if e_type == "System_Initialization":
                exp_title = data.get('experiment', 'Unknown')
                total_steps = data.get('total_steps', 0)
                
                # Parse Initial Containers
                init_state = data.get('initial_state', {})
                containers = init_state.get('containers', {})
                
                hw_html = ""
                for name, props in containers.items():
                    # 简化显示：名字 + 颜色 + 体积
                    color = props.get('color_desc', '')
                    vol = props.get('volume_ml', 0)
                    detail = f"{vol}ml"
                    if color and color != "空":
                        detail += f" ({color})"
                    
                    hw_html += f"<span class='hw-tag'>{name} <span class='hw-detail'>{detail}</span></span>"

                content_html = f"""
                <div class="setup-box">
                    <div class="setup-title">🧪 {exp_title}</div>
                    <div style="margin-bottom:10px; color:#555;"><strong>Steps:</strong> {total_steps}</div>
                    <div class="setup-hardware">{hw_html}</div>
                </div>
                """
            
            # 2. TEACHER
            elif role_name == "Teacher":
                if "Request" in e_type:
                    sys_prompt = str(data.get('system_prompt_preview', ''))[:300] + "..."
                    mode = data.get('strategy_mode', 'N/A')
                    review_status = data.get('review_status', 'N/A')
                    
                    # 状态 Badge
                    status_class = f"status-{review_status}" if review_status in ["PASS", "FAIL", "WAIT", "INFO"] else ""
                    
                    content_html = f"""
                    <div style='color:#7f8c8d; font-size:0.9em; display:flex; align-items:center;'>
                        Mode: <strong>{mode}</strong>
                        <span class='status-badge {status_class}'>{review_status}</span>
                    </div>
                    """
                    content_html += f"<div style='margin-top:5px;'><span class='json-toggle' onclick='toggleJson(\"{unique_id}\")'>Show Prompt</span><div id='{unique_id}' class='code-box'>{sys_prompt}</div></div>"
                else:
                    raw_content = data.get('raw_content', {})
                    parsed_content = safe_parse_json(raw_content)
                    
                    thought = parsed_content.get('thought') or data.get('thought', '')
                    response = parsed_content.get('response') or data.get('response', '')
                    
                    if thought: content_html += f"<div class='thought-block'>💭 {thought}</div>"
                    if response: content_html += f"<div class='speak-block'>🗣️ {response}</div>"

            # 3. STUDENT
            elif role_name == "Student":
                if "Request" in e_type:
                    persona = data.get('persona', '')
                    content_html = f"<div style='color:#7f8c8d; font-size:0.9em;'>Student Persona: <strong>{persona}</strong></div>"
                else:
                    thought = data.get('thought', '')
                    speak = data.get('speak', '')
                    
                    if thought: content_html += f"<div class='thought-block'>💭 {thought}</div>"
                    if speak: content_html += f"<div class='speak-block'>🗣️ {speak}</div>"

            # 4. SYSTEM MONITOR (Failures/Ignored)
            elif role_name == "System Monitor":
                if e_type == "Step_Failure":
                    step_name = data.get('step', '')
                    reason = data.get('reason', '')
                    diagnosis = data.get('diagnosis', {})
                    
                    diag_str = ""
                    if diagnosis:
                        d_type = diagnosis.get('type', 'ERROR')
                        diag_items = []
                        for k, v in diagnosis.items():
                            if k != 'type': diag_items.append(f"{k}: {v}")
                        diag_str = f"{d_type} | " + ", ".join(diag_items)

                    content_html = f"""
                    <div class="error-box">
                        <div class="error-title">❌ Step Failed: {step_name}</div>
                        <div class="error-detail"><strong>Reason:</strong> {reason}</div>
                        <div class="error-detail" style="font-family:monospace; margin-top:5px;">🔍 {diag_str}</div>
                    </div>
                    """
                elif e_type == "Action_Ignored":
                    raw_actions = data.get('raw', [])
                    action_list_html = "".join([f"<li>{a}</li>" for a in raw_actions])
                    content_html = f"""
                    <div class="warning-box">
                        <strong>⚠️ Action Ignored (Conversation Only)</strong>
                        <ul style="margin:5px 0 0 20px; padding:0;">{action_list_html}</ul>
                    </div>
                    """

            # 5. PHYSICS ENGINE
            elif role_name == "Physics Engine":
                action = data.get('action', 'Unknown Action')
                obs = data.get('observation', '')
                content_html += f"<div class='action-line physics-action'>⚡ <strong>{action}</strong></div>"
                if obs:
                    content_html += f"<div style='font-size:0.9em; color:#303f9f; margin-bottom:5px;'>👁️ {obs}</div>"
                
                # Snapshot visualizer
                snapshot = data.get('containers')
                if snapshot:
                    snap_json = json.dumps(snapshot, indent=2, ensure_ascii=False)
                    content_html += f"<div style='margin-top:5px;'><span class='json-toggle' onclick='toggleJson(\"{unique_id}\")'>📊 Container State</span><div id='{unique_id}' class='code-box'>{snap_json}</div></div>"

            # 6. SIMULATION ACTION
            elif role_name == "Simulation":
                actions = data.get('actions', [])
                result_text = data.get('result_text', '')
                
                for act in actions:
                    act_desc = f"{act.get('action')} {act.get('vessel', '')}"
                    if act.get('reagent'): act_desc += f" + {act.get('reagent')}"
                    if act.get('volume'): act_desc += f" ({act.get('volume')}ml)"
                    if act.get('tool'): act_desc += f" using {act.get('tool')}"
                    
                    content_html += f"<div class='action-line'>⚡ {act_desc}</div>"

                if result_text:
                    content_html += f"<div style='margin-top:8px; color:#006064; font-weight:500; font-family:monospace; white-space: pre-wrap;'>{result_text}</div>"

            # Fallback
            else:
                 content_html += f"<pre style='font-size:0.8em; overflow:auto;'>{json.dumps(data, ensure_ascii=False)}</pre>"

            # HTML Assembly
            html_content += f"""
            <div class="timeline-item {role_class}">
                <div class="avatar {role_class}">{avatar_icon}</div>
                <div class="bubble-content">
                    <div class="meta-header">
                        <span class="role-name">{role_name}</span>
                        <span class="timestamp">{event['time'].split(' ')[1] if event['time'] else ''}</span>
                    </div>
                    {content_html}
                </div>
            </div>
            """
        
        html_content += "</div>" # End Turn Section

    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ Report generated: {os.path.abspath(output_html)}")
    return output_html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML report from Agent logs.")
    parser.add_argument("-f", "--file", help="Specific .jsonl file to load", default=None)
    
    args = parser.parse_args()

    logs = None
    filename = None

    # Logic: Load specific file if provided, otherwise load latest
    if args.file:
        logs = read_jsonl_file(args.file)
        filename = args.file
    else:
        logs, filename = load_latest_log()

    if logs and filename:
        path = render_html(logs, filename)
        if path:
            try:
                # 尝试自动打开浏览器，失败则提示
                webbrowser.open('file://' + os.path.abspath(path))
            except:
                print("Could not open browser automatically. Please open the file manually.")
    else:
        print("❌ No valid logs loaded.")