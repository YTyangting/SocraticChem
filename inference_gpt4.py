import json
import os
import time
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
from config import Config

# ================= 配置区域 =================
API_KEY = Config.OPENAI_API_KEY
BASE_URL = Config.OPENAI_BASE_URL

# 推荐使用 GPT-4o，因为新数据包含大量 Context (XDL, DAG)，需要长窗口和强逻辑
MODEL_NAME = "gpt-4o" 

# 文件路径 (根据您的 vLLM 示例更新)
TEST_FILE = "/home/yjh/socChem_final/distribution_shift.json"
OUTPUT_FILE = "new_test/predictions_gpt4o_distribution_shift.jsonl" 

MAX_WORKERS = 8 # 根据您的 API Rate Limit 调整
# ===========================================

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# === System Prompt: 强制规定输出格式 ===
# 注意：原本的 Strategy Menu 不再适用，因为新数据的 instruction 字段里已经包含了具体的推理规则。
# 这里我们只需要设定“人设”和“格式约束”。
# ================= Few-Shot 示例区域 =================
# 这里提供一个完美的“输入 -> 输出”样本，教模型怎么写 XML
FEW_SHOT_EXAMPLE = """
### Instruction:
你是一个具备全知视角的苏格拉底式化学实验导师。请基于提供的实验设计(XDL)、任务流程(DAG)、操作前环境(BeforeEnvironment)及系统反馈(Observation)，严格按照以下逻辑链条输出 XML 格式的思维过程与回复：
1. <FindcurrentNode>: 分析操作意图与实验进度...
2. <vertify>: 根据锁定的节点推导验证标准...
3. <Thought>: 诊断学生的认知误区，制定对应的教学策略...
4. <Response>: 将策略转化为自然的教学语言回复...

### Input:
<Input>
  <Profile><Name>测试学生</Name></Profile>
  <DAG>...</DAG>
  <History>
    [{"role": "system", "content": "实验开始"}]
  </History>
  <Student><Speak>"老师好，今天做什么实验？"</Speak></Student>
</Input>

### Response:
<FindcurrentNode>
  <LogicTrace>学生正在发起对话，实验尚未开始物理操作。</LogicTrace>
  <StatusCheck>
    <Completed>[]</Completed>
    <UnfinishedCandidates>["node_start"]</UnfinishedCandidates>
  </StatusCheck>
  <Decision>
    <CurrentNode>node_start</CurrentNode>
    <State>IDLE</State>
  </Decision>
</FindcurrentNode>

<vertify>
  <CheckList>
    - 意图识别: GREETING [PASS]
  </CheckList>
  <Result> ✅ </Result>
</vertify>

<Thought>
  <CognitiveDiagnosis>
    学生态度积极，需要建立良好的师生关系并引入实验主题。
  </CognitiveDiagnosis>
  <Strategy>RAPPORT_BUILDING</Strategy>
  <Trace>
    检测到开场白，策略选择建立融洽关系，并介绍实验目标。
  </Trace>
  <Instruction>热情回应，并简述实验目标。</Instruction>
</Thought>

<Response>
你好呀！今天我们要进行非常有趣的离子反应实验。准备好探索化学的奥秘了吗？
</Response>
"""
# ====================================================

SYSTEM_PROMPT = f"""
You are an omniscient Socratic Chemistry Tutor.
Your task is to analyze the student's behavior based on the provided Experiment Design (XDL), Task Graph (DAG), and Environment.

You MUST strictly follow the XML output format below without any markdown code blocks (```xml ... ```):

<FindcurrentNode>
  <LogicTrace>Analyze operation intent and experiment progress...</LogicTrace>
  <StatusCheck>...</StatusCheck>
  <Decision>
    <CurrentNode>...</CurrentNode>
    <State>...</State>
  </Decision>
</FindcurrentNode>

<vertify>
  <CheckList>...</CheckList>
  <Result>...</Result>
</vertify>

<Thought>
  <CognitiveDiagnosis>...</CognitiveDiagnosis>
  <Strategy>...</Strategy>
  <Trace>...</Trace>
  <Instruction>...</Instruction>
</Thought>

<Response>
Your final dialogue to the student.
</Response>

##Example:
{FEW_SHOT_EXAMPLE}

Ensure all XML tags are correctly closed. Do not output any text outside these XML tags.
"""

def load_data(file_path):
    print(f"📖 Loading data from: {file_path}...")
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                data_list = json.loads(content)
            else:
                for line in content.splitlines():
                    if line.strip(): data_list.append(json.loads(line))
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []
    print(f"✅ Loaded {len(data_list)} samples.")
    return data_list

def process_one(item):
    """
    处理单条数据
    """
    # 获取数据中的指令和输入
    instruction = item.get("instruction", "")
    input_context = item.get("input", "")

    # 构建 User Content
    # 我们把具体的任务指令 (Instruction) 和 庞大的上下文 (Input) 组合在一起
    user_content = f"""
    {instruction}

    {input_context}
    """

    generated_text = ""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1, # 降低温度以保证 XML 结构稳定
            max_tokens=2048, # 增加 token 上限，因为 XML 思维链很长
            top_p=0.95
        )
        generated_text = response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        generated_text = "<Error>API Failure</Error>"

    # 简单清洗：有时候模型会即使在 System Prompt 禁止后依然输出 ```xml
    generated_text = generated_text.replace("```xml", "").replace("```", "").strip()

    return {
        "instruction": instruction,
        "input": input_context,
        "ground_truth": item.get("output", ""),
        "generated_output": generated_text
    }

def main():
    # 1. 加载数据
    raw_data = load_data(TEST_FILE)
    if not raw_data: return

    # 2. 检查断点
    start_index = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start_index = len(lines)
            print(f"⏩ Found existing file, skipping {start_index} lines...")

    tasks_to_run = raw_data[start_index:]
    if not tasks_to_run:
        print("🎉 All tasks completed!")
        return

    print(f"🚀 Starting concurrent inference (Workers: {MAX_WORKERS})...")
    
    # 3. 多线程执行
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {executor.submit(process_one, item): item for item in tasks_to_run}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(tasks_to_run)):
                try:
                    result = future.result()
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                except Exception as exc:
                    print(f"❌ Task generated an exception: {exc}")

    print(f"🎉 Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()