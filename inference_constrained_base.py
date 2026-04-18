import json
import os
from vllm import LLM, SamplingParams

# ================= 配置区域 =================
# [注意] 请确保 TENSOR_PARALLEL_SIZE 能整除模型词表大小 (如 4, 6, 8)
# 且小于等于实际显卡数量
TENSOR_PARALLEL_SIZE = 4            
MODEL_PATH = "/home/yjh/soclm_v5"   # 您的模型路径
TEST_FILE = "/home/yjh/socChem_final/distribution_shift.json" # 新数据路径
OUTPUT_FILE = "new_test/predictions_soclm_v5_distribution_shift.jsonl" 

# [关键修改] 新数据的逻辑链起始标签
FORCE_PREFIX = "<FindcurrentNode>\n" 
# ===========================================

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

def build_prompt(instruction, input_text):
    """
    构建 Prompt：System Instruction + Few-Shot Example + Real Task
    """
    # 1. 拼接 Few-Shot 示例（作为教案）
    # 注意：这里我们让模型先看一遍例子
    prompt = (
        f"{FEW_SHOT_EXAMPLE}\n\n"  # <--- 关键：先展示示例
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{str(input_text)}\n\n"
        f"### Response:\n{FORCE_PREFIX}" # <--- 强制引导
    )
    return prompt

def load_data(file_path):
    print(f"📖 Loading data from: {file_path}...")
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 兼容 JSON Array 和 JSONL 两种格式
            if content.startswith('['):
                data_list = json.loads(content)
            else:
                lines = content.splitlines()
                for line in lines:
                    if line.strip():
                        data_list.append(json.loads(line))
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []
        
    print(f"✅ Loaded {len(data_list)} samples.")
    return data_list

def main():
    # 1. 加载数据
    raw_data = load_data(TEST_FILE)
    if not raw_data: return

    # 2. 构建 Prompts
    # 过滤掉 input 为空的数据以防报错
    prompts = [build_prompt(d.get("instruction", ""), d.get("input", "")) for d in raw_data]

    # 3. 初始化 vLLM
    # 保持保守的显存配置，防止 OOM
    print(f"🚀 Initializing vLLM (TP={TENSOR_PARALLEL_SIZE})...")
    try:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,  # 稍微留点余量
            max_model_len=4096,          # [重要] XML 上下文通常较长，需保证窗口足够
            max_num_seqs=16,             # 降低并发数以节省显存
            enforce_eager=True           # 强制 Eager 模式，减少计算图显存占用
        )
    except Exception as e:
        print(f"❌ vLLM Init Failed: {e}")
        print("💡 建议检查显卡数量是否匹配 TENSOR_PARALLEL_SIZE")
        return

    # 4. 采样参数
    sampling_params = SamplingParams(
        temperature=0.1,        # 低温，保证 XML 格式严格正确
        top_p=0.95,
        max_tokens=2048,        # [重要] 完整的思维链输出很长，增加 max_tokens
        stop=["</Response>"]    # 遇到 Response 闭合标签即停止
    )

    # 5. 执行推理
    print(f"⚡ Starting Inference on {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. 保存结果
    print(f"💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for original_item, output in zip(raw_data, outputs):
            # 获取生成内容
            generated_suffix = output.outputs[0].text
            
            # [拼接] 强制前缀 + 生成内容
            full_output = FORCE_PREFIX + generated_suffix
            
            # [补全] 简单的鲁棒性处理：如果被截断，手动闭合
            if not full_output.strip().endswith("</Response>"):
                # 检查是否以 > 结尾，防止截断在标签中间
                if full_output.strip().endswith(">"):
                    full_output += "\n</Response>"
            
            result_item = {
                "instruction": original_item.get("instruction", ""),
                "input": original_item.get("input", ""),
                "generated_output": full_output, 
                "ground_truth": original_item.get("output", "")
            }
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    print("🎉 Done!")

if __name__ == "__main__":
    main()