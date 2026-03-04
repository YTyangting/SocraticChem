import json
import os
from vllm import LLM, SamplingParams

# ================= 配置区域 =================
MODEL_PATH = "/home/yjh/soclm_v3"  # 模型路径
TEST_FILE = "/home/yjh/socChemlab/sft_finetune_chemlab_test.json"  # 测试文件
OUTPUT_FILE = "new_test/predictions_chem_lab_xml.jsonl"  # 输出文件

TENSOR_PARALLEL_SIZE = 4  # 显卡数量

# [关键策略更新] 
# 根据新的数据分布，Output 总是以 <FindcurrentNode> 开头。
# 我们强制模型从这里开始写，避免它输出无关的寒暄。
FORCE_PREFIX = "<FindcurrentNode>\n"
# ===========================================

def build_prompt(instruction, input_text):
    """
    针对长文本 XML 推理任务构建 Prompt。
    格式采用标准的 Instruction / Input / Response 结构。
    """
    # 确保 input_text 是字符串，防止数据中有非预期类型
    input_str = str(input_text)
    
    # 构建 Prompt
    # 这种格式对于遵循复杂指令（如您的苏格拉底导师）通常效果较好
    prompt = (
        f"Below is an instruction that describes a task, paired with an input that provides further context. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_str}\n\n"
        f"### Response:\n{FORCE_PREFIX}"  # 强制以首个 XML 标签开头
    )
    return prompt

def load_test_data(file_path):
    print(f"📖 正在读取测试数据: {file_path}...")
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 尝试作为整个 JSON 数组加载
            if content.startswith('['):
                data_list = json.loads(content)
            else:
                # 尝试作为 JSONL (每行一个 JSON) 加载
                lines = content.splitlines()
                for line in lines:
                    if line.strip():
                        data_list.append(json.loads(line))
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return []
        
    print(f"✅ 成功加载 {len(data_list)} 条测试样本。")
    return data_list

def main():
    # 1. 加载数据
    raw_data = load_test_data(TEST_FILE)
    if not raw_data: 
        print("数据为空，程序退出。")
        return

    # 2. 构建 Prompts
    print("🔨 正在构建 Prompts...")
    prompts = [build_prompt(d.get("instruction", ""), d.get("input", "")) for d in raw_data]

    # 3. 初始化 vLLM
    print(f"🚀 初始化 vLLM (Model: {MODEL_PATH})...")
    # 注意：如果您的模型显存占用较高，可以适当调低 gpu_memory_utilization
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        gpu_memory_utilization=0.9, # 稍微调高一点利用率，因为 Context 很长
        max_model_len=4096,         # [重要] XML 上下文通常很长，建议增加到 4096 或 8192
        max_num_seqs=16,
    )

    # 4. 采样参数设置
    sampling_params = SamplingParams(
        temperature=0.1,       # 降低温度，因为 XML 结构需要高度确定性
        top_p=0.95,
        max_tokens=2048,       # 增加生成长度，因为 Output 包含完整的思维链
        stop=["</Response>"]   # 遇到 Response 闭合标签即停止
    )

    # 5. 执行推理
    print(f"⚡ 开始批量推理 ({len(prompts)} 条样本)...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. 保存结果
    print(f"💾 保存结果到: {OUTPUT_FILE}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for original_item, output in zip(raw_data, outputs):
            # 获取生成内容
            generated_suffix = output.outputs[0].text
            
            # [关键] 拼接前缀和生成内容
            full_output = FORCE_PREFIX + generated_suffix
            
            # 鲁棒性处理：如果模型因为 max_tokens 截断没写完闭合标签，我们手动补全（可选）
            if "</Response>" not in full_output and not full_output.strip().endswith(">"):
                full_output += "\n</Response>"
            elif full_output.strip().endswith("</Response>"):
                pass # 完美结束
            else:
                # 如果没有闭合，可能需要补一个
                full_output += "</Response>"

            result_item = {
                "instruction": original_item.get("instruction", ""),
                "input": original_item.get("input", ""),
                "generated_output": full_output, 
                "ground_truth": original_item.get("output", "")
            }
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    print("🎉 推理完成！")

if __name__ == "__main__":
    main()