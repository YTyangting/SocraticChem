import json
import os
import gc
import torch
from vllm import LLM, SamplingParams
# 导入 vllm 的分布式销毁函数，确保释放显存不残留
from vllm.distributed.parallel_state import destroy_model_parallel
import multiprocessing as mp

# ================= 配置区域 =================
TENSOR_PARALLEL_SIZE = 4            
TEST_FILE = "/home/yjh/socChemlab/sft_finetune_chemlab_test_1.json"  # 测试文件

# [核心修改] 将模型配置改为列表，您可以添加任意多个模型
MODELS_CONFIG = [
    {
        "model_path": "/home/yjh/qwen-v-finetune",  # 替换为您第二个模型的路径
        "output_file": "new_test/predictions_qwen-v-finetune_noxml_t3_fewshot.jsonl"
    },
    
    # {
    #     "model_path": "/home/yjh/SocraticLM",  # 替换为您第二个模型的路径
    #     "output_file": "new_test/predictions_SocraticLM_noxml_t2_fewshot.jsonl"
    # },
    # {
    #     "model_path": "/home/yjh/MathDial-SFT-Qwen2.5-1.5B-Instruct",  # 替换为您第二个模型的路径
    #     "output_file": "new_test/predictions_MathDial_noxml_t2_fewshot.jsonl"
    # },
    
    # 在这里继续添加更多模型...
]
# ===========================================

# Few-shot 示例
FEW_SHOT_EXAMPLES = [
    {
        "instruction": "你是一个具备全知视角的苏格拉底式化学实验导师。请基于提供的实验设计(XDL)、任务流程(DAG)、操作前环境(BeforeEnvironment)及系统反馈(Observation)，给出苏格拉底式教学回复。",
        "input": "<Input>...</Input>",
        "output": "你的想法很有趣！不过在我们开始实验之前，你觉得应该先准备哪些器材呢？比如试管和滴管，你会怎么安排它们？"
    },
    {
        "instruction": "你是一个具备全知视角的苏格拉底式化学实验导师。请基于提供的实验设计(XDL)、任务流程(DAG)、操作前环境(BeforeEnvironment)及系统反馈(Observation)，给出苏格拉底式教学回复。",
        "input": "<Input>...</Input>",
        "output": "你已经把溶液加进试管了，做得不错！现在请你仔细观察试管里的现象，描述一下你看到了什么变化？"
    }
]

def build_prompt_with_fewshot(instruction, input_text):
    # 构建包含few-shot示例的prompt
    prompt_parts = []
    
    # 添加指令
    prompt_parts.append(f"### 角色说明:\n{instruction}\n")
    
    # 添加格式要求
    prompt_parts.append("### 回复要求:\n")
    prompt_parts.append("1. 直接给出苏格拉底式教学回复，不要输出XML格式")
    prompt_parts.append("2. 通过引导性提问帮助学生自己发现答案")
    prompt_parts.append("3. 必要时指出安全注意事项")
    prompt_parts.append("4. 语气温和，鼓励学生思考")
    prompt_parts.append("5. 回复应当简洁、自然，避免重复")
    prompt_parts.append("6. 回复结束后不要添加额外内容\n")
    
    # 添加few-shot示例
    prompt_parts.append("### 示例:")
    for i, example in enumerate(FEW_SHOT_EXAMPLES[:2], 1):  # 只使用前2个示例
        prompt_parts.append(f"\n示例 {i}:")
        prompt_parts.append(f"输入: {example['input'][:200]}...")  # 截取部分输入
        prompt_parts.append(f"回复: {example['output']}")
    
    prompt_parts.append(f"\n### 当前输入:\n{str(input_text)}\n")
    prompt_parts.append("### 回复:")
    
    return "\n".join(prompt_parts)

def build_prompt_simple(instruction, input_text):
    # 简单的prompt构建（备用）
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"请直接给出苏格拉底式教学回复，不要输出XML格式。\n"
        f"回复要求：\n"
        f"1. 通过引导性提问帮助学生自己发现答案\n"
        f"2. 必要时指出安全注意事项\n"
        f"3. 语气温和，鼓励学生思考\n"
        f"4. 回复应当简洁、自然，避免重复\n"
        f"5. 回复结束后不要添加额外内容\n\n"
        f"### Input:\n{str(input_text)}\n\n"
        f"### Response:\n"
    )

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
                    if line.strip():
                        data_list.append(json.loads(line))
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []
    print(f"✅ Loaded {len(data_list)} samples.")
    return data_list

# 后处理函数：移除重复内容
def remove_repetition(text, max_repeat=2):
    """简单的去重函数，移除过度重复的句子"""
    lines = text.split('\n')
    if len(lines) <= 1:
        return text
    
    # 检测重复模式
    seen_patterns = []
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            cleaned_lines.append(line)
            continue
            
        # 检查是否与最近几行相似
        is_repeat = False
        for pattern in seen_patterns[-3:]:  # 检查最近3个模式
            if line_stripped in pattern or pattern in line_stripped:
                if line_stripped.count(pattern) > 1 or pattern.count(line_stripped) > 1:
                    is_repeat = True
                    break
        
        if not is_repeat:
            cleaned_lines.append(line)
            seen_patterns.append(line_stripped)
    
    return '\n'.join(cleaned_lines)

# [核心修改] 将单个模型的推理过程封装成一个独立的函数，用于子进程运行
def run_model_inference(model_path, output_file, prompts, raw_data, tp_size, use_fewshot=True):
    # 注意：vLLM 和 torch 相关的导入必须放在子进程内部！
    # 这样可以防止 CUDA Context 在主进程被提前初始化
    import torch
    from vllm import LLM, SamplingParams
    
    print(f"\n" + "="*50)
    print(f"🚀 [Start Process] Initializing model: {model_path}")
    print("="*50)
    
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,  
            max_model_len=4096,          
            max_num_seqs=16,             
            enforce_eager=True           
        )
        
        # 优化的采样参数
        sampling_params = SamplingParams(
            temperature=0.7,           # 提高温度，增加多样性
            top_p=0.9,                # 稍微降低top_p
            repetition_penalty=1.2,    # 重复惩罚，减少重复
            frequency_penalty=0.1,     # 频率惩罚
            presence_penalty=0.1,      # 存在惩罚
            max_tokens=512,           # 限制最大长度
            stop=["\n\n", "###", "Instruction:", "Input:", "Response:", "回复:"]  # 停止token
        )

        print(f"⚡ Inferencing on {len(prompts)} samples...")
        outputs = llm.generate(prompts, sampling_params)

        print(f"💾 Saving to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for original_item, output in zip(raw_data, outputs):
                generated_text = output.outputs[0].text.strip()
                
                # 后处理：移除重复内容
                cleaned_text = remove_repetition(generated_text)
                
                result_item = {
                    "instruction": original_item.get("instruction", ""),
                    "input": original_item.get("input", ""),
                    "generated_output": cleaned_text,  # 使用清理后的文本
                    "ground_truth": original_item.get("output", "")
                }
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                
        print(f"✅ Model {model_path} finished successfully.")
        
    except Exception as e:
        print(f"❌ Error during inference for {model_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    # [关键配置] 必须设置为 'spawn' 模式，这是 PyTorch 和 vLLM 多进程调用 CUDA 的硬性要求
    mp.set_start_method('spawn', force=True)
    
    # 1. 在主进程加载一次数据即可
    raw_data = load_data(TEST_FILE)
    if not raw_data: return
    
    # 2. 构建prompts（可以选择使用few-shot或简单版本）
    use_fewshot = True  # 设置为True使用few-shot，False使用简单版本
    if use_fewshot:
        prompts = [build_prompt_with_fewshot(d.get("instruction", ""), d.get("input", "")) for d in raw_data]
        print(f"📝 Using few-shot prompts ({len(prompts)} samples)")
    else:
        prompts = [build_prompt_simple(d.get("instruction", ""), d.get("input", "")) for d in raw_data]
        print(f"📝 Using simple prompts ({len(prompts)} samples)")

    # 3. 遍历模型配置，依次启动子进程
    for config in MODELS_CONFIG:
        model_path = config["model_path"]
        output_file = config["output_file"]
        
        # 创建子进程
        p = mp.Process(
            target=run_model_inference, 
            args=(model_path, output_file, prompts, raw_data, TENSOR_PARALLEL_SIZE, use_fewshot)
        )
        
        p.start()  # 启动子进程
        p.join()   # 阻塞主进程，直到当前模型的子进程彻底结束退出
        
        # 检查子进程是否异常退出
        if p.exitcode != 0:
            print(f"⚠️ Warning: Process for {model_path} exited with code {p.exitcode}")
            
        print(f"🧹 [Cleanup] OS has reclaimed all GPU memory. Ready for next.")

    print("\n🎉 All models have been processed successfully!")

if __name__ == "__main__":
    main()