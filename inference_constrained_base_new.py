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
        "output_file": "new_test/predictions_qwen-v-finetune_noxml_t2.jsonl"
    },
    
    {
        "model_path": "/home/yjh/SocraticLM",  # 替换为您第二个模型的路径
        "output_file": "new_test/predictions_SocraticLM_noxml_t2.jsonl"
    },
    {
        "model_path": "/home/yjh/MathDial-SFT-Qwen2.5-1.5B-Instruct",  # 替换为您第二个模型的路径
        "output_file": "new_test/predictions_MathDial_noxml_t2.jsonl"
    },
    
    # 在这里继续添加更多模型...
]
# ===========================================

# def build_prompt(instruction, input_text):
#     # 改进的prompt构建，添加更明确的格式指导
#     return (
#         f"### Instruction:\n{instruction}\n\n"
#         f"请直接给出苏格拉底式教学回复，不要输出XML格式。\n"
#         f"回复要求：\n"
#         f"1. 通过引导性提问帮助学生自己发现答案\n"
#         f"2. 必要时指出安全注意事项\n"
#         f"3. 语气温和，鼓励学生思考\n"
#         f"4. 回复应当简洁、自然，避免重复\n"
#         f"5. 回复结束后不要添加额外内容\n\n"
#         f"### Input:\n{str(input_text)}\n\n"
#         f"### Response:\n"
#     )


def build_prompt(instruction, input_text):
    """
    构建 Prompt：System Instruction + Few-Shot Example + Real Task
    """
    # 1. 拼接 Few-Shot 示例（作为教案）
    # 注意：这里我们让模型先看一遍例子
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{str(input_text)}\n\n"
        f"### Response:\n" # <--- 强制引导
    )
    return prompt

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

# [核心修改] 将单个模型的推理过程封装成一个独立的函数，用于子进程运行
def run_model_inference(model_path, output_file, prompts, raw_data, tp_size):
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
        
        sampling_params = SamplingParams(
            temperature=0.7,          # 提高温度，增加多样性
            top_p=0.9,               # 稍微降低top_p
            repetition_penalty=1.2,   # 添加重复惩罚，减少重复
            frequency_penalty=0.1,    # 频率惩罚，减少常见词重复
            presence_penalty=0.1,     # 存在惩罚，鼓励多样性
            max_tokens=512,          # 减少最大token数，避免过长回复
            stop=["\n\n", "###", "Instruction:", "Input:"]  # 添加停止token
        )

        print(f"⚡ Inferencing on {len(prompts)} samples...")
        outputs = llm.generate(prompts, sampling_params)

        print(f"💾 Saving to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for original_item, output in zip(raw_data, outputs):
                generated_text = output.outputs[0].text.strip()
                result_item = {
                    "instruction": original_item.get("instruction", ""),
                    "input": original_item.get("input", ""),
                    "generated_output": generated_text, 
                    "ground_truth": original_item.get("output", "")
                }
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                
        print(f"✅ Model {model_path} finished successfully.")
        
    except Exception as e:
        print(f"❌ Error during inference for {model_path}: {e}")
    # 子进程执行到这里结束，OS 会自动接管释放所有的显存和 worker 进程。

def main():
    # [关键配置] 必须设置为 'spawn' 模式，这是 PyTorch 和 vLLM 多进程调用 CUDA 的硬性要求
    mp.set_start_method('spawn', force=True)
    
    # 1. 在主进程加载一次数据即可
    raw_data = load_data(TEST_FILE)
    if not raw_data: return
    prompts = [build_prompt(d.get("instruction", ""), d.get("input", "")) for d in raw_data]

    # 2. 遍历模型配置，依次启动子进程
    for config in MODELS_CONFIG:
        model_path = config["model_path"]
        output_file = config["output_file"]
        
        # 创建子进程
        p = mp.Process(
            target=run_model_inference, 
            args=(model_path, output_file, prompts, raw_data, TENSOR_PARALLEL_SIZE)
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