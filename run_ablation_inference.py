import json
import os
import re
from vllm import LLM, SamplingParams

# ================= 配置区域 =================
MODEL_PATH = "/home/yjh/soclm_v5"      # 您的微调模型路径
TEST_FILE = "/home/yjh/socChemlab/chemlab_safety_eval_set.json"  # 您的测试集路径
TENSOR_PARALLEL_SIZE = 4                # 显卡数量 (根据您的硬件调整)

# 强制前缀 (保持一致性，引导模型输出)
FORCE_PREFIX = "<FindcurrentNode>\n" 
# ===========================================

def remove_xml_block(text, tag):
    """
    鲁棒的 XML 屏蔽函数：将 <tag>...</tag> 及其内容完全移除
    """
    if not isinstance(text, str): return text
    # 模式匹配：<Tag ...>...内容...</Tag> (DOTALL模式，匹配换行符)
    # 支持带属性的标签，如 <DAG id="1">
    pattern = f"<{tag}[^>]*>.*?</{tag}>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    # 清理多余的空行，保持 Prompt 整洁
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text).strip()
    return cleaned_text

def build_prompt(instruction, input_text):
    """构建 Alpaca 风格 Prompt"""
    return (
        f"Below is an instruction that describes a task, paired with an input that provides further context. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n{FORCE_PREFIX}" 
    )

def load_data(file_path):
    print(f"📖 Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 兼容 JSON 列表和 JSONL
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                return json.load(f)
            else:
                return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return []

def run_ablation_batch(llm, raw_data, mode):
    print(f"\n🔥 Running Ablation Mode: [{mode}]")
    
    prompts = []
    valid_indices = [] # 记录有效数据的索引，防止数据加载失败导致错位

    for idx, item in enumerate(raw_data):
        original_input = item.get("input", "")
        instruction = item.get("instruction", "")
        
        # --- 核心消融逻辑 (Masking) ---
        masked_input = original_input
        
        if mode == "full_context":
            pass # 完整上下文 (基准对照组)
            
        elif mode == "no_obs":
            # 策略 A: 移除观察 (测试状态判定能力)
            masked_input = remove_xml_block(original_input, "Observation")
            
        elif mode == "no_logic":
            # 策略 B: 移除逻辑图 (测试流程锁定能力)
            # 同时移除 DAG 和 XDL
            masked_input = remove_xml_block(original_input, "DAG")
            masked_input = remove_xml_block(masked_input, "XDL")
            
        elif mode == "no_env":
            # 策略 C: 移除环境数据 (测试危险拦截能力)
            masked_input = remove_xml_block(original_input, "BeforeEnvironment")
            
        elif mode == "no_profile":
            # 策略 D: 移除画像 (测试个性化教学)
            masked_input = remove_xml_block(original_input, "Profile")
            
        elif mode == "no_history":
            # 策略 E: 移除历史 (测试多轮对话能力)
            masked_input = remove_xml_block(original_input, "History")

        prompts.append(build_prompt(instruction, masked_input))
        valid_indices.append(idx)

    # 2. 批量推理
    sampling_params = SamplingParams(
        temperature=0.1,    # 消融实验建议低温，减少随机性干扰
        top_p=0.9,
        max_tokens=2048,
        stop=["</Response>"] 
    )
    
    print(f"⚡ Generating {len(prompts)} responses for mode [{mode}]...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 3. 保存结果
    output_filename = f"/home/yjh/socChemlab/new_test/predictions_ablation_{mode}_safety.jsonl"
    print(f"💾 Saving to {output_filename}...")
    
    with open(output_filename, 'w', encoding='utf-8') as f_out:
        for i, output in enumerate(outputs):
            original_idx = valid_indices[i]
            original_item = raw_data[original_idx]
            
            # 补全 Response 结构
            generated_suffix = output.outputs[0].text
            full_output = FORCE_PREFIX + generated_suffix
            if not full_output.strip().endswith("</Response>"):
                full_output += "\n</Response>"
            
            # 构造结果对象
            result_item = {
                "instruction": original_item.get("instruction", ""),
                
                # [关键] 这里保存的是**原始未消融**的 input
                # 评估代码(Parser)会读取这里的 DAG/GT 来做对比
                "input": original_item.get("input", ""), 
                
                # 这里保存模型的预测结果
                "generated_output": full_output,
                
                # 这里保存 Ground Truth
                "ground_truth": original_item.get("output", ""),
                
                # 记录消融模式，方便后续分析
                "meta_ablation_mode": mode
            }
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            
    print(f"✅ Mode [{mode}] Finished. Saved {len(outputs)} entries.")

def main():
    # 1. 加载模型
    print(f"🚀 Initializing vLLM (Model: {MODEL_PATH})...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        gpu_memory_utilization=0.8, # 稍微调高一点，防止 OOM
        max_model_len=4096 # 根据你的显存调整
    )
    
    # 2. 加载原始数据
    raw_data = load_data(TEST_FILE)
    if not raw_data: return
    print(f"📊 Loaded {len(raw_data)} samples.")

    # 3. 依次运行所有消融模式
    # 建议顺序：先跑 Full 做基准，再跑其他的
    ablation_modes = [
        "full_context",  # 基准 (Upper Bound)
        "no_logic",      # 验证 DAG 的作用
        "no_obs",        # 验证 Observation 的作用
        "no_env",        # 验证 BeforeEnvironment 的作用
        "no_profile",     # 验证 Profile 的作用
        'no_history'
    ]
    
    for mode in ablation_modes:
        try:
            run_ablation_batch(llm, raw_data, mode=mode)
        except Exception as e:
            print(f"⚠️ Error running mode {mode}: {e}")
            continue
    
    print("\n🎉 All Ablation Studies Completed!")
    print("Next Step: Run 'eval_sochem_v25_production.py' on the generated .jsonl files.")

if __name__ == "__main__":
    main()