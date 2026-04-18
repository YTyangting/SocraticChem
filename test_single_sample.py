#!/usr/bin/env python3
"""
测试单个样本的推理，检查文本循环问题是否解决
"""

import json
from vllm import LLM, SamplingParams

def test_single_sample():
    # 加载一个测试样本
    test_file = "/home/yjh/socChemlab/sft_finetune_chemlab_test_1.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 取第一个样本
    sample = data[0]
    instruction = sample["instruction"]
    input_text = sample["input"]
    
    # 构建prompt（使用改进的版本）
    prompt = f"""### Instruction:
{instruction}

请直接给出苏格拉底式教学回复，不要输出XML格式。
回复要求：
1. 通过引导性提问帮助学生自己发现答案
2. 必要时指出安全注意事项
3. 语气温和，鼓励学生思考
4. 回复应当简洁、自然，避免重复
5. 回复结束后不要添加额外内容

### Input:
{input_text}

### Response:
"""
    
    # 加载模型
    print("🚀 加载模型...")
    llm = LLM(
        model="/home/yjh/qwen-v-finetune",
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
    )
    
    # 测试不同的采样参数
    test_configs = [
        {
            "name": "原始参数 (temperature=0.1)",
            "params": SamplingParams(temperature=0.1, top_p=0.95, max_tokens=512)
        },
        {
            "name": "改进参数 (temperature=0.7, repetition_penalty=1.2)",
            "params": SamplingParams(
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                max_tokens=512,
                stop=["\n\n", "###", "Instruction:", "Input:", "Response:"]
            )
        }
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"测试配置: {config['name']}")
        print(f"{'='*60}")
        
        outputs = llm.generate([prompt], config['params'])
        generated_text = outputs[0].outputs[0].text.strip()
        
        print(f"生成的回复:\n{generated_text}")
        print(f"\n回复长度: {len(generated_text)} 字符")
        
        # 检查是否有重复
        lines = generated_text.split('\n')
        unique_lines = set(lines)
        repetition_ratio = 1 - (len(unique_lines) / max(len(lines), 1))
        
        print(f"重复率: {repetition_ratio:.2%}")
        if repetition_ratio > 0.3:
            print("⚠️  警告: 检测到高重复率!")
        
        # 检查是否有明显的循环模式
        if "你已经很认真地完成了前面的实验步骤" in generated_text:
            count = generated_text.count("你已经很认真地完成了前面的实验步骤")
            print(f"⚠️  检测到重复模式 '你已经很认真地完成了前面的实验步骤' {count} 次")
        
        if "如果遇到困难" in generated_text:
            count = generated_text.count("如果遇到困难")
            print(f"⚠️  检测到重复模式 '如果遇到困难' {count} 次")

if __name__ == "__main__":
    test_single_sample()