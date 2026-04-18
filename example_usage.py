#!/usr/bin/env python3
"""
苏格拉底式教学数据处理使用示例
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_data_extraction():
    """数据提取示例"""
    print("=== 数据提取示例 ===")
    
    from socratic_finetune_preprocessor import SocraticDialogueProcessor
    
    # 初始化处理器
    data_dir = "/home/yjh/socChem_final/finetune_data_dataset"
    processor = SocraticDialogueProcessor(data_dir)
    
    # 处理少量文件进行测试
    print("处理前5个文件...")
    dialogues = processor.process_all_files(max_files=5)
    
    if dialogues:
        print(f"提取了 {len(dialogues)} 个对话轮次")
        
        # 显示示例对话
        print("\n=== 对话示例 ===")
        for i, dialogue in enumerate(dialogues[:3]):
            print(f"\n对话 {i+1}:")
            print(f"学生: {dialogue.student_speak}")
            if dialogue.student_actions:
                print(f"操作: {dialogue.student_actions}")
            print(f"教师: {dialogue.teacher_response}")
            print(f"策略: {dialogue.strategy_type}")
            print("-" * 50)
        
        # 分析数据分布
        processor.analyze_data_distribution(dialogues)
        
        # 格式化数据
        print("\n=== 格式化数据 ===")
        formatted_data = processor.format_for_finetuning(dialogues, output_format="qwen")
        
        # 保存数据
        output_dir = "./output_example"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "socratic_dialogue_example.jsonl")
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in formatted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"数据保存到: {output_path}")
        
        return formatted_data
    else:
        print("没有提取到数据")
        return None

def example_finetune_preparation():
    """微调准备示例"""
    print("\n=== 微调准备示例 ===")
    
    from socratic_finetune_preprocessor import main as preprocess_main
    import argparse
    
    # 模拟命令行参数
    class Args:
        def __init__(self):
            self.data_dir = "/home/yjh/socChem_final/finetune_data_dataset"
            self.output_dir = "./output_full"
            self.output_format = "qwen"
            self.max_files = 10  # 限制文件数用于测试
            self.split_ratio = 0.8
            self.analyze = True
    
    args = Args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行预处理
    print("运行完整预处理流程...")
    
    # 由于main函数需要命令行参数，我们直接调用处理器
    from socratic_finetune_preprocessor import SocraticDialogueProcessor
    
    processor = SocraticDialogueProcessor(args.data_dir)
    dialogues = processor.process_all_files(max_files=args.max_files)
    
    if dialogues:
        # 分析数据
        if args.analyze:
            processor.analyze_data_distribution(dialogues)
        
        # 格式化数据
        formatted_data = processor.format_for_finetuning(dialogues, args.output_format)
        
        # 保存数据
        output_path = os.path.join(args.output_dir, f"socratic_dialogue_{args.output_format}.jsonl")
        train_path, val_path = processor.save_data(formatted_data, output_path, args.split_ratio)
        
        print(f"\n训练数据: {train_path}")
        print(f"验证数据: {val_path}")
        
        return train_path, val_path
    else:
        print("没有提取到数据")
        return None, None

def example_inference():
    """推理示例（使用微调后的模型）"""
    print("\n=== 推理示例 ===")
    
    # 注意：这需要先运行微调训练
    print("""
    要使用微调后的模型进行推理，需要：
    1. 先运行数据预处理提取数据
    2. 运行微调训练得到模型
    3. 使用以下代码加载模型进行推理
    
    示例代码：
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    # 加载微调后的模型
    model_path = "./socratic_qwen_model/final_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 准备输入
    prompt = "学生说: 老师，这个实验真的有必要吗？我觉得直接告诉我结果都行，非得做一遍？"
    input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"模型响应: {response}")
    """)

def create_bash_script():
    """创建bash脚本示例"""
    print("\n=== Bash脚本示例 ===")
    
    script_content = """#!/bin/bash

# 苏格拉底式教学数据处理和微调脚本

echo "=== 步骤1: 数据预处理 ==="
python socratic_finetune_preprocessor.py \
    --data_dir /home/yjh/socChem_final/finetune_data_dataset \
    --output_dir ./processed_data \
    --output_format qwen \
    --split_ratio 0.8 \
    --analyze

echo "=== 步骤2: 微调训练 ==="
python finetune_qwen_socratic.py \
    --train_data ./processed_data/socratic_dialogue_qwen_train.jsonl \
    --val_data ./processed_data/socratic_dialogue_qwen_val.jsonl \
    --output_dir ./socratic_qwen_model \
    --model_name Qwen/Qwen-7B \
    --format qwen \
    --epochs 3 \
    --batch_size 4 \
    --use_lora \
    --quantization

echo "=== 步骤3: 测试生成 ==="
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型
model_path = './socratic_qwen_model/final_model'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto'
)

# 测试提示
test_prompts = [
    '学生说: 老师，这个实验真的有必要吗？我觉得直接告诉我结果都行，非得做一遍？',
    '学生说: 我已经把稀氯化钠溶液加到烧杯里了，接下来要做什么？',
    '学生说: 我不明白为什么要用量筒，直接倒不行吗？'
]

for prompt in test_prompts:
    input_text = f'<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n'
    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'提示: {prompt[:50]}...')
    print(f'响应: {response}\\n')
"
"""
    
    script_path = "./run_socratic_finetune.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"创建脚本: {script_path}")
    print("运行命令: ./run_socratic_finetune.sh")

def main():
    """主函数"""
    print("苏格拉底式教学数据处理和微调示例")
    print("=" * 60)
    
    # 运行示例
    example_data_extraction()
    
    # 如果需要运行完整流程，取消注释下面的行
    # train_path, val_path = example_finetune_preparation()
    
    example_inference()
    create_bash_script()
    
    print("\n=== 使用说明 ===")
    print("1. 数据提取: python socratic_finetune_preprocessor.py --data_dir /path/to/data --output_dir ./output")
    print("2. 微调训练: python finetune_qwen_socratic.py --train_data ./output/xxx_train.jsonl --val_data ./output/xxx_val.jsonl")
    print("3. 或运行脚本: ./run_socratic_finetune.sh")

if __name__ == "__main__":
    main()