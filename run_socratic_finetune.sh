#!/bin/bash

# 苏格拉底式教学数据处理和微调脚本

echo "=== 步骤1: 数据预处理 ==="
python socratic_finetune_preprocessor.py     --data_dir /home/yjh/socChem_final/finetune_data_dataset     --output_dir ./processed_data     --output_format qwen     --split_ratio 0.8     --analyze

echo "=== 步骤2: 微调训练 ==="
python finetune_qwen_socratic.py     --train_data ./processed_data/socratic_dialogue_qwen_train.jsonl     --val_data ./processed_data/socratic_dialogue_qwen_val.jsonl     --output_dir ./socratic_qwen_model     --model_name Qwen/Qwen-7B     --format qwen     --epochs 3     --batch_size 4     --use_lora     --quantization

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
    input_text = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
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
    print(f'响应: {response}\n')
"
