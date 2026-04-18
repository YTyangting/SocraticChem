# 快速开始指南

## 1. 环境准备

```bash
# 安装依赖
pip install torch transformers datasets peft accelerate bitsandbytes

# 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "from transformers import __version__; print(f'Transformers版本: {__version__}')"
```

## 2. 数据预处理（第一步）

```bash
# 基本用法（处理所有文件）
python socratic_finetune_preprocessor.py \
  --data_dir finetune_data_dataset \
  --output_dir ./processed_data \
  --output_format qwen \
  --analyze

# 测试用法（只处理5个文件）
python socratic_finetune_preprocessor.py \
  --data_dir finetune_data_dataset \
  --output_dir ./test_output \
  --max_files 5 \
  --analyze
```

**输出文件**：
- `./processed_data/socratic_dialogue_qwen_train.jsonl` - 训练集
- `./processed_data/socratic_dialogue_qwen_val.jsonl` - 验证集

## 3. 模型微调（第二步）

### 选项A：使用LoRA微调（推荐）
```bash
python finetune_qwen_socratic.py \
  --train_data ./processed_data/socratic_dialogue_qwen_train.jsonl \
  --val_data ./processed_data/socratic_dialogue_qwen_val.jsonl \
  --output_dir ./socratic_model_lora \
  --model_name Qwen/Qwen-7B \
  --epochs 3 \
  --batch_size 4 \
  --use_lora
```

### 选项B：使用量化+LoRA（显存不足时）
```bash
python finetune_qwen_socratic.py \
  --train_data ./processed_data/socratic_dialogue_qwen_train.jsonl \
  --val_data ./processed_data/socratic_dialogue_qwen_val.jsonl \
  --output_dir ./socratic_model_quant \
  --model_name Qwen/Qwen-7B \
  --epochs 3 \
  --batch_size 2 \
  --use_lora \
  --quantization
```

## 4. 测试模型（第三步）

```python
# test_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载微调后的模型
model_path = "./socratic_model_lora/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 测试提示
test_cases = [
    "学生说: 老师，这个实验真的有必要吗？我觉得直接告诉我结果都行，非得做一遍？",
    "学生说: 我已经把稀氯化钠溶液加到烧杯里了，接下来要做什么？",
    "学生说: 我不明白为什么要用量筒，直接倒不行吗？",
    "学生说: 我看到溶液变红了，这是正常的吗？",
    "学生说: 我忘了加酚酞指示剂，现在补加还来得及吗？"
]

for prompt in test_cases:
    # 构建输入
    input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码响应
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取assistant的响应
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").strip()
    
    print(f"输入: {prompt}")
    print(f"输出: {response}")
    print("-" * 80)
```

运行测试：
```bash
python test_model.py
```

## 5. 一键运行

```bash
# 给脚本执行权限
chmod +x run_socratic_finetune.sh

# 运行完整流程
./run_socratic_finetune.sh
```

## 6. 自定义配置

### 调整训练参数
```python
# 在finetune_qwen_socratic.py中修改TrainingArguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,           # 增加训练轮数
    per_device_train_batch_size=2, # 减小批次大小
    learning_rate=2e-4,           # 调整学习率
    warmup_steps=200,             # 增加warmup步数
    weight_decay=0.01,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    ...
)
```

### 调整LoRA配置
```python
# 在finetune_qwen_socratic.py中修改LoraConfig
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                         # 增加秩
    lora_alpha=64,                # 调整alpha
    lora_dropout=0.05,            # 减小dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="lora_only"
)
```

## 7. 常见问题

### Q1: 显存不足怎么办？
**A**: 使用量化选项：
```bash
--quantization --batch_size 2
```

### Q2: 训练时间太长怎么办？
**A**: 
1. 减少训练轮数：`--epochs 2`
2. 使用更多GPU：设置`CUDA_VISIBLE_DEVICES=0,1`
3. 减少数据量：预处理时使用`--max_files 100`

### Q3: 模型生成质量不高怎么办？
**A**:
1. 增加训练数据量
2. 调整生成参数：`temperature=0.3, top_p=0.95`
3. 增加训练轮数

### Q4: 如何评估模型效果？
**A**:
1. 查看训练日志中的eval_loss
2. 人工评估生成的问题质量
3. 设计测试集进行自动评估

## 8. 高级用法

### 多GPU训练
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python finetune_qwen_socratic.py ... --batch_size 8
```

### 继续训练
```python
# 加载已有模型继续训练
model = AutoModelForCausalLM.from_pretrained(
    "./socratic_model_lora/final_model",
    trust_remote_code=True
)
```

### 导出为安全格式
```python
# 合并LoRA权重到基础模型
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B")
model = PeftModel.from_pretrained(base_model, "./socratic_model_lora/final_model")
model = model.merge_and_unload()
model.save_pretrained("./socratic_model_merged")
```

## 9. 监控训练

### 查看训练日志
```bash
# 查看损失曲线
tensorboard --logdir ./socratic_model_lora/logs

# 查看保存的检查点
ls ./socratic_model_lora/checkpoint-*
```

### 实时监控
```python
# 添加回调函数
from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: {logs}")

# 在Trainer中添加
trainer = Trainer(
    ...,
    callbacks=[LoggingCallback()]
)
```

## 10. 生产部署

### 创建API服务
```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

class Request(BaseModel):
    student_input: str
    context: str = ""

# 加载模型
model = AutoModelForCausalLM.from_pretrained("./socratic_model_lora/final_model")
tokenizer = AutoTokenizer.from_pretrained("./socratic_model_lora/final_model")

@app.post("/generate")
async def generate_question(request: Request):
    input_text = f"<|im_start|>user\n{request.context}\n学生说: {request.student_input}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"question": response}
```

运行API：
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 总结

按照以下步骤操作：
1. **准备环境**：安装依赖
2. **预处理数据**：提取和格式化对话数据
3. **微调模型**：使用LoRA训练Qwen-7B
4. **测试模型**：验证生成质量
5. **部署应用**：创建API服务

如有问题，查看日志文件或调整参数重新训练。