# 苏格拉底式教学对话数据微调

本项目提供了一套完整的工具链，用于处理苏格拉底式教学对话数据并微调Qwen-7B模型，使其能够基于学生输入（speak和action）生成合适的苏格拉底式问题。

## 项目结构

```
socChem_final/
├── finetune_data_dataset/          # 原始数据目录（包含大量JSONL文件）
├── socratic_finetune_preprocessor.py  # 数据预处理脚本
├── finetune_qwen_socratic.py       # Qwen-7B微调脚本
├── example_usage.py                # 使用示例
├── run_socratic_finetune.sh        # 一键运行脚本
└── README.md                       # 说明文档
```

## 数据格式

原始数据为JSONL格式，每个文件包含多个对话轮次，每行一个JSON对象：

```json
{
  "session_id": "xxx",
  "turn_index": 0,
  "data": {
    "meta": {...},
    "agents": {
      "student": {
        "input": {...},
        "output": {
          "speak": "学生说的话",
          "actions_ready_to_run": [...],
          "thought": "..."
        }
      },
      "teacher": {
        "output": {
          "response": "教师的苏格拉底式问题",
          "strategy": "教学策略"
        }
      }
    }
  }
}
```

## 功能模块

### 1. 数据预处理 (`socratic_finetune_preprocessor.py`)

**功能**：
- 从JSONL文件中提取对话轮次
- 格式化数据为微调格式（支持Qwen、Instruction、Simple格式）
- 分析数据分布（教学策略、对话长度等）
- 分割训练集和验证集

**使用方法**：
```bash
python socratic_finetune_preprocessor.py \
  --data_dir /path/to/finetune_data_dataset \
  --output_dir ./processed_data \
  --output_format qwen \
  --max_files 100 \
  --split_ratio 0.8 \
  --analyze
```

**参数说明**：
- `--data_dir`: 原始数据目录
- `--output_dir`: 输出目录
- `--output_format`: 输出格式（qwen/instruction/simple）
- `--max_files`: 最大处理文件数（用于测试）
- `--split_ratio`: 训练集分割比例
- `--analyze`: 分析数据分布

### 2. 模型微调 (`finetune_qwen_socratic.py`)

**功能**：
- 加载Qwen-7B模型
- 使用LoRA进行高效微调
- 支持4-bit量化（减少显存占用）
- 训练和评估模型

**使用方法**：
```bash
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
```

**参数说明**：
- `--train_data`: 训练数据路径
- `--val_data`: 验证数据路径
- `--output_dir`: 模型输出目录
- `--model_name`: 基础模型名称
- `--format`: 数据格式
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--use_lora`: 使用LoRA微调
- `--quantization`: 使用4-bit量化

### 3. 一键运行脚本 (`run_socratic_finetune.sh`)

**功能**：
- 自动化整个流程：数据预处理 → 模型微调 → 测试生成

**使用方法**：
```bash
chmod +x run_socratic_finetune.sh
./run_socratic_finetune.sh
```

## 微调数据格式

### Qwen格式（推荐）
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "当前任务: 量取稀氯化钠溶液并转移至烧杯...\n学生说: 老师，这个实验真的有必要吗？..."
    },
    {
      "role": "assistant",
      "content": "你的想法很有趣——如果直接告诉结果，确实省事。不过你觉得..."
    }
  ]
}
```

### Instruction格式
```json
{
  "instruction": "你是一位化学老师，正在进行苏格拉底式教学...",
  "input": "学生说: 老师，这个实验真的有必要吗？...",
  "output": "你的想法很有趣——如果直接告诉结果，确实省事..."
}
```

## 模型输出示例

**输入**（学生）：
```
老师，这个实验真的有必要吗？不就是用稀盐酸滴定稀氯化钠溶液，然后加点酚酞看看变不变色吗？我觉得直接告诉我结果都行，非得做一遍？
```

**输出**（模型生成的苏格拉底式问题）：
```
你的想法很有趣——如果直接告诉结果，确实省事。不过你觉得，如果我们想看到溶液的变化，第一步需要先准备哪些材料？比如，稀氯化钠溶液和烧杯，你觉得应该先做哪一步：直接加盐酸，还是先量取氯化钠溶液？为什么？
```

## 教学策略

模型学习到的苏格拉底式教学策略包括：
- **助产法**：引导学生自己发现答案
- **反诘**：通过反问让学生思考
- **肯定与引导**：先肯定学生的操作，再引导改进
- **关系建立**：建立良好的师生关系

## 硬件要求

- **最低要求**：16GB GPU内存（不使用量化）
- **推荐配置**：24GB+ GPU内存
- **量化模式**：8GB GPU内存（使用4-bit量化）

## 安装依赖

```bash
pip install torch transformers datasets peft accelerate bitsandbytes
```

## 快速开始

1. **数据预处理**：
```bash
python socratic_finetune_preprocessor.py \
  --data_dir /home/yjh/socChem_final/finetune_data_dataset \
  --output_dir ./processed_data \
  --analyze
```

2. **查看数据分布**：
```bash
python example_usage.py
```

3. **微调模型**：
```bash
python finetune_qwen_socratic.py \
  --train_data ./processed_data/socratic_dialogue_qwen_train.jsonl \
  --val_data ./processed_data/socratic_dialogue_qwen_val.jsonl \
  --output_dir ./socratic_model \
  --use_lora \
  --quantization
```

4. **测试模型**：
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./socratic_model/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 生成苏格拉底式问题
prompt = "学生说: 我已经把稀氯化钠溶液加到烧杯里了，接下来要做什么？"
input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 注意事项

1. **数据量**：原始数据量很大，建议先使用`--max_files`参数测试
2. **显存管理**：使用`--quantization`参数可以减少显存占用
3. **模型选择**：可以根据需要更换其他Qwen系列模型
4. **微调策略**：LoRA微调可以保持原始模型权重，只训练少量参数

## 预期效果

微调后的模型能够：
1. 理解学生的化学实验操作意图
2. 根据学生认知状态调整问题难度
3. 使用苏格拉底式提问引导学生思考
4. 针对不同学生人设（叛逆、好奇、恐惧等）调整教学策略

## 扩展应用

1. **多学科教学**：可以扩展到物理、生物等其他学科
2. **个性化教学**：根据学生历史表现调整教学策略
3. **自动评估**：评估学生的实验操作和思考过程
4. **教学助手**：辅助教师设计苏格拉底式问题

## 引用

如果使用本项目，请引用：
```
苏格拉底式教学对话数据微调系统 - 基于Qwen-7B的化学实验教学助手
```

## 许可证

本项目仅供研究使用，请遵守相关法律法规和模型使用协议。