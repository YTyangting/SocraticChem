#!/usr/bin/env python3
"""
Qwen-7B苏格拉底式教学微调脚本
基于提取的对话数据进行模型微调
"""

import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse
from typing import Dict, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SocraticFinetuner:
    """苏格拉底式教学模型微调器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen-7B", use_lora: bool = True):
        self.model_name = model_name
        self.use_lora = use_lora
        self.tokenizer = None
        self.model = None
        self.peft_config = None
        
    def load_tokenizer(self):
        """加载tokenizer"""
        logger.info(f"加载tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right",
            use_fast=False
        )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        return self.tokenizer
    
    def load_model(self, quantization: bool = False):
        """加载模型"""
        logger.info(f"加载模型: {self.model_name}")
        
        if quantization:
            # 使用4-bit量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model_kwargs = {"quantization_config": bnb_config}
        else:
            model_kwargs = {}
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            **model_kwargs
        )
        
        # 应用LoRA配置
        if self.use_lora:
            self._setup_lora()
            
        return self.model
    
    def _setup_lora(self):
        """设置LoRA配置"""
        logger.info("设置LoRA配置")
        
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # LoRA秩
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
    
    def load_data(self, data_path: str, format_type: str = "qwen") -> Dataset:
        """加载数据"""
        logger.info(f"加载数据: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        
        # 根据格式类型处理数据
        if format_type == "qwen":
            processed_data = self._process_qwen_format(data)
        elif format_type == "instruction":
            processed_data = self._process_instruction_format(data)
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")
        
        # 创建Dataset
        dataset = Dataset.from_list(processed_data)
        return dataset
    
    def _process_qwen_format(self, data: List[Dict]) -> List[Dict]:
        """处理Qwen格式数据"""
        processed = []
        
        for item in data:
            conversations = item.get("conversations", [])
            if len(conversations) >= 2:
                # 构建对话文本
                text = ""
                for conv in conversations:
                    role = conv.get("role", "")
                    content = conv.get("content", "")
                    if role == "user":
                        text += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
                processed.append({"text": text})
        
        return processed
    
    def _process_instruction_format(self, data: List[Dict]) -> List[Dict]:
        """处理指令格式数据"""
        processed = []
        
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            
            # 构建指令文本
            text = f"<|im_start|>system\n你是一位化学老师，使用苏格拉底式教学方法引导学生思考。<|im_end|>\n"
            text += f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n"
            text += f"<|im_start|>assistant\n{output_text}<|im_end|>\n"
            
            processed.append({"text": text})
        
        return processed
    
    def tokenize_function(self, examples):
        """tokenize函数"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
    
    def train(self, train_dataset, val_dataset, output_dir: str, 
              num_epochs: int = 3, batch_size: int = 4):
        """训练模型"""
        logger.info("开始训练")
        
        # 准备训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_total_limit=3,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to="none"
        )
        
        # 数据collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model(f"{output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{output_dir}/final_model")
        
        logger.info(f"训练完成，模型保存到: {output_dir}/final_model")
    
    def generate_example(self, prompt: str, max_length: int = 200):
        """生成示例响应"""
        if not self.model or not self.tokenizer:
            raise ValueError("请先加载模型和tokenizer")
        
        inputs = self.tokenizer(
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取assistant的响应
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            response = response.replace("<|im_end|>", "").strip()
        
        return response


def main():
    parser = argparse.ArgumentParser(description="Qwen-7B苏格拉底式教学微调")
    parser.add_argument("--train_data", type=str, required=True,
                       help="训练数据路径")
    parser.add_argument("--val_data", type=str, required=True,
                       help="验证数据路径")
    parser.add_argument("--output_dir", type=str, default="./socratic_qwen_model",
                       help="输出目录")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B",
                       help="基础模型名称")
    parser.add_argument("--format", type=str, default="qwen",
                       choices=["qwen", "instruction"],
                       help="数据格式")
    parser.add_argument("--epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--use_lora", action="store_true",
                       help="使用LoRA微调")
    parser.add_argument("--quantization", action="store_true",
                       help="使用4-bit量化")
    parser.add_argument("--test_prompt", type=str, default=None,
                       help="测试提示词")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化微调器
    finetuner = SocraticFinetuner(
        model_name=args.model_name,
        use_lora=args.use_lora
    )
    
    # 加载tokenizer和模型
    tokenizer = finetuner.load_tokenizer()
    model = finetuner.load_model(quantization=args.quantization)
    
    # 加载数据
    train_dataset = finetuner.load_data(args.train_data, args.format)
    val_dataset = finetuner.load_data(args.val_data, args.format)
    
    # Tokenize数据
    logger.info("Tokenize数据...")
    tokenized_train = train_dataset.map(
        finetuner.tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        finetuner.tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # 训练模型
    finetuner.train(
        train_dataset=tokenized_train,
        val_dataset=tokenized_val,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # 测试生成
    if args.test_prompt:
        logger.info("测试生成...")
        response = finetuner.generate_example(args.test_prompt)
        print(f"\n测试提示: {args.test_prompt}")
        print(f"模型响应: {response}")


if __name__ == "__main__":
    main()