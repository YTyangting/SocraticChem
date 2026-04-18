#!/usr/bin/env python3
"""
苏格拉底式教学对话数据预处理脚本
用于提取数据并准备Qwen-7B模型微调格式

输入：JSONL格式的苏格拉底式教学对话数据
输出：适合微调的对话格式，包含学生输入和教师响应
"""

import json
import os
import glob
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import random
from collections import defaultdict


@dataclass
class DialogueTurn:
    """对话轮次数据结构"""
    turn_index: int
    student_speak: str  # 学生说的话
    student_actions: List[Dict]  # 学生准备执行的操作
    teacher_response: str  # 教师的苏格拉底式问题
    experiment_context: str  # 实验背景
    current_task: str  # 当前任务描述
    student_traits: Dict  # 学生人设特征
    cognitive_state: str  # 学生认知状态
    strategy_type: str  # 教师策略类型


class SocraticDialogueProcessor:
    """苏格拉底式对话数据处理器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.jsonl_files = self._find_jsonl_files()
        self.processed_dialogues = []
        
    def _find_jsonl_files(self) -> List[str]:
        """查找所有JSONL文件"""
        pattern = os.path.join(self.data_dir, "*.jsonl")
        files = glob.glob(pattern)
        print(f"找到 {len(files)} 个JSONL文件")
        return files
    
    def _extract_experiment_metadata(self, lines: List[Dict]) -> Tuple[str, str, Dict]:
        """从第一条记录中提取实验元数据"""
        experiment_title = "化学实验"
        experiment_goal = ""
        student_traits = {}
        
        for line in lines:
            if line.get("record_type") == "experiment_metadata":
                data = line.get("data", {})
                xdl_enriched = data.get("xdl_enriched", "")
                # 简单提取实验标题和目标
                if "title=" in xdl_enriched:
                    start = xdl_enriched.find('title="') + 7
                    end = xdl_enriched.find('"', start)
                    experiment_title = xdl_enriched[start:end]
                if "goal=" in xdl_enriched:
                    start = xdl_enriched.find('goal="') + 6
                    end = xdl_enriched.find('"', start)
                    experiment_goal = xdl_enriched[start:end]
                break
        
        return experiment_title, experiment_goal, student_traits
    
    def _extract_dialogue_turn(self, line: Dict, session_id: str) -> Optional[DialogueTurn]:
        """从单行数据中提取对话轮次"""
        try:
            data = line.get("data", {})
            agents = data.get("agents", {})
            
            # 提取学生信息
            student_data = agents.get("student", {})
            student_input = student_data.get("input", {})
            student_output = student_data.get("output", {})
            
            # 提取教师信息
            teacher_data = agents.get("teacher", {})
            teacher_input = teacher_data.get("input", {})
            teacher_output = teacher_data.get("output", {})
            
            # 提取元数据
            meta = data.get("meta", {})
            bkt_status = data.get("bkt_status", {})
            
            # 构建对话轮次
            turn = DialogueTurn(
                turn_index=line.get("turn_index", 0),
                student_speak=student_output.get("speak", ""),
                student_actions=student_output.get("actions_ready_to_run", []),
                teacher_response=teacher_output.get("response", ""),
                experiment_context=meta.get("current_task_desc", ""),
                current_task=meta.get("current_task_desc", ""),
                student_traits=student_input.get("traits", {}),
                cognitive_state=student_input.get("cognitive_state", ""),
                strategy_type=teacher_output.get("strategy", "")
            )
            
            # 验证必要字段
            if not turn.student_speak or not turn.teacher_response:
                return None
                
            return turn
            
        except Exception as e:
            print(f"提取对话轮次时出错: {e}")
            return None
    
    def process_file(self, file_path: str) -> List[DialogueTurn]:
        """处理单个JSONL文件"""
        dialogues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [json.loads(line.strip()) for line in f if line.strip()]
            
            if not lines:
                return []
            
            # 提取实验元数据
            experiment_title, experiment_goal, _ = self._extract_experiment_metadata(lines)
            
            # 提取对话轮次
            current_session = None
            session_dialogues = []
            
            for line in lines:
                if line.get("record_type") == "experiment_metadata":
                    continue
                    
                session_id = line.get("session_id")
                if session_id != current_session:
                    if session_dialogues:
                        dialogues.extend(session_dialogues)
                    current_session = session_id
                    session_dialogues = []
                
                turn = self._extract_dialogue_turn(line, session_id)
                if turn:
                    session_dialogues.append(turn)
            
            if session_dialogues:
                dialogues.extend(session_dialogues)
                
            print(f"从 {os.path.basename(file_path)} 提取了 {len(dialogues)} 个对话轮次")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            
        return dialogues
    
    def process_all_files(self, max_files: Optional[int] = None) -> List[DialogueTurn]:
        """处理所有文件"""
        all_dialogues = []
        
        files_to_process = self.jsonl_files
        if max_files:
            files_to_process = files_to_process[:max_files]
        
        for i, file_path in enumerate(files_to_process):
            print(f"处理文件 {i+1}/{len(files_to_process)}: {os.path.basename(file_path)}")
            dialogues = self.process_file(file_path)
            all_dialogues.extend(dialogues)
        
        print(f"总共提取了 {len(all_dialogues)} 个对话轮次")
        return all_dialogues
    
    def format_for_finetuning(self, dialogues: List[DialogueTurn], 
                            output_format: str = "qwen") -> List[Dict]:
        """将对话格式化为微调格式"""
        
        formatted_data = []
        
        for dialogue in dialogues:
            # 构建学生输入：包含说的话和准备执行的操作
            student_input = dialogue.student_speak
            if dialogue.student_actions:
                actions_str = json.dumps(dialogue.student_actions, ensure_ascii=False)
                student_input += f"\n[准备执行的操作]: {actions_str}"
            
            # 构建上下文信息
            context_parts = []
            if dialogue.experiment_context:
                context_parts.append(f"当前任务: {dialogue.experiment_context}")
            if dialogue.cognitive_state:
                context_parts.append(f"学生认知状态: {dialogue.cognitive_state}")
            if dialogue.strategy_type:
                context_parts.append(f"教学策略: {dialogue.strategy_type}")
            
            context = "\n".join(context_parts) if context_parts else ""
            
            # 根据输出格式构建数据
            if output_format == "qwen":
                # Qwen微调格式（对话格式）
                formatted = {
                    "conversations": [
                        {
                            "role": "user",
                            "content": f"{context}\n学生说: {student_input}"
                        },
                        {
                            "role": "assistant",
                            "content": dialogue.teacher_response
                        }
                    ]
                }
            elif output_format == "instruction":
                # 指令微调格式
                formatted = {
                    "instruction": f"你是一位化学老师，正在进行苏格拉底式教学。{context}",
                    "input": f"学生说: {student_input}",
                    "output": dialogue.teacher_response
                }
            elif output_format == "simple":
                # 简单对话格式
                formatted = {
                    "user": student_input,
                    "assistant": dialogue.teacher_response,
                    "context": context
                }
            else:
                raise ValueError(f"不支持的输出格式: {output_format}")
            
            formatted_data.append(formatted)
        
        return formatted_data
    
    def analyze_data_distribution(self, dialogues: List[DialogueTurn]):
        """分析数据分布"""
        print("\n=== 数据分布分析 ===")
        
        # 统计教学策略
        strategy_counts = defaultdict(int)
        for dialogue in dialogues:
            strategy_counts[dialogue.strategy_type] += 1
        
        print("教学策略分布:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy}: {count} ({count/len(dialogues)*100:.1f}%)")
        
        # 统计对话长度
        student_lengths = [len(d.student_speak) for d in dialogues]
        teacher_lengths = [len(d.teacher_response) for d in dialogues]
        
        print(f"\n学生发言平均长度: {sum(student_lengths)/len(student_lengths):.1f} 字符")
        print(f"教师回应平均长度: {sum(teacher_lengths)/len(teacher_lengths):.1f} 字符")
        
        # 统计操作类型
        action_types = defaultdict(int)
        for dialogue in dialogues:
            for action in dialogue.student_actions:
                action_type = action.get("action", "unknown")
                action_types[action_type] += 1
        
        if action_types:
            print("\n学生操作类型分布:")
            for action_type, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {action_type}: {count}")
    
    def save_data(self, data: List[Dict], output_path: str, split_ratio: float = 0.8):
        """保存数据，可选分割为训练集和验证集"""
        
        # 随机打乱数据
        random.shuffle(data)
        
        # 分割数据
        split_idx = int(len(data) * split_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # 保存训练集
        train_path = output_path.replace(".jsonl", "_train.jsonl")
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"训练集保存到: {train_path} ({len(train_data)} 条)")
        
        # 保存验证集
        val_path = output_path.replace(".jsonl", "_val.jsonl")
        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"验证集保存到: {val_path} ({len(val_data)} 条)")
        
        return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="苏格拉底式教学对话数据预处理")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="包含JSONL文件的目录路径")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="输出目录路径")
    parser.add_argument("--output_format", type=str, default="qwen",
                       choices=["qwen", "instruction", "simple"],
                       help="输出数据格式")
    parser.add_argument("--max_files", type=int, default=None,
                       help="最大处理文件数（用于测试）")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                       help="训练集分割比例")
    parser.add_argument("--analyze", action="store_true",
                       help="分析数据分布")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化处理器
    processor = SocraticDialogueProcessor(args.data_dir)
    
    # 处理数据
    print("开始处理数据...")
    dialogues = processor.process_all_files(max_files=args.max_files)
    
    if not dialogues:
        print("没有提取到有效对话数据")
        return
    
    # 分析数据分布
    if args.analyze:
        processor.analyze_data_distribution(dialogues)
    
    # 格式化数据
    print(f"\n格式化为 {args.output_format} 格式...")
    formatted_data = processor.format_for_finetuning(dialogues, args.output_format)
    
    # 保存数据
    output_path = os.path.join(args.output_dir, f"socratic_dialogue_{args.output_format}.jsonl")
    train_path, val_path = processor.save_data(formatted_data, output_path, args.split_ratio)
    
    print(f"\n=== 处理完成 ===")
    print(f"总对话轮次: {len(dialogues)}")
    print(f"格式化数据: {len(formatted_data)} 条")
    print(f"输出目录: {args.output_dir}")
    
    # 显示示例
    print("\n=== 数据示例 ===")
    if formatted_data:
        example = formatted_data[0]
        print(json.dumps(example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()