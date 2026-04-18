#!/usr/bin/env python3
"""
完整版学生-教师交互式化学实验平台
包含：实验选择、双输入模式、苏格拉底式引导
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Optional
import re

# 添加路径以便导入原有模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from soc_chem_dia_refactored import ChemSimEngine
    print("✅ 成功导入物理引擎")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

class SocratesTeacher:
    """苏格拉底式教师代理"""
    
    def __init__(self):
        self.dialogue_history = []
        
        # 引导问题库（按实验阶段分类）
        self.question_bank = {
            "planning": [  # 计划阶段
                "你为什么要进行这个操作？",
                "这个操作的目标是什么？",
                "操作前需要考虑哪些安全事项？",
                "你预计会看到什么现象？",
                "如果出现意外情况，你的应对方案是什么？"
            ],
            "execution": [  # 执行阶段
                "这个操作的关键步骤是什么？",
                "如何确保操作的准确性？",
                "你观察到了什么现象？",
                "这个现象说明了什么？",
                "操作中遇到了什么困难？如何解决的？"
            ],
            "reflection": [  # 反思阶段
                "操作结果符合你的预期吗？为什么？",
                "如果重做一次，你会改进什么？",
                "这个操作背后的化学原理是什么？",
                "操作中的关键学习点是什么？",
                "如何将这个方法应用到其他实验中？"
            ]
        }
        
        # 反馈语句
        self.feedback_templates = {
            "encourage": [
                "很好！你考虑得很周到。",
                "这个思路很清晰，继续努力。",
                "观察得很仔细，这正是科学探究的精神。",
                "你的思考很有逻辑性。",
                "做得不错，让我们深入思考一下..."
            ],
            "guide": [
                "让我们从另一个角度思考这个问题...",
                "这里有一个重要的安全注意事项...",
                "实验步骤可以这样优化...",
                "你漏掉了一个关键环节...",
                "让我们回顾一下这个操作的原理..."
            ],
            "challenge": [
                "为什么必须按照这个顺序操作？",
                "如果改变某个条件，结果会怎样？",
                "如何证明你的猜想是正确的？",
                "这个现象有没有其他解释？",
                "操作中的误差来源有哪些？"
            ]
        }
    
    def get_question(self, stage: str, context: Dict) -> str:
        """获取引导性问题"""
        questions = self.question_bank.get(stage, self.question_bank["planning"])
        question = random.choice(questions)
        
        # 根据上下文个性化
        action = context.get("action", "")
        if "加热" in str(action) or "heat" in str(action).lower():
            question += " 特别是加热操作的安全性和温度控制。"
        elif "添加" in str(action) or "add" in str(action).lower():
            question += " 注意试剂的用量和添加顺序。"
        elif "连接" in str(action) or "attach" in str(action).lower():
            question += " 装置的气密性和稳定性很重要。"
        
        return question
    
    def get_feedback(self, feedback_type: str = "encourage") -> str:
        """获取反馈语句"""
        templates = self.feedback_templates.get(feedback_type, self.feedback_templates["encourage"])
        return random.choice(templates)
    
    def generate_dialogue(self, student_input: Dict, experiment_info: Dict, 
                         step: int, success: bool) -> str:
        """生成苏格拉底式对话"""
        
        # 确定阶段
        if step == 1:
            stage = "planning"
        elif step <= 3:
            stage = "execution"
        else:
            stage = "reflection"
        
        # 确定反馈类型
        if success:
            feedback_type = "encourage"
        else:
            feedback_type = "guide"
        
        # 构建上下文
        context = {
            "step": step,
            "action": student_input.get("action", {}),
            "experiment": experiment_info.get("name", ""),
            "stage": stage
        }
        
        # 生成对话
        feedback = self.get_feedback(feedback_type)
        question = self.get_question(stage, context)
        
        # 50%概率添加挑战性问题
        if random.random() > 0.5:
            challenge = self.get_feedback("challenge")
            dialogue = f"{feedback}\n\n💭 {question}\n\n🤔 {challenge}"
        else:
            dialogue = f"{feedback}\n\n💭 {question}"
        
        # 记录
        self.dialogue_history.append({
            "step": step,
            "stage": stage,
            "dialogue": dialogue,
            "time": datetime.now().isoformat()
        })
        
        return dialogue

class TeachingPlatform:
    """
    教学平台主类
    实现完整教学流程：
    1. 实验选择（3个实验）
    2. 双输入模式（语言意图 + XDL动作）
    3. 状态显示
    4. 苏格拉底式引导
    """
    
    def __init__(self):
        self.session_id = f"teach_{int(time.time())}"
        self.step_number = 0
        self.current_experiment = None
        self.history = []
        
        # 初始化组件
        self.teacher = SocratesTeacher()
        self.engine = None
        
        # 实验配置
        self.experiments = self._setup_experiments()
        
        # 记录文件
        self.log_file = f"teaching_session_{self.session_id}.jsonl"
        self._init_log_file()
        
        print(f"🎓 教学平台初始化完成 | 会话ID: {self.session_id}")
    
    def _setup_experiments(self) -> Dict:
        """设置实验项目"""
        return {
            "1": {
                "id": "exp_oxygen",
                "name": "高锰酸钾制取氧气",
                "desc": "学习固体加热制取气体的方法，掌握装置组装和安全操作",
                "hardware": [
                    {"id": "tube1", "type": "tube", "name": "试管"},
                    {"id": "burner1", "type": "heater", "name": "酒精灯"},
                    {"id": "stand1", "type": "stand", "name": "铁架台"},
                    {"id": "clamp1", "type": "clamp", "name": "试管夹"},
                    {"id": "bottle1", "type": "bottle", "name": "集气瓶"},
                    {"id": "trough1", "type": "trough", "name": "水槽"}
                ],
                "reagents": {"KMnO4": "KMnO4"},
                "objectives": [
                    "安全组装加热装置",
                    "正确加热高锰酸钾",
                    "用排水法收集气体",
                    "验证氧气性质"
                ]
            },
            "2": {
                "id": "exp_neutralization",
                "name": "酸碱中和反应",
                "desc": "探究酸和碱的反应，学习使用指示剂和测量pH",
                "hardware": [
                    {"id": "beaker1", "type": "beaker", "name": "烧杯(50ml)"},
                    {"id": "beaker2", "type": "beaker", "name": "烧杯(100ml)"},
                    {"id": "dropper1", "type": "dropper", "name": "滴管"},
                    {"id": "rod1", "type": "rod", "name": "玻璃棒"},
                    {"id": "paper_ph", "type": "paper", "name": "pH试纸"}
                ],
                "reagents": {"HCl": "HCl", "NaOH": "NaOH", "phenol": "C6H6O"},
                "objectives": [
                    "准确量取试剂",
                    "观察中和现象",
                    "使用指示剂",
                    "理解反应原理"
                ]
            },
            "3": {
                "id": "exp_electrolysis",
                "name": "水的电解",
                "desc": "通过电解验证水的组成，学习电解原理",
                "hardware": [
                    {"id": "electrolyzer1", "type": "apparatus", "name": "电解槽"},
                    {"id": "power1", "type": "power", "name": "电源"},
                    {"id": "tube_a", "type": "tube", "name": "试管A"},
                    {"id": "tube_b", "type": "tube", "name": "试管B"},
                    {"id": "electrode1", "type": "electrode", "name": "电极"},
                    {"id": "wire1", "type": "wire", "name": "导线"}
                ],
                "reagents": {"water": "H2O", "H2SO4": "H2SO4"},
                "objectives": [
                    "正确连接电路",
                    "观察气体产生",
                    "验证气体性质",
                    "计算体积比"
                ]
            }
        }
    
    def _init_log_file(self):
        """初始化日志文件"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                meta = {
                    "type": "session_start",
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "platform": "TeachingPlatform v2.0"
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        except:
            print("⚠️  日志文件创建失败，但平台可以继续运行")
    
    def _log_event(self, event_type: str, data: Dict):
        """记录事件"""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "step": self.step_number,
            "data": data
        }
        
        self.history.append(event)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except:
            pass
    
    def show_experiment_menu(self):
        """显示实验菜单"""
        print("\n" + "="*60)
        print("🔬 化学实验教学平台 - 实验选择")
        print("="*60)
        
        for key, exp in self.experiments.items():
            print(f"\n【{key}】 {exp['name']}")
            print(f"   简介: {exp['desc']}")
            print(f"   学习目标: {', '.join(exp['objectives'][:2])}...")
        
        print("\n" + "="*60)
        
        while True:
            choice = input("请选择实验编号 (1/2/3): ").strip()
            if choice in self.experiments:
                self.current_experiment = self.experiments[choice]
                break
            else:
                print("❌ 无效选择，请重新输入")
        
        # 显示选择结果
        exp = self.current_experiment
        print(f"\n✅ 已选择实验: {exp['name']}")
        print("📋 实验目标:")
        for i, obj in enumerate(exp['objectives'], 1):
            print(f"   {i}. {obj}")
        
        # 初始化物理引擎
        self._init_physics_engine()
        
        # 记录实验选择
        self._log_event("experiment_selected", {
            "experiment_id": exp["id"],
            "experiment_name": exp["name"]
        })
        
        return exp
    
    def _init_physics_engine(self):
        """初始化物理引擎"""
        try:
            exp = self.current_experiment
            self.engine = ChemSimEngine(
                hardware_config=exp["hardware"],
                reagent_map=exp["reagents"],
                clumsiness_level="low",
                logger_name="TeachingEngine",
                silent_mode=True
            )
            print("✅ 实验环境准备就绪")
        except Exception as e:
            print(f"❌ 物理引擎初始化失败: {e}")
            raise
    
    def show_current_state(self):
        """显示当前实验状态"""
        if not self.engine:
            print("⚠️  实验环境未就绪")
            return
        
        print("\n" + "="*60)
        print("📊 当前实验状态")
        print("="*60)
        
        try:
            snapshot = self.engine.get_snapshot()
            equipment = snapshot.get("hardware", {})
            
            print("\n🧪 实验器材:")
            for eq_id, eq_data in equipment.items():
                name = eq_data.get("name", eq_id)
                status_parts = []
                
                # 温度
                temp = eq_data.get("temperature", 25)
                if temp != 25:
                    status_parts.append(f"{temp}°C")
                
                # 内容物
                contents = eq_data.get("contents", {})
                if contents:
                    for chem, amount in contents.items():
                        if amount > 0.01:
                            status_parts.append(f"{chem}:{amount:.3f}")
                
                # 体积
                volume = eq_data.get("volume_ml", 0)
                if volume > 0.1:
                    status_parts.append(f"{volume}ml")
                
                # 状态标记
                if eq_data.get("is_sealed"):
                    status_parts.append("密封")
                if eq_data.get("is_on", False):
                    status_parts.append("加热中")
                
                status = " | ".join(status_parts) if status_parts else "就绪"
                print(f"  {name}: {status}")
            
            # 连接状态
            connections = snapshot.get("topology", [])
            if connections:
                print("\n🔗 装置连接:")
                for conn in connections:
                    print(f"  {conn['child']} → {conn['parent']}")
            
            print(f"\n📈 实验进度: 第{self.step_number}步")
            
        except Exception as e:
            print(f"❌ 状态获取失败: {e}")
        
        print("="*60)
    
    def get_language_intent(self) -> str:
        """获取学生语言意图"""
        print("\n💬 【第一步：描述意图】")
        print("请用自然语言描述你打算进行的操作")
        print("例如: '我想组装加热装置' 或 '准备添加试剂'")
        print("输入: ", end="")
        
        try:
            intent = input().strip()
            while not intent:
                print("输入不能为空，请重新输入: ", end="")
                intent = input().strip()
            return intent
        except (EOFError, KeyboardInterrupt):
            return "退出"
    
    def get_action_command(self) -> Optional[Dict]:
        """获取学生动作命令（XDL格式）"""
        print("\n🔧 【第二步：执行操作】")
        print("请输入XDL格式的操作命令")
        print("例如: {\"action\":\"Attach\",\"vessel\":\"clamp1\",\"support\":\"stand1\"}")
        print("输入 '帮助' 查看帮助，'示例' 查看示例")
        print("输入: ", end="")
        
        try:
            command = input().strip()
            
            if command.lower() in ['退出', 'exit', 'quit']:
                return {"action": "Exit"}
            elif command.lower() in ['帮助', 'help']:
                self._show_command_help()
                return self.get_action_command()
            elif command.lower() in ['示例', 'example']:
                self._show_command_examples()
                return self.get_action_command()
            
            # 解析JSON命令
            if command.startswith("{") and command.endswith("}"):
                try:
                    action = json.loads(command)
                    if "action" in action:
                        return action
                    else:
                        print("❌ 命令必须包含'action'字段")
                        return None
                except json.JSONDecodeError as e:
                    print(f"❌ JSON解析错误: {e}")
                    return None
            else:
                print("❌ 请输入有效的JSON格式")
                return None
                
        except (EOFError, KeyboardInterrupt):
            return {"action": "Exit"}
    
    def _show_command_help(self):
        """显示命令帮助"""
        print("\n📖 XDL命令帮助")
        print("="*40)
        print("基本格式: {\"action\":\"动作类型\", ...参数}")
        print("\n常用动作:")
        print("  加料: {\"action\":\"Add\",\"vessel\":\"容器\",\"reagent\":\"试剂\",\"volume\":\"10ml\"}")
        print("  加热: {\"action\":\"Heat\",\"vessel\":\"容器\",\"temperature\":\"100\"}")
        print("  连接: {\"action\":\"Attach\",\"vessel\":\"子部件\",\"support\":\"父部件\"}")
        print("  转移: {\"action\":\"Transfer\",\"from_vessel\":\"源\",\"to_vessel\":\"目标\"}")
        print("  搅拌: {\"action\":\"Stir\",\"vessel\":\"容器\"}")
        print("  等待: {\"action\":\"Wait\",\"duration\":\"秒数\"}")
        print("="*40)
    
    def _show_command_examples(self):
        """显示命令示例"""
        exp_id = self.current_experiment["id"]
        
        print("\n📋 命令示例")
        print("="*40)
        
        if exp_id == "exp_oxygen":
            examples = [
                '{"action":"Attach","vessel":"clamp1","support":"stand1"}',
                '{"action":"Add","vessel":"tube1","reagent":"KMnO4","mass":"5g"}',
                '{"action":"Heat","vessel":"tube1","temperature":"200"}'
            ]
        elif exp_id == "exp_neutralization":
            examples = [
                '{"action":"Add","vessel":"beaker1","reagent":"HCl","