#!/usr/bin/env python3
"""
学生-教师交互式化学实验平台
包含苏格拉底式引导和双输入模式
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

# 添加路径以便导入原有模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from soc_chem_dia_refactored import ChemSimEngine
    print("✅ 成功导入物理引擎")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

class TeacherAgent:
    """教师代理，负责苏格拉底式引导"""
    
    def __init__(self):
        self.guide_history = []
        
        # 引导问题库（按实验类型分类）
        self.guide_questions = {
            "oxygen_production": [
                "你认为加热高锰酸钾时需要注意什么安全事项？",
                "为什么试管口要略向下倾斜？",
                "如何判断氧气已经收集满了？",
                "实验结束后应该先撤酒精灯还是先撤导管？为什么？",
                "如果加热时试管破裂了，可能是什么原因？"
            ],
            "acid_base": [
                "如何用pH试纸检测溶液的酸碱性？",
                "中和反应有什么明显的现象？",
                "为什么酸和碱混合后温度会升高？",
                "如何判断中和反应恰好完全？",
                "如果不小心把酸溅到皮肤上，应该怎么处理？"
            ],
            "electrolysis": [
                "电解水时为什么要在水中加入少量硫酸？",
                "如何区分正负极产生的气体？",
                "为什么产生的氢气和氧气体积比是2:1？",
                "如果电极接反了会有什么后果？",
                "电解过程中溶液浓度会变化吗？为什么？"
            ]
        }
        
        # 反馈语句库
        self.feedback_phrases = {
            "good": [
                "很好！你注意到了关键点。",
                "这个操作很规范，继续努力。",
                "观察得很仔细，这正是实验的重点。",
                "你的思考方向是正确的。",
                "做得不错，接下来可以思考..."
            ],
            "need_improve": [
                "这个操作有点问题，让我们思考一下...",
                "安全第一！这里需要注意...",
                "实验步骤可以优化，比如...",
                "你漏掉了一个重要环节...",
                "让我们回顾一下这个操作的原理..."
            ],
            "question": [
                "你为什么选择这样做？",
                "你觉得这个现象说明了什么？",
                "如果改变某个条件，结果会怎样？",
                "这个操作背后的化学原理是什么？",
                "如何验证你的猜想？"
            ]
        }
    
    def get_guide_question(self, experiment_type: str, student_action: str, 
                          current_state: Dict, step: int) -> str:
        """获取引导性问题"""
        
        # 根据实验类型和步骤选择问题
        questions = self.guide_questions.get(experiment_type, [])
        if questions:
            # 根据步骤选择问题（循环使用）
            question_idx = step % len(questions)
            base_question = questions[question_idx]
        else:
            base_question = "你认为这个操作的目的是什么？"
        
        # 个性化调整
        if "加热" in student_action or "heat" in student_action.lower():
            return f"{base_question} 特别是加热操作的安全注意事项。"
        elif "加入" in student_action or "add" in student_action.lower():
            return f"{base_question} 思考一下试剂的添加顺序和用量。"
        elif "连接" in student_action or "attach" in student_action.lower():
            return f"{base_question} 装置组装对实验成功很重要。"
        else:
            return base_question
    
    def get_feedback(self, action_success: bool, action_type: str) -> str:
        """获取反馈语句"""
        if action_success:
            phrases = self.feedback_phrases["good"]
        else:
            phrases = self.feedback_phrases["need_improve"]
        
        return random.choice(phrases)
    
    def generate_guide(self, experiment_type: str, student_lang: str, 
                      student_action: Dict, action_result: Dict, 
                      current_state: Dict, step: int) -> str:
        """生成完整的引导语句"""
        
        # 1. 反馈
        success = action_result.get("success", False)
        feedback = self.get_feedback(success, student_action.get("action", ""))
        
        # 2. 引导问题
        question = self.get_guide_question(experiment_type, student_lang, current_state, step)
        
        # 3. 组合
        guide = f"{feedback}\n\n💭 思考题：{question}"
        
        # 记录
        self.guide_history.append({
            "step": step,
            "student_lang": student_lang,
            "guide": guide,
            "time": datetime.now().isoformat()
        })
        
        return guide

class StudentTeacherPlatform:
    """
    学生-教师交互平台
    特点：
    1. 实验选择（3个实验）
    2. 双输入模式（语言 + XDL动作）
    3. 苏格拉底式引导
    4. 完整教学流程
    """
    
    def __init__(self):
        self.session_id = f"teach_{int(time.time())}"
        self.step_count = 0
        self.current_experiment = None
        self.operations = []
        
        # 实验配置
        self.experiments = self._load_experiments()
        
        # 初始化组件
        self.teacher = TeacherAgent()
        self.engine = None
        
        # 记录文件
        self.record_file = f"teaching_log_{self.session_id}.jsonl"
        self._init_records()
        
        print(f"🎓 教学平台已初始化 | 会话: {self.session_id}")
    
    def _load_experiments(self) -> Dict:
        """加载实验配置"""
        return {
            "1": {
                "id": "oxygen_production",
                "name": "高锰酸钾制取氧气",
                "description": "学习加热高锰酸钾制取氧气的方法，掌握气体收集和装置组装",
                "hardware": [
                    {"id": "test_tube", "type": "tube", "name": "试管"},
                    {"id": "alcohol_burner", "type": "heater", "name": "酒精灯"},
                    {"id": "iron_stand", "type": "stand", "name": "铁架台"},
                    {"id": "tube_clamp", "type": "clamp", "name": "试管夹"},
                    {"id": "gas_collector", "type": "bottle", "name": "集气瓶"},
                    {"id": "water_trough", "type": "trough", "name": "水槽"}
                ],
                "reagents": {"KMnO4": "KMnO4", "water": "H2O"},
                "goals": [
                    "正确组装实验装置",
                    "安全加热高锰酸钾",
                    "用排水法收集氧气",
                    "验证氧气性质"
                ]
            },
            "2": {
                "id": "acid_base",
                "name": "酸碱中和反应",
                "description": "探究酸和碱的中和反应，学习使用指示剂和pH试纸",
                "hardware": [
                    {"id": "beaker_50ml", "type": "beaker", "name": "50ml烧杯"},
                    {"id": "beaker_100ml", "type": "beaker", "name": "100ml烧杯"},
                    {"id": "dropper", "type": "dropper", "name": "滴管"},
                    {"id": "glass_rod", "type": "rod", "name": "玻璃棒"},
                    {"id": "pH_paper", "type": "paper", "name": "pH试纸"}
                ],
                "reagents": {"HCl": "HCl", "NaOH": "NaOH", "phenolphthalein": "C20H14O4"},
                "goals": [
                    "准确量取酸和碱",
                    "观察中和反应现象",
                    "使用指示剂判断终点",
                    "理解中和反应原理"
                ]
            },
            "3": {
                "id": "electrolysis",
                "name": "水的电解",
                "description": "通过电解水验证水的组成，学习电解原理",
                "hardware": [
                    {"id": "electrolyzer", "type": "apparatus", "name": "电解器"},
                    {"id": "dc_power", "type": "power", "name": "直流电源"},
                    {"id": "test_tube_x2", "type": "tube", "name": "试管(2支)"},
                    {"id": "electrode", "type": "electrode", "name": "电极"},
                    {"id": "wire", "type": "wire", "name": "导线"}
                ],
                "reagents": {"water": "H2O", "H2SO4": "H2SO4"},
                "goals": [
                    "正确连接电路",
                    "观察气体产生现象",
                    "验证气体性质",
                    "理解电解原理"
                ]
            }
        }
    
    def _init_records(self):
        """初始化记录文件"""
        try:
            with open(self.record_file, 'w', encoding='utf-8') as f:
                meta = {
                    "type": "session_start",
                    "session": self.session_id,
                    "time": datetime.now().isoformat(),
                    "platform": "StudentTeacher v1.0"
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        except:
            print("⚠️  记录文件创建失败，但平台可以继续运行")
    
    def _save_record(self, record: Dict):
        """保存记录"""
        try:
            with open(self.record_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except:
            pass
    
    def select_experiment(self):
        """实验选择界面"""
        print("\n" + "="*60)
        print("🔬 请选择实验项目")
        print("="*60)
        
        for key, exp in self.experiments.items():
            print(f"\n【{key}】{exp['name']}")
            print(f"   描述: {exp['description']}")
            print(f"   目标: {' | '.join(exp['goals'][:2])}...")
        
        print("\n" + "="*60)
        
        while True:
            choice = input("请输入实验编号 (1/2/3): ").strip()
            if choice in self.experiments:
                self.current_experiment = self.experiments[choice]
                break
            else:
                print("❌ 无效选择，请重新输入")
        
        # 显示选择的实验
        exp = self.current_experiment
        print(f"\n✅ 已选择: {exp['name']}")
        print(f"📋 实验目标:")
        for i, goal in enumerate(exp['goals'], 1):
            print(f"   {i}. {goal}")
        
        # 初始化引擎
        self._init_engine()
        
        # 记录实验选择
        self._save_record({
            "type": "experiment_selected",
            "time": datetime.now().isoformat(),
            "experiment": exp['name'],
            "experiment_id": exp['id']
        })
        
        return exp
    
    def _init_engine(self):
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
            print("✅ 实验环境已准备就绪")
        except Exception as e:
            print(f"❌ 引擎初始化失败: {e}")
            raise
    
    def display_state(self):
        """显示当前实验状态"""
        if not self.engine:
            print("⚠️  实验环境未初始化")
            return
        
        print("\n" + "="*60)
        print("📊 实 验 状 态")
        print("="*60)
        
        try:
            snap = self.engine.get_snapshot()
            items = snap.get("hardware", {})
            
            print("\n🧪 器 材 状 态:")
            for obj_id, data in items.items():
                name = data.get("name", obj_id)
                info = []
                
                # 温度
                temp = data.get("temperature", 25)
                if temp != 25:
                    info.append(f"{temp}°C")
                
                # 内容物
                contents = data.get("contents", {})
                if contents:
                    for chem, amt in contents.items():
                        if amt > 0.01:
                            info.append(f"{chem}:{amt:.2f}mol")
                
                # 体积
                vol = data.get("volume_ml", 0)
                if vol > 0.1:
                    info.append(f"{vol}ml")
                
                # 特殊状态
                if data.get("is_sealed"):
                    info.append("🔒密封")
                if data.get("is_on", False):
                    info.append("🔥加热中")
                if data.get("is_connected", False):
                    info.append("🔗已连接")
                
                status = " | ".join(info) if info else "待用"
                print(f"  {name}: {status}")
            
            # 连接状态
            links = snap.get("topology", [])
            if links:
                print("\n🔗 装 置 连 接:")
                for link in links:
                    print(f"  {link['child']} → {link['parent']} ({link.get('type', '连接')})")
            
            print(f"\n📈 实 验 进 度: 第{self.step_count}步")
            
        except Exception as e:
            print(f"❌ 状态显示错误: {e}")
        
        print("="*60)
    
    def get_student_language_input(self) -> str:
        """获取学生语言输入"""
        print("\n💬 请描述你打算进行的操作（自然语言）:")
        print("   例如: '我想组装实验装置' 或 '准备加热试管'")
        print("   > ", end="")
        
        try:
            user_input = input().strip()
            while not user_input:
                print("⚠️  输入不能为空，请重新输入:")
                print("   > ", end="")
                user_input = input().strip()
            return user_input
        except (EOFError, KeyboardInterrupt):
            return "退出"
    
    def get_student_action_input(self) -> Optional[Dict]:
        """获取学生动作输入（XDL格式）"""
        print("\n🔧 请输入具体操作指令（XDL格式）:")
        print("   例如: {\"action\":\"Attach\",\"vessel\":\"tube_clamp\",\"support\":\"iron_stand\"}")
        print("   或输入 '示例' 查看示例")
        print("   > ", end="")
        
        try:
            user_input = input().strip()
            
            if user_input.lower() in ['退出', 'exit', 'quit']:
                return {"action": "Exit"}
            elif user_input.lower() in ['示例', 'example']:
                self._show_action_examples()
                return self.get_student_action_input()
            elif user_input.lower() in ['帮助', 'help']:
                self._show_help()
                return self.get_student_action_input()
            
            # 尝试解析JSON
            if user_input.startswith("{") and user_input.endswith("}"):
                try:
                    action = json.loads(user_input)
                    if "action" in action:
                        return action
                    else:
                        print("❌ JSON中缺少'action'字段")
                        return None
                except json.JSONDecodeError as e:
                    print(f"❌ JSON解析错误: {e}")
                    return None
            else:
                print("❌ 请输入有效的JSON格式")
                return None
                
        except (EOFError, KeyboardInterrupt):
            return {"action": "Exit"}
    
    def _show_action_examples(self):
        """显示动作示例"""
        exp_type = self.current_experiment["id"]
        
        print("\n📋 操 作 示 例:")
        print("="*40)
        
        if exp_type == "oxygen_production":
            examples = [
                '{"action":"Attach","vessel":"tube_clamp","support":"iron_stand"}',
                '{"action":"Add","vessel":"test_tube","reagent":"KMnO4","mass":"5g"}',
                '{"action":"Heat","vessel":"test_tube","temperature":"200"}',
                '{"action":"Attach","vessel":"test_tube","support":"tube_clamp"}'
            ]
        elif exp_type == "acid_base":
            examples = [
                '{"action":"Add","vessel":"beaker_50ml","reagent":"HCl","volume":"20ml"}',
                '{"action":"Add","vessel":"beaker_50ml","reagent":"NaOH","volume":"20ml"}',
                '{"action":"Stir","vessel":"beaker_50ml"}',
                '{"action":"Wait","duration":"10"}'
            ]
        else:  # electrolysis
            examples = [
                '{"action":"Attach","vessel":"electrode","support":"electrolyzer"}',
                '{"action":"Add","v