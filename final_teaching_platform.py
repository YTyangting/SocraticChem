#!/usr/bin/env python3
"""
最终版：学生-教师交互式化学实验平台
完整实现：
1. 实验选择（3个实验）
2. 双输入模式（语言意图 + XDL动作）
3. 实时状态显示
4. 苏格拉底式引导问题
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Optional

# 添加路径以便导入原有模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from soc_chem_dia_refactored import ChemSimEngine
    print("✅ 成功导入物理引擎")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

class SocratesGuide:
    """苏格拉底式引导系统"""
    
    def __init__(self):
        self.guide_history = []
        
        # 引导问题库
        self.guide_questions = {
            "planning": [
                "你为什么要进行这个操作？目的是什么？",
                "操作前需要考虑哪些安全事项？",
                "你预计会看到什么现象？为什么？",
                "如果出现意外，你的应对方案是什么？",
                "这个操作的关键步骤是什么？"
            ],
            "execution": [
                "如何确保操作的准确性？",
                "你观察到了什么现象？这说明了什么？",
                "操作中遇到了什么困难？如何解决？",
                "这个现象是否符合预期？为什么？",
                "操作的关键控制点是什么？"
            ],
            "reflection": [
                "操作结果符合预期吗？为什么？",
                "如果重做，你会改进什么？",
                "这个操作背后的化学原理是什么？",
                "关键学习点是什么？",
                "如何应用到其他实验？"
            ]
        }
        
        # 反馈语句
        self.feedbacks = {
            "good": [
                "很好！思考很全面。",
                "操作规范，继续努力。",
                "观察仔细，科学精神可嘉。",
                "思路清晰，逻辑性强。",
                "做得不错，让我们深入思考..."
            ],
            "improve": [
                "让我们从安全角度再思考一下...",
                "这里有个重要细节需要注意...",
                "操作顺序可以优化...",
                "漏掉了一个关键环节...",
                "让我们回顾一下原理..."
            ]
        }
    
    def get_guide(self, step: int, action_type: str, success: bool) -> str:
        """获取引导内容"""
        # 确定阶段
        if step <= 2:
            stage = "planning"
        elif step <= 5:
            stage = "execution"
        else:
            stage = "reflection"
        
        # 选择问题
        questions = self.guide_questions[stage]
        question = random.choice(questions)
        
        # 个性化问题
        if "heat" in action_type.lower() or "加热" in action_type:
            question += " 特别是加热操作的安全控制。"
        elif "add" in action_type.lower() or "添加" in action_type:
            question += " 注意试剂用量和添加顺序。"
        elif "attach" in action_type.lower() or "连接" in action_type:
            question += " 装置稳定性和气密性很重要。"
        
        # 选择反馈
        feedback_type = "good" if success else "improve"
        feedback = random.choice(self.feedbacks[feedback_type])
        
        # 组合
        guide = f"{feedback}\n\n💭 引导问题：{question}"
        
        # 记录
        self.guide_history.append({
            "step": step,
            "guide": guide,
            "time": datetime.now().isoformat()
        })
        
        return guide

class TeachingExperimentPlatform:
    """教学实验平台"""
    
    def __init__(self):
        self.session_id = f"session_{int(time.time())}"
        self.step = 0
        self.current_exp = None
        self.records = []
        
        # 初始化组件
        self.guide_system = SocratesGuide()
        self.physics_engine = None
        
        # 实验配置
        self.experiment_options = self._create_experiments()
        
        # 日志文件
        self.log_file = f"teach_log_{self.session_id}.jsonl"
        self._setup_logging()
        
        print(f"🎓 教学平台已就绪 | ID: {self.session_id}")
    
    def _create_experiments(self) -> Dict:
        """创建实验选项"""
        return {
            "1": {
                "name": "制取氧气实验",
                "desc": "加热高锰酸钾制取氧气，学习气体收集方法",
                "setup": [
                    {"id": "tube1", "type": "tube", "name": "试管"},
                    {"id": "heater1", "type": "heater", "name": "酒精灯"},
                    {"id": "stand1", "type": "stand", "name": "铁架台"},
                    {"id": "clamp1", "type": "clamp", "name": "试管夹"},
                    {"id": "bottle1", "type": "bottle", "name": "集气瓶"}
                ],
                "chems": {"KMnO4": "KMnO4"},
                "goals": ["组装装置", "加热固体", "收集气体", "验证性质"]
            },
            "2": {
                "name": "中和反应实验",
                "desc": "酸碱中和反应，学习使用指示剂",
                "setup": [
                    {"id": "beaker1", "type": "beaker", "name": "烧杯"},
                    {"id": "dropper1", "type": "dropper", "name": "滴管"},
                    {"id": "rod1", "type": "rod", "name": "玻璃棒"},
                    {"id": "paper1", "type": "paper", "name": "pH试纸"}
                ],
                "chems": {"HCl": "HCl", "NaOH": "NaOH"},
                "goals": ["量取试剂", "观察现象", "使用指示剂", "理解原理"]
            },
            "3": {
                "name": "电解水实验",
                "desc": "电解水验证组成，学习电解原理",
                "setup": [
                    {"id": "cell1", "type": "cell", "name": "电解槽"},
                    {"id": "power1", "type": "power", "name": "电源"},
                    {"id": "tube_a", "type": "tube", "name": "试管A"},
                    {"id": "tube_b", "type": "tube", "name": "试管B"},
                    {"id": "wire1", "type": "wire", "name": "导线"}
                ],
                "chems": {"water": "H2O", "acid": "H2SO4"},
                "goals": ["连接电路", "观察气体", "验证性质", "计算比例"]
            }
        }
    
    def _setup_logging(self):
        """设置日志"""
        try:
            with open(self.log_file, 'w') as f:
                start_info = {
                    "type": "start",
                    "id": self.session_id,
                    "time": datetime.now().isoformat()
                }
                f.write(json.dumps(start_info) + "\n")
        except:
            print("⚠️  日志初始化失败")
    
    def _log(self, event: str, data: dict):
        """记录事件"""
        record = {
            "event": event,
            "step": self.step,
            "time": datetime.now().isoformat(),
            "data": data
        }
        self.records.append(record)
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(record) + "\n")
        except:
            pass
    
    def choose_experiment(self):
        """选择实验"""
        print("\n" + "="*60)
        print("🔬 请选择实验项目")
        print("="*60)
        
        for num, exp in self.experiment_options.items():
            print(f"\n【{num}】 {exp['name']}")
            print(f"   {exp['desc']}")
            print(f"   目标: {', '.join(exp['goals'][:2])}...")
        
        print("\n" + "="*60)
        
        while True:
            choice = input("输入编号 (1/2/3): ").strip()
            if choice in self.experiment_options:
                self.current_exp = self.experiment_options[choice]
                break
            print("❌ 无效选择")
        
        # 显示选择
        exp = self.current_exp
        print(f"\n✅ 已选择: {exp['name']}")
        print("🎯 实验目标:")
        for i, goal in enumerate(exp['goals'], 1):
            print(f"   {i}. {goal}")
        
        # 初始化引擎
        self._init_engine()
        
        self._log("experiment_chosen", {
            "name": exp["name"],
            "id": choice
        })
        
        return exp
    
    def _init_engine(self):
        """初始化物理引擎"""
        try:
            exp = self.current_exp
            self.physics_engine = ChemSimEngine(
                hardware_config=exp["setup"],
                reagent_map=exp["chems"],
                clumsiness_level="low",
                silent_mode=True
            )
            print("✅ 实验环境准备完成")
        except Exception as e:
            print(f"❌ 引擎初始化失败: {e}")
            raise
    
    def show_state(self):
        """显示状态"""
        if not self.physics_engine:
            print("⚠️  引擎未就绪")
            return
        
        print("\n" + "="*60)
        print("📊 实 验 状 态")
        print("="*60)
        
        try:
            snap = self.physics_engine.get_snapshot()
            items = snap.get("hardware", {})
            
            print("\n🧪 器 材:")
            for obj_id, data in items.items():
                name = data.get("name", obj_id)
                info = []
                
                # 温度
                temp = data.get("temperature", 25)
                if temp != 25:
                    info.append(f"{temp}°C")
                
                # 内容
                contents = data.get("contents", {})
                if contents:
                    for chem, amt in contents.items():
                        if amt > 0.01:
                            info.append(f"{chem}:{amt:.2f}")
                
                # 体积
                vol = data.get("volume_ml", 0)
                if vol > 0.1:
                    info.append(f"{vol}ml")
                
                # 状态
                if data.get("is_sealed"):
                    info.append("密封")
                if data.get("is_on", False):
                    info.append("加热")
                
                status = " | ".join(info) if info else "就绪"
                print(f"  {name}: {status}")
            
            # 连接
            links = snap.get("topology", [])
            if links:
                print("\n🔗 连 接:")
                for link in links:
                    print(f"  {link['child']} → {link['parent']}")
            
            print(f"\n📈 进 度: 第{self.step}步")
            
        except Exception as e:
            print(f"❌ 状态错误: {e}")
        
        print("="*60)
    
    def get_intent(self) -> str:
        """获取意图描述"""
        print("\n💬 【描述意图】")
        print("你想做什么？用语言描述")
        print("例: '组装装置' 或 '添加试剂'")
        print("输入: ", end="")
        
        try:
            text = input().strip()
            while not text:
                print("请描述意图: ", end="")
                text = input().strip()
            return text
        except:
            return "退出"
    
    def get_action(self) -> Optional[dict]:
        """获取动作命令"""
        print("\n🔧 【执行操作】")
        print("输入XDL命令 (JSON格式)")
        print("例: {\"action\":\"Attach\",\"vessel\":\"clamp1\",\"support\":\"stand1\"}")
        print("输入 'help' 查看帮助")
        print("输入: ", end="")
        
        try:
            cmd = input().strip()
            
            if cmd.lower() in ['退出', 'exit']:
                return {"action": "Exit"}
            elif cmd.lower() in ['帮助', 'help']:
                self._show_help()
                return self.get_action()
            
            # 解析JSON
            if cmd.startswith("{") and cmd.endswith("}"):
                try:
                    action = json.loads(cmd)
                    if "action" in action:
                        return action
                    else:
                        print("❌ 缺少action字段")
                        return None
                except Exception as e:
                    print(f"❌ JSON错误: {e}")
                    return None
            else:
                print("❌ 需要JSON格式")
                return None
                
        except:
            return {"action": "Exit"}
    
    def _show_help(self):
        """显示帮助"""
        print("\n📖 命 令 帮 助")
        print("="*40)
        print("格式: {\"action\":\"类型\", ...参数}")
        print("\n动作类型:")
        print("  Add - 加料")
        print("    {\"action\":\"Add\",\"vessel\":\"容器\",\"reagent\":\"试剂\",\"volume\":\"10ml\"}")
        print("  Heat - 加热")
        print("    {\"action\":\"Heat\",\"vessel\":\"容器\",\"temperature\":\"100\"}")
        print("  Attach - 连接")
        print("    {\"action\":\"Attach\",\"vessel\":\"子\",\"support\":\"父\"}")
        print("  Stir - 搅拌")
        print("    {\"action\":\"Stir\",\"vessel\":\"容器\"}")
        print("  Wait - 等待")
        print("    {\"action\":\"Wait\",\"duration\":\"5\"}")
        print("="*40)
    
    def execute(self, action: dict) -> dict:
        """执行动作"""
        if not action or "action" not in action:
            return {"ok": False, "msg": "无效命令"}
        
        if action["action"] == "Exit":
            return {"ok": True, "msg": "退出", "exit": True}
        
        try:
            result, _ = self.physics_engine.execute(action)
            return {"ok": True, "msg": result}
        except Exception as e:
            return {"ok": False, "msg": str(e)}
    
    def run(self):
        """运行主流程"""
        print("\n" + "="*60)
        print("🎓 化 学 实 验 教 学")
        print("="*60)
        print("流程: 选择实验 → 描述意图 → 执行操作 → 获得引导")
        print("="*60)
        
        # 1. 选择实验
        self.choose_experiment()
        
        print("\n✅ 实验准备完成！可以开始了。")
        print("💡 提示: 先想清楚要做什么，再具体操作")
        
        # 主循环
        while True:
            self.step += 1
            print(f"\n{'='*60}")
            print(f"🔄 第 {self.step} 步")
            print('='*60)
            
            # 2. 获取意图
            print("\n【第一步：意图描述】")
            intent = self.get_intent()
            if intent == "退出":
                break
            
            self._log("intent", {"text": intent})
            
            # 3. 获取动作
            print("\n【第二步：具体操作】")
            action = self.get_action()
            if not action:
                print("⚠️  无效操作，重新开始")
                self.step -= 1
                continue
            
            if action.get("action") == "Exit":
                break
            
            self._log("action", action)
            
            # 4. 执行
            print(f"\n⚡ 执行: {json.dumps(action)}")
            result = self.execute(action)
            
            if result.get("exit"):
                break
            
            self._log("result", result)
            
            if result["ok"]:
                print(f"✅ 成功: {result['msg'][:50]}...")
            else:
                print(f"❌ 失败: {result['msg']}")
            
            # 5. 显示状态
            print("\n【当前状态】")
            self.show_state()
            
            # 6. 引导
            print("\n【教师引导】")
            guide = self.guide_system.get_guide(
                step=self.step,
                action_type=action.get("action", ""),
                success=result["ok"]
            )
            print(guide)
            
            self._log("guide", {"text": guide})
            
            # 询问继续
            print("\n" + "-"*40)
            cont = input("继续下一步？(y/n): ").lower()
            if cont not in ['y', 'yes', '是']:
                break
        
        # 总结
        self._show_summary()
    
    def _show_summary(self):
        """显示总结"""
        print("\n" + "="*60)
        print("📊 实 验 总 结")
        print("="*60)
        
        print(f"🧪 实验: {self.current_exp['name']}")
        print(f"⏱️  步数: {self.step}")
        print(f"📝 记录: {len(self.records)} 条")
        
        # 统计
        actions = {}
        for r in self.records:
            if r["event"] == "action":
                act = r["data"].get("action", "unknown")
                actions[act] = actions.get(act, 0) + 1
        
        if actions:
            print("\n🔧 操作统计:")
            for act, count in actions.items():
                print(f"  {act}: {count}