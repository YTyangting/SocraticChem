#!/usr/bin/env python3
"""
完整版教学平台：实验选择 + 双输入 + 状态显示 + 苏格拉底引导
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
    """苏格拉底式引导"""
    
    def __init__(self):
        self.questions = {
            "start": [
                "你为什么要进行这个操作？目的是什么？",
                "操作前需要考虑哪些安全事项？",
                "你预计会看到什么现象？为什么？"
            ],
            "middle": [
                "如何确保操作的准确性？",
                "你观察到了什么现象？这说明了什么？",
                "操作的关键控制点是什么？"
            ],
            "end": [
                "操作结果符合预期吗？为什么？",
                "如果重做，你会改进什么？",
                "这个操作背后的化学原理是什么？"
            ]
        }
        
        self.feedbacks = {
            "good": ["很好！", "不错！", "思考全面！"],
            "improve": ["注意安全！", "可以优化！", "再思考一下！"]
        }
    
    def get_guide(self, step: int, action: str, success: bool) -> str:
        """获取引导"""
        # 阶段
        if step <= 2:
            stage = "start"
        elif step <= 5:
            stage = "middle"
        else:
            stage = "end"
        
        # 问题
        question = random.choice(self.questions[stage])
        
        # 个性化
        if "heat" in action.lower():
            question += " 注意加热安全！"
        elif "add" in action.lower():
            question += " 注意试剂用量！"
        
        # 反馈
        feedback_type = "good" if success else "improve"
        feedback = random.choice(self.feedbacks[feedback_type])
        
        return f"{feedback}\n\n💭 {question}"

class TeachingPlatform:
    """教学平台"""
    
    def __init__(self):
        self.session_id = f"teach_{int(time.time())}"
        self.step = 0
        self.exp = None
        self.logs = []
        
        # 组件
        self.guide = SocratesGuide()
        self.engine = None
        
        # 实验
        self.exps = {
            "1": {
                "name": "制取氧气",
                "setup": [
                    {"id": "tube1", "type": "tube", "name": "试管"},
                    {"id": "heater1", "type": "heater", "name": "酒精灯"},
                    {"id": "stand1", "type": "stand", "name": "铁架台"},
                    {"id": "clamp1", "type": "clamp", "name": "试管夹"}
                ],
                "chems": {"KMnO4": "KMnO4"}
            },
            "2": {
                "name": "中和反应",
                "setup": [
                    {"id": "beaker1", "type": "beaker", "name": "烧杯"},
                    {"id": "dropper1", "type": "dropper", "name": "滴管"}
                ],
                "chems": {"HCl": "HCl", "NaOH": "NaOH"}
            },
            "3": {
                "name": "电解水",
                "setup": [
                    {"id": "cell1", "type": "cell", "name": "电解槽"},
                    {"id": "power1", "type": "power", "name": "电源"}
                ],
                "chems": {"water": "H2O"}
            }
        }
        
        # 日志
        self.log_file = f"teach_{self.session_id}.jsonl"
        self._init_log()
        
        print(f"🎓 平台就绪 | ID: {self.session_id}")
    
    def _init_log(self):
        """初始化日志"""
        try:
            with open(self.log_file, 'w') as f:
                start = {
                    "type": "start",
                    "id": self.session_id,
                    "time": datetime.now().isoformat()
                }
                f.write(json.dumps(start) + "\n")
        except:
            pass
    
    def _log(self, type: str, data: dict):
        """记录"""
        record = {
            "type": type,
            "step": self.step,
            "time": datetime.now().isoformat(),
            "data": data
        }
        self.logs.append(record)
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(record) + "\n")
        except:
            pass
    
    def choose_exp(self):
        """选择实验"""
        print("\n" + "="*60)
        print("🔬 选择实验")
        print("="*60)
        
        for num, exp in self.exps.items():
            print(f"\n【{num}】 {exp['name']}")
        
        print("\n" + "="*60)
        
        while True:
            choice = input("输入编号 (1/2/3): ").strip()
            if choice in self.exps:
                self.exp = self.exps[choice]
                break
            print("❌ 无效")
        
        print(f"\n✅ 已选择: {self.exp['name']}")
        
        # 初始化引擎
        try:
            self.engine = ChemSimEngine(
                hardware_config=self.exp["setup"],
                reagent_map=self.exp["chems"],
                silent_mode=True
            )
            print("✅ 环境就绪")
        except Exception as e:
            print(f"❌ 引擎失败: {e}")
            raise
        
        self._log("exp_chosen", {"name": self.exp["name"]})
        
        return self.exp
    
    def show_state(self):
        """显示状态"""
        if not self.engine:
            return
        
        print("\n" + "="*60)
        print("📊 状 态")
        print("="*60)
        
        try:
            snap = self.engine.get_snapshot()
            items = snap.get("hardware", {})
            
            print("\n🧪 器材:")
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
                print("\n🔗 连接:")
                for link in links:
                    print(f"  {link['child']} → {link['parent']}")
            
            print(f"\n📈 进度: 第{self.step}步")
            
        except Exception as e:
            print(f"❌ 状态错误: {e}")
        
        print("="*60)
    
    def get_intent(self) -> str:
        """获取意图"""
        print("\n💬 【描述意图】")
        print("你想做什么？")
        print("输入: ", end="")
        
        try:
            text = input().strip()
            while not text:
                print("请描述: ", end="")
                text = input().strip()
            return text
        except:
            return "退出"
    
    def get_action(self) -> Optional[dict]:
        """获取动作"""
        print("\n🔧 【执行操作】")
        print("输入XDL命令 (JSON)")
        print("例: {\"action\":\"Attach\",\"vessel\":\"clamp1\",\"support\":\"stand1\"}")
        print("输入: ", end="")
        
        try:
            cmd = input().strip()
            
            if cmd.lower() in ['退出', 'exit']:
                return {"action": "Exit"}
            
            # 解析JSON
            if cmd.startswith("{") and cmd.endswith("}"):
                try:
                    action = json.loads(cmd)
                    if "action" in action:
                        return action
                    else:
                        print("❌ 缺少action")
                        return None
                except:
                    print("❌ JSON错误")
                    return None
            else:
                print("❌ 需要JSON")
                return None
                
        except:
            return {"action": "Exit"}
    
    def execute(self, action: dict) -> dict:
        """执行"""
        if not action or "action" not in action:
            return {"ok": False, "msg": "无效"}
        
        if action["action"] == "Exit":
            return {"ok": True, "msg": "退出", "exit": True}
        
        try:
            result, _ = self.engine.execute(action)
            return {"ok": True, "msg": result}
        except Exception as e:
            return {"ok": False, "msg": str(e)}
    
    def run(self):
        """运行"""
        print("\n" + "="*60)
        print("🎓 化学实验教学")
        print("="*60)
        
        # 1. 选择实验
        self.choose_exp()
        
        print("\n✅ 准备完成！开始实验。")
        
        # 主循环
        while True:
            self.step += 1
            print(f"\n{'='*60}")
            print(f"🔄 第 {self.step} 步")
            print('='*60)
            
            # 2. 意图
            print("\n【意图】")
            intent = self.get_intent()
            if intent == "退出":
                break
            
            self._log("intent", {"text": intent})
            
            # 3. 动作
            print("\n【动作】")
            action = self.get_action()
            if not action:
                print("⚠️  无效")
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
            
            # 5. 状态
            print("\n【状态】")
            self.show_state()
            
            # 6. 引导
            print("\n【引导】")
            guide = self.guide.get_guide(
                step=self.step,
                action=action.get("action", ""),
                success=result["ok"]
            )
            print(guide)
            
            self._log("guide", {"text": guide})
            
            # 继续？
            print("\n" + "-"*40)
            cont = input("继续？(y/n): ").lower()
            if cont not in ['y', 'yes', '是']:
                break
        
        # 总结
        print("\n" + "="*60)
        print("📊 总 结")
        print("="*60)
        print(f"实验: {self.exp['name']}")
        print(f"步数: {self.step}")
        print(f"日志: {self.log_file}")
        print("🎉 完成！")
        print("="*60)

def main():
    """主函数"""
    print("🎓 化学实验教学平台")
    
    try:
        platform = TeachingPlatform()
        platform.run()
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())