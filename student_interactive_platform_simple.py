#!/usr/bin/env python3
"""
简化版学生交互式化学实验平台
基于 soc_chem_dia_refactored.py 改造
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import re

# 添加路径以便导入原有模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尝试导入核心组件
try:
    from soc_chem_dia_refactored import ChemSimEngine
    print("✅ 成功导入物理引擎")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保 soc_chem_dia_refactored.py 在同一目录下")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.WARNING)  # 减少日志输出
logger = logging.getLogger("StudentPlatform")

class SimpleStudentPlatform:
    """
    简化版学生交互平台
    特点：
    1. 极简设计，减少bug
    2. 支持自然语言和XDL命令
    3. 实时显示状态
    4. 自动记录操作
    """
    
    def __init__(self):
        """初始化平台"""
        self.session_id = f"session_{int(time.time())}"
        self.operations = []
        
        # 简单实验配置
        self.config = {
            "title": "化学实验平台",
            "hardware": [
                {"id": "tube1", "type": "tube", "name": "试管1"},
                {"id": "beaker1", "type": "beaker", "name": "烧杯1"},
                {"id": "burner1", "type": "heater", "name": "酒精灯"},
                {"id": "stand1", "type": "stand", "name": "铁架台"},
                {"id": "clamp1", "type": "clamp", "name": "试管夹"}
            ],
            "reagents": {
                "water": "H2O",
                "KMnO4": "KMnO4",
                "H2O": "H2O"
            }
        }
        
        # 初始化引擎
        self.engine = self._init_engine()
        
        # 记录文件
        self.record_file = f"student_log_{self.session_id}.jsonl"
        self._save_metadata()
        
        print(f"🎯 平台已就绪 | 会话: {self.session_id}")
        print(f"📁 记录文件: {self.record_file}")
    
    def _init_engine(self):
        """初始化模拟引擎"""
        try:
            engine = ChemSimEngine(
                hardware_config=self.config["hardware"],
                reagent_map=self.config["reagents"],
                clumsiness_level="low",
                logger_name="StudentEngine",
                silent_mode=True  # 静默模式，减少输出
            )
            return engine
        except Exception as e:
            print(f"❌ 引擎初始化失败: {e}")
            raise
    
    def _save_metadata(self):
        """保存会话元数据"""
        try:
            with open(self.record_file, 'w') as f:
                meta = {
                    "type": "session_start",
                    "time": datetime.now().isoformat(),
                    "session": self.session_id,
                    "experiment": self.config["title"]
                }
                f.write(json.dumps(meta) + "\n")
        except:
            pass  # 如果保存失败，继续运行
    
    def _save_operation(self, cmd: str, result: dict):
        """保存操作记录"""
        try:
            record = {
                "type": "operation",
                "time": datetime.now().isoformat(),
                "command": cmd,
                "result": result
            }
            with open(self.record_file, 'a') as f:
                f.write(json.dumps(record) + "\n")
        except:
            pass
    
    def show_status(self):
        """显示当前状态"""
        print("\n" + "="*50)
        print("📊 实 验 状 态")
        print("="*50)
        
        try:
            snap = self.engine.get_snapshot()
            items = snap.get("hardware", {})
            
            # 显示器材
            print("\n🧪 器 材:")
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
                    chems = []
                    for chem, amt in contents.items():
                        if amt > 0.01:
                            chems.append(f"{chem}:{amt:.2f}")
                    if chems:
                        info.append(f"内容:{','.join(chems)}")
                
                # 体积
                vol = data.get("volume_ml", 0)
                if vol > 0.1:
                    info.append(f"{vol}ml")
                
                # 状态
                if data.get("is_sealed"):
                    info.append("密封")
                if data.get("is_on", False):
                    info.append("加热中")
                
                status = " | ".join(info) if info else "空"
                print(f"  {name}: {status}")
            
            # 连接
            links = snap.get("topology", [])
            if links:
                print("\n🔗 连 接:")
                for link in links:
                    print(f"  {link['child']} → {link['parent']}")
            
            # 操作计数
            print(f"\n📈 操 作: {len(self.operations)} 次")
            
        except Exception as e:
            print(f"❌ 状态获取失败: {e}")
        
        print("="*50)
    
    def parse_command(self, cmd: str) -> Optional[dict]:
        """解析命令"""
        cmd = cmd.strip()
        if not cmd:
            return None
        
        # 1. 直接JSON命令
        if cmd.startswith("{") and cmd.endswith("}"):
            try:
                action = json.loads(cmd)
                if "action" in action:
                    return action
            except:
                pass
        
        # 2. 自然语言命令
        cmd_lower = cmd.lower()
        
        # 加料
        if "加入" in cmd or "add" in cmd_lower:
            return self._make_add(cmd)
        
        # 加热
        elif "加热" in cmd or "heat" in cmd_lower:
            return self._make_heat(cmd)
        
        # 连接
        elif "连接" in cmd or "attach" in cmd_lower:
            return self._make_attach(cmd)
        
        # 转移
        elif "倒入" in cmd or "transfer" in cmd_lower:
            return self._make_transfer(cmd)
        
        # 等待
        elif "等待" in cmd or "wait" in cmd_lower:
            return self._make_wait(cmd)
        
        # 搅拌
        elif "搅拌" in cmd or "stir" in cmd_lower:
            return self._make_stir(cmd)
        
        else:
            print("❓ 无法识别，请用: 加入/加热/连接/倒入/等待/搅拌 或 JSON格式")
            return None
    
    def _make_add(self, cmd: str) -> dict:
        """创建加料命令"""
        # 简单匹配: 向[容器]加入[试剂][量]
        match = re.search(r'向(.+?)加入(.+?)(\d+\.?\d*)(ml|g)?', cmd)
        if match:
            vessel, reagent, amount, unit = match.groups()
            unit = unit or "ml"
            return {
                "action": "Add",
                "vessel": vessel.strip(),
                "reagent": reagent.strip(),
                "volume" if unit == "ml" else "mass": f"{amount}{unit}"
            }
        
        # 默认
        return {"action": "Add", "vessel": "tube1", "reagent": "water", "volume": "10ml"}
    
    def _make_heat(self, cmd: str) -> dict:
        """创建加热命令"""
        match = re.search(r'加热(.+?)(?:到(\d+))?', cmd)
        if match:
            vessel, temp = match.groups()
            action = {"action": "Heat", "vessel": vessel.strip()}
            if temp:
                action["temperature"] = temp
            return action
        
        return {"action": "Heat", "vessel": "tube1"}
    
    def _make_attach(self, cmd: str) -> dict:
        """创建连接命令"""
        match = re.search(r'将(.+?)连接到(.+?)', cmd)
        if match:
            child, parent = match.groups()
            return {
                "action": "Attach",
                "vessel": child.strip(),
                "support": parent.strip()
            }
        
        return {"action": "Attach", "vessel": "clamp1", "support": "stand1"}
    
    def _make_transfer(self, cmd: str) -> dict:
        """创建转移命令"""
        match = re.search(r'从(.+?)倒入(.+?)(\d+ml)?', cmd)
        if match:
            src, dst, vol = match.groups()
            action = {
                "action": "Transfer",
                "from_vessel": src.strip(),
                "to_vessel": dst.strip()
            }
            if vol:
                action["volume"] = vol
            return action
        
        return {"action": "Transfer", "from_vessel": "beaker1", "to_vessel": "tube1"}
    
    def _make_wait(self, cmd: str) -> dict:
        """创建等待命令"""
        match = re.search(r'等待(\d+)', cmd)
        if match:
            sec = match.group(1)
            return {"action": "Wait", "duration": sec}
        
        return {"action": "Wait", "duration": "5"}
    
    def _make_stir(self, cmd: str) -> dict:
        """创建搅拌命令"""
        match = re.search(r'搅拌(.+)', cmd)
        if match:
            vessel = match.group(1).strip()
            return {"action": "Stir", "vessel": vessel}
        
        return {"action": "Stir", "vessel": "tube1"}
    
    def execute(self, action: dict) -> dict:
        """执行命令"""
        if not action or "action" not in action:
            return {"ok": False, "msg": "无效命令"}
        
        try:
            # 执行
            msg, _ = self.engine.execute(action)
            
            # 记录
            self.operations.append({
                "action": action["action"],
                "time": time.time(),
                "success": True
            })
            
            return {"ok": True, "msg": msg}
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ 执行错误: {error_msg}")
            return {"ok": False, "msg": error_msg}
    
    def run(self):
        """运行主循环"""
        print("\n" + "="*50)
        print("🧪 化 学 实 验 平 台")
        print("="*50)
        print("输入命令开始实验:")
        print("  - 自然语言: '向试管1加入水10ml', '加热试管1'")
        print("  - JSON格式: '{\"action\":\"Add\",\"vessel\":\"tube1\",\"reagent\":\"water\",\"volume\":\"10ml\"}'")
        print("  - 状态: '状态'")
        print("  - 帮助: '帮助'")
        print("  - 退出: '退出'")
        print("="*50)
        
        # 初始状态
        self.show_status()
        
        step = 0
        while True:
            step += 1
            print(f"\n🔄 第{step}步 > ", end="")
            
            try:
                cmd = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 实验结束")
                break
            
            # 特殊命令
            if cmd in ["退出", "exit", "quit", "q"]:
                print("👋 再见!")
                break
            elif cmd in ["状态", "status", "s"]:
                self.show_status()
                continue
            elif cmd in ["帮助", "help", "h", "?"]:
                self._show_help()
                continue
            elif cmd in ["历史", "history"]:
                self._show_history()
                continue
            
            # 解析和执行
            action = self.parse_command(cmd)
            if not action:
                continue
            
            print(f"🔧 执行: {json.dumps(action, ensure_ascii=False)}")
            
            result = self.execute(action)
            if result["ok"]:
                print(f"✅ 成功: {result['msg']}")
            else:
                print(f"❌ 失败: {result['msg']}")
            
            # 保存记录
            self._save_operation(cmd, result)
            
            # 显示新状态
            self.show_status()
    
    def _show_help(self):
        """显示帮助"""
        print("\n📖 命 令 帮 助")
        print("="*30)
        print("加料: 向试管1加入水10ml")
        print("加热: 加热试管1 或 加热试管1到100")
        print("连接: 将试管夹连接到铁架台")
        print("转移: 从烧杯1倒入试管1 20ml")
        print("等待: 等待10秒")
        print("搅拌: 搅拌试管1")
        print("\nJSON示例:")
        print('  {"action":"Add","vessel":"tube1","reagent":"water","volume":"10ml"}')
        print('  {"action":"Heat","vessel":"tube1","temperature":"100"}')
        print("="*30)
    
    def _show_history(self):
        """显示历史"""
        if not self.operations:
            print("📭 暂无操作历史")
            return
        
        print("\n📜 操 作 历 史")
        print("="*30)
        for i, op in enumerate(self.operations[-10:], 1):
            status = "✅" if op.get("success") else "❌"
            print(f"{i}. {status} {op.get('action', '未知')}")
        print("="*30)


def main():
    """主函数"""
    print("🧪 化学实验学生平台 v1.0")
    
    try:
        platform = SimpleStudentPlatform()
        platform.run()
        
        print(f"\n📊 实 验 总 结")
        print(f"  操作次数: {len(platform.operations)}")
        print(f"  记录文件: {platform.record_file}")
        print("🎉 实验完成!")
        
    except Exception as e:
        print(f"❌ 平台错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())