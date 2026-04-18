#!/usr/bin/env python3
"""
学生交互式化学实验平台
基于 soc_chem_dia_refactored.py 改造，让真实学生手动操作
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import re

# 添加路径以便导入原有模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入原有框架的核心组件
try:
    # 这里我们只导入必要的组件，避免复杂的依赖
    from soc_chem_dia_refactored import (
        ChemSimEngine, ObservationEngine, HardwareManager,
        EquipmentFactory, Vessel, Heater, LabObject,
        parse_quantity, CHEM_DB, REACTIONS_DB
    )
    print("✅ 成功导入原有框架组件")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保 soc_chem_dia_refactored.py 在同一目录下")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StudentPlatform")

class StudentInteractivePlatform:
    """
    学生交互式平台
    核心功能：
    1. 显示当前实验状态
    2. 接收学生输入（自然语言或XDL命令）
    3. 执行操作并反馈结果
    4. 记录实验过程
    """
    
    def __init__(self, xdl_config_path: str):
        """
        初始化平台
        Args:
            xdl_config_path: XDL实验配置文件路径
        """
        self.xdl_path = xdl_config_path
        self.session_id = f"student_session_{int(time.time())}"
        self.operation_history = []
        
        # 实验配置
        self.exp_config = self._load_experiment_config()
        
        # 初始化物理引擎
        self.sim_engine = self._init_simulation_engine()
        
        # 任务追踪
        self.current_task = None
        self.task_progress = {}
        
        # 记录文件
        self.record_file = f"student_records_{self.session_id}.jsonl"
        self._init_record_file()
        
        print(f"🎯 学生交互平台已初始化 (会话ID: {self.session_id})")
        print(f"📁 实验记录将保存到: {self.record_file}")
    
    def _load_experiment_config(self) -> Dict:
        """加载实验配置（简化版）"""
        # 这里简化处理，实际应该解析XDL文件
        # 为了快速实现，我们使用硬编码的配置
        return {
            "title": "高锰酸钾制取氧气实验",
            "hardware": [
                {"id": "test_tube_1", "type": "tube", "name": "试管1", "capacity": "50ml"},
                {"id": "beaker_1", "type": "beaker", "name": "烧杯1", "capacity": "100ml"},
                {"id": "alcohol_burner", "type": "heater", "name": "酒精灯"},
                {"id": "iron_stand", "type": "stand", "name": "铁架台"},
                {"id": "test_tube_clamp", "type": "clamp", "name": "试管夹"}
            ],
            "reagent_map": {
                "KMnO4": "KMnO4",
                "water": "H2O",
                "H2O": "H2O",
                "potassium_permanganate": "KMnO4"
            },
            "initial_setup": [
                {"action": "Attach", "vessel": "test_tube_clamp", "support": "iron_stand"},
                {"action": "Attach", "vessel": "test_tube_1", "support": "test_tube_clamp"}
            ]
        }
    
    def _init_simulation_engine(self) -> ChemSimEngine:
        """初始化物理模拟引擎"""
        try:
            # 创建引擎实例
            engine = ChemSimEngine(
                hardware_config=self.exp_config["hardware"],
                reagent_map=self.exp_config["reagent_map"],
                clumsiness_level="low",  # 学生操作，默认低笨拙度
                logger_name="StudentEngine",
                silent_mode=False
            )
            
            # 执行初始设置
            if "initial_setup" in self.exp_config:
                for action in self.exp_config["initial_setup"]:
                    result, _ = engine.execute(action)
                    logger.info(f"初始设置: {result}")
            
            return engine
            
        except Exception as e:
            logger.error(f"初始化模拟引擎失败: {e}")
            raise
    
    def _init_record_file(self):
        """初始化记录文件"""
        try:
            with open(self.record_file, 'w', encoding='utf-8') as f:
                # 写入会话元数据
                metadata = {
                    "record_type": "session_metadata",
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "experiment_title": self.exp_config.get("title", "未知实验"),
                    "student_platform": "v1.0"
                }
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            logger.info(f"记录文件已创建: {self.record_file}")
        except Exception as e:
            logger.error(f"创建记录文件失败: {e}")
    
    def _record_operation(self, student_input: str, operation_result: Dict):
        """记录学生操作"""
        record = {
            "record_type": "student_operation",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "student_input": student_input,
            "operation_result": operation_result,
            "environment_snapshot": self.get_current_snapshot()
        }
        
        self.operation_history.append(record)
        
        # 写入文件
        try:
            with open(self.record_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"写入记录失败: {e}")
    
    def get_current_snapshot(self) -> Dict:
        """获取当前环境快照"""
        return self.sim_engine.get_snapshot()
    
    def display_current_state(self):
        """显示当前实验状态"""
        print("\n" + "="*60)
        print("📊 当前实验状态")
        print("="*60)
        
        snapshot = self.get_current_snapshot()
        hardware = snapshot.get("hardware", {})
        
        # 显示器材状态
        print("\n🧪 器材状态:")
        for obj_id, data in hardware.items():
            obj_name = data.get("name", obj_id)
            obj_type = data.get("type", "unknown")
            
            # 基础信息
            info_parts = []
            
            # 容器类显示内容物
            if obj_type == "vessel" or "tube" in str(obj_type).lower() or "beaker" in str(obj_type).lower():
                # 温度
                temp = data.get("temperature", 25.0)
                if abs(temp - 25.0) > 2.0:
                    info_parts.append(f"温度: {temp:.1f}°C")
                
                # 内容物
                contents = data.get("contents", {})
                if contents:
                    chem_list = []
                    for chem, amount in contents.items():
                        if amount > 0.001:  # 只显示显著量
                            chem_list.append(f"{chem}: {amount:.3f}mol")
                    if chem_list:
                        info_parts.append(f"内容: {', '.join(chem_list)}")
                
                # 体积
                volume = data.get("volume_ml", 0.0)
                if volume > 0.1:
                    info_parts.append(f"体积: {volume:.1f}ml")
                
                # 密封状态
                if data.get("is_sealed", False):
                    info_parts.append("🔒 已密封")
                if data.get("is_covered", False):
                    info_parts.append("🛡️ 已盖上")
            
            # 加热器类
            elif "heater" in str(obj_type).lower():
                if data.get("is_on", False):
                    info_parts.append("🔥 开启中")
                    temp = data.get("current_temp", data.get("temperature", 25.0))
                    info_parts.append(f"温度: {temp:.1f}°C")
                else:
                    info_parts.append("⚪ 关闭")
            
            # 组装信息
            info_str = " | ".join(info_parts) if info_parts else "空/默认状态"
            print(f"  - {obj_name} ({obj_id}): {info_str}")
        
        # 显示连接状态
        topology = snapshot.get("topology", [])
        if topology:
            print("\n🔗 装置连接:")
            for link in topology:
                print(f"  - {link['child']} → {link['parent']} ({link.get('type', 'mechanical')})")
        
        # 显示当前任务（如果有）
        if self.current_task:
            print(f"\n🎯 当前任务: {self.current_task}")
            if self.task_progress:
                print("📈 任务进度:")
                for task, status in self.task_progress.items():
                    print(f"  - {task}: {status}")
        
        print("="*60 + "\n")
    
    def parse_student_input(self, user_input: str) -> Optional[Dict]:
        """
        解析学生输入
        支持两种格式：
        1. 自然语言（简单解析）
        2. XDL结构化命令（JSON格式）
        """
        user_input = user_input.strip()
        if not user_input:
            return None
        
        # 尝试解析为JSON（XDL命令）
        if user_input.startswith("{") and user_input.endswith("}"):
            try:
                action = json.loads(user_input)
                if "action" in action:
                    return action
            except json.JSONDecodeError:
                pass  # 不是有效的JSON，继续尝试自然语言解析
        
        # 自然语言解析（简化版）
        return self._parse_natural_language(user_input)
    
    def _parse_natural_language(self, text: str) -> Optional[Dict]:
        """
        简单自然语言解析
        注意：这是一个简化版本，实际应用需要更复杂的NLP
        """
        text_lower = text.lower()
        
        # 加料操作
        if "加入" in text or "添加" in text or "add" in text_lower:
            return self._parse_add_action(text)
        
        # 转移操作
        elif "倒入" in text or "转移" in text or "transfer" in text_lower:
            return self._parse_transfer_action(text)
        
        # 加热操作
        elif "加热" in text or "heat" in text_lower:
            return self._parse_heat_action(text)
        
        # 连接操作
        elif "连接" in text or "固定" in text or "attach" in text_lower:
            return self._parse_attach_action(text)
        
        # 搅拌操作
        elif "搅拌" in text or "stir" in text_lower:
            return self._parse_stir_action(text)
        
        # 等待操作
        elif "等待" in text or "wait" in text_lower:
            return self._parse_wait_action(text)
        
        # 测量温度
        elif "温度" in text or "测温" in text or "measure" in text_lower:
            return self._parse_measure_temp_action(text)
        
        else:
            print("⚠️  无法识别的指令，请使用更明确的表述或XDL格式")
            return None
    
    def _parse_add_action(self, text: str) -> Dict:
        """解析加料操作"""
        # 简单正则匹配
        patterns = [
            r'向(.+?)中加入(.+?)(\d+\.?\d*)(ml|g)',
            r'向(.+?)添加(.+?)(\d+\.?\d*)(ml|g)',
            r'add (.+?) to (.+?) with (\d+\.?\d*)(ml|g)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 4:
                    vessel, reagent, amount, unit = match.groups()
                    return {
                        "action": "Add",
                        "vessel": vessel.strip(),
                        "reagent": reagent.strip(),
                        "volume" if unit.lower() == "ml" else "mass": f"{amount}{unit}"
                    }
        
        # 简化匹配
        words = text.split()
        if len(words) >= 4:
            return {
                "action": "Add",
                "vessel": words[1] if "向" in text else words[-1],
                "reagent": words[2] if "加入" in text else words[1],
                "volume": "10ml"  # 默认值
            }
        
        return {"action": "Add", "error": "无法解析加料参数"}
    
    def _parse_transfer_action(self, text: str) -> Dict:
        """解析转移操作"""
        match = re.search(r'从(.+?)倒入(.+?)(?:(\d+\.?\d*)(ml))?', text)
        if match:
            from_vessel, to_vessel, amount, unit = match.groups()
            action = {
                "action": "Transfer",
                "from_vessel": from_vessel.strip(),
                "to_vessel": to_vessel.strip()
            }
            if amount:
                action["volume"] = f"{amount}{unit or 'ml'}"
            return action
        
        return {"action": "Transfer", "error": "无法解析转移参数"}
    
    def _parse_heat_action(self, text: str) -> Dict:
        """解析加热操作"""
        match = re.search(r'加热(.+?)(?:到(\d+\.?\d*)度)?', text)
        if match:
            vessel, temp = match.groups()
            action = {"action": "Heat", "vessel": vessel.strip()}
            if temp:
                action["temperature"] = temp
            return action
        
        return {"action": "Heat", "error": "无法解析加热参数"}
    
    def _parse_attach_action(self, text: str) -> Dict:
        """解析连接操作"""
        match = re.search(r'将(.+?)(?:连接|固定)到(.+?)上', text)
        if match:
            child, parent = match.groups()
            return {
                "action": "Attach",
                "vessel": child.strip(),
                "support": parent.strip()
            }
        
        return {"action": "Attach", "error": "无法解析连接参数"}
    
    def _parse_stir_action(self, text: str) -> Dict:
        """解析搅拌操作"""
        match = re.search(r'搅拌(.+)', text)
        if match:
            vessel = match.group(1).strip()
            return {"action": "Stir", "vessel": vessel}
        
        return {"action": "Stir", "error": "无法解析搅拌参数"}
    
    def _parse_wait_action(self, text: str) -> Dict:
        """解析等待操作"""
        match = re.search(r'等待(\d+\.?\d*)(?:秒|s)?', text)
        if match:
            seconds = match.group(1)
            return {"action": "Wait", "duration": seconds}
        
        return {"action": "Wait", "duration": "5"}
    
    def _parse_measure_temp_action(self, text: str) -> Dict:
        """解析测量温度操作"""
        match = re.search(r'测量(.+?)的温度', text)
        if match:
            vessel = match.group(1).strip()
            return {"action": "MeasureTemperature", "vessel": vessel}
        
        # 如果没有指定容器，测量所有
        return {"action": "MeasureTemperature", "vessel": "all"}
    
    def execute_student_action(self, action: Dict) -> Dict:
        """
        执行学生操作并返回结果
        """
        if not action or "action" not in action:
            return {"success": False, "message": "无效的操作指令"}
        
        try:
            # 执行操作
            result_message, new_snapshot = self.sim_engine.execute(action)
            
            # 构建结果
            result = {
                "success": True,
                "action": action["action"],
                "message": result_message,
                "timestamp": datetime.now().isoformat()
            }
            
            # 添加错误信息（如果有）
            if "error" in action:
                result["warning"] = f"解析警告: {action['error']}"
            
            return result
            
        except Exception as e:
            logger.error(f"执行操作失败: {e}")
            return {
                "success": False,
                "action": action.get("action", "unknown"),
                "message": f"操作失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def run_interactive_session(self):
        """运行交互式会话"""
        print("\n" + "="*60)
        print("🧪 化学实验交互平台")
        print("="*60)
        print("欢迎使用化学实验交互平台！")
        print("你可以：")
        print("1. 输入自然语言指令（如'向试管中加入5ml水'）")
        print("2. 输入XDL格式命令（如'{\"action\": \"Add\", \"vessel\": \"test_tube_1\", \"reagent\": \"water\", \"volume\": \"5ml\"}'）")
        print("3. 输入 '状态' 查看当前实验状态")
        print("4. 输入 '帮助' 查看可用命令")
        print("5. 输入 '退出' 结束实验")
        print("="*60)
        
        # 显示初始状态
