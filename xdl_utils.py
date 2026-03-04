import xml.etree.ElementTree as ET
import os
import re  # <--- [新增] 用于提取数字
from typing import Dict, Any, List, Tuple

class XDLParser:
    """
    XDL (XML Description Language) 解析器
    职责：将 .xdl XML 文件转化为 Python 字典配置
    """

    @staticmethod
    def parse_xdl(filepath: str) -> Dict[str, Any]:
        """
        解析 XDL 文件的主入口
        :param filepath: .xdl 文件的路径
        :return: 包含实验完整配置的字典
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"XDL file not found: {filepath}")

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"XML Syntax Error in {filepath}: {e}")

        # 1. 解析 Metadata
        meta_node = root.find("Metadata")
        metadata = {}
        if meta_node is not None:
            metadata = {
                "title": meta_node.attrib.get("title", "未命名实验"),
                "goal": meta_node.attrib.get("goal", "无明确目标"),
                "difficulty": meta_node.attrib.get("difficulty", "Low")
            }

        # 2. 定位 Synthesis 节点
        synthesis = root.find("Synthesis")
        if synthesis is None:
            raise ValueError("Invalid XDL: Missing <Synthesis> block.")

        # 3. 解析试剂 (核心升级：包含 Mapping)
        reagents_node = synthesis.find("Reagents")
        reagents_data, reagent_map = XDLParser._parse_reagents(reagents_node)

        # 4. 解析硬件
        hardware_node = synthesis.find("Hardware")
        hardware_data = XDLParser._parse_hardware(hardware_node)

        # 5. 解析步骤
        procedure_node = synthesis.find("Procedure")
        procedure_data = XDLParser._parse_procedure(procedure_node)

        # 6. 组装最终 Config 对象
        return {
            # 顶层快捷字段
            "title": metadata.get("title"),
            "goal": metadata.get("goal"),
            
            # 完整数据块
            "metadata": metadata,
            "reagents": reagents_data,
            "reagent_map": reagent_map,  # <--- [重点] 动态映射表
            "hardware": hardware_data,
            "procedure": procedure_data
        }

    @staticmethod
    def _parse_reagents(node: ET.Element) -> Tuple[List[Dict], Dict[str, str]]:
        """
        解析试剂节点，并构建 name -> formula 的映射表
        """
        reagents_list = []
        reagent_map = {}

        if node is None:
            return reagents_list, reagent_map

        for r in node.findall("Reagent"):
            name = r.attrib.get("name")
            if not name: continue

            # [关键逻辑] 提取 formula，如果未定义，默认等于 name
            # 这允许 XDL 写法灵活：<Reagent name="铁钉" formula="Fe" />
            formula = r.attrib.get("formula", name)
            
            desc = r.attrib.get("description", "")
            state = r.attrib.get("state", "liquid")

            reagents_list.append({
                "name": name,
                "formula": formula,
                "state": state,
                "desc": desc
            })

            # 构建映射：剧本名 -> 化学式
            reagent_map[name] = formula

        return reagents_list, reagent_map

    @staticmethod
    def _parse_hardware(node: ET.Element) -> List[Dict]:
        """
        解析硬件节点
        """
        hardware_list = []
        if node is None:
            return hardware_list

        for c in node.findall("Component"):
            c_id = c.attrib.get("id")
            if not c_id: continue

            item = {
                "id": c_id,
                "type": c.attrib.get("type", "container"),
                "capacity": c.attrib.get("capacity", "100ml"),
                "desc": c.attrib.get("description", "")
            }
            
            # 支持预置物 (例如预先装了水的烧杯)
            if "contains" in c.attrib:
                item["contains"] = c.attrib["contains"]
                item["vol"] = float(c.attrib.get("vol", 0.0))

            hardware_list.append(item)

        # 添加默认硬件（如量筒，药匙，滴管等）
        # <Component id="test_tube_1" type="test_tube" capacity="50ml"/>
        # <Component id="marble_chunks" type="solid_holder"/>
        # <Component id="rubber_stop_1" type="rubber_stop"/>
        default_hardware = [
            {"id": "measuring_cylinder_100ml", "type": "measuring_cylinder", "capacity": "100ml", "desc": "100ml 量筒"},
            {"id": "spatula", "type": "spatula", "desc": "药匙"},
            {"id": "dropper", "type": "dropper", "desc": "滴管"},
        ]
        hardware_list.extend(default_hardware)
            
        return hardware_list

    @staticmethod
    def _parse_procedure(node: ET.Element) -> List[Dict]:
        """
        解析步骤节点，将 XML 标签转换为 Action 字典
        """
        steps = []
        if node is None:
            return steps

        for child in node:
            action_type = child.tag # 例如 "Add", "Transfer", "Heat"

            # === [修改] 过滤 Wait 操作 ===
            if action_type == "Wait":
                continue 
            # ==========================
            params = child.attrib.copy() # XML 属性即参数
            
            # === [修改] 增强数值解析逻辑 ===
            for k, v in params.items():
                if k in ["volume", "temp", "time", "mass", "amount"]:
                    # 1. 尝试直接转换
                    try:
                        params[k] = float(v)
                    except ValueError:
                        # 2. 如果失败（例如 "0 mL"），尝试提取数字
                        if isinstance(v, str):
                            # 匹配开头的数字部分 (支持整数、小数、负数)
                            match = re.match(r"([-+]?\d*\.?\d+)", v.strip())
                            if match:
                                params[k] = float(match.group(1))
                            # 否则保持原样（可能是变量名等）
            # ==============================

            # === [修改] 零值/空值 自动填充默认值 ===
            # 现在 params['volume'] 已经被转成了 float 0.0，即使原文本是 "0 mL"
            
            if action_type == "Add":
                vol = params.get("volume", 0.0)
                mass = params.get("mass", 0.0)
                
                # 检查是否有效 (判定阈值)
                is_vol_zero = (isinstance(vol, (int, float)) and vol <= 1e-6)
                is_mass_zero = (isinstance(mass, (int, float)) and mass <= 1e-6)

                # 如果 volume 和 mass 都几乎为 0
                if is_vol_zero and is_mass_zero:
                     params["volume"] = 5.0 # Add 默认 5.0
            
            elif action_type == "Transfer":
                vol = params.get("volume", 0.0)
                
                # 检查是否有效
                if isinstance(vol, (int, float)) and vol <= 1e-6:
                    params["volume"] = 10.0 # Transfer 默认 10.0
            # ======================================

            # 获取或生成描述
            # 优先使用 XDL 里的 description，如果没有，自动生成
            desc = params.pop("description", None)
            if not desc:
                desc = XDLParser._generate_description(action_type, params)

            step_data = {
                "action": action_type,
                "params": params,
                "desc": desc,
                "is_virtual": False # 标记这是物理步骤，而非 Teacher 动态生成的辅导步骤
            }
            steps.append(step_data)

        return steps

    @staticmethod
    def _generate_description(action: str, params: Dict) -> str:
        """
        [增强版] 覆盖所有物理引擎支持的动作
        """
        vessel = params.get('vessel', '?')
        reagent = params.get('reagent', '?')

        # === 核心动作 ===
        if action == "Add":
            if "mass" in params:
                return f"向 {vessel} 中加入 {params['mass']} {reagent}"
            if "amount" in params: # 兼容旧写法
                return f"向 {vessel} 中加入 {params['amount']} {reagent}"
            vol = params.get('volume', '?')
            return f"向 {vessel} 中加入 {vol}ml {reagent}"
        
        elif action == "Transfer":
            vol = params.get('volume', '?')
            src = params.get('from_vessel', '?')
            dst = params.get('to_vessel', '?')
            return f"将 {vol}ml 液体从 {src} 转移至 {dst}"
        
        # === 硬件操作 ===
        elif action == "Attach":
            return f"将 {vessel} 固定到 {params.get('support', '?')} 上"
        
        elif action == "Insert":
            obj = params.get('object') or params.get('tool')
            target = params.get('target') or params.get('vessel')
            return f"将 {obj} 插入 {target} 中"
            
        elif action == "Stir":
            return f"搅拌 {vessel} 中的溶液"
            
        elif action == "Heat":
            target_temp = params.get("target_temperature", "")
            desc = f"至 {target_temp}°C" if target_temp else ""
            return f"加热容器 {vessel} {desc}"
            
        elif action == "Cool":
            return f"冷却容器 {vessel}"
            
        elif action == "MeasureTemperature":
            return f"测量 {vessel} 的温度"
            
        elif action == "MeasureMass":
            # 可能是测容器，也可能是测药品
            target = vessel if vessel != '?' else reagent
            return f"称量 {target} 的质量"

        elif action == "CollectGas":
            return f"利用 {params.get('collector', '?')} 收集 {params.get('source_vessel', '?')} 产生的气体"
            
        elif action == "Filter":
            return f"过滤 {params.get('from_vessel', '?')} 中的液体至 {params.get('to_vessel', '?')}"

        # === 最后的兜底 ===
        return f"执行操作: {action}"