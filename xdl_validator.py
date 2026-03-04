import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

class XDLVerifyError:
    def __init__(self, step: str, errors: List[str]):
        self.step = step
        self.errors = errors
    def __repr__(self):
        return f"[Step: {self.step}] Errors: {', '.join(self.errors)}"

class XDLValidator:
    MANDATORY_PROPERTIES = {
        "Attach": ["vessel", "support"],
        "Insert": ["tool", "vessel"],
        "Add": ["vessel", "reagent"], 
        "Transfer": ["from_vessel", "to_vessel", "volume"],
        "Heat": ["vessel"],
        "Cool": ["vessel"],
        "Stir": ["vessel"],
        "Wait": ["time"],
        "MeasureTemperature": ["vessel"],
        "MeasureMass": [],
        "Filter": ["from_vessel", "to_vessel"],
        "CollectGas": ["source_vessel", "collector"]
    }

    @staticmethod
    def verify_xdl(xdl_string: str) -> List[XDLVerifyError]:
        errors = []
        try:
            clean_xdl = xdl_string.replace("```xml", "").replace("```", "").strip()
            if "<?xml" not in clean_xdl:
                clean_xdl = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + clean_xdl
            root = ET.fromstring(clean_xdl)
        except ET.ParseError as e:
            return [XDLVerifyError("XML Parse", [f"Invalid XML syntax: {e}"])]
        
        # === [核心修复] 严格结构检查 ===
        # 1. 检查根节点下是否有 Metadata
        metadata = root.find("Metadata")
        if metadata is None:
            return [XDLVerifyError("Structure", ["Missing <Metadata> section"])]
        
        # 2. 检查根节点下是否有 Synthesis (必须是直接子节点)
        synthesis = root.find("Synthesis")
        if synthesis is None:
            # 这是一个致命结构错误，直接返回，不再继续检查内部
            return [XDLVerifyError("Structure", ["Missing <Synthesis> block. Hardware/Reagents/Procedure must be inside <Synthesis>."])]

        # === 之后的检查都基于 synthesis 节点进行 ===
        # 这样确保了层级正确
        
        # Metadata 属性检查
        meta_errors = []
        for attr in ["title", "goal"]:
            if not metadata.get(attr):
                meta_errors.append(f"Metadata missing attribute '{attr}'")
        if meta_errors:
             errors.append(XDLVerifyError("Metadata", meta_errors))

        # 传入 synthesis 节点而不是 root，防止跨层级查找
        defined_hw_ids, hw_errors = XDLValidator._parse_hardware(synthesis)
        errors.extend(hw_errors)

        defined_reagents_info, r_errors = XDLValidator._parse_reagents(synthesis)
        errors.extend(r_errors)

        proc_errors = XDLValidator._verify_procedure(synthesis, defined_hw_ids, defined_reagents_info)
        errors.extend(proc_errors)

        return errors

    @staticmethod
    def _parse_hardware(parent_node):
        hw_ids = set()
        errors = []
        # 直接查找子节点
        hardware_node = parent_node.find("Hardware")
        if hardware_node is None:
            return set(), [XDLVerifyError("Hardware", ["Missing <Hardware> section inside <Synthesis>"])]
        
        for comp in hardware_node:
            errs = []
            if comp.tag != "Component": continue 
            cid = comp.get("id")
            if not cid: errs.append("Component missing 'id'")
            else: hw_ids.add(cid)
            if errs: errors.append(XDLVerifyError(ET.tostring(comp, encoding='unicode'), errs))
        return hw_ids, errors

    @staticmethod
    def _parse_reagents(parent_node):
        r_info = {} 
        errors = []
        reagents_node = parent_node.find("Reagents")
        
        if reagents_node is None:
            return {}, [XDLVerifyError("Reagents", ["Missing <Reagents> section inside <Synthesis>"])]

        for r in reagents_node:
            errs = []
            name = r.get("name")
            state = r.get("state")
            
            if not name: errs.append("Reagent missing 'name'")
            if not state: errs.append("Reagent missing 'state'")
            
            if name and state:
                r_info[name] = state.lower()

            if errs:
                errors.append(XDLVerifyError(ET.tostring(r, encoding='unicode'), errs))

        return r_info, errors

    @staticmethod
    def _verify_procedure(parent_node, defined_hw_ids, defined_reagents_info):
        errors = []
        proc_node = parent_node.find("Procedure")
        if proc_node is None:
            return [XDLVerifyError("Procedure", ["Missing <Procedure> section inside <Synthesis>"])]

        for step in proc_node:
            if step.tag is ET.Comment: continue
            action = step.tag
            errs = []
            
            if action in XDLValidator.MANDATORY_PROPERTIES:
                for prop in XDLValidator.MANDATORY_PROPERTIES[action]:
                    val = step.get(prop)
                    if not val:
                        errs.append(f"Action '{action}' missing attribute '{prop}'")
                        continue
                    if prop in ["vessel", "from_vessel", "to_vessel", "tool", "support", "source_vessel", "collector"]:
                        if val not in defined_hw_ids:
                            errs.append(f"Undefined Hardware ID '{val}'")
                    if prop == "reagent":
                        if val not in defined_reagents_info:
                            errs.append(f"Undefined Reagent '{val}'")

            if action == "Add":
                r_name = step.get("reagent")
                if r_name and r_name in defined_reagents_info:
                    r_state = defined_reagents_info[r_name]
                    is_solid = "solid" in r_state or "powder" in r_state
                    if is_solid:
                        if not step.get("mass"): errs.append(f"Adding SOLID '{r_name}' requires 'mass'.")
                    else:
                        if not step.get("volume"): errs.append(f"Adding LIQUID '{r_name}' requires 'volume'.")

            if action == "Refill":
                r_name = step.get("reagent")
                if r_name and r_name not in defined_reagents_info:
                     errs.append(f"Refill uses undefined reagent '{r_name}'")

            if action == "MeasureMass":
                if not step.get("vessel") and not step.get("reagent"):
                    errs.append("MeasureMass requires 'vessel' or 'reagent'.")

            if errs:
                errors.append(XDLVerifyError(ET.tostring(step, encoding='unicode').strip(), errs))
        
        return errors