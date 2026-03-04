import os
import sys
import glob
import json
import yaml
import logging
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Set, Literal
from pathlib import Path

# 第三方库
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from colorama import Fore, Style, init
from config import Config

# 初始化颜色和日志
init(autoreset=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AutoRegistry")

# ==========================================
# 1. 数据模型定义 (Schema Definition)
# ==========================================
# 这部分确保 LLM 生成的数据严格符合 equipment_manifest.yaml 的结构

class EquipmentConfig(BaseModel):
    """单个器材的配置模型"""
    name: str = Field(..., description="中文名称，如'试管'")
    class_category: Literal["Vessel", "Heater", "Support", "Tool"] = Field(
        ..., description="对应 Python 类: Vessel(容器), Heater(加热器), Support(支架), Tool(工具)"
    )
    tags: List[str] = Field(default_factory=list, description="通用标签, e.g. glassware, metal")
    my_tags: List[str] = Field(default_factory=list, description="自身特有标签, e.g. tube_tag")
    connectable_to: List[str] = Field(default_factory=list, description="该物体可以安装在哪些 tag 上")
    ports: List[str] = Field(default_factory=lambda: ["center"], description="物理端口列表")
    capacity: int = Field(default=0, description="作为父级时能挂载的数量，容器通常为0或导管数")

class ManifestSchema(BaseModel):
    """整体输出结构"""
    types: Dict[str, EquipmentConfig]

# ==========================================
# 2. XDL 解析器 (Extractor)
# ==========================================

class XDLExtractor:
    def __init__(self, xdl_dir: str):
        self.xdl_dir = Path(xdl_dir)

    def scan_all_types(self) -> Set[str]:
        """扫描目录下所有 XDL，提取去重的 Component type"""
        files = list(self.xdl_dir.glob("*.xdl"))
        logger.info(f"📂 Scanning {len(files)} XDL files in {self.xdl_dir}...")
        
        found_types = set()
        
        for f_path in files:
            try:
                tree = ET.parse(f_path)
                root = tree.getroot()
                # 查找 Hardware 下的所有 Component
                for comp in root.findall(".//Hardware/Component"):
                    t = comp.get("type")
                    if t:
                        found_types.add(t.strip())
            except ET.ParseError:
                logger.error(f"❌ XML Parse Error: {f_path.name}")
            except Exception as e:
                logger.error(f"❌ Failed to process {f_path.name}: {e}")
                
        logger.info(f"🔍 Found {len(found_types)} unique equipment types.")
        return found_types

# ==========================================
# 3. LLM 生成器 (Generator)
# ==========================================

class ConfigGenerator:
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @retry(
        retry=retry_if_exception_type((Exception)), # 捕获所有生成错误重试
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_configs(self, new_types: List[str]) -> Dict[str, dict]:
        """调用 LLM 为新类型生成配置"""
        if not new_types:
            return {}

        logger.info(f"🧠 Asking LLM to generate config for: {new_types}")

        system_prompt = """
        你是一个精通化学实验物理引擎的架构师。
        任务：为输入的化学器材类型列表生成对应的 YAML 配置。
        
        ### 判定规则 (Class Category)
        1. Vessel: 能盛装液体、气体或固体的容器（如：tube, beaker, flask, bottle, trough）。
        2. Heater: 能主动产生热量的设备（如：burner, alcohol_lamp, hotplate）。
        3. Support: 用于固定、支撑的被动设备（如：stand, clamp, tripod, net, rack, table）。
        4. Tool: 辅助工具、连接件、密封件（如：stopper, tubing, rod, spoon, dropper）。

        ### 属性填充指南
        - name: 准确的中文名称。
        - my_tags: 通常是 {type}_tag，例如 tube -> ["tube_tag"]。
        - connectable_to: 基于物理常识。例如 Vessel 通常连 clamp_tag 或 rack_tag；Clamp 连 stand_tag。
        - capacity: 
           - 容器(Vessel)若能插塞子/导管，capacity通常为 2-3。
           - 支架(Support)如铁架台 capacity=5，铁夹=1。
        
        ### 输出格式
        必须是合法的 JSON，符合以下 TypeScript 接口：
        interface Output {
            types: {
                [key: string]: {
                    name: string;
                    class_category: "Vessel" | "Heater" | "Support" | "Tool";
                    tags: string[];
                    my_tags: string[];
                    connectable_to: string[];
                    ports: string[];
                    capacity: number;
                }
            }
        }
        """

        user_content = f"请为以下器材生成配置: {json.dumps(new_types)}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.1 # 低温确保稳定
        )
        
        content = response.choices[0].message.content
        
        # 验证 JSON 并清洗
        try:
            data = json.loads(content)
            # 使用 Pydantic 校验数据结构完整性
            validated = ManifestSchema(**data)
            # 转回 dict，exclude_none 保持整洁
            return validated.model_dump(mode='json')['types']
        except ValidationError as e:
            logger.error(f"Schema Validation Failed: {e}")
            raise e # 触发重试
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            raise e

# ==========================================
# 4. 流程管理器 (Pipeline Manager)
# ==========================================

class RegistryManager:
    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.current_data = {"types": {}}

    def load(self):
        """加载现有的 Manifest"""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    self.current_data = yaml.safe_load(f) or {"types": {}}
                logger.info(f"✅ Loaded existing manifest with {len(self.current_data.get('types', {}))} items.")
            except Exception as e:
                logger.error(f"❌ Failed to load manifest: {e}")
                self.current_data = {"types": {}}
        else:
            logger.warning("⚠️ No existing manifest found. Starting fresh.")

    def save(self):
        """保存 Manifest"""
        # 备份旧文件
        if self.manifest_path.exists():
            backup_path = self.manifest_path.with_suffix(".yaml.bak")
            self.manifest_path.rename(backup_path)
            
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            # allow_unicode=True 确保中文正常显示
            yaml.dump(self.current_data, f, allow_unicode=True, sort_keys=True, indent=2)
        logger.info(f"💾 Manifest saved to {self.manifest_path}")

    def update(self, new_configs: Dict[str, dict]):
        """合并新配置"""
        if not new_configs:
            return
            
        if "types" not in self.current_data:
            self.current_data["types"] = {}
            
        for k, v in new_configs.items():
            # 只有当 key 不存在时才写入，保护人工修改
            if k not in self.current_data["types"]:
                self.current_data["types"][k] = v
                logger.info(Fore.GREEN + f"   + Added new equipment: {k} ({v['name']})")
            else:
                logger.debug(f"   . Skipped existing: {k}")

# ==========================================
# 5. 主入口 (Main)
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Auto-register XDL equipment types to YAML manifest.")
    parser.add_argument("--xdl_dir", type=str, default="./experiments", help="Directory containing .xdl files")
    parser.add_argument("--manifest", type=str, default="./database/equipment_manifest.yaml", help="Path to output YAML")
    parser.add_argument("--api_key", type=str, default=Config.OPENAI_API_KEY, help="LLM API Key")
    parser.add_argument("--base_url", type=str, default=Config.OPENAI_BASE_URL, help="LLM Base URL")
    
    args = parser.parse_args()

    if not args.api_key:
        logger.critical("❌ Error: API Key is missing. Set OPENAI_API_KEY env var or pass --api_key.")
        sys.exit(1)

    # 1. 初始化管理器
    manager = RegistryManager(args.manifest)
    manager.load()
    existing_types = set(manager.current_data.get("types", {}).keys())

    # 2. 扫描 XDL
    extractor = XDLExtractor(args.xdl_dir)
    xdl_types = extractor.scan_all_types()

    # 3. 计算差异 (增量更新核心)
    # 找出 XDL 里有，但 YAML 里没有的
    new_types = list(xdl_types - existing_types)

    if not new_types:
        logger.info(Fore.GREEN + "🎉 All equipment types are already registered. Nothing to do.")
        sys.exit(0)

    logger.info(Fore.CYAN + f"🚀 Identifying {len(new_types)} new types to register: {new_types}")

    # 4. 调用 LLM 生成
    generator = ConfigGenerator(args.api_key, args.base_url)
    try:
        generated_configs = generator.generate_configs(new_types)
        
        # 5. 合并并保存
        manager.update(generated_configs)
        manager.save()
        
        print(Fore.GREEN + f"\n✅ Successfully registered {len(generated_configs)} new items!")
        
    except Exception as e:
        logger.critical(f"❌ Process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()