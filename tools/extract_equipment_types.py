import os
import glob
import xml.etree.ElementTree as ET
from collections import Counter

def scan_xdl_hardware(xdl_dir):
    unique_types = set()
    type_counts = Counter()
    
    # 查找所有 xdl 文件
    files = glob.glob(os.path.join(xdl_dir, "*.xdl"))
    print(f"🔍 扫描了 {len(files)} 个 XDL 文件...")

    for file_path in files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # 查找所有 Hardware 下的 Component
            for comp in root.findall(".//Hardware/Component"):
                hw_type = comp.get("type")
                if hw_type:
                    # 统一转小写，防止 Test_Tube 和 test_tube 被当成两个
                    normalized_type = hw_type.lower().strip()
                    unique_types.add(normalized_type)
                    type_counts[normalized_type] += 1
                    
        except Exception as e:
            print(f"❌ 解析错误 {file_path}: {e}")

    return unique_types, type_counts

if __name__ == "__main__":
    # 替换为你存放 XDL 文件的真实路径
    XDL_FOLDER = "./experiments" 
    
    types, counts = scan_xdl_hardware(XDL_FOLDER)
    
    print("\n✅ 发现以下器材类型 (按频次排序):")
    print("-" * 40)
    for t, c in counts.most_common():
        print(f"{t}: {c} 次")
        
    # 自动生成一个 YAML 模板供你填写
    print("\n📝 正在生成 equipment_manifest.yaml 模板...")
    with open("equipment_manifest_template.yaml", "w", encoding="utf-8") as f:
        f.write("types:\n")
        for t in sorted(list(types)):
            f.write(f"  {t}:\n")
            f.write(f"    category: unknown  # vessel / heater / tool / support\n")
            f.write(f"    components: []     # [storage, thermal, topology]\n")
            f.write(f"    tags: []           # [glassware, transparent]\n\n")
            
    print("✅ 模板已生成，请打开 equipment_manifest_template.yaml 进行配置！")