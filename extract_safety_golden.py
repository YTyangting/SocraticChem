import json
import os

def extract_safety_golden_set(input_file, output_file):
    print(f"🔍 Scanning {input_file} for Safety Interception cases...")
    
    safety_cases = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f) # 假设 chem_test.json 是标准 JSON 列表
        
    for idx, item in enumerate(data):
        # 1. 检查策略是否为 PREDICTIVE_QUESTIONING
        # 注意：您的数据格式可能不同，请根据实际 key 调整
        # 假设结构是 item['output'] 或 item['ground_truth'] 里包含 strategy
        
        # 尝试解析 GT Strategy
        gt_output = item.get('output', '')
        if isinstance(gt_output, dict):
            strategy = gt_output.get('strategy', '')
        else:
            # 如果是字符串 XML 格式
            import re
            match = re.search(r"<Strategy>(.*?)</Strategy>", str(gt_output))
            strategy = match.group(1) if match else ""
        
        # 筛选条件：策略匹配 + 确实涉及安全
        if strategy == "PREDICTIVE_QUESTIONING":
            # 记录原始索引，方便回溯
            item['_original_index'] = idx
            safety_cases.append(item)

    print(f"✅ Found {len(safety_cases)} confirmed Safety Interception cases.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(safety_cases, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to {output_file}")

if __name__ == "__main__":
    # 请修改为您的原始测试集路径
    extract_safety_golden_set("chem_test.json", "safety_golden_set.json")