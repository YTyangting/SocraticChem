"""
SoChemDataset 人工检查工具 v3
==============================
修复版本：
1. 正确提取并展示 XDL、Action、Result、XML CoT 原文
2. 新增 Action 检查（XDL转换正确性核心）
3. 每条数据单独呈现，标注员填写判断
4. 支持多人独立标注，结果分开汇总

使用方法：
1. python dataset_validator.py --sample 150
2. 打开生成的 CSV，每个标注员独立检查同一份数据
3. python dataset_validator.py --summarize dataset_check_template.csv
"""

import json
import random
import argparse
import os
import re
from collections import defaultdict

# ===========================
# 配置
# ===========================
ANNOTATORS = ["标注员A", "标注员B", "标注员C"]
SAMPLE_SIZE = 150
INPUT_FILE = "sft_finetune_chemlab_train_1.json"
OUTPUT_PREFIX = "dataset_check"

# ===========================
# 错误类型定义
# ===========================
XDL_ERROR_TYPES = {
    "XDL_OK": "正确",
    "XDL_ACTION_WRONG": "Action与XDL不符",
    "XDL_ACTION_OBS_MISMATCH": "Observation与Action不符",
    "XDL_REAGENT_WRONG": "试剂错误",
    "XDL_EQUIPMENT_WRONG": "器材错误",
    "XDL_OTHER": "其他错误"
}

SAFE_ERROR_TYPES = {
    "SAFE_OK": "正确",
    "SAFE_MISSING": "应标FAIL但标PASS",
    "SAFE_WRONG": "不应标FAIL但标了",
    "SAFE_INCOMPLETE": "CheckList不完整",
    "SAFE_OTHER": "其他错误"
}

CAUSAL_ERROR_TYPES = {
    "CAUSAL_OK": "正确",
    "CAUSAL_NODE_ERROR": "CurrentNode错误",
    "CAUSAL_LOGIC_ERROR": "逻辑推理错误",
    "CAUSAL_STRATEGY_ERROR": "策略选择错误",
    "CAUSAL_RESPONSE_ERROR": "Response与策略不符",
    "CAUSAL_OTHER": "其他错误"
}

# ===========================
# XML 解析工具
# ===========================
def extract_xml_tag(xml, tag):
    """提取 XML 标签内容"""
    pattern = f"<{tag}[^>]*>(.*?)</{tag}>"
    match = re.search(pattern, xml, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_student_blocks(input_xml):
    """提取 Student 标签内的所有子块"""
    student_tag = extract_xml_tag(input_xml, 'Student')
    return {
        'speak': extract_xml_tag(student_tag, 'Speak'),
        'action': extract_xml_tag(student_tag, 'Action'),
    }

def extract_all_xml_blocks(xml):
    """提取所有关键 XML 块"""
    return {
        'xdl': extract_xml_tag(xml, 'XDL'),
        'dag': extract_xml_tag(xml, 'DAG'),
        'observation': extract_xml_tag(xml, 'Observation'),
        'student': extract_xml_tag(xml, 'Student'),
        'findcurrentnode': extract_xml_tag(xml, 'FindcurrentNode'),
        'vertify': extract_xml_tag(xml, 'vertify'),
        'thought': extract_xml_tag(xml, 'Thought'),
        'response': extract_xml_tag(xml, 'Response'),
        'result': extract_xml_tag(xml, 'Result'),
    }

# ===========================
# 抽样和生成检查表
# ===========================
def sample_and_generate(data, n, output_prefix):
    """抽样并生成检查表"""
    
    # 随机抽样
    random.seed(42)
    sample_indices = random.sample(range(len(data)), min(n, len(data)))
    sample_data = [(data[i], i) for i in sample_indices]
    
    print(f"📊 抽样完成: {len(sample_data)} 条数据")
    
    # 保存完整 JSONL（供标注员查阅）
    jsonl_path = f"{output_prefix}_sample.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item, idx in sample_data:
            item['_sample_index'] = idx
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"💾 保存完整数据: {jsonl_path}")
    
    # 生成 CSV 检查表（包含供标注员查看的原文）
    csv_path = f"{output_prefix}_template.csv"
    
    with open(csv_path, 'w', encoding='utf-8-sig') as f:
        # 标题行
        f.write("=" * 80 + "\n")
        f.write("SoChemDataset 质量检查表\n")
        f.write("检查说明：Action与Observation生成正确性、安全策略、因果追踪（详见评估指南）\n")
        f.write("评分标准：XDL转换(正确/错误)、安全策略(正确/错误)、因果追踪(正确/错误)\n")
        f.write("总体评分(1-5): 1=很差, 2=较差, 3=一般, 4=较好, 5=很好\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
        
        # 每条数据的标题
        for seq_num, (item, orig_idx) in enumerate(sample_data, 1):
            input_xml = item.get('input', '')
            output_xml = item.get('output', '')
            
            # 提取所有 XML 块
            blocks = extract_all_xml_blocks(input_xml)
            out_blocks = extract_all_xml_blocks(output_xml)
            student_blocks = extract_student_blocks(input_xml)
            
            # 写入数据
            f.write(f"【数据 {seq_num}】(原始索引: {orig_idx})\n")
            f.write("-" * 40 + "\n")
            
            f.write("【Student Say】(学生输入):\n")
            f.write(f"{student_blocks['speak']}\n")
            f.write("\n")
            
            f.write("【Action】(学生操作 - 检查是否与XDL Procedure一致):\n")
            f.write(f"{student_blocks['action']}\n")
            f.write("\n")
            
            f.write("【Observation】(系统观察 - 检查是否与Action一致):\n")
            f.write(f"{blocks['observation']}\n")
            f.write("\n")
            
            f.write("【XDL Procedure】(实验步骤 - 用于对照Action):\n")
            xdl_content = blocks['xdl']
            # 提取 Procedure 部分
            procedure_match = re.search(r'<Procedure>(.*?)</Procedure>', xdl_content, re.DOTALL)
            if procedure_match:
                f.write(f"{procedure_match.group(0)}\n")
            else:
                f.write(f"{xdl_content}\n")
            f.write("\n")
            
            f.write("【Result & vertify】(安全策略 - 检查是否正确):\n")
            f.write(f"Result: {out_blocks['result']}\n")
            f.write(f"CheckList: {extract_xml_tag(output_xml, 'CheckList')}\n")
            f.write("\n")
            
            f.write("【FindcurrentNode & vertify】(因果追踪 - 检查逻辑是否正确):\n")
            f.write(f"FindcurrentNode:\n{out_blocks['findcurrentnode']}\n")
            f.write(f"\nvertify:\n{out_blocks['vertify']}\n")
            f.write(f"\nThought:\n{out_blocks['thought']}\n")
            f.write("\n")
            
            f.write("【Response】(教学回复 - 检查是否遵循策略):\n")
            f.write(f"{out_blocks['response']}\n")
            f.write("\n")
            
            # 标注区域
            f.write("【标注区域】\n")
            f.write(f"XDL转换是否正确(Action与Observation):, XDL错误类型:\n")
            f.write(f"安全策略是否正确:, 安全错误类型:\n")
            f.write(f"因果追踪是否正确:, 因果错误类型:\n")
            f.write(f"总体质量评分(1-5):\n")
            f.write(f"备注:\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("\n")
    
    print(f"💾 保存检查表: {csv_path}")
    print(f"\n📋 检查表格式说明:")
    print(f"   - 每条数据包含: Student Say, Action, Observation, XDL Procedure, Result, XML CoT, Response")
    print(f"   - XDL转换检查: 对比 Action 与 XDL Procedure 是否一致，Observation 与 Action 是否一致")
    print(f"   - 标注员在【标注区域】填写判断")
    print(f"   - 同一份表格可由多人独立使用（分别填写）")
    print(f"   - 或将标注区域复制到新文件分发给每位标注员")
    
    return sample_data

# ===========================
# 汇总统计
# ===========================
def summarize(csv_file):
    """汇总标注结果"""
    print(f"\n{'='*60}")
    print(f"📊 汇总统计: {csv_file}")
    print(f"{'='*60}")
    
    # 提示用户如何进行汇总
    print("""
⚠️ 手动汇总说明：
由于标注员可能使用不同的文件，请分别统计每位标注员的结果。

汇总方法：
1. 打开标注员填写的 CSV 文件
2. 统计以下三项的"正确"和"错误"数量：
   - XDL转换是否正确?
   - 安全策略是否正确?
   - 因果追踪是否正确?
3. 计算各项错误率

报告模板：
---
Dataset Validation Summary:
- Total samples: 150
- XDL conversion errors: X%
- Safety strategy errors: X%
- Causal trace errors: X%
- Inter-annotator agreement: XX%
---
""")

# ===========================
# 主函数
# ===========================
def main():
    parser = argparse.ArgumentParser(description="SoChemDataset 人工检查工具 v3")
    parser.add_argument("--sample", type=int, default=150, help="抽样数量")
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="输入数据文件")
    parser.add_argument("--output", type=str, default=OUTPUT_PREFIX, help="输出文件前缀")
    parser.add_argument("--summarize", type=str, help="汇总 CSV 文件")
    
    args = parser.parse_args()
    
    if args.summarize:
        summarize(args.summarize)
        return
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        print(f"请确保已将文件上传到服务器或修改 --input 参数")
        return
    
    print(f"📂 加载数据: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   总数据量: {len(data)}")
    
    sample_and_generate(data, args.sample, args.output)
    
    print(f"\n📌 下一步:")
    print(f"   1. 下载生成的 CSV 文件")
    print(f"   2. 分发给标注员检查")
    print(f"   3. 标注完成后运行汇总统计")

if __name__ == "__main__":
    main()
