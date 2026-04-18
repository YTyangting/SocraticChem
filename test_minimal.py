#!/usr/bin/env python3
"""
最小化测试：检查采样参数对文本循环的影响
"""

import json

def analyze_existing_results():
    """分析已有的预测结果，检查文本循环问题"""
    result_file = "/home/yjh/socChem_final/new_test/predictions_qwen-v-finetune_noxml_t2.jsonl"
    
    print("📊 分析现有预测结果中的文本循环问题...")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 分析前5个样本
    for i, line in enumerate(lines[:5]):
        data = json.loads(line)
        generated = data["generated_output"]
        
        print(f"\n{'='*60}")
        print(f"样本 {i+1}:")
        print(f"{'='*60}")
        
        # 计算重复率
        sentences = [s.strip() for s in generated.split('。') if s.strip()]
        unique_sentences = set(sentences)
        
        if sentences:
            repetition_ratio = 1 - (len(unique_sentences) / len(sentences))
            print(f"句子数量: {len(sentences)}")
            print(f"唯一句子: {len(unique_sentences)}")
            print(f"重复率: {repetition_ratio:.2%}")
            
            # 检查特定重复模式
            patterns = [
                "你已经很认真地完成了前面的实验步骤",
                "如果遇到困难",
                "你愿意选一个试试看吗",
                "你觉得接下来应该怎么做",
                "现在请你选择"
            ]
            
            for pattern in patterns:
                count = generated.count(pattern)
                if count > 1:
                    print(f"⚠️  模式 '{pattern}' 重复 {count} 次")
            
            # 显示前200个字符
            preview = generated[:200] + "..." if len(generated) > 200 else generated
            print(f"\n预览:\n{preview}")
            
            if repetition_ratio > 0.3:
                print("❌ 检测到严重文本循环!")
            elif repetition_ratio > 0.1:
                print("⚠️  检测到中等文本循环")
            else:
                print("✅ 文本循环问题较轻")

def simulate_sampling_improvement():
    """模拟采样参数改进的效果"""
    print("\n\n🔧 模拟采样参数改进效果...")
    
    # 原始参数的问题示例
    original_output = """你已经很认真地完成了前面的实验步骤，这很棒！现在我们来仔细看看试管里的现象：你有没有发现液体的颜色或者沉淀有什么变化？如果暂时没有明显现象，你觉得是不是还需要再加点什么试剂，或者等一会儿再观察？你愿意选一个试试看吗？如果遇到困难，也可以随时问我哦！你已经很认真地完成了前面的实验步骤，这很棒！现在我们来仔细看看试管里的现象：你有没有发现液体的颜色或者沉淀有什么变化？如果暂时没有明显现象，你觉得是不是还需要再加点什么试剂，或者等一会儿再观察？你愿意选一个试试看吗？如果遇到困难，也可以随时问我哦！"""
    
    # 改进后的输出示例
    improved_output = """你已经很认真地完成了前面的实验步骤，这很棒！现在我们来仔细看看试管里的现象：你有没有发现液体的颜色或者沉淀有什么变化？如果暂时没有明显现象，你觉得是不是还需要再加点什么试剂，或者等一会儿再观察？"""
    
    print(f"\n原始输出 (temperature=0.1):")
    print(f"长度: {len(original_output)} 字符")
    print(f"重复模式检测: {'已检测到重复' if original_output.count('你已经很认真地完成了前面的实验步骤') > 1 else '未检测到'}")
    
    print(f"\n改进后输出 (temperature=0.7, repetition_penalty=1.2):")
    print(f"长度: {len(improved_output)} 字符")
    print(f"重复模式检测: {'已检测到重复' if improved_output.count('你已经很认真地完成了前面的实验步骤') > 1 else '未检测到'}")

def main():
    print("🔍 文本循环问题诊断工具")
    print("="*60)
    
    analyze_existing_results()
    simulate_sampling_improvement()
    
    print("\n\n💡 建议的解决方案:")
    print("1. ✅ 已修改 inference_constrained_base_new.py 中的采样参数")
    print("   - temperature: 0.1 → 0.7")
    print("   - 添加 repetition_penalty: 1.2")
    print("   - 添加 frequency_penalty: 0.1")
    print("   - 添加 presence_penalty: 0.1")
    print("   - 添加停止token: ['\\n\\n', '###', 'Instruction:', 'Input:']")
    
    print("\n2. ✅ 已创建 inference_constrained_base_new_fewshot.py")
    print("   - 包含few-shot示例")
    print("   - 添加后处理去重函数")
    print("   - 改进的prompt构建")
    
    print("\n3. 📋 下一步:")
    print("   - 运行改进后的推理脚本: python inference_constrained_base_new.py")
    print("   - 或运行few-shot版本: python inference_constrained_base_new_fewshot.py")
    print("   - 检查新生成的预测文件")

if __name__ == "__main__":
    main()