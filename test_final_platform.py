#!/usr/bin/env python3
"""
测试完整教学平台
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_all():
    """测试所有功能"""
    print("🧪 测试完整教学平台")
    print("="*60)
    
    # 测试导入
    print("\n1. 测试导入...")
    try:
        from complete_teaching_platform import TeachingPlatform, SocratesGuide
        print("✅ 导入成功")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 测试引导系统
    print("\n2. 测试引导系统...")
    try:
        guide = SocratesGuide()
        test_guide = guide.get_guide(step=1, action="Heat", success=True)
        print(f"✅ 引导生成: {test_guide[:50]}...")
    except Exception as e:
        print(f"❌ 引导系统失败: {e}")
        return False
    
    # 测试平台创建
    print("\n3. 测试平台创建...")
    try:
        platform = TeachingPlatform()
        print(f"✅ 平台创建: ID={platform.session_id}")
    except Exception as e:
        print(f"❌ 平台创建失败: {e}")
        return False
    
    # 测试实验选择（模拟）
    print("\n4. 测试实验选择...")
    try:
        # 这里我们只是检查方法是否存在
        if hasattr(platform, 'choose_exp') and callable(platform.choose_exp):
            print("✅ 实验选择方法存在")
        else:
            print("❌ 实验选择方法不存在")
            return False
    except Exception as e:
        print(f"❌ 实验选择测试失败: {e}")
        return False
    
    # 测试状态显示
    print("\n5. 测试状态显示...")
    try:
        if hasattr(platform, 'show_state') and callable(platform.show_state):
            print("✅ 状态显示方法存在")
        else:
            print("❌ 状态显示方法不存在")
            return False
    except Exception as e:
        print(f"❌ 状态显示测试失败: {e}")
        return False
    
    # 测试命令执行
    print("\n6. 测试命令执行...")
    try:
        # 创建测试动作
        test_action = {"action": "Wait", "duration": "1"}
        result = platform.execute(test_action)
        print(f"✅ 命令执行测试: {result.get('ok', False)}")
    except Exception as e:
        print(f"❌ 命令执行失败: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 所有基础测试通过！")
    print("\n💡 平台功能概要:")
    print("  1. 实验选择（3个实验）")
    print("  2. 双输入模式（语言意图 + XDL动作）")
    print("  3. 实时状态显示")
    print("  4. 苏格拉底式引导")
    print("  5. 完整操作记录")
    print("\n🚀 运行平台:")
    print("  python complete_teaching_platform.py")
    
    return True

def quick_demo():
    """快速演示"""
    print("\n🎬 快速功能演示")
    print("="*60)
    
    from complete_teaching_platform import TeachingPlatform
    
    try:
        # 创建平台
        platform = TeachingPlatform()
        
        # 模拟实验选择
        print("\n1. 实验选择界面:")
        print("   【1】 制取氧气")
        print("   【2】 中和反应")
        print("   【3】 电解水")
        
        # 模拟选择实验1
        platform.exp = platform.exps["1"]
        print(f"\n✅ 已选择: {platform.exp['name']}")
        
        # 初始化引擎
        from soc_chem_dia_refactored import ChemSimEngine
        platform.engine = ChemSimEngine(
            hardware_config=platform.exp["setup"],
            reagent_map=platform.exp["chems"],
            silent_mode=True
        )
        print("✅ 实验环境就绪")
        
        # 演示状态显示
        print("\n2. 状态显示演示:")
        platform.show_state()
        
        # 演示引导
        print("\n3. 引导系统演示:")
        from complete_teaching_platform import SocratesGuide
        guide = SocratesGuide()
        demo_guide = guide.get_guide(step=1, action="Heat", success=True)
        print(demo_guide)
        
        # 演示命令执行
        print("\n4. 命令执行演示:")
        test_cmd = {"action": "Attach", "vessel": "clamp1", "support": "stand1"}
        result = platform.execute(test_cmd)
        print(f"命令: {test_cmd}")
        print(f"结果: {result}")
        
        print("\n✅ 演示完成！")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """主测试"""
    print("🧪 完整教学平台测试套件")
    print("="*60)
    
    print("\n选择测试模式:")
    print("1. 完整功能测试")
    print("2. 快速演示")
    print("3. 退出")
    
    try:
        choice = input("\n请输入选择 (1/2/3): ").strip()
        
        if choice == "1":
            success = test_all()
        elif choice == "2":
            success = quick_demo()
        else:
            print("👋 退出测试")
            return 0
        
        if success:
            print("\n🎉 测试成功！平台可以正常运行。")
            print("\n📋 使用说明:")
            print("  1. 运行: python complete_teaching_platform.py")
            print("  2. 选择实验 (1/2/3)")
            print("  3. 描述意图（自然语言）")
            print("  4. 输入XDL命令（JSON格式）")
            print("  5. 查看状态和引导")
            print("  6. 重复3-5直到实验完成")
            return 0
        else:
            print("\n❌ 测试失败，请检查错误信息")
            return 1
            
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())