#!/usr/bin/env python3
"""
测试学生交互平台
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """测试导入"""
    print("🧪 测试导入模块...")
    try:
        # 测试导入原有框架
        from soc_chem_dia_refactored import ChemSimEngine
        print("✅ 成功导入 ChemSimEngine")
        
        # 测试导入新平台
        from student_interactive_platform_simple import SimpleStudentPlatform
        print("✅ 成功导入 SimpleStudentPlatform")
        
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_platform_creation():
    """测试平台创建"""
    print("\n🧪 测试平台创建...")
    try:
        from student_interactive_platform_simple import SimpleStudentPlatform
        
        platform = SimpleStudentPlatform()
        print("✅ 平台创建成功")
        print(f"   会话ID: {platform.session_id}")
        print(f"   记录文件: {platform.record_file}")
        
        # 测试状态显示
        platform.show_status()
        
        return True
    except Exception as e:
        print(f"❌ 平台创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_command_parsing():
    """测试命令解析"""
    print("\n🧪 测试命令解析...")
    try:
        from student_interactive_platform_simple import SimpleStudentPlatform
        
        platform = SimpleStudentPlatform()
        
        test_commands = [
            "向试管1加入水10ml",
            "加热试管1",
            "将试管夹连接到铁架台",
            "从烧杯1倒入试管1",
            '{"action":"Add","vessel":"tube1","reagent":"water","volume":"10ml"}',
            "等待5秒",
            "搅拌试管1"
        ]
        
        for cmd in test_commands:
            print(f"\n  测试命令: {cmd}")
            action = platform.parse_command(cmd)
            if action:
                print(f"    解析结果: {action}")
            else:
                print(f"    解析失败")
        
        return True
    except Exception as e:
        print(f"❌ 命令解析测试失败: {e}")
        return False

def test_execution():
    """测试命令执行"""
    print("\n🧪 测试命令执行...")
    try:
        from student_interactive_platform_simple import SimpleStudentPlatform
        
        platform = SimpleStudentPlatform()
        
        # 测试几个基本操作
        test_actions = [
            {"action": "Attach", "vessel": "clamp1", "support": "stand1"},
            {"action": "Add", "vessel": "tube1", "reagent": "water", "volume": "10ml"},
            {"action": "Wait", "duration": "2"}
        ]
        
        for action in test_actions:
            print(f"\n  执行: {action}")
            result = platform.execute(action)
            if result.get("ok"):
                print(f"    成功: {result.get('msg', '')[:50]}...")
            else:
                print(f"    失败: {result.get('msg', '')}")
        
        # 显示最终状态
        platform.show_status()
        
        return True
    except Exception as e:
        print(f"❌ 执行测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("🧪 学生交互平台测试")
    print("="*60)
    
    tests = [
        ("导入测试", test_import),
        ("平台创建测试", test_platform_creation),
        ("命令解析测试", test_command_parsing),
        ("命令执行测试", test_execution)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        if test_func():
            print(f"✅ {test_name} 通过")
            passed += 1
        else:
            print(f"❌ {test_name} 失败")
    
    print("\n" + "="*60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！平台可以正常运行。")
        print("\n💡 使用说明:")
        print("   1. 运行平台: python student_interactive_platform_simple.py")
        print("   2. 输入自然语言命令或JSON格式命令")
        print("   3. 输入'状态'查看当前状态")
        print("   4. 输入'退出'结束实验")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())