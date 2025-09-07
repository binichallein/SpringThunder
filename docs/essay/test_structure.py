#!/usr/bin/env python3
"""
项目结构和功能测试脚本
"""

import sys
import os
sys.path.insert(0, '.')

def test_imports():
    """测试各模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        # 测试API模块
        from sanwen.api.test_siliconflow_api import get_beginning_and_outline
        print("✅ API模块导入成功")
        
        # 测试包初始化
        import sanwen
        print(f"✅ 主包导入成功 (版本: {sanwen.__version__})")
        
        # 测试子包
        from sanwen import api, finetune, models, utils
        print("✅ 子包导入成功")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    return True

def test_api_functionality():
    """测试API功能"""
    print("\n📡 测试API功能...")
    
    try:
        from sanwen.api.test_siliconflow_api import SiliconFlowAPITester
        
        tester = SiliconFlowAPITester()
        result = tester.test_api_connection()
        
        if result["success"]:
            print("✅ API连接测试成功")
            return True
        else:
            print(f"❌ API连接失败: {result.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n📁 测试文件结构...")
    
    expected_structure = [
        "sanwen/__init__.py",
        "sanwen/api/__init__.py",
        "sanwen/finetune/__init__.py",
        "sanwen/models/__init__.py",
        "sanwen/utils/__init__.py",
        "tools/model_tools/test_model.py",
        "data/raw/zhu_ziqing_continuations.json",
        "data/models/qwen_zhu_ziqing_lora",
        "setup.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file_path in expected_structure:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return False
    else:
        print("✅ 文件结构完整")
        return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("Sanwen 项目结构测试")
    print("=" * 60)
    
    results = []
    
    # 运行各项测试
    results.append(test_file_structure())
    results.append(test_imports())
    results.append(test_api_functionality())
    
    # 总结结果
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    
    if all(results):
        print("🎉 所有测试通过！项目结构重组成功")
        print("\n💡 使用建议:")
        print("   - 运行 'python main.py' 使用主程序")
        print("   - 运行 'python tools/model_tools/test_model.py' 进行模型测试")
        print("   - 运行 'python sanwen/api/test_siliconflow_api.py' 进行API测试")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
        failed_count = len([r for r in results if not r])
        print(f"失败项目数: {failed_count}/{len(results)}")

if __name__ == "__main__":
    main()