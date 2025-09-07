#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
朱自清散文续写项目主程序入口
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数 - 提供命令行菜单"""
    print("=" * 60)
    print("朱自清散文续写项目 (Sanwen)")
    print("=" * 60)
    print("请选择功能:")
    print("1. 模型训练 (微调)")
    print("2. 散文生成测试")
    print("3. API功能测试") 
    print("4. 退出")
    print("-" * 60)
    
    while True:
        try:
            choice = input("请输入选项 (1-4): ").strip()
            
            if choice == "1":
                print("\n启动模型训练...")
                from sanwen.finetune.sft import main as sft_main
                sft_main()
                break
                
            elif choice == "2":
                print("\n启动散文生成测试...")
                from tools.model_tools.test_model import main as test_main
                test_main()
                break
                
            elif choice == "3":
                print("\n启动API功能测试...")
                from sanwen.api.test_siliconflow_api import main as api_main
                api_main()
                break
                
            elif choice == "4":
                print("退出程序")
                sys.exit(0)
                
            else:
                print("无效选项，请输入 1-4")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            sys.exit(0)
        except Exception as e:
            print(f"发生错误: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()