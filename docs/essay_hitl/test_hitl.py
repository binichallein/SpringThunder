#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HITL（人机协同）交互编辑功能自动化测试
不依赖外部API与大模型，仅调用交互编辑方法并验证结果。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import List, Dict, Any

from run_model import ZhuZiQingModelTester


def run_hitl_scripted_case(commands: List[str]) -> Dict[str, Any]:
    tester = ZhuZiQingModelTester()
    # 初始数据：
    beginning = "初始开头。"
    nodes = [
        {"node_id": 1, "description": "节点一"},
        {"node_id": 2, "description": "节点二"},
    ]
    edited_beginning, edited_nodes = tester.edit_outline_interactively(beginning, nodes, scripted_commands=commands)
    return {"beginning": edited_beginning, "nodes": edited_nodes}


def main():
    # 用例1：修改开头 + 新增节点 + 完成
    case1 = run_hitl_scripted_case([
        "show",
        "setbegin 新的开头文本",
        "add 新增的节点三",
        "done",
    ])
    assert case1["beginning"].startswith("新的开头文本"), "用例1：开头未更新"
    assert len(case1["nodes"]) == 3 and case1["nodes"][2]["description"] == "新增的节点三", "用例1：新增节点失败"

    # 用例2：插入、编辑、删除、移动与重新编号
    case2 = run_hitl_scripted_case([
        "insert 2 插入在第二位",
        "edit 1 第一项修订",
        "move 3 1",
        "del 2",
        "renumber",
        "done",
    ])
    # 预期顺序：
    # 初始 [节点一, 节点二]
    # 插入第二位 -> [节点一, 插入在第二位, 节点二]
    # 编辑第1项 -> [第一项修订, 插入在第二位, 节点二]
    # 移动第3到第1 -> [节点二, 第一项修订, 插入在第二位]
    # 删除第2项 -> [节点二, 插入在第二位]
    # 重新编号 -> id按顺序
    assert case2["nodes"][0]["description"].startswith("节点二"), "用例2：移动结果不符合预期"
    assert case2["nodes"][1]["description"].startswith("插入在第二位"), "用例2：插入/删除结果不符合预期"
    assert len(case2["nodes"]) == 2, "用例2：删除失败"

    print("所有HITL测试通过 ✅")


if __name__ == "__main__":
    main()


