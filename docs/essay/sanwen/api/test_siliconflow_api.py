#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硅基流动API测试工具
专门用于测试API调用和获取开头、情节规划等信息
"""

import os
import json
import requests
import logging
from typing import Dict, Any, List, Tuple
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SiliconFlowAPITester:
    """硅基流动API测试器"""
    
    def __init__(self, api_key: str = "sk-zrwphtyaynbrvgqflgvnjptkptdoeeylucqmlzegxiwlkzex"):
        self.api_key = api_key
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.model = "deepseek-ai/DeepSeek-V3"
        
    def test_api_connection(self) -> Dict[str, Any]:
        """测试API连接"""
        logger.info("测试API连接...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 简单的测试请求
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "请回复'API连接正常'"
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            logger.info(f"API连接测试成功，响应: {content}")
            return {
                "success": True,
                "response": content,
                "status_code": response.status_code
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API连接测试失败: {e}")
            return {
                "success": False,
                "error": f"API连接失败: {e}"
            }
        except Exception as e:
            logger.error(f"API连接测试出错: {e}")
            return {
                "success": False,
                "error": f"API连接测试出错: {e}"
            }
    
    def generate_narrative_outline(self, user_topic: str) -> Dict[str, Any]:
        """生成情节规划"""
        logger.info(f"开始为主题 '{user_topic}' 生成情节规划")
        
        # 注意：不要用 str.format 直接填充，JSON 花括号会被误当占位符。
        prompt_template = """# 角色与任务
你是一位精通朱自清散文风格的叙事架构师。你的任务是：根据用户提供的简单主题，首先创作一个符合朱自清文风的开头段落，随后直接生成一个简洁明了、可用于后续散文创作的叙事大纲。

# 执行步骤
1.  **风格化开头创作**：首先，模仿朱自清的风格（语言朴实细腻、善用具体意象、情感含蓄深沉、节奏舒缓自然）创作一个开头段落。这个开头应能自然地引出故事。
2.  **叙事大纲规划**：接着，基于你生成的开头，规划一个3-4个节点的叙事大纲。每个节点必须是**一个具体的、可被描写的场景或事件**，并为每个节点指明其**核心情感或动作**。
3.  **结尾**：保证结尾的情节能收束整篇文章的情感，不要让人产生文章是否已经结束的疑惑
# 输出格式
你必须严格按照以下JSON格式输出，你的输出将被其他程序直接解析：

{
  "generated_beginning": "在这里生成完整的开头段落，要求不少于50字。",
  "narrative_outline": [
    {
      "node_id": 1,
      "description": "用一句话描述第一个核心场景或事件。"
    },
    {
      "node_id": 2,
      "description": "用一句话描述第二个承上启下的场景或事件。"
    },
    {
      "node_id": 3,
      "description": "用一句话描述第三个推向深化的场景或事件。"
    }
  ]
}

# 规则与禁忌
- **必须**：输出必须是且仅是一个合法的JSON对象。
- **必须**：`narrative_outline`中的每个`description`字段必须描述**具体动作或场景**（例如："父亲穿过铁道去买橘子"、"月光下独自漫步荷塘"）。
- **禁忌**：禁止在描述中使用抽象的情感概括（如："他感到很悲伤"），必须转化为具体行为（如："他望着远方，轻轻叹了口气"）。
- **禁忌**：避免规划过多或过少的节点，3-4个为佳，确保每个节点都能发展为一个完整的散文段落。

# 用户输入主题
"{user_topic}" """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 仅替换我们定义的占位符，避免 format 解析 JSON 花括号
        prompt_filled = prompt_template.replace("{user_topic}", user_topic)

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_filled
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False
        }
        
        try:
            start_time = time.time()
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            request_time = time.time() - start_time
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            logger.info(f"API请求耗时: {request_time:.2f}秒")
            logger.info(f"原始响应内容长度: {len(content)} 字符")
            logger.info(f"原始响应内容预览: {content[:200]}...")
            
            # 尝试解析JSON
            parsed_result = self._parse_json_response(content)
            parsed_result["request_time"] = request_time
            parsed_result["raw_content"] = content
            
            return parsed_result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {e}")
            return {
                "success": False,
                "error": f"API请求失败: {e}",
                "error_type": "RequestException"
            }
        except Exception as e:
            logger.error(f"处理API响应时出错: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"处理API响应时出错: {e}",
                "error_type": type(e).__name__
            }
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析JSON响应"""
        logger.info("开始解析JSON响应...")
        
        # 策略1: 直接解析
        try:
            parsed_content = json.loads(content)
            logger.info("直接JSON解析成功")
            return {
                "success": True,
                "data": parsed_content,
                "parse_method": "direct"
            }
        except json.JSONDecodeError as e:
            logger.warning(f"直接JSON解析失败: {e}")
        
        # 策略2: 提取JSON部分
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                extracted_json = json_match.group()
                logger.info(f"提取到JSON片段，长度: {len(extracted_json)} 字符")
                parsed_content = json.loads(extracted_json)
                logger.info("提取JSON解析成功")
                return {
                    "success": True,
                    "data": parsed_content,
                    "parse_method": "extracted"
                }
            except json.JSONDecodeError as e2:
                logger.warning(f"提取的JSON片段解析失败: {e2}")
        
        # 策略3: 清理markdown标记
        try:
            cleaned_content = re.sub(r'```json\s*', '', content)
            cleaned_content = re.sub(r'```\s*$', '', cleaned_content)
            cleaned_content = cleaned_content.strip()
            
            parsed_content = json.loads(cleaned_content)
            logger.info("清理后JSON解析成功")
            return {
                "success": True,
                "data": parsed_content,
                "parse_method": "cleaned"
            }
        except json.JSONDecodeError as e3:
            logger.warning(f"清理后JSON解析失败: {e3}")
        
        # 策略4: 手动提取字段
        try:
            beginning_match = re.search(r'"generated_beginning":\s*"([^"]*(?:\\.[^"]*)*)"', content, re.DOTALL)
            outline_match = re.search(r'"narrative_outline":\s*(\[.*?\])', content, re.DOTALL)
            
            if beginning_match and outline_match:
                beginning_text = beginning_match.group(1)
                outline_text = outline_match.group(1)
                
                # 手动构建JSON
                manual_json = {
                    "generated_beginning": beginning_text,
                    "narrative_outline": json.loads(outline_text)
                }
                
                logger.info("手动提取字段成功")
                return {
                    "success": True,
                    "data": manual_json,
                    "parse_method": "manual_extraction"
                }
        except Exception as e4:
            logger.warning(f"手动提取字段失败: {e4}")
        
        # 所有策略都失败
        logger.error("所有JSON解析策略都失败")
        return {
            "success": False,
            "error": "无法解析API返回的JSON内容",
            "raw_content": content
        }
    
    def test_multiple_topics(self, topics: List[str]) -> Dict[str, Any]:
        """测试多个主题"""
        logger.info(f"开始测试 {len(topics)} 个主题")
        
        results = {}
        success_count = 0
        
        for i, topic in enumerate(topics, 1):
            logger.info(f"测试主题 {i}/{len(topics)}: {topic}")
            
            result = self.generate_narrative_outline(topic)
            results[topic] = result
            
            if result["success"]:
                success_count += 1
                logger.info(f"主题 '{topic}' 测试成功")
            else:
                logger.error(f"主题 '{topic}' 测试失败: {result.get('error', '未知错误')}")
            
            # 避免请求过于频繁
            if i < len(topics):
                time.sleep(1)
        
        logger.info(f"测试完成: {success_count}/{len(topics)} 个主题成功")
        
        return {
            "total_topics": len(topics),
            "success_count": success_count,
            "success_rate": success_count / len(topics) if topics else 0,
            "results": results
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """保存测试结果"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"siliconflow_api_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试结果已保存到: {filename}")
        return filename

def get_beginning_and_outline(topic: str, api_key: str | None = None) -> Tuple[str, Dict[int, str]]:
    """简化接口：返回(开头字符串, 情节节点字典)
    - 字典按顺序编号: {1: 描述, 2: 描述, ...}
    - API失败时返回("", {})
    """
    tester = SiliconFlowAPITester(api_key or "sk-zrwphtyaynbrvgqflgvnjptkptdoeeylucqmlzegxiwlkzex")
    result = tester.generate_narrative_outline(topic)
    if not result.get("success"):
        return "", {}
    data = result.get("data", {})
    beginning = data.get("generated_beginning", "") or ""
    nodes = data.get("narrative_outline", []) or []
    ordered_outline: Dict[int, str] = {i + 1: (node.get("description", "") or "") for i, node in enumerate(nodes)}
    return beginning, ordered_outline

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="硅基流动API测试工具")
    parser.add_argument("--test_connection", action="store_true", help="测试API连接")
    parser.add_argument("--topic", type=str, help="测试单个主题")
    parser.add_argument("--topics", nargs="+", help="测试多个主题")
    parser.add_argument("--api_key", type=str, help="API密钥")
    parser.add_argument("--save", action="store_true", help="保存测试结果")
    parser.add_argument("--simple", action="store_true", help="使用简化接口返回开头与编号情节字典")
    
    args = parser.parse_args()
    
    # 创建测试器
    api_key = args.api_key or "sk-zrwphtyaynbrvgqflgvnjptkptdoeeylucqmlzegxiwlkzex"
    tester = SiliconFlowAPITester(api_key)
    
    # 测试API连接
    if args.test_connection:
        print("=" * 60)
        print("测试API连接")
        print("=" * 60)
        connection_result = tester.test_api_connection()
        print(f"连接结果: {connection_result}")
        print()
    
    # 测试单个主题
    if args.topic and args.simple:
        print("=" * 60)
        print(f"简化接口 - 主题: {args.topic}")
        print("=" * 60)
        beginning, outline = get_beginning_and_outline(args.topic, args.api_key)
        if beginning:
            print("【开头】\n" + beginning)
            print("\n【情节节点（按序号）】")
            for k in sorted(outline.keys()):
                print(f"{k}. {outline[k]}")
        else:
            print("❌ 获取失败")
        return

    if args.topic:
        print("=" * 60)
        print(f"测试主题: {args.topic}")
        print("=" * 60)
        result = tester.generate_narrative_outline(args.topic)
        
        if result["success"]:
            print("✅ 生成成功!")
            print(f"请求耗时: {result.get('request_time', 0):.2f}秒")
            print(f"解析方法: {result.get('parse_method', 'unknown')}")
            print()
            
            data = result["data"]
            print("【生成的开头段落】")
            print(data.get("generated_beginning", ""))
            print()
            
            print("【情节规划】")
            for node in data.get("narrative_outline", []):
                print(f"节点{node.get('node_id', '?')}: {node.get('description', '')}")
            print()
        else:
            print("❌ 生成失败!")
            print(f"错误: {result.get('error', '未知错误')}")
            print()
        
        if args.save:
            tester.save_results(result, f"single_topic_test_{int(time.time())}.json")
    
    # 测试多个主题
    if args.topics:
        print("=" * 60)
        print(f"测试多个主题: {args.topics}")
        print("=" * 60)
        results = tester.test_multiple_topics(args.topics)
        
        print(f"总体结果: {results['success_count']}/{results['total_topics']} 成功")
        print(f"成功率: {results['success_rate']:.2%}")
        print()
        
        for topic, result in results["results"].items():
            status = "✅" if result["success"] else "❌"
            print(f"{status} {topic}: {result.get('error', '成功')}")
        
        if args.save:
            tester.save_results(results, f"multiple_topics_test_{int(time.time())}.json")
    
    # 如果没有指定任何参数，运行默认测试
    if not any([args.test_connection, args.topic, args.topics]):
        print("=" * 60)
        print("运行默认测试")
        print("=" * 60)
        
        # 测试连接
        connection_result = tester.test_api_connection()
        print(f"API连接测试: {'✅ 成功' if connection_result['success'] else '❌ 失败'}")
        print()
        
        # 测试默认主题
        default_topics = [
            "在一个秋日，旧友在茶舍重逢",
            "荷塘月色的夜晚",
            "春天的校园"
        ]
        
        print("测试默认主题...")
        results = tester.test_multiple_topics(default_topics)
        
        print(f"测试结果: {results['success_count']}/{results['total_topics']} 成功")
        print(f"成功率: {results['success_rate']:.2%}")
        
        # 保存结果
        filename = tester.save_results(results)
        print(f"结果已保存到: {filename}")

if __name__ == "__main__":
    main()
