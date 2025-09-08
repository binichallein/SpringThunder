#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
朱自清散文续写模型测试工具
支持交互式测试和自动测试
"""

import os
import json
import torch
import logging
import argparse
import random
import requests
from typing import List, Dict, Any, Optional
from transformers import GenerationConfig
from modelscope import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel
import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
from sanwen.api.test_siliconflow_api import get_beginning_and_outline
import time

class SiliconFlowAPIClient:
    """硅基流动API客户端"""
    
    def __init__(self, api_key: str = "sk-zrwphtyaynbrvgqflgvnjptkptdoeeylucqmlzegxiwlkzex"):
        self.api_key = api_key
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.model = "deepseek-ai/DeepSeek-V3"
        
    def generate_narrative_outline_local(self, user_topic: str) -> Dict[str, Any]:
        """本地备用生成方法（当API失败时使用）"""
        logger.info("使用本地备用方法生成情节规划")
        
        # 简单的本地生成逻辑
        beginning_templates = [
            f"秋日的午后，阳光透过窗棂洒在茶桌上，{user_topic}。",
            f"时光荏苒，{user_topic}，仿佛昨日重现。",
            f"在这个宁静的秋日，{user_topic}，心中涌起万千思绪。"
        ]
        
        outline_templates = [
            [
                {"node_id": 1, "description": "描述重逢时的场景和氛围"},
                {"node_id": 2, "description": "回忆往昔的友谊和共同经历"},
                {"node_id": 3, "description": "表达对时光流逝的感慨和对未来的期许"}
            ],
            [
                {"node_id": 1, "description": "描绘茶舍的环境和茶香"},
                {"node_id": 2, "description": "通过对话展现友情的深厚"},
                {"node_id": 3, "description": "以景抒情，表达内心的感受"}
            ]
        ]
        
        import random
        beginning = random.choice(beginning_templates)
        outline = random.choice(outline_templates)
        
        return {
            "success": True,
            "data": {
                "generated_beginning": beginning,
                "narrative_outline": outline
            }
        }
        
    def generate_narrative_outline(self, user_topic: str) -> Dict[str, Any]:
        """根据用户主题生成情节规划"""
        # 首先尝试API调用
        api_result = self._call_api_for_outline(user_topic)
        if api_result["success"]:
            return api_result
        
        # API失败时使用本地备用方法
        logger.warning("API调用失败，使用本地备用方法")
        return self.generate_narrative_outline_local(user_topic)
    
    def _call_api_for_outline(self, user_topic: str) -> Dict[str, Any]:
        """调用API生成情节规划"""
        prompt_template = """# 角色与任务
你是一位精通朱自清散文风格的叙事架构师。你的任务是：根据用户提供的简单主题，首先创作一个符合朱自清文风的开头段落，随后直接生成一个简洁明了、可用于后续散文创作的叙事大纲。

# 执行步骤
1.  **风格化开头创作**：首先，模仿朱自清的风格（语言朴实细腻、善用具体意象、情感含蓄深沉、节奏舒缓自然）创作一个开头段落。这个开头应能自然地引出故事。
2.  **叙事大纲规划**：接着，基于你生成的开头，规划一个3-4个节点的叙事大纲。每个节点必须是**一个具体的、可被描写的场景或事件**，并为每个节点指明其**核心情感或动作**。

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
- **必须**：`narrative_outline`中的每个`description`字段必须描述**具体动作或场景**。
- **禁忌**：禁止在描述中使用抽象的情感概括，必须转化为具体行为。
- **禁忌**：避免规划过多或过少的节点，3-4个为佳。

# 用户输入主题
"{user_topic}" """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_template.format(user_topic=user_topic)
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False
        }
        
        try:
            logger.info(f"调用硅基流动API生成情节规划，主题: {user_topic}")
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 尝试解析JSON
            try:
                parsed_content = json.loads(content)
                return {
                    "success": True,
                    "data": parsed_content
                }
            except json.JSONDecodeError as e:
                logger.warning(f"初始JSON解析失败: {e}")
                logger.warning(f"原始内容: {content[:500]}...")
                
                # 尝试多种JSON提取策略
                import re
                
                # 策略1: 提取第一个完整的JSON对象
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        extracted_json = json_match.group()
                        logger.info(f"提取到JSON片段: {extracted_json[:200]}...")
                        parsed_content = json.loads(extracted_json)
                        return {
                            "success": True,
                            "data": parsed_content
                        }
                    except json.JSONDecodeError as e2:
                        logger.warning(f"提取的JSON片段也无法解析: {e2}")
                
                # 策略2: 尝试修复常见的JSON格式问题
                try:
                    # 移除可能的markdown代码块标记
                    cleaned_content = re.sub(r'```json\s*', '', content)
                    cleaned_content = re.sub(r'```\s*$', '', cleaned_content)
                    cleaned_content = cleaned_content.strip()
                    
                    # 尝试解析清理后的内容
                    parsed_content = json.loads(cleaned_content)
                    return {
                        "success": True,
                        "data": parsed_content
                    }
                except json.JSONDecodeError as e3:
                    logger.warning(f"清理后的内容仍无法解析: {e3}")
                
                # 策略3: 尝试手动构建JSON（如果内容结构清晰）
                try:
                    # 提取关键字段
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
                        
                        logger.info("通过手动提取成功构建JSON")
                        return {
                            "success": True,
                            "data": manual_json
                        }
                except Exception as e4:
                    logger.warning(f"手动构建JSON失败: {e4}")
                
                return {
                    "success": False,
                    "error": f"API返回的内容不是有效的JSON格式: {e}",
                    "raw_content": content
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API调用失败: {e}")
            return {
                "success": False,
                "error": f"API调用失败: {e}"
            }
        except Exception as e:
            logger.error(f"处理API响应时出错: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"处理API响应时出错: {e}",
                "error_type": type(e).__name__
            }

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZhuZiQingModelTester:
    """朱自清散文续写模型测试器"""
    
    def __init__(self, 
                 base_model_name: str = "qwen/Qwen2.5-7B-Instruct",
                 lora_adapter_path: str = "../../data/models/qwen_zhu_ziqing_lora",
                 cache_dir: str = "../../data/models"):
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        self.api_client = SiliconFlowAPIClient()  # 初始化API客户端
        
    def load_model(self):
        """加载基础模型和LoRA适配器"""
        logger.info("开始加载模型...")
        
        # 加载分词器
        logger.info(f"加载分词器: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="right",
            cache_dir=self.cache_dir
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 加载基础模型
        logger.info(f"加载基础模型: {self.base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=self.cache_dir
        )
        
        # 加载LoRA适配器
        if os.path.exists(self.lora_adapter_path):
            logger.info(f"加载LoRA适配器: {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.lora_adapter_path,
                torch_dtype=torch.float16
            )
            logger.info("LoRA适配器加载完成")
        else:
            logger.warning(f"LoRA适配器路径不存在: {self.lora_adapter_path}")
            logger.warning("将使用原始模型进行测试")
            
        logger.info("模型加载完成")
        
    def parse_sections(self, text: str) -> Dict[str, str]:
        """从模型输出中解析【风格诊断】与【散文续写】内容"""
        style_content = ""
        continuation_content = ""
        content = text
        
        # 只保留 assistant 段
        if "<|im_start|>assistant" in content:
            content = content.split("<|im_start|>assistant", 1)[1]
            if "<|im_end|>" in content:
                content = content.split("<|im_end|>", 1)[0]
        
        # 消除可能残留的后续对话块
        for marker in ["<|im_start|>user", "<|im_start|>system", "<|im_start|>assistant"]:
            if marker in content:
                content = content.split(marker, 1)[0]
        content = content.strip()
        
        # 标准格式解析
        if "【散文续写】" in content:
            before_cont = content.split("【散文续写】", 1)[0]
            after_cont = content.split("【散文续写】", 1)[1]
            # 风格诊断（可选）
            if "【风格诊断】" in before_cont:
                style_content = before_cont.split("【风格诊断】", 1)[1].strip()
            # 续写正文，去掉尾部杂质
            continuation_content = after_cont.strip()
        else:
            # 兜底：未找到明确标记，尝试整段作为续写
            continuation_content = content.strip()
        
        # 去除多余的提示语句（常见污染）
        noisy_prefixes = [
            "两个部分，无需任何其他内容。",
            "请完成以下任务：",
        ]
        for p in noisy_prefixes:
            if continuation_content.startswith(p):
                continuation_content = continuation_content[len(p):].lstrip()
        
        return {
            "style": style_content,
            "continuation": continuation_content
        }

    def create_prompt(self, original_text: str, plot_summary: str = "") -> str:
        """创建测试提示（与训练模板保持一致）"""
        prompt = f"""<|im_start|>system
你是一位专业的朱自清散文风格续写助手。你的核心职责是：基于对原文风格的精准分析，并严格遵循用户提供的情节概要，创作出无缝衔接、文风仿真的续写内容。

**核心原则：**
1.  **风格诊断**：精准捕捉原文在修辞、节奏与情感上的典型特征，分析需简洁、切中要害。
2.  **指令遵循**：将"情节概要"视为绝对指令。续写内容必须完全符合其要求，严禁自行构思情节走向。
3.  **节点选择规则**：如果"情节概要"中列出了多个编号节点（如1. 2. 3.），**你的唯一任务是依据最后一个节点（即最新、最当前的情节指令）进行续写**。其他节点仅作为背景参考以确保连贯性。
4.  **连贯性保障**：需确保续写内容与此前所有情节逻辑自洽，绝对避免出现前后矛盾。
5.  **文风把控**：模仿朱自清先生质朴含蓄、清新自然的文风，避免辞藻堆砌和过度抒情。

你的输出必须严格限定为【风格诊断】和【散文续写】两个部分，无需任何其他内容。
<|im_end|>
<|im_start|>user
请完成以下任务：
1.  **分析**下方"原文"的文学风格。
2.  **续写**下文，必须严格遵循"情节概要"的指令。

**原文**：
{original_text}

**情节概要**：
{plot_summary}

**重要规则**：如上所列，若情节概要以编号形式呈现，你的续写必须**严格依据最后一个编号节点的指令**执行。其他节点用于提供上下文，确保逻辑自洽。

**输出格式**（必须严格遵守）：
【风格诊断】
（你的分析）
【散文续写】
（你的续写）
<|im_end|>
<|im_start|>assistant
"""
        return prompt
        
    def generate_response(self, prompt: str, max_length: int = 1024, temperature: float = 0.7) -> str:
        """生成回应"""
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # 生成
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
            generation_time = time.time() - start_time
            
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取第一个 assistant 段并在第一个 <|im_end|> 处截断
        if "<|im_start|>assistant" in response:
            assistant_response = response.split("<|im_start|>assistant", 1)[1]
            if "<|im_end|>" in assistant_response:
                assistant_response = assistant_response.split("<|im_end|>", 1)[0]
            # 若assistant段中意外包含后续对话块，进一步截断
            for marker in ["<|im_start|>user", "<|im_start|>system", "<|im_start|>assistant"]:
                if marker in assistant_response:
                    assistant_response = assistant_response.split(marker, 1)[0]
            response = assistant_response.strip()
        
        logger.info(f"生成耗时: {generation_time:.2f}秒")
        return response
    
    def build_plot_summary_with_nodes(self, narrative_nodes: List[Dict], current_node_index: int) -> str:
        """构建包含所有节点的情节概要，突出当前节点"""
        if not narrative_nodes or current_node_index >= len(narrative_nodes):
            return ""
        
        plot_lines = []
        
        # 添加所有节点，用编号标识
        for i, node in enumerate(narrative_nodes):
            node_desc = node.get('description', '')
            if i <= current_node_index:
                # 已完成或当前节点
                if i == current_node_index:
                    # 当前要生成的节点，用特殊标记
                    plot_lines.append(f"{i+1}. 【当前续写目标】{node_desc}")
                else:
                    # 之前的节点，作为背景
                    plot_lines.append(f"{i+1}. {node_desc}")
            else:
                # 未来节点，作为整体规划参考（可选）
                plot_lines.append(f"{i+1}. （后续规划）{node_desc}")
        
        return "\n".join(plot_lines)
    
    def generate_complete_essay_optimized(self, topic: str) -> Dict[str, Any]:
        """优化版：只保留前一段落 + 完整情节节点信息"""
        result = {
            "topic": topic,
            "narrative_planning": None,
            "generated_beginning": "",
            "narrative_nodes": [],
            "full_essay_sections": [],
            "success": False,
            "error": None
        }
        
        try:
            # 1. 获取开头与情节规划（优先使用封装函数）
            logger.info(f"开始为主题 '{topic}' 获取情节规划（封装函数）")
            beginning, outline_dict = get_beginning_and_outline(topic)
            if not beginning:
                logger.warning("封装函数获取失败，回退到内置客户端")
                planning_result = self.api_client.generate_narrative_outline(topic)
                if not planning_result["success"]:
                    error_msg = planning_result.get('error', '未知错误')
                    raw_content = planning_result.get('raw_content', '')
                    result["error"] = f"情节规划生成失败: {error_msg}"
                    if raw_content:
                        logger.error(f"API原始返回内容: {raw_content[:500]}...")
                        result["api_raw_content"] = raw_content
                    return result
                planning_data = planning_result["data"]
                result["narrative_planning"] = planning_data
                result["generated_beginning"] = planning_data.get("generated_beginning", "")
                result["narrative_nodes"] = planning_data.get("narrative_outline", [])
            else:
                # 将字典转换为列表形式，保持下游结构一致
                narrative_outline = [{"node_id": k, "description": v} for k, v in outline_dict.items()]
                result["narrative_planning"] = {
                    "generated_beginning": beginning,
                    "narrative_outline": narrative_outline
                }
                result["generated_beginning"] = beginning
                result["narrative_nodes"] = narrative_outline
            
            logger.info(f"情节规划生成成功，包含 {len(result['narrative_nodes'])} 个节点")
            
            # 2. 逐个生成段落，每次只使用前一段落 + 完整节点信息
            full_essay_sections = []
            previous_section = result["generated_beginning"]  # 前一段落
            
            for i, node in enumerate(result["narrative_nodes"]):
                logger.info(f"生成第 {i+1} 个节点的散文段落...")
                logger.info(f"当前节点描述: {node['description']}")
                logger.info(f"前一段落长度: {len(previous_section)} 字符")
                
                # 构建包含所有节点信息的情节概要，突出当前节点
                complete_plot_summary = self.build_plot_summary_with_nodes(result["narrative_nodes"], i)
                
                # 创建提示：使用前一段落作为原文 + 完整节点信息作为情节概要
                prompt = self.create_prompt(previous_section, complete_plot_summary)
                
                # 生成当前节点的散文段落
                response = self.generate_response(prompt, max_length=512)
                parsed = self.parse_sections(response)
                
                section_result = {
                    "node_id": node["node_id"],
                    "node_description": node["description"],
                    "previous_section_length": len(previous_section),
                    "complete_plot_summary": complete_plot_summary,
                    "style_analysis": parsed.get("style", ""),
                    "continuation": parsed.get("continuation", ""),
                    "full_response": response
                }
                
                full_essay_sections.append(section_result)
                
                # 更新前一段落：使用新生成的段落作为下一次的上下文
                if parsed.get("continuation"):
                    previous_section = parsed["continuation"]
                    logger.info(f"已更新前一段落，新长度: {len(previous_section)} 字符")
                else:
                    logger.warning(f"第 {i+1} 个节点未生成有效内容，保持前一段落不变")
                    
            result["full_essay_sections"] = full_essay_sections
            result["success"] = True
            
            logger.info("优化版完整散文生成成功")
            return result
            
        except Exception as e:
            logger.error(f"生成完整散文时出错: {e}")
            result["error"] = str(e)
            return result
    
    def manage_context_length(self, accumulated_text: str, max_length: int = 1500, strategy: str = "smart") -> str:
        """智能管理上下文长度，避免超出模型输入限制
        
        Args:
            accumulated_text: 累积的文本
            max_length: 最大长度限制
            strategy: 管理策略
                - "smart": 智能截断，保持重要内容
                - "sliding": 滑动窗口，保留最近内容
                - "summary": 摘要压缩（简化版）
                - "none": 不处理（可能超长）
        """
        if len(accumulated_text) <= max_length:
            return accumulated_text
        
        if strategy == "none":
            logger.warning(f"文本长度 {len(accumulated_text)} 超过限制 {max_length}，但选择不处理")
            return accumulated_text
        
        elif strategy == "sliding":
            # 滑动窗口：保留最后的内容
            return self._sliding_window_truncate(accumulated_text, max_length)
        
        elif strategy == "summary":
            # 简化版摘要：保留开头和最近的内容
            return self._summary_truncate(accumulated_text, max_length)
        
        else:  # smart策略
            # 智能截断：尝试保持文本的连贯性和重要信息
            return self._smart_truncate(accumulated_text, max_length)
    
    def _sliding_window_truncate(self, text: str, max_length: int) -> str:
        """滑动窗口截断：保留最后的内容，在合适位置截断"""
        truncated = text[-max_length:]
        
        # 寻找最佳截断点
        best_cut = self._find_best_cut_point(truncated)
        if best_cut > 0:
            truncated = truncated[best_cut:]
            
        logger.info(f"滑动窗口截断：从 {len(text)} 字符截断到 {len(truncated)} 字符")
        return truncated.strip()
    
    def _summary_truncate(self, text: str, max_length: int) -> str:
        """摘要式截断：保留开头重要信息 + 最近内容"""
        # 保留开头部分（约1/3）
        head_length = max_length // 3
        head_part = text[:head_length]
        
        # 保留结尾部分（约2/3）
        tail_length = max_length - head_length - 10  # 留10字符给连接符
        tail_part = text[-tail_length:]
        
        # 在合适位置截断
        head_cut = self._find_best_cut_point(head_part, reverse=True)
        if head_cut < len(head_part) - 50:  # 确保不会切得太短
            head_part = head_part[:head_cut]
        
        tail_cut = self._find_best_cut_point(tail_part)
        if tail_cut > 0:
            tail_part = tail_part[tail_cut:]
        
        result = head_part + "\n\n[...]\n\n" + tail_part
        logger.info(f"摘要截断：从 {len(text)} 字符压缩到 {len(result)} 字符（保留开头+结尾）")
        return result
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """智能截断：综合考虑段落、句子边界和内容重要性"""
        # 先尝试滑动窗口
        truncated = text[-max_length:]
        
        # 寻找最佳截断点
        best_cut = self._find_best_cut_point(truncated)
        if best_cut > 0:
            truncated = truncated[best_cut:]
        
        # 如果截断后太短，尝试摘要式处理
        if len(truncated) < max_length * 0.7:
            return self._summary_truncate(text, max_length)
        
        logger.info(f"智能截断：从 {len(text)} 字符截断到 {len(truncated)} 字符")
        return truncated.strip()
    
    def _find_best_cut_point(self, text: str, reverse: bool = False) -> int:
        """寻找最佳截断点，优先选择段落、句子边界"""
        if reverse:
            # 从后往前找截断点（用于头部截断）
            text_to_search = text[::-1]  # 反转字符串
            # 段落分隔符（反转后）
            paragraph_pos = text_to_search.find('\n\n')
            if paragraph_pos > 50:
                return len(text) - paragraph_pos - 2
            
            # 句号（反转后）
            sentence_pos = text_to_search.find('。')
            if sentence_pos > 20:
                return len(text) - sentence_pos
            
            return len(text)
        else:
            # 从前往后找截断点
            # 优先选择段落分隔符
            paragraph_pos = text.find('\n\n')
            if paragraph_pos > 50:  # 确保不会截断得太早
                return paragraph_pos + 2
            
            # 其次选择句号
            sentence_pos = text.find('。')
            if sentence_pos > 20:
                return sentence_pos + 1
            
            # 最后选择逗号或分号
            for punct in ['；', '，']:
                pos = text.find(punct)
                if pos > 10:
                    return pos + 1
            
            return 0
    
    def generate_complete_essay(self, topic: str, context_strategy: str = "smart", max_context_length: int = 1500) -> Dict[str, Any]:
        """根据主题生成完整的散文（包含情节规划和续写）
        
        Args:
            topic: 散文主题
            context_strategy: 上下文管理策略 ("smart", "sliding", "summary", "none")
            max_context_length: 最大上下文长度
        """
        result = {
            "topic": topic,
            "context_strategy": context_strategy,
            "max_context_length": max_context_length,
            "narrative_planning": None,
            "generated_beginning": "",
            "narrative_nodes": [],
            "full_essay_sections": [],
            "success": False,
            "error": None
        }
        
        try:
            # 1. 获取开头与情节规划（优先使用封装函数）
            logger.info(f"开始为主题 '{topic}' 获取情节规划（封装函数）")
            beginning, outline_dict = get_beginning_and_outline(topic)
            if not beginning:
                logger.warning("封装函数获取失败，回退到内置客户端")
                planning_result = self.api_client.generate_narrative_outline(topic)
                if not planning_result["success"]:
                    error_msg = planning_result.get('error', '未知错误')
                    raw_content = planning_result.get('raw_content', '')
                    result["error"] = f"情节规划生成失败: {error_msg}"
                    if raw_content:
                        logger.error(f"API原始返回内容: {raw_content[:500]}...")
                        result["api_raw_content"] = raw_content
                    return result
                planning_data = planning_result["data"]
                result["narrative_planning"] = planning_data
                result["generated_beginning"] = planning_data.get("generated_beginning", "")
                result["narrative_nodes"] = planning_data.get("narrative_outline", [])
            else:
                # 将字典转换为列表形式，保持下游结构一致
                narrative_outline = [{"node_id": k, "description": v} for k, v in outline_dict.items()]
                result["narrative_planning"] = {
                    "generated_beginning": beginning,
                    "narrative_outline": narrative_outline
                }
                result["generated_beginning"] = beginning
                result["narrative_nodes"] = narrative_outline
            
            logger.info(f"情节规划生成成功，包含 {len(result['narrative_nodes'])} 个节点")
            
            # 2. 使用朱自清模型为每个节点生成散文段落
            full_essay_sections = []
            # 初始文本为API生成的开头段落
            accumulated_text = result["generated_beginning"]
            
            for i, node in enumerate(result["narrative_nodes"]):
                logger.info(f"生成第 {i+1} 个节点的散文段落...")
                logger.info(f"当前节点描述: {node['description']}")
                logger.info(f"当前累积文本长度: {len(accumulated_text)} 字符")
                
                # 管理上下文长度，避免过长
                managed_text = self.manage_context_length(
                    accumulated_text, 
                    max_length=max_context_length, 
                    strategy=context_strategy
                )
                
                # 创建提示：
                # - 原文：使用管理后的累积文本作为上下文
                # - 情节概要：只使用当前节点的描述
                prompt = self.create_prompt(managed_text, node["description"])
                
                # 生成当前节点的散文段落
                response = self.generate_response(prompt, max_length=512)
                parsed = self.parse_sections(response)
                
                section_result = {
                    "node_id": node["node_id"],
                    "node_description": node["description"],
                    "context_length": len(managed_text),  # 记录使用的上下文长度
                    "original_context_length": len(accumulated_text),  # 记录原始累积长度
                    "style_analysis": parsed.get("style", ""),
                    "continuation": parsed.get("continuation", ""),
                    "full_response": response
                }
                
                full_essay_sections.append(section_result)
                
                # 更新累积文本：将新生成的段落加入到完整累积文本中
                if parsed.get("continuation"):
                    accumulated_text = accumulated_text + "\n\n" + parsed["continuation"]
                    logger.info(f"已将第 {i+1} 个节点的内容加入累积文本，新长度: {len(accumulated_text)} 字符")
                else:
                    logger.warning(f"第 {i+1} 个节点未生成有效内容")
                    
            result["full_essay_sections"] = full_essay_sections
            result["success"] = True
            
            logger.info("完整散文生成成功")
            return result
            
        except Exception as e:
            logger.error(f"生成完整散文时出错: {e}")
            result["error"] = str(e)
            return result
        
    def interactive_test(self):
        """交互式测试"""
        logger.info("进入交互式测试模式")
        logger.info("输入 'quit' 或 'exit' 退出")
        logger.info("输入 'clear' 清屏")
        logger.info("输入 'topic:主题内容' 生成完整散文（优化版）")
        logger.info("输入 'topic_old:主题内容 [strategy:策略] [length:长度]' 使用旧版本")
        logger.info("  优化版：只保留前一段落+完整情节节点，更高效准确")
        logger.info("  旧版本：累积所有内容，支持多种截断策略")
        logger.info("-" * 50)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n请输入 (朱自清散文片段 | topic:主题): ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    logger.info("退出交互式测试")
                    break
                    
                if user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                    
                if not user_input:
                    continue
                
                # 检查是否是主题生成模式
                if user_input.startswith('topic:'):
                    # 优化版：只使用主题，不需要额外参数
                    topic = user_input[6:].strip()
                    if not topic:
                        logger.warning("主题不能为空")
                        continue
                    
                    logger.info(f"开始根据主题生成完整散文（优化版）: {topic}")
                    result = self.generate_complete_essay_optimized(topic)
                    
                    if result["success"]:
                        print("\n" + "="*80)
                        print(f"主题: {result['topic']} （优化版）")
                        print("="*80)
                        
                        print(f"\n【生成的开头段落】")
                        print(result["generated_beginning"])
                        
                        print(f"\n【情节规划】")
                        for node in result["narrative_nodes"]:
                            print(f"节点{node['node_id']}: {node['description']}")
                        
                        print(f"\n【完整散文生成过程】")
                        full_essay = result["generated_beginning"]
                        for section in result["full_essay_sections"]:
                            print(f"\n--- 节点{section['node_id']}: {section['node_description']} ---")
                            print(f"【前一段落长度】{section['previous_section_length']} 字符")
                            print(f"【使用的情节概要】")
                            print(section['complete_plot_summary'])
                            print(f"【风格分析】{section['style_analysis']}")
                            print(f"【续写内容】{section['continuation']}")
                            if section['continuation']:
                                full_essay += "\n\n" + section['continuation']
                        
                        print(f"\n【最终完整散文】")
                        print("="*60)
                        print(full_essay)
                        print("="*80)
                        
                    else:
                        print(f"\n❌ 生成失败: {result['error']}")
                    
                    continue
                
                elif user_input.startswith('topic_old:'):
                    # 旧版本：支持策略参数
                    parts = user_input[10:].split()
                    if not parts:
                        logger.warning("主题不能为空")
                        continue
                    
                    topic = parts[0]
                    context_strategy = "smart"  # 默认策略
                    max_context_length = 1500   # 默认长度
                    
                    # 解析可选参数
                    for part in parts[1:]:
                        if part.startswith('strategy:'):
                            strategy_name = part.split(':', 1)[1]
                            if strategy_name in ["smart", "sliding", "summary", "none"]:
                                context_strategy = strategy_name
                            else:
                                print(f"⚠️  未知策略 '{strategy_name}'，使用默认策略 'smart'")
                        elif part.startswith('length:'):
                            try:
                                max_context_length = int(part.split(':', 1)[1])
                                if max_context_length < 500:
                                    max_context_length = 500
                                    print("⚠️  长度限制过小，调整为500")
                            except ValueError:
                                print("⚠️  长度参数无效，使用默认长度1500")
                    
                    logger.info(f"开始根据主题生成完整散文（旧版本）: {topic}")
                    logger.info(f"上下文策略: {context_strategy}, 最大长度: {max_context_length}")
                    
                    result = self.generate_complete_essay(topic, context_strategy, max_context_length)
                    
                    if result["success"]:
                        print("\n" + "="*80)
                        print(f"主题: {result['topic']}")
                        print("="*80)
                        
                        print(f"\n【生成的开头段落】")
                        print(result["generated_beginning"])
                        
                        print(f"\n【情节规划】")
                        for node in result["narrative_nodes"]:
                            print(f"节点{node['node_id']}: {node['description']}")
                        
                        print(f"\n【完整散文】")
                        full_essay = result["generated_beginning"]
                        for section in result["full_essay_sections"]:
                            print(f"\n--- 节点{section['node_id']}: {section['node_description']} ---")
                            print(f"【上下文长度】原始: {section['original_context_length']} 字符, 使用: {section['context_length']} 字符")
                            print(f"【风格分析】{section['style_analysis']}")
                            print(f"【续写内容】{section['continuation']}")
                            if section['continuation']:
                                full_essay += "\n\n" + section['continuation']
                        
                        print(f"\n【最终完整散文】")
                        print("="*60)
                        print(full_essay)
                        print("="*80)
                        
                    else:
                        print(f"\n❌ 生成失败: {result['error']}")
                    
                    continue
                
                # 原始模式：散文片段续写
                original_text = user_input
                
                # 强制要求情节概要
                plot_summary = input("\n请输入情节概要: ").strip()
                if not plot_summary:
                    logger.warning("情节概要不能为空")
                    continue
                    
                # 生成回应
                prompt = self.create_prompt(original_text, plot_summary)
                response = self.generate_response(prompt)
                parsed = self.parse_sections(response)
                
                # 显示结果
                print("\n" + "="*60)
                print("模型回应:")
                print("="*60)
                print("【风格诊断】\n" + (parsed["style"] or "(未解析到)") )
                print("\n【散文续写】\n" + (parsed["continuation"] or "(未解析到)") )
                print("="*60)
                
            except KeyboardInterrupt:
                logger.info("\n用户中断，退出测试")
                break
            except Exception as e:
                logger.error(f"生成过程中出现错误: {e}")
                
    def load_test_data(self, test_data_path: str) -> List[Dict[str, Any]]:
        """加载测试数据"""
        if not os.path.exists(test_data_path):
            logger.error(f"测试数据文件不存在: {test_data_path}")
            return []
            
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 过滤有效数据
        valid_data = []
        for item in data:
            if (item.get("原文", "").strip() and 
                item.get("风格诊断", "").strip() and 
                item.get("散文续写", "").strip() and
                item.get("情节概要", "").strip()):
                if not (item["风格诊断"].startswith("处理出错") or 
                       item["散文续写"].startswith("处理出错") or
                       item["情节概要"].startswith("处理出错")):
                    valid_data.append(item)
                    
        logger.info(f"加载测试数据: {len(valid_data)} 条")
        return valid_data
        
    def auto_test(self, test_data_path: str = "zhu_ziqing_continuations.json", 
                  num_samples: int = 5, save_results: bool = True):
        """自动测试"""
        logger.info("开始自动测试")
        
        # 加载测试数据
        test_data = self.load_test_data(test_data_path)
        if not test_data:
            logger.error("没有可用的测试数据")
            return
            
        # 随机选择测试样本
        if num_samples > len(test_data):
            num_samples = len(test_data)
            
        test_samples = random.sample(test_data, num_samples)
        logger.info(f"选择 {num_samples} 个样本进行测试")
        
        results = []
        
        for i, sample in enumerate(test_samples, 1):
            logger.info(f"测试样本 {i}/{num_samples}")
            
            original_text = sample["原文"]
            expected_style = sample["风格诊断"]
            expected_continuation = sample["散文续写"]
            plot_summary = sample.get("情节概要", "")
            
            # 生成回应
            prompt = self.create_prompt(original_text, plot_summary)
            response = self.generate_response(prompt)
            parsed = self.parse_sections(response)
            
            # 保存结果
            result = {
                "sample_id": i,
                "original_text": original_text,
                "expected_style": expected_style,
                "expected_continuation": expected_continuation,
                "generated_response": response,
                "parsed_style": parsed.get("style", ""),
                "parsed_continuation": parsed.get("continuation", ""),
                "prompt": prompt
            }
            results.append(result)
            
            # 显示结果
            print(f"\n{'='*20} 测试样本 {i} {'='*20}")
            print(f"原文: {original_text[:100]}...")
            print(f"\n期望风格诊断: {expected_style[:100]}...")
            print(f"\n期望续写: {expected_continuation[:100]}...")
            print(f"\n模型生成(解析后):")
            print("【风格诊断】\n" + (parsed.get("style", "")[:500] or "(未解析到)") )
            print("\n【散文续写】\n" + (parsed.get("continuation", "")[:800] or "(未解析到)") )
            print("="*60)
            
        # 保存测试结果
        if save_results:
            output_file = f"test_results_{int(time.time())}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"测试结果已保存到: {output_file}")
            
        return results
        
    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估测试结果"""
        logger.info("开始评估测试结果")
        
        # 简单的评估指标
        metrics = {
            "total_samples": len(results),
            "avg_response_length": 0,
            "style_analysis_rate": 0,
            "continuation_rate": 0
        }
        
        total_length = 0
        style_count = 0
        continuation_count = 0
        
        for result in results:
            response = result["generated_response"]
            total_length += len(response)
            
            # 检查是否包含风格分析
            if "【风格诊断】" in response or "风格" in response:
                style_count += 1
                
            # 检查是否包含续写
            if "【散文续写】" in response or "续写" in response:
                continuation_count += 1
                
        metrics["avg_response_length"] = total_length / len(results) if results else 0
        metrics["style_analysis_rate"] = style_count / len(results) if results else 0
        metrics["continuation_rate"] = continuation_count / len(results) if results else 0
        
        # 显示评估结果
        print("\n" + "="*40)
        print("评估结果:")
        print("="*40)
        print(f"测试样本数: {metrics['total_samples']}")
        print(f"平均回应长度: {metrics['avg_response_length']:.1f} 字符")
        print(f"风格分析完成率: {metrics['style_analysis_rate']:.2%}")
        print(f"续写完成率: {metrics['continuation_rate']:.2%}")
        print("="*40)
        
        return metrics

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="朱自清散文续写模型测试工具")
    parser.add_argument("--mode", choices=["interactive", "auto", "both", "oneshot", "topic", "topic_opt"], 
                       default="interactive", help="测试模式")
    parser.add_argument("--base_model", default="qwen/Qwen2.5-7B-Instruct", 
                       help="基础模型名称")
    parser.add_argument("--lora_path", default="../../data/models/qwen_zhu_ziqing_lora", 
                       help="LoRA适配器路径")
    parser.add_argument("--test_data", default="../../data/raw/zhu_ziqing_continuations.json", 
                       help="测试数据路径")
    parser.add_argument("--num_samples", type=int, default=5, 
                       help="自动测试样本数")
    parser.add_argument("--cache_dir", default="../../data/models", 
                       help="模型缓存目录")
    parser.add_argument("--original", default=None, help="oneshot模式：原文片段")
    parser.add_argument("--summary", default=None, help="oneshot模式：情节概要")
    parser.add_argument("--topic", default=None, help="topic模式：主题内容")
    parser.add_argument("--context_strategy", default="smart", 
                       choices=["smart", "sliding", "summary", "none"],
                       help="上下文管理策略（仅topic模式）")
    parser.add_argument("--max_context_length", type=int, default=1500,
                       help="最大上下文长度（仅topic模式）")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ZhuZiQingModelTester(
        base_model_name=args.base_model,
        lora_adapter_path=args.lora_path,
        cache_dir=args.cache_dir
    )
    
    # 加载模型
    tester.load_model()
    
    # 根据模式执行测试
    if args.mode == "interactive":
        tester.interactive_test()
    elif args.mode == "auto":
        results = tester.auto_test(args.test_data, args.num_samples)
        if results:
            tester.evaluate_results(results)
    elif args.mode == "both":
        # 先自动测试
        logger.info("开始自动测试...")
        results = tester.auto_test(args.test_data, args.num_samples)
        if results:
            tester.evaluate_results(results)
        
        # 再交互测试
        input("\n按回车键进入交互式测试...")
        tester.interactive_test()
    elif args.mode == "oneshot":
        # 单次生成：从命令行读取原文与情节概要
        if not args.original or not args.summary:
            logger.error("oneshot模式需要 --original 与 --summary 参数")
            return
        prompt = tester.create_prompt(args.original, args.summary)
        response = tester.generate_response(prompt)
        parsed = tester.parse_sections(response)
        print("\n" + "="*60)
        print("【风格诊断】\n" + (parsed.get("style", "") or "(未解析到)") )
        print("\n【散文续写】\n" + (parsed.get("continuation", "") or "(未解析到)") )
        print("="*60)
    elif args.mode == "topic":
        # 主题生成完整散文模式
        if not args.topic:
            logger.error("topic模式需要 --topic 参数")
            return
        
        logger.info(f"根据主题生成完整散文: {args.topic}")
        logger.info(f"上下文策略: {args.context_strategy}, 最大长度: {args.max_context_length}")
        result = tester.generate_complete_essay(args.topic, args.context_strategy, args.max_context_length)
        
        if result["success"]:
            print("\n" + "="*80)
            print(f"主题: {result['topic']}")
            print("="*80)
            
            print(f"\n【生成的开头段落】")
            print(result["generated_beginning"])
            
            print(f"\n【情节规划】")
            for node in result["narrative_nodes"]:
                print(f"节点{node['node_id']}: {node['description']}")
            
            print(f"\n【完整散文】")
            full_essay = result["generated_beginning"]
            for section in result["full_essay_sections"]:
                print(f"\n--- 节点{section['node_id']}: {section['node_description']} ---")
                print(f"【上下文长度】原始: {section['original_context_length']} 字符, 使用: {section['context_length']} 字符")
                print(f"【风格分析】{section['style_analysis']}")
                print(f"【续写内容】{section['continuation']}")
                if section['continuation']:
                    full_essay += "\n\n" + section['continuation']
            
            print(f"\n【最终完整散文】")
            print("="*60)
            print(full_essay)
            print("="*80)
            
            # 仅终端输出，不再保存到文件
            
        else:
            print(f"\n❌ 生成失败: {result['error']}")
    elif args.mode == "topic_opt":
        # 优化版主题生成完整散文模式
        if not args.topic:
            logger.error("topic_opt模式需要 --topic 参数")
            return
        
        logger.info(f"根据主题生成完整散文（优化版）: {args.topic}")
        result = tester.generate_complete_essay_optimized(args.topic)
        
        if result["success"]:
            print("\n" + "="*80)
            print(f"主题: {result['topic']} （优化版）")
            print("="*80)
            
            print(f"\n【生成的开头段落】")
            print(result["generated_beginning"])
            
            print(f"\n【情节规划】")
            for node in result["narrative_nodes"]:
                print(f"节点{node['node_id']}: {node['description']}")
            
            print(f"\n【完整散文生成过程】")
            full_essay = result["generated_beginning"]
            for section in result["full_essay_sections"]:
                print(f"\n--- 节点{section['node_id']}: {section['node_description']} ---")
                print(f"【前一段落长度】{section['previous_section_length']} 字符")
                print(f"【使用的情节概要】")
                print(section['complete_plot_summary'])
                print(f"【风格分析】{section['style_analysis']}")
                print(f"【续写内容】{section['continuation']}")
                if section['continuation']:
                    full_essay += "\n\n" + section['continuation']
            
            print(f"\n【最终完整散文】")
            print("="*60)
            print(full_essay)
            print("="*80)
            
            # 仅终端输出，不再保存到文件
            
        else:
            print(f"\n❌ 生成失败: {result['error']}")

if __name__ == "__main__":
    main()
