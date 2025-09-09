#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
朱自清散文续写 - Qwen2.5-7B Instruct LoRA微调
使用ModelScope框架进行微调
"""

import os
import json
import torch
import logging
from typing import List, Dict, Any, Tuple
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from modelscope import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import numpy as np
from sklearn.model_selection import train_test_split

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZhuZiQingDatasetProcessor:
    """朱自清散文数据集处理器"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.tokenizer = None
        
    def load_data(self) -> List[Dict[str, Any]]:
        """加载原始数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"加载了 {len(data)} 条数据")
        return data
    
    def filter_valid_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤有效数据"""
        valid_data = []
        for item in data:
            # 检查必要字段是否存在且不为空
            if (item.get("原文", "").strip() and 
                item.get("风格诊断", "").strip() and 
                item.get("散文续写", "").strip()):
                
                # 过滤掉错误信息
                plot_summary = item.get("情节概要", "")
                if not (item["风格诊断"].startswith("处理出错") or 
                       item["散文续写"].startswith("处理出错") or
                       (plot_summary and plot_summary.startswith("处理出错"))):
                    valid_data.append(item)
        
        logger.info(f"过滤后有效数据: {len(valid_data)} 条")
        return valid_data
    
    def create_prompt_template(self, original_text: str, plot_summary: str, style_analysis: str, continuation: str) -> str:
        """创建训练用的提示模板"""
        prompt = f"""<|im_start|>system
你是一位专业的朱自清散文风格续写助手。你的核心职责是：基于对原文风格的精准分析，并严格遵循用户提供的情节概要，创作出无缝衔接、文风仿真的续写内容。

**核心原则：**
1.  **风格诊断**：精准捕捉原文在修辞、节奏与情感上的典型特征，分析需简洁、切中要害。
2.  **指令遵循**：将“情节概要”视为绝对指令。续写内容必须完全符合其要求，严禁自行构思情节走向。
3.  **节点选择规则**：如果“情节概要”中列出了多个编号节点（如1. 2. 3.），**你的唯一任务是依据最后一个节点（即最新、最当前的情节指令）进行续写**。其他节点仅作为背景参考以确保连贯性。
4.  **连贯性保障**：需确保续写内容与此前所有情节逻辑自洽，绝对避免出现前后矛盾。
5.  **文风把控**：模仿朱自清先生质朴含蓄、清新自然的文风，避免辞藻堆砌和过度抒情。

你的输出必须严格限定为【风格诊断】和【散文续写】两个部分，无需任何其他内容。
<|im_end|>
<|im_start|>user
请完成以下任务：
1.  **分析**下方“原文”的文学风格。
2.  **续写**下文，必须严格遵循“情节概要”的指令。

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
【风格诊断】
{style_analysis}

【散文续写】
{continuation}
<|im_end|>"""
        return prompt
    
    def prepare_training_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """准备训练数据"""
        training_data = []
        
        for item in data:
            prompt = self.create_prompt_template(
                item["原文"],
                item.get("情节概要", ""),
                item["风格诊断"], 
                item["散文续写"]
            )
            training_data.append({"text": prompt})
        
        logger.info(f"准备训练数据: {len(training_data)} 条")
        return training_data
    
    def split_data(self, data: List[Dict[str, Any]], train_ratio: float = 0.8, random_state: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """将数据分割为训练集和验证集"""
        if len(data) < 2:
            logger.warning("数据量太少，无法分割验证集，将全部用作训练集")
            return data, []
        
        train_data, val_data = train_test_split(
            data, 
            train_size=train_ratio, 
            random_state=random_state,
            shuffle=True
        )
        
        logger.info(f"数据分割完成 - 训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")
        return train_data, val_data
    

    
    def create_dataset(self, training_data: List[Dict[str, str]]) -> Dataset:
        """创建HuggingFace数据集"""
        return Dataset.from_list(training_data)

class QwenLoRATrainer:
    """Qwen2.5-7B LoRA训练器"""
    
    def __init__(self, model_name: str = "qwen/Qwen2.5-7B-Instruct", cache_dir: str = "./models"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        self.lora_config = None
        
        # 确保模型缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def setup_tokenizer(self):
        """设置分词器"""
        logger.info(f"从ModelScope下载并加载分词器: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right",
            cache_dir=self.cache_dir
        )
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("分词器加载完成")
        
    def setup_model(self, load_in_8bit: bool = True):
        """设置模型"""
        logger.info(f"从ModelScope下载并加载模型: {self.model_name}")
        logger.info(f"模型将缓存到: {self.cache_dir}")
        
        # 模型配置
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": {
                "model.embed_tokens": 0,
                "model.layers.0": 0,
                "model.layers.1": 0,
                "model.layers.2": 0,
                "model.layers.3": 0,
                "model.layers.4": 0,
                "model.layers.5": 0,
                "model.layers.6": 0,
                "model.layers.7": 0,
                "model.layers.8": 0,
                "model.layers.9": 0,
                "model.layers.10": 1,
                "model.layers.11": 1,
                "model.layers.12": 1,
                "model.layers.13": 1,
                "model.layers.14": 1,
                "model.layers.15": 1,
                "model.layers.16": 1,
                "model.layers.17": 1,
                "model.layers.18": 1,
                "model.layers.19": 1,
                "model.layers.20": 3,
                "model.layers.21": 3,
                "model.layers.22": 3,
                "model.layers.23": 3,
                "model.layers.24": 3,
                "model.layers.25": 3,
                "model.layers.26": 3,
                "model.layers.27": 3,
                "lm_head": 3,
                "model.norm": 3
            },
            "cache_dir": self.cache_dir
        }
        
        if load_in_8bit:
            # 使用新的量化配置方式
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            model_kwargs["quantization_config"] = quantization_config
        else:
            # 4bit量化配置
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # 准备模型进行训练 - 冻结原始参数
        self.model = prepare_model_for_kbit_training(self.model)
        
        # 确保只训练LoRA参数，冻结原始模型参数
        for name, param in self.model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
                logger.info(f"可训练参数: {name}")
        
        logger.info("模型加载完成")
        
    def setup_lora_config(self):
        """设置LoRA配置"""
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        )
        
        logger.info("LoRA配置设置完成")
        
    def apply_lora(self):
        """应用LoRA到模型"""
        logger.info("应用LoRA...")
        self.model = get_peft_model(self.model, self.lora_config)
        
        # 再次确保只训练LoRA参数
        for name, param in self.model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 打印可训练参数统计
        self.model.print_trainable_parameters()
        
        # 打印详细的参数信息
        trainable_params = 0
        all_params = 0
        for name, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                logger.info(f"可训练参数: {name} - {param.numel():,} 参数")
        
        logger.info(f"总参数: {all_params:,}, 可训练参数: {trainable_params:,}")
        logger.info(f"可训练参数比例: {trainable_params/all_params*100:.2f}%")
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """对数据集进行分词"""
        logger.info("对数据集进行分词...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
                padding=False,
                return_tensors=None
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info("分词完成")
        return tokenized_dataset
        
    def setup_training_args(self, output_dir: str = "./qwen_zhu_ziqing_lora") -> TrainingArguments:
        """设置训练参数"""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,  # 设置验证batch size
            gradient_accumulation_steps=8,
            warmup_steps=100,
            learning_rate=2e-4,
            weight_decay=0.01,
            logging_steps=10,
            eval_steps=100,  # 每100步进行一次验证
            eval_strategy="steps",  # 按步数进行验证
            save_steps=500,
            save_strategy="steps",
            load_best_model_at_end=True,  # 训练结束时加载最佳模型
            metric_for_best_model="eval_loss",  # 使用验证损失作为最佳模型指标
            greater_is_better=False,  # 损失越小越好
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # 禁用wandb等报告工具
        )
        
    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """开始训练"""
        logger.info("开始训练...")
        
        # 验证只有LoRA参数可训练
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"训练前验证 - 可训练参数: {trainable_params:,}, 总参数: {all_params:,}")
        logger.info(f"可训练参数比例: {trainable_params/all_params*100:.4f}%")
        
        # 对数据集进行分词
        tokenized_train_dataset = self.tokenize_dataset(train_dataset)
        tokenized_eval_dataset = None
        if eval_dataset is not None:
            tokenized_eval_dataset = self.tokenize_dataset(eval_dataset)
            logger.info(f"验证集: {len(tokenized_eval_dataset)} 条")
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 训练参数
        training_args = self.setup_training_args()
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,  # 添加验证集
            data_collator=data_collator
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型 - 只保存LoRA适配器
        logger.info("保存LoRA适配器...")
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # 保存适配器配置信息
        adapter_info = {
            "base_model": self.model_name,
            "adapter_type": "LoRA",
            "lora_config": {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "lora_dropout": self.lora_config.lora_dropout,
                "target_modules": list(self.lora_config.target_modules)  # 转换为list
            },
            "trainable_parameters": trainable_params,
            "total_parameters": all_params,
            "trainable_ratio": trainable_params/all_params*100
        }
        
        with open(os.path.join(training_args.output_dir, "adapter_info.json"), "w", encoding="utf-8") as f:
            json.dump(adapter_info, f, ensure_ascii=False, indent=2)
        
        logger.info("训练完成！")
        logger.info(f"LoRA适配器已保存到: {training_args.output_dir}")
        
    def save_model_info(self, output_dir: str):
        """保存模型信息"""
        model_info = {
            "base_model": self.model_name,
            "adapter_type": "LoRA",
            "task": "朱自清散文续写",
            "lora_config": {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "lora_dropout": self.lora_config.lora_dropout,
                "target_modules": list(self.lora_config.target_modules)  # 转换为list
            },
            "description": "基于Qwen2.5-7B-Instruct的LoRA适配器，专门用于朱自清散文风格分析和续写任务",
            "note": "此文件仅包含LoRA适配器，需要与原始模型配合使用"
        }
        
        with open(os.path.join(output_dir, "model_info.json"), "w", encoding="utf-8") as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 数据路径
    data_path = "zhu_ziqing_continuations.json"
    
    # 1. 处理数据
    processor = ZhuZiQingDatasetProcessor(data_path)
    raw_data = processor.load_data()
    valid_data = processor.filter_valid_data(raw_data)
    
    # 2. 分割训练集和验证集
    train_data, val_data = processor.split_data(valid_data, train_ratio=0.8, random_state=42)
    
    # 3. 准备训练和验证数据集
    train_prepared_data = processor.prepare_training_data(train_data)
    train_dataset = processor.create_dataset(train_prepared_data)
    
    val_dataset = None
    if val_data:
        val_prepared_data = processor.prepare_training_data(val_data)
        val_dataset = processor.create_dataset(val_prepared_data)
        logger.info(f"验证集: {len(val_dataset)} 条")
    else:
        logger.info("未创建验证集")
    
    logger.info(f"训练集: {len(train_dataset)} 条")
    
    # 4. 设置模型和训练器
    trainer = QwenLoRATrainer(
        model_name="qwen/Qwen2.5-7B-Instruct",
        cache_dir="./models"
    )
    trainer.setup_tokenizer()
    trainer.setup_model(load_in_8bit=True)  # 使用8bit量化节省显存
    trainer.setup_lora_config()
    trainer.apply_lora()
    
    # 5. 开始训练（包含验证集）
    trainer.train(train_dataset, val_dataset)
    
    # 6. 保存模型信息
    output_dir = "./qwen_zhu_ziqing_lora"
    trainer.save_model_info(output_dir)
    
    logger.info("所有任务完成！")

if __name__ == "__main__":
    main()
