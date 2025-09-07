"""
朱自清散文续写项目 (Sanwen)
=============================

这是一个专注于朱自清散文风格续写的深度学习项目，包含：
- 微调 (finetune): 基于Qwen模型的LoRA微调
- API (api): 硅基流动API集成和调用
- 模型 (models): 模型推理和生成工具
- 工具 (utils): 通用工具函数

主要功能：
- 朱自清散文风格分析
- 基于情节概要的散文续写
- 完整散文生成流程

版本: 1.0.0
作者: Kaiwen
"""

__version__ = "1.0.0"
__author__ = "Kaiwen"
__description__ = "朱自清散文续写项目"

from . import finetune
from . import api
from . import models
from . import utils

__all__ = [
    "finetune",
    "api", 
    "models",
    "utils"
]