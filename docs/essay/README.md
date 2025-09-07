# 朱自清散文续写项目 (Sanwen)

这是一个专注于朱自清散文风格续写的深度学习项目，基于Qwen模型进行LoRA微调，实现高质量的中文散文生成。

## 🌟 项目特色

- **朱自清风格仿写**：专门针对朱自清散文风格进行训练
- **情节驱动生成**：支持基于情节概要的散文续写
- **API集成**：集成硅基流动API进行情节规划
- **完整工具链**：从数据处理到模型训练再到推理部署的完整流程

## 📁 项目结构

```
sanwen/
├── sanwen/                 # 主要代码包
│   ├── finetune/          # 模型微调
│   │   └── sft.py         # 监督微调脚本
│   ├── api/               # API集成
│   │   └── test_siliconflow_api.py  # 硅基流动API
│   ├── models/            # 模型推理
│   └── utils/             # 工具函数
├── tools/                 # 工具脚本
│   ├── model_tools/       # 模型工具
│   │   └── test_model.py  # 模型测试脚本
│   ├── data_maker/        # 数据制作工具
│   └── evaluation/        # 评估工具
├── data/                  # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── models/            # 模型文件
├── test/                  # 测试代码
├── docs/                  # 文档
│   ├── finetune/          # 微调文档
│   ├── api/               # API文档
│   └── usage/             # 使用指南
├── run/                   # 运行脚本
└── assets/                # 资源文件
```

## 🚀 快速开始

### 环境安装

```bash
# 克隆项目
git clone <project-url>
cd sanwen

# 安装依赖
pip install -r requirements.txt

# 或者使用开发模式安装
pip install -e .
```

### 模型训练

```bash
# 使用SFT进行微调
cd sanwen/finetune
python sft.py
```

### 生成散文

```bash
# 使用优化版生成完整散文
python tools/model_tools/test_model.py --mode topic_opt --topic "在一个秋日，旧友在茶舍重逢"
```

### API测试

```bash
# 测试硅基流动API
python sanwen/api/test_siliconflow_api.py --test_connection --topic "春天的校园"
```

## 📊 数据说明

- **原始数据**: `data/raw/zhu_ziqing_continuations.json` - 朱自清散文续写训练数据
- **模型文件**: `data/models/qwen_zhu_ziqing_lora/` - LoRA适配器模型

## 🛠️ 主要功能

### 1. 模型微调
- 基于Qwen2.5-7B-Instruct的LoRA微调
- 支持朱自清散文风格学习
- 包含风格诊断和续写双重功能

### 2. 智能生成
- 基于主题的完整散文生成
- 情节规划与续写结合
- 多种上下文管理策略

### 3. API集成
- 硅基流动API集成
- 自动情节规划生成
- 多重JSON解析策略

## 🔧 配置说明

项目使用环境变量进行配置，可以创建 `.env` 文件：

```bash
# API配置
SILICONFLOW_API_KEY=your_api_key_here
MODEL_CACHE_DIR=./data/models
BASE_MODEL_NAME=qwen/Qwen2.5-7B-Instruct
LORA_PATH=./data/models/qwen_zhu_ziqing_lora
```

## 📝 使用示例

### 生成完整散文

```python
from tools.model_tools.test_model import ZhuZiQingModelTester

# 初始化测试器
tester = ZhuZiQingModelTester()
tester.load_model()

# 生成散文
result = tester.generate_complete_essay_optimized("在一个秋日，旧友在茶舍重逢")
if result["success"]:
    print("生成的散文：")
    print(result["generated_beginning"])
    for section in result["full_essay_sections"]:
        print(section["continuation"])
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- 感谢Qwen团队提供的优秀基础模型
- 感谢硅基流动提供的API服务
- 参考了[Tianji](https://github.com/SocialAI-tianji/Tianji)项目的结构设计