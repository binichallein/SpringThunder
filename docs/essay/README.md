## 朱自清散文续写（essay）运行说明

本目录包含用于生成/测试散文续写的脚本与数据。核心脚本为 `test_model.py`，既可调用硅基流动 API 获取情节规划，也可在 API 失败时使用本地兜底逻辑。模型加载依赖 ModelScope 与 PEFT。

### 环境准备
- Python 3.9+
- 建议在虚拟环境中安装依赖：
```bash
cd docs/essay
pip install -r requirements.txt
```

### 快速开始
基于“主题”生成完整散文（优化版；推荐）
```bash
cd docs/essay
python test_model.py --mode topic_opt --topic "你的主题"
```

基于“主题”并启用人机协同（HITL）先审核与编辑大纲，再生成：
```bash
cd docs/essay
python test_model.py --mode topic_opt --topic "你的主题" --human_in_loop
# 进入命令行交互：
# 可用命令：
#   show | setbegin <文本> | add <描述> | insert <索引> <描述> |
#   edit <索引> <新描述> | del <索引> | move <源索引> <目标索引> |
#   renumber | done | cancel
```

在 CI/自动化中使用脚本化 HITL 命令（无需人工输入）：
```bash
python test_model.py --mode topic_opt --topic "你的主题" \
  --human_in_loop \
  --hitl_commands "setbegin 新的开头 | add 新增情节节点 | edit 1 修订第一节点 | renumber | done"
```


### 重要参数与路径
- 基础模型与缓存目录（可通过参数覆盖）：
  - `--base_model` 默认：`qwen/Qwen2.5-7B-Instruct`
  - `--cache_dir` 默认：`../../data/models`
- LoRA 适配器路径：
  - `--lora_path` 默认：`../../data/models/qwen_zhu_ziqing_lora`
  - 若路径不存在，脚本会警告并使用基础模型直接测试
- 测试数据：
  - `--test_data` 默认：`zhu_ziqing_continuations.json`（位于本目录）

### 关于硅基流动 API
- 脚本优先调用硅基流动 API 获取“开头段落 + 叙事大纲”。若调用失败，将自动回退到本地备用生成逻辑。
- API Key 设置：当前代码中 `SiliconFlowAPIClient` 使用硬编码 key，仅用于占位与调试。建议将你自己的 key 写入 `test_model.py` 中该类的 `api_key` 参数；或者自行改造为从环境变量读取（例如 `SILICONFLOW_API_KEY`）。

### Human-in-the-loop（HITL）说明
- 在生成散文正文之前，系统先展示 deepseek 规划的开头与情节节点。
- 用户可在命令行中进行编辑：修改开头、增删改节点、调整顺序、重新编号。
- 编辑结束后输入 `done`，随后进入段落生成流程。
- 也可用 `--hitl_commands` 提前提供命令序列以批处理测试。

### 运行示例
以“天安门广场阅兵”为主题，生成完整散文（优化版）：
```bash
python test_model.py --mode topic_opt --topic "围绕一场在天安门广场的阅兵进行创作，主旨是兵强马壮，家国安康"
```

### 常见问题
- ImportError: No module named 'sanwen'
  - 已修复为本地导入：`from test_siliconflow_api import get_beginning_and_outline`
- 模型下载过慢或显存不足
  - 可先仅运行 API 生成流程（不加载大模型），或在低资源机器上仅进行情节规划环节；必要时切换更小模型或关闭 LoRA。
- 路径不存在（如 LoRA 路径）
  - 若 `--lora_path` 不存在，脚本会继续使用基础模型进行测试；可调整到你实际训练输出目录。

### 目录说明（节选）
```
docs/essay/
  README.md                 # 本说明
  requirements.txt          # 运行依赖
  test_model.py             # 主测试脚本
  test_siliconflow_api.py   # API 封装与获取大纲
  sft.py                    # 训练/微调脚本
  zhu_ziqing_continuations.json  # 测试数据样例
```



