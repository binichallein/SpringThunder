## 朱自清散文续写（essay）运行说明

作者：binichallein
本目录包含用于生成/测试散文续写的脚本与数据，已经成功地复刻了朱自清的行文风格，可供生成朱自清风格的散文。核心脚本为 `test_model.py`，强烈建议使用显存大于20GB的GPU运行本项目，例如RTX4090

### 环境准备
- Python 3.9+
- 建议在虚拟环境中安装依赖：
```bash
git clone https://github.com/SocialAI-tianji/SpringThunder.git
cd SpringThunder/docs/essay
pip install -r requirements.txt
```

###快速开始

使用以下命令，程序会自动下载qwen2.5-7B instruct模型并从modelscope下载微调得到的lora适配器，无需再进行本地微调。
基于“主题”并启用人机协同（HITL）先审核与编辑大纲，再生成：
```bash

python test_model.py --mode topic_opt --topic "围绕一场在天安门广场的阅兵进行创作，主旨是兵强马壮，家国安康" --human_in_loop
# 进入命令行交互：
# 可用命令：
#   show | setbegin <文本> | add <描述> | insert <索引> <描述> |
#   edit <索引> <新描述> | del <索引> | move <源索引> <目标索引> |
#   renumber | done | cancel

hitl>done #确认情节规划没有问题后输入done开始生成
```
### Human-in-the-loop（HITL）人机协同情节规划编辑

本功能用于在生成散文正文之前，让用户先审阅并编辑 deepseek 给出的"开头 + 情节节点"，确保生成过程遵循人工审校后的意图。

#### 功能概览
- 展示 deepseek 生成的 `generated_beginning` 与 `narrative_outline` 列表
- 命令行中支持对开头与节点的增删改、插入、移动、重新编号

#### 命令说明（交互界面）
- `show`：显示当前开头与全部情节节点
- `setbegin <文本>`：设置/替换开头段落
- `add <描述>`：在末尾新增一个节点
- `insert <索引> <描述>`：在指定位置（1-based）插入节点
- `edit <索引> <新描述>`：修改指定节点的描述
- `del <索引>`：删除指定节点
- `move <源索引> <目标索引>`：移动节点到目标位置
- `renumber`：按当前顺序重排 `node_id`
- `done`：结束编辑并保存修改，进入正文生成
- `cancel`：放弃编辑并退出（保留当前状态）
- `help`：显示简要帮助



### 关于硅基流动 API
- 脚本优先调用硅基流动 API 获取“开头段落 + 叙事大纲”。若调用失败，将自动回退到本地备用生成逻辑。
- API Key 设置：当前代码中 `SiliconFlowAPIClient` 使用硬编码 key，。你也可以将自己的 key 写入 `test_model.py` 中该类的 `api_key` 参数；


### 目录说明（节选）
```
docs/essay/
  README.md                 # 本说明（包含HITL人机协同功能说明）
  requirements.txt          # 运行依赖
  run_model.py             # 主测试脚本
  test_siliconflow_api.py   # API 封装与获取大纲
  test_hitl.py             # HITL功能自动化测试脚本
  run_sft.py                    # 训练/微调脚本
  zhu_ziqing_continuations.json  # 测试数据样例
```



