## Human-in-the-loop（HITL）人机协同情节规划编辑指南

本功能用于在生成散文正文之前，让用户先审阅并编辑 deepseek 给出的“开头 + 情节节点”，确保生成过程遵循人工审校后的意图。

### 功能概览
- 展示 deepseek 生成的 `generated_beginning` 与 `narrative_outline` 列表。
- 命令行中支持对开头与节点的增删改、插入、移动、重新编号。
- 支持交互式人工输入，也支持 `--hitl_commands` 以脚本化批处理（CI 场景）。

### 入口与开关
- 交互入口（交互模式）：
  - 在 `test_model.py` 的交互测试中输入：`topic_hitl:你的主题`
- 命令行入口（非交互/脚本化也可）：
  - `python test_model.py --mode topic_opt --topic "你的主题" --human_in_loop`
  - 可选：`--hitl_commands "setbegin 新的开头 | add 新节点 | done"`

### 命令说明（交互界面）
- `show`：显示当前开头与全部情节节点。
- `setbegin <文本>`：设置/替换开头段落。
- `add <描述>`：在末尾新增一个节点。
- `insert <索引> <描述>`：在指定位置（1-based）插入节点。
- `edit <索引> <新描述>`：修改指定节点的描述。
- `del <索引>`：删除指定节点。
- `move <源索引> <目标索引>`：移动节点到目标位置。
- `renumber`：按当前顺序重排 `node_id`。
- `done`：结束编辑并保存修改，进入正文生成。
- `cancel`：放弃编辑并退出（保留当前状态）。
- `help`：显示简要帮助。

### 脚本化命令（CI/批处理）
使用 `--hitl_commands` 传入由 `|` 分隔的命令序列，例如：
```bash
python test_model.py --mode topic_opt --topic "春天的校园" \
  --human_in_loop \
  --hitl_commands "setbegin 清晨的风带着青草的气息拂过走廊 | insert 1 第一段描写操场的清新 | add 傍晚的余晖洒在红墙上 | renumber | done"
```

### 测试与验证
- 自动化测试脚本：`docs/essay/test_hitl.py`
  - 覆盖典型操作：修改开头、增/插/改/删、移动与重新编号。
  - 运行：`python docs/essay/test_hitl.py`

### 与主体生成流程的衔接
- 在 `test_model.py` 中，优化版生成方法 `generate_complete_essay_optimized` 支持：
  - `human_in_loop: bool` 开关；
  - `hitl_commands: Optional[List[str]]` 脚本化命令输入；
  - 若开启 HITL，则在正文生成前进入编辑；编辑完成后使用新的开头与节点生成散文。

### 注意事项
- 命令中的索引均为 1-based。
- `insert` 与 `move` 会自动对 `node_id` 进行一致性维护，亦可手动执行 `renumber`。
- 若不想等待人工输入，可使用 `--hitl_commands` 进行无交互批处理。


