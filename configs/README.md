# configs — 实验登记与配置标准化

## 1. 目录用途

`configs/` 存放策略实验的结构化登记和配置文件。目标是让每一次策略实验都有可追溯的记录，包括策略假设、参数、命令、输出目录、分析结论和推进决策。

当前文件：

| 文件 | 用途 |
|------|------|
| `research_experiments.json` | 实验登记主文件，记录所有实验的配置和结论 |
| `research_experiments.schema.json` | JSON Schema (draft-07)，定义实验字段的结构和约束 |
| `README.md` | 本文件 |

## 2. research_experiments.json 维护规则

### 顶层结构

```json
{
  "schema_version": "1.0",
  "updated_at": "2026-05-16",
  "experiments": [...]
}
```

- `schema_version`：当前为 `"1.0"`，仅在 schema 结构变更时升级。
- `updated_at`：最后一次修改的日期（YYYY-MM-DD）。
- `experiments`：实验数组，每个元素是一个完整的实验登记。

### 每次新增实验

复制现有实验对象作为模板，修改以下必填字段：

1. `experiment_id` — 唯一编号，格式 `exp_NNN_简短描述`
2. `status` — 初始状态设为 `planned`
3. `strategy_family` — 策略类别（如 `ma_cross`、`momentum`）
4. `strategy_version` — 版本号（如 `v3`）
5. `strategy_name` — 人类可读名称
6. `hypothesis` — 一句话说清楚在测什么
7. `signal_definition` — 入场、出场、过滤、排序、组合逻辑
8. `universe` — 市场、证券类型、数据源
9. `date_range` — 训练起始、测试区间
10. `parameters` — 所有策略参数和训练筛选条件
11. `cost_model` — 佣金、印花税、滑点假设
12. `commands` — 每个阶段的可复现命令
13. `outputs` — 每个阶段的输出目录
14. `decision` — 初始决策设为 `decision: "continue"`

可选字段在实验完成后补充：

- `key_results` — 量化结果指标
- `diagnosis_summary` — 诊断结论
- `notes` — 自由备注

### 更新已有实验

每次推进实验（跑完新阶段、拿到新结论、做出新决策）时：

1. 更新 `status`
2. 补充或更新 `key_results`、`diagnosis_summary`
3. 更新 `decision`（reason、decided_at、owner）
4. 更新顶层 `updated_at`

## 3. status 含义

| status | 含义 | 使用场景 |
|--------|------|----------|
| `planned` | 已登记，尚未运行 | 新实验刚创建 |
| `running` | 正在执行某个阶段 | 批量回测或 walk-forward 运行中 |
| `completed` | 所有阶段跑完，尚未诊断 | walk-forward 和分析都已完成 |
| `diagnosed` | 已完成诊断，有结论 | 诊断报告已生成 |
| `abandoned` | 已放弃，不再推进 | 诊断结论为不值得继续 |
| `promoted` | 已推进到下一阶段 | 已进入 portfolio backtest 或模拟盘 |

状态流转：

```
planned → running → completed → diagnosed
                                  ├── abandoned
                                  ├── revise (回到 planned，修改策略)
                                  └── promoted
```

## 4. decision 含义

| decision | 含义 |
|----------|------|
| `continue` | 继续按当前设计推进 |
| `revise` | 需要修改策略后重新实验 |
| `revise_alpha_signal` | alpha 信号需要修正（稳健性验证不通过） |
| `abandon` | 放弃该策略方向 |
| `promote_to_portfolio_backtest` | 通过验证，进入真实组合约束回测 |
| `promote_to_simulation` | 通过组合回测，进入模拟盘 |

## 5. 从 MA v2 创建 MA v3 的步骤

1. 在 `research_experiments.json` 的 `experiments` 数组末尾添加新对象。
2. 复制 `exp_002_ma_cross_with_market_filter` 作为模板。
3. 修改以下字段：
   - `experiment_id`: `exp_003_ma_v3_描述`
   - `status`: `planned`
   - `strategy_version`: `v3`
   - `strategy_name`: 描述新策略逻辑
   - `hypothesis`: 写清楚 MA v3 相对 v2 改了什么、为什么
   - `signal_definition`: 更新入场/出场/过滤逻辑
   - `parameters`: 更新参数范围
   - `commands`: 更新为新脚本的运行命令
   - `outputs`: 更新为新输出目录
   - `decision`: 设为 `{"decision": "continue", ...}`
4. 清空 `key_results` 和 `diagnosis_summary`（设为 `null` 或删除）。
5. 更新顶层 `updated_at`。

## 6. 重要说明

- **本配置文件用于登记和复现。** 自动化执行请使用 `scripts/run_research_pipeline.py`（见第 7 节）。
- 不要把 MA v1 登记为后续推进对象。MA v1 已停止推进，仅在 notes 中说明即可。
- 数值指标允许 `null`，因为 `planned` 状态的实验还没有结果。

## 7. 运行流水线

`scripts/run_research_pipeline.py` 是 configs 驱动的研究流水线入口，可读取实验配置、按阶段执行命令。

### 激活环境

正式运行前必须激活 research-env 环境：

```bash
conda activate research-env
```

### 列出所有实验

```bash
python scripts/run_research_pipeline.py --list-experiments
```

### 展示单个实验详情

```bash
python scripts/run_research_pipeline.py --experiment-id exp_002_ma_cross_with_market_filter --show-experiment
```

### 预览（dry-run）

```bash
python scripts/run_research_pipeline.py --experiment-id exp_002_ma_cross_with_market_filter --dry-run
```

### 只跑部分 stage

```bash
python scripts/run_research_pipeline.py --experiment-id exp_002_ma_cross_with_market_filter --stages analysis,diagnosis --execute
```

### 跳过已有输出的 stage

```bash
python scripts/run_research_pipeline.py --experiment-id exp_002_ma_cross_with_market_filter --skip-existing --execute
```

### 调试：允许环境不匹配

如果当前不在 research-env 环境，可以用 `--allow-env-mismatch` 跳过环境阻断（仅用于调试）：

```bash
python scripts/run_research_pipeline.py --experiment-id exp_002_ma_cross_with_market_filter --dry-run --allow-env-mismatch
```

### 可用 stage

按顺序：`single_symbol_check` → `batch_backtest` → `walk_forward` → `analysis` → `diagnosis` → `robustness` → `portfolio_backtest`
