# 同花顺概念/行业板块成分股数据

## 概述

同花顺（THS）概念/行业板块成分股数据采集与查询工具。通过问财接口获取板块列表，抓取板块成分股 HTML 表格，并提供 DuckDB 查询接口。

## 数据来源

- **板块列表**：通过 `pywencai` 从同花顺问财接口获取
- **成分股**：通过 `q.10jqka.com.cn` 的 AJAX 接口抓取 HTML 表格
- **依赖**：`pywencai`、`browser_cookie3`、`requests`、`pandas`、`duckdb`、`tqdm`

## 目录结构

```
factors/concept_industry/
├── README.md                    # 本文件
├── scripts/
│   ├── fetch_ths_boards.py      # 通过问财获取板块列表
│   ├── make_boards_from_seed.py # 从 seed CSV 手动构建板块列表
│   ├── fetch_ths_members.py     # 抓取板块成分股
│   ├── retry_failed_members.py  # 补跑失败/空结果的板块
│   ├── build_ths_duckdb.py      # 从 parquet 生成 DuckDB
│   └── query_examples.py        # DuckDB 查询示例
├── parquet/                     # 采集结果（git ignored）
│   ├── ths_boards.csv/parquet   # 板块列表
│   ├── ths_board_members.csv/parquet  # 成分股映射
│   ├── ths_fetch_log.csv/parquet      # 采集日志
│   └── seed_*.csv               # 手动种子数据
├── raw/                         # 原始 HTML（git ignored）
└── ths_concept_industry.duckdb  # 查询数据库（git ignored）
```

## 字段说明

### ths_boards（板块列表）

| 字段 | 说明 |
|------|------|
| board_code | 板块代码（881xxx/885xxx） |
| board_name | 板块名称 |
| board_type | concept 或 industry |
| source | 数据来源（pywencai/10jqka_page_seed） |
| fetch_date | 采集日期 |
| fetch_time | 采集时间 |
| raw_query | 问财查询语句 |

### ths_board_members（成分股映射）

| 字段 | 说明 |
|------|------|
| board_code | 板块代码 |
| board_name | 板块名称 |
| board_type | concept 或 industry |
| stock_code | 股票代码（000001.SZ 格式） |
| stock_name | 股票名称 |
| fetch_date | 采集日期 |
| fetch_time | 采集时间 |
| source | 数据来源（10jqka_q_ajax） |

### ths_fetch_log（采集日志）

| 字段 | 说明 |
|------|------|
| fetch_time | 采集时间 |
| target | 采集目标（members） |
| board_code | 板块代码 |
| board_name | 板块名称 |
| board_type | 板块类型 |
| status | success/failed/empty |
| message | 错误信息（仅 failed） |
| rows | 采集到的行数 |

## 使用方法

### 1. 安装依赖

```bash
pip install pywencai browser_cookie3 requests pandas duckdb tqdm
```

### 2. 获取板块列表

```bash
cd factors/concept_industry
python scripts/fetch_ths_boards.py
```

需要设置 `PYWENCAI_COOKIE` 环境变量。

### 3. 抓取成分股

```bash
python scripts/fetch_ths_members.py
```

需要浏览器 Cookie 或设置 `THS_COOKIE` 环境变量。

### 4. 补跑失败板块

```bash
python scripts/retry_failed_members.py --statuses failed,empty --limit 50
```

### 5. 生成 DuckDB

```bash
python scripts/build_ths_duckdb.py
```

### 6. 查询示例

```bash
python scripts/query_examples.py --stock 000858.SZ
python scripts/query_examples.py --board 白酒
```

## Cookie 风险说明

本工具依赖同花顺网站 Cookie 进行数据采集：

- `browser_cookie3` 尝试从 Firefox 读取 Cookie，需要关闭 Firefox 后运行
- `THS_COOKIE` 环境变量可手动设置，优先级高于浏览器 Cookie
- Cookie 过期后需要重新获取
- 频繁请求可能触发反爬机制（401/403），建议使用 `retry_failed_members.py` 的 cooldown 参数

## 当前状态

详见 [STATUS.md](STATUS.md)。数据要点：

- 板块列表：452 个（concept 362 + industry 90）
- 成分股采集成功率低（主要为 401 反爬），成功板块的数据有效可用
- DuckDB：已生成（约 1 MB），从 parquet 数据构建

## 已知限制

1. 采集成功率受 Cookie 有效性和反爬策略影响
2. 同花顺页面编码为 GBK/GB18030，已做兼容处理
3. 成分股采集依赖 HTML 表格解析，页面结构变化可能导致解析失败
4. 当前未接入 D's_Flow 的 factor registry，仅为独立数据素材

## 后续计划

- [ ] 提高采集成功率（优化 Cookie 管理、增加重试策略）
- [x] 生成有效的 DuckDB 查询数据库
- [ ] 接入 factor registry，明确 PIT 语义
- [ ] 支持增量更新（只补跑新板块和失败板块）
