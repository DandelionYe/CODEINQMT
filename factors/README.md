# factors/ — 因子数据目录

## 说明

本公开仓库 **不包含真实第三方因子数据、上市公司信息数据、概念行业原始数据**。用户需要自行准备或生成本地因子数据。

## 目录结构

```
factors/
  README.md                          # 本文件
  STK_LISTEDCOINFOANL.csv            # 上市公司信息（需自行准备）
  concept_industry/                  # 同花顺概念/行业板块数据
    README.md                        # 概念/行业模块说明
    FACTOR_REGISTRY.md               # 因子注册表草案
    STATUS.md                        # 数据采集状态
    scripts/                         # 数据采集脚本
      fetch_ths_boards.py            # 通过问财获取板块列表
      fetch_ths_members.py           # 抓取板块成分股
      retry_failed_members.py        # 补跑失败板块
      build_ths_duckdb.py            # 从 parquet 生成 DuckDB
      make_boards_from_seed.py       # 从 seed CSV 构建板块列表
      query_examples.py              # DuckDB 查询示例
    parquet/                         # 采集结果（需自行生成）
    raw/                             # 原始 HTML（需自行生成）
    ths_concept_industry.duckdb      # 查询数据库（需自行生成）
  processed/                         # 处理后的因子数据（需自行生成）
    feature_matrix/                  # 特征矩阵
```

## 数据来源

### 上市公司信息

`STK_LISTEDCOINFOANL.csv` 包含上市公司基本信息，字段包括：
- Symbol：股票代码
- ShortName：股票简称
- EndDate：截止日期
- ListedCoID：上市公司 ID
- SecurityID：证券 ID
- IndustryCode：行业代码
- LISTINGDATE：上市日期

用户可从 Wind、同花顺等数据源获取类似数据。

### 概念/行业板块数据

`concept_industry/` 模块提供同花顺概念/行业板块成分股的采集工具：

1. **板块列表**：通过 `pywencai` 从同花顺问财接口获取
2. **成分股**：通过 `q.10jqka.com.cn` 的 AJAX 接口抓取
3. **查询**：通过 DuckDB 提供 SQL 查询接口

使用前需安装依赖：
```bash
pip install pywencai browser_cookie3 requests pandas duckdb tqdm
```

## 注意事项

- 本仓库不提供任何真实因子数据、上市公司信息或概念行业数据。
- 用户应确保数据来源合法，遵守相关数据使用协议。
- 采集脚本仅供学习和研究使用，请勿用于商业用途。
