# data/ — 行情数据目录

## 说明

本公开仓库 **不包含真实行情数据**。用户需要自行准备本地 CSV 或 Parquet 数据。

## 推荐目录结构

```
data/
  qmt_export/          # QMT 导出的原始 CSV 数据
    SH/                # 上交所股票
      price_600000.csv
      ...
    SZ/                # 深交所股票
      price_000001.csv
      ...
  qmt_parquet/         # CSV 转换后的 Parquet 数据（可选）
    SH/
      price_600000.parquet
      ...
    SZ/
      price_000001.parquet
      ...
  qmt_export_catalog.csv  # 数据目录索引（可选，由 convert_qmt_csv_to_parquet.py 生成）
```

## CSV 数据格式

QMT 导出的 CSV 文件应包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| date | str | 交易日期，格式 YYYY-MM-DD 或 YYYYMMDD |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| volume | float | 成交量 |
| amount | float | 成交额（可选） |

## 数据来源

- **QMT（QMT 兼容交易终端）**：通过 QMT 客户端导出历史行情数据。
- **其他来源**：任何符合上述格式的 CSV 或 Parquet 数据均可使用。

## 数据转换

如需将 CSV 转换为 Parquet 格式，可运行：

```bash
python scripts/convert_qmt_csv_to_parquet.py
```

## 注意事项

- 本仓库不提供任何真实行情数据、券商数据或第三方数据。
- 用户应确保数据来源合法，遵守相关数据使用协议。
- `data/qmt_export_catalog.csv` 包含数据目录索引，用于快速查找股票对应的 CSV 文件路径。
