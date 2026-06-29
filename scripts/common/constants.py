# -*- coding: utf-8 -*-
"""
scripts/common/constants.py

共享常量定义。所有策略和脚本应从此模块导入，避免硬编码。
"""

import math
from pathlib import Path

# 年化交易日数。A 股市场每年约 252 个交易日。
# 用于年化收益、年化波动率、Sharpe ratio 等指标计算。
TRADING_DAYS_PER_YEAR = 252

# 预计算的 sqrt(TRADING_DAYS_PER_YEAR)，用于年化波动率和 Sharpe 计算。
# 避免在每个 calc_metrics 调用中重复执行 np.sqrt(252)。
SQRT_TRADING_DAYS_PER_YEAR = math.sqrt(TRADING_DAYS_PER_YEAR)

# 默认基准指数（沪深 300）。
# 策略的 --benchmark 默认值应与此一致；
# 如有策略特定原因使用不同基准，需在策略 docstring 中明确说明。
DEFAULT_BENCHMARK = "000300.SH"

# 默认基准指数列表（沪深 300、中证 500、中证 1000）。
# 批量回测、Walk-Forward 验证、诊断等脚本的 --benchmark-list / --benchmarks 默认值。
DEFAULT_BENCHMARK_LIST = "000300.SH,000905.SH,000852.SH"

# ---------------------------------------------------------------------------
# 项目根目录与数据目录
# ---------------------------------------------------------------------------

# 从本文件位置向上推 2 级：scripts/common/constants.py -> scripts -> CODEINQMT
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# QMT 导出 CSV 数据根目录
DEFAULT_EXPORT_ROOT: Path = PROJECT_ROOT / "data" / "qmt_export"

# Parquet 转换数据根目录
DEFAULT_PARQUET_ROOT: Path = PROJECT_ROOT / "data" / "qmt_parquet"
