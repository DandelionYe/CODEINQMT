# -*- coding: utf-8 -*-
"""
benchmark.py

大盘 regime filter 共享逻辑。

提供 prepare_benchmark_regime()，从基准指数日线数据计算 MA 交叉信号，
用于策略的大盘过滤（market_filter）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_benchmark_regime(
    benchmark_df: pd.DataFrame,
    benchmark_ma: int,
) -> pd.DataFrame:
    """计算大盘 regime filter：benchmark MA 交叉。

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        基准指数日线数据，必须包含 ``date`` 和 ``close`` 列。
    benchmark_ma : int
        短期 MA 窗口（交易日数）。长期窗口固定为 ``int(benchmark_ma * 2.5)``。

    Returns
    -------
    pd.DataFrame
        包含 ``date``、``close``、``benchmark_ma_short``、``benchmark_ma_long``、
        ``market_filter`` 列。``market_filter=1`` 表示短期 MA 高于长期 MA（看多）。
    """
    bench = benchmark_df.copy()
    bench = bench.set_index("date", drop=False).sort_index()
    bench["benchmark_ma_short"] = bench["close"].rolling(benchmark_ma).mean()
    bench["benchmark_ma_long"] = bench["close"].rolling(int(benchmark_ma * 2.5)).mean()
    bench["market_filter"] = np.where(
        bench["benchmark_ma_short"] > bench["benchmark_ma_long"],
        1,
        0,
    )
    return bench[["date", "close", "benchmark_ma_short", "benchmark_ma_long", "market_filter"]].copy()
