# -*- coding: utf-8 -*-
"""
tests/test_benchmark.py

scripts/common/benchmark.py 的单元测试。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.common.benchmark import prepare_benchmark_regime


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_benchmark_df(dates: list[str], closes: list[float]) -> pd.DataFrame:
    """构造最小基准日线 DataFrame。"""
    return pd.DataFrame({"date": dates, "close": closes})


# ---------------------------------------------------------------------------
# 基础功能
# ---------------------------------------------------------------------------

class TestPrepareBenchmarkRegime:
    """prepare_benchmark_regime 基础功能测试。"""

    def test_output_columns(self):
        """输出必须包含 5 个指定列。"""
        dates = [f"2024-01-{d:02d}" for d in range(1, 61)]
        closes = list(range(100, 160))
        df = _make_benchmark_df(dates, closes)
        result = prepare_benchmark_regime(df, benchmark_ma=20)
        assert list(result.columns) == [
            "date", "close", "benchmark_ma_short", "benchmark_ma_long", "market_filter",
        ]

    def test_output_length_preserved(self):
        """行数应与输入一致。"""
        n = 50
        dates = [f"2024-01-{d:02d}" for d in range(1, n + 1)]
        closes = list(range(100, 100 + n))
        df = _make_benchmark_df(dates, closes)
        result = prepare_benchmark_regime(df, benchmark_ma=10)
        assert len(result) == n

    def test_market_filter_is_binary(self):
        """market_filter 只能是 0 或 1。"""
        n = 60
        dates = [f"2024-01-{d:02d}" for d in range(1, n + 1)]
        closes = list(range(100, 100 + n))
        df = _make_benchmark_df(dates, closes)
        result = prepare_benchmark_regime(df, benchmark_ma=20)
        valid = result["market_filter"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_short_ma_window(self):
        """benchmark_ma_short 应为 benchmark_ma 期滚动均值。"""
        n = 40
        dates = [f"2024-01-{d:02d}" for d in range(1, n + 1)]
        closes = [float(i) for i in range(1, n + 1)]
        df = _make_benchmark_df(dates, closes)
        result = prepare_benchmark_regime(df, benchmark_ma=10)
        expected = pd.Series(closes).rolling(10).mean().values
        np.testing.assert_allclose(
            result["benchmark_ma_short"].values, expected, rtol=1e-10,
        )

    def test_long_ma_window(self):
        """benchmark_ma_long 应为 int(benchmark_ma * 2.5) 期滚动均值。"""
        bm_ma = 20
        long_ma = int(bm_ma * 2.5)  # 50
        n = 80
        dates = [f"2024-01-{d:02d}" for d in range(1, n + 1)]
        closes = [float(i) for i in range(1, n + 1)]
        df = _make_benchmark_df(dates, closes)
        result = prepare_benchmark_regime(df, benchmark_ma=bm_ma)
        expected = pd.Series(closes).rolling(long_ma).mean().values
        np.testing.assert_allclose(
            result["benchmark_ma_long"].values, expected, rtol=1e-10,
        )

    def test_does_not_mutate_input(self):
        """不应修改原始 DataFrame。"""
        dates = [f"2024-01-{d:02d}" for d in range(1, 41)]
        closes = list(range(100, 140))
        df = _make_benchmark_df(dates, closes)
        original_cols = list(df.columns)
        prepare_benchmark_regime(df, benchmark_ma=10)
        assert list(df.columns) == original_cols

    def test_sorted_by_date(self):
        """输入乱序日期时，输出应按日期排序后再计算 rolling。"""
        dates = [f"2024-01-{d:02d}" for d in range(1, 31)]
        closes = [float(i) for i in range(1, 31)]
        # 打乱顺序
        df = _make_benchmark_df(dates[::-1], closes[::-1])
        result = prepare_benchmark_regime(df, benchmark_ma=10)
        # 输出应按日期排序
        assert list(result["date"]) == sorted(dates)


# ---------------------------------------------------------------------------
# 统一导入验证
# ---------------------------------------------------------------------------

class TestPrepareBenchmarkRegimeUnifiedImport:
    """验证旧脚本中的 prepare_benchmark_regime 来自共享模块。"""

    _EXPECTED_MODULE = "scripts.common.benchmark"

    @pytest.mark.parametrize("module_path", [
        "strategies.alpha_v6_research_strategy_csv",
        "strategies.alpha_v7_research_strategy_csv",
        "strategies.alpha_v4_research_strategy_csv",
        "strategies.alpha_v5_research_strategy_csv",
        "strategies.ma_v3_momentum_strategy_csv",
        "scripts.batch_ma_v3_momentum_backtest_csv",
        "scripts.validate_ma_v3_momentum_candidates",
    ])
    def test_import_origin(self, module_path: str):
        """prepare_benchmark_regime 应来自 scripts.common.benchmark。"""
        import importlib
        mod = importlib.import_module(module_path)
        fn = getattr(mod, "prepare_benchmark_regime")
        assert fn.__module__ == self._EXPECTED_MODULE, (
            f"{module_path}.prepare_benchmark_regime 来自 {fn.__module__}，"
            f"期望 {self._EXPECTED_MODULE}"
        )
