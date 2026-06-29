# -*- coding: utf-8 -*-
"""
test_v7_engine_migration.py

验证 Alpha v7 的 run_backtest() 已正确迁移到共享回测引擎。

核心验证：
1. run_backtest() 输出包含引擎提供的所有列
2. metrics 包含所有必需 key
3. 与手动计算的回测逻辑一致（position/strategy_ret/equity）
4. 金融正确性：次日持仓、成本公式、收益公式
5. 4 个 variant 均能正常运行
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.alpha_v7_research_strategy_csv import (  # noqa: E402
    run_backtest,
    _adapt_engine_metrics,
    compute_alpha_v7_signals,
    prepare_benchmark_regime,
)
from scripts.common.backtest.engine import single_asset_backtest  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_stock_df() -> pd.DataFrame:
    """生成模拟股票数据。"""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame({
        "date": dates,
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": volume,
        "amount": close * volume,
    })


@pytest.fixture
def sample_benchmark_df() -> pd.DataFrame:
    """生成模拟基准数据。"""
    np.random.seed(123)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 3000 * np.cumprod(1 + np.random.randn(n) * 0.01)

    return pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.randint(100_000_000, 500_000_000, n).astype(float),
        "amount": np.random.randn(n),
    })


# ---------------------------------------------------------------------------
# 测试：run_backtest 输出完整性
# ---------------------------------------------------------------------------

class TestRunBacktestOutput:
    """run_backtest() 应返回引擎计算的所有列和指标。"""

    def test_result_columns(self, sample_stock_df, sample_benchmark_df):
        """result 应包含引擎计算的所有列。"""
        result, _ = run_backtest(
            stock_df=sample_stock_df,
            benchmark_df=sample_benchmark_df,
            stock_symbol="000001.SZ",
            benchmark_symbol="000300.SH",
            alpha_variant="short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
            benchmark_ma=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        # 引擎核心列
        expected_cols = [
            "date", "close",
            "alpha_signal", "raw_alpha_score", "alpha_score",
            "market_filter", "final_signal",
            "position", "stock_ret", "cost", "strategy_ret",
            "equity", "buy_hold_equity", "drawdown",
            "benchmark_close", "benchmark_ret", "benchmark_equity",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少列: {col}"

    def test_metrics_keys(self, sample_stock_df, sample_benchmark_df):
        """metrics 应包含所有 CLI 需要的 key。"""
        _, metrics = run_backtest(
            stock_df=sample_stock_df,
            benchmark_df=sample_benchmark_df,
            stock_symbol="000001.SZ",
            benchmark_symbol="000300.SH",
            alpha_variant="short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
            benchmark_ma=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        expected_keys = [
            "total_return", "annual_return", "annual_volatility",
            "max_drawdown", "sharpe", "trade_count", "final_equity",
            "benchmark_total_return", "market_filter_on_ratio", "strategy_exposure_ratio",
        ]
        for k in expected_keys:
            assert k in metrics, f"缺少 key: {k}"


# ---------------------------------------------------------------------------
# 测试：引擎一致性（手动 vs 引擎）
# ---------------------------------------------------------------------------

class TestEngineConsistency:
    """验证 run_backtest() 的结果与手动调用引擎一致。"""

    def test_position_matches_manual(self, sample_stock_df, sample_benchmark_df):
        """position 应与手动引擎调用一致。"""
        result, _ = run_backtest(
            stock_df=sample_stock_df,
            benchmark_df=sample_benchmark_df,
            stock_symbol="000001.SZ",
            benchmark_symbol="000300.SH",
            alpha_variant="short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
            benchmark_ma=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )

        # 手动重现引擎逻辑
        signals = compute_alpha_v7_signals(
            sample_stock_df, "short_term_reversal", 10, 60, 10, 60, 20,
        )
        bench_filter = prepare_benchmark_regime(sample_benchmark_df, 120)
        manual_result, _ = single_asset_backtest(
            signals, bench_filter, signal_col="alpha_signal",
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
            compute_benchmark_ret=True,
        )

        pd.testing.assert_series_equal(
            result["position"], manual_result["position"],
            atol=1e-10, check_names=False,
        )
        pd.testing.assert_series_equal(
            result["strategy_ret"], manual_result["strategy_ret"],
            atol=1e-10, check_names=False,
        )
        pd.testing.assert_series_equal(
            result["equity"], manual_result["equity"],
            atol=1e-6, check_names=False,
        )

    def test_metrics_consistency(self, sample_stock_df, sample_benchmark_df):
        """metrics 应与手动引擎调用一致。"""
        _, metrics = run_backtest(
            stock_df=sample_stock_df,
            benchmark_df=sample_benchmark_df,
            stock_symbol="000001.SZ",
            benchmark_symbol="000300.SH",
            alpha_variant="short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
            benchmark_ma=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )

        signals = compute_alpha_v7_signals(
            sample_stock_df, "short_term_reversal", 10, 60, 10, 60, 20,
        )
        bench_filter = prepare_benchmark_regime(sample_benchmark_df, 120)
        _, engine_metrics = single_asset_backtest(
            signals, bench_filter, signal_col="alpha_signal",
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
            compute_benchmark_ret=True,
        )
        expected = _adapt_engine_metrics(engine_metrics)

        for k in expected:
            if isinstance(expected[k], float) and np.isnan(expected[k]):
                assert np.isnan(metrics[k]), f"key {k}: expected NaN"
            else:
                assert abs(metrics[k] - expected[k]) < 1e-10, \
                    f"key {k}: {metrics[k]} != {expected[k]}"


# ---------------------------------------------------------------------------
# 测试：4 个 variant 均能正常运行
# ---------------------------------------------------------------------------

class TestAllVariants:
    """所有 4 个 variant 均能正常运行并返回完整结果。"""

    @pytest.mark.parametrize("variant", [
        "short_term_reversal",
        "low_volatility",
        "turnover_reversal",
        "volume_price_divergence",
    ])
    def test_variant_runs(self, sample_stock_df, sample_benchmark_df, variant):
        result, metrics = run_backtest(
            stock_df=sample_stock_df,
            benchmark_df=sample_benchmark_df,
            stock_symbol="000001.SZ",
            benchmark_symbol="000300.SH",
            alpha_variant=variant,
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
            benchmark_ma=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        assert len(result) == len(sample_stock_df)
        assert "total_return" in metrics
        assert "equity" in result.columns
        assert "position" in result.columns


# ---------------------------------------------------------------------------
# 测试：金融正确性
# ---------------------------------------------------------------------------

class TestFinancialCorrectness:
    """金融正确性检查。"""

    def test_next_day_position(self, sample_stock_df, sample_benchmark_df):
        """position = final_signal.shift(1)，次日生效。"""
        result, _ = run_backtest(
            stock_df=sample_stock_df,
            benchmark_df=sample_benchmark_df,
            stock_symbol="000001.SZ",
            benchmark_symbol="000300.SH",
            alpha_variant="short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
            benchmark_ma=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        # position 应等于 final_signal.shift(1)
        expected_position = result["final_signal"].shift(1, fill_value=0).astype(float)
        pd.testing.assert_series_equal(
            result["position"], expected_position,
            atol=1e-10, check_names=False,
        )

    def test_strategy_ret_formula(self, sample_stock_df, sample_benchmark_df):
        """strategy_ret = position * stock_ret - cost。"""
        result, _ = run_backtest(
            stock_df=sample_stock_df,
            benchmark_df=sample_benchmark_df,
            stock_symbol="000001.SZ",
            benchmark_symbol="000300.SH",
            alpha_variant="short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
            benchmark_ma=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        expected = result["position"] * result["stock_ret"] - result["cost"]
        pd.testing.assert_series_equal(
            result["strategy_ret"], expected,
            atol=1e-10, check_names=False,
        )

    def test_equity_curve(self, sample_stock_df, sample_benchmark_df):
        """equity = cash * cumprod(1 + strategy_ret)。"""
        cash = 1_000_000.0
        result, _ = run_backtest(
            stock_df=sample_stock_df,
            benchmark_df=sample_benchmark_df,
            stock_symbol="000001.SZ",
            benchmark_symbol="000300.SH",
            alpha_variant="short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
            benchmark_ma=120,
            cash=cash, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        expected_equity = cash * (1 + result["strategy_ret"]).cumprod()
        pd.testing.assert_series_equal(
            result["equity"], expected_equity,
            atol=1e-6, check_names=False,
        )

    def test_drawdown_non_positive(self, sample_stock_df, sample_benchmark_df):
        """drawdown 应始终 <= 0。"""
        result, _ = run_backtest(
            stock_df=sample_stock_df,
            benchmark_df=sample_benchmark_df,
            stock_symbol="000001.SZ",
            benchmark_symbol="000300.SH",
            alpha_variant="short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
            benchmark_ma=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        assert (result["drawdown"] <= 1e-10).all()


# ---------------------------------------------------------------------------
# 测试：_adapt_engine_metrics
# ---------------------------------------------------------------------------

class TestAdaptEngineMetrics:
    """测试 _adapt_engine_metrics 正确映射 key。"""

    def test_key_mapping(self):
        engine_metrics = {
            "strategy_total_return": 0.15,
            "strategy_annual_return": 0.12,
            "strategy_max_drawdown": -0.08,
            "strategy_sharpe": 1.5,
            "strategy_annual_volatility": 0.20,
            "strategy_trade_count": 42,
            "strategy_final_equity": 1_150_000.0,
            "buy_hold_total_return": 0.10,
            "excess_vs_buy_hold_total_return": 0.05,
            "market_filter_on_ratio": 0.75,
            "strategy_exposure_ratio": 0.60,
        }
        adapted = _adapt_engine_metrics(engine_metrics)

        assert adapted["total_return"] == 0.15
        assert adapted["annual_return"] == 0.12
        assert adapted["max_drawdown"] == -0.08
        assert adapted["sharpe"] == 1.5
        assert adapted["annual_volatility"] == 0.20
        assert adapted["trade_count"] == 42
        assert adapted["final_equity"] == 1_150_000.0
        assert adapted["benchmark_total_return"] == 0.10
        assert adapted["market_filter_on_ratio"] == 0.75
        assert adapted["strategy_exposure_ratio"] == 0.60
