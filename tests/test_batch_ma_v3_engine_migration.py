# -*- coding: utf-8 -*-
"""
test_batch_ma_v3_engine_migration.py

验证 batch_ma_v3_momentum_backtest_csv.py 的 run_one_backtest() 已正确迁移到共享回测引擎。

核心验证：
1. run_one_backtest() 输出包含引擎提供的所有列 + stock_only 对照列
2. metrics 包含所有必需 key（含 stock_only_* 和 excess_vs_stock_only_*）
3. 与手动调用引擎 comparison_signal_col 一致
4. stock_only 对照路径正确计算（不受 market_filter 影响）
5. 金融正确性：次日持仓、成本公式、收益公式、权益曲线
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_ma_v3_momentum_backtest_csv import (  # noqa: E402
    run_one_backtest,
    compute_v3_signals,
    prepare_benchmark_regime,
    calc_score,
)
from scripts.common.backtest.engine import single_asset_backtest  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_stock_df() -> pd.DataFrame:
    """生成模拟股票数据。"""
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
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
    n = 400
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
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


@pytest.fixture
def benchmark_filter(sample_benchmark_df) -> pd.DataFrame:
    """生成基准过滤数据。"""
    return prepare_benchmark_regime(sample_benchmark_df, 120)


# ---------------------------------------------------------------------------
# 测试：输出完整性
# ---------------------------------------------------------------------------

class TestOutputCompleteness:
    """run_one_backtest() 应返回引擎计算的所有列 + stock_only 对照列。"""

    def test_result_columns(self, sample_stock_df, benchmark_filter):
        result, _ = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        # 引擎核心列
        engine_cols = [
            "date", "close", "market_filter", "final_signal",
            "position", "stock_ret", "cost", "strategy_ret",
            "equity", "buy_hold_equity", "drawdown",
        ]
        for col in engine_cols:
            assert col in result.columns, f"缺少引擎列: {col}"

        # stock_only 对照列（由 comparison_signal_col 自动计算）
        stock_only_cols = [
            "stock_only_signal", "stock_only_position",
            "stock_only_cost", "stock_only_ret", "stock_only_equity",
        ]
        for col in stock_only_cols:
            assert col in result.columns, f"缺少对照列: {col}"

    def test_metrics_keys(self, sample_stock_df, benchmark_filter):
        _, metrics = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        expected_keys = [
            "strategy_total_return", "strategy_annual_return", "strategy_sharpe",
            "strategy_max_drawdown", "strategy_trade_count", "strategy_final_equity",
            "buy_hold_total_return",
            "stock_only_total_return", "stock_only_annual_return",
            "stock_only_sharpe", "stock_only_max_drawdown",
            "excess_vs_stock_only_total_return", "excess_vs_buy_hold_total_return",
            "market_filter_on_ratio", "strategy_exposure_ratio",
            "stock_only_exposure_ratio",
        ]
        for k in expected_keys:
            assert k in metrics, f"缺少 key: {k}"


# ---------------------------------------------------------------------------
# 测试：引擎一致性（手动 vs 批量脚本）
# ---------------------------------------------------------------------------

class TestEngineConsistency:
    """验证 run_one_backtest() 的结果与手动调用引擎一致。"""

    def test_position_matches_manual(self, sample_stock_df, benchmark_filter):
        result, _ = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )

        # 手动重现引擎逻辑
        signals = compute_v3_signals(sample_stock_df, 60, 250, 120)
        manual_result, _ = single_asset_backtest(
            signals, benchmark_filter, signal_col="trend_confirm",
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
            comparison_signal_col="trend_confirm",
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
            result["stock_only_ret"], manual_result["stock_only_ret"],
            atol=1e-10, check_names=False,
        )

    def test_metrics_consistency(self, sample_stock_df, benchmark_filter):
        _, metrics = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )

        signals = compute_v3_signals(sample_stock_df, 60, 250, 120)
        _, engine_metrics = single_asset_backtest(
            signals, benchmark_filter, signal_col="trend_confirm",
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
            comparison_signal_col="trend_confirm",
        )

        # 关键指标一致
        for k in ["strategy_total_return", "stock_only_total_return",
                   "excess_vs_stock_only_total_return", "excess_vs_buy_hold_total_return"]:
            assert abs(metrics[k] - engine_metrics[k]) < 1e-10, \
                f"key {k}: {metrics[k]} != {engine_metrics[k]}"


# ---------------------------------------------------------------------------
# 测试：stock_only 对照路径
# ---------------------------------------------------------------------------

class TestStockOnlyPath:
    """验证 stock_only 对照路径正确计算。"""

    def test_stock_only_position_is_trend_confirm_shifted(self, sample_stock_df, benchmark_filter):
        """stock_only_position = trend_confirm.shift(1)。"""
        result, _ = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        expected_position = result["trend_confirm"].shift(1, fill_value=0).astype(float)
        pd.testing.assert_series_equal(
            result["stock_only_position"], expected_position,
            atol=1e-10, check_names=False,
        )

    def test_stock_only_ret_formula(self, sample_stock_df, benchmark_filter):
        """stock_only_ret = stock_only_position * stock_ret - stock_only_cost。"""
        result, _ = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        expected = result["stock_only_position"] * result["stock_ret"] - result["stock_only_cost"]
        pd.testing.assert_series_equal(
            result["stock_only_ret"], expected,
            atol=1e-10, check_names=False,
        )

    def test_stock_only_equity_curve(self, sample_stock_df, benchmark_filter):
        """stock_only_equity = cash * cumprod(1 + stock_only_ret)。"""
        cash = 1_000_000.0
        result, _ = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=cash, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        expected_equity = cash * (1 + result["stock_only_ret"]).cumprod()
        pd.testing.assert_series_equal(
            result["stock_only_equity"], expected_equity,
            atol=1e-6, check_names=False,
        )

    def test_stock_only_no_market_filter(self, sample_stock_df, benchmark_filter):
        """stock_only 信号不应受 market_filter 影响。"""
        result, _ = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        # stock_only_signal 应等于 trend_confirm，不受 market_filter 影响
        pd.testing.assert_series_equal(
            result["stock_only_signal"], result["trend_confirm"],
            check_names=False,
        )
        # final_signal 受 market_filter 影响，但 stock_only_signal 不受
        mask_no_filter = result["market_filter"] == 0
        if mask_no_filter.any():
            assert (result.loc[mask_no_filter, "final_signal"] == 0).all()


# ---------------------------------------------------------------------------
# 测试：金融正确性
# ---------------------------------------------------------------------------

class TestFinancialCorrectness:
    """金融正确性检查。"""

    def test_next_day_position(self, sample_stock_df, benchmark_filter):
        """position = final_signal.shift(1)，次日生效。"""
        result, _ = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        expected_position = result["final_signal"].shift(1, fill_value=0).astype(float)
        pd.testing.assert_series_equal(
            result["position"], expected_position,
            atol=1e-10, check_names=False,
        )

    def test_strategy_ret_formula(self, sample_stock_df, benchmark_filter):
        """strategy_ret = position * stock_ret - cost。"""
        result, _ = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        expected = result["position"] * result["stock_ret"] - result["cost"]
        pd.testing.assert_series_equal(
            result["strategy_ret"], expected,
            atol=1e-10, check_names=False,
        )

    def test_drawdown_non_positive(self, sample_stock_df, benchmark_filter):
        """drawdown 应始终 <= 0。"""
        result, _ = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        assert (result["drawdown"] <= 1e-10).all()

    def test_cost_non_negative(self, sample_stock_df, benchmark_filter):
        """cost 应始终 >= 0。"""
        result, _ = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=60, ma_long=250, momentum_window=120,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        assert (result["cost"] >= -1e-10).all()
        assert (result["stock_only_cost"] >= -1e-10).all()


# ---------------------------------------------------------------------------
# 测试：calc_score 包含 stock_only 权重
# ---------------------------------------------------------------------------

class TestCalcScore:
    """验证 calc_score 使用 excess_vs_stock_only 权重。"""

    def test_score_with_stock_only_excess(self):
        """calc_score 应包含 excess_vs_stock_only_total_return 项。"""
        metrics = {
            "strategy_annual_return": 0.10,
            "strategy_sharpe": 1.0,
            "excess_vs_stock_only_total_return": 0.05,
            "excess_vs_buy_hold_total_return": 0.03,
            "strategy_max_drawdown": -0.10,
        }
        score = calc_score(metrics)
        # 手动验证公式
        expected = (
            0.10
            + 0.20 * 1.0
            + 0.40 * 0.05
            + 0.30 * 0.03
            + (-0.10)
        )
        assert abs(score - expected) < 1e-10


# ---------------------------------------------------------------------------
# 测试：参数化
# ---------------------------------------------------------------------------

class TestParameterized:
    """参数化测试覆盖不同 MA 参数组合。"""

    @pytest.mark.parametrize("ma_mid,ma_long,momentum_window", [
        (20, 120, 60),
        (60, 250, 120),
        (30, 200, 90),
    ])
    def test_different_params(self, sample_stock_df, benchmark_filter,
                              ma_mid, ma_long, momentum_window):
        """不同参数组合均能正常运行。"""
        result, metrics = run_one_backtest(
            stock_df=sample_stock_df,
            benchmark_filter_df=benchmark_filter,
            ma_mid=ma_mid, ma_long=ma_long, momentum_window=momentum_window,
            cash=1_000_000.0, commission=0.0001, sell_tax=0.0005, slippage=0.0,
        )
        assert len(result) == len(sample_stock_df)
        assert "strategy_total_return" in metrics
        assert "stock_only_total_return" in metrics
        assert "equity" in result.columns
        assert "stock_only_equity" in result.columns
