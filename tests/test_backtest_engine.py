# -*- coding: utf-8 -*-
"""
tests/test_backtest_engine.py

scripts/common/backtest/engine.py 的单元测试。

覆盖：
- BacktestCostModel / BacktestConfig 数据类
- apply_market_filter：基准对齐、缺失填充、不引入未来数据
- compute_final_signal：alpha_signal AND market_filter，自定义信号列
- compute_position_and_returns：次日持仓、成本计算、收益计算
- compute_equity_curves：权益曲线、回撤、基准权益
- build_backtest_metrics：策略指标、买入持有指标、超额收益
- single_asset_backtest：完整流程端到端
- single_asset_backtest_lite：精简版
- 金融正确性：无未来函数、shift 正确、成本公式正确
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.common.backtest.engine import (
    BacktestConfig,
    BacktestCostModel,
    apply_market_filter,
    build_backtest_metrics,
    compute_equity_curves,
    compute_final_signal,
    compute_position_and_returns,
    single_asset_backtest,
    single_asset_backtest_lite,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_stock_df(dates, close, alpha_signal):
    """构建股票信号 DataFrame。"""
    return pd.DataFrame({
        "date": dates,
        "close": close,
        "alpha_signal": alpha_signal,
    })


def _make_benchmark_filter(dates, close, ma_short, ma_long, market_filter):
    """构建基准过滤 DataFrame。"""
    return pd.DataFrame({
        "date": dates,
        "close": close,
        "benchmark_ma_short": ma_short,
        "benchmark_ma_long": ma_long,
        "market_filter": market_filter,
    })


@pytest.fixture
def simple_data():
    """简单测试数据：5 天，前 2 天无信号，后 3 天有信号。"""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    stock_df = _make_stock_df(
        dates,
        close=[10.0, 10.5, 11.0, 10.8, 11.2],
        alpha_signal=[0, 0, 1, 1, 1],
    )
    bench_df = _make_benchmark_filter(
        dates,
        close=[100.0, 101.0, 102.0, 101.5, 103.0],
        ma_short=[100.0, 100.5, 101.0, 101.2, 101.8],
        ma_long=[99.0, 99.5, 100.0, 100.2, 100.8],
        market_filter=[1, 1, 1, 1, 1],
    )
    return stock_df, bench_df, dates


@pytest.fixture
def partial_market_filter_data():
    """大盘过滤部分关闭的数据。"""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    stock_df = _make_stock_df(
        dates,
        close=[10.0, 10.5, 11.0, 10.8, 11.2],
        alpha_signal=[1, 1, 1, 1, 1],
    )
    bench_df = _make_benchmark_filter(
        dates,
        close=[100.0, 101.0, 102.0, 101.5, 103.0],
        ma_short=[100.0, 100.5, 101.0, 101.2, 101.8],
        ma_long=[99.0, 99.5, 100.0, 100.2, 100.8],
        market_filter=[0, 0, 1, 1, 0],  # 只有第 3、4 天开启
    )
    return stock_df, bench_df, dates


# ---------------------------------------------------------------------------
# BacktestCostModel / BacktestConfig
# ---------------------------------------------------------------------------

class TestBacktestCostModel:
    def test_default_values(self):
        m = BacktestCostModel()
        assert m.commission == 0.0003
        assert m.sell_tax == 0.0005
        assert m.slippage == 0.001

    def test_a_share_default(self):
        m = BacktestCostModel.a_share_default()
        assert m.commission == 0.0003
        assert m.sell_tax == 0.0005

    def test_custom_values(self):
        m = BacktestCostModel(commission=0.001, sell_tax=0.001, slippage=0.002)
        assert m.commission == 0.001


class TestBacktestConfig:
    def test_default(self):
        c = BacktestConfig()
        assert c.cash == 1_000_000.0
        assert c.cost_model is not None
        assert c.cost_model.commission == 0.0003

    def test_custom_cash(self):
        c = BacktestConfig(cash=500_000.0)
        assert c.cash == 500_000.0


# ---------------------------------------------------------------------------
# apply_market_filter
# ---------------------------------------------------------------------------

class TestApplyMarketFilter:
    def test_basic_alignment(self, simple_data):
        stock_df, bench_df, dates = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        assert "benchmark_close" in result.columns
        assert "market_filter" in result.columns
        assert len(result) == 5
        # 基准收盘价对齐
        assert result["benchmark_close"].iloc[0] == 100.0
        assert result["benchmark_close"].iloc[-1] == 103.0

    def test_market_filter_is_int(self, simple_data):
        _, bench_df, _ = simple_data
        result = apply_market_filter(simple_data[0].copy(), bench_df)
        assert result["market_filter"].dtype in (np.int32, np.int64, int)

    def test_ffill_on_missing_dates(self):
        """如果基准数据缺少某些日期，应 forward-fill。"""
        stock_dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stock_df = _make_stock_df(stock_dates, [10.0] * 5, [0] * 5)

        # 基准只有第 1、3 天数据
        bench_dates = pd.DatetimeIndex(["2024-01-01", "2024-01-03"])
        bench_df = pd.DataFrame({
            "date": bench_dates,
            "close": [100.0, 102.0],
            "benchmark_ma_short": [100.0, 101.0],
            "benchmark_ma_long": [99.0, 100.0],
            "market_filter": [1, 1],
        })
        result = apply_market_filter(stock_df, bench_df)
        # 第 2 天应 ffill 第 1 天的值
        assert result["benchmark_close"].iloc[1] == 100.0
        # 第 4、5 天应 ffill 第 3 天的值
        assert result["benchmark_close"].iloc[3] == 102.0

    def test_no_future_data_leakage(self):
        """基准数据的日期晚于股票数据时，不应泄漏未来数据。"""
        stock_dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stock_df = _make_stock_df(stock_dates, [10.0, 10.5, 11.0], [0, 0, 0])

        bench_dates = pd.date_range("2024-01-03", periods=3, freq="B")
        bench_df = pd.DataFrame({
            "date": bench_dates,
            "close": [100.0, 101.0, 102.0],
            "benchmark_ma_short": [100.0, 100.5, 101.0],
            "benchmark_ma_long": [99.0, 99.5, 100.0],
            "market_filter": [1, 1, 1],
        })
        result = apply_market_filter(stock_df, bench_df)
        # 第 1、2 天没有基准数据，应为 NaN（ffill 没有前值）
        assert pd.isna(result["benchmark_close"].iloc[0])
        assert pd.isna(result["benchmark_close"].iloc[1])
        # 第 3 天有基准数据
        assert result["benchmark_close"].iloc[2] == 100.0


# ---------------------------------------------------------------------------
# compute_final_signal
# ---------------------------------------------------------------------------

class TestComputeFinalSignal:
    def test_alpha_signal_and_market_filter(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        # alpha_signal=[0,0,1,1,1], market_filter=[1,1,1,1,1]
        # final_signal=[0,0,1,1,1]
        assert list(result["final_signal"]) == [0, 0, 1, 1, 1]

    def test_market_filter_blocks_signal(self, partial_market_filter_data):
        stock_df, bench_df, _ = partial_market_filter_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        # alpha_signal=[1,1,1,1,1], market_filter=[0,0,1,1,0]
        # final_signal=[0,0,1,1,0]
        assert list(result["final_signal"]) == [0, 0, 1, 1, 0]

    def test_custom_signal_col(self, simple_data):
        stock_df, bench_df, _ = simple_data
        stock_df = stock_df.rename(columns={"alpha_signal": "trend_confirm"})
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result, signal_col="trend_confirm")
        assert list(result["final_signal"]) == [0, 0, 1, 1, 1]


# ---------------------------------------------------------------------------
# compute_position_and_returns
# ---------------------------------------------------------------------------

class TestComputePositionAndReturns:
    def test_next_day_position(self, simple_data):
        """position 应在 final_signal 的次日生效。"""
        stock_df, bench_df, _ = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        result = compute_position_and_returns(result, 0.0003, 0.0005, 0.001)
        # final_signal=[0,0,1,1,1]
        # position=[0,0,0,1,1]  (shift 1)
        assert result["position"].iloc[0] == 0.0
        assert result["position"].iloc[1] == 0.0
        assert result["position"].iloc[2] == 0.0  # signal=1 但 position 还是 0
        assert result["position"].iloc[3] == 1.0  # signal=1 的次日
        assert result["position"].iloc[4] == 1.0

    def test_stock_ret_calculation(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        result = compute_position_and_returns(result, 0.0003, 0.0005, 0.001)
        # close=[10.0, 10.5, 11.0, 10.8, 11.2]
        # stock_ret: [0, 0.05, 0.0476..., -0.01818..., 0.03703...]
        assert result["stock_ret"].iloc[0] == 0.0
        expected_ret_1 = (10.5 - 10.0) / 10.0
        assert abs(result["stock_ret"].iloc[1] - expected_ret_1) < 1e-10

    def test_cost_on_position_change(self):
        """持仓变化时应产生交易成本。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stock_df = _make_stock_df(dates, [10.0, 10.5, 11.0, 11.5], [1, 1, 0, 0])
        bench_df = _make_benchmark_filter(dates, [100.0] * 4, [100.0] * 4, [99.0] * 4, [1] * 4)
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        # final_signal=[1,1,0,0]
        # position=[0,1,1,0]  (shift 1)
        result = compute_position_and_returns(result, 0.001, 0.001, 0.001)
        # 第 2 天：从 0→1，买入成本 = 1 * (0.001 + 0.001) = 0.002
        assert abs(result["cost"].iloc[1] - 0.002) < 1e-10
        # 第 3 天：持仓不变，无成本
        assert result["cost"].iloc[2] == 0.0
        # 第 4 天：从 1→0，卖出成本 = 1 * (0.001 + 0.001 + 0.001) = 0.003
        assert abs(result["cost"].iloc[3] - 0.003) < 1e-10

    def test_strategy_ret_formula(self, simple_data):
        """strategy_ret = position * stock_ret - cost。"""
        stock_df, bench_df, _ = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        result = compute_position_and_returns(result, 0.0003, 0.0005, 0.001)
        for i in range(len(result)):
            expected = result["position"].iloc[i] * result["stock_ret"].iloc[i] - result["cost"].iloc[i]
            assert abs(result["strategy_ret"].iloc[i] - expected) < 1e-10


# ---------------------------------------------------------------------------
# compute_equity_curves
# ---------------------------------------------------------------------------

class TestComputeEquityCurves:
    def test_equity_starts_at_cash(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        result = compute_position_and_returns(result, 0.0003, 0.0005, 0.001)
        result = compute_equity_curves(result, cash=1_000_000.0)
        # 第一天 strategy_ret=0，equity 应等于 cash
        assert abs(result["equity"].iloc[0] - 1_000_000.0) < 1e-6

    def test_buy_hold_equity(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        result = compute_position_and_returns(result, 0.0003, 0.0005, 0.001)
        result = compute_equity_curves(result, cash=1_000_000.0)
        # buy_hold_equity = cash * cumprod(1 + stock_ret)
        expected_bh = 1_000_000.0 * (1 + result["stock_ret"]).cumprod()
        pd.testing.assert_series_equal(result["buy_hold_equity"], expected_bh, atol=1e-6, check_names=False)

    def test_drawdown_non_positive(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        result = compute_position_and_returns(result, 0.0003, 0.0005, 0.001)
        result = compute_equity_curves(result, cash=1_000_000.0)
        # 回撤应 <= 0
        assert (result["drawdown"] <= 1e-10).all()

    def test_benchmark_equity_when_present(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        result = compute_position_and_returns(result, 0.0003, 0.0005, 0.001)
        result["benchmark_ret"] = result["benchmark_close"].pct_change().fillna(0)
        result = compute_equity_curves(result, cash=1_000_000.0)
        assert "benchmark_equity" in result.columns
        expected_be = 1_000_000.0 * (1 + result["benchmark_ret"]).cumprod()
        pd.testing.assert_series_equal(result["benchmark_equity"], expected_be, atol=1e-6, check_names=False)

    def test_no_benchmark_equity_without_column(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        result = compute_position_and_returns(result, 0.0003, 0.0005, 0.001)
        # 移除 benchmark_ret 列
        result = result.drop(columns=["benchmark_ret"], errors="ignore")
        result = compute_equity_curves(result, cash=1_000_000.0)
        assert "benchmark_equity" not in result.columns


# ---------------------------------------------------------------------------
# build_backtest_metrics
# ---------------------------------------------------------------------------

class TestBuildBacktestMetrics:
    def test_returns_all_expected_keys(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        result = compute_position_and_returns(result, 0.0003, 0.0005, 0.001)
        metrics = build_backtest_metrics(result, 1_000_000.0)
        expected_keys = [
            "strategy_total_return", "strategy_annual_return", "strategy_sharpe",
            "strategy_max_drawdown", "strategy_annual_volatility",
            "buy_hold_total_return", "buy_hold_annual_return",
            "excess_vs_buy_hold_total_return",
            "market_filter_on_ratio", "strategy_exposure_ratio",
        ]
        for k in expected_keys:
            assert k in metrics, f"Missing key: {k}"

    def test_excess_return_calculation(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        result = compute_position_and_returns(result, 0.0003, 0.0005, 0.001)
        metrics = build_backtest_metrics(result, 1_000_000.0)
        expected_excess = metrics["strategy_total_return"] - metrics["buy_hold_total_return"]
        assert abs(metrics["excess_vs_buy_hold_total_return"] - expected_excess) < 1e-10

    def test_exposure_ratio(self, partial_market_filter_data):
        stock_df, bench_df, _ = partial_market_filter_data
        result = apply_market_filter(stock_df.copy(), bench_df)
        result = compute_final_signal(result)
        result = compute_position_and_returns(result, 0.0003, 0.0005, 0.001)
        metrics = build_backtest_metrics(result, 1_000_000.0)
        # final_signal=[0,0,1,1,0], position=[0,0,0,1,1]
        # exposure = mean([0,0,0,1,1]) = 0.4
        assert abs(metrics["strategy_exposure_ratio"] - 0.4) < 1e-10


# ---------------------------------------------------------------------------
# single_asset_backtest (端到端)
# ---------------------------------------------------------------------------

class TestSingleAssetBacktest:
    def test_basic_run(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result, metrics = single_asset_backtest(stock_df.copy(), bench_df)
        assert "equity" in result.columns
        assert "strategy_total_return" in metrics
        assert len(result) == 5

    def test_with_benchmark_ret(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result, metrics = single_asset_backtest(
            stock_df.copy(), bench_df, compute_benchmark_ret=True,
        )
        assert "benchmark_ret" in result.columns
        assert "benchmark_equity" in result.columns

    def test_custom_signal_col(self, simple_data):
        stock_df, bench_df, _ = simple_data
        stock_df = stock_df.rename(columns={"alpha_signal": "trend_confirm"})
        result, metrics = single_asset_backtest(
            stock_df.copy(), bench_df, signal_col="trend_confirm",
        )
        assert "equity" in result.columns

    def test_custom_cost_model(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result1, m1 = single_asset_backtest(
            stock_df.copy(), bench_df, commission=0.0003, sell_tax=0.0005, slippage=0.001,
        )
        result2, m2 = single_asset_backtest(
            stock_df.copy(), bench_df, commission=0.01, sell_tax=0.01, slippage=0.01,
        )
        # 高成本应导致更低的收益
        assert m2["strategy_total_return"] < m1["strategy_total_return"]

    def test_all_signals_off(self):
        """所有信号为 0 时，策略收益应接近 0（只有成本）。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stock_df = _make_stock_df(dates, [10.0, 10.5, 11.0, 10.8, 11.2], [0, 0, 0, 0, 0])
        bench_df = _make_benchmark_filter(dates, [100.0] * 5, [100.0] * 5, [99.0] * 5, [1] * 5)
        result, metrics = single_asset_backtest(stock_df.copy(), bench_df)
        # 无持仓，策略收益应全为 0
        assert (result["strategy_ret"] == 0).all()
        assert metrics["strategy_total_return"] == 0.0

    def test_all_signals_on_with_market_filter(self):
        """信号全开 + 大盘全开 = 全程持仓。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stock_df = _make_stock_df(dates, [10.0, 10.5, 11.0, 10.8, 11.2], [1, 1, 1, 1, 1])
        bench_df = _make_benchmark_filter(dates, [100.0] * 5, [100.0] * 5, [99.0] * 5, [1] * 5)
        result, metrics = single_asset_backtest(stock_df.copy(), bench_df)
        # position 应为 [0, 1, 1, 1, 1]（第 1 天 signal=1 但 shift 后为 0）
        assert result["position"].iloc[0] == 0.0
        assert (result["position"].iloc[1:] == 1.0).all()
        # exposure = 4/5 = 0.8
        assert abs(metrics["strategy_exposure_ratio"] - 0.8) < 1e-10


# ---------------------------------------------------------------------------
# single_asset_backtest_lite
# ---------------------------------------------------------------------------

class TestSingleAssetBacktestLite:
    def test_no_equity_columns(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result = single_asset_backtest_lite(stock_df.copy(), bench_df)
        assert "equity" not in result.columns
        assert "drawdown" not in result.columns
        assert "buy_hold_equity" not in result.columns

    def test_has_core_columns(self, simple_data):
        stock_df, bench_df, _ = simple_data
        result = single_asset_backtest_lite(stock_df.copy(), bench_df)
        for col in ["position", "stock_ret", "cost", "strategy_ret", "final_signal"]:
            assert col in result.columns

    def test_consistency_with_full_version(self, simple_data):
        """lite 版本的 position/strategy_ret 应与完整版一致。"""
        stock_df, bench_df, _ = simple_data
        result_lite = single_asset_backtest_lite(stock_df.copy(), bench_df)
        result_full, _ = single_asset_backtest(stock_df.copy(), bench_df)
        pd.testing.assert_series_equal(
            result_lite["position"], result_full["position"], atol=1e-10,
        )
        pd.testing.assert_series_equal(
            result_lite["strategy_ret"], result_full["strategy_ret"], atol=1e-10,
        )


# ---------------------------------------------------------------------------
# 金融正确性专项测试
# ---------------------------------------------------------------------------

class TestFinancialCorrectness:
    def test_no_future_function_in_position(self):
        """position 使用 shift(1)，不使用当日信号。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        # 第 3 天突然出现信号
        stock_df = _make_stock_df(dates, [10.0, 10.0, 10.0, 10.0], [0, 0, 1, 1])
        bench_df = _make_benchmark_filter(dates, [100.0] * 4, [100.0] * 4, [99.0] * 4, [1] * 4)
        result, _ = single_asset_backtest(stock_df.copy(), bench_df)
        # 第 3 天 signal=1，但 position 应为 0（shift 1）
        assert result["position"].iloc[2] == 0.0
        # 第 4 天 position 才变为 1
        assert result["position"].iloc[3] == 1.0

    def test_cost_formula_buy(self):
        """买入成本 = delta_position * (commission + slippage)。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stock_df = _make_stock_df(dates, [10.0, 10.0, 10.0], [1, 1, 1])
        bench_df = _make_benchmark_filter(dates, [100.0] * 3, [100.0] * 3, [99.0] * 3, [1] * 3)
        result, _ = single_asset_backtest(
            stock_df.copy(), bench_df, commission=0.001, sell_tax=0.001, slippage=0.002,
        )
        # position=[0, 1, 1]，第 2 天从 0→1
        # 买入成本 = 1 * (0.001 + 0.002) = 0.003
        assert abs(result["cost"].iloc[1] - 0.003) < 1e-10

    def test_cost_formula_sell(self):
        """卖出成本 = delta_position * (commission + sell_tax + slippage)。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stock_df = _make_stock_df(dates, [10.0, 10.0, 10.0, 10.0], [1, 1, 0, 0])
        bench_df = _make_benchmark_filter(dates, [100.0] * 4, [100.0] * 4, [99.0] * 4, [1] * 4)
        result, _ = single_asset_backtest(
            stock_df.copy(), bench_df, commission=0.001, sell_tax=0.001, slippage=0.002,
        )
        # position=[0, 1, 1, 0]，第 4 天从 1→0
        # 卖出成本 = 1 * (0.001 + 0.001 + 0.002) = 0.004
        assert abs(result["cost"].iloc[3] - 0.004) < 1e-10

    def test_strategy_ret_never_exceeds_stock_ret_when_long(self):
        """持仓时，策略收益不应超过股票收益（成本为正）。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        np.random.seed(42)
        close = 10.0 * np.cumprod(1 + np.random.randn(10) * 0.02)
        stock_df = _make_stock_df(dates, close.tolist(), [1] * 10)
        bench_df = _make_benchmark_filter(dates, [100.0] * 10, [100.0] * 10, [99.0] * 10, [1] * 10)
        result, _ = single_asset_backtest(stock_df.copy(), bench_df)
        # 持仓期间，strategy_ret <= stock_ret（因为 cost >= 0）
        long_mask = result["position"] > 0
        if long_mask.any():
            assert (result.loc[long_mask, "strategy_ret"] <= result.loc[long_mask, "stock_ret"] + 1e-10).all()

    def test_equity_curve_monotonic_when_no_cost_no_trade(self):
        """无交易成本且持续持仓时，权益曲线应与 cumprod(stock_ret) 一致。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stock_df = _make_stock_df(dates, [10.0, 11.0, 12.0, 11.5, 13.0], [1, 1, 1, 1, 1])
        bench_df = _make_benchmark_filter(dates, [100.0] * 5, [100.0] * 5, [99.0] * 5, [1] * 5)
        result, _ = single_asset_backtest(
            stock_df.copy(), bench_df, commission=0.0, sell_tax=0.0, slippage=0.0,
        )
        # 持仓期间 strategy_ret = stock_ret（cost=0）
        long_mask = result["position"] > 0
        if long_mask.any():
            pd.testing.assert_series_equal(
                result.loc[long_mask, "strategy_ret"],
                result.loc[long_mask, "stock_ret"],
                atol=1e-10,
                check_names=False,
            )
