# -*- coding: utf-8 -*-
"""
scripts/common/backtest/engine.py

共享单标的回测引擎。

设计原则：
- 从 8 个策略脚本的 run_backtest() 和 wf_batch_shared.py 的 run_one_backtest() / run_alpha_frame()
  中提取重复的回测核心逻辑。
- 引擎不关心信号如何生成，只接收已包含 alpha_signal 列的 DataFrame。
- 所有回测参数（佣金、滑点、印花税、初始资金）显式传入，不依赖全局状态。
- 次日持仓（position = final_signal.shift(1)），避免未来函数。

金融正确性检查：
- position = final_signal.shift(1, fill_value=0)，次日生效。
- cost = buy_turnover * (commission + slippage) + sell_turnover * (commission + sell_tax + slippage)。
- strategy_ret = position * stock_ret - cost。
- equity = cash * cumprod(1 + strategy_ret)。
- 滚动窗口只使用历史数据（由信号计算层负责）。
- 基准对齐使用 reindex + ffill，不引入未来数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from scripts.common.metrics import calc_metrics_from_returns


# ---------------------------------------------------------------------------
# 1. 回测参数配置
# ---------------------------------------------------------------------------

@dataclass
class BacktestCostModel:
    """A 股成本模型。"""
    commission: float = 0.0003      # 买入/卖出佣金
    sell_tax: float = 0.0005        # 卖出印花税
    slippage: float = 0.001         # 滑点

    @classmethod
    def a_share_default(cls) -> "BacktestCostModel":
        return cls(commission=0.0003, sell_tax=0.0005, slippage=0.001)


@dataclass
class BacktestConfig:
    """回测配置。"""
    cash: float = 1_000_000.0
    cost_model: BacktestCostModel = None

    def __post_init__(self):
        if self.cost_model is None:
            self.cost_model = BacktestCostModel.a_share_default()


# ---------------------------------------------------------------------------
# 2. 核心计算函数
# ---------------------------------------------------------------------------

def apply_market_filter(
    result: pd.DataFrame,
    benchmark_filter_df: pd.DataFrame,
) -> pd.DataFrame:
    """将大盘 regime filter 对齐到股票日期。

    参数：
        result: 包含 date 列的股票数据 DataFrame。
        benchmark_filter_df: 包含 date, close, benchmark_ma_short,
            benchmark_ma_long, market_filter 列的基准过滤结果。

    返回：
        添加了 benchmark_close, benchmark_ma_short, benchmark_ma_long,
        market_filter 列的 DataFrame。
    """
    bench = benchmark_filter_df.rename(columns={"close": "benchmark_close"})
    bench = bench.set_index("date", drop=False)
    result = result.set_index("date", drop=False).sort_index()

    result["benchmark_close"] = bench["benchmark_close"].reindex(result.index).ffill()
    result["benchmark_ma_short"] = bench["benchmark_ma_short"].reindex(result.index).ffill()
    result["benchmark_ma_long"] = bench["benchmark_ma_long"].reindex(result.index).ffill()
    result["market_filter"] = (
        bench["market_filter"].reindex(result.index).ffill().fillna(0).astype(int)
    )
    return result


def compute_final_signal(
    result: pd.DataFrame,
    signal_col: str = "alpha_signal",
) -> pd.DataFrame:
    """计算 final_signal = signal AND market_filter。

    参数：
        result: 包含 signal_col 和 market_filter 列的 DataFrame。
        signal_col: 信号列名，默认 "alpha_signal"。
            MA 策略可传 "trend_confirm"。

    返回：
        添加了 final_signal 列的 DataFrame。
    """
    result["final_signal"] = (
        (result[signal_col] == 1) & (result["market_filter"] == 1)
    ).astype(int)
    return result


def _compute_single_path(
    result: pd.DataFrame,
    signal_series: pd.Series,
    prefix: str,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> pd.DataFrame:
    """为一条信号路径计算 position / cost / ret 列。

    参数：
        result: 目标 DataFrame。
        signal_series: 0/1 信号序列（已对齐到 result.index）。
        prefix: 列名前缀，"" 表示主路径（position/cost/strategy_ret），
            "stock_only_" 表示对照路径。
        commission, sell_tax, slippage: 成本参数。

    返回：
        添加了 {prefix}position, {prefix}cost, {prefix}{suffix}_ret 列的 DataFrame。
    """
    pos_col = f"{prefix}position"
    cost_col = f"{prefix}cost"
    ret_col = f"{prefix}strategy_ret" if prefix == "" else f"{prefix}ret"

    result[pos_col] = signal_series.shift(1, fill_value=0).astype(float)

    pos_change = result[pos_col].diff().fillna(0)
    buy_turnover = pos_change.clip(lower=0)
    sell_turnover = (-pos_change).clip(lower=0)
    result[cost_col] = (
        buy_turnover * (commission + slippage)
        + sell_turnover * (commission + sell_tax + slippage)
    )

    result[ret_col] = result[pos_col] * result["stock_ret"] - result[cost_col]
    return result


def compute_position_and_returns(
    result: pd.DataFrame,
    commission: float,
    sell_tax: float,
    slippage: float,
    comparison_signal_col: str | None = None,
) -> pd.DataFrame:
    """计算次日持仓、交易成本和策略收益。

    核心逻辑（所有策略共享）：
    - position = final_signal.shift(1, fill_value=0)
    - stock_ret = close.pct_change().fillna(0)
    - cost = buy_turnover * (commission + slippage)
           + sell_turnover * (commission + sell_tax + slippage)
    - strategy_ret = position * stock_ret - cost

    可选对照路径（用于 MA 策略的 stock_only 对照）：
    - 如果 comparison_signal_col 不为 None，额外计算 stock_only_position,
      stock_only_cost, stock_only_ret 列。

    参数：
        result: 包含 close 和 final_signal 列的 DataFrame。
        commission: 佣金率。
        sell_tax: 卖出印花税率。
        slippage: 滑点率。
        comparison_signal_col: 可选对照信号列名。传入时额外计算
            stock_only_position / stock_only_cost / stock_only_ret。

    返回：
        添加了 position, stock_ret, cost, strategy_ret 列的 DataFrame。
        如果 comparison_signal_col 不为 None，还包含对照路径列。
    """
    result["stock_ret"] = result["close"].pct_change().fillna(0)

    # 主路径（final_signal）
    result = _compute_single_path(
        result, result["final_signal"], "",
        commission, sell_tax, slippage,
    )

    # 可选对照路径
    if comparison_signal_col is not None:
        result["stock_only_signal"] = result[comparison_signal_col]
        result = _compute_single_path(
            result, result[comparison_signal_col], "stock_only_",
            commission, sell_tax, slippage,
        )

    return result


def compute_equity_curves(
    result: pd.DataFrame,
    cash: float,
) -> pd.DataFrame:
    """计算权益曲线、买入持有权益、基准权益和回撤。

    参数：
        result: 包含 strategy_ret, stock_ret 列的 DataFrame。
            如果有 benchmark_ret 列，也会计算 benchmark_equity。
            如果有 stock_only_ret 列，也会计算 stock_only_equity。
        cash: 初始资金。

    返回：
        添加了 equity, buy_hold_equity, drawdown 列的 DataFrame。
        如果有 benchmark_ret 列，还会添加 benchmark_equity。
        如果有 stock_only_ret 列，还会添加 stock_only_equity。
    """
    result["equity"] = cash * (1.0 + result["strategy_ret"]).cumprod()
    result["buy_hold_equity"] = cash * (1.0 + result["stock_ret"]).cumprod()
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1.0

    if "benchmark_ret" in result.columns:
        result["benchmark_equity"] = cash * (1.0 + result["benchmark_ret"]).cumprod()

    if "stock_only_ret" in result.columns:
        result["stock_only_equity"] = cash * (1.0 + result["stock_only_ret"]).cumprod()

    return result


def build_backtest_metrics(
    result: pd.DataFrame,
    cash: float,
) -> dict:
    """从回测结果 DataFrame 构建完整指标 dict。

    返回的 dict 包含：
    - strategy_*: 策略指标
    - buy_hold_*: 买入持有指标
    - excess_vs_buy_hold_total_return: 超额收益
    - market_filter_on_ratio: 大盘过滤开启比例
    - strategy_exposure_ratio: 策略持仓比例

    如果 result 包含 stock_only_ret / stock_only_position 列，还会包含：
    - stock_only_*: 对照路径指标
    - excess_vs_stock_only_total_return: 超额 vs 对照路径
    - stock_only_exposure_ratio: 对照路径持仓比例

    参数：
        result: 包含 strategy_ret, stock_ret, position, market_filter 列的 DataFrame。
        cash: 初始资金。
    """
    strategy_metrics = calc_metrics_from_returns(result["strategy_ret"], result["position"], cash)
    buy_hold_metrics = calc_metrics_from_returns(result["stock_ret"], None, cash)

    metrics = {}
    for k, v in strategy_metrics.items():
        metrics[f"strategy_{k}"] = v
    for k, v in buy_hold_metrics.items():
        metrics[f"buy_hold_{k}"] = v

    metrics["excess_vs_buy_hold_total_return"] = (
        metrics["strategy_total_return"] - metrics["buy_hold_total_return"]
    )
    metrics["market_filter_on_ratio"] = float(result["market_filter"].mean())
    metrics["strategy_exposure_ratio"] = float(result["position"].mean())

    # 可选对照路径指标
    if "stock_only_ret" in result.columns and "stock_only_position" in result.columns:
        so_metrics = calc_metrics_from_returns(
            result["stock_only_ret"], result["stock_only_position"], cash,
        )
        for k, v in so_metrics.items():
            metrics[f"stock_only_{k}"] = v
        metrics["excess_vs_stock_only_total_return"] = (
            metrics["strategy_total_return"] - metrics["stock_only_total_return"]
        )
        metrics["stock_only_exposure_ratio"] = float(result["stock_only_position"].mean())

    return metrics


# ---------------------------------------------------------------------------
# 3. 高层入口
# ---------------------------------------------------------------------------

def single_asset_backtest(
    result: pd.DataFrame,
    benchmark_filter_df: pd.DataFrame,
    signal_col: str = "alpha_signal",
    cash: float = 1_000_000.0,
    commission: float = 0.0003,
    sell_tax: float = 0.0005,
    slippage: float = 0.001,
    compute_benchmark_ret: bool = False,
    comparison_signal_col: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """单标的回测完整流程。

    将 apply_market_filter → compute_final_signal → compute_position_and_returns
    → compute_equity_curves → build_backtest_metrics 串联为一个入口。

    参数：
        result: 已计算信号的 DataFrame，必须包含 date, close 和 signal_col 列。
        benchmark_filter_df: 大盘 regime filter 结果。
        signal_col: 信号列名，alpha 策略用 "alpha_signal"，MA 策略用 "trend_confirm"。
        cash: 初始资金。
        commission: 佣金率。
        sell_tax: 卖出印花税率。
        slippage: 滑点率。
        compute_benchmark_ret: 是否计算 benchmark_ret（需要 benchmark_close 列）。
        comparison_signal_col: 可选对照信号列名。传入时额外计算
            stock_only_position / stock_only_cost / stock_only_ret 及对应指标。

    返回：
        (result_df, metrics_dict)
    """
    result = apply_market_filter(result, benchmark_filter_df)
    result = compute_final_signal(result, signal_col)
    result = compute_position_and_returns(
        result, commission, sell_tax, slippage,
        comparison_signal_col=comparison_signal_col,
    )

    if compute_benchmark_ret and "benchmark_close" in result.columns:
        result["benchmark_ret"] = result["benchmark_close"].pct_change().fillna(0)

    result = compute_equity_curves(result, cash)
    metrics = build_backtest_metrics(result, cash)

    return result, metrics


def single_asset_backtest_lite(
    result: pd.DataFrame,
    benchmark_filter_df: pd.DataFrame,
    signal_col: str = "alpha_signal",
    commission: float = 0.0003,
    sell_tax: float = 0.0005,
    slippage: float = 0.001,
    comparison_signal_col: str | None = None,
) -> pd.DataFrame:
    """单标的回测精简版（不计算权益曲线和指标）。

    对应 wf_batch_shared.run_alpha_frame()，用于 walk-forward 训练期评估。
    不需要 cash 参数，不计算 equity 曲线。

    参数：
        result: 已计算信号的 DataFrame。
        benchmark_filter_df: 大盘 regime filter 结果。
        signal_col: 信号列名。
        commission: 佣金率。
        sell_tax: 卖出印花税率。
        slippage: 滑点率。
        comparison_signal_col: 可选对照信号列名。

    返回：
        包含 position, stock_ret, cost, strategy_ret 列的 DataFrame。
        如果 comparison_signal_col 不为 None，还包含对照路径列。
    """
    result = apply_market_filter(result, benchmark_filter_df)
    result = compute_final_signal(result, signal_col)
    result = compute_position_and_returns(
        result, commission, sell_tax, slippage,
        comparison_signal_col=comparison_signal_col,
    )
    return result
