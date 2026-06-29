# -*- coding: utf-8 -*-
"""
scripts/common/metrics.py

共享绩效指标计算。从 18 个文件中抽取的重复逻辑。

使用方式：
    from scripts.common.metrics import (
        format_pct,
        format_float,
        max_drawdown_from_equity,
        calc_metrics_from_returns,
        calc_metrics_from_dataframe,
        calc_metrics_simple,
        calc_portfolio_metrics,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# 格式化工具（原分散在 14+ 个文件中）
# ---------------------------------------------------------------------------

def format_pct(x: float) -> str:
    """格式化百分比，None 或 NaN 返回 'N/A'。"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.2%}"


def format_float(x: float) -> str:
    """格式化浮点数，None 或 NaN 返回 'N/A'。"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.4f}"


# ---------------------------------------------------------------------------
# 最大回撤
# ---------------------------------------------------------------------------

def max_drawdown_from_equity(equity: pd.Series) -> float:
    """从净值序列计算最大回撤（返回负数或 0）。

    原分散在 16 个文件中，函数名有 max_drawdown / calc_max_drawdown /
    max_drawdown_from_equity 三种变体，逻辑完全相同。
    """
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


# ---------------------------------------------------------------------------
# 从收益率序列计算指标（最通用的版本）
# ---------------------------------------------------------------------------

def calc_metrics_from_returns(
    ret: pd.Series,
    position: pd.Series | None = None,
    cash: float = 1_000_000.0,
    benchmark_ret: pd.Series | None = None,
    stock_ret: pd.Series | None = None,
) -> dict:
    """从日收益率序列计算核心绩效指标。

    原分散在 batch_*/validate_*/analyze_* 的 11 个文件中。
    此版本合并了所有常见参数变体，未使用的参数不影响结果。
    """
    ret = ret.dropna().astype(float)
    days = len(ret)
    if days == 0:
        return {
            "total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0,
            "sharpe": 0.0, "annual_volatility": 0.0, "days": 0,
            "trade_count": 0, "final_equity": cash, "calmar": 0.0,
        }

    equity = cash * (1 + ret).cumprod()
    total_return = float(equity.iloc[-1] / cash - 1.0)
    annual_return = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1.0
    std = float(ret.std())
    annual_volatility = std * SQRT_TRADING_DAYS_PER_YEAR
    mdd = max_drawdown_from_equity(equity)

    sharpe = float(ret.mean() / std * SQRT_TRADING_DAYS_PER_YEAR) if std > 0 else 0.0
    calmar = annual_return / abs(mdd) if mdd != 0 else 0.0

    trade_count = 0
    if position is not None:
        pos = position.dropna()
        if len(pos) > 1:
            changes = pos.diff().fillna(0)
            trade_count = int((changes != 0).sum())

    result = {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "annual_volatility": annual_volatility,
        "days": days,
        "trade_count": trade_count,
        "final_equity": float(equity.iloc[-1]),
        "calmar": calmar,
    }

    # 可选：buy-hold 和超额收益
    if stock_ret is not None:
        sr = stock_ret.dropna().astype(float)
        if len(sr) > 0:
            buy_hold_total_return = float((1 + sr).prod() - 1.0)
            result["buy_hold_total_return"] = buy_hold_total_return
            result["excess_total_return"] = total_return - buy_hold_total_return

    if benchmark_ret is not None:
        br = benchmark_ret.dropna().astype(float)
        if len(br) > 0:
            bm_total_return = float((1 + br).prod() - 1.0)
            result["benchmark_total_return"] = bm_total_return
            result["excess_vs_benchmark"] = total_return - bm_total_return

    return result


# ---------------------------------------------------------------------------
# 从 DataFrame 计算指标（策略文件用）
# ---------------------------------------------------------------------------

def calc_metrics_from_dataframe(
    result: pd.DataFrame,
    ret_col: str = "strategy_ret",
    equity_col: str = "equity",
) -> dict:
    """从策略回测结果 DataFrame 计算核心绩效指标。

    从 DataFrame 提取收益率和净值列，委托给 calc_metrics_from_returns。
    调用方可根据需要补充额外字段（如 buy_hold_total_return 等）。
    """
    ret = result[ret_col].dropna().astype(float)
    equity = result[equity_col].dropna()
    if len(ret) == 0 or len(equity) == 0:
        return {
            "total_return": 0.0, "annual_return": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "annual_volatility": 0.0,
            "trade_count": 0, "final_equity": 0.0, "days": 0, "calmar": 0.0,
        }

    position = result["position"] if "position" in result.columns else None
    metrics = calc_metrics_from_returns(ret, position=position, cash=float(equity.iloc[0]))
    # 用实际净值覆盖 final_equity（DataFrame 中的净值可能含前缀数据）
    metrics["final_equity"] = float(equity.iloc[-1])
    return metrics


# ---------------------------------------------------------------------------
# 简化版指标（walk-forward 分析用）
# ---------------------------------------------------------------------------

def calc_metrics_simple(ret: pd.Series) -> dict:
    """从日收益率序列计算简化绩效指标（含 calmar）。

    原分散在 analyze_walk_forward_results 等 4 个文件中。
    委托给 calc_metrics_from_returns，返回核心字段子集。
    """
    m = calc_metrics_from_returns(ret)
    return {
        "total_return": m["total_return"],
        "annual_return": m["annual_return"],
        "max_drawdown": m["max_drawdown"],
        "sharpe": m["sharpe"],
        "annual_volatility": m["annual_volatility"],
        "days": m["days"],
        "calmar": m["calmar"],
    }


# ---------------------------------------------------------------------------
# 从日收益率序列计算指标（统一版本）
# ---------------------------------------------------------------------------

def calc_metrics_from_daily(
    daily_returns: pd.Series,
    keys_only: bool = False,
    sharpe_method: str = "annualized_ratio",
) -> dict:
    """从日收益率序列计算核心绩效指标（统一版本。

    统一了以下两个历史实现：
    - wf_robustness_shared.compute_metrics_from_daily（Sharpe 用 annualized_ratio）
    - wf_report_shared.calc_metrics（Sharpe 用 mean_std，额外返回 days）

    Parameters
    ----------
    daily_returns : pd.Series
        日收益率序列。
    keys_only : bool, default False
        若 True，只返回 {total_return, annual_return, annual_volatility,
        max_drawdown, sharpe, calmar}，不含 turnover/commission/slippage。
    sharpe_method : str, default "annualized_ratio"
        Sharpe 计算方法：
        - "annualized_ratio": annual_return / annual_volatility
          （与 calc_portfolio_metrics、compute_metrics_from_daily 一致）
        - "mean_std": mean(r) / std(r) * sqrt(252)
          （与 calc_metrics 一致，用于 walk-forward 分析）

    Returns
    -------
    dict
        包含 total_return, annual_return, annual_volatility, max_drawdown,
        sharpe, calmar。当 keys_only=False 时额外包含 turnover,
        total_commission, total_slippage_cost（默认值 0/NaN）。
    """
    if daily_returns.empty:
        result = {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "calmar": np.nan,
        }
        if not keys_only:
            result["turnover"] = np.nan
            result["total_commission"] = 0.0
            result["total_slippage_cost"] = 0.0
        return result

    n_days = len(daily_returns)
    total_return = float((1 + daily_returns).prod() - 1)
    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
    annual_vol = float(daily_returns.std()) * SQRT_TRADING_DAYS_PER_YEAR

    equity = (1 + daily_returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())

    if sharpe_method == "mean_std":
        std = float(daily_returns.std())
        sharpe = float(daily_returns.mean() / std * SQRT_TRADING_DAYS_PER_YEAR) if std > 0 else np.nan
    else:
        sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    calmar = annual_return / abs(max_dd) if max_dd != 0 else np.nan

    result = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
    }
    if not keys_only:
        result["turnover"] = np.nan
        result["total_commission"] = 0.0
        result["total_slippage_cost"] = 0.0
    return result


# ---------------------------------------------------------------------------
# 组合层指标（原 portfolio_backtest_csv.py 内 compute_metrics）
# ---------------------------------------------------------------------------

def calc_portfolio_metrics(
    daily_returns: pd.Series,
    total_commission: float = 0.0,
    total_slippage: float = 0.0,
    turnover: float = np.nan,
) -> dict:
    """从组合日收益率序列计算组合层绩效指标。

    与 calc_metrics_from_returns 的区别：
    - 额外输出 total_commission、total_slippage_cost、turnover
    - 无 cash/equity 概念（组合净值已隐含在 daily_returns 中）
    - Sharpe 使用 annual_return / annual_volatility 口径（与原 portfolio_backtest_csv 一致）

    原位于 portfolio_backtest_csv.py 的 compute_metrics()。
    """
    if daily_returns.empty:
        return {
            "total_return": np.nan, "annual_return": np.nan, "annual_volatility": np.nan,
            "max_drawdown": np.nan, "sharpe": np.nan, "calmar": np.nan,
            "turnover": np.nan, "total_commission": 0.0, "total_slippage_cost": 0.0,
        }
    n_days = len(daily_returns)
    total_return = float((1 + daily_returns).prod() - 1)
    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
    annual_vol = float(daily_returns.std()) * SQRT_TRADING_DAYS_PER_YEAR
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    equity = (1 + daily_returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())
    calmar = annual_return / abs(max_dd) if max_dd != 0 else np.nan
    return {
        "total_return": total_return, "annual_return": annual_return,
        "annual_volatility": annual_vol, "max_drawdown": max_dd,
        "sharpe": sharpe, "calmar": calmar, "turnover": turnover,
        "total_commission": total_commission, "total_slippage_cost": total_slippage,
    }
