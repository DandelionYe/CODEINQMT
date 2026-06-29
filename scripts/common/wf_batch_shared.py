# -*- coding: utf-8 -*-
"""
wf_batch_shared.py

Alpha v6/v7 batch 回测和 walk-forward validate 脚本共享逻辑。

设计原则：
- 所有版本无关的工具函数、评分、参数组合、回测框架、目录构建等集中于此。
- 每个 batch/validate 脚本只需定义 WFConfig/WFValidateConfig 并调用共享函数。
- 与 wf_report_shared.py（analysis/diagnosis 共享）对应。
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from scripts.common.backtest.engine import (
    single_asset_backtest,
    single_asset_backtest_lite,
)
from scripts.common.constants import (
    DEFAULT_BENCHMARK_LIST,
    SQRT_TRADING_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
)
from scripts.common.metrics import calc_metrics_from_returns, max_drawdown_from_equity
from scripts.common.validation import (
    parse_date_yyyymmdd,
    parse_int_list,
    parse_list,
    safe_symbol_tag,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. WFConfig — 批量回测配置
# ---------------------------------------------------------------------------

@dataclass
class WFConfig:
    """封装 v6/v7 batch 脚本的版本差异。"""

    alpha_version: str  # e.g. "v6" or "v7"
    output_dir_name: str  # e.g. "batch_alpha_v6_research_csv"
    compute_signals_fn: Callable  # compute_alpha_v6_signals or compute_alpha_v7_signals
    prepare_benchmark_regime_fn: Callable  # prepare_benchmark_regime
    valid_variants: list[str]
    variant_tags: dict[str, str] = field(default_factory=lambda: {
        "short_term_reversal": "str",
        "low_volatility": "lv",
        "turnover_reversal": "tr",
        "volume_price_divergence": "vpd",
    })


# ---------------------------------------------------------------------------
# 2. WFValidateConfig — walk-forward 验证配置
# ---------------------------------------------------------------------------

@dataclass
class WFValidateConfig:
    """封装 v6/v7 validate 脚本的版本差异。"""

    alpha_version: str  # e.g. "v6" or "v7"
    output_dir_name: str  # e.g. "walk_forward_alpha_v6_research_csv"
    file_prefix: str  # e.g. "wf_alpha_v6_stock"
    report_title: str  # e.g. "Alpha v6 Walk-Forward Report"
    compute_signals_fn: Callable
    prepare_benchmark_regime_fn: Callable
    valid_variants: list[str]
    variant_tags: dict[str, str] = field(default_factory=lambda: {
        "short_term_reversal": "str",
        "low_volatility": "lv",
        "turnover_reversal": "tr",
        "volume_price_divergence": "vpd",
    })


# ---------------------------------------------------------------------------
# 3. 通用工具函数（batch + validate 共享）
# ---------------------------------------------------------------------------

def build_variant_param_combos(
    alpha_variant_list: list[str],
    reversal_window_list: list[int],
    vol_window_list: list[int],
    turnover_short_list: list[int],
    turnover_long_list: list[int],
    divergence_window_list: list[int],
) -> list[tuple[str, int, int, int, int, int]]:
    """根据 alpha variant 生成 variant-aware 参数组合，避免无意义重复。

    返回 (alpha_variant, reversal_window, vol_window, turnover_short, turnover_long, divergence_window)。
    """
    combos = []
    for av in alpha_variant_list:
        if av == "short_term_reversal":
            for rw in reversal_window_list:
                combos.append((av, rw, vol_window_list[0], turnover_short_list[0], turnover_long_list[0], divergence_window_list[0]))
        elif av == "low_volatility":
            for vw in vol_window_list:
                combos.append((av, reversal_window_list[0], vw, turnover_short_list[0], turnover_long_list[0], divergence_window_list[0]))
        elif av == "turnover_reversal":
            for ts in turnover_short_list:
                for tl in turnover_long_list:
                    combos.append((av, reversal_window_list[0], vol_window_list[0], ts, tl, divergence_window_list[0]))
        elif av == "volume_price_divergence":
            for dw in divergence_window_list:
                combos.append((av, reversal_window_list[0], vol_window_list[0], turnover_short_list[0], turnover_long_list[0], dw))
        else:
            raise ValueError(f"未知的 alpha_variant: {av}，请检查 --alpha-variant-list")
    return combos


def compact_int_list(values: list[int]) -> str:
    vals = sorted(dict.fromkeys(int(v) for v in values))
    if not vals:
        return "none"
    if len(vals) == 1:
        return str(vals[0])
    return f"{vals[0]}-{vals[-1]}x{len(vals)}"


def compact_variant_list(values: list[str], variant_tags: dict[str, str]) -> str:
    tags = [variant_tags.get(v, v[:3]) for v in values]
    if len(tags) == len(variant_tags) and set(values) == set(variant_tags):
        return "all4"
    return "-".join(tags)


def compact_benchmark_list(values: list[str]) -> str:
    if len(values) == 1:
        return safe_symbol_tag(values[0])
    return f"bm{len(values)}"


def build_param_signature(
    alpha_variant_list: list[str],
    reversal_window_list: list[int],
    vol_window_list: list[int],
    turnover_short_list: list[int],
    turnover_long_list: list[int],
    divergence_window_list: list[int],
    benchmark_list: list[str],
    benchmark_ma_list: list[int],
) -> str:
    return "|".join([
        ",".join(alpha_variant_list),
        ",".join(map(str, reversal_window_list)),
        ",".join(map(str, vol_window_list)),
        ",".join(map(str, turnover_short_list)),
        ",".join(map(str, turnover_long_list)),
        ",".join(map(str, divergence_window_list)),
        ",".join(benchmark_list),
        ",".join(map(str, benchmark_ma_list)),
    ])


def build_batch_tag(
    cfg: WFConfig,
    alpha_variant_list: list[str],
    reversal_window_list: list[int],
    vol_window_list: list[int],
    turnover_short_list: list[int],
    turnover_long_list: list[int],
    divergence_window_list: list[int],
    benchmark_list: list[str],
    benchmark_ma_list: list[int],
    sample_mode: str,
) -> str:
    signature = build_param_signature(
        alpha_variant_list, reversal_window_list, vol_window_list,
        turnover_short_list, turnover_long_list, divergence_window_list,
        benchmark_list, benchmark_ma_list,
    )
    short_hash = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    return (
        f"{cfg.alpha_version}_av{compact_variant_list(alpha_variant_list, cfg.variant_tags)}"
        f"_rw{compact_int_list(reversal_window_list)}"
        f"_vw{compact_int_list(vol_window_list)}"
        f"_ts{compact_int_list(turnover_short_list)}"
        f"_tl{compact_int_list(turnover_long_list)}"
        f"_dw{compact_int_list(divergence_window_list)}"
        f"_{compact_benchmark_list(benchmark_list)}"
        f"_bma{compact_int_list(benchmark_ma_list)}"
        f"_{sample_mode}"
        f"_h{short_hash}"
    )


# ---------------------------------------------------------------------------
# 4. 批量回测核心函数
# ---------------------------------------------------------------------------

def run_one_backtest(
    stock_df: pd.DataFrame,
    benchmark_filter_df: pd.DataFrame,
    compute_signals_fn: Callable,
    alpha_variant: str,
    reversal_window: int,
    vol_window: int,
    turnover_short: int,
    turnover_long: int,
    divergence_window: int,
    cash: float,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> tuple[pd.DataFrame, dict]:
    """使用指定信号函数运行单只股票回测，返回结果 DataFrame 和指标。

    委托给共享回测引擎 single_asset_backtest()，保证与所有策略版本回测口径一致。
    """
    result = compute_signals_fn(
        stock_df, alpha_variant, reversal_window, vol_window,
        turnover_short, turnover_long, divergence_window,
    )

    result, metrics = single_asset_backtest(
        result=result,
        benchmark_filter_df=benchmark_filter_df,
        signal_col="alpha_signal",
        cash=cash,
        commission=commission,
        sell_tax=sell_tax,
        slippage=slippage,
    )

    return result, metrics


def calc_score(metrics: dict) -> float:
    return (
        metrics.get("strategy_annual_return", 0)
        + 0.20 * metrics.get("strategy_sharpe", 0)
        + 0.30 * metrics.get("excess_vs_buy_hold_total_return", 0)
        + metrics.get("strategy_max_drawdown", 0)
    )


def get_required_rows(
    sample_mode: str, reversal_window: int, vol_window: int,
    turnover_long: int, divergence_window: int,
    benchmark_ma: int, warmup_buffer: int,
    long_min_rows: int, min_rows_arg: int,
) -> int:
    warmup_required = max(
        reversal_window, vol_window, turnover_long, divergence_window,
        int(benchmark_ma * 2.5),
    ) + warmup_buffer
    if sample_mode == "short":
        return warmup_required
    elif sample_mode == "long":
        return max(long_min_rows, warmup_required)
    else:
        return max(min_rows_arg, warmup_required)


def process_one_stock(args_tuple):
    """单进程 worker：对一只股票遍历所有参数组合。"""
    (item, param_combos, benchmark_cache_data, start, end,
     cash, commission, sell_tax, slippage,
     sample_mode, warmup_buffer, long_min_rows, min_rows_arg,
     compute_signals_fn) = args_tuple

    rows = []
    skipped = []
    errors = []

    try:
        # Lazy import to avoid circular dependency at module load time
        from strategies import ma_demo_strategy_csv as ma

        csv_path = Path(item["csv_path"])
        try:
            df = ma.load_qmt_price_csv(csv_path=csv_path, start=start, end=end)
        except RuntimeError:
            skipped.append({"symbol": item.get("symbol", ""), "reason": "数据为空或不足"})
            return rows, skipped, errors

        for alpha_variant, reversal_window, vol_window, turnover_short, turnover_long, divergence_window in param_combos:
            for bm_key, bm_data in benchmark_cache_data.items():
                benchmark_symbol = bm_data["benchmark"]
                benchmark_ma = bm_data["benchmark_ma"]
                benchmark_csv_path = Path(bm_data["csv_path"])
                filter_df = bm_data["filter_df"]

                required_rows = get_required_rows(
                    sample_mode, reversal_window, vol_window, turnover_long, divergence_window,
                    benchmark_ma, warmup_buffer, long_min_rows, min_rows_arg,
                )

                if len(df) < required_rows:
                    skipped.append({
                        "symbol": item.get("symbol", ""),
                        "reason": f"rows={len(df)} < {required_rows}",
                    })
                    continue

                try:
                    _, metrics = run_one_backtest(
                        stock_df=df,
                        benchmark_filter_df=filter_df,
                        compute_signals_fn=compute_signals_fn,
                        alpha_variant=alpha_variant,
                        reversal_window=reversal_window,
                        vol_window=vol_window,
                        turnover_short=turnover_short,
                        turnover_long=turnover_long,
                        divergence_window=divergence_window,
                        cash=cash,
                        commission=commission,
                        sell_tax=sell_tax,
                        slippage=slippage,
                    )

                    row = {
                        "symbol": item.get("symbol", ""),
                        "market": item.get("market", ""),
                        "security_type": item.get("security_type", ""),
                        "alpha_variant": alpha_variant,
                        "reversal_window": reversal_window,
                        "vol_window": vol_window,
                        "turnover_short": turnover_short,
                        "turnover_long": turnover_long,
                        "divergence_window": divergence_window,
                        "benchmark": benchmark_symbol,
                        "benchmark_ma": benchmark_ma,
                        "sample_mode": sample_mode,
                        "required_rows": required_rows,
                        "start_date": str(df["date"].min()) if not df.empty else "",
                        "end_date": str(df["date"].max()) if not df.empty else "",
                        "rows": len(df),
                        "csv_path": str(item.get("csv_path", "")),
                        "benchmark_csv_path": str(benchmark_csv_path),
                    }
                    row.update(metrics)
                    row["score"] = calc_score(metrics)
                    rows.append(row)
                except Exception as e:
                    errors.append({
                        "symbol": item.get("symbol", ""),
                        "alpha_variant": alpha_variant,
                        "error": str(e),
                    })
    except Exception as e:
        errors.append({"symbol": item.get("symbol", ""), "error": str(e)})

    return rows, skipped, errors


# ---------------------------------------------------------------------------
# 5. Walk-forward 验证核心函数
# ---------------------------------------------------------------------------

def calc_train_score(metrics: dict) -> float:
    return (
        metrics.get("annual_return", 0)
        + 0.25 * metrics.get("sharpe", 0)
        + 0.20 * metrics.get("excess_vs_buy_hold_total_return", 0)
        + 0.80 * metrics.get("max_drawdown", 0)
    )


def pass_train_filters_simple(
    metrics: dict,
    min_train_rows: int,
    min_train_trades: int,
    max_train_drawdown: float,
    min_train_sharpe: float,
    min_train_annual_return: float,
    max_train_volatility: float,
    min_train_calmar: float,
) -> bool:
    if pd.isna(metrics["annual_return"]) or pd.isna(metrics["max_drawdown"]) or pd.isna(metrics["sharpe"]):
        return False
    if metrics["days"] < min_train_rows:
        return False
    if metrics["trade_count"] < min_train_trades:
        return False
    if metrics["max_drawdown"] < max_train_drawdown:
        return False
    if metrics["sharpe"] < min_train_sharpe:
        return False
    if metrics["annual_return"] < min_train_annual_return:
        return False
    if max_train_volatility > 0 and metrics["annual_volatility"] > max_train_volatility:
        return False
    if min_train_calmar > 0:
        if metrics["max_drawdown"] == 0:
            return False
        calmar = metrics["annual_return"] / abs(metrics["max_drawdown"])
        if calmar < min_train_calmar:
            return False
    return True


def run_alpha_frame(
    stock_df: pd.DataFrame,
    benchmark_filter_df: pd.DataFrame,
    compute_signals_fn: Callable,
    alpha_variant: str,
    reversal_window: int,
    vol_window: int,
    turnover_short: int,
    turnover_long: int,
    divergence_window: int,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> pd.DataFrame:
    """使用指定信号函数运行策略框架，返回完整结果 DataFrame。

    委托给共享回测引擎 single_asset_backtest_lite()，保证与所有策略版本回测口径一致。
    """
    result = compute_signals_fn(
        stock_df, alpha_variant, reversal_window, vol_window,
        turnover_short, turnover_long, divergence_window,
    )

    result = single_asset_backtest_lite(
        result=result,
        benchmark_filter_df=benchmark_filter_df,
        signal_col="alpha_signal",
        commission=commission,
        sell_tax=sell_tax,
        slippage=slippage,
    )

    return result


def train_one_stock_all_years(args_tuple, compute_signals_fn: Callable):
    """处理一只股票跨所有测试年份的训练候选，供并行 worker 使用。

    需要外部传入 compute_signals_fn，因为 multiprocessing worker
    不能直接序列化闭包。
    """
    from strategies import ma_demo_strategy_csv as ma

    (
        item_dict,
        test_years,
        train_start,
        end,
        param_combos,
        benchmark_cache_data,
        commission,
        sell_tax,
        slippage,
        cash,
        warmup_buffer,
        min_train_rows,
        min_train_trades,
        max_train_drawdown,
        min_train_sharpe,
        min_train_annual_return,
        max_train_volatility,
        min_train_calmar,
    ) = args_tuple

    symbol = item_dict["symbol"]
    market = item_dict["market"]
    security_type = item_dict["security_type"]
    csv_path = Path(item_dict["csv_path"])

    try:
        full_df = ma.load_qmt_price_csv(csv_path, train_start, end)
    except Exception:
        return {}

    train_start_dt = parse_date_yyyymmdd(train_start)
    results_by_year = {}

    for year in test_years:
        train_end_dt = pd.Timestamp(year=year - 1, month=12, day=31)
        if train_end_dt < train_start_dt:
            continue

        train_df = full_df[(full_df["date"] >= train_start_dt) & (full_df["date"] <= train_end_dt)].copy()
        if len(train_df) < min_train_rows:
            continue

        best_candidate = None
        best_score = -np.inf

        for alpha_variant, reversal_window, vol_window, turnover_short, turnover_long, divergence_window in param_combos:
            for (bm_symbol, bm_ma), bm_data in benchmark_cache_data.items():
                required_rows = max(reversal_window, vol_window, turnover_long, divergence_window, int(bm_ma * 2.5)) + warmup_buffer
                if len(train_df) < required_rows:
                    continue

                filter_df = bm_data["filter_df"]
                filter_train = filter_df[
                    (filter_df["date"] >= train_start_dt)
                    & (filter_df["date"] <= train_end_dt)
                ].copy()

                if filter_train.empty:
                    continue

                try:
                    result = run_alpha_frame(
                        train_df, filter_train,
                        compute_signals_fn,
                        alpha_variant, reversal_window, vol_window,
                        turnover_short, turnover_long, divergence_window,
                        commission, sell_tax, slippage,
                    )
                    metrics = calc_metrics_from_returns(
                        result["strategy_ret"],
                        position=result["position"],
                        cash=cash,
                        stock_ret=result["stock_ret"],
                    )
                    metrics["excess_vs_buy_hold_total_return"] = metrics.pop("excess_total_return", np.nan)

                    if not pass_train_filters_simple(
                        metrics,
                        min_train_rows,
                        min_train_trades,
                        max_train_drawdown,
                        min_train_sharpe,
                        min_train_annual_return,
                        max_train_volatility,
                        min_train_calmar,
                    ):
                        continue

                    score = calc_train_score(metrics)
                    if score <= best_score:
                        continue

                    best_score = score
                    best_candidate = {
                        "symbol": symbol,
                        "market": market,
                        "security_type": security_type,
                        "csv_path": str(csv_path),
                        "alpha_variant": alpha_variant,
                        "reversal_window": reversal_window,
                        "vol_window": vol_window,
                        "turnover_short": turnover_short,
                        "turnover_long": turnover_long,
                        "divergence_window": divergence_window,
                        "benchmark": bm_symbol,
                        "benchmark_ma": bm_ma,
                        "benchmark_csv_path": bm_data["csv_path"],
                        "train_start": str(train_start_dt.date()),
                        "train_end": str(train_end_dt.date()),
                        "train_rows": len(train_df),
                        "train_score": score,
                        "train_total_return": metrics["total_return"],
                        "train_annual_return": metrics["annual_return"],
                        "train_sharpe": metrics["sharpe"],
                        "train_max_drawdown": metrics["max_drawdown"],
                        "train_annual_volatility": metrics["annual_volatility"],
                        "train_trade_count": metrics["trade_count"],
                        "train_buy_hold_total_return": metrics["buy_hold_total_return"],
                        "train_excess_vs_buy_hold": metrics["excess_vs_buy_hold_total_return"],
                        "train_market_filter_on_ratio": float(result["market_filter"].mean()),
                        "train_strategy_exposure_ratio": float(result["position"].mean()),
                    }
                except Exception:
                    continue

        if best_candidate is not None:
            results_by_year[year] = best_candidate

    return results_by_year


def test_selected_for_period(
    selected: pd.DataFrame,
    data_cache: dict,
    benchmark_cache: dict,
    test_start_dt: pd.Timestamp,
    test_end_dt: pd.Timestamp,
    args: argparse.Namespace,
    compute_signals_fn: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对选中候选执行测试期回测，返回明细和收益矩阵。"""
    detail_rows = []
    returns_dict = {}

    for _, cand in selected.iterrows():
        symbol = cand["symbol"]
        csv_path = cand["csv_path"]
        alpha_variant = cand["alpha_variant"]
        reversal_window = int(cand["reversal_window"])
        vol_window = int(cand["vol_window"])
        turnover_short = int(cand["turnover_short"])
        turnover_long = int(cand["turnover_long"])
        divergence_window = int(cand["divergence_window"])
        bm_symbol = cand["benchmark"]
        bm_ma = int(cand["benchmark_ma"])

        if csv_path not in data_cache:
            continue
        full_df = data_cache[csv_path]
        bm_data = benchmark_cache.get((bm_symbol, bm_ma))
        if bm_data is None:
            continue

        full_result = run_alpha_frame(
            full_df, bm_data["filter_df"],
            compute_signals_fn,
            alpha_variant, reversal_window, vol_window,
            turnover_short, turnover_long, divergence_window,
            args.commission, args.sell_tax, args.slippage,
        )

        test_result = full_result[
            (full_result["date"] >= test_start_dt) & (full_result["date"] <= test_end_dt)
        ].copy()

        if test_result.empty:
            continue

        test_metrics = calc_metrics_from_returns(
            test_result["strategy_ret"],
            position=test_result["position"],
            stock_ret=test_result["stock_ret"],
        )
        test_metrics["excess_vs_buy_hold_total_return"] = test_metrics.pop("excess_total_return", np.nan)

        detail = {
            "symbol": symbol,
            "alpha_variant": alpha_variant,
            "reversal_window": reversal_window,
            "vol_window": vol_window,
            "turnover_short": turnover_short,
            "turnover_long": turnover_long,
            "divergence_window": divergence_window,
            "benchmark": bm_symbol,
            "benchmark_ma": bm_ma,
            "selected_rank": cand.get("selected_rank", 0),
            "train_score": cand.get("train_score", np.nan),
            "train_annual_return": cand.get("train_annual_return", np.nan),
            "train_sharpe": cand.get("train_sharpe", np.nan),
            "train_max_drawdown": cand.get("train_max_drawdown", np.nan),
            "train_annual_volatility": cand.get("train_annual_volatility", np.nan),
            "train_excess_vs_buy_hold": cand.get("train_excess_vs_buy_hold", np.nan),
            "test_total_return": test_metrics["total_return"],
            "test_annual_return": test_metrics["annual_return"],
            "test_sharpe": test_metrics["sharpe"],
            "test_max_drawdown": test_metrics["max_drawdown"],
            "test_annual_volatility": test_metrics["annual_volatility"],
            "test_excess_vs_buy_hold": test_metrics["excess_vs_buy_hold_total_return"],
            "test_days": test_metrics["days"],
            "test_trade_count": test_metrics["trade_count"],
        }
        detail_rows.append(detail)

        test_ret = test_result[["date", "strategy_ret"]].copy()
        test_ret = test_ret.rename(columns={"strategy_ret": symbol})
        test_ret = test_ret.set_index("date")
        returns_dict[symbol] = test_ret[symbol]

    detail_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()
    returns_df = pd.DataFrame(returns_dict) if returns_dict else pd.DataFrame()
    return detail_df, returns_df


def calc_portfolio_period_metrics(
    returns_df: pd.DataFrame,
    test_year: int,
    cash: float,
    portfolio_ret: pd.Series | None = None,
) -> dict:
    if returns_df.empty:
        return {"test_year": test_year, "portfolio_size_actual": 0}

    if portfolio_ret is None:
        portfolio_ret = returns_df.mean(axis=1)
    total_return = (1.0 + portfolio_ret).prod() - 1.0
    equity = cash * (1.0 + portfolio_ret).cumprod()
    days = max(len(portfolio_ret), 1)
    annual_return = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1.0
    annual_volatility = portfolio_ret.std() * SQRT_TRADING_DAYS_PER_YEAR
    sharpe = portfolio_ret.mean() / portfolio_ret.std() * SQRT_TRADING_DAYS_PER_YEAR if portfolio_ret.std() > 0 else np.nan

    return {
        "test_year": test_year,
        "portfolio_size_actual": len(returns_df.columns),
        "period_start": str(returns_df.index.min()),
        "period_end": str(returns_df.index.max()),
        "days": days,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "max_drawdown": max_drawdown_from_equity(equity),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
    }


def build_validate_tag(cfg: WFValidateConfig, args: argparse.Namespace) -> str:
    """构建 walk-forward 输出文件名标签。"""
    alpha_variant_list = parse_list(args.alpha_variant_list, upper=False)
    reversal_window_list = parse_int_list(args.reversal_window_list)
    vol_window_list = parse_int_list(args.vol_window_list)
    turnover_short_list = parse_int_list(args.turnover_short_list)
    turnover_long_list = parse_int_list(args.turnover_long_list)
    divergence_window_list = parse_int_list(args.divergence_window_list)
    bm_list = [x.strip().upper() for x in args.benchmark_list.split(",") if x.strip()]
    bm_ma_list = parse_int_list(args.benchmark_ma_list)
    last_test = args.last_test_year if args.last_test_year > 0 else "latest"
    limit_tag = f"l{args.limit}" if args.limit > 0 else "all"
    signature = build_param_signature(
        alpha_variant_list, reversal_window_list, vol_window_list,
        turnover_short_list, turnover_long_list, divergence_window_list,
        bm_list, bm_ma_list,
    )
    short_hash = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]

    return (
        f"{cfg.alpha_version}_{args.market}_{args.security_type}"
        f"_ts{args.train_start}"
        f"_fy{args.first_test_year}-{last_test}"
        f"_av{compact_variant_list(alpha_variant_list, cfg.variant_tags)}"
        f"_rw{compact_int_list(reversal_window_list)}"
        f"_vw{compact_int_list(vol_window_list)}"
        f"_ts{compact_int_list(turnover_short_list)}"
        f"_tl{compact_int_list(turnover_long_list)}"
        f"_dw{compact_int_list(divergence_window_list)}"
        f"_{compact_benchmark_list(bm_list)}"
        f"_bma{compact_int_list(bm_ma_list)}"
        f"_top{args.portfolio_size}"
        f"_{limit_tag}"
        f"_h{short_hash}"
    )


def save_validate_outputs(
    cfg: WFValidateConfig,
    selected_all: pd.DataFrame,
    test_detail_all: pd.DataFrame,
    portfolio_daily_all: pd.DataFrame,
    portfolio_periods: list[dict],
    args: argparse.Namespace,
    output_dir: Path | None = None,
) -> None:
    """保存 walk-forward 验证输出文件。"""
    tag = build_validate_tag(cfg, args)
    if output_dir is None:
        output_dir = Path(cfg.output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = output_dir / f"{cfg.file_prefix}_{tag}_selected_by_year.csv"
    detail_path = output_dir / f"{cfg.file_prefix}_{tag}_test_detail.csv"
    daily_path = output_dir / f"{cfg.file_prefix}_{tag}_portfolio_daily.csv"
    period_path = output_dir / f"{cfg.file_prefix}_{tag}_portfolio_period_summary.csv"
    report_path = output_dir / f"{cfg.file_prefix}_{tag}_report.txt"

    if not selected_all.empty:
        selected_all.to_csv(selected_path, encoding="utf-8-sig", index=False)
    if not test_detail_all.empty:
        test_detail_all.to_csv(detail_path, encoding="utf-8-sig", index=False)
    if not portfolio_daily_all.empty:
        portfolio_daily_all.to_csv(daily_path, encoding="utf-8-sig", index=False)
    if portfolio_periods:
        pd.DataFrame(portfolio_periods).to_csv(period_path, encoding="utf-8-sig", index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{cfg.report_title}\n")
        f.write("=" * 60 + "\n\n")
        if cfg.alpha_version == "v7":
            f.write("signal_source: expression_layer (feature_expression.py)\n")
        f.write(f"train_start: {args.train_start}\n")
        f.write(f"first_test_year: {args.first_test_year}\n")
        f.write(f"last_test_year: {args.last_test_year}\n")
        f.write(f"portfolio_size: {args.portfolio_size}\n")
        f.write(f"alpha_variant_list: {args.alpha_variant_list}\n")
        f.write(f"reversal_window_list: {args.reversal_window_list}\n")
        f.write(f"vol_window_list: {args.vol_window_list}\n")
        f.write(f"turnover_short_list: {args.turnover_short_list}\n")
        f.write(f"turnover_long_list: {args.turnover_long_list}\n")
        f.write(f"divergence_window_list: {args.divergence_window_list}\n")
        f.write(f"benchmark_list: {args.benchmark_list}\n")
        f.write(f"benchmark_ma_list: {args.benchmark_ma_list}\n")
        f.write(f"market: {args.market}\n")
        f.write(f"security_type: {args.security_type}\n\n")

        if portfolio_periods:
            f.write("Portfolio Period Summary:\n")
            f.write("-" * 40 + "\n")
            for p in portfolio_periods:
                f.write(f"  {p['test_year']}: return={p.get('total_return', np.nan):.4f}, "
                        f"sharpe={p.get('sharpe', np.nan):.4f}, "
                        f"max_dd={p.get('max_drawdown', np.nan):.4f}, "
                        f"size={p.get('portfolio_size_actual', 0)}\n")

            valid_periods = [p for p in portfolio_periods if "total_return" in p and not np.isnan(p["total_return"])]
            if valid_periods:
                overall_equity = 1.0
                for p in valid_periods:
                    overall_equity *= (1.0 + p["total_return"])
                overall_return = overall_equity - 1.0
                years = len(valid_periods)
                overall_annual = (1.0 + overall_return) ** (1.0 / years) - 1.0 if years > 0 else 0
                f.write(f"\nOverall: total_return={overall_return:.4f}, annual_return={overall_annual:.4f}, years={years}\n")

    logger.info("结果已保存到：%s", output_dir)
    logger.info("  %s", selected_path)
    logger.info("  %s", detail_path)
    logger.info("  %s", daily_path)
    logger.info("  %s", period_path)
    logger.info("  %s", report_path)


# ---------------------------------------------------------------------------
# 6. 数据加载和目录构建
# ---------------------------------------------------------------------------

def load_symbol_data(csv_path: Path, start: str, end: str):
    """加载单只股票数据，返回 DataFrame 或 None。"""
    from strategies import ma_demo_strategy_csv as ma
    try:
        return ma.load_qmt_price_csv(csv_path=csv_path, start=start, end=end)
    except RuntimeError:
        return None


def get_test_years(args: argparse.Namespace) -> list[int]:
    first = args.first_test_year
    last = args.last_test_year
    if last <= 0:
        last = datetime.now().year
    return list(range(first, last + 1))


def build_catalog(args: argparse.Namespace) -> pd.DataFrame:
    """根据 CLI 参数构建股票目录。"""
    from strategies import ma_demo_strategy_csv as ma

    export_root = Path(args.export_root)
    from pathlib import PurePosixPath
    project_root = Path(__file__).resolve().parents[2]
    if not export_root.is_absolute():
        export_root = project_root / export_root
    catalog = ma.scan_qmt_export(export_root)
    if args.market != "ALL":
        catalog = catalog[catalog["market"] == args.market]
    if args.security_type != "ALL":
        catalog = catalog[catalog["security_type"] == args.security_type]
    if args.limit > 0:
        catalog = catalog.head(args.limit)
    return catalog


def load_benchmark_cache(
    prepare_benchmark_regime_fn: Callable,
    benchmark_list_str: str,
    benchmark_ma_list_str: str,
    train_start: str,
    end: str,
    export_root: Path,
) -> dict:
    """预加载基准数据到缓存。"""
    from strategies import ma_demo_strategy_csv as ma

    benchmark_list = [x.strip().upper() for x in benchmark_list_str.split(",") if x.strip()]
    benchmark_ma_list = parse_int_list(benchmark_ma_list_str)

    cache = {}
    for bm_symbol in benchmark_list:
        for bm_ma in benchmark_ma_list:
            bm_csv_path, _, _, _ = ma.find_csv_for_stock(bm_symbol, export_root)
            bm_df = ma.load_qmt_price_csv(bm_csv_path, train_start, end)
            filter_df = prepare_benchmark_regime_fn(bm_df, bm_ma)
            cache[(bm_symbol, bm_ma)] = {
                "benchmark": bm_symbol,
                "benchmark_ma": bm_ma,
                "csv_path": str(bm_csv_path),
                "filter_df": filter_df,
            }
    return cache
