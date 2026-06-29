# -*- coding: utf-8 -*-
"""
validate_ma_v3_momentum_candidates.py

MA v3 walk-forward / 滚动样本外验证：
个股趋势强度 / 动量排序 + 趋势确认 + 风险过滤

核心思想：
每一个测试年度，只使用该年度之前的数据进行训练、选股、选参数；
然后在该年度进行样本外测试。

策略逻辑：
1. 趋势确认：
   close > ma_long AND ma_mid > ma_long AND ma_long_slope > 0

2. 动量 / 强度评分：
   trend_strength + momentum_return + trend_slope 组合

3. 大盘 regime filter：
   benchmark_ma_short > benchmark_ma_long

4. 最终信号：
   final_signal = trend_confirm AND market_filter

运行示例：

1. 小规模测试：
python scripts\\validate_ma_v3_momentum_candidates.py --market SZ --limit 300 --first-test-year 2024 --last-test-year 2024 --portfolio-size 10 --ma-mid-list 60 --ma-long-list 250 --momentum-window-list 120

2. 全市场 v3 walk-forward：
python scripts\\validate_ma_v3_momentum_candidates.py --market ALL --security-type stock --first-test-year 2021 --portfolio-size 20 --ma-mid-list 20,60 --ma-long-list 120,250 --momentum-window-list 60,120 --benchmark-list 000300.SH,000905.SH,000852.SH --benchmark-ma-list 120,250
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR, DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.metrics import max_drawdown_from_equity, calc_metrics_from_returns  # noqa: E402
from scripts.common.validation import parse_list as parse_symbol_list, parse_int_list, parse_workers, safe_symbol_tag, parse_date_yyyymmdd  # noqa: E402
from scripts.common.feature_expression import normalize_zscore  # noqa: E402
from scripts.common.benchmark import prepare_benchmark_regime  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "backtests" / "walk_forward_ma_v3_momentum_csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _enrich_stock_only_metrics(metrics: dict, stock_only_ret: pd.Series) -> None:
    """将 stock_only_ret 的超额收益注入 calc_metrics_from_returns 的结果 dict。"""
    metrics["excess_vs_buy_hold_total_return"] = metrics.pop("excess_total_return", np.nan)
    sr = stock_only_ret.dropna().astype(float)
    if len(sr) > 0:
        metrics["stock_only_total_return"] = float((1 + sr).prod() - 1.0)
        metrics["excess_vs_stock_only_total_return"] = float(
            metrics.get("total_return", 0.0) - metrics["stock_only_total_return"]
        )
    else:
        metrics["stock_only_total_return"] = np.nan
        metrics["excess_vs_stock_only_total_return"] = np.nan


def compute_v3_signals(stock_df: pd.DataFrame, ma_mid: int, ma_long: int, momentum_window: int) -> pd.DataFrame:
    result = stock_df.copy()
    result["ma_mid"] = result["close"].rolling(ma_mid).mean()
    result["ma_long"] = result["close"].rolling(ma_long).mean()
    result["ma_long_slope"] = result["ma_long"].pct_change(20)
    result["momentum_return"] = result["close"].pct_change(momentum_window)
    result["trend_strength"] = result["close"] / result["ma_long"] - 1.0

    result["trend_confirm"] = (
        (result["close"] > result["ma_long"])
        & (result["ma_mid"] > result["ma_long"])
        & (result["ma_long_slope"] > 0)
    ).astype(int)

    score_parts = normalize_zscore(result["trend_strength"]) + \
                  normalize_zscore(result["momentum_return"]) + \
                  normalize_zscore(result["ma_long_slope"])
    result["train_score"] = score_parts

    return result


def run_v3_frame(
    stock_df: pd.DataFrame,
    benchmark_filter_df: pd.DataFrame,
    ma_mid: int,
    ma_long: int,
    momentum_window: int,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> pd.DataFrame:
    """运行 v3 策略，返回完整结果 DataFrame（不含 metrics）。"""
    result = compute_v3_signals(stock_df, ma_mid, ma_long, momentum_window)

    bench_filter = benchmark_filter_df.rename(columns={"close": "benchmark_close"})
    bench_filter = bench_filter.set_index("date", drop=False)
    result = result.set_index("date", drop=False).sort_index()

    result["benchmark_close"] = bench_filter["benchmark_close"].reindex(result.index).ffill()
    result["benchmark_ma_short"] = bench_filter["benchmark_ma_short"].reindex(result.index).ffill()
    result["benchmark_ma_long"] = bench_filter["benchmark_ma_long"].reindex(result.index).ffill()
    result["market_filter"] = bench_filter["market_filter"].reindex(result.index).ffill().fillna(0).astype(int)

    result["final_signal"] = (
        (result["trend_confirm"] == 1) & (result["market_filter"] == 1)
    ).astype(int)
    result["stock_only_signal"] = result["trend_confirm"]

    result["position"] = result["final_signal"].shift(1, fill_value=0).astype(float)
    result["stock_only_position"] = result["stock_only_signal"].shift(1, fill_value=0).astype(float)
    result["stock_ret"] = result["close"].pct_change().fillna(0)

    # Strategy costs
    pos_change = result["position"].diff().fillna(0)
    buy_turnover = pos_change.clip(lower=0)
    sell_turnover = (-pos_change).clip(lower=0)
    result["cost"] = buy_turnover * (commission + slippage) + sell_turnover * (commission + sell_tax + slippage)
    result["strategy_ret"] = result["position"] * result["stock_ret"] - result["cost"]

    # Stock-only costs
    so_change = result["stock_only_position"].diff().fillna(0)
    so_buy = so_change.clip(lower=0)
    so_sell = (-so_change).clip(lower=0)
    result["stock_only_cost"] = so_buy * (commission + slippage) + so_sell * (commission + sell_tax + slippage)
    result["stock_only_ret"] = result["stock_only_position"] * result["stock_ret"] - result["stock_only_cost"]

    return result


def calc_train_score(metrics: dict) -> float:
    """训练期评分公式。"""
    return (
        metrics.get("annual_return", 0)
        + 0.25 * metrics.get("sharpe", 0)
        + 0.40 * metrics.get("excess_vs_stock_only_total_return", 0)
        + 0.20 * metrics.get("excess_vs_buy_hold_total_return", 0)
        + 0.80 * metrics.get("max_drawdown", 0)  # 负值，惩罚回撤
    )


def pass_train_filters(metrics: dict, args: argparse.Namespace) -> bool:
    """检查训练期质量门控。"""
    if pd.isna(metrics["annual_return"]) or pd.isna(metrics["max_drawdown"]) or pd.isna(metrics["sharpe"]):
        return False
    if metrics["days"] < args.min_train_rows:
        return False
    if metrics["trade_count"] < args.min_train_trades:
        return False
    if metrics["max_drawdown"] < args.max_train_drawdown:  # drawdown is negative
        return False
    if metrics["sharpe"] < args.min_train_sharpe:
        return False
    if metrics["annual_return"] < args.min_train_annual_return:
        return False

    # 波动率过滤
    if args.max_train_volatility > 0 and metrics["annual_volatility"] > args.max_train_volatility:
        return False

    # Calmar 过滤 (收益/回撤比)
    if args.min_train_calmar > 0:
        if metrics["max_drawdown"] == 0:
            return False
        calmar = metrics["annual_return"] / abs(metrics["max_drawdown"])
        if calmar < args.min_train_calmar:
            return False

    # 超额收益过滤
    if not args.allow_negative_train_excess and args.train_excess_mode != "none":
        if args.train_excess_mode == "stock_only":
            excess = metrics.get("excess_vs_stock_only_total_return", 0)
        else:
            excess = metrics.get("excess_vs_buy_hold_total_return", 0)
        if excess < args.min_train_excess_total_return:
            return False

    return True


def pass_train_filters_simple(
    metrics, min_train_rows, min_train_trades, max_train_drawdown,
    min_train_sharpe, min_train_annual_return,
    max_train_volatility, min_train_calmar,
    allow_negative_train_excess, train_excess_mode, min_train_excess_total_return,
):
    """纯函数版本，用于 worker 进程。"""
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
    if not allow_negative_train_excess and train_excess_mode != "none":
        if train_excess_mode == "stock_only":
            excess = metrics.get("excess_vs_stock_only_total_return", 0)
        else:
            excess = metrics.get("excess_vs_buy_hold_total_return", 0)
        if excess < min_train_excess_total_return:
            return False
    return True


def train_one_stock_all_years_v3(args_tuple):
    """处理一只股票跨所有测试年份的训练候选，供并行 worker 使用。"""
    (
        item_dict,
        test_years,
        train_start,
        end,
        stock_pairs,
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
        allow_negative_train_excess,
        train_excess_mode,
        min_train_excess_total_return,
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

        for ma_mid, ma_long, momentum_window in stock_pairs:
            for (bm_symbol, bm_ma), bm_data in benchmark_cache_data.items():
                required_rows = max(ma_long, momentum_window, int(bm_ma * 2.5)) + warmup_buffer
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
                    result = run_v3_frame(
                        train_df,
                        filter_train,
                        ma_mid,
                        ma_long,
                        momentum_window,
                        commission,
                        sell_tax,
                        slippage,
                    )
                    metrics = calc_metrics_from_returns(
                        result["strategy_ret"],
                        position=result["position"],
                        cash=cash,
                        stock_ret=result["stock_ret"],
                    )
                    _enrich_stock_only_metrics(metrics, result["stock_only_ret"])

                    if not pass_train_filters_simple(
                        metrics,
                        min_train_rows,
                        min_train_trades,
                        max_train_drawdown,
                        min_train_sharpe,
                        min_train_annual_return,
                        max_train_volatility,
                        min_train_calmar,
                        allow_negative_train_excess,
                        train_excess_mode,
                        min_train_excess_total_return,
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
                        "ma_mid": ma_mid,
                        "ma_long": ma_long,
                        "momentum_window": momentum_window,
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
                        "train_stock_only_total_return": metrics["stock_only_total_return"],
                        "train_excess_vs_stock_only": metrics["excess_vs_stock_only_total_return"],
                        "train_excess_vs_buy_hold": metrics["excess_vs_buy_hold_total_return"],
                        "train_market_filter_on_ratio": float(result["market_filter"].mean()),
                        "train_strategy_exposure_ratio": float(result["position"].mean()),
                    }
                except Exception:
                    continue

        if best_candidate is not None:
            results_by_year[year] = best_candidate

    return results_by_year


def build_catalog(args: argparse.Namespace) -> pd.DataFrame:
    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root
    catalog = ma.scan_qmt_export(export_root)
    if args.market != "ALL":
        catalog = catalog[catalog["market"] == args.market]
    if args.security_type != "ALL":
        catalog = catalog[catalog["security_type"] == args.security_type]
    if args.limit > 0:
        catalog = catalog.head(args.limit)
    return catalog


def get_test_years(args: argparse.Namespace) -> list[int]:
    first = args.first_test_year
    last = args.last_test_year
    if last <= 0:
        last = datetime.now().year
    return list(range(first, last + 1))


def load_symbol_data(csv_path: Path, args: argparse.Namespace) -> pd.DataFrame | None:
    try:
        return ma.load_qmt_price_csv(csv_path=csv_path, start=args.train_start, end=args.end)
    except RuntimeError:
        return None


def load_benchmark_cache(args: argparse.Namespace) -> dict:
    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root

    benchmark_list = parse_symbol_list(args.benchmark_list)
    benchmark_ma_list = parse_int_list(args.benchmark_ma_list)

    cache = {}
    for bm_symbol in benchmark_list:
        for bm_ma in benchmark_ma_list:
            bm_csv_path, _, _, _ = ma.find_csv_for_stock(bm_symbol, export_root)
            bm_df = ma.load_qmt_price_csv(bm_csv_path, args.train_start, args.end)
            filter_df = prepare_benchmark_regime(bm_df, bm_ma)
            cache[(bm_symbol, bm_ma)] = {
                "benchmark": bm_symbol,
                "benchmark_ma": bm_ma,
                "csv_path": str(bm_csv_path),
                "filter_df": filter_df,
            }
    return cache


def train_select_for_period(
    catalog: pd.DataFrame,
    data_cache: dict,
    benchmark_cache: dict,
    train_start_dt: pd.Timestamp,
    train_end_dt: pd.Timestamp,
    stock_pairs: list[tuple[int, int, int]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """单进程训练：对一个训练期选股。"""
    candidates = []

    for _, item in catalog.iterrows():
        symbol = item["symbol"]
        csv_path = item["csv_path"]

        if csv_path not in data_cache:
            continue
        df = data_cache[csv_path]
        train_df = df[(df["date"] >= train_start_dt) & (df["date"] <= train_end_dt)].copy()
        if len(train_df) < args.min_train_rows:
            continue

        for ma_mid, ma_long, momentum_window in stock_pairs:
            for (bm_symbol, bm_ma), bm_data in benchmark_cache.items():
                filter_df = bm_data["filter_df"]
                filter_train = filter_df[(filter_df["date"] >= train_start_dt) & (filter_df["date"] <= train_end_dt)]

                try:
                    result = run_v3_frame(
                        train_df, filter_train,
                        ma_mid, ma_long, momentum_window,
                        args.commission, args.sell_tax, args.slippage,
                    )
                    ret = result["strategy_ret"]
                    pos = result["position"]
                    metrics = calc_metrics_from_returns(ret, position=pos, stock_ret=result["stock_ret"])
                    _enrich_stock_only_metrics(metrics, result["stock_only_ret"])

                    if not pass_train_filters(metrics, args):
                        continue

                    score = calc_train_score(metrics)

                    candidates.append({
                        "symbol": symbol,
                        "csv_path": csv_path,
                        "ma_mid": ma_mid,
                        "ma_long": ma_long,
                        "momentum_window": momentum_window,
                        "benchmark": bm_symbol,
                        "benchmark_ma": bm_ma,
                        "train_score": score,
                        "train_annual_return": metrics["annual_return"],
                        "train_sharpe": metrics["sharpe"],
                        "train_max_drawdown": metrics["max_drawdown"],
                        "train_annual_volatility": metrics["annual_volatility"],
                        "train_excess_vs_stock_only": metrics["excess_vs_stock_only_total_return"],
                        "train_excess_vs_buy_hold": metrics["excess_vs_buy_hold_total_return"],
                        "train_days": metrics["days"],
                        "train_trade_count": metrics["trade_count"],
                    })
                except Exception:
                    continue

    if not candidates:
        return pd.DataFrame()

    cand_df = pd.DataFrame(candidates)
    cand_df = cand_df.sort_values(["train_score", "train_sharpe", "train_annual_return"], ascending=False)
    # 每只股票只保留最佳参数
    cand_df = cand_df.drop_duplicates(subset=["symbol"], keep="first")
    cand_df = cand_df.head(args.portfolio_size).copy()
    cand_df["selected_rank"] = range(1, len(cand_df) + 1)
    return cand_df


def test_selected_for_period(
    selected: pd.DataFrame,
    data_cache: dict,
    benchmark_cache: dict,
    test_start_dt: pd.Timestamp,
    test_end_dt: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对选中的候选进行样本外测试。"""
    detail_rows = []
    returns_dict = {}

    for _, cand in selected.iterrows():
        symbol = cand["symbol"]
        csv_path = cand["csv_path"]
        ma_mid = int(cand["ma_mid"])
        ma_long = int(cand["ma_long"])
        momentum_window = int(cand["momentum_window"])
        bm_symbol = cand["benchmark"]
        bm_ma = int(cand["benchmark_ma"])

        if csv_path not in data_cache:
            continue
        full_df = data_cache[csv_path]
        bm_data = benchmark_cache.get((bm_symbol, bm_ma))
        if bm_data is None:
            continue

        # 使用完整历史计算指标，然后切片到测试期
        full_result = run_v3_frame(
            full_df, bm_data["filter_df"],
            ma_mid, ma_long, momentum_window,
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
        _enrich_stock_only_metrics(test_metrics, test_result["stock_only_ret"])

        detail = {
            "symbol": symbol,
            "ma_mid": ma_mid,
            "ma_long": ma_long,
            "momentum_window": momentum_window,
            "benchmark": bm_symbol,
            "benchmark_ma": bm_ma,
            "selected_rank": cand.get("selected_rank", 0),
            "train_score": cand.get("train_score", np.nan),
            "train_annual_return": cand.get("train_annual_return", np.nan),
            "train_sharpe": cand.get("train_sharpe", np.nan),
            "train_max_drawdown": cand.get("train_max_drawdown", np.nan),
            "train_annual_volatility": cand.get("train_annual_volatility", np.nan),
            "train_excess_vs_stock_only": cand.get("train_excess_vs_stock_only", np.nan),
            "train_excess_vs_buy_hold": cand.get("train_excess_vs_buy_hold", np.nan),
            "test_total_return": test_metrics["total_return"],
            "test_annual_return": test_metrics["annual_return"],
            "test_sharpe": test_metrics["sharpe"],
            "test_max_drawdown": test_metrics["max_drawdown"],
            "test_annual_volatility": test_metrics["annual_volatility"],
            "test_excess_vs_stock_only": test_metrics["excess_vs_stock_only_total_return"],
            "test_excess_vs_buy_hold": test_metrics["excess_vs_buy_hold_total_return"],
            "test_days": test_metrics["days"],
            "test_trade_count": test_metrics["trade_count"],
        }
        detail_rows.append(detail)

        # 每日收益用于等权组合
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
    """等权组合指标。portfolio_ret 若已计算则直接传入，避免重复计算。"""
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


def build_tag(args: argparse.Namespace) -> str:
    ma_mid_list = parse_int_list(args.ma_mid_list)
    ma_long_list = parse_int_list(args.ma_long_list)
    mom_list = parse_int_list(args.momentum_window_list)
    bm_list = parse_symbol_list(args.benchmark_list)
    bm_ma_list = parse_int_list(args.benchmark_ma_list)
    last_test = args.last_test_year if args.last_test_year > 0 else "latest"
    limit_tag = f"limit{args.limit}" if args.limit > 0 else "limitALL"

    return (
        f"v3_mom_{args.market}_{args.security_type}"
        f"_ts{args.train_start}"
        f"_fy{args.first_test_year}-{last_test}"
        f"_mid{'_'.join(map(str, ma_mid_list))}"
        f"_long{'_'.join(map(str, ma_long_list))}"
        f"_mom{'_'.join(map(str, mom_list))}"
        f"_bm{'_'.join(safe_symbol_tag(b) for b in bm_list)}"
        f"_bma{'_'.join(map(str, bm_ma_list))}"
        f"_top{args.portfolio_size}"
        f"_{limit_tag}"
    )


def save_outputs(
    selected_all: pd.DataFrame,
    test_detail_all: pd.DataFrame,
    portfolio_daily_all: pd.DataFrame,
    portfolio_periods: list[dict],
    args: argparse.Namespace,
) -> None:
    tag = build_tag(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_path = OUTPUT_DIR / f"wf_ma_v3_stock_{tag}_selected_by_year.csv"
    detail_path = OUTPUT_DIR / f"wf_ma_v3_stock_{tag}_test_detail.csv"
    daily_path = OUTPUT_DIR / f"wf_ma_v3_stock_{tag}_portfolio_daily.csv"
    period_path = OUTPUT_DIR / f"wf_ma_v3_stock_{tag}_portfolio_period_summary.csv"
    report_path = OUTPUT_DIR / f"wf_ma_v3_stock_{tag}_report.txt"

    if not selected_all.empty:
        selected_all.to_csv(selected_path, encoding="utf-8-sig", index=False)
    if not test_detail_all.empty:
        test_detail_all.to_csv(detail_path, encoding="utf-8-sig", index=False)
    if not portfolio_daily_all.empty:
        portfolio_daily_all.to_csv(daily_path, encoding="utf-8-sig", index=False)
    if portfolio_periods:
        pd.DataFrame(portfolio_periods).to_csv(period_path, encoding="utf-8-sig", index=False)

    # 报告
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MA v3 Walk-Forward Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"train_start: {args.train_start}\n")
        f.write(f"first_test_year: {args.first_test_year}\n")
        f.write(f"last_test_year: {args.last_test_year}\n")
        f.write(f"portfolio_size: {args.portfolio_size}\n")
        f.write(f"ma_mid_list: {args.ma_mid_list}\n")
        f.write(f"ma_long_list: {args.ma_long_list}\n")
        f.write(f"momentum_window_list: {args.momentum_window_list}\n")
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

            # 整体指标
            valid_periods = [p for p in portfolio_periods if "total_return" in p and not np.isnan(p["total_return"])]
            if valid_periods:
                overall_equity = 1.0
                for p in valid_periods:
                    overall_equity *= (1.0 + p["total_return"])
                overall_return = overall_equity - 1.0
                years = len(valid_periods)
                overall_annual = (1.0 + overall_return) ** (1.0 / years) - 1.0 if years > 0 else 0
                f.write(f"\nOverall: total_return={overall_return:.4f}, annual_return={overall_annual:.4f}, years={years}\n")

    logger.info("结果已保存到：%s", OUTPUT_DIR)
    logger.info("  %s", selected_path)
    logger.info("  %s", detail_path)
    logger.info("  %s", daily_path)
    logger.info("  %s", period_path)
    logger.info("  %s", report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MA v3 walk-forward：趋势强度 / 动量 + 趋势确认")

    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT))
    parser.add_argument("--market", default="ALL")
    parser.add_argument("--security-type", default="stock")
    parser.add_argument("--train-start", default="20150101")
    parser.add_argument("--first-test-year", type=int, default=2021)
    parser.add_argument("--last-test-year", type=int, default=0)
    parser.add_argument("--end", default="")

    parser.add_argument("--ma-mid-list", default="20,60")
    parser.add_argument("--ma-long-list", default="120,250")
    parser.add_argument("--momentum-window-list", default="60,120")
    parser.add_argument("--benchmark-list", default=DEFAULT_BENCHMARK_LIST)
    parser.add_argument("--benchmark-ma-list", default="120,250")

    parser.add_argument("--warmup-buffer", type=int, default=10)
    parser.add_argument("--portfolio-size", type=int, default=20)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission", type=float, default=0.0001)
    parser.add_argument("--sell-tax", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0)

    # 训练筛选
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--min-train-trades", type=int, default=4)
    parser.add_argument("--max-train-drawdown", type=float, default=-0.55)
    parser.add_argument("--min-train-sharpe", type=float, default=0.2)
    parser.add_argument("--min-train-annual-return", type=float, default=0.02)
    parser.add_argument("--min-train-excess-total-return", type=float, default=0.0)
    parser.add_argument("--train-excess-mode", default="stock_only", choices=["stock_only", "buy_hold", "none"])
    parser.add_argument("--allow-negative-train-excess", action="store_true")
    parser.add_argument("--max-train-volatility", type=float, default=0.0, help="最大训练期年化波动率，0=不限制")
    parser.add_argument("--min-train-calmar", type=float, default=0.0, help="最小训练期 Calmar 比率，0=不限制")

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--output-name", default="wf_ma_v3")
    parser.add_argument("--workers", type=parse_workers, default=1, help="并行进程数，支持 'auto' 或正整数")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = datetime.now()

    stock_pairs = [
        (mid, long_, mom)
        for mid in parse_int_list(args.ma_mid_list)
        for long_ in parse_int_list(args.ma_long_list)
        for mom in parse_int_list(args.momentum_window_list)
        if mid < long_
    ]

    catalog = build_catalog(args)
    test_years = get_test_years(args)
    benchmark_cache = load_benchmark_cache(args)

    logger.info("股票数量: %d", len(catalog))
    logger.info("参数组合: %d", len(stock_pairs))
    logger.info("测试年份: %s", test_years)
    logger.info("基准组合: %d", len(benchmark_cache))

    data_cache = {}
    selected_all_list = []
    test_detail_all_list = []
    portfolio_daily_list = []
    portfolio_periods = []
    running_equity = args.cash

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        benchmark_cache_data = {
            k: {
                "benchmark": v["benchmark"],
                "benchmark_ma": v["benchmark_ma"],
                "csv_path": v["csv_path"],
                "filter_df": v["filter_df"].reset_index(drop=True),
            }
            for k, v in benchmark_cache.items()
        }

        tasks = []
        for _, item in catalog.iterrows():
            tasks.append((
                item.to_dict(),
                test_years,
                args.train_start,
                args.end,
                stock_pairs,
                benchmark_cache_data,
                args.commission,
                args.sell_tax,
                args.slippage,
                args.cash,
                args.warmup_buffer,
                args.min_train_rows,
                args.min_train_trades,
                args.max_train_drawdown,
                args.min_train_sharpe,
                args.min_train_annual_return,
                args.max_train_volatility,
                args.min_train_calmar,
                args.allow_negative_train_excess,
                args.train_excess_mode,
                args.min_train_excess_total_return,
            ))

        all_candidates_by_year = {year: [] for year in test_years}
        completed_stocks = 0

        logger.info("并行训练：%d 进程处理 %d 只股票，跨 %d 个年份...", args.workers, len(tasks), len(test_years))
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(train_one_stock_all_years_v3, task): task[0]["symbol"] for task in tasks}
            for future in as_completed(futures):
                completed_stocks += 1
                if completed_stocks % max(args.progress_every, 1) == 0:
                    logger.info("训练进度：%d/%d 只股票完成", completed_stocks, len(tasks))

                try:
                    stock_results = future.result()
                    for year, candidate in stock_results.items():
                        all_candidates_by_year[year].append(candidate)
                except Exception as exc:
                    logger.error("训练错误 %s: %s", futures[future], repr(exc))

        for test_year in test_years:
            logger.info("=" * 60)
            logger.info("测试年份: %d", test_year)
            test_start_dt = pd.Timestamp(f"{test_year}-01-01")
            test_end_dt = pd.Timestamp(f"{test_year}-12-31")
            if args.end:
                end_dt = parse_date_yyyymmdd(args.end)
                if test_end_dt > end_dt:
                    test_end_dt = end_dt

            candidates = all_candidates_by_year.get(test_year, [])
            if not candidates:
                logger.warning("%d: 无候选，跳过", test_year)
                portfolio_periods.append({"test_year": test_year, "portfolio_size_actual": 0})
                continue

            candidates_df = pd.DataFrame(candidates)
            candidates_df = candidates_df.sort_values(
                ["train_score", "train_sharpe", "train_annual_return"],
                ascending=False,
            )
            selected = candidates_df.drop_duplicates(subset=["symbol"], keep="first")
            selected = selected.head(args.portfolio_size).reset_index(drop=True)
            selected["selected_rank"] = range(1, len(selected) + 1)
            selected["test_year"] = test_year
            selected_all_list.append(selected)
            logger.info("%d: 训练候选 %d 条，选中 %d 只", test_year, len(candidates_df), len(selected))

            for csv_path in selected["csv_path"].unique():
                if csv_path not in data_cache:
                    data_cache[csv_path] = load_symbol_data(Path(csv_path), args)

            detail, returns_df = test_selected_for_period(
                selected, data_cache, benchmark_cache,
                test_start_dt, test_end_dt, args,
            )

            if not detail.empty:
                detail["test_year"] = test_year
                test_detail_all_list.append(detail)

            if not returns_df.empty:
                portfolio_ret = returns_df.mean(axis=1)
                equity = running_equity * (1.0 + portfolio_ret).cumprod()
                daily_df = pd.DataFrame({
                    "date": returns_df.index,
                    "portfolio_ret": portfolio_ret.values,
                    "equity": equity.values,
                    "test_year": test_year,
                })
                portfolio_daily_list.append(daily_df)
                period_metrics = calc_portfolio_period_metrics(returns_df, test_year, args.cash, portfolio_ret)
                portfolio_periods.append(period_metrics)
                running_equity = equity.iloc[-1]
                logger.info("%d: return=%.4f, sharpe=%.4f",
                            test_year, period_metrics.get('total_return', np.nan),
                            period_metrics.get('sharpe', np.nan))
    else:
        # 单进程模式保留预加载，便于小样本调试。
        skipped_load = 0
        for _, item in catalog.iterrows():
            csv_path = item["csv_path"]
            if csv_path not in data_cache:
                df = load_symbol_data(Path(csv_path), args)
                if df is not None and not df.empty:
                    data_cache[csv_path] = df
                else:
                    skipped_load += 1

        logger.info("已加载 %d 只股票数据 (跳过 %d 只数据不足)", len(data_cache), skipped_load)

        for test_year in test_years:
            logger.info("=" * 60)
            logger.info("测试年份: %d", test_year)
            train_start_dt = parse_date_yyyymmdd(args.train_start)
            train_end_dt = pd.Timestamp(f"{test_year - 1}-12-31")
            test_start_dt = pd.Timestamp(f"{test_year}-01-01")
            test_end_dt = pd.Timestamp(f"{test_year}-12-31")
            if args.end:
                end_dt = parse_date_yyyymmdd(args.end)
                if test_end_dt > end_dt:
                    test_end_dt = end_dt

            selected = train_select_for_period(
                catalog, data_cache, benchmark_cache,
                train_start_dt, train_end_dt, stock_pairs, args,
            )

            if selected.empty:
                logger.warning("%d: 无候选，跳过", test_year)
                portfolio_periods.append({"test_year": test_year, "portfolio_size_actual": 0})
                continue

            selected["test_year"] = test_year
            selected_all_list.append(selected)
            logger.info("%d: 选中 %d 只", test_year, len(selected))

            detail, returns_df = test_selected_for_period(
                selected, data_cache, benchmark_cache,
                test_start_dt, test_end_dt, args,
            )

            if not detail.empty:
                detail["test_year"] = test_year
                test_detail_all_list.append(detail)

            if not returns_df.empty:
                portfolio_ret = returns_df.mean(axis=1)
                equity = running_equity * (1.0 + portfolio_ret).cumprod()
                daily_df = pd.DataFrame({
                    "date": returns_df.index,
                    "portfolio_ret": portfolio_ret.values,
                    "equity": equity.values,
                    "test_year": test_year,
                })
                portfolio_daily_list.append(daily_df)
                period_metrics = calc_portfolio_period_metrics(returns_df, test_year, args.cash, portfolio_ret)
                portfolio_periods.append(period_metrics)
                running_equity = equity.iloc[-1]
                logger.info("%d: return=%.4f, sharpe=%.4f",
                            test_year, period_metrics.get('total_return', np.nan),
                            period_metrics.get('sharpe', np.nan))

    # 汇总
    selected_all = pd.concat(selected_all_list, ignore_index=True) if selected_all_list else pd.DataFrame()
    test_detail_all = pd.concat(test_detail_all_list, ignore_index=True) if test_detail_all_list else pd.DataFrame()
    portfolio_daily_all = pd.concat(portfolio_daily_list, ignore_index=True) if portfolio_daily_list else pd.DataFrame()

    save_outputs(selected_all, test_detail_all, portfolio_daily_all, portfolio_periods, args)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("总耗时: %.1fs", elapsed)


if __name__ == "__main__":
    setup_cli_logging()
    main()
