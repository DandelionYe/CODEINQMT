# -*- coding: utf-8 -*-
"""
validate_ma_market_filter_candidates.py

MA v2 walk-forward / 滚动样本外验证：
个股均线交叉 + 大盘趋势过滤

核心思想：
每一个测试年度，只使用该年度之前的数据进行训练、选股、选参数、选过滤基准；
然后在该年度进行样本外测试。

策略逻辑：
1. 个股信号：
   stock_fast_ma > stock_slow_ma

2. 大盘过滤：
   benchmark_fast_ma > benchmark_slow_ma

3. 最终信号：
   final_signal = stock_signal AND market_filter

4. 执行规则：
   当日收盘生成信号，次日持仓，避免未来函数。

默认测试：
- 个股参数：5/60, 5/120, 5/250, 10/60, 10/120, 10/250, 20/60, 20/120, 20/250
- 大盘过滤基准：000300.SH, 000905.SH, 000852.SH
- 大盘过滤参数：20/120
- 每年训练期选 Top 20
- 下一年样本外等权测试

运行示例：

1. 小规模测试，只跑深市前 300 个标的，只测试 2024 年：
python scripts\\validate_ma_market_filter_candidates.py --market SZ --limit 300 --first-test-year 2024 --last-test-year 2024 --portfolio-size 10

2. 深市 v2 walk-forward：
python scripts\\validate_ma_market_filter_candidates.py --market SZ --security-type stock --first-test-year 2021 --portfolio-size 20

3. 沪市 v2 walk-forward：
python scripts\\validate_ma_market_filter_candidates.py --market SH --security-type stock --first-test-year 2021 --portfolio-size 20

4. 全市场 v2 walk-forward：
python scripts\\validate_ma_market_filter_candidates.py --market ALL --security-type stock --first-test-year 2021 --portfolio-size 20

5. 指定过滤基准：
python scripts\\validate_ma_market_filter_candidates.py --market SZ --benchmark-list 000905.SH --first-test-year 2021 --portfolio-size 20

6. 放宽训练筛选条件：
python scripts\\validate_ma_market_filter_candidates.py --market SZ --min-train-rows 700 --min-train-trades 3 --max-train-drawdown -0.60 --min-train-sharpe 0.2 --min-train-annual-return 0.02
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


OUTPUT_DIR = PROJECT_ROOT / "backtests" / "walk_forward_ma_market_filter_csv"
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


def prepare_benchmark_filter(
    benchmark_df: pd.DataFrame,
    benchmark_fast: int,
    benchmark_slow: int,
) -> pd.DataFrame:
    bench = benchmark_df.copy()
    bench = bench.set_index("date", drop=False).sort_index()

    bench["benchmark_fast_ma"] = bench["close"].rolling(benchmark_fast).mean()
    bench["benchmark_slow_ma"] = bench["close"].rolling(benchmark_slow).mean()

    bench["market_filter"] = np.where(
        bench["benchmark_fast_ma"] > bench["benchmark_slow_ma"],
        1,
        0,
    )

    return bench[["date", "close", "benchmark_fast_ma", "benchmark_slow_ma", "market_filter"]].copy()


def run_ma_market_filter_frame(
    stock_df: pd.DataFrame,
    benchmark_filter_df: pd.DataFrame,
    fast: int,
    slow: int,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> pd.DataFrame:
    result = stock_df.copy()
    result = result.set_index("date", drop=False).sort_index()

    result["fast_ma"] = result["close"].rolling(fast).mean()
    result["slow_ma"] = result["close"].rolling(slow).mean()

    result["stock_signal"] = np.where(result["fast_ma"] > result["slow_ma"], 1, 0)

    bench = benchmark_filter_df.copy()
    bench = bench.rename(columns={"close": "benchmark_close"})
    bench = bench.set_index("date", drop=False).sort_index()

    # 使用股票交易日为主索引，基准过滤信号对齐到股票日期。
    result["benchmark_close"] = bench["benchmark_close"].reindex(result.index).ffill()
    result["benchmark_fast_ma"] = bench["benchmark_fast_ma"].reindex(result.index).ffill()
    result["benchmark_slow_ma"] = bench["benchmark_slow_ma"].reindex(result.index).ffill()
    result["market_filter"] = bench["market_filter"].reindex(result.index).ffill().fillna(0).astype(int)

    # v2：个股趋势 + 大盘过滤
    result["final_signal"] = (
        (result["stock_signal"] == 1)
        & (result["market_filter"] == 1)
    ).astype(int)

    # v1 对照：只看个股趋势
    result["stock_only_signal"] = result["stock_signal"]

    # 次日持仓，避免未来函数
    result["position"] = result["final_signal"].shift(1).fillna(0)
    result["stock_only_position"] = result["stock_only_signal"].shift(1).fillna(0)

    result["stock_ret"] = result["close"].pct_change().fillna(0)

    # v2 成本
    pos_change = result["position"].diff().fillna(result["position"])
    buy_turnover = pos_change.clip(lower=0)
    sell_turnover = (-pos_change).clip(lower=0)

    result["cost"] = (
        buy_turnover * (commission + slippage)
        + sell_turnover * (commission + sell_tax + slippage)
    )

    result["strategy_ret"] = result["position"] * result["stock_ret"] - result["cost"]

    # v1 对照成本
    stock_only_pos_change = result["stock_only_position"].diff().fillna(result["stock_only_position"])
    stock_only_buy = stock_only_pos_change.clip(lower=0)
    stock_only_sell = (-stock_only_pos_change).clip(lower=0)

    result["stock_only_cost"] = (
        stock_only_buy * (commission + slippage)
        + stock_only_sell * (commission + sell_tax + slippage)
    )

    result["stock_only_ret"] = result["stock_only_position"] * result["stock_ret"] - result["stock_only_cost"]

    return result


def calc_train_score(metrics: dict) -> float:
    sharpe = 0.0 if pd.isna(metrics["sharpe"]) else metrics["sharpe"]
    excess_stock_only = 0.0 if pd.isna(metrics["excess_vs_stock_only_total_return"]) else metrics["excess_vs_stock_only_total_return"]
    excess_buy_hold = 0.0 if pd.isna(metrics["excess_vs_buy_hold_total_return"]) else metrics["excess_vs_buy_hold_total_return"]

    return float(
        metrics["annual_return"]
        + 0.25 * sharpe
        + 0.40 * excess_stock_only
        + 0.20 * excess_buy_hold
        + 0.80 * metrics["max_drawdown"]
    )


def pass_train_filters(metrics: dict, args: argparse.Namespace) -> bool:
    if pd.isna(metrics["annual_return"]) or pd.isna(metrics["max_drawdown"]) or pd.isna(metrics["sharpe"]):
        return False

    if metrics["days"] < args.min_train_rows:
        return False

    if metrics["trade_count"] < args.min_train_trades:
        return False

    if metrics["max_drawdown"] < args.max_train_drawdown:
        return False

    if metrics["sharpe"] < args.min_train_sharpe:
        return False

    if metrics["annual_return"] < args.min_train_annual_return:
        return False

    if not args.allow_negative_train_excess:
        if args.train_excess_mode == "stock_only":
            value = metrics["excess_vs_stock_only_total_return"]
        elif args.train_excess_mode == "buy_hold":
            value = metrics["excess_vs_buy_hold_total_return"]
        else:
            value = 0.0

        if args.train_excess_mode != "none":
            if pd.isna(value) or value <= args.min_train_excess_total_return:
                return False

    return True


def pass_train_filters_simple(
    metrics: dict,
    min_train_rows: int,
    min_train_trades: int,
    max_train_drawdown: float,
    min_train_sharpe: float,
    min_train_annual_return: float,
    min_train_excess_total_return: float,
    allow_negative_train_excess: bool,
    train_excess_mode: str,
) -> bool:
    """训练期筛选条件（纯函数版本，供 worker 使用）。"""
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

    if not allow_negative_train_excess:
        if train_excess_mode == "stock_only":
            value = metrics["excess_vs_stock_only_total_return"]
        elif train_excess_mode == "buy_hold":
            value = metrics["excess_vs_buy_hold_total_return"]
        else:
            value = 0.0

        if train_excess_mode != "none":
            if pd.isna(value) or value <= min_train_excess_total_return:
                return False

    return True


def train_one_stock_all_years_v2(args_tuple):
    """
    处理一只股票（含大盘过滤）：加载数据一次，对所有测试年份运行训练阶段。
    供 ProcessPoolExecutor 使用的 worker 函数。
    """
    (item_dict, test_years, train_start, end, stock_pairs, benchmark_cache_data,
     commission, sell_tax, slippage, cash, warmup_buffer,
     min_train_rows, min_train_trades, max_train_drawdown,
     min_train_sharpe, min_train_annual_return,
     min_train_excess_total_return, allow_negative_train_excess,
     train_excess_mode) = args_tuple

    symbol = item_dict["symbol"]
    market = item_dict["market"]
    security_type = item_dict["security_type"]
    csv_path = Path(item_dict["csv_path"])

    try:
        full_df = ma.load_qmt_price_csv(csv_path, train_start, end)
    except Exception:
        return {}

    results_by_year = {}

    for year in test_years:
        train_end_dt = pd.Timestamp(year=year - 1, month=12, day=31)
        train_start_dt = pd.to_datetime(train_start)

        if train_end_dt < train_start_dt:
            continue

        train_df = full_df[(full_df.index >= train_start_dt) & (full_df.index <= train_end_dt)]

        if train_df.empty:
            continue

        best_candidate = None
        best_score = -np.inf

        for fast, slow in stock_pairs:
            for bench_key, bench_filter_df, bench_csv_path_str, bench_symbol in benchmark_cache_data:
                _, benchmark_fast, benchmark_slow = bench_key

                required_rows = max(slow, benchmark_slow) + warmup_buffer
                if len(train_df) < required_rows:
                    continue

                train_benchmark_filter = bench_filter_df[
                    (bench_filter_df["date"] >= train_start_dt)
                    & (bench_filter_df["date"] <= train_end_dt)
                ].copy()

                if train_benchmark_filter.empty:
                    continue

                result = run_ma_market_filter_frame(
                    stock_df=train_df,
                    benchmark_filter_df=train_benchmark_filter,
                    fast=fast,
                    slow=slow,
                    commission=commission,
                    sell_tax=sell_tax,
                    slippage=slippage,
                )

                metrics = calc_metrics_from_returns(
                    result["strategy_ret"],
                    position=result["position"],
                    cash=cash,
                    stock_ret=result["stock_ret"],
                )
                _enrich_stock_only_metrics(metrics, result["stock_only_ret"])

                if not pass_train_filters_simple(
                    metrics, min_train_rows, min_train_trades, max_train_drawdown,
                    min_train_sharpe, min_train_annual_return,
                    min_train_excess_total_return, allow_negative_train_excess,
                    train_excess_mode,
                ):
                    continue

                train_score = calc_train_score(metrics)

                if train_score > best_score:
                    best_score = train_score
                    best_candidate = {
                        "symbol": symbol,
                        "market": market,
                        "security_type": security_type,
                        "csv_path": str(csv_path),
                        "fast": fast,
                        "slow": slow,
                        "benchmark": bench_symbol,
                        "benchmark_fast": benchmark_fast,
                        "benchmark_slow": benchmark_slow,
                        "benchmark_csv_path": bench_csv_path_str,
                        "train_start": str(train_start_dt.date()),
                        "train_end": str(train_end_dt.date()),
                        "train_rows": len(train_df),
                        "train_total_return": metrics["total_return"],
                        "train_annual_return": metrics["annual_return"],
                        "train_annual_volatility": metrics["annual_volatility"],
                        "train_max_drawdown": metrics["max_drawdown"],
                        "train_sharpe": metrics["sharpe"],
                        "train_trade_count": metrics["trade_count"],
                        "train_buy_hold_total_return": metrics["buy_hold_total_return"],
                        "train_stock_only_total_return": metrics["stock_only_total_return"],
                        "train_excess_vs_buy_hold_total_return": metrics["excess_vs_buy_hold_total_return"],
                        "train_excess_vs_stock_only_total_return": metrics["excess_vs_stock_only_total_return"],
                        "train_score": train_score,
                        "train_market_filter_on_ratio": float(result["market_filter"].mean()),
                        "train_strategy_exposure_ratio": float(result["position"].mean()),
                    }

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

    catalog = catalog.sort_values(["market", "symbol"]).reset_index(drop=True)

    if args.limit and args.limit > 0:
        catalog = catalog.head(args.limit)

    if catalog.empty:
        raise RuntimeError("catalog 为空，请检查 market / security-type / limit 参数。")

    return catalog


def get_test_years(args: argparse.Namespace) -> list[int]:
    first_year = args.first_test_year

    if args.last_test_year:
        last_year = args.last_test_year
    elif args.end:
        last_year = parse_date_yyyymmdd(args.end).year
    else:
        last_year = datetime.today().year

    if first_year > last_year:
        raise ValueError(f"first-test-year 不能大于 last-test-year：{first_year} > {last_year}")

    return list(range(first_year, last_year + 1))


def load_symbol_data(csv_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    return ma.load_qmt_price_csv(csv_path, args.train_start, args.end)


def load_benchmark_cache(args: argparse.Namespace) -> dict[tuple[str, int, int], dict]:
    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root

    benchmark_list = parse_symbol_list(args.benchmark_list)
    benchmark_fast_list = parse_int_list(args.benchmark_fast_list)
    benchmark_slow_list = parse_int_list(args.benchmark_slow_list)

    benchmark_pairs = [
        (f, s)
        for f in benchmark_fast_list
        for s in benchmark_slow_list
        if f < s
    ]

    if not benchmark_pairs:
        raise ValueError("没有有效基准均线组合。要求 benchmark_fast < benchmark_slow。")

    cache: dict[tuple[str, int, int], dict] = {}

    for benchmark in benchmark_list:
        csv_path, symbol, _, _ = ma.find_csv_for_stock(
            stock=benchmark,
            export_root=export_root,
        )

        df = ma.load_qmt_price_csv(
            csv_path=csv_path,
            start=args.train_start,
            end=args.end,
        )

        for bf, bs in benchmark_pairs:
            filter_df = prepare_benchmark_filter(
                benchmark_df=df,
                benchmark_fast=bf,
                benchmark_slow=bs,
            )

            cache[(symbol, bf, bs)] = {
                "benchmark": symbol,
                "benchmark_fast": bf,
                "benchmark_slow": bs,
                "csv_path": str(csv_path),
                "filter_df": filter_df,
            }

    return cache


def train_select_for_period(
    catalog: pd.DataFrame,
    data_cache: dict[str, pd.DataFrame],
    benchmark_cache: dict[tuple[str, int, int], dict],
    train_start_dt: pd.Timestamp,
    train_end_dt: pd.Timestamp,
    stock_pairs: list[tuple[int, int]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []

    for idx, item in catalog.iterrows():
        symbol = item["symbol"]
        market = item["market"]
        security_type = item["security_type"]
        csv_path = Path(item["csv_path"])

        try:
            if symbol not in data_cache:
                data_cache[symbol] = load_symbol_data(csv_path, args)

            full_df = data_cache[symbol]
            train_df = full_df[(full_df.index >= train_start_dt) & (full_df.index <= train_end_dt)]

            if train_df.empty:
                continue

            for fast, slow in stock_pairs:
                for (_, _, _), bench_item in benchmark_cache.items():
                    benchmark_symbol = bench_item["benchmark"]
                    benchmark_fast = bench_item["benchmark_fast"]
                    benchmark_slow = bench_item["benchmark_slow"]
                    benchmark_filter_df = bench_item["filter_df"]

                    required_rows = max(slow, benchmark_slow) + args.warmup_buffer
                    if len(train_df) < required_rows:
                        continue

                    # 训练期只取 train_end 以前的基准信号，避免年度之外数据参与统计。
                    train_benchmark_filter = benchmark_filter_df[
                        (benchmark_filter_df["date"] >= train_start_dt)
                        & (benchmark_filter_df["date"] <= train_end_dt)
                    ].copy()

                    if train_benchmark_filter.empty:
                        continue

                    result = run_ma_market_filter_frame(
                        stock_df=train_df,
                        benchmark_filter_df=train_benchmark_filter,
                        fast=fast,
                        slow=slow,
                        commission=args.commission,
                        sell_tax=args.sell_tax,
                        slippage=args.slippage,
                    )

                    metrics = calc_metrics_from_returns(
                        result["strategy_ret"],
                        position=result["position"],
                        cash=args.cash,
                        stock_ret=result["stock_ret"],
                    )
                    _enrich_stock_only_metrics(metrics, result["stock_only_ret"])

                    if not pass_train_filters(metrics, args):
                        continue

                    train_score = calc_train_score(metrics)

                    rows.append(
                        {
                            "symbol": symbol,
                            "market": market,
                            "security_type": security_type,
                            "csv_path": str(csv_path),
                            "fast": fast,
                            "slow": slow,
                            "benchmark": benchmark_symbol,
                            "benchmark_fast": benchmark_fast,
                            "benchmark_slow": benchmark_slow,
                            "benchmark_csv_path": bench_item["csv_path"],
                            "train_start": str(train_start_dt.date()),
                            "train_end": str(train_end_dt.date()),
                            "train_rows": len(train_df),
                            "train_total_return": metrics["total_return"],
                            "train_annual_return": metrics["annual_return"],
                            "train_annual_volatility": metrics["annual_volatility"],
                            "train_max_drawdown": metrics["max_drawdown"],
                            "train_sharpe": metrics["sharpe"],
                            "train_trade_count": metrics["trade_count"],
                            "train_buy_hold_total_return": metrics["buy_hold_total_return"],
                            "train_stock_only_total_return": metrics["stock_only_total_return"],
                            "train_excess_vs_buy_hold_total_return": metrics["excess_vs_buy_hold_total_return"],
                            "train_excess_vs_stock_only_total_return": metrics["excess_vs_stock_only_total_return"],
                            "train_score": train_score,
                            "train_market_filter_on_ratio": float(result["market_filter"].mean()),
                            "train_strategy_exposure_ratio": float(result["position"].mean()),
                        }
                    )

        except Exception as exc:
            logger.warning("训练跳过 %s: %s", symbol, repr(exc))
            continue

        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            logger.info("训练进度：%d/%d", idx + 1, len(catalog))

    if not rows:
        return pd.DataFrame()

    candidates = pd.DataFrame(rows)

    candidates = candidates.sort_values(
        by=["train_score", "train_sharpe", "train_annual_return"],
        ascending=[False, False, False],
    )

    # 每只股票只保留训练期最优：包括个股参数 + 过滤基准 + 过滤参数
    best_per_symbol = candidates.drop_duplicates(subset=["symbol"], keep="first")

    selected = best_per_symbol.head(args.portfolio_size).reset_index(drop=True)
    selected["selected_rank"] = range(1, len(selected) + 1)

    return selected


def test_selected_for_period(
    selected: pd.DataFrame,
    data_cache: dict[str, pd.DataFrame],
    benchmark_cache: dict[tuple[str, int, int], dict],
    test_start_dt: pd.Timestamp,
    test_end_dt: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    return_series = {}

    for _, item in selected.iterrows():
        symbol = item["symbol"]
        full_df = data_cache.get(symbol)

        if full_df is None or full_df.empty:
            continue

        fast = int(item["fast"])
        slow = int(item["slow"])
        benchmark = str(item["benchmark"])
        benchmark_fast = int(item["benchmark_fast"])
        benchmark_slow = int(item["benchmark_slow"])

        bench_key = (benchmark, benchmark_fast, benchmark_slow)
        if bench_key not in benchmark_cache:
            continue

        benchmark_filter_df = benchmark_cache[bench_key]["filter_df"]

        # 测试时不能只传测试期数据，否则测试期初均线没有足够历史。
        # 正确做法：使用 test_end 之前的全部历史计算指标，然后只截取测试期收益。
        calc_df = full_df[full_df.index <= test_end_dt].copy()
        calc_benchmark_filter = benchmark_filter_df[benchmark_filter_df["date"] <= test_end_dt].copy()

        if len(calc_df) < max(slow, benchmark_slow) + args.warmup_buffer:
            continue

        result = run_ma_market_filter_frame(
            stock_df=calc_df,
            benchmark_filter_df=calc_benchmark_filter,
            fast=fast,
            slow=slow,
            commission=args.commission,
            sell_tax=args.sell_tax,
            slippage=args.slippage,
        )

        test_result = result[
            (result.index >= test_start_dt)
            & (result.index <= test_end_dt)
        ].copy()

        if test_result.empty:
            continue

        metrics = calc_metrics_from_returns(
            test_result["strategy_ret"],
            position=test_result["position"],
            cash=args.cash,
            stock_ret=test_result["stock_ret"],
        )
        _enrich_stock_only_metrics(metrics, test_result["stock_only_ret"])

        return_series[symbol] = test_result["strategy_ret"].rename(symbol)

        detail_rows.append(
            {
                "test_year": test_start_dt.year,
                "symbol": symbol,
                "market": item["market"],
                "security_type": item["security_type"],
                "selected_rank": item["selected_rank"],
                "fast": fast,
                "slow": slow,
                "benchmark": benchmark,
                "benchmark_fast": benchmark_fast,
                "benchmark_slow": benchmark_slow,
                "train_start": item["train_start"],
                "train_end": item["train_end"],
                "train_rows": item["train_rows"],
                "train_total_return": item["train_total_return"],
                "train_annual_return": item["train_annual_return"],
                "train_max_drawdown": item["train_max_drawdown"],
                "train_sharpe": item["train_sharpe"],
                "train_trade_count": item["train_trade_count"],
                "train_buy_hold_total_return": item["train_buy_hold_total_return"],
                "train_stock_only_total_return": item["train_stock_only_total_return"],
                "train_excess_vs_buy_hold_total_return": item["train_excess_vs_buy_hold_total_return"],
                "train_excess_vs_stock_only_total_return": item["train_excess_vs_stock_only_total_return"],
                "train_score": item["train_score"],
                "train_market_filter_on_ratio": item["train_market_filter_on_ratio"],
                "train_strategy_exposure_ratio": item["train_strategy_exposure_ratio"],
                "test_start": str(test_start_dt.date()),
                "test_end": str(test_result.index.max().date()),
                "test_rows": len(test_result),
                "test_total_return": metrics["total_return"],
                "test_annual_return": metrics["annual_return"],
                "test_annual_volatility": metrics["annual_volatility"],
                "test_max_drawdown": metrics["max_drawdown"],
                "test_sharpe": metrics["sharpe"],
                "test_trade_count": metrics["trade_count"],
                "test_buy_hold_total_return": metrics["buy_hold_total_return"],
                "test_stock_only_total_return": metrics["stock_only_total_return"],
                "test_excess_vs_buy_hold_total_return": metrics["excess_vs_buy_hold_total_return"],
                "test_excess_vs_stock_only_total_return": metrics["excess_vs_stock_only_total_return"],
                "test_market_filter_on_ratio": float(test_result["market_filter"].mean()),
                "test_strategy_exposure_ratio": float(test_result["position"].mean()),
            }
        )

    detail = pd.DataFrame(detail_rows)

    if return_series:
        returns_df = pd.concat(return_series.values(), axis=1).sort_index()
        returns_df = returns_df.fillna(0.0)
    else:
        returns_df = pd.DataFrame()

    return detail, returns_df


def calc_portfolio_period_metrics(
    returns_df: pd.DataFrame,
    test_year: int,
    cash: float,
) -> dict:
    if returns_df.empty:
        return {
            "test_year": test_year,
            "portfolio_size_actual": 0,
            "period_start": "",
            "period_end": "",
            "days": 0,
            "portfolio_total_return": np.nan,
            "portfolio_annual_return": np.nan,
            "portfolio_annual_volatility": np.nan,
            "portfolio_max_drawdown": np.nan,
            "portfolio_sharpe": np.nan,
            "portfolio_final_equity": cash,
        }

    portfolio_ret = returns_df.mean(axis=1).sort_index()

    metrics = calc_metrics_from_returns(
        portfolio_ret,
        position=None,
        cash=cash,
    )
    metrics.setdefault("buy_hold_total_return", np.nan)
    metrics.setdefault("excess_vs_buy_hold_total_return", np.nan)
    metrics.setdefault("stock_only_total_return", np.nan)
    metrics.setdefault("excess_vs_stock_only_total_return", np.nan)

    return {
        "test_year": test_year,
        "portfolio_size_actual": returns_df.shape[1],
        "period_start": str(portfolio_ret.index.min().date()),
        "period_end": str(portfolio_ret.index.max().date()),
        "days": metrics["days"],
        "portfolio_total_return": metrics["total_return"],
        "portfolio_annual_return": metrics["annual_return"],
        "portfolio_annual_volatility": metrics["annual_volatility"],
        "portfolio_max_drawdown": metrics["max_drawdown"],
        "portfolio_sharpe": metrics["sharpe"],
        "portfolio_final_equity": metrics["final_equity"],
    }


def build_tag(args: argparse.Namespace) -> str:
    fast_tag = "f" + "-".join(map(str, parse_int_list(args.fast_list)))
    slow_tag = "s" + "-".join(map(str, parse_int_list(args.slow_list)))

    benchmark_tag = "b" + "-".join(safe_symbol_tag(x) for x in parse_symbol_list(args.benchmark_list))
    benchmark_fast_tag = "bf" + "-".join(map(str, parse_int_list(args.benchmark_fast_list)))
    benchmark_slow_tag = "bs" + "-".join(map(str, parse_int_list(args.benchmark_slow_list)))

    return (
        f"{args.output_name}_"
        f"{args.security_type}_{args.market}_"
        f"train{args.train_start}_"
        f"test{args.first_test_year}-{args.last_test_year or 'latest'}_"
        f"{fast_tag}_{slow_tag}_"
        f"{benchmark_tag}_{benchmark_fast_tag}_{benchmark_slow_tag}_"
        f"top{args.portfolio_size}"
    )


def save_outputs(
    selected_all: pd.DataFrame,
    test_detail_all: pd.DataFrame,
    portfolio_daily_all: pd.DataFrame,
    portfolio_periods: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    tag = build_tag(args)

    selected_path = OUTPUT_DIR / f"{tag}_selected_by_year.csv"
    test_detail_path = OUTPUT_DIR / f"{tag}_test_detail.csv"
    portfolio_daily_path = OUTPUT_DIR / f"{tag}_portfolio_daily.csv"
    portfolio_period_path = OUTPUT_DIR / f"{tag}_portfolio_period_summary.csv"
    report_path = OUTPUT_DIR / f"{tag}_report.txt"

    selected_all.to_csv(selected_path, index=False, encoding="utf-8-sig")
    test_detail_all.to_csv(test_detail_path, index=False, encoding="utf-8-sig")
    portfolio_daily_all.to_csv(portfolio_daily_path, index=False, encoding="utf-8-sig")
    portfolio_periods.to_csv(portfolio_period_path, index=False, encoding="utf-8-sig")

    if not portfolio_daily_all.empty and "portfolio_ret" in portfolio_daily_all.columns:
        ret = portfolio_daily_all.set_index("date")["portfolio_ret"].astype(float)
        overall_metrics = calc_metrics_from_returns(
            ret,
            position=None,
            cash=args.cash,
        )
        overall_metrics.setdefault("buy_hold_total_return", np.nan)
        overall_metrics.setdefault("excess_vs_buy_hold_total_return", np.nan)
        overall_metrics.setdefault("stock_only_total_return", np.nan)
        overall_metrics.setdefault("excess_vs_stock_only_total_return", np.nan)
    else:
        overall_metrics = {}

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MA v2 walk-forward 验证报告：个股均线交叉 + 大盘趋势过滤\n")
        f.write("=" * 80 + "\n\n")

        f.write("参数设置：\n")
        f.write(f"market: {args.market}\n")
        f.write(f"security_type: {args.security_type}\n")
        f.write(f"train_start: {args.train_start}\n")
        f.write(f"first_test_year: {args.first_test_year}\n")
        f.write(f"last_test_year: {args.last_test_year or 'latest'}\n")
        f.write(f"fast_list: {args.fast_list}\n")
        f.write(f"slow_list: {args.slow_list}\n")
        f.write(f"benchmark_list: {args.benchmark_list}\n")
        f.write(f"benchmark_fast_list: {args.benchmark_fast_list}\n")
        f.write(f"benchmark_slow_list: {args.benchmark_slow_list}\n")
        f.write(f"portfolio_size: {args.portfolio_size}\n")
        f.write(f"min_train_rows: {args.min_train_rows}\n")
        f.write(f"min_train_trades: {args.min_train_trades}\n")
        f.write(f"max_train_drawdown: {args.max_train_drawdown}\n")
        f.write(f"min_train_sharpe: {args.min_train_sharpe}\n")
        f.write(f"min_train_annual_return: {args.min_train_annual_return}\n")
        f.write(f"train_excess_mode: {args.train_excess_mode}\n")
        f.write(f"allow_negative_train_excess: {args.allow_negative_train_excess}\n\n")

        f.write("输出文件：\n")
        f.write(f"selected_by_year: {selected_path}\n")
        f.write(f"test_detail: {test_detail_path}\n")
        f.write(f"portfolio_daily: {portfolio_daily_path}\n")
        f.write(f"portfolio_period_summary: {portfolio_period_path}\n\n")

        f.write("数量统计：\n")
        f.write(f"selected rows: {len(selected_all)}\n")
        f.write(f"test detail rows: {len(test_detail_all)}\n")
        f.write(f"portfolio daily rows: {len(portfolio_daily_all)}\n")
        f.write(f"portfolio period rows: {len(portfolio_periods)}\n\n")

        if overall_metrics:
            f.write("组合整体样本外表现：\n")
            f.write(f"total_return: {overall_metrics['total_return']:.8f}\n")
            f.write(f"annual_return: {overall_metrics['annual_return']:.8f}\n")
            f.write(f"annual_volatility: {overall_metrics['annual_volatility']:.8f}\n")
            f.write(f"max_drawdown: {overall_metrics['max_drawdown']:.8f}\n")
            f.write(f"sharpe: {overall_metrics['sharpe']:.8f}\n")
            f.write(f"days: {overall_metrics['days']}\n")

    logger.info("MA v2 walk-forward 验证完成。")
    logger.info("年度入选组合：%s", selected_path)
    logger.info("样本外单股明细：%s", test_detail_path)
    logger.info("组合日收益：%s", portfolio_daily_path)
    logger.info("组合年度汇总：%s", portfolio_period_path)
    logger.info("报告：%s", report_path)

    if not portfolio_periods.empty:
        logger.info("组合年度表现：")
        print(portfolio_periods.to_string(index=False))

    if overall_metrics:
        logger.info("组合整体样本外表现：")
        logger.info("总收益: %.2f%%", overall_metrics['total_return'] * 100)
        logger.info("年化收益: %.2f%%", overall_metrics['annual_return'] * 100)
        logger.info("最大回撤: %.2f%%", overall_metrics['max_drawdown'] * 100)
        logger.info("夏普比率: %.4f", overall_metrics['sharpe'])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MA v2 walk-forward：个股均线交叉 + 大盘趋势过滤")

    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT), help="QMT CSV 导出根目录")
    parser.add_argument("--market", default="ALL", choices=["ALL", "SH", "SZ"], help="市场筛选")
    parser.add_argument("--security-type", default="stock", choices=["stock", "index", "other", "ALL"], help="证券类型筛选")

    parser.add_argument("--train-start", default="20150101", help="训练起点，例如 20150101")
    parser.add_argument("--first-test-year", type=int, default=2021, help="第一个样本外测试年份")
    parser.add_argument("--last-test-year", type=int, default=0, help="最后一个测试年份；0 表示自动使用当前年份")
    parser.add_argument("--end", default="", help="数据结束日期，例如 20260514；留空表示使用数据最新日期")

    parser.add_argument("--fast-list", default="5,10,20", help="个股快均线列表，例如 5,10,20")
    parser.add_argument("--slow-list", default="60,120,250", help="个股慢均线列表，例如 60,120,250")

    parser.add_argument(
        "--benchmark-list",
        default=DEFAULT_BENCHMARK_LIST,
        help="大盘过滤基准列表，例如 000300.SH,000905.SH,000852.SH",
    )
    parser.add_argument("--benchmark-fast-list", default="20", help="基准快均线列表，默认 20")
    parser.add_argument("--benchmark-slow-list", default="120", help="基准慢均线列表，默认 120")

    parser.add_argument("--warmup-buffer", type=int, default=10, help="慢均线之外额外要求的缓冲行数")
    parser.add_argument("--portfolio-size", type=int, default=20, help="每期入选股票数量")
    parser.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金，仅用于指标换算")

    parser.add_argument("--commission", type=float, default=0.0001, help="佣金率，默认万一")
    parser.add_argument("--sell-tax", type=float, default=0.0005, help="卖出印花税，默认万五")
    parser.add_argument("--slippage", type=float, default=0.0, help="滑点率，默认 0")

    # 训练期筛选条件
    parser.add_argument("--min-train-rows", type=int, default=1000, help="训练期最少样本行数")
    parser.add_argument("--min-train-trades", type=int, default=4, help="训练期最少换仓次数")
    parser.add_argument("--max-train-drawdown", type=float, default=-0.55, help="训练期最大回撤下限，例如 -0.55 表示不低于 -55%")
    parser.add_argument("--min-train-sharpe", type=float, default=0.2, help="训练期最低夏普")
    parser.add_argument("--min-train-annual-return", type=float, default=0.02, help="训练期最低年化收益")
    parser.add_argument("--min-train-excess-total-return", type=float, default=0.0, help="训练期最低超额收益")
    parser.add_argument(
        "--train-excess-mode",
        choices=["stock_only", "buy_hold", "none"],
        default="stock_only",
        help="训练期超额收益约束口径：stock_only=要求跑赢无大盘过滤的个股均线；buy_hold=要求跑赢买入持有；none=不约束。",
    )
    parser.add_argument("--allow-negative-train-excess", action="store_true", help="允许训练期超额收益为负")

    parser.add_argument("--limit", type=int, default=0, help="只测试前 N 个标的；用于调试")
    parser.add_argument("--progress-every", type=int, default=500, help="训练阶段每处理 N 个标的打印一次进度")
    parser.add_argument("--output-name", default="wf_ma_mf", help="输出文件名前缀")
    parser.add_argument("--workers", type=parse_workers, default=1, help="并行进程数。默认 1（单进程）。使用 'auto' 自动检测 CPU 核心数，或指定整数如 --workers 6")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.last_test_year == 0:
        args.last_test_year = None

    train_start_dt = parse_date_yyyymmdd(args.train_start)

    fast_list = parse_int_list(args.fast_list)
    slow_list = parse_int_list(args.slow_list)

    stock_pairs = [(f, s) for f in fast_list for s in slow_list if f < s]
    if not stock_pairs:
        raise ValueError("没有有效个股参数组合，要求 fast < slow。")

    catalog = build_catalog(args)
    benchmark_cache = load_benchmark_cache(args)
    test_years = get_test_years(args)

    logger.info("MA v2 walk-forward 验证配置：")
    logger.info("市场：%s", args.market)
    logger.info("证券类型：%s", args.security_type)
    logger.info("标的数量：%d", len(catalog))
    logger.info("测试年份：%s", test_years)
    logger.info("个股参数组合：%s", stock_pairs)
    logger.info("大盘过滤组合：%s", list(benchmark_cache.keys()))
    logger.info("每期组合数量：%d", args.portfolio_size)
    logger.info("训练起点：%s", args.train_start)
    logger.info("训练期超额约束：%s", args.train_excess_mode)

    data_cache: dict[str, pd.DataFrame] = {}

    selected_all_rows = []
    test_detail_all_rows = []
    portfolio_daily_rows = []
    portfolio_period_rows = []

    running_equity = args.cash

    # 将 benchmark_cache 转为可序列化的列表格式
    benchmark_cache_data = []
    for (bench_symbol, bench_fast, bench_slow), bench_item in benchmark_cache.items():
        benchmark_cache_data.append((
            (bench_symbol, bench_fast, bench_slow),
            bench_item["filter_df"],
            bench_item["csv_path"],
            bench_item["benchmark"],
        ))

    if args.workers > 1:
        # 并行模式：按股票跨所有年份并行训练
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = []
        for _, item in catalog.iterrows():
            item_dict = item.to_dict()
            tasks.append((
                item_dict, test_years, args.train_start, args.end,
                stock_pairs, benchmark_cache_data,
                args.commission, args.sell_tax, args.slippage,
                args.cash, args.warmup_buffer,
                args.min_train_rows, args.min_train_trades, args.max_train_drawdown,
                args.min_train_sharpe, args.min_train_annual_return,
                args.min_train_excess_total_return, args.allow_negative_train_excess,
                args.train_excess_mode,
            ))

        total_stocks = len(tasks)
        completed_stocks = 0
        all_candidates_by_year = {year: [] for year in test_years}

        logger.info("并行训练：%d 进程处理 %d 只股票，跨 %d 个年份...", args.workers, total_stocks, len(test_years))

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(train_one_stock_all_years_v2, t): t[0]["symbol"] for t in tasks}

            for future in as_completed(futures):
                completed_stocks += 1
                if completed_stocks % 100 == 0:
                    logger.info("训练进度：%d/%d 只股票完成", completed_stocks, total_stocks)

                try:
                    stock_results = future.result()
                    for year, candidate in stock_results.items():
                        all_candidates_by_year[year].append(candidate)
                except Exception as exc:
                    symbol = futures[future]
                    logger.error("训练错误 %s: %s", symbol, repr(exc))

        # Phase 2 & 3: 串行 - 每年选 top N + 测试
        for year in test_years:
            train_end_dt = pd.Timestamp(year=year - 1, month=12, day=31)
            test_start_dt = pd.Timestamp(year=year, month=1, day=1)
            test_end_dt = pd.Timestamp(year=year, month=12, day=31)

            if args.end:
                end_dt = parse_date_yyyymmdd(args.end)
                if test_end_dt > end_dt:
                    test_end_dt = end_dt

            if train_end_dt < train_start_dt:
                logger.warning("跳过年份 %d: 训练期为空", year)
                continue

            logger.info("=" * 80)
            logger.info("测试年份：%d", year)
            logger.info("训练区间：%s 至 %s", train_start_dt.date(), train_end_dt.date())
            logger.info("测试区间：%s 至 %s", test_start_dt.date(), test_end_dt.date())

            candidates = all_candidates_by_year.get(year, [])
            if not candidates:
                logger.warning("无入选标的 %d", year)
                continue

            candidates_df = pd.DataFrame(candidates)
            candidates_df = candidates_df.sort_values(
                by=["train_score", "train_sharpe", "train_annual_return"],
                ascending=[False, False, False],
            )
            best_per_symbol = candidates_df.drop_duplicates(subset=["symbol"], keep="first")
            selected = best_per_symbol.head(args.portfolio_size).reset_index(drop=True)
            selected["selected_rank"] = range(1, len(selected) + 1)

            if selected.empty:
                logger.warning("无入选标的 %d", year)
                continue

            selected["test_year"] = year
            selected_all_rows.append(selected)

            logger.info("训练筛选后入选数量：%d", len(selected))
            print(
                selected[
                    [
                        "selected_rank",
                        "symbol",
                        "fast",
                        "slow",
                        "benchmark",
                        "benchmark_fast",
                        "benchmark_slow",
                        "train_annual_return",
                        "train_max_drawdown",
                        "train_sharpe",
                        "train_score",
                        "train_excess_vs_stock_only_total_return",
                    ]
                ].head(20).to_string(index=False)
            )

            # 测试阶段需要 data_cache，为选中的股票加载数据
            for symbol in selected["symbol"].unique():
                if symbol not in data_cache:
                    csv_path_str = selected.loc[selected["symbol"] == symbol, "csv_path"].iloc[0]
                    data_cache[symbol] = ma.load_qmt_price_csv(Path(csv_path_str), args.train_start, args.end)

            test_detail, returns_df = test_selected_for_period(
                selected=selected,
                data_cache=data_cache,
                benchmark_cache=benchmark_cache,
                test_start_dt=test_start_dt,
                test_end_dt=test_end_dt,
                args=args,
            )

            if not test_detail.empty:
                test_detail_all_rows.append(test_detail)

            period_metrics = calc_portfolio_period_metrics(
                returns_df=returns_df,
                test_year=year,
                cash=args.cash,
            )
            portfolio_period_rows.append(period_metrics)

            if not returns_df.empty:
                portfolio_ret = returns_df.mean(axis=1).sort_index()
                period_equity = running_equity * (1.0 + portfolio_ret).cumprod()
                running_equity = float(period_equity.iloc[-1])

                period_daily = pd.DataFrame(
                    {
                        "date": portfolio_ret.index,
                        "test_year": year,
                        "portfolio_ret": portfolio_ret.values,
                        "portfolio_equity": period_equity.values,
                        "portfolio_size_actual": returns_df.shape[1],
                    }
                )
                portfolio_daily_rows.append(period_daily)

            logger.info("样本外组合表现：")
            print(pd.DataFrame([period_metrics]).to_string(index=False))

    else:
        # 单进程模式：原有逻辑
        for year in test_years:
            train_end_dt = pd.Timestamp(year=year - 1, month=12, day=31)
            test_start_dt = pd.Timestamp(year=year, month=1, day=1)
            test_end_dt = pd.Timestamp(year=year, month=12, day=31)

            if args.end:
                end_dt = parse_date_yyyymmdd(args.end)
                if test_end_dt > end_dt:
                    test_end_dt = end_dt

            if train_end_dt < train_start_dt:
                logger.warning("跳过年份 %d: 训练期为空", year)
                continue

            logger.info("=" * 80)
            logger.info("测试年份：%d", year)
            logger.info("训练区间：%s 至 %s", train_start_dt.date(), train_end_dt.date())
            logger.info("测试区间：%s 至 %s", test_start_dt.date(), test_end_dt.date())

            selected = train_select_for_period(
                catalog=catalog,
                data_cache=data_cache,
                benchmark_cache=benchmark_cache,
                train_start_dt=train_start_dt,
                train_end_dt=train_end_dt,
                stock_pairs=stock_pairs,
                args=args,
            )

            if selected.empty:
                logger.warning("无入选标的 %d", year)
                continue

            selected["test_year"] = year
            selected_all_rows.append(selected)

            logger.info("训练筛选后入选数量：%d", len(selected))
            print(
                selected[
                    [
                        "selected_rank",
                        "symbol",
                        "fast",
                        "slow",
                        "benchmark",
                        "benchmark_fast",
                        "benchmark_slow",
                        "train_annual_return",
                        "train_max_drawdown",
                        "train_sharpe",
                        "train_score",
                        "train_excess_vs_stock_only_total_return",
                    ]
                ].head(20).to_string(index=False)
            )

            test_detail, returns_df = test_selected_for_period(
                selected=selected,
                data_cache=data_cache,
                benchmark_cache=benchmark_cache,
                test_start_dt=test_start_dt,
                test_end_dt=test_end_dt,
                args=args,
            )

            if not test_detail.empty:
                test_detail_all_rows.append(test_detail)

            period_metrics = calc_portfolio_period_metrics(
                returns_df=returns_df,
                test_year=year,
                cash=args.cash,
            )
            portfolio_period_rows.append(period_metrics)

            if not returns_df.empty:
                portfolio_ret = returns_df.mean(axis=1).sort_index()
                period_equity = running_equity * (1.0 + portfolio_ret).cumprod()
                running_equity = float(period_equity.iloc[-1])

                period_daily = pd.DataFrame(
                    {
                        "date": portfolio_ret.index,
                        "test_year": year,
                        "portfolio_ret": portfolio_ret.values,
                        "portfolio_equity": period_equity.values,
                        "portfolio_size_actual": returns_df.shape[1],
                    }
                )
                portfolio_daily_rows.append(period_daily)

            logger.info("样本外组合表现：")
            print(pd.DataFrame([period_metrics]).to_string(index=False))

    if not selected_all_rows:
        raise RuntimeError("没有任何年度产生入选标的。请放宽训练筛选条件。")

    selected_all = pd.concat(selected_all_rows, ignore_index=True)

    if test_detail_all_rows:
        test_detail_all = pd.concat(test_detail_all_rows, ignore_index=True)
    else:
        test_detail_all = pd.DataFrame()

    if portfolio_daily_rows:
        portfolio_daily_all = pd.concat(portfolio_daily_rows, ignore_index=True)
    else:
        portfolio_daily_all = pd.DataFrame()

    portfolio_periods = pd.DataFrame(portfolio_period_rows)

    save_outputs(
        selected_all=selected_all,
        test_detail_all=test_detail_all,
        portfolio_daily_all=portfolio_daily_all,
        portfolio_periods=portfolio_periods,
        args=args,
    )


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception as exc:
        logger.error("程序异常：%s", repr(exc))
        raise