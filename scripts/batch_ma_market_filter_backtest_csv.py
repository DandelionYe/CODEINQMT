# -*- coding: utf-8 -*-
"""
batch_ma_market_filter_backtest_csv.py

MA v2 批量回测：
个股均线交叉 + 大盘趋势过滤

策略逻辑：
1. 个股信号：
   stock_fast_ma > stock_slow_ma

2. 大盘过滤：
   benchmark_fast_ma > benchmark_slow_ma

3. 最终信号：
   final_signal = stock_signal AND market_filter

4. 执行规则：
   当日收盘生成信号，次日持仓，避免未来函数。

默认设计：
- 个股参数：fast-list / slow-list
- 大盘过滤基准：000300.SH,000905.SH,000852.SH
- 大盘过滤参数：20 / 120
- 大盘过滤为 False 时直接空仓
- 输出目录：backtests\\batch_ma_market_filter_csv

运行示例：

1. 小规模测试，只跑深市前 50 个股票，单组参数，单个过滤基准：
python scripts\\batch_ma_market_filter_backtest_csv.py --market SZ --security-type stock --limit 50 --fast-list 20 --slow-list 120 --benchmark-list 000905.SH

2. 深市 v2 参数网格，三个过滤基准：
python scripts\\batch_ma_market_filter_backtest_csv.py --market SZ --security-type stock --fast-list 5,10,20 --slow-list 60,120,250 --benchmark-list 000300.SH,000905.SH,000852.SH --sample-mode short

3. 沪市 v2 参数网格：
python scripts\\batch_ma_market_filter_backtest_csv.py --market SH --security-type stock --fast-list 5,10,20 --slow-list 60,120,250 --benchmark-list 000300.SH,000905.SH,000852.SH --sample-mode short

4. 全市场 v2 参数网格：
python scripts\\batch_ma_market_filter_backtest_csv.py --market ALL --security-type stock --fast-list 5,10,20 --slow-list 60,120,250 --benchmark-list 000300.SH,000905.SH,000852.SH --sample-mode short

5. 长样本稳健版：
python scripts\\batch_ma_market_filter_backtest_csv.py --market SZ --security-type stock --fast-list 5,10,20 --slow-list 60,120,250 --sample-mode long
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR, DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.metrics import max_drawdown_from_equity, calc_metrics_from_returns  # noqa: E402
from scripts.common.validation import parse_list as parse_symbol_list, parse_int_list, parse_workers, safe_symbol_tag  # noqa: E402

logger = logging.getLogger(__name__)

BACKTEST_DIR = PROJECT_ROOT / "backtests" / "batch_ma_market_filter_csv"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)





def prepare_benchmark_filter(
    benchmark_df: pd.DataFrame,
    benchmark_fast: int,
    benchmark_slow: int,
) -> pd.DataFrame:
    bench = benchmark_df.copy()

    bench["benchmark_fast_ma"] = bench["close"].rolling(benchmark_fast).mean()
    bench["benchmark_slow_ma"] = bench["close"].rolling(benchmark_slow).mean()

    bench["market_filter"] = np.where(
        bench["benchmark_fast_ma"] > bench["benchmark_slow_ma"],
        1,
        0,
    )

    return bench[["date", "close", "benchmark_fast_ma", "benchmark_slow_ma", "market_filter"]].copy()


def run_one_backtest(
    stock_df: pd.DataFrame,
    benchmark_filter_df: pd.DataFrame,
    fast: int,
    slow: int,
    cash: float,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> tuple[pd.DataFrame, dict]:
    result = stock_df.copy()
    result = result.set_index("date", drop=False).sort_index()

    result["fast_ma"] = result["close"].rolling(fast).mean()
    result["slow_ma"] = result["close"].rolling(slow).mean()
    result["stock_signal"] = np.where(result["fast_ma"] > result["slow_ma"], 1, 0)

    bench = benchmark_filter_df.copy()
    bench = bench.rename(columns={"close": "benchmark_close"})
    bench = bench.set_index("date", drop=False).sort_index()

    result["benchmark_close"] = bench["benchmark_close"].reindex(result.index).ffill()
    result["benchmark_fast_ma"] = bench["benchmark_fast_ma"].reindex(result.index).ffill()
    result["benchmark_slow_ma"] = bench["benchmark_slow_ma"].reindex(result.index).ffill()
    result["market_filter"] = bench["market_filter"].reindex(result.index).ffill().fillna(0).astype(int)

    result["final_signal"] = (
        (result["stock_signal"] == 1)
        & (result["market_filter"] == 1)
    ).astype(int)

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

    # v1 对照成本：只用个股均线，不加大盘过滤
    stock_only_pos_change = result["stock_only_position"].diff().fillna(result["stock_only_position"])
    stock_only_buy = stock_only_pos_change.clip(lower=0)
    stock_only_sell = (-stock_only_pos_change).clip(lower=0)

    result["stock_only_cost"] = (
        stock_only_buy * (commission + slippage)
        + stock_only_sell * (commission + sell_tax + slippage)
    )

    result["stock_only_ret"] = result["stock_only_position"] * result["stock_ret"] - result["stock_only_cost"]

    result["equity"] = cash * (1.0 + result["strategy_ret"]).cumprod()
    result["stock_only_equity"] = cash * (1.0 + result["stock_only_ret"]).cumprod()
    result["buy_hold_equity"] = cash * (1.0 + result["stock_ret"]).cumprod()

    strategy_metrics = calc_metrics_from_returns(
        ret=result["strategy_ret"],
        position=result["position"],
        cash=cash,
    )

    stock_only_metrics = calc_metrics_from_returns(
        ret=result["stock_only_ret"],
        position=result["stock_only_position"],
        cash=cash,
    )

    buy_hold_metrics = calc_metrics_from_returns(
        ret=result["stock_ret"],
        position=None,
        cash=cash,
    )

    metrics = {
        "strategy_total_return": strategy_metrics["total_return"],
        "strategy_annual_return": strategy_metrics["annual_return"],
        "strategy_annual_volatility": strategy_metrics["annual_volatility"],
        "strategy_max_drawdown": strategy_metrics["max_drawdown"],
        "strategy_sharpe": strategy_metrics["sharpe"],
        "strategy_trade_count": strategy_metrics["trade_count"],
        "strategy_final_equity": strategy_metrics["final_equity"],
        "stock_only_total_return": stock_only_metrics["total_return"],
        "stock_only_annual_return": stock_only_metrics["annual_return"],
        "stock_only_annual_volatility": stock_only_metrics["annual_volatility"],
        "stock_only_max_drawdown": stock_only_metrics["max_drawdown"],
        "stock_only_sharpe": stock_only_metrics["sharpe"],
        "stock_only_trade_count": stock_only_metrics["trade_count"],
        "buy_hold_total_return": buy_hold_metrics["total_return"],
        "buy_hold_annual_return": buy_hold_metrics["annual_return"],
        "buy_hold_max_drawdown": buy_hold_metrics["max_drawdown"],
        "excess_vs_stock_only_total_return": strategy_metrics["total_return"] - stock_only_metrics["total_return"],
        "excess_vs_buy_hold_total_return": strategy_metrics["total_return"] - buy_hold_metrics["total_return"],
        "market_filter_on_ratio": float(result["market_filter"].mean()),
        "strategy_exposure_ratio": float(result["position"].mean()),
        "stock_only_exposure_ratio": float(result["stock_only_position"].mean()),
        "days": strategy_metrics["days"],
    }

    return result, metrics


def calc_score(metrics: dict) -> float:
    sharpe = 0.0 if pd.isna(metrics["strategy_sharpe"]) else metrics["strategy_sharpe"]

    return float(
        metrics["strategy_annual_return"]
        + 0.20 * sharpe
        + 0.40 * metrics["excess_vs_stock_only_total_return"]
        + 0.30 * metrics["excess_vs_buy_hold_total_return"]
        + metrics["strategy_max_drawdown"]
    )


def get_required_rows(args: argparse.Namespace, slow: int, benchmark_slow: int) -> int:
    warmup_required = max(slow, benchmark_slow) + args.warmup_buffer

    if args.sample_mode == "short":
        return warmup_required

    if args.sample_mode == "long":
        return max(args.long_min_rows, warmup_required)

    if args.sample_mode == "custom":
        if args.min_rows is None:
            raise ValueError("--sample-mode custom 时必须提供 --min-rows。")
        return max(args.min_rows, warmup_required)

    raise ValueError(f"未知 sample_mode: {args.sample_mode}")


def build_summary_row(
    item: pd.Series,
    benchmark_symbol: str,
    benchmark_csv_path: Path,
    fast: int,
    slow: int,
    benchmark_fast: int,
    benchmark_slow: int,
    required_rows: int,
    sample_mode: str,
    df: pd.DataFrame,
    metrics: dict,
) -> dict:
    row = {
        "symbol": item["symbol"],
        "market": item["market"],
        "security_type": item["security_type"],
        "benchmark": benchmark_symbol,
        "benchmark_fast": benchmark_fast,
        "benchmark_slow": benchmark_slow,
        "fast": fast,
        "slow": slow,
        "sample_mode": sample_mode,
        "required_rows": required_rows,
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "rows": len(df),
        "csv_path": item["csv_path"],
        "benchmark_csv_path": str(benchmark_csv_path),
    }

    row.update(metrics)
    row["score"] = calc_score(metrics)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MA v2 批量回测：个股均线交叉 + 大盘趋势过滤")

    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT), help="QMT CSV 导出根目录")
    parser.add_argument("--market", default="ALL", choices=["ALL", "SH", "SZ"], help="市场筛选")
    parser.add_argument("--security-type", default="stock", choices=["stock", "index", "other", "ALL"], help="证券类型筛选")

    parser.add_argument("--start", default="20150101", help="开始日期，例如 20150101")
    parser.add_argument("--end", default="", help="结束日期，例如 20260514；留空表示最新")

    parser.add_argument("--fast-list", default="20", help="个股快均线列表，例如 5,10,20")
    parser.add_argument("--slow-list", default="120", help="个股慢均线列表，例如 60,120,250")

    parser.add_argument(
        "--benchmark-list",
        default=DEFAULT_BENCHMARK_LIST,
        help="大盘过滤基准列表，例如 000300.SH,000905.SH,000852.SH",
    )
    parser.add_argument("--benchmark-fast-list", default="20", help="基准快均线列表，默认 20")
    parser.add_argument("--benchmark-slow-list", default="120", help="基准慢均线列表，默认 120")

    parser.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0001, help="佣金率，默认万一")
    parser.add_argument("--sell-tax", type=float, default=0.0005, help="卖出印花税，默认万五")
    parser.add_argument("--slippage", type=float, default=0.0, help="滑点率，默认 0")

    parser.add_argument(
        "--sample-mode",
        choices=["short", "long", "custom"],
        default="short",
        help=(
            "样本模式："
            "short=只要求 rows >= max(slow, benchmark_slow)+warmup-buffer；"
            "long=长样本筛选，默认 rows >= 1500；"
            "custom=使用 --min-rows 自定义。"
        ),
    )
    parser.add_argument("--warmup-buffer", type=int, default=10, help="慢均线之外额外要求的缓冲行数")
    parser.add_argument("--long-min-rows", type=int, default=1500, help="long 模式下要求的最少行数")
    parser.add_argument("--min-rows", type=int, default=None, help="custom 模式下要求的最少行数")

    parser.add_argument("--limit", type=int, default=0, help="只测试前 N 个标的；0 表示不限制")
    parser.add_argument("--print-skips", action="store_true", help="打印被跳过的标的；默认只记录到 skipped 文件")
    parser.add_argument("--workers", type=parse_workers, default=1, help="并行进程数。默认 1（单进程）。使用 'auto' 自动检测 CPU 核心数，或指定整数如 --workers 6")

    return parser.parse_args()


def get_required_rows_simple(sample_mode: str, slow: int, benchmark_slow: int, warmup_buffer: int, long_min_rows: int, min_rows_arg: int | None) -> int:
    """根据 sample-mode 计算最少数据行数（纯函数版本，供 worker 使用）。"""
    warmup_required = max(slow, benchmark_slow) + warmup_buffer

    if sample_mode == "short":
        return warmup_required

    if sample_mode == "long":
        return max(long_min_rows, warmup_required)

    if sample_mode == "custom":
        if min_rows_arg is None:
            raise ValueError("--sample-mode custom 时必须提供 --min-rows。")
        return max(min_rows_arg, warmup_required)

    raise ValueError(f"未知 sample_mode: {sample_mode}")


def process_one_stock_v2(args_tuple):
    """
    处理一只股票（含大盘过滤）：加载 CSV，运行所有 stock_pairs × benchmark_cache 组合。
    供 ProcessPoolExecutor 使用的 worker 函数。
    """
    (item_dict, stock_pairs, benchmark_cache_data, start, end,
     cash, commission, sell_tax, slippage, sample_mode,
     warmup_buffer, long_min_rows, min_rows_arg) = args_tuple

    symbol = item_dict["symbol"]
    market = item_dict["market"]
    security_type = item_dict["security_type"]
    csv_path = Path(item_dict["csv_path"])

    rows = []
    skipped = []
    errors = []

    try:
        stock_df = ma.load_qmt_price_csv(csv_path, start, end)

        for fast, slow in stock_pairs:
            for bench_key, bench_filter_df, bench_csv_path_str, bench_symbol in benchmark_cache_data:
                _, benchmark_fast, benchmark_slow = bench_key

                required_rows = get_required_rows_simple(
                    sample_mode, slow, benchmark_slow, warmup_buffer, long_min_rows, min_rows_arg,
                )

                if len(stock_df) < required_rows:
                    skip_item = {
                        "symbol": symbol,
                        "market": market,
                        "security_type": security_type,
                        "benchmark": bench_symbol,
                        "fast": fast,
                        "slow": slow,
                        "benchmark_fast": benchmark_fast,
                        "benchmark_slow": benchmark_slow,
                        "sample_mode": sample_mode,
                        "rows": len(stock_df),
                        "required_rows": required_rows,
                        "csv_path": str(csv_path),
                        "reason": "rows_not_enough",
                    }
                    skipped.append(skip_item)
                    continue

                _, metrics = run_one_backtest(
                    stock_df=stock_df,
                    benchmark_filter_df=bench_filter_df,
                    fast=fast,
                    slow=slow,
                    cash=cash,
                    commission=commission,
                    sell_tax=sell_tax,
                    slippage=slippage,
                )

                row = {
                    "symbol": symbol,
                    "market": market,
                    "security_type": security_type,
                    "benchmark": bench_symbol,
                    "benchmark_fast": benchmark_fast,
                    "benchmark_slow": benchmark_slow,
                    "fast": fast,
                    "slow": slow,
                    "sample_mode": sample_mode,
                    "required_rows": required_rows,
                    "start_date": str(stock_df.index.min().date()),
                    "end_date": str(stock_df.index.max().date()),
                    "rows": len(stock_df),
                    "csv_path": str(csv_path),
                    "benchmark_csv_path": bench_csv_path_str,
                }
                row.update(metrics)
                row["score"] = calc_score(metrics)
                rows.append(row)

    except Exception as exc:
        errors.append(
            {
                "symbol": symbol,
                "market": market,
                "security_type": security_type,
                "csv_path": str(csv_path),
                "error": repr(exc),
            }
        )

    return rows, skipped, errors


def main() -> None:
    args = parse_args()

    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root

    fast_list = parse_int_list(args.fast_list)
    slow_list = parse_int_list(args.slow_list)
    benchmark_fast_list = parse_int_list(args.benchmark_fast_list)
    benchmark_slow_list = parse_int_list(args.benchmark_slow_list)
    benchmark_list = parse_symbol_list(args.benchmark_list)

    stock_pairs = [(f, s) for f in fast_list for s in slow_list if f < s]
    benchmark_pairs = [(f, s) for f in benchmark_fast_list for s in benchmark_slow_list if f < s]

    if not stock_pairs:
        raise ValueError("没有有效个股均线组合。要求 fast < slow。")

    if not benchmark_pairs:
        raise ValueError("没有有效基准均线组合。要求 benchmark_fast < benchmark_slow。")

    catalog = ma.scan_qmt_export(export_root)

    if args.market != "ALL":
        catalog = catalog[catalog["market"] == args.market]

    if args.security_type != "ALL":
        catalog = catalog[catalog["security_type"] == args.security_type]

    catalog = catalog.sort_values(["market", "symbol"]).reset_index(drop=True)

    if args.limit and args.limit > 0:
        catalog = catalog.head(args.limit)

    logger.info("MA v2 批量回测配置：")
    logger.info(f"市场：{args.market}")
    logger.info(f"证券类型：{args.security_type}")
    logger.info(f"标的数量：{len(catalog)}")
    logger.info(f"个股参数组合：{stock_pairs}")
    logger.info(f"基准列表：{benchmark_list}")
    logger.info(f"基准参数组合：{benchmark_pairs}")
    logger.info(f"日期区间：{args.start} 至 {args.end or 'latest'}")
    logger.info(f"样本模式：{args.sample_mode}")
    logger.info(f"warmup_buffer：{args.warmup_buffer}")

    # 预加载基准数据和过滤信号
    benchmark_cache: dict[tuple[str, int, int], tuple[pd.DataFrame, Path, str]] = {}

    for benchmark in benchmark_list:
        benchmark_csv_path, benchmark_symbol, _, _ = ma.find_csv_for_stock(
            stock=benchmark,
            export_root=export_root,
        )

        benchmark_df = ma.load_qmt_price_csv(
            csv_path=benchmark_csv_path,
            start=args.start,
            end=args.end,
        )

        for benchmark_fast, benchmark_slow in benchmark_pairs:
            benchmark_filter_df = prepare_benchmark_filter(
                benchmark_df=benchmark_df,
                benchmark_fast=benchmark_fast,
                benchmark_slow=benchmark_slow,
            )

            benchmark_cache[(benchmark_symbol, benchmark_fast, benchmark_slow)] = (
                benchmark_filter_df,
                benchmark_csv_path,
                benchmark_symbol,
            )

    rows = []
    skipped = []
    errors = []

    # 将 benchmark_cache 转为可序列化的列表格式
    benchmark_cache_data = []
    for (bench_symbol, bench_fast, bench_slow), (bench_filter_df, bench_csv_path, bench_symbol_for_row) in benchmark_cache.items():
        benchmark_cache_data.append((
            (bench_symbol, bench_fast, bench_slow),
            bench_filter_df,
            str(bench_csv_path),
            bench_symbol_for_row,
        ))

    if args.workers > 1:
        # 并行模式：按股票并行
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = []
        for _, item in catalog.iterrows():
            item_dict = item.to_dict()
            tasks.append((
                item_dict, stock_pairs, benchmark_cache_data, args.start, args.end,
                args.cash, args.commission, args.sell_tax, args.slippage,
                args.sample_mode, args.warmup_buffer, args.long_min_rows, args.min_rows,
            ))

        total_stocks = len(tasks)
        completed_stocks = 0
        logger.info(f"并行模式：{args.workers} 进程处理 {total_stocks} 只股票...")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_stock_v2, t): t[0]["symbol"] for t in tasks}

            for future in as_completed(futures):
                completed_stocks += 1
                if completed_stocks % 50 == 0:
                    logger.info(f"进度：{completed_stocks}/{total_stocks} 只股票完成")

                try:
                    stock_rows, stock_skipped, stock_errors = future.result()
                    rows.extend(stock_rows)
                    skipped.extend(stock_skipped)
                    errors.extend(stock_errors)
                except Exception as exc:
                    symbol = futures[future]
                    errors.append({"symbol": symbol, "error": repr(exc)})

    else:
        # 单进程模式：原有逻辑
        total_tasks = len(catalog) * len(stock_pairs) * len(benchmark_cache)
        task_no = 0

        for idx, item in catalog.iterrows():
            symbol = item["symbol"]
            csv_path = Path(item["csv_path"])

            try:
                stock_df = ma.load_qmt_price_csv(
                    csv_path=csv_path,
                    start=args.start,
                    end=args.end,
                )

                for fast, slow in stock_pairs:
                    for (benchmark_symbol, benchmark_fast, benchmark_slow), (
                        benchmark_filter_df,
                        benchmark_csv_path,
                        benchmark_symbol_for_row,
                    ) in benchmark_cache.items():
                        task_no += 1

                        if task_no % 500 == 0:
                            logger.info(f"进度：{task_no}/{total_tasks}")

                        required_rows = get_required_rows(
                            args=args,
                            slow=slow,
                            benchmark_slow=benchmark_slow,
                        )

                        if len(stock_df) < required_rows:
                            skip_item = {
                                "symbol": symbol,
                                "market": item["market"],
                                "security_type": item["security_type"],
                                "benchmark": benchmark_symbol,
                                "fast": fast,
                                "slow": slow,
                                "benchmark_fast": benchmark_fast,
                                "benchmark_slow": benchmark_slow,
                                "sample_mode": args.sample_mode,
                                "rows": len(stock_df),
                                "required_rows": required_rows,
                                "csv_path": str(csv_path),
                                "reason": "rows_not_enough",
                            }
                            skipped.append(skip_item)

                            if args.print_skips:
                                logger.warning(
                                    f"{symbol}: rows={len(stock_df)}, "
                                    f"required_rows={required_rows}, "
                                    f"fast={fast}, slow={slow}, "
                                    f"benchmark={benchmark_symbol}"
                                )
                            continue

                        result, metrics = run_one_backtest(
                            stock_df=stock_df,
                            benchmark_filter_df=benchmark_filter_df,
                            fast=fast,
                            slow=slow,
                            cash=args.cash,
                            commission=args.commission,
                            sell_tax=args.sell_tax,
                            slippage=args.slippage,
                        )

                        rows.append(
                            build_summary_row(
                                item=item,
                                benchmark_symbol=benchmark_symbol_for_row,
                                benchmark_csv_path=benchmark_csv_path,
                                fast=fast,
                                slow=slow,
                                benchmark_fast=benchmark_fast,
                                benchmark_slow=benchmark_slow,
                                required_rows=required_rows,
                                sample_mode=args.sample_mode,
                                df=stock_df,
                                metrics=metrics,
                            )
                        )

            except Exception as exc:
                errors.append(
                    {
                        "symbol": symbol,
                        "market": item.get("market", ""),
                        "security_type": item.get("security_type", ""),
                        "csv_path": str(csv_path),
                        "error": repr(exc),
                    }
                )

    if not rows:
        raise RuntimeError("没有生成任何回测结果。")

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=["score", "strategy_sharpe", "strategy_annual_return"],
        ascending=[False, False, False],
    )

    benchmark_tag = "b" + "-".join(safe_symbol_tag(x) for x in benchmark_list)
    fast_tag = "f" + "-".join(map(str, fast_list))
    slow_tag = "s" + "-".join(map(str, slow_list))
    bench_fast_tag = "bf" + "-".join(map(str, benchmark_fast_list))
    bench_slow_tag = "bs" + "-".join(map(str, benchmark_slow_list))

    if args.sample_mode == "short":
        sample_tag = f"short_warmup{args.warmup_buffer}"
    elif args.sample_mode == "long":
        sample_tag = f"long_minrows{args.long_min_rows}_warmup{args.warmup_buffer}"
    else:
        sample_tag = f"custom_minrows{args.min_rows}_warmup{args.warmup_buffer}"

    tag = (
        f"stock_{args.market}_{args.start}_{args.end or 'latest'}_"
        f"{fast_tag}_{slow_tag}_"
        f"{benchmark_tag}_{bench_fast_tag}_{bench_slow_tag}_"
        f"{sample_tag}"
    )

    summary_path = BACKTEST_DIR / f"batch_ma_mf_summary_{tag}.csv"
    top50_path = BACKTEST_DIR / f"batch_ma_mf_top50_{tag}.csv"
    skipped_path = BACKTEST_DIR / f"batch_ma_mf_skipped_{tag}.csv"
    errors_path = BACKTEST_DIR / f"batch_ma_mf_errors_{tag}.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary.head(50).to_csv(top50_path, index=False, encoding="utf-8-sig")

    if skipped:
        pd.DataFrame(skipped).to_csv(skipped_path, index=False, encoding="utf-8-sig")

    if errors:
        pd.DataFrame(errors).to_csv(errors_path, index=False, encoding="utf-8-sig")

    logger.info("MA v2 批量回测完成。")
    logger.info(f"结果数量：{len(summary)}")
    logger.info(f"跳过数量：{len(skipped)}")
    logger.info(f"错误数量：{len(errors)}")
    logger.info(f"汇总文件：{summary_path}")
    logger.info(f"Top50 文件：{top50_path}")

    if skipped:
        logger.info(f"跳过记录文件：{skipped_path}")

    if errors:
        logger.info(f"错误文件：{errors_path}")

    print("\nTop 20：")
    cols = [
        "symbol",
        "market",
        "benchmark",
        "fast",
        "slow",
        "benchmark_fast",
        "benchmark_slow",
        "strategy_annual_return",
        "strategy_max_drawdown",
        "strategy_sharpe",
        "strategy_total_return",
        "stock_only_total_return",
        "buy_hold_total_return",
        "excess_vs_stock_only_total_return",
        "excess_vs_buy_hold_total_return",
        "market_filter_on_ratio",
        "strategy_exposure_ratio",
        "strategy_trade_count",
        "rows",
        "score",
    ]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception:
        logger.error("程序异常：", exc_info=True)