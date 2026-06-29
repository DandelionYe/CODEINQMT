# -*- coding: utf-8 -*-
"""
batch_alpha_v4_research_backtest_csv.py

Alpha v4 批量回测：
支持 4 个 alpha variant 参数网格搜索。

运行示例：

1. 小规模测试：
python scripts\\batch_alpha_v4_research_backtest_csv.py --market SZ --security-type stock --limit 50 --alpha-variant-list pure_momentum,simplified_trend_momentum --momentum-window-list 60,120 --trend-ma-list 120,250 --vol-window-list 60 --breakout-window-list 120 --benchmark-list 000300.SH --benchmark-ma-list 120 --start 20150101 --workers 4

2. 全市场参数网格：
python scripts\\batch_alpha_v4_research_backtest_csv.py --market ALL --security-type stock --alpha-variant-list pure_momentum,simplified_trend_momentum,volatility_adjusted_momentum,breakout_momentum --momentum-window-list 60,120,250 --trend-ma-list 120,250 --vol-window-list 60,120 --breakout-window-list 60,120,250 --benchmark-list 000300.SH,000905.SH,000852.SH --benchmark-ma-list 120,250 --start 20150101 --sample-mode short --workers 10
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from strategies.alpha_v4_research_strategy_csv import (
    compute_alpha_v4_signals,
    prepare_benchmark_regime,
    VALID_VARIANTS,
)
from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR, DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.metrics import max_drawdown_from_equity, calc_metrics_from_returns  # noqa: E402
from scripts.common.validation import parse_list, parse_int_list, parse_workers, safe_symbol_tag  # noqa: E402

logger = logging.getLogger(__name__)

BACKTEST_DIR = PROJECT_ROOT / "backtests" / "batch_alpha_v4_research_csv"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)



VARIANT_TAGS = {
    "pure_momentum": "pm",
    "simplified_trend_momentum": "stm",
    "volatility_adjusted_momentum": "vam",
    "breakout_momentum": "bm",
}


def build_variant_param_combos(
    alpha_variant_list: list[str],
    momentum_window_list: list[int],
    trend_ma_list: list[int],
    vol_window_list: list[int],
    breakout_window_list: list[int],
) -> list[tuple[str, int, int, int, int]]:
    """根据 alpha variant 生成 variant-aware 参数组合，避免无意义重复。"""
    combos = []
    for av in alpha_variant_list:
        if av == "pure_momentum":
            for mw in momentum_window_list:
                combos.append((av, mw, trend_ma_list[0], vol_window_list[0], breakout_window_list[0]))
        elif av == "simplified_trend_momentum":
            for mw in momentum_window_list:
                for tma in trend_ma_list:
                    combos.append((av, mw, tma, vol_window_list[0], breakout_window_list[0]))
        elif av == "volatility_adjusted_momentum":
            for mw in momentum_window_list:
                for vw in vol_window_list:
                    combos.append((av, mw, trend_ma_list[0], vw, breakout_window_list[0]))
        elif av == "breakout_momentum":
            for bkw in breakout_window_list:
                combos.append((av, momentum_window_list[0], trend_ma_list[0], vol_window_list[0], bkw))
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


def compact_variant_list(values: list[str]) -> str:
    tags = [VARIANT_TAGS.get(v, v[:3]) for v in values]
    if len(tags) == len(VARIANT_TAGS) and set(values) == set(VARIANT_TAGS):
        return "all4"
    return "-".join(tags)


def compact_benchmark_list(values: list[str]) -> str:
    if len(values) == 1:
        return safe_symbol_tag(values[0])
    return f"bm{len(values)}"


def build_param_signature(
    alpha_variant_list: list[str],
    momentum_window_list: list[int],
    trend_ma_list: list[int],
    vol_window_list: list[int],
    breakout_window_list: list[int],
    benchmark_list: list[str],
    benchmark_ma_list: list[int],
) -> str:
    return "|".join([
        ",".join(alpha_variant_list),
        ",".join(map(str, momentum_window_list)),
        ",".join(map(str, trend_ma_list)),
        ",".join(map(str, vol_window_list)),
        ",".join(map(str, breakout_window_list)),
        ",".join(benchmark_list),
        ",".join(map(str, benchmark_ma_list)),
    ])


def build_batch_tag(
    alpha_variant_list: list[str],
    momentum_window_list: list[int],
    trend_ma_list: list[int],
    vol_window_list: list[int],
    breakout_window_list: list[int],
    benchmark_list: list[str],
    benchmark_ma_list: list[int],
    sample_mode: str,
) -> str:
    signature = build_param_signature(
        alpha_variant_list,
        momentum_window_list,
        trend_ma_list,
        vol_window_list,
        breakout_window_list,
        benchmark_list,
        benchmark_ma_list,
    )
    short_hash = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    return (
        f"alpha_v4_av{compact_variant_list(alpha_variant_list)}"
        f"_mw{compact_int_list(momentum_window_list)}"
        f"_tm{compact_int_list(trend_ma_list)}"
        f"_vw{compact_int_list(vol_window_list)}"
        f"_bw{compact_int_list(breakout_window_list)}"
        f"_{compact_benchmark_list(benchmark_list)}"
        f"_bma{compact_int_list(benchmark_ma_list)}"
        f"_{sample_mode}"
        f"_h{short_hash}"
    )



def run_one_backtest(
    stock_df: pd.DataFrame,
    benchmark_filter_df: pd.DataFrame,
    alpha_variant: str,
    momentum_window: int,
    trend_ma: int,
    vol_window: int,
    breakout_window: int,
    cash: float,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> tuple[pd.DataFrame, dict]:
    result = compute_alpha_v4_signals(
        stock_df, alpha_variant, momentum_window, trend_ma, vol_window, breakout_window,
    )

    bench_filter = benchmark_filter_df.rename(columns={"close": "benchmark_close"})
    bench_filter = bench_filter.set_index("date", drop=False)
    result = result.set_index("date", drop=False)

    result["benchmark_close"] = bench_filter["benchmark_close"].reindex(result.index).ffill()
    result["benchmark_ma_short"] = bench_filter["benchmark_ma_short"].reindex(result.index).ffill()
    result["benchmark_ma_long"] = bench_filter["benchmark_ma_long"].reindex(result.index).ffill()
    result["market_filter"] = bench_filter["market_filter"].reindex(result.index).ffill().fillna(0).astype(int)

    result["final_signal"] = (
        (result["alpha_signal"] == 1)
        & (result["market_filter"] == 1)
    ).astype(int)

    result["position"] = result["final_signal"].shift(1, fill_value=0).astype(float)
    result["stock_ret"] = result["close"].pct_change().fillna(0)

    pos_change = result["position"].diff().fillna(0)
    buy_turnover = pos_change.clip(lower=0)
    sell_turnover = (-pos_change).clip(lower=0)
    result["cost"] = (
        buy_turnover * (commission + slippage)
        + sell_turnover * (commission + sell_tax + slippage)
    )
    result["strategy_ret"] = result["position"] * result["stock_ret"] - result["cost"]

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

    return result, metrics


def calc_score(metrics: dict) -> float:
    return (
        metrics.get("strategy_annual_return", 0)
        + 0.20 * metrics.get("strategy_sharpe", 0)
        + 0.30 * metrics.get("excess_vs_buy_hold_total_return", 0)
        + metrics.get("strategy_max_drawdown", 0)
    )


def get_required_rows(sample_mode: str, trend_ma: int, breakout_window: int,
                      benchmark_ma: int, warmup_buffer: int, long_min_rows: int, min_rows_arg: int) -> int:
    warmup_required = max(trend_ma, breakout_window, int(benchmark_ma * 2.5)) + warmup_buffer
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
     sample_mode, warmup_buffer, long_min_rows, min_rows_arg) = args_tuple

    rows = []
    skipped = []
    errors = []

    try:
        csv_path = Path(item["csv_path"])
        try:
            df = ma.load_qmt_price_csv(csv_path=csv_path, start=start, end=end)
        except RuntimeError:
            skipped.append({"symbol": item.get("symbol", ""), "reason": "数据为空或不足"})
            return rows, skipped, errors

        for alpha_variant, momentum_window, trend_ma, vol_window, breakout_window in param_combos:
            for bm_key, bm_data in benchmark_cache_data.items():
                benchmark_symbol = bm_data["benchmark"]
                benchmark_ma = bm_data["benchmark_ma"]
                benchmark_csv_path = Path(bm_data["csv_path"])
                filter_df = bm_data["filter_df"]

                required_rows = get_required_rows(
                    sample_mode, trend_ma, breakout_window, benchmark_ma,
                    warmup_buffer, long_min_rows, min_rows_arg,
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
                        alpha_variant=alpha_variant,
                        momentum_window=momentum_window,
                        trend_ma=trend_ma,
                        vol_window=vol_window,
                        breakout_window=breakout_window,
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
                        "momentum_window": momentum_window,
                        "trend_ma": trend_ma,
                        "vol_window": vol_window,
                        "breakout_window": breakout_window,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha v4 Research 批量回测")

    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT))
    parser.add_argument("--market", default="ALL", help="市场：ALL / SH / SZ")
    parser.add_argument("--security-type", default="stock", help="证券类型")
    parser.add_argument("--start", default="20150101", help="开始日期")
    parser.add_argument("--end", default="", help="结束日期")

    parser.add_argument("--alpha-variant-list", default=",".join(VALID_VARIANTS), help="Alpha variant 列表")
    parser.add_argument("--momentum-window-list", default="60,120,250", help="动量窗口列表")
    parser.add_argument("--trend-ma-list", default="120,250", help="趋势均线列表")
    parser.add_argument("--vol-window-list", default="60,120", help="波动率窗口列表")
    parser.add_argument("--breakout-window-list", default="60,120,250", help="突破窗口列表")
    parser.add_argument("--benchmark-list", default=DEFAULT_BENCHMARK_LIST, help="基准列表")
    parser.add_argument("--benchmark-ma-list", default="120,250", help="基准 MA 列表")

    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission", type=float, default=0.0001)
    parser.add_argument("--sell-tax", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0)

    parser.add_argument("--sample-mode", default="short", choices=["short", "long", "custom"])
    parser.add_argument("--warmup-buffer", type=int, default=10)
    parser.add_argument("--long-min-rows", type=int, default=1500)
    parser.add_argument("--min-rows", type=int, default=0)

    parser.add_argument("--limit", type=int, default=0, help="限制股票数量，0=不限制")
    parser.add_argument("--print-skips", action="store_true")
    parser.add_argument("--workers", default="1", help="并行 worker 数量")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    alpha_variant_list = parse_list(args.alpha_variant_list, upper=False)
    momentum_window_list = parse_int_list(args.momentum_window_list)
    trend_ma_list = parse_int_list(args.trend_ma_list)
    vol_window_list = parse_int_list(args.vol_window_list)
    breakout_window_list = parse_int_list(args.breakout_window_list)
    benchmark_list = [x.strip().upper() for x in args.benchmark_list.split(",") if x.strip()]
    benchmark_ma_list = parse_int_list(args.benchmark_ma_list)

    # 构建 variant-aware 参数组合
    param_combos = build_variant_param_combos(
        alpha_variant_list, momentum_window_list, trend_ma_list, vol_window_list, breakout_window_list,
    )

    benchmark_pairs = [
        (bm, bm_ma)
        for bm in benchmark_list
        for bm_ma in benchmark_ma_list
    ]

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

    logger.info(f"股票数量: {len(catalog)}")
    logger.info(f"Alpha variants: {alpha_variant_list}")
    logger.info(f"参数组合: {len(param_combos)}")
    logger.info(f"基准组合: {len(benchmark_pairs)}")
    logger.info(f"总任务数: {len(catalog) * len(param_combos) * len(benchmark_pairs)}")

    # 预加载基准数据
    benchmark_cache = {}
    for bm_symbol, bm_ma in benchmark_pairs:
        bm_csv_path, bm_sym, _, _ = ma.find_csv_for_stock(bm_symbol, export_root)
        bm_df = ma.load_qmt_price_csv(bm_csv_path, args.start, args.end)
        filter_df = prepare_benchmark_regime(bm_df, bm_ma)
        filter_df = filter_df.set_index("date", drop=False).sort_index()
        benchmark_cache[(bm_symbol, bm_ma)] = {
            "benchmark": bm_symbol,
            "benchmark_ma": bm_ma,
            "csv_path": str(bm_csv_path),
            "filter_df": filter_df,
        }

    workers = parse_workers(args.workers)
    all_rows = []
    all_skipped = []
    all_errors = []

    if workers > 1:
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
            item_dict = item.to_dict()
            tasks.append((
                item_dict, param_combos, benchmark_cache_data,
                args.start, args.end,
                args.cash, args.commission, args.sell_tax, args.slippage,
                args.sample_mode, args.warmup_buffer, args.long_min_rows, args.min_rows,
            ))

        logger.info(f"启动 {workers} 个 worker...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one_stock, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                rows, skipped, errors = future.result()
                all_rows.extend(rows)
                all_skipped.extend(skipped)
                all_errors.extend(errors)
                if i % 100 == 0:
                    logger.info(f"完成 {i}/{len(tasks)}")
    else:
        for idx, (_, item) in enumerate(catalog.iterrows(), 1):
            rows, skipped, errors = process_one_stock((
                item.to_dict(), param_combos,
                {k: {"benchmark": v["benchmark"], "benchmark_ma": v["benchmark_ma"],
                     "csv_path": v["csv_path"], "filter_df": v["filter_df"]}
                 for k, v in benchmark_cache.items()},
                args.start, args.end,
                args.cash, args.commission, args.sell_tax, args.slippage,
                args.sample_mode, args.warmup_buffer, args.long_min_rows, args.min_rows,
            ))
            all_rows.extend(rows)
            all_skipped.extend(skipped)
            all_errors.extend(errors)
            if idx % 100 == 0:
                logger.info(f"完成 {idx}/{len(catalog)}")

    # 构建 summary
    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        summary_df = summary_df.sort_values(["score", "strategy_sharpe", "strategy_annual_return"], ascending=False)
    else:
        summary_df = pd.DataFrame()

    # 输出文件名标签
    tag = build_batch_tag(
        alpha_variant_list,
        momentum_window_list,
        trend_ma_list,
        vol_window_list,
        breakout_window_list,
        benchmark_list,
        benchmark_ma_list,
        args.sample_mode,
    )

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = BACKTEST_DIR / f"batch_alpha_v4_summary_{tag}.csv"
    top50_path = BACKTEST_DIR / f"batch_alpha_v4_top50_{tag}.csv"
    skipped_path = BACKTEST_DIR / f"batch_alpha_v4_skipped_{tag}.csv"
    errors_path = BACKTEST_DIR / f"batch_alpha_v4_errors_{tag}.csv"

    if not summary_df.empty:
        summary_df.to_csv(summary_path, encoding="utf-8-sig", index=False)
        summary_df.head(50).to_csv(top50_path, encoding="utf-8-sig", index=False)
        logger.info(f"Summary: {summary_path}")
        logger.info(f"Top50: {top50_path}")

    if all_skipped:
        pd.DataFrame(all_skipped).to_csv(skipped_path, encoding="utf-8-sig", index=False)
        if args.print_skips:
            logger.info(f"Skipped: {skipped_path} ({len(all_skipped)} rows)")

    if all_errors:
        pd.DataFrame(all_errors).to_csv(errors_path, encoding="utf-8-sig", index=False)
        logger.info(f"Errors: {errors_path} ({len(all_errors)} rows)")

    # 打印 top 20
    if not summary_df.empty:
        display_cols = [
            "symbol", "alpha_variant", "momentum_window", "trend_ma", "vol_window", "breakout_window",
            "benchmark", "benchmark_ma",
            "strategy_total_return", "strategy_annual_return", "strategy_sharpe",
            "strategy_max_drawdown", "excess_vs_buy_hold_total_return", "score",
        ]
        available = [c for c in display_cols if c in summary_df.columns]
        print(f"\nTop 20:")
        print(summary_df[available].head(20).to_string(index=False))

    print(f"\n完成。共 {len(all_rows)} 条结果，{len(all_skipped)} 跳过，{len(all_errors)} 错误。")


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception:
        logger.error("程序异常：", exc_info=True)
