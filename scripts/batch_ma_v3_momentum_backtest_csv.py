# -*- coding: utf-8 -*-
"""
batch_ma_v3_momentum_backtest_csv.py

MA v3 批量回测：
个股趋势强度 / 动量排序 + 趋势确认 + 风险过滤

策略逻辑：
1. 趋势确认：
   close > ma_long AND ma_mid > ma_long AND ma_long_slope > 0

2. 动量 / 强度评分：
   trend_strength + momentum_return + trend_slope 组合

3. 大盘 regime filter：
   benchmark_ma_short > benchmark_ma_long

4. 最终信号：
   final_signal = trend_confirm AND market_filter

默认设计：
- 个股参数：ma-mid-list / ma-long-list / momentum-window-list
- 大盘过滤基准：000300.SH,000905.SH,000852.SH
- 大盘过滤参数：benchmark-ma-list
- 输出目录：backtests\\batch_ma_v3_momentum_csv

运行示例：

1. 小规模测试：
python scripts\\batch_ma_v3_momentum_backtest_csv.py --market SZ --security-type stock --limit 50 --ma-mid-list 60 --ma-long-list 250 --momentum-window-list 120 --benchmark-list 000300.SH --benchmark-ma-list 120 --sample-mode short

2. 全市场 v3 参数网格：
python scripts\\batch_ma_v3_momentum_backtest_csv.py --market ALL --security-type stock --ma-mid-list 20,60 --ma-long-list 120,250 --momentum-window-list 60,120 --benchmark-list 000300.SH,000905.SH,000852.SH --benchmark-ma-list 120,250 --sample-mode short
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

from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR, DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.metrics import max_drawdown_from_equity, calc_metrics_from_returns  # noqa: E402
from scripts.common.validation import parse_list as parse_symbol_list, parse_int_list, parse_workers, safe_symbol_tag  # noqa: E402
from scripts.common.backtest.engine import single_asset_backtest  # noqa: E402

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from scripts.common.feature_expression import normalize_zscore  # noqa: E402
from scripts.common.benchmark import prepare_benchmark_regime  # noqa: E402

logger = logging.getLogger(__name__)

BACKTEST_DIR = PROJECT_ROOT / "backtests" / "batch_ma_v3_momentum_csv"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


def compute_v3_signals(
    stock_df: pd.DataFrame,
    ma_mid: int,
    ma_long: int,
    momentum_window: int,
) -> pd.DataFrame:
    """计算 MA v3 信号列。"""
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


def run_one_backtest(
    stock_df: pd.DataFrame,
    benchmark_filter_df: pd.DataFrame,
    ma_mid: int,
    ma_long: int,
    momentum_window: int,
    cash: float,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> tuple[pd.DataFrame, dict]:
    """单股票 MA v3 回测，委托给共享引擎。

    主路径：final_signal = trend_confirm AND market_filter。
    对照路径：stock_only_signal = trend_confirm（不加大盘过滤）。
    对照路径通过引擎的 comparison_signal_col 参数自动计算。
    """
    result = compute_v3_signals(stock_df, ma_mid, ma_long, momentum_window)

    # 委托给共享回测引擎（market filter 对齐 + 信号合成 + 持仓/收益/成本 + 对照路径）
    result, metrics = single_asset_backtest(
        result=result,
        benchmark_filter_df=benchmark_filter_df,
        signal_col="trend_confirm",
        cash=cash,
        commission=commission,
        sell_tax=sell_tax,
        slippage=slippage,
        comparison_signal_col="trend_confirm",
    )

    return result, metrics


def calc_score(metrics: dict) -> float:
    """评分公式：综合收益、夏普、超额、回撤。"""
    return (
        metrics.get("strategy_annual_return", 0)
        + 0.20 * metrics.get("strategy_sharpe", 0)
        + 0.40 * metrics.get("excess_vs_stock_only_total_return", 0)
        + 0.30 * metrics.get("excess_vs_buy_hold_total_return", 0)
        + metrics.get("strategy_max_drawdown", 0)  # 负值，惩罚回撤
    )


def get_required_rows(sample_mode: str, ma_long: int, benchmark_ma: int,
                      warmup_buffer: int, long_min_rows: int, min_rows_arg: int) -> int:
    warmup_required = max(ma_long, int(benchmark_ma * 2.5)) + warmup_buffer
    if sample_mode == "short":
        return warmup_required
    elif sample_mode == "long":
        return max(long_min_rows, warmup_required)
    else:
        return max(min_rows_arg, warmup_required)


def build_summary_row(
    item: dict,
    benchmark_symbol: str,
    benchmark_csv_path: Path,
    ma_mid: int,
    ma_long: int,
    momentum_window: int,
    benchmark_ma: int,
    required_rows: int,
    sample_mode: str,
    df: pd.DataFrame,
    metrics: dict,
) -> dict:
    row = {
        "symbol": item.get("symbol", ""),
        "market": item.get("market", ""),
        "security_type": item.get("security_type", ""),
        "benchmark": benchmark_symbol,
        "benchmark_ma": benchmark_ma,
        "ma_mid": ma_mid,
        "ma_long": ma_long,
        "momentum_window": momentum_window,
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
    return row


def get_required_rows_simple(sample_mode, ma_long, benchmark_ma, warmup_buffer, long_min_rows, min_rows_arg):
    return get_required_rows(sample_mode, ma_long, benchmark_ma, warmup_buffer, long_min_rows, min_rows_arg)


def process_one_stock_v3(args_tuple):
    """单进程 worker：对一只股票遍历所有参数组合。"""
    (item, stock_pairs, benchmark_cache_data, start, end,
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

        for ma_mid, ma_long, momentum_window in stock_pairs:
            for bm_key, bm_data in benchmark_cache_data.items():
                benchmark_symbol = bm_data["benchmark"]
                benchmark_ma = bm_data["benchmark_ma"]
                benchmark_csv_path = Path(bm_data["csv_path"])
                filter_df = bm_data["filter_df"]

                required_rows = get_required_rows_simple(
                    sample_mode, ma_long, benchmark_ma,
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
                        ma_mid=ma_mid,
                        ma_long=ma_long,
                        momentum_window=momentum_window,
                        cash=cash,
                        commission=commission,
                        sell_tax=sell_tax,
                        slippage=slippage,
                    )
                    row = build_summary_row(
                        item, benchmark_symbol, benchmark_csv_path,
                        ma_mid, ma_long, momentum_window, benchmark_ma,
                        required_rows, sample_mode, df, metrics,
                    )
                    rows.append(row)
                except Exception as e:
                    errors.append({
                        "symbol": item.get("symbol", ""),
                        "error": str(e),
                    })
    except Exception as e:
        errors.append({"symbol": item.get("symbol", ""), "error": str(e)})

    return rows, skipped, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MA v3 批量回测：趋势强度 / 动量 + 趋势确认")

    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT))
    parser.add_argument("--market", default="ALL", help="市场：ALL / SH / SZ")
    parser.add_argument("--security-type", default="stock", help="证券类型：stock / index / other / ALL")
    parser.add_argument("--start", default="20150101", help="开始日期")
    parser.add_argument("--end", default="", help="结束日期")

    parser.add_argument("--ma-mid-list", default="20,60", help="中期均线列表，逗号分隔")
    parser.add_argument("--ma-long-list", default="120,250", help="长期均线列表，逗号分隔")
    parser.add_argument("--momentum-window-list", default="60,120", help="动量窗口列表，逗号分隔")
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
    parser.add_argument("--workers", default="1", help="并行 worker 数量，支持 'auto'")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ma_mid_list = parse_int_list(args.ma_mid_list)
    ma_long_list = parse_int_list(args.ma_long_list)
    momentum_window_list = parse_int_list(args.momentum_window_list)
    benchmark_list = parse_symbol_list(args.benchmark_list)
    benchmark_ma_list = parse_int_list(args.benchmark_ma_list)

    # 构建参数组合：ma_mid < ma_long
    stock_pairs = [
        (mid, long_, mom)
        for mid in ma_mid_list
        for long_ in ma_long_list
        for mom in momentum_window_list
        if mid < long_
    ]

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
    logger.info(f"参数组合: {len(stock_pairs)}")
    logger.info(f"基准组合: {len(benchmark_pairs)}")
    logger.info(f"总任务数: {len(catalog) * len(stock_pairs) * len(benchmark_pairs)}")

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
                item_dict, stock_pairs, benchmark_cache_data,
                args.start, args.end,
                args.cash, args.commission, args.sell_tax, args.slippage,
                args.sample_mode, args.warmup_buffer, args.long_min_rows, args.min_rows,
            ))

        logger.info(f"启动 {workers} 个 worker...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one_stock_v3, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                rows, skipped, errors = future.result()
                all_rows.extend(rows)
                all_skipped.extend(skipped)
                all_errors.extend(errors)
                if i % 100 == 0:
                    logger.info(f"完成 {i}/{len(tasks)}")
    else:
        for idx, (_, item) in enumerate(catalog.iterrows(), 1):
            rows, skipped, errors = process_one_stock_v3((
                item.to_dict(), stock_pairs,
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
    tag = (
        f"v3_mom_mid{'_'.join(map(str, ma_mid_list))}"
        f"_long{'_'.join(map(str, ma_long_list))}"
        f"_mom{'_'.join(map(str, momentum_window_list))}"
        f"_bm{'_'.join(safe_symbol_tag(b) for b in benchmark_list)}"
        f"_bma{'_'.join(map(str, benchmark_ma_list))}"
        f"_{args.sample_mode}"
    )

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = BACKTEST_DIR / f"batch_v3_momentum_summary_{tag}.csv"
    top50_path = BACKTEST_DIR / f"batch_v3_momentum_top50_{tag}.csv"
    skipped_path = BACKTEST_DIR / f"batch_v3_momentum_skipped_{tag}.csv"
    errors_path = BACKTEST_DIR / f"batch_v3_momentum_errors_{tag}.csv"

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
            "symbol", "ma_mid", "ma_long", "momentum_window", "benchmark", "benchmark_ma",
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
