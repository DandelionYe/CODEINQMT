# -*- coding: utf-8 -*-
"""
batch_alpha_v7_research_backtest_csv.py

Alpha v7 批量回测：
使用 UGQlib 表达式层（feature_expression.py）计算信号，支持 4 个 alpha variant 参数网格搜索。

共享逻辑已提取到 scripts/common/wf_batch_shared.py。

运行示例：

1. 小规模测试：
python scripts\\batch_alpha_v7_research_backtest_csv.py --market SZ --security-type stock --limit 50 --alpha-variant-list short_term_reversal,low_volatility --reversal-window-list 10 --vol-window-list 60 --benchmark-list 000300.SH --benchmark-ma-list 120 --start 20150101 --workers 4

2. 全市场参数网格：
python scripts\\batch_alpha_v7_research_backtest_csv.py --market ALL --security-type stock --alpha-variant-list short_term_reversal,low_volatility,turnover_reversal,volume_price_divergence --reversal-window-list 5,10,20 --vol-window-list 20,60,120 --turnover-short-list 10 --turnover-long-list 60 --divergence-window-list 10,20,60 --benchmark-list 000300.SH,000905.SH,000852.SH --benchmark-ma-list 120,250 --start 20150101 --sample-mode short --workers 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from strategies.alpha_v7_research_strategy_csv import (
    compute_alpha_v7_signals,
    prepare_benchmark_regime,
    VALID_VARIANTS,
)
from scripts.common.constants import DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.validation import parse_list, parse_int_list, parse_workers  # noqa: E402
from scripts.common.wf_batch_shared import (  # noqa: E402
    WFConfig,
    build_batch_tag,
    build_variant_param_combos,
    compact_benchmark_list,
    compact_int_list,
    compact_variant_list,
    load_benchmark_cache,
    process_one_stock,
)

logger = logging.getLogger(__name__)

BACKTEST_DIR = PROJECT_ROOT / "backtests" / "batch_alpha_v7_research_csv"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

CFG = WFConfig(
    alpha_version="alpha_v7",
    output_dir_name="batch_alpha_v7_research_csv",
    compute_signals_fn=compute_alpha_v7_signals,
    prepare_benchmark_regime_fn=prepare_benchmark_regime,
    valid_variants=VALID_VARIANTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha v7 Research 批量回测（表达式层）")

    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT))
    parser.add_argument("--market", default="ALL", help="市场：ALL / SH / SZ")
    parser.add_argument("--security-type", default="stock", help="证券类型")
    parser.add_argument("--start", default="20150101", help="开始日期")
    parser.add_argument("--end", default="", help="结束日期")

    parser.add_argument("--alpha-variant-list", default=",".join(VALID_VARIANTS), help="Alpha variant 列表")
    parser.add_argument("--reversal-window-list", default="5,10,20", help="反转窗口列表")
    parser.add_argument("--vol-window-list", default="20,60,120", help="波动率窗口列表")
    parser.add_argument("--turnover-short-list", default="10", help="短期换手率窗口列表")
    parser.add_argument("--turnover-long-list", default="60", help="长期换手率窗口列表")
    parser.add_argument("--divergence-window-list", default="10,20,60", help="量价分歧窗口列表")
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
    reversal_window_list = parse_int_list(args.reversal_window_list)
    vol_window_list = parse_int_list(args.vol_window_list)
    turnover_short_list = parse_int_list(args.turnover_short_list)
    turnover_long_list = parse_int_list(args.turnover_long_list)
    divergence_window_list = parse_int_list(args.divergence_window_list)
    benchmark_list = [x.strip().upper() for x in args.benchmark_list.split(",") if x.strip()]
    benchmark_ma_list = parse_int_list(args.benchmark_ma_list)

    param_combos = build_variant_param_combos(
        alpha_variant_list, reversal_window_list, vol_window_list,
        turnover_short_list, turnover_long_list, divergence_window_list,
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

    # 预加载基准数据（使用 v7 的 prepare_benchmark_regime）
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
                CFG.compute_signals_fn,
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
                CFG.compute_signals_fn,
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

    tag = build_batch_tag(
        CFG,
        alpha_variant_list, reversal_window_list, vol_window_list,
        turnover_short_list, turnover_long_list, divergence_window_list,
        benchmark_list, benchmark_ma_list, args.sample_mode,
    )

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = BACKTEST_DIR / f"batch_alpha_v7_summary_{tag}.csv"
    top50_path = BACKTEST_DIR / f"batch_alpha_v7_top50_{tag}.csv"
    skipped_path = BACKTEST_DIR / f"batch_alpha_v7_skipped_{tag}.csv"
    errors_path = BACKTEST_DIR / f"batch_alpha_v7_errors_{tag}.csv"

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

    if not summary_df.empty:
        display_cols = [
            "symbol", "alpha_variant", "reversal_window", "vol_window",
            "turnover_short", "turnover_long", "divergence_window",
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
