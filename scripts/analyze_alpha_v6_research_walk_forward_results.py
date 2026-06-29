# -*- coding: utf-8 -*-
"""
analyze_alpha_v6_research_walk_forward_results.py

分析 Alpha v6 walk-forward 样本外结果。

输入：validate_alpha_v6_research_candidates.py 的输出
输出：11 个 CSV + 1 个 TXT 报告 + 可选 PNG 图表

运行示例：
python scripts\\analyze_alpha_v6_research_walk_forward_results.py --input-tag alpha_v6_ALL_stock_ts20150101_fy2021-2025_avmomentum_reversion_blend_adaptive_momentum_multi_timeframe_momentum_volatility_regime_momentum_mom60_120_rev20_mshort60_mlong250_vol60_brk120_bm000300SH_000905SH_000852SH_bma120_250_top20_limitALL --no-png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.constants import DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.validation import resolve_path, parse_list  # noqa: E402
from scripts.common.wf_report_shared import (  # noqa: E402
    WFReportConfig,
    make_v6_config,
    load_walk_forward_group,
    load_benchmark_returns,
    build_combined_returns,
    build_overall_comparison,
    build_excess_comparison,
    build_yearly_comparison,
    build_yearly_excess,
    analyze_selected_frequency,
    analyze_parameter_frequency,
    analyze_alpha_variant_frequency,
    analyze_benchmark_filter_frequency,
    analyze_single_stock_contribution,
    save_tables,
    save_plots,
    write_analysis_report,
    infer_incomplete_year,
)

CFG: WFReportConfig = make_v6_config(PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=CFG.analyze_description)
    parser.add_argument("--input-dir", default=str(CFG.default_input_dir))
    parser.add_argument("--output-dir", default=str(CFG.default_output_dir))
    parser.add_argument("--export-root", default=str(PROJECT_ROOT / "data" / "qmt_export"))
    parser.add_argument("--markets", default="ALL")
    parser.add_argument("--portfolio-size", type=int, default=20)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARK_LIST)
    parser.add_argument("--incomplete-year", type=int, default=2026)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--input-tag", default="", help="精确匹配 walk-forward 文件名中的 tag 部分")
    parser.add_argument("--allow-fallback", action="store_true", help="未传 --input-tag 时允许使用最近匹配文件")
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    export_root = resolve_path(args.export_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = parse_list(args.benchmarks)

    # 加载数据
    markets = parse_list(args.markets) if args.markets != "ALL" else ["ALL"]
    try:
        groups = load_walk_forward_group(
            input_dir,
            markets[0] if len(markets) == 1 else "ALL",
            args.portfolio_size,
            CFG,
            args.input_tag,
            args.allow_fallback,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    daily = groups.get("daily", pd.DataFrame())
    if daily.empty:
        logger.error("未找到 portfolio_daily 数据")
        sys.exit(1)

    # 基准收益
    all_dates = daily["date"].unique()
    bench_returns = load_benchmark_returns(benchmarks, export_root, pd.DatetimeIndex(all_dates))

    # 合并
    combined = build_combined_returns(groups, bench_returns)
    incomplete_year = infer_incomplete_year(combined, args.incomplete_year)

    # 分析表
    overall = build_overall_comparison(combined, incomplete_year)
    excess = build_excess_comparison(overall)
    yearly = build_yearly_comparison(combined, incomplete_year)
    yearly_excess = build_yearly_excess(yearly)
    selected_freq = analyze_selected_frequency(groups)
    param_freq = analyze_parameter_frequency(groups, CFG)
    alpha_variant_freq = analyze_alpha_variant_frequency(groups)
    benchmark_filter_freq, ratio_stats = analyze_benchmark_filter_frequency(groups)
    contribution = analyze_single_stock_contribution(groups)

    # 保存
    table_paths = save_tables(
        output_dir, CFG,
        combined_daily_returns=combined.reset_index(),
        overall_comparison=overall,
        excess_comparison=excess,
        yearly_comparison=yearly,
        yearly_excess=yearly_excess,
        selected_frequency=selected_freq,
        parameter_frequency=param_freq,
        alpha_variant_frequency=alpha_variant_freq,
        benchmark_filter_frequency=benchmark_filter_freq,
        single_stock_contribution=contribution,
    )

    plot_paths = {}
    if not args.no_png:
        plot_paths = save_plots(output_dir, combined, yearly, selected_freq, param_freq, benchmark_filter_freq, CFG)

    # 报告
    report_path = write_analysis_report(
        output_dir, args, groups, benchmarks, incomplete_year,
        overall, excess, yearly, yearly_excess, selected_freq,
        param_freq, alpha_variant_freq, benchmark_filter_freq, ratio_stats, contribution,
        table_paths, plot_paths, CFG,
    )

    logger.info("分析完成，输出到: %s", output_dir)
    logger.info("报告: %s", report_path)


if __name__ == "__main__":
    setup_cli_logging()
    main()
