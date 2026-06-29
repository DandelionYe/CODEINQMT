# -*- coding: utf-8 -*-
"""
diagnose_alpha_v6_research_strategy_results.py

诊断 Alpha v6 walk-forward 样本外结果，找出策略弱点并给出改进建议。

输入：analyze_alpha_v6_research_walk_forward_results.py 的输出 + walk-forward 原始 CSV
输出：10 个诊断 CSV + 1 个 TXT 建议报告 + 可选 PNG 图表

运行示例：
python scripts\\diagnose_alpha_v6_research_strategy_results.py --run-id exp006_alpha_v6_smoke --input-tag <tag> --no-png
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.constants import DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.validation import resolve_path, parse_list  # noqa: E402
from scripts.common.wf_report_shared import (  # noqa: E402
    WFReportConfig,
    make_v6_config,
    load_walk_forward_raw,
    load_analysis_tables,
    build_summary,
    build_yearly_weakness,
    build_train_test_gap,
    build_train_test_correlation,
    build_contributors,
    build_alpha_variant_stability,
    build_parameter_stability,
    build_selected_repetition,
    save_diagnosis_outputs,
    make_train_vs_test_scatter,
    make_yearly_excess_heatmap,
    make_top_draggers_chart,
    write_recommendations,
    build_signal_quality_section,
)

CFG: WFReportConfig = make_v6_config(PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=CFG.diagnose_description)
    parser.add_argument("--analysis-dir", default=str(CFG.default_analysis_dir))
    parser.add_argument("--walk-forward-dir", default=str(CFG.default_wf_dir))
    parser.add_argument("--output-root", default=str(CFG.default_output_root))
    parser.add_argument("--markets", default="ALL")
    parser.add_argument("--portfolio-size", type=int, default=20)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARK_LIST)
    parser.add_argument("--exclude-year", type=int, default=2026)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--top-n-chart", type=int, default=30)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--input-tag", default="", help="精确匹配 walk-forward 文件名中的 tag 部分")
    parser.add_argument("--allow-fallback", action="store_true", help="未传 --input-tag 时允许使用最近匹配文件")
    parser.add_argument("--signal-eval-dir", default="",
                        help="signal_evaluation 输出目录，用于加载 IC/RankIC 摘要（可选）")
    args = parser.parse_args()

    analysis_dir = resolve_path(args.analysis_dir)
    wf_dir = resolve_path(args.walk_forward_dir)
    output_root = resolve_path(args.output_root)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{CFG.diagnosis_dir_prefix}{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_year = args.exclude_year if args.exclude_year > 0 else None
    markets = parse_list(args.markets)

    # Load data
    tables = load_analysis_tables(analysis_dir, CFG)
    selected_raw, detail_raw = load_walk_forward_raw(
        wf_dir, markets, args.portfolio_size, CFG, args.input_tag, args.allow_fallback,
    )

    # Build diagnoses
    summary = build_summary(tables.get("overall_comparison", pd.DataFrame()),
                            tables.get("excess_comparison", pd.DataFrame()), exclude_year)
    yearly_weakness = build_yearly_weakness(
        tables.get("yearly_comparison", pd.DataFrame()),
        tables.get("yearly_excess", pd.DataFrame()), exclude_year)
    gap_df = build_train_test_gap(detail_raw, exclude_year)
    gap_corr = build_train_test_correlation(gap_df)
    bad_contrib, good_contrib = build_contributors(detail_raw, exclude_year)
    alpha_variant_stab = build_alpha_variant_stability(detail_raw, exclude_year)
    param_stab = build_parameter_stability(detail_raw, exclude_year, CFG)
    selected_rep = build_selected_repetition(
        tables.get("selected_frequency", pd.DataFrame()), detail_raw, exclude_year)

    # Save
    diag_tables = {
        "summary": summary,
        "yearly_weakness": yearly_weakness,
        "train_test_gap": gap_df,
        "train_test_correlation": gap_corr,
        "bad_contributors": bad_contrib,
        "good_contributors": good_contrib,
        "alpha_variant_stability": alpha_variant_stab,
        "parameter_stability": param_stab,
        "selected_repetition": selected_rep,
    }
    table_paths = save_diagnosis_outputs(output_dir, diag_tables, CFG)

    # Charts
    plot_paths = {}
    if not args.no_png:
        try:
            p = make_train_vs_test_scatter(gap_df, output_dir, CFG)
            if p:
                plot_paths["train_vs_test_scatter"] = p
        except Exception:
            pass
        try:
            p = make_yearly_excess_heatmap(yearly_weakness, output_dir, CFG)
            if p:
                plot_paths["yearly_returns"] = p
        except Exception:
            pass
        try:
            p = make_top_draggers_chart(bad_contrib, output_dir, args.top_n_chart, CFG)
            if p:
                plot_paths["top_draggers"] = p
        except Exception:
            pass

    # Signal quality (optional)
    signal_quality = None
    if args.signal_eval_dir:
        signal_eval_dir = resolve_path(args.signal_eval_dir)
        if signal_eval_dir.is_dir():
            signal_quality = build_signal_quality_section(signal_eval_dir)
            if not signal_quality.empty:
                sq_path = output_dir / f"{CFG.diagnosis_output_prefix}_signal_quality.csv"
                signal_quality.to_csv(sq_path, encoding="utf-8-sig", index=False)
                table_paths["signal_quality"] = sq_path
        else:
            print(f"警告: signal-eval-dir 不存在: {signal_eval_dir}")

    # Report
    report_path = write_recommendations(
        output_dir, args, summary, yearly_weakness, gap_df, gap_corr,
        bad_contrib, good_contrib, alpha_variant_stab, param_stab, selected_rep,
        table_paths, plot_paths, exclude_year, CFG, signal_quality=signal_quality)

    print(f"\n诊断完成，输出到: {output_dir}")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
