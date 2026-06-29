# -*- coding: utf-8 -*-
"""
validate_alpha_v7_robustness.py

Alpha v7 稳健性验证模块：判断 full-run walk-forward 结果是否足够稳健，
能否进入 portfolio backtest 阶段。

输入：walk-forward CSV（portfolio_daily, portfolio_period_summary, test_detail, selected_by_year）
输出：10 个 CSV + 1 个 TXT 报告 + 可选 PNG 图表

运行示例：
python scripts/validate_alpha_v7_robustness.py --input-tag <tag> --run-id exp007_alpha_v7_full --no-png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.validation import resolve_path, parse_list  # noqa: E402
from scripts.common.wf_robustness_shared import (  # noqa: E402
    WFRobustnessConfig,
    make_v7_config,
    load_input_files,
    load_benchmark_data,
    build_scenarios,
    build_benchmark_comparison,
    build_variant_stability,
    build_parameter_stability,
    build_concentration,
    build_train_test_stability,
    evaluate_gates,
    write_report,
    make_equity_chart,
    make_loyo_chart,
    make_yearly_chart,
    make_variant_chart,
    make_concentration_chart,
)
from scripts.common.data_io import safe_to_numeric  # noqa: E402

# Keep PARAM_COLS for backward compatibility (tests import it)
PARAM_COLS = ["reversal_window", "vol_window", "turnover_short", "turnover_long", "divergence_window"]


def main() -> None:
    cfg = make_v7_config(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Alpha v7 稳健性验证")
    parser.add_argument("--input-tag", required=True,
                        help="精确匹配 walk-forward 文件名中的 tag 部分")
    parser.add_argument("--walk-forward-dir", default=str(cfg.default_wf_dir))
    parser.add_argument("--analysis-dir", default=str(cfg.default_analysis_dir))
    parser.add_argument("--output-root", default=str(cfg.default_output_root))
    parser.add_argument("--run-id", default=cfg.default_run_id)
    parser.add_argument("--benchmarks", default="000300.SH,000905.SH")
    parser.add_argument("--exclude-year", default="2025",
                        help="排除年份，逗号分隔")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    wf_dir = resolve_path(args.walk_forward_dir)
    output_root = resolve_path(args.output_root)
    benchmarks = parse_list(args.benchmarks)
    exclude_years = [int(y) for y in parse_list(args.exclude_year, upper=False)]

    output_dir = output_root / f"alpha_v7_robustness_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {wf_dir} ...")
    data = load_input_files(wf_dir, args.input_tag, cfg.file_prefix)
    daily = data["portfolio_daily"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = safe_to_numeric(daily, ["portfolio_ret", "equity", "test_year"])
    daily["test_year"] = daily["test_year"].astype(int)

    period_summary = data["portfolio_period_summary"]
    detail = data["test_detail"]
    selected = data["selected_by_year"]

    all_years = sorted(daily["test_year"].unique())
    print(f"  Test years: {all_years}")
    print(f"  Trading days: {len(daily)}")

    # Load benchmarks
    print("Loading benchmark data ...")
    benchmarks_data = load_benchmark_data(benchmarks, PROJECT_ROOT)

    # Build analyses
    print("Building robustness analyses ...")
    summary_df, loyo_df, exclude_df, yearly_df = build_scenarios(
        daily, period_summary, benchmarks_data, exclude_years)
    benchmark_df = build_benchmark_comparison(daily, benchmarks_data, all_years, exclude_years)
    variant_df = build_variant_stability(detail)
    param_df = build_parameter_stability(detail, cfg.param_cols)
    concentration_df = build_concentration(detail)
    train_test_df = build_train_test_stability(detail)

    # Evaluate gates
    print("Evaluating gates ...")
    gates, decision = evaluate_gates(
        summary_df, loyo_df, variant_df, train_test_df, concentration_df, yearly_df,
        exclude_years)

    # Save CSVs
    print("Saving outputs ...")
    csv_tables = {
        "alpha_v7_robustness_summary": summary_df,
        "alpha_v7_robustness_leave_one_year_out": loyo_df,
        "alpha_v7_robustness_exclude_years": exclude_df,
        "alpha_v7_robustness_yearly_contribution": yearly_df,
        "alpha_v7_robustness_benchmark": benchmark_df,
        "alpha_v7_robustness_variant": variant_df,
        "alpha_v7_robustness_parameters": param_df,
        "alpha_v7_robustness_concentration": concentration_df,
        "alpha_v7_robustness_train_test": train_test_df,
        "alpha_v7_robustness_decision": pd.DataFrame(gates),
    }
    for name, df in csv_tables.items():
        if df is not None and not df.empty:
            path = output_dir / f"{name}.csv"
            df.to_csv(path, encoding="utf-8-sig", index=False)
            print(f"  Saved: {path.name}")

    # Write report
    report_path = write_report(
        output_dir, cfg, args, summary_df, loyo_df, exclude_df, yearly_df,
        benchmark_df, variant_df, param_df, concentration_df, train_test_df,
        gates, decision)
    print(f"  Report: {report_path.name}")

    # Charts
    if not args.no_png:
        print("Generating charts ...")
        chart_funcs = [
            ("equity_full_vs_exclude", lambda: make_equity_chart(daily, exclude_years, output_dir, cfg.display_name)),
            ("leave_one_year_out_returns", lambda: make_loyo_chart(loyo_df, output_dir, cfg.display_name)),
            ("yearly_contribution", lambda: make_yearly_chart(yearly_df, output_dir, cfg.display_name)),
            ("variant_stability", lambda: make_variant_chart(variant_df, output_dir, cfg.display_name)),
            ("contributor_concentration", lambda: make_concentration_chart(concentration_df, output_dir, cfg.display_name)),
        ]
        for name, func in chart_funcs:
            try:
                p = func()
                print(f"  Chart: {p.name}")
            except Exception as e:
                print(f"  [WARNING] Chart {name} failed: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Decision: {decision}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
