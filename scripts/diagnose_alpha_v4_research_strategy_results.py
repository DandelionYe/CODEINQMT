# -*- coding: utf-8 -*-
"""
diagnose_alpha_v4_research_strategy_results.py

诊断 Alpha v4 walk-forward 样本外结果，找出策略弱点并给出改进建议。

输入：analyze_alpha_v4_research_walk_forward_results.py 的输出 + walk-forward 原始 CSV
输出：10 个诊断 CSV + 1 个 TXT 建议报告 + 可选 PNG 图表

运行示例：
python scripts\\diagnose_alpha_v4_research_strategy_results.py --run-id exp004_alpha_v4_smoke --input-tag <tag> --no-png
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.constants import DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.data_io import safe_to_numeric, read_csv_required  # noqa: E402
from scripts.common.metrics import format_pct, format_float  # noqa: E402
from scripts.common.validation import resolve_path, parse_list  # noqa: E402

DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "backtests" / "walk_forward_alpha_v4_research_analysis"
DEFAULT_WF_DIR = PROJECT_ROOT / "backtests" / "walk_forward_alpha_v4_research_csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "backtests" / "strategy_diagnosis"

BENCHMARK_ENTITY_MAP = {
    "000300.SH": "BENCH_000300.SH_CSI300",
    "000905.SH": "BENCH_000905.SH_CSI500",
    "000852.SH": "BENCH_000852.SH_CSI1000",
}
BENCHMARK_SHORT = {v: k for k, v in BENCHMARK_ENTITY_MAP.items()}

OVERFITTING_BLOCKER_RATE = 0.20
BAD_TO_GOOD_CONTRIBUTOR_BLOCKER_RATIO = 2.0




def list_candidate_files(input_dir: Path, kind: str) -> str:
    candidates = sorted(input_dir.glob(f"wf_alpha_v4_stock_*_{kind}.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not candidates:
        return "  (none)"
    return "\n".join(f"  - {p.name}" for p in candidates[:20])


def find_wf_file(
    input_dir: Path,
    market: str,
    portfolio_size: int,
    kind: str,
    file_tag: str = "",
    allow_fallback: bool = False,
) -> Path:
    if file_tag:
        pattern = f"wf_alpha_v4_stock_{file_tag}_{kind}.csv"
        files = sorted(input_dir.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            raise FileNotFoundError(f"未找到精确匹配文件: {input_dir / pattern}")
        return files[0]

    if not allow_fallback:
        raise FileNotFoundError(
            "Alpha v4 diagnosis requires --input-tag to bind an exact walk-forward file group.\n"
            f"Candidate {kind} files:\n{list_candidate_files(input_dir, kind)}"
        )

    files = sorted(input_dir.glob(f"wf_alpha_v4_stock_*top{portfolio_size}_{kind}.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        files = sorted(input_dir.glob(f"wf_alpha_v4_stock_*_{kind}.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"未找到 {kind} 文件: {input_dir}")
    if len(files) > 1:
        print(f"  [WARNING] 找到 {len(files)} 个匹配文件，使用最新: {files[0].name}")
    return files[0]


def load_walk_forward_raw(
    wf_dir: Path,
    markets: list[str],
    portfolio_size: int,
    file_tag: str = "",
    allow_fallback: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_path = find_wf_file(
        wf_dir, markets[0] if len(markets) == 1 else "ALL",
        portfolio_size, "selected_by_year", file_tag, allow_fallback,
    )
    detail_path = find_wf_file(
        wf_dir, markets[0] if len(markets) == 1 else "ALL",
        portfolio_size, "test_detail", file_tag, allow_fallback,
    )

    selected = read_csv_required(selected_path)
    detail = read_csv_required(detail_path)

    numeric_cols = [c for c in detail.columns if c.startswith(("train_", "test_"))]
    detail = safe_to_numeric(detail, numeric_cols)

    return selected, detail


def load_analysis_tables(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    prefix = "alpha_v4_wf_analysis"
    table_names = [
        "overall_comparison", "excess_comparison", "yearly_comparison", "yearly_excess",
        "selected_frequency", "parameter_frequency", "alpha_variant_frequency",
        "benchmark_filter_frequency", "single_stock_contribution", "combined_daily_returns",
    ]
    tables = {}
    for name in table_names:
        path = analysis_dir / f"{prefix}_{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path, encoding="utf-8-sig")
        else:
            tables[name] = pd.DataFrame()
    return tables


def filter_exclude_year(df: pd.DataFrame, exclude_year: int | None, year_col: str = "year") -> pd.DataFrame:
    if exclude_year and year_col in df.columns:
        return df[df[year_col] != exclude_year]
    return df


def build_summary(overall: pd.DataFrame, excess: pd.DataFrame, exclude_year: int | None) -> pd.DataFrame:
    strat = overall[(overall["entity"] == "strategy") & (overall["period"] == "all_years")]
    if strat.empty:
        return pd.DataFrame()
    s = strat.iloc[0]

    row = {
        "total_return": s.get("total_return", np.nan),
        "annual_return": s.get("annual_return", np.nan),
        "max_drawdown": s.get("max_drawdown", np.nan),
        "sharpe": s.get("sharpe", np.nan),
        "annual_volatility": s.get("annual_volatility", np.nan),
        "negative_return": s.get("total_return", 0) < 0,
        "low_sharpe": s.get("sharpe", 0) < 0.3,
        "high_drawdown": s.get("max_drawdown", 0) < -0.25,
    }

    if not excess.empty:
        for _, e in excess[excess["period"] == "all_years"].iterrows():
            bm = e["benchmark"]
            row[f"excess_vs_{bm}"] = e.get("excess_return", np.nan)
            row[f"underperform_{bm}"] = e.get("excess_return", 0) < 0

    return pd.DataFrame([row])


def collect_decision_blockers(
    gap_df: pd.DataFrame,
    bad_contrib: pd.DataFrame | None,
    good_contrib: pd.DataFrame | None,
    alpha_variant_stab: pd.DataFrame | None,
) -> list[str]:
    blockers = []

    if not gap_df.empty and "gap_flag" in gap_df.columns:
        flags = gap_df["gap_flag"].value_counts()
        good_bad = int(flags.get("train_good_test_bad", 0))
        total = int(len(gap_df))
        rate = good_bad / total if total else 0.0
        if rate > OVERFITTING_BLOCKER_RATE:
            blockers.append(
                "severe_overfitting: "
                f"train_good_test_bad={good_bad}/{total} ({rate:.0%}) "
                f"> {OVERFITTING_BLOCKER_RATE:.0%}"
            )

    if alpha_variant_stab is None or alpha_variant_stab.empty:
        blockers.append("no_stable_alpha_variant: alpha_variant_stability_table_empty")
    else:
        stable_av = alpha_variant_stab[
            alpha_variant_stab.get("stability_label", "") == "relatively_stable"
        ]
        if len(stable_av) == 0:
            blockers.append("no_stable_alpha_variant: stable_variant_count=0")

    if bad_contrib is not None and good_contrib is not None:
        bad_count = int(len(bad_contrib))
        good_count = int(len(good_contrib))
        if bad_count > 0 and (
            good_count == 0
            or bad_count > good_count * BAD_TO_GOOD_CONTRIBUTOR_BLOCKER_RATIO
        ):
            blockers.append(
                "bad_contributors_dominant: "
                f"bad={bad_count}, good={good_count}, "
                f"threshold={BAD_TO_GOOD_CONTRIBUTOR_BLOCKER_RATIO:.1f}x"
            )

    return blockers


def build_yearly_weakness(yearly: pd.DataFrame, yearly_excess: pd.DataFrame, exclude_year: int | None) -> pd.DataFrame:
    strat = yearly[yearly["entity"] == "strategy"]
    if strat.empty:
        return pd.DataFrame()
    strat = filter_exclude_year(strat, exclude_year)

    rows = []
    for _, s in strat.iterrows():
        year = int(s["year"])
        row = {
            "year": year,
            "total_return": s.get("total_return", np.nan),
            "max_drawdown": s.get("max_drawdown", np.nan),
            "sharpe": s.get("sharpe", np.nan),
            "lost_money": s.get("total_return", 0) < 0,
            "deep_drawdown": s.get("max_drawdown", 0) < -0.2,
            "negative_sharpe": s.get("sharpe", 0) < 0,
        }

        ye = yearly_excess[yearly_excess["year"] == year] if not yearly_excess.empty else pd.DataFrame()
        if not ye.empty:
            row["underperformed_any_benchmark"] = any(ye["excess_return"] < 0)
            row["benchmark_up_strategy_down"] = any((ye["benchmark_return"] > 0) & (ye["strategy_return"] < 0))

        rows.append(row)

    return pd.DataFrame(rows)


def build_train_test_gap(detail: pd.DataFrame, exclude_year: int | None) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    df = filter_exclude_year(detail, exclude_year)

    gap_df = df.copy()
    if "train_annual_return" in df.columns and "test_total_return" in df.columns:
        gap_df["train_test_gap"] = df["test_total_return"] - df["train_annual_return"]
    if "train_sharpe" in df.columns and "test_sharpe" in df.columns:
        gap_df["sharpe_gap"] = df["test_sharpe"] - df["train_sharpe"]

    def flag(row):
        train_ret = row.get("train_annual_return", np.nan)
        test_ret = row.get("test_total_return", np.nan)
        if np.isnan(train_ret) or np.isnan(test_ret):
            return "unknown"
        if train_ret > 0.05 and test_ret < 0:
            return "train_good_test_bad"
        if train_ret > 0.05 and test_ret > 0:
            return "train_good_test_ok"
        if train_ret < 0 and test_ret > 0:
            return "train_bad_test_good"
        return "neutral"

    gap_df["gap_flag"] = gap_df.apply(flag, axis=1)
    return gap_df


def build_train_test_correlation(gap_df: pd.DataFrame) -> pd.DataFrame:
    if gap_df.empty:
        return pd.DataFrame()

    pairs = [
        ("train_score", "test_total_return"),
        ("train_annual_return", "test_total_return"),
        ("train_sharpe", "test_total_return"),
    ]

    rows = []
    for train_col, test_col in pairs:
        if train_col in gap_df.columns and test_col in gap_df.columns:
            valid = gap_df[[train_col, test_col]].dropna()
            if len(valid) > 5:
                corr = valid[train_col].corr(valid[test_col])
                rows.append({"train_metric": train_col, "test_metric": test_col, "correlation": corr, "n": len(valid)})

    return pd.DataFrame(rows)


def build_contributors(detail: pd.DataFrame, exclude_year: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detail.empty or "symbol" not in detail.columns:
        return pd.DataFrame(), pd.DataFrame()

    df = filter_exclude_year(detail, exclude_year)

    agg = df.groupby("symbol").agg(
        count=("symbol", "count"),
        sum_return=("test_total_return", "sum") if "test_total_return" in df.columns else ("symbol", "count"),
        avg_return=("test_total_return", "mean") if "test_total_return" in df.columns else ("symbol", "count"),
    ).reset_index()

    if "test_total_return" in df.columns:
        agg["win_count"] = df.groupby("symbol")["test_total_return"].apply(lambda x: (x > 0).sum()).values
        agg["win_rate"] = agg["win_count"] / agg["count"]

    bad = agg[(agg["sum_return"] < 0) | ((agg["win_rate"] < 0.5) & (agg["count"] >= 2))].copy()
    bad = bad.sort_values("sum_return", ascending=True)

    good = agg[(agg["sum_return"] > 0) & (agg["win_rate"] >= 0.5)].copy()
    good = good.sort_values("sum_return", ascending=False)

    return bad, good


def build_alpha_variant_stability(detail: pd.DataFrame, exclude_year: int | None) -> pd.DataFrame:
    if detail.empty or "alpha_variant" not in detail.columns:
        return pd.DataFrame()

    df = filter_exclude_year(detail, exclude_year)

    agg = df.groupby("alpha_variant").agg(
        selected_count=("symbol", "count"),
        avg_test_return=("test_total_return", "mean") if "test_total_return" in df.columns else ("symbol", "count"),
    ).reset_index()

    if "test_total_return" in df.columns:
        agg["win_count"] = df.groupby("alpha_variant")["test_total_return"].apply(lambda x: (x > 0).sum()).values
        agg["win_rate"] = agg["win_count"] / agg["selected_count"]

    def label(row):
        if row.get("selected_count", 0) >= 3 and row.get("win_rate", 0) >= 0.5 and row.get("avg_test_return", 0) > 0:
            return "relatively_stable"
        return "unstable_or_weak"

    agg["stability_label"] = agg.apply(label, axis=1)
    return agg.sort_values("avg_test_return", ascending=False)


def build_parameter_stability(detail: pd.DataFrame, exclude_year: int | None) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()

    df = filter_exclude_year(detail, exclude_year)
    param_cols = [c for c in ["momentum_window", "trend_ma", "vol_window", "breakout_window"] if c in df.columns]
    if not param_cols:
        return pd.DataFrame()

    agg = df.groupby(param_cols).agg(
        selected_count=("symbol", "count"),
        avg_test_return=("test_total_return", "mean") if "test_total_return" in df.columns else ("symbol", "count"),
    ).reset_index()

    if "test_total_return" in df.columns:
        agg["win_count"] = df.groupby(param_cols)["test_total_return"].apply(lambda x: (x > 0).sum()).values
        agg["win_rate"] = agg["win_count"] / agg["selected_count"]

    def label(row):
        if row.get("selected_count", 0) >= 3 and row.get("win_rate", 0) >= 0.5 and row.get("avg_test_return", 0) > 0:
            return "relatively_stable"
        return "unstable_or_weak"

    agg["stability_label"] = agg.apply(label, axis=1)
    return agg.sort_values("avg_test_return", ascending=False)


def build_selected_repetition(selected_freq: pd.DataFrame, detail: pd.DataFrame, exclude_year: int | None) -> pd.DataFrame:
    if selected_freq.empty or detail.empty:
        return pd.DataFrame()
    df = filter_exclude_year(detail, exclude_year)
    if "symbol" not in df.columns:
        return pd.DataFrame()

    test_agg = df.groupby("symbol").agg(
        avg_test_return=("test_total_return", "mean") if "test_total_return" in df.columns else ("symbol", "count"),
    ).reset_index()

    merged = selected_freq.merge(test_agg, on="symbol", how="left")
    return merged.sort_values("selected_count", ascending=False)


def make_train_vs_test_scatter(gap_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if gap_df.empty or "train_annual_return" not in gap_df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(gap_df["train_annual_return"], gap_df["test_total_return"], alpha=0.3, s=10)
    lim = max(abs(gap_df["train_annual_return"].max()), abs(gap_df["test_total_return"].max()), 0.1)
    ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5)
    ax.set_xlabel("Train Annual Return")
    ax.set_ylabel("Test Total Return")
    ax.set_title("Alpha v4: Train vs Test Returns")
    plt.tight_layout()
    path = output_dir / "alpha_v4_diagnosis_train_vs_test_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_yearly_excess_heatmap(yearly_weakness: pd.DataFrame, output_dir: Path) -> Path | None:
    if yearly_weakness.empty or "total_return" not in yearly_weakness.columns:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    data = yearly_weakness[["year", "total_return"]].set_index("year")
    ax.bar(data.index.astype(str), data["total_return"])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Alpha v4: Yearly Returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    plt.tight_layout()
    path = output_dir / "alpha_v4_diagnosis_yearly_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_top_draggers_chart(contrib: pd.DataFrame, output_dir: Path, top_n: int) -> Path | None:
    if contrib.empty or "sum_return" not in contrib.columns:
        return None
    top = contrib.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
    ax.barh(top["symbol"], top["sum_return"])
    ax.set_xlabel("Sum Test Return")
    ax.set_title(f"Alpha v4: Worst {top_n} Stock Contributors")
    plt.tight_layout()
    path = output_dir / "alpha_v4_diagnosis_top_draggers.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_outputs(output_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    paths = {}
    prefix = "alpha_v4_diagnosis"
    for name, df in tables.items():
        if df is not None and not df.empty:
            path = output_dir / f"{prefix}_{name}.csv"
            df.to_csv(path, encoding="utf-8-sig", index=False)
            paths[name] = path
    return paths


def write_recommendations(output_dir: Path, args, summary, yearly_weakness, gap_df, gap_corr,
                          bad_contrib, good_contrib, alpha_variant_stab, param_stab, selected_rep,
                          table_paths, plot_paths, exclude_year) -> Path:
    path = output_dir / "alpha_v4_diagnosis_recommendations.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("Alpha v4 Strategy Diagnosis Recommendations\n")
        f.write("=" * 60 + "\n\n")

        f.write("Settings:\n")
        f.write(f"  analysis_dir: {args.analysis_dir}\n")
        f.write(f"  walk_forward_dir: {args.walk_forward_dir}\n")
        f.write(f"  run_id: {args.run_id}\n")
        f.write(f"  exclude_year: {exclude_year}\n\n")

        # Summary
        if not summary.empty:
            s = summary.iloc[0]
            f.write("Overall Summary:\n")
            f.write(f"  total_return: {format_pct(s.get('total_return', np.nan))}\n")
            f.write(f"  annual_return: {format_pct(s.get('annual_return', np.nan))}\n")
            f.write(f"  max_drawdown: {format_pct(s.get('max_drawdown', np.nan))}\n")
            f.write(f"  sharpe: {format_float(s.get('sharpe', np.nan))}\n\n")

            annual = s.get("annual_return", 0)
            sharpe = s.get("sharpe", 0)

            if annual < 0 and sharpe < 0:
                f.write("  *** CRITICAL: 年化收益和夏普均为负，核心信号没有产生 alpha。***\n")
                f.write("  *** 不应进入 portfolio backtest。需要重新设计信号逻辑。***\n\n")
            elif annual < 0.03:
                f.write("  *** CAUTION: 年化收益低于 3%，策略复杂度可能不值得。***\n\n")

        # Yearly weakness
        if not yearly_weakness.empty:
            weak_years = yearly_weakness[yearly_weakness.get("lost_money", False) == True] if "lost_money" in yearly_weakness.columns else pd.DataFrame()
            f.write(f"Yearly Weakness: {len(weak_years)} / {len(yearly_weakness)} years lost money\n\n")

        # Train-test gap
        if not gap_df.empty and "gap_flag" in gap_df.columns:
            flags = gap_df["gap_flag"].value_counts()
            f.write("Train-Test Gap Flags:\n")
            for flag, count in flags.items():
                f.write(f"  {flag}: {count}\n")
            f.write("\n")

            good_bad = flags.get("train_good_test_bad", 0)
            total = len(gap_df)
            if total > 0 and good_bad / total > 0.2:
                f.write(f"  *** OVERFITTING WARNING: {good_bad}/{total} ({good_bad/total:.0%}) train-good-test-bad ***\n\n")

        # Correlation
        if not gap_corr.empty:
            f.write("Train-Test Correlations:\n")
            for _, row in gap_corr.iterrows():
                f.write(f"  {row['train_metric']} vs {row['test_metric']}: r={row['correlation']:.4f} (n={int(row['n'])})\n")
            f.write("\n")

        # Alpha variant stability
        if alpha_variant_stab is not None and not alpha_variant_stab.empty:
            f.write("Alpha Variant Stability:\n")
            for _, row in alpha_variant_stab.iterrows():
                f.write(f"  {row['alpha_variant']}: count={int(row['selected_count'])}, "
                        f"avg_return={row.get('avg_test_return', 0):.4f}, "
                        f"win_rate={row.get('win_rate', 0):.2f}, "
                        f"label={row.get('stability_label', 'unknown')}\n")
            stable = alpha_variant_stab[alpha_variant_stab.get("stability_label", "") == "relatively_stable"]
            if len(stable) == 0:
                f.write("  *** NO STABLE ALPHA VARIANTS: 没有 variant 在样本外表现稳定。***\n")
            f.write("\n")

        # Parameter stability
        if param_stab is not None and not param_stab.empty:
            stable = param_stab[param_stab.get("stability_label", "") == "relatively_stable"]
            f.write(f"Parameter Stability: {len(stable)} / {len(param_stab)} relatively stable\n")
            if len(stable) == 0:
                f.write("  *** NO STABLE PARAMETERS: 策略参数不稳定，可能过拟合。***\n")
            f.write("\n")

        # Contributors
        if bad_contrib is not None and not bad_contrib.empty:
            f.write(f"Bad Contributors: {len(bad_contrib)} stocks\n")
            f.write("Worst 10:\n")
            for _, row in bad_contrib.head(10).iterrows():
                f.write(f"  {row['symbol']}: sum={row.get('sum_return', 0):.4f}, win_rate={row.get('win_rate', 0):.2f}\n")
            f.write("\n")

        if good_contrib is not None and not good_contrib.empty:
            f.write(f"Good Contributors: {len(good_contrib)} stocks\n\n")

        # Decision
        f.write("Decision:\n")
        f.write("-" * 40 + "\n")

        proceed = False
        decision = "revise"
        decision_blockers = collect_decision_blockers(
            gap_df, bad_contrib, good_contrib, alpha_variant_stab
        )

        if not summary.empty:
            s = summary.iloc[0]
            annual = s.get("annual_return", 0)
            sharpe = s.get("sharpe", 0)
            headline_pass = annual > 0.03 and sharpe > 0.3

            if headline_pass and not decision_blockers:
                proceed = True
                decision = "promote_to_portfolio_backtest"
                f.write("  proceed_to_portfolio_backtest: true\n")
                f.write("  decision: promote_to_portfolio_backtest\n")
                f.write("  策略表现可接受，可以考虑推进到 portfolio backtest。\n")
            elif headline_pass:
                decision = "continue_to_robustness_validation"
                f.write("  proceed_to_portfolio_backtest: false\n")
                f.write("  decision: continue_to_robustness_validation\n")
                f.write("  reason: headline metrics pass, but hard risk blockers are present\n")
            elif annual > 0:
                f.write("  proceed_to_portfolio_backtest: false\n")
                f.write("  decision: revise\n")
                f.write("  策略有正收益但不显著，建议继续 revise。\n")
            else:
                f.write("  proceed_to_portfolio_backtest: false\n")
                f.write("  decision: revise\n")
                f.write("  策略表现不佳，建议重新设计信号逻辑。\n")
                f.write("  不要推进到 portfolio_backtest_csv.py。\n")
        else:
            f.write("  proceed_to_portfolio_backtest: false\n")
            f.write("  decision: revise\n")
            f.write("  reason: missing_summary_metrics\n")

        if decision_blockers:
            f.write("  blocking_reasons:\n")
            for blocker in decision_blockers:
                f.write(f"    - {blocker}\n")
        f.write("\n")

        # 主要问题来源
        f.write("Problem Source Analysis:\n")
        f.write("-" * 40 + "\n")

        issues = []
        if alpha_variant_stab is not None and not alpha_variant_stab.empty:
            stable_av = alpha_variant_stab[alpha_variant_stab.get("stability_label", "") == "relatively_stable"]
            if len(stable_av) == 0:
                issues.append("alpha_variant: 没有 variant 在样本外稳定盈利")

        if param_stab is not None and not param_stab.empty:
            stable_params = param_stab[param_stab.get("stability_label", "") == "relatively_stable"]
            if len(stable_params) == 0:
                issues.append("parameter: 参数组合不稳定，可能过拟合")

        if not gap_df.empty and "gap_flag" in gap_df.columns:
            flags = gap_df["gap_flag"].value_counts()
            good_bad = flags.get("train_good_test_bad", 0)
            total = len(gap_df)
            if total > 0 and good_bad / total > 0.2:
                issues.append("train_test_gap: 过拟合严重，train-good-test-bad 比例过高")

        if bad_contrib is not None and good_contrib is not None:
            if len(bad_contrib) > len(good_contrib) * 2:
                issues.append("stock_selection: 坏贡献者远多于好贡献者，选股质量差")

        if not yearly_weakness.empty and "lost_money" in yearly_weakness.columns:
            lost = yearly_weakness[yearly_weakness["lost_money"] == True]
            if len(lost) > len(yearly_weakness) * 0.6:
                issues.append("market_environment: 多数年份亏损，可能受市场环境影响")

        if not issues:
            issues.append("需要进一步分析")

        for issue in issues:
            f.write(f"  - {issue}\n")
        f.write("\n")

        # Output files
        f.write("Output Files:\n")
        for name, p in table_paths.items():
            f.write(f"  {name}: {p.name}\n")
        for name, p in plot_paths.items():
            f.write(f"  {name}: {p.name}\n")

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断 Alpha v4 walk-forward 策略表现")
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--walk-forward-dir", default=str(DEFAULT_WF_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--markets", default="ALL")
    parser.add_argument("--portfolio-size", type=int, default=20)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARK_LIST)
    parser.add_argument("--exclude-year", type=int, default=2026)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--top-n-chart", type=int, default=30)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--input-tag", default="", help="精确匹配 walk-forward 文件名中的 tag 部分")
    parser.add_argument("--allow-fallback", action="store_true", help="未传 --input-tag 时允许使用最近匹配文件")
    args = parser.parse_args()

    analysis_dir = resolve_path(args.analysis_dir)
    wf_dir = resolve_path(args.walk_forward_dir)
    output_root = resolve_path(args.output_root)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"alpha_v4_research_diagnosis_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_year = args.exclude_year if args.exclude_year > 0 else None
    markets = parse_list(args.markets)

    # Load data
    tables = load_analysis_tables(analysis_dir)
    selected_raw, detail_raw = load_walk_forward_raw(
        wf_dir, markets, args.portfolio_size, args.input_tag, args.allow_fallback,
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
    param_stab = build_parameter_stability(detail_raw, exclude_year)
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
    table_paths = save_outputs(output_dir, diag_tables)

    # Charts
    plot_paths = {}
    if not args.no_png:
        try:
            p = make_train_vs_test_scatter(gap_df, output_dir)
            if p:
                plot_paths["train_vs_test_scatter"] = p
        except Exception:
            pass
        try:
            p = make_yearly_excess_heatmap(yearly_weakness, output_dir)
            if p:
                plot_paths["yearly_returns"] = p
        except Exception:
            pass
        try:
            p = make_top_draggers_chart(bad_contrib, output_dir, args.top_n_chart)
            if p:
                plot_paths["top_draggers"] = p
        except Exception:
            pass

    # Report
    report_path = write_recommendations(
        output_dir, args, summary, yearly_weakness, gap_df, gap_corr,
        bad_contrib, good_contrib, alpha_variant_stab, param_stab, selected_rep,
        table_paths, plot_paths, exclude_year)

    print(f"\n诊断完成，输出到: {output_dir}")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
