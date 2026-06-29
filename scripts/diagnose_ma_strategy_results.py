# -*- coding: utf-8 -*-
"""
diagnose_ma_strategy_results.py

用途：
对 MA walk-forward 结果进行策略诊断，判断当前策略问题主要来自：
1. 年度稳定性不足
2. 回撤偏大
3. 相对中证500 / 中证1000 超额不足
4. 训练期表现与样本外表现落差过大
5. 参数不稳定
6. 个别股票拖累明显
7. SZ / SH / ALL 表现差异

本脚本只读取已有结果，不重新跑策略。

默认输入：
backtests/walk_forward_analysis
backtests/walk_forward_ma_csv

默认输出：
backtests/strategy_diagnosis/ma_diagnosis_<run_id>

运行示例：

1. 默认诊断：
python scripts\\diagnose_ma_strategy_results.py

2. 指定 run id，方便记录实验：
python scripts\\diagnose_ma_strategy_results.py --run-id exp001_ma_baseline_diag

3. 不排除 2026：
python scripts\\diagnose_ma_strategy_results.py --exclude-year 0

4. 只诊断 SZ：
python scripts\\diagnose_ma_strategy_results.py --markets SZ
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.constants import DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.data_io import safe_to_numeric, read_csv_required  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.validation import resolve_path, parse_list  # noqa: E402

DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "backtests" / "walk_forward_analysis"
DEFAULT_WF_DIR = PROJECT_ROOT / "backtests" / "walk_forward_ma_csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "backtests" / "strategy_diagnosis"


BENCHMARK_ENTITY_MAP = {
    "000300.SH": "BENCH_000300.SH_CSI300",
    "000905.SH": "BENCH_000905.SH_CSI500",
    "000852.SH": "BENCH_000852.SH_CSI1000",
    "000001.SH": "BENCH_000001.SH_SSE Composite",
    "399001.SZ": "BENCH_399001.SZ_SZ Component",
    "399006.SZ": "BENCH_399006.SZ_ChiNext",
}




def normalize_benchmark_entities(items: list[str]) -> list[str]:
    result = []
    for item in items:
        if item.startswith("BENCH_"):
            result.append(item)
        else:
            result.append(BENCHMARK_ENTITY_MAP.get(item, item))
    return result


def find_wf_file(wf_dir: Path, market: str, portfolio_size: int, kind: str) -> Path:
    pattern = f"wf_ma_stock_{market}_*_top{portfolio_size}_{kind}.csv"
    files = sorted(wf_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"找不到 {market} 的 {kind} 文件，pattern={pattern}")

    if len(files) > 1:
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    return files[0]


def load_walk_forward_raw(
    wf_dir: Path,
    markets: list[str],
    portfolio_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames = []
    detail_frames = []

    for market in markets:
        selected_path = find_wf_file(wf_dir, market, portfolio_size, "selected_by_year")
        detail_path = find_wf_file(wf_dir, market, portfolio_size, "test_detail")

        selected = pd.read_csv(selected_path)
        detail = pd.read_csv(detail_path)

        selected["market_group"] = market
        detail["market_group"] = market
        selected["source_file"] = str(selected_path)
        detail["source_file"] = str(detail_path)

        selected_frames.append(selected)
        detail_frames.append(detail)

    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    numeric_cols = [
        "test_year", "selected_rank", "fast", "slow",
        "train_rows", "train_annual_return", "train_max_drawdown",
        "train_sharpe", "train_trade_count", "train_excess_total_return",
        "train_score",
        "test_rows", "test_total_return", "test_annual_return",
        "test_max_drawdown", "test_sharpe", "test_trade_count",
        "test_excess_total_return", "test_buy_hold_total_return",
    ]

    selected_all = safe_to_numeric(selected_all, numeric_cols)
    detail_all = safe_to_numeric(detail_all, numeric_cols)

    return selected_all, detail_all


def load_analysis_tables(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "overall": analysis_dir / "wf_analysis_overall_comparison.csv",
        "excess": analysis_dir / "wf_analysis_excess_comparison.csv",
        "yearly": analysis_dir / "wf_analysis_yearly_comparison.csv",
        "yearly_excess": analysis_dir / "wf_analysis_yearly_excess.csv",
        "selected_frequency": analysis_dir / "wf_analysis_selected_frequency.csv",
        "parameter_frequency": analysis_dir / "wf_analysis_parameter_frequency.csv",
        "single_stock_contribution": analysis_dir / "wf_analysis_single_stock_contribution.csv",
    }

    tables = {}
    for name, path in files.items():
        tables[name] = read_csv_required(path)

    numeric_map = {
        "overall": ["days", "total_return", "annual_return", "annual_volatility", "max_drawdown", "sharpe", "calmar"],
        "excess": [
            "strategy_total_return", "benchmark_total_return", "excess_total_return",
            "strategy_annual_return", "benchmark_annual_return", "excess_annual_return",
            "strategy_max_drawdown", "benchmark_max_drawdown", "strategy_sharpe", "benchmark_sharpe",
        ],
        "yearly": ["year", "total_return", "annual_return", "max_drawdown", "sharpe"],
        "yearly_excess": [
            "year", "strategy_total_return", "benchmark_total_return", "excess_total_return",
            "strategy_annual_return", "benchmark_annual_return", "excess_annual_return",
            "strategy_max_drawdown", "benchmark_max_drawdown", "strategy_sharpe", "benchmark_sharpe",
        ],
        "selected_frequency": ["selected_count", "avg_rank", "best_rank", "avg_train_annual_return", "avg_train_max_drawdown", "avg_train_sharpe", "avg_train_score"],
        "parameter_frequency": ["fast", "slow", "selected_count", "avg_rank", "avg_train_annual_return", "avg_train_max_drawdown", "avg_train_sharpe", "avg_train_score"],
        "single_stock_contribution": [
            "selected_count", "avg_selected_rank", "avg_test_total_return", "sum_test_total_return",
            "median_test_total_return", "min_test_total_return", "max_test_total_return",
            "avg_test_max_drawdown", "avg_test_sharpe", "avg_test_excess_total_return",
            "win_year_count", "avg_train_score", "win_rate",
        ],
    }

    for name, cols in numeric_map.items():
        tables[name] = safe_to_numeric(tables[name], cols)

    return tables


def filter_main_period(df: pd.DataFrame, year_col: str, exclude_year: int | None) -> pd.DataFrame:
    if exclude_year is None or exclude_year == 0 or year_col not in df.columns:
        return df.copy()
    return df[df[year_col] != exclude_year].copy()


def build_summary(
    overall: pd.DataFrame,
    excess: pd.DataFrame,
    benchmarks: list[str],
) -> pd.DataFrame:
    rows = []

    strategies = overall[overall["entity_type"] == "strategy"].copy()

    for _, s in strategies.iterrows():
        period = s["period"]
        strategy = s["entity"]

        row = {
            "period": period,
            "strategy": strategy,
            "total_return": s.get("total_return", np.nan),
            "annual_return": s.get("annual_return", np.nan),
            "annual_volatility": s.get("annual_volatility", np.nan),
            "max_drawdown": s.get("max_drawdown", np.nan),
            "sharpe": s.get("sharpe", np.nan),
            "calmar": s.get("calmar", np.nan),
        }

        for bench in benchmarks:
            m = excess[
                (excess["period"] == period)
                & (excess["strategy"] == strategy)
                & (excess["benchmark"] == bench)
            ]

            suffix = bench.replace("BENCH_", "").replace(".", "_").replace(" ", "_").replace("-", "_")

            if m.empty:
                row[f"excess_total_return_vs_{suffix}"] = np.nan
                row[f"excess_annual_return_vs_{suffix}"] = np.nan
            else:
                item = m.iloc[0]
                row[f"excess_total_return_vs_{suffix}"] = item.get("excess_total_return", np.nan)
                row[f"excess_annual_return_vs_{suffix}"] = item.get("excess_annual_return", np.nan)

        row["diagnosis_level"] = classify_strategy_level(row)
        rows.append(row)

    result = pd.DataFrame(rows)
    return result.sort_values(["period", "annual_return"], ascending=[True, False])


def classify_strategy_level(row: dict) -> str:
    annual = row.get("annual_return", np.nan)
    mdd = row.get("max_drawdown", np.nan)
    sharpe = row.get("sharpe", np.nan)

    if pd.isna(annual) or pd.isna(mdd) or pd.isna(sharpe):
        return "unknown"

    if annual > 0.08 and mdd > -0.30 and sharpe > 0.8:
        return "strong_candidate"

    if annual > 0.03 and mdd > -0.35 and sharpe > 0.4:
        return "needs_validation"

    if annual > 0 and mdd > -0.40:
        return "weak_positive"

    return "not_pass"


def build_yearly_weakness(
    yearly: pd.DataFrame,
    yearly_excess: pd.DataFrame,
    benchmarks: list[str],
    exclude_year: int | None,
) -> pd.DataFrame:
    y = filter_main_period(yearly, "year", exclude_year)
    ye = filter_main_period(yearly_excess, "year", exclude_year)

    ye = ye[ye["benchmark"].isin(benchmarks)].copy()

    if ye.empty:
        return pd.DataFrame()

    beat_stats = ye.groupby(["strategy", "year"]).agg(
        benchmark_count=("benchmark", "size"),
        beat_count=("beat_benchmark", "sum"),
        avg_excess_total_return=("excess_total_return", "mean"),
        median_excess_total_return=("excess_total_return", "median"),
        worst_excess_total_return=("excess_total_return", "min"),
        avg_excess_annual_return=("excess_annual_return", "mean"),
    ).reset_index()

    beat_stats["beat_rate"] = beat_stats["beat_count"] / beat_stats["benchmark_count"]

    strategy_yearly = y[y["entity_type"] == "strategy"].copy()
    strategy_yearly = strategy_yearly.rename(
        columns={
            "entity": "strategy",
            "total_return": "strategy_total_return",
            "annual_return": "strategy_annual_return",
            "max_drawdown": "strategy_max_drawdown",
            "sharpe": "strategy_sharpe",
        }
    )

    cols = [
        "year", "strategy", "strategy_total_return", "strategy_annual_return",
        "strategy_max_drawdown", "strategy_sharpe",
    ]
    merged = beat_stats.merge(strategy_yearly[cols], on=["strategy", "year"], how="left")

    merged["negative_year"] = merged["strategy_total_return"] < 0
    merged["weakness_score"] = (
        -merged["avg_excess_total_return"].fillna(0)
        + np.maximum(-merged["strategy_total_return"].fillna(0), 0)
        + np.maximum(-merged["strategy_sharpe"].fillna(0), 0) * 0.05
    )

    def reason(row: pd.Series) -> str:
        reasons = []
        if row["strategy_total_return"] < 0:
            reasons.append("negative_return")
        if row["beat_rate"] < 0.5:
            reasons.append("underperform_benchmarks")
        if row["strategy_max_drawdown"] < -0.20:
            reasons.append("deep_drawdown")
        if row["strategy_sharpe"] < 0:
            reasons.append("negative_sharpe")
        return ",".join(reasons) if reasons else "acceptable"

    merged["weakness_reason"] = merged.apply(reason, axis=1)

    return merged.sort_values(["weakness_score", "year"], ascending=[False, True])


def build_train_test_gap(
    detail: pd.DataFrame,
    exclude_year: int | None,
) -> pd.DataFrame:
    d = filter_main_period(detail, "test_year", exclude_year)

    if d.empty:
        return d

    d = d.copy()
    d["annual_return_gap"] = d["test_annual_return"] - d["train_annual_return"]
    d["total_return_gap"] = d["test_total_return"] - d.get("train_excess_total_return", 0)
    d["sharpe_gap"] = d["test_sharpe"] - d["train_sharpe"]
    d["drawdown_gap"] = d["test_max_drawdown"] - d["train_max_drawdown"]

    d["is_test_loss"] = d["test_total_return"] < 0
    d["is_train_good_test_bad"] = (
        (d["train_annual_return"] > 0.10)
        & (d["train_sharpe"] > 0.6)
        & (d["test_total_return"] < 0)
    )

    keep_cols = [
        "market_group", "test_year", "symbol", "selected_rank",
        "fast", "slow",
        "train_annual_return", "test_annual_return", "annual_return_gap",
        "train_sharpe", "test_sharpe", "sharpe_gap",
        "train_max_drawdown", "test_max_drawdown", "drawdown_gap",
        "train_excess_total_return", "test_excess_total_return",
        "test_total_return", "test_trade_count",
        "is_test_loss", "is_train_good_test_bad",
    ]
    keep_cols = [c for c in keep_cols if c in d.columns]

    return d[keep_cols].sort_values(["annual_return_gap", "test_total_return"], ascending=[True, True])


def build_bad_contributors(
    detail: pd.DataFrame,
    exclude_year: int | None,
) -> pd.DataFrame:
    d = filter_main_period(detail, "test_year", exclude_year)

    if d.empty:
        return pd.DataFrame()

    result = d.groupby(["market_group", "symbol"]).agg(
        selected_count=("symbol", "size"),
        years=("test_year", lambda x: ",".join(map(str, sorted(pd.Series(x).dropna().astype(int).unique())))),
        avg_selected_rank=("selected_rank", "mean"),
        avg_train_annual_return=("train_annual_return", "mean"),
        avg_train_sharpe=("train_sharpe", "mean"),
        avg_train_score=("train_score", "mean"),
        avg_test_total_return=("test_total_return", "mean"),
        sum_test_total_return=("test_total_return", "sum"),
        median_test_total_return=("test_total_return", "median"),
        min_test_total_return=("test_total_return", "min"),
        max_test_total_return=("test_total_return", "max"),
        avg_test_max_drawdown=("test_max_drawdown", "mean"),
        avg_test_sharpe=("test_sharpe", "mean"),
        avg_test_excess_total_return=("test_excess_total_return", "mean"),
        win_year_count=("test_total_return", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
    ).reset_index()

    result["win_rate"] = result["win_year_count"] / result["selected_count"]

    def bad_reason(row: pd.Series) -> str:
        reasons = []
        if row["sum_test_total_return"] < 0:
            reasons.append("negative_total_contribution")
        if row["win_rate"] < 0.5:
            reasons.append("low_win_rate")
        if row["avg_test_max_drawdown"] < -0.20:
            reasons.append("deep_avg_drawdown")
        if row["avg_test_excess_total_return"] < 0:
            reasons.append("negative_avg_excess")
        return ",".join(reasons) if reasons else "not_bad"

    result["bad_reason"] = result.apply(bad_reason, axis=1)

    return result.sort_values(["sum_test_total_return", "avg_test_total_return"], ascending=[True, True])


def build_parameter_stability(
    detail: pd.DataFrame,
    exclude_year: int | None,
) -> pd.DataFrame:
    d = filter_main_period(detail, "test_year", exclude_year)

    if d.empty:
        return pd.DataFrame()

    d = d.copy()
    d["param"] = d["fast"].astype(int).astype(str) + "/" + d["slow"].astype(int).astype(str)
    d["annual_return_gap"] = d["test_annual_return"] - d["train_annual_return"]

    result = d.groupby(["market_group", "fast", "slow", "param"]).agg(
        selected_count=("param", "size"),
        years=("test_year", lambda x: ",".join(map(str, sorted(pd.Series(x).dropna().astype(int).unique())))),
        avg_selected_rank=("selected_rank", "mean"),
        avg_train_annual_return=("train_annual_return", "mean"),
        avg_train_sharpe=("train_sharpe", "mean"),
        avg_train_score=("train_score", "mean"),
        avg_test_total_return=("test_total_return", "mean"),
        median_test_total_return=("test_total_return", "median"),
        sum_test_total_return=("test_total_return", "sum"),
        avg_test_sharpe=("test_sharpe", "mean"),
        avg_test_max_drawdown=("test_max_drawdown", "mean"),
        avg_test_excess_total_return=("test_excess_total_return", "mean"),
        avg_annual_return_gap=("annual_return_gap", "mean"),
        win_year_count=("test_total_return", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
    ).reset_index()

    result["win_rate"] = result["win_year_count"] / result["selected_count"]

    result["stability_label"] = np.where(
        (result["selected_count"] >= 3) & (result["win_rate"] >= 0.5) & (result["avg_test_total_return"] > 0),
        "relatively_stable",
        "unstable_or_weak",
    )

    return result.sort_values(
        ["market_group", "stability_label", "sum_test_total_return"],
        ascending=[True, True, False],
    )


def build_selected_repetition(
    selected: pd.DataFrame,
    detail: pd.DataFrame,
    exclude_year: int | None,
) -> pd.DataFrame:
    s = filter_main_period(selected, "test_year", exclude_year)
    d = filter_main_period(detail, "test_year", exclude_year)

    if s.empty:
        return pd.DataFrame()

    selected_grouped = s.groupby(["market_group", "symbol"]).agg(
        selected_count=("symbol", "size"),
        years=("test_year", lambda x: ",".join(map(str, sorted(pd.Series(x).dropna().astype(int).unique())))),
        avg_rank=("selected_rank", "mean"),
        best_rank=("selected_rank", "min"),
        avg_train_annual_return=("train_annual_return", "mean"),
        avg_train_max_drawdown=("train_max_drawdown", "mean"),
        avg_train_sharpe=("train_sharpe", "mean"),
        avg_train_score=("train_score", "mean"),
    ).reset_index()

    if not d.empty:
        contrib = d.groupby(["market_group", "symbol"]).agg(
            avg_test_total_return=("test_total_return", "mean"),
            sum_test_total_return=("test_total_return", "sum"),
            avg_test_sharpe=("test_sharpe", "mean"),
            avg_test_excess_total_return=("test_excess_total_return", "mean"),
            win_year_count=("test_total_return", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
        ).reset_index()

        result = selected_grouped.merge(contrib, on=["market_group", "symbol"], how="left")
        result["win_rate"] = result["win_year_count"] / result["selected_count"]
    else:
        result = selected_grouped

    return result.sort_values(["selected_count", "avg_train_score"], ascending=[False, False])


def make_train_vs_test_scatter(train_test_gap: pd.DataFrame, output_dir: Path) -> Path | None:
    if train_test_gap.empty:
        return None

    path = output_dir / "ma_diagnosis_train_vs_test_scatter.png"

    plt.figure(figsize=(10, 7))

    for market, df in train_test_gap.groupby("market_group"):
        plt.scatter(
            df["train_annual_return"],
            df["test_annual_return"],
            alpha=0.65,
            label=market,
        )

    min_val = min(train_test_gap["train_annual_return"].min(), train_test_gap["test_annual_return"].min())
    max_val = max(train_test_gap["train_annual_return"].max(), train_test_gap["test_annual_return"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="train = test")

    plt.title("Train Annual Return vs Test Annual Return")
    plt.xlabel("Train Annual Return")
    plt.ylabel("Test Annual Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def make_yearly_excess_heatmap(
    yearly_weakness: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if yearly_weakness.empty:
        return None

    path = output_dir / "ma_diagnosis_yearly_excess_heatmap.png"

    pivot = yearly_weakness.pivot_table(
        index="strategy",
        columns="year",
        values="avg_excess_total_return",
        aggfunc="mean",
    ).sort_index()

    if pivot.empty:
        return None

    plt.figure(figsize=(12, 5))
    data = pivot.values

    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label="Avg Excess Total Return")

    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if not np.isnan(value):
                plt.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=8)

    plt.title("Yearly Avg Excess Return vs Benchmarks")
    plt.xlabel("Year")
    plt.ylabel("Strategy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def make_top_draggers_chart(
    bad_contributors: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> Path | None:
    if bad_contributors.empty:
        return None

    path = output_dir / "ma_diagnosis_top_draggers.png"

    tmp = bad_contributors.sort_values("sum_test_total_return", ascending=True).head(top_n).copy()
    tmp["label"] = tmp["market_group"].astype(str) + " " + tmp["symbol"].astype(str)

    plt.figure(figsize=(12, 8))
    plt.barh(tmp["label"], tmp["sum_test_total_return"])
    plt.title(f"Top {top_n} Draggers by Sum Test Total Return")
    plt.xlabel("Sum Test Total Return")
    plt.ylabel("Market / Symbol")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def save_outputs(
    output_dir: Path,
    summary: pd.DataFrame,
    yearly_weakness: pd.DataFrame,
    train_test_gap: pd.DataFrame,
    bad_contributors: pd.DataFrame,
    parameter_stability: pd.DataFrame,
    selected_repetition: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "summary": output_dir / "ma_diagnosis_summary.csv",
        "yearly_weakness": output_dir / "ma_diagnosis_yearly_weakness.csv",
        "train_test_gap": output_dir / "ma_diagnosis_train_test_gap.csv",
        "bad_contributors": output_dir / "ma_diagnosis_bad_contributors.csv",
        "parameter_stability": output_dir / "ma_diagnosis_parameter_stability.csv",
        "selected_repetition": output_dir / "ma_diagnosis_selected_repetition.csv",
    }

    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    yearly_weakness.to_csv(paths["yearly_weakness"], index=False, encoding="utf-8-sig")
    train_test_gap.to_csv(paths["train_test_gap"], index=False, encoding="utf-8-sig")
    bad_contributors.to_csv(paths["bad_contributors"], index=False, encoding="utf-8-sig")
    parameter_stability.to_csv(paths["parameter_stability"], index=False, encoding="utf-8-sig")
    selected_repetition.to_csv(paths["selected_repetition"], index=False, encoding="utf-8-sig")

    return paths


def write_recommendations(
    output_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    yearly_weakness: pd.DataFrame,
    train_test_gap: pd.DataFrame,
    bad_contributors: pd.DataFrame,
    parameter_stability: pd.DataFrame,
    selected_repetition: pd.DataFrame,
    output_paths: dict[str, Path],
    plot_paths: dict[str, Path | None],
) -> Path:
    path = output_dir / "ma_diagnosis_recommendations.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("MA 策略诊断与改进建议\n")
        f.write("=" * 80 + "\n\n")

        f.write("一、诊断设置\n")
        f.write("-" * 80 + "\n")
        f.write(f"analysis_dir: {args.analysis_dir}\n")
        f.write(f"walk_forward_dir: {args.walk_forward_dir}\n")
        f.write(f"markets: {args.markets}\n")
        f.write(f"portfolio_size: {args.portfolio_size}\n")
        f.write(f"benchmarks: {args.benchmarks}\n")
        f.write(f"exclude_year: {args.exclude_year}\n")
        f.write("说明：本模块只读取已有结果，不重新回测。\n\n")

        f.write("二、总体诊断\n")
        f.write("-" * 80 + "\n")
        if not summary.empty:
            f.write(summary.to_string(index=False))
            f.write("\n\n")

            main_period = "exclude_incomplete_2026" if args.exclude_year == 2026 else summary["period"].iloc[0]
            s_main = summary[summary["period"] == main_period].copy()

            if s_main.empty:
                s_main = summary.copy()

            best = s_main.sort_values("annual_return", ascending=False).iloc[0]
            f.write(
                f"主诊断区间内，年化收益最高的是 {best['strategy']}，"
                f"年化收益 {best['annual_return']:.2%}，"
                f"最大回撤 {best['max_drawdown']:.2%}，"
                f"夏普 {best['sharpe']:.4f}。\n"
            )

            sz = s_main[s_main["strategy"] == "STRATEGY_SZ"]
            all_m = s_main[s_main["strategy"] == "STRATEGY_ALL"]

            if not sz.empty and not all_m.empty:
                sz_ret = float(sz.iloc[0]["annual_return"])
                all_ret = float(all_m.iloc[0]["annual_return"])
                if sz_ret > all_ret:
                    f.write("诊断：SZ 明显好于 ALL，说明全市场股票池可能引入了更多噪音标的。\n")
                    f.write("建议：v2 策略应加入股票池过滤，至少区分 SZ / SH / ALL，不宜默认全市场混合。\n")

        f.write("\n三、年度稳定性诊断\n")
        f.write("-" * 80 + "\n")
        if not yearly_weakness.empty:
            f.write(yearly_weakness.head(30).to_string(index=False))
            f.write("\n\n")

            weak_years = yearly_weakness[
                yearly_weakness["weakness_reason"].astype(str).str.contains("negative_return|underperform_benchmarks", na=False)
            ]

            if not weak_years.empty:
                f.write("诊断：存在负收益或跑输基准的年度，当前 MA 策略年度稳定性不足。\n")
                f.write("建议：v2 优先加入大盘趋势过滤，候选过滤指数包括 000300.SH、000905.SH、000852.SH。\n")
        else:
            f.write("无年度弱点数据。\n")

        f.write("\n四、训练期与样本外落差诊断\n")
        f.write("-" * 80 + "\n")
        if not train_test_gap.empty:
            f.write(train_test_gap.head(30).to_string(index=False))
            f.write("\n\n")

            bad_gap_rate = float(train_test_gap["is_train_good_test_bad"].mean()) if "is_train_good_test_bad" in train_test_gap.columns else 0.0
            f.write(f"训练好但样本外为负的比例：{bad_gap_rate:.2%}\n")

            if bad_gap_rate > 0.20:
                f.write("诊断：训练期强势股在样本外失效较多，存在趋势延续不足或均值回归问题。\n")
                f.write("建议：v2 不应只按训练期收益排序，应加入近期动量、回撤恢复、成交量确认等过滤。\n")
        else:
            f.write("无训练/测试落差数据。\n")

        f.write("\n五、拖累个股诊断\n")
        f.write("-" * 80 + "\n")
        if not bad_contributors.empty:
            f.write(bad_contributors.head(30).to_string(index=False))
            f.write("\n\n")
            f.write("建议：v2 可加入个股级风控，例如训练期最大回撤更严格、低胜率标的剔除、测试前近端趋势确认。\n")
        else:
            f.write("无拖累个股数据。\n")

        f.write("\n六、参数稳定性诊断\n")
        f.write("-" * 80 + "\n")
        if not parameter_stability.empty:
            f.write(parameter_stability.to_string(index=False))
            f.write("\n\n")

            stable_params = parameter_stability[parameter_stability["stability_label"] == "relatively_stable"]
            if stable_params.empty:
                f.write("诊断：参数组合稳定性不足，没有明显稳定参数。\n")
                f.write("建议：v2 应降低参数自由度，避免过度依赖训练窗口内的最优参数。\n")
            else:
                f.write("诊断：存在相对稳定参数组合，v2 可优先围绕这些参数做受限测试。\n")
        else:
            f.write("无参数稳定性数据。\n")

        f.write("\n七、入选重复度诊断\n")
        f.write("-" * 80 + "\n")
        if not selected_repetition.empty:
            f.write(selected_repetition.head(30).to_string(index=False))
            f.write("\n\n")
            f.write("说明：重复入选股票既可能代表稳定趋势特征，也可能代表策略对少数股票依赖过高。\n")
        else:
            f.write("无入选重复度数据。\n")

        f.write("\n八、v2 策略建议优先级\n")
        f.write("-" * 80 + "\n")
        f.write("1. 先加入大盘趋势过滤：分别测试 000300.SH、000905.SH、000852.SH。\n")
        f.write("2. 加入股票池过滤：不要默认 ALL，全市场结果弱于 SZ 时，应区分市场池。\n")
        f.write("3. 加入风险控制：限制训练期最大回撤更深的标的，或加入单股止损/组合回撤控制。\n")
        f.write("4. 降低参数过拟合：优先测试表现相对稳定的参数，而不是每期完全自由选最优参数。\n")
        f.write("5. 暂缓行业约束：当前本地没有行业数据，后续可通过 QMT API 或板块数据补充行业分类。\n")

        f.write("\n九、输出文件\n")
        f.write("-" * 80 + "\n")
        for name, p in output_paths.items():
            f.write(f"{name}: {p}\n")
        for name, p in plot_paths.items():
            if p is not None:
                f.write(f"{name}: {p}\n")

    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断 MA walk-forward 策略结果")

    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR), help="walk-forward analysis 输出目录")
    parser.add_argument("--walk-forward-dir", default=str(DEFAULT_WF_DIR), help="walk-forward 原始输出目录")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="诊断结果输出根目录")

    parser.add_argument("--markets", default="SZ,SH,ALL", help="诊断市场组合，例如 SZ,SH,ALL")
    parser.add_argument("--portfolio-size", type=int, default=20, help="对应 walk-forward 文件名中的 top20")

    parser.add_argument(
        "--benchmarks",
        default=DEFAULT_BENCHMARK_LIST,
        help="基准列表，例如 000300.SH,000905.SH,000852.SH",
    )

    parser.add_argument("--exclude-year", type=int, default=2026, help="主诊断排除年份。默认排除 2026。传 0 表示不排除")
    parser.add_argument("--run-id", default="", help="自定义诊断 run id；为空则自动生成时间戳")
    parser.add_argument("--top-n-chart", type=int, default=30, help="图中展示的 Top N 数量")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    analysis_dir = resolve_path(args.analysis_dir)
    wf_dir = resolve_path(args.walk_forward_dir)
    output_root = resolve_path(args.output_root)

    run_id = args.run_id.strip()
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = output_root / f"ma_diagnosis_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    markets = parse_list(args.markets)
    benchmark_entities = normalize_benchmark_entities(parse_list(args.benchmarks))
    exclude_year = None if args.exclude_year == 0 else args.exclude_year

    logger.info("MA 策略诊断配置：")
    logger.info("analysis_dir: %s", analysis_dir)
    logger.info("walk_forward_dir: %s", wf_dir)
    logger.info("output_dir: %s", output_dir)
    logger.info("markets: %s", markets)
    logger.info("benchmarks: %s", benchmark_entities)
    logger.info("exclude_year: %s", exclude_year)

    tables = load_analysis_tables(analysis_dir)
    selected_all, detail_all = load_walk_forward_raw(
        wf_dir=wf_dir,
        markets=markets,
        portfolio_size=args.portfolio_size,
    )

    summary = build_summary(
        overall=tables["overall"],
        excess=tables["excess"],
        benchmarks=benchmark_entities,
    )

    yearly_weakness = build_yearly_weakness(
        yearly=tables["yearly"],
        yearly_excess=tables["yearly_excess"],
        benchmarks=benchmark_entities,
        exclude_year=exclude_year,
    )

    train_test_gap = build_train_test_gap(
        detail=detail_all,
        exclude_year=exclude_year,
    )

    bad_contributors = build_bad_contributors(
        detail=detail_all,
        exclude_year=exclude_year,
    )

    parameter_stability = build_parameter_stability(
        detail=detail_all,
        exclude_year=exclude_year,
    )

    selected_repetition = build_selected_repetition(
        selected=selected_all,
        detail=detail_all,
        exclude_year=exclude_year,
    )

    output_paths = save_outputs(
        output_dir=output_dir,
        summary=summary,
        yearly_weakness=yearly_weakness,
        train_test_gap=train_test_gap,
        bad_contributors=bad_contributors,
        parameter_stability=parameter_stability,
        selected_repetition=selected_repetition,
    )

    plot_paths = {
        "train_vs_test_scatter": make_train_vs_test_scatter(train_test_gap, output_dir),
        "yearly_excess_heatmap": make_yearly_excess_heatmap(yearly_weakness, output_dir),
        "top_draggers": make_top_draggers_chart(bad_contributors, output_dir, args.top_n_chart),
    }

    recommendations_path = write_recommendations(
        output_dir=output_dir,
        args=args,
        summary=summary,
        yearly_weakness=yearly_weakness,
        train_test_gap=train_test_gap,
        bad_contributors=bad_contributors,
        parameter_stability=parameter_stability,
        selected_repetition=selected_repetition,
        output_paths=output_paths,
        plot_paths=plot_paths,
    )

    logger.info("诊断完成。")
    logger.info("输出目录：%s", output_dir)
    logger.info("建议报告：%s", recommendations_path)

    logger.info("主要输出：")
    for name, path in output_paths.items():
        logger.info("%s: %s", name, path)

    logger.info("图片输出：")
    for name, path in plot_paths.items():
        if path is not None:
            logger.info("%s: %s", name, path)

    print("\n诊断摘要 Top：")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception as exc:
        logger.error("程序异常：%s", repr(exc))
        raise