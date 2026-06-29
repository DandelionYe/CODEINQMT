# -*- coding: utf-8 -*-
"""
wf_report_shared.py

Walk-forward 分析与诊断脚本的共享逻辑。

v6/v7 的 analyze 和 diagnose 脚本大量函数完全重复，差异仅在于：
1. 版本字符串（文件前缀、标题、输出目录）
2. 参数列名（momentum vs reversal 参数体系）
3. v7 diagnosis 新增 --input-dir 参数

本模块将所有共享函数抽取为参数化版本，通过 WFReportConfig 注入差异。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.common.constants import (
    TRADING_DAYS_PER_YEAR,
    SQRT_TRADING_DAYS_PER_YEAR,
    DEFAULT_BENCHMARK_LIST,
)
from scripts.common.data_io import safe_to_numeric, read_csv_required
from scripts.common.metrics import format_pct, format_float, max_drawdown_from_equity, calc_metrics_from_daily
from scripts.common.validation import resolve_path, parse_list

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WFReportConfig:
    """版本特定的配置差异点。"""

    # 分析脚本 defaults
    default_input_dir: Path
    default_output_dir: Path

    # 诊断脚本 defaults
    default_analysis_dir: Path
    default_wf_dir: Path
    default_output_root: Path

    # 文件前缀（glob 匹配用）
    file_prefix: str  # e.g. "wf_alpha_v6_stock_" or "wf_alpha_v7_stock_"

    # 输出前缀
    analysis_output_prefix: str  # e.g. "alpha_v6_wf_analysis"
    diagnosis_output_prefix: str  # e.g. "alpha_v6_diagnosis"

    # 参数列名（分析/诊断脚本的 parameter_frequency / parameter_stability）
    param_cols: list[str]

    # 显示名称（图表标题、报告标题等）
    display_name: str  # e.g. "Alpha v6"
    display_suffix: str = ""  # e.g. " (expression layer)" for v7

    # 诊断输出目录前缀
    diagnosis_dir_prefix: str = ""  # e.g. "alpha_v6_research_diagnosis_"

    # argparse 描述
    analyze_description: str = ""
    diagnose_description: str = ""


def make_v6_config(project_root: Path) -> WFReportConfig:
    return WFReportConfig(
        default_input_dir=project_root / "backtests" / "walk_forward_alpha_v6_research_csv",
        default_output_dir=project_root / "backtests" / "walk_forward_alpha_v6_research_analysis",
        default_analysis_dir=project_root / "backtests" / "walk_forward_alpha_v6_research_analysis",
        default_wf_dir=project_root / "backtests" / "walk_forward_alpha_v6_research_csv",
        default_output_root=project_root / "backtests" / "strategy_diagnosis",
        file_prefix="wf_alpha_v6_stock_",
        analysis_output_prefix="alpha_v6_wf_analysis",
        diagnosis_output_prefix="alpha_v6_diagnosis",
        param_cols=["momentum_window", "trend_ma", "vol_window", "breakout_window"],
        display_name="Alpha v6",
        display_suffix="",
        diagnosis_dir_prefix="alpha_v6_research_diagnosis_",
        analyze_description="分析 Alpha v6 walk-forward 样本外结果",
        diagnose_description="诊断 Alpha v6 walk-forward 策略表现",
    )


def make_v7_config(project_root: Path) -> WFReportConfig:
    return WFReportConfig(
        default_input_dir=project_root / "backtests" / "walk_forward_alpha_v7_research_csv",
        default_output_dir=project_root / "backtests" / "walk_forward_alpha_v7_research_analysis",
        default_analysis_dir=project_root / "backtests" / "walk_forward_alpha_v7_research_analysis",
        default_wf_dir=project_root / "backtests" / "walk_forward_alpha_v7_research_csv",
        default_output_root=project_root / "backtests" / "strategy_diagnosis",
        file_prefix="wf_alpha_v7_stock_",
        analysis_output_prefix="alpha_v7_wf_analysis",
        diagnosis_output_prefix="alpha_v7_diagnosis",
        param_cols=["reversal_window", "vol_window", "turnover_short", "turnover_long", "divergence_window"],
        display_name="Alpha v7",
        display_suffix=" (expression layer)",
        diagnosis_dir_prefix="alpha_v7_research_diagnosis_",
        analyze_description="分析 Alpha v7 walk-forward 样本外结果",
        diagnose_description="诊断 Alpha v7 walk-forward 策略表现",
    )


# ---------------------------------------------------------------------------
# Constants (shared between v6/v7)
# ---------------------------------------------------------------------------

BENCHMARK_NAMES = {
    "000300.SH": "CSI300",
    "000905.SH": "CSI500",
    "000852.SH": "CSI1000",
    "000001.SH": "SSE",
    "399001.SZ": "SZComp",
    "399006.SZ": "ChiNext",
}

BENCHMARK_ENTITY_MAP = {
    "000300.SH": "BENCH_000300.SH_CSI300",
    "000905.SH": "BENCH_000905.SH_CSI500",
    "000852.SH": "BENCH_000852.SH_CSI1000",
}
BENCHMARK_SHORT = {v: k for k, v in BENCHMARK_ENTITY_MAP.items()}

OVERFITTING_BLOCKER_RATE = 0.20
BAD_TO_GOOD_CONTRIBUTOR_BLOCKER_RATIO = 2.0


# ---------------------------------------------------------------------------
# Formatting helpers — imported from scripts.common.metrics
# ---------------------------------------------------------------------------


def infer_incomplete_year(combined: pd.DataFrame, user_value: int) -> int | None:
    if user_value and user_value > 0:
        return user_value
    if combined.empty:
        return None
    latest = combined.index.max()
    if latest.month < 12 or (latest.month == 12 and latest.day < 15):
        return latest.year
    return None


# ---------------------------------------------------------------------------
# File finding (parameterized by config.file_prefix)
# ---------------------------------------------------------------------------

def list_candidate_files(input_dir: Path, kind: str, cfg: WFReportConfig) -> str:
    candidates = sorted(
        input_dir.glob(f"{cfg.file_prefix}*_{kind}.csv"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return "  (none)"
    return "\n".join(f"  - {p.name}" for p in candidates[:20])


def find_one_file(
    input_dir: Path,
    market: str,
    portfolio_size: int,
    kind: str,
    cfg: WFReportConfig,
    file_tag: str = "",
    allow_fallback: bool = False,
) -> Path:
    if file_tag:
        pattern = f"{cfg.file_prefix}{file_tag}_{kind}.csv"
        files = sorted(input_dir.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            raise FileNotFoundError(f"未找到精确匹配文件: {input_dir / pattern}")
        return files[0]

    if not allow_fallback:
        raise FileNotFoundError(
            f"{cfg.display_name} diagnosis requires --input-tag to bind an exact walk-forward file group.\n"
            f"Candidate {kind} files:\n{list_candidate_files(input_dir, kind, cfg)}"
        )

    files = sorted(
        input_dir.glob(f"{cfg.file_prefix}*top{portfolio_size}_{kind}.csv"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not files:
        files = sorted(
            input_dir.glob(f"{cfg.file_prefix}*_{kind}.csv"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
    if not files:
        raise FileNotFoundError(f"未找到 {kind} 文件: {input_dir}")
    if len(files) > 1:
        logger.warning("找到 %d 个匹配文件，使用最新: %s", len(files), files[0].name)
    return files[0]


# ---------------------------------------------------------------------------
# Data loading (analyze side)
# ---------------------------------------------------------------------------

def load_walk_forward_group(
    input_dir: Path,
    market: str,
    portfolio_size: int,
    cfg: WFReportConfig,
    file_tag: str = "",
    allow_fallback: bool = False,
) -> dict:
    group = {"market": market, "files": {}}

    kind_map = {
        "daily": "portfolio_daily",
        "period": "portfolio_period_summary",
        "selected": "selected_by_year",
        "detail": "test_detail",
    }

    for key, kind in kind_map.items():
        path = find_one_file(input_dir, market, portfolio_size, kind, cfg, file_tag, allow_fallback)
        group["files"][key] = path
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        group[key] = df

    return group


def load_walk_forward_raw(
    wf_dir: Path,
    markets: list[str],
    portfolio_size: int,
    cfg: WFReportConfig,
    file_tag: str = "",
    allow_fallback: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_path = find_one_file(
        wf_dir, markets[0] if len(markets) == 1 else "ALL",
        portfolio_size, "selected_by_year", cfg, file_tag, allow_fallback,
    )
    detail_path = find_one_file(
        wf_dir, markets[0] if len(markets) == 1 else "ALL",
        portfolio_size, "test_detail", cfg, file_tag, allow_fallback,
    )

    selected = read_csv_required(selected_path)
    detail = read_csv_required(detail_path)

    numeric_cols = [c for c in detail.columns if c.startswith(("train_", "test_"))]
    detail = safe_to_numeric(detail, numeric_cols)

    return selected, detail


# ---------------------------------------------------------------------------
# Benchmark loading (analyze side)
# ---------------------------------------------------------------------------

def symbol_to_qmt_csv(symbol: str, export_root: Path) -> Path:
    from strategies import ma_demo_strategy_csv as ma
    csv_path, _, _, _ = ma.find_csv_for_stock(symbol, export_root)
    return csv_path


def load_benchmark_returns(
    benchmarks: list[str],
    export_root: Path,
    all_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    from strategies import ma_demo_strategy_csv as ma
    returns = {}
    for bm in benchmarks:
        try:
            csv_path = symbol_to_qmt_csv(bm, export_root)
            df = ma.load_qmt_price_csv(csv_path, start="", end="")
            df = df.set_index("date").sort_index()
            ret = df["close"].pct_change()
            ret = ret.reindex(all_dates).ffill()
            name = BENCHMARK_NAMES.get(bm, bm)
            returns[f"BENCH_{bm}_{name}"] = ret
            logger.info("已加载基准: %s (%s)", bm, name)
        except Exception as e:
            logger.warning("加载基准 %s 失败: %s", bm, e)
            continue
    return pd.DataFrame(returns, index=all_dates)


# ---------------------------------------------------------------------------
# Metrics (analyze side)
# ---------------------------------------------------------------------------

def drawdown_series(ret: pd.Series) -> pd.Series:
    equity = (1 + ret).cumprod()
    running_max = equity.cummax()
    return equity / running_max - 1.0


def calc_metrics(ret: pd.Series) -> dict:
    """从日收益率序列计算分析指标。

    委托给 scripts.common.metrics.calc_metrics_from_daily，
    使用 mean_std Sharpe 方法（与原 walk-forward 分析口径一致）。
    """
    m = calc_metrics_from_daily(ret, keys_only=True, sharpe_method="mean_std")
    m["days"] = len(ret)
    return m


# ---------------------------------------------------------------------------
# Analysis builders (analyze side)
# ---------------------------------------------------------------------------

def build_combined_returns(groups: dict, bench_returns: pd.DataFrame) -> pd.DataFrame:
    daily = groups.get("daily", pd.DataFrame())
    if daily.empty:
        return pd.DataFrame()

    daily = daily.set_index("date").sort_index()
    combined = pd.DataFrame(index=daily.index)
    combined["strategy"] = daily["portfolio_ret"]

    for col in bench_returns.columns:
        combined[col] = bench_returns[col].reindex(combined.index).ffill()

    return combined


def build_overall_comparison(combined: pd.DataFrame, incomplete_year: int | None) -> pd.DataFrame:
    rows = []
    for col in combined.columns:
        ret = combined[col].dropna()
        metrics = calc_metrics(ret)
        metrics["entity"] = col
        metrics["period"] = "all_years"
        rows.append(metrics)

        if incomplete_year:
            ret_excl = ret[ret.index.year != incomplete_year]
            if not ret_excl.empty:
                m = calc_metrics(ret_excl)
                m["entity"] = col
                m["period"] = f"exclude_incomplete_{incomplete_year}"
                rows.append(m)

    return pd.DataFrame(rows)


def build_excess_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    strategy_rows = overall[overall["entity"] == "strategy"]
    bench_rows = overall[overall["entity"] != "strategy"]

    rows = []
    for _, strat in strategy_rows.iterrows():
        period = strat["period"]
        for _, bench in bench_rows[bench_rows["period"] == period].iterrows():
            rows.append({
                "period": period,
                "benchmark": bench["entity"],
                "strategy_return": strat["total_return"],
                "benchmark_return": bench["total_return"],
                "excess_return": strat["total_return"] - bench["total_return"],
                "strategy_annual": strat["annual_return"],
                "benchmark_annual": bench["annual_return"],
                "excess_annual": strat["annual_return"] - bench["annual_return"],
                "strategy_sharpe": strat["sharpe"],
                "benchmark_sharpe": bench["sharpe"],
                "strategy_max_drawdown": strat["max_drawdown"],
                "benchmark_max_drawdown": bench["max_drawdown"],
            })
    return pd.DataFrame(rows)


def build_yearly_comparison(combined: pd.DataFrame, incomplete_year: int | None) -> pd.DataFrame:
    rows = []
    for year in combined.index.year.unique():
        year_data = combined[combined.index.year == year]
        for col in combined.columns:
            ret = year_data[col].dropna()
            metrics = calc_metrics(ret)
            metrics["entity"] = col
            metrics["year"] = year
            metrics["incomplete"] = (incomplete_year is not None and year == incomplete_year)
            rows.append(metrics)
    return pd.DataFrame(rows)


def build_yearly_excess(yearly: pd.DataFrame) -> pd.DataFrame:
    strategy_yearly = yearly[yearly["entity"] == "strategy"]
    rows = []
    for year in yearly["year"].unique():
        strat = strategy_yearly[strategy_yearly["year"] == year]
        if strat.empty:
            continue
        strat = strat.iloc[0]
        bench_yearly = yearly[(yearly["year"] == year) & (yearly["entity"] != "strategy")]
        for _, bench in bench_yearly.iterrows():
            rows.append({
                "year": year,
                "benchmark": bench["entity"],
                "strategy_return": strat["total_return"],
                "benchmark_return": bench["total_return"],
                "excess_return": strat["total_return"] - bench["total_return"],
                "beat_benchmark": strat["total_return"] > bench["total_return"],
            })
    return pd.DataFrame(rows)


def analyze_selected_frequency(groups: dict) -> pd.DataFrame:
    selected = groups.get("selected", pd.DataFrame())
    if selected.empty:
        return pd.DataFrame()
    freq = selected.groupby("symbol").agg(
        selected_count=("symbol", "count"),
        avg_rank=("selected_rank", "mean"),
        best_rank=("selected_rank", "min"),
        avg_train_score=("train_score", "mean") if "train_score" in selected.columns else ("symbol", "count"),
    ).reset_index()
    freq = freq.sort_values("selected_count", ascending=False)
    return freq


def analyze_parameter_frequency(groups: dict, cfg: WFReportConfig) -> pd.DataFrame:
    selected = groups.get("selected", pd.DataFrame())
    if selected.empty:
        return pd.DataFrame()

    param_cols = [c for c in cfg.param_cols if c in selected.columns]
    if not param_cols:
        return pd.DataFrame()

    freq = selected.groupby(param_cols).size().reset_index(name="count")
    freq = freq.sort_values("count", ascending=False)
    return freq


def analyze_alpha_variant_frequency(groups: dict) -> pd.DataFrame:
    selected = groups.get("selected", pd.DataFrame())
    if selected.empty or "alpha_variant" not in selected.columns:
        return pd.DataFrame()

    freq = selected.groupby("alpha_variant").agg(
        count=("alpha_variant", "size"),
        avg_train_score=("train_score", "mean") if "train_score" in selected.columns else ("alpha_variant", "size"),
    ).reset_index()
    freq = freq.sort_values("count", ascending=False)
    return freq


def analyze_benchmark_filter_frequency(groups: dict) -> tuple[pd.DataFrame, dict]:
    selected = groups.get("selected", pd.DataFrame())
    if selected.empty:
        return pd.DataFrame(), {}

    freq_cols = [c for c in ["benchmark", "benchmark_ma"] if c in selected.columns]
    if not freq_cols:
        return pd.DataFrame(), {}

    freq = selected.groupby(freq_cols).size().reset_index(name="count")
    freq = freq.sort_values("count", ascending=False)

    detail = groups.get("detail", pd.DataFrame())
    ratio_stats = {}
    if not detail.empty:
        for col in ["test_annual_volatility", "test_max_drawdown"]:
            if col in detail.columns:
                vals = detail[col].dropna()
                if not vals.empty:
                    ratio_stats[col] = {
                        "mean": float(vals.mean()),
                        "median": float(vals.median()),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                    }

    return freq, ratio_stats


def analyze_single_stock_contribution(groups: dict) -> pd.DataFrame:
    detail = groups.get("detail", pd.DataFrame())
    if detail.empty or "symbol" not in detail.columns:
        return pd.DataFrame()

    contrib = detail.groupby("symbol").agg(
        count=("symbol", "count"),
        avg_test_return=("test_total_return", "mean") if "test_total_return" in detail.columns else ("symbol", "count"),
        sum_test_return=("test_total_return", "sum") if "test_total_return" in detail.columns else ("symbol", "count"),
    ).reset_index()

    if "test_total_return" in detail.columns:
        contrib["win_count"] = detail.groupby("symbol")["test_total_return"].apply(lambda x: (x > 0).sum()).values
        contrib["win_rate"] = contrib["win_count"] / contrib["count"]

    contrib = contrib.sort_values("sum_test_return", ascending=True)
    return contrib


# ---------------------------------------------------------------------------
# Save tables (parameterized by prefix)
# ---------------------------------------------------------------------------

def save_tables(output_dir: Path, cfg: WFReportConfig, **tables) -> dict[str, Path]:
    paths = {}
    for name, df in tables.items():
        if df is not None and not df.empty:
            path = output_dir / f"{cfg.analysis_output_prefix}_{name}.csv"
            df.to_csv(path, encoding="utf-8-sig", index=False)
            paths[name] = path
    return paths


# ---------------------------------------------------------------------------
# Plotting (parameterized by display_name)
# ---------------------------------------------------------------------------

def plot_equity_curve(combined: pd.DataFrame, output_dir: Path, cfg: WFReportConfig) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    equity = (1 + combined).cumprod()
    for col in equity.columns:
        ax.plot(equity.index, equity[col], label=col)
    ax.set_title(f"{cfg.display_name} Walk-Forward Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = output_dir / f"{cfg.analysis_output_prefix}_equity_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_drawdown_curve(combined: pd.DataFrame, output_dir: Path, cfg: WFReportConfig) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4))
    dd = drawdown_series(combined["strategy"])
    ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color="red")
    ax.set_title(f"{cfg.display_name} Strategy Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    plt.tight_layout()
    path = output_dir / f"{cfg.analysis_output_prefix}_drawdown_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_yearly_return_bar(yearly: pd.DataFrame, output_dir: Path, cfg: WFReportConfig) -> Path:
    strategy_yearly = yearly[yearly["entity"] == "strategy"]
    if strategy_yearly.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(strategy_yearly["year"].astype(str), strategy_yearly["total_return"])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{cfg.display_name} Yearly Returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    plt.tight_layout()
    path = output_dir / f"{cfg.analysis_output_prefix}_yearly_return_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_plots(output_dir: Path, combined: pd.DataFrame, yearly: pd.DataFrame,
               selected_freq: pd.DataFrame, param_freq: pd.DataFrame,
               benchmark_filter_freq: pd.DataFrame, cfg: WFReportConfig) -> dict[str, Path]:
    paths = {}
    try:
        paths["equity_curve"] = plot_equity_curve(combined, output_dir, cfg)
    except Exception:
        pass
    try:
        paths["drawdown_curve"] = plot_drawdown_curve(combined, output_dir, cfg)
    except Exception:
        pass
    try:
        p = plot_yearly_return_bar(yearly, output_dir, cfg)
        if p:
            paths["yearly_return_bar"] = p
    except Exception:
        pass
    return paths


# ---------------------------------------------------------------------------
# Report writing (analyze side)
# ---------------------------------------------------------------------------

def write_analysis_report(
    output_dir: Path,
    args,
    groups: dict,
    benchmarks: list[str],
    incomplete_year: int | None,
    overall: pd.DataFrame,
    excess: pd.DataFrame,
    yearly: pd.DataFrame,
    yearly_excess: pd.DataFrame,
    selected_freq: pd.DataFrame,
    param_freq: pd.DataFrame,
    alpha_variant_freq: pd.DataFrame,
    benchmark_filter_freq: pd.DataFrame,
    ratio_stats: dict,
    contribution: pd.DataFrame,
    table_paths: dict,
    plot_paths: dict,
    cfg: WFReportConfig,
) -> Path:
    path = output_dir / f"{cfg.analysis_output_prefix}_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{cfg.display_name} Walk-Forward Analysis Report{cfg.display_suffix}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Settings:\n")
        f.write(f"  input_dir: {args.input_dir}\n")
        f.write(f"  output_dir: {args.output_dir}\n")
        f.write(f"  markets: {args.markets}\n")
        f.write(f"  portfolio_size: {args.portfolio_size}\n")
        f.write(f"  benchmarks: {args.benchmarks}\n")
        f.write(f"  incomplete_year: {args.incomplete_year}\n\n")

        # Overall comparison
        strat_overall = overall[(overall["entity"] == "strategy") & (overall["period"] == "all_years")]
        if not strat_overall.empty:
            s = strat_overall.iloc[0]
            f.write("Overall Strategy Performance:\n")
            f.write(f"  total_return: {format_pct(s['total_return'])}\n")
            f.write(f"  annual_return: {format_pct(s['annual_return'])}\n")
            f.write(f"  max_drawdown: {format_pct(s['max_drawdown'])}\n")
            f.write(f"  sharpe: {format_float(s['sharpe'])}\n\n")

        # Excess comparison
        if not excess.empty:
            f.write("Excess vs Benchmarks (all years):\n")
            for _, row in excess[excess["period"] == "all_years"].iterrows():
                f.write(f"  {row['benchmark']}: excess={format_pct(row['excess_return'])}, "
                        f"strategy={format_pct(row['strategy_return'])}, "
                        f"benchmark={format_pct(row['benchmark_return'])}\n")
            f.write("\n")

        # Yearly
        if not yearly.empty:
            f.write("Yearly Strategy Returns:\n")
            for _, row in yearly[(yearly["entity"] == "strategy")].iterrows():
                f.write(f"  {int(row['year'])}: {format_pct(row['total_return'])} "
                        f"(sharpe={format_float(row['sharpe'])}, max_dd={format_pct(row['max_drawdown'])})\n")
            f.write("\n")

        # Alpha variant frequency
        if alpha_variant_freq is not None and not alpha_variant_freq.empty:
            f.write("Alpha Variant Frequency:\n")
            for _, row in alpha_variant_freq.iterrows():
                f.write(f"  {row['alpha_variant']}: {int(row['count'])} selections\n")
            f.write("\n")

        # Selected frequency
        if selected_freq is not None and not selected_freq.empty:
            f.write("Top 30 Selected Stocks:\n")
            for _, row in selected_freq.head(30).iterrows():
                f.write(f"  {row['symbol']}: selected {int(row['selected_count'])} times, "
                        f"avg_rank={row.get('avg_rank', 0):.1f}\n")
            f.write("\n")

        # Parameter frequency
        if param_freq is not None and not param_freq.empty:
            f.write("Parameter Frequency (top 10):\n")
            for _, row in param_freq.head(10).iterrows():
                parts = [f"{c}={row[c]}" for c in param_freq.columns if c != "count"]
                f.write(f"  {', '.join(parts)}: {int(row['count'])}\n")
            f.write("\n")

        # Benchmark filter frequency
        if benchmark_filter_freq is not None and not benchmark_filter_freq.empty:
            f.write("Benchmark Filter Frequency:\n")
            for _, row in benchmark_filter_freq.head(10).iterrows():
                parts = [f"{c}={row[c]}" for c in benchmark_filter_freq.columns if c != "count"]
                f.write(f"  {', '.join(parts)}: {int(row['count'])}\n")
            f.write("\n")

        # Contribution
        if contribution is not None and not contribution.empty:
            f.write("Worst 30 Stock Contributors:\n")
            for _, row in contribution.head(30).iterrows():
                f.write(f"  {row['symbol']}: sum_return={row.get('sum_test_return', 0):.4f}, "
                        f"count={int(row['count'])}\n")
            f.write("\n")

        # Output files
        f.write("Output Files:\n")
        for name, p in table_paths.items():
            f.write(f"  {name}: {p.name}\n")
        for name, p in plot_paths.items():
            f.write(f"  {name}: {p.name}\n")

    return path


# ---------------------------------------------------------------------------
# Diagnosis builders
# ---------------------------------------------------------------------------

def filter_exclude_year(df: pd.DataFrame, exclude_year: int | None, year_col: str = "year") -> pd.DataFrame:
    if exclude_year and year_col in df.columns:
        return df[df[year_col] != exclude_year]
    return df


def load_analysis_tables(analysis_dir: Path, cfg: WFReportConfig) -> dict[str, pd.DataFrame]:
    table_names = [
        "overall_comparison", "excess_comparison", "yearly_comparison", "yearly_excess",
        "selected_frequency", "parameter_frequency", "alpha_variant_frequency",
        "benchmark_filter_frequency", "single_stock_contribution", "combined_daily_returns",
    ]
    tables = {}
    for name in table_names:
        path = analysis_dir / f"{cfg.analysis_output_prefix}_{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path, encoding="utf-8-sig")
        else:
            tables[name] = pd.DataFrame()
    return tables


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


def build_parameter_stability(detail: pd.DataFrame, exclude_year: int | None, cfg: WFReportConfig) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()

    df = filter_exclude_year(detail, exclude_year)
    param_cols = [c for c in cfg.param_cols if c in df.columns]
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


# ---------------------------------------------------------------------------
# Diagnosis plotting (parameterized by display_name / output_prefix)
# ---------------------------------------------------------------------------

def make_train_vs_test_scatter(gap_df: pd.DataFrame, output_dir: Path, cfg: WFReportConfig) -> Path | None:
    if gap_df.empty or "train_annual_return" not in gap_df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(gap_df["train_annual_return"], gap_df["test_total_return"], alpha=0.3, s=10)
    lim = max(abs(gap_df["train_annual_return"].max()), abs(gap_df["test_total_return"].max()), 0.1)
    ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5)
    ax.set_xlabel("Train Annual Return")
    ax.set_ylabel("Test Total Return")
    ax.set_title(f"{cfg.display_name}: Train vs Test Returns")
    plt.tight_layout()
    path = output_dir / f"{cfg.diagnosis_output_prefix}_train_vs_test_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_yearly_excess_heatmap(yearly_weakness: pd.DataFrame, output_dir: Path, cfg: WFReportConfig) -> Path | None:
    if yearly_weakness.empty or "total_return" not in yearly_weakness.columns:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    data = yearly_weakness[["year", "total_return"]].set_index("year")
    ax.bar(data.index.astype(str), data["total_return"])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{cfg.display_name}: Yearly Returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    plt.tight_layout()
    path = output_dir / f"{cfg.diagnosis_output_prefix}_yearly_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_top_draggers_chart(contrib: pd.DataFrame, output_dir: Path, top_n: int, cfg: WFReportConfig) -> Path | None:
    if contrib.empty or "sum_return" not in contrib.columns:
        return None
    top = contrib.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
    ax.barh(top["symbol"], top["sum_return"])
    ax.set_xlabel("Sum Test Return")
    ax.set_title(f"{cfg.display_name}: Worst {top_n} Stock Contributors")
    plt.tight_layout()
    path = output_dir / f"{cfg.diagnosis_output_prefix}_top_draggers.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_diagnosis_outputs(output_dir: Path, tables: dict[str, pd.DataFrame], cfg: WFReportConfig) -> dict[str, Path]:
    paths = {}
    for name, df in tables.items():
        if df is not None and not df.empty:
            path = output_dir / f"{cfg.diagnosis_output_prefix}_{name}.csv"
            df.to_csv(path, encoding="utf-8-sig", index=False)
            paths[name] = path
    return paths


# ---------------------------------------------------------------------------
# Signal evaluation integration
# ---------------------------------------------------------------------------

VALID_VARIANTS = [
    "short_term_reversal",
    "low_volatility",
    "turnover_reversal",
    "volume_price_divergence",
]


def load_signal_evaluation_summaries(
    eval_dir: Path,
    variants: list[str] | None = None,
    label_col: str = "ret_1d",
) -> pd.DataFrame:
    """从 signal_evaluation 输出目录加载各 variant 的 IC 摘要。

    Parameters
    ----------
    eval_dir : Path
        signal_evaluation 输出根目录，结构为：
        <eval_dir>/<variant>/<label_col>/signal_ic_summary.csv
    variants : list[str], optional
        要加载的 variant 列表，默认全部 4 个。
    label_col : str
        前瞻收益列名，默认 "ret_1d"。

    Returns
    -------
    pd.DataFrame
        合并的 IC 摘要表，含 alpha_variant 列。
    """
    if variants is None:
        variants = VALID_VARIANTS

    frames: list[pd.DataFrame] = []
    for variant in variants:
        summary_path = eval_dir / variant / label_col / "signal_ic_summary.csv"
        if not summary_path.exists():
            logger.debug("信号评估文件不存在: %s", summary_path)
            continue
        try:
            df = pd.read_csv(summary_path, encoding="utf-8-sig")
            df["alpha_variant"] = variant
            frames.append(df)
        except Exception as e:
            logger.warning("读取信号评估文件失败 %s: %s", summary_path, e)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_signal_quality_section(
    eval_dir: Path,
    variants: list[str] | None = None,
    label_cols: list[str] | None = None,
) -> pd.DataFrame:
    """构建信号质量综合摘要表，用于诊断报告。

    对每个 variant × label_col 组合加载 IC 摘要，
    返回一个 DataFrame 包含：alpha_variant, label_col, ic_mean, rank_ic_mean, icir, ic_positive_rate, signal_quality_label。

    Parameters
    ----------
    eval_dir : Path
        signal_evaluation 输出根目录。
    variants : list[str], optional
        要加载的 variant 列表，默认全部 4 个。
    label_cols : list[str], optional
        要加载的前瞻收益列名，默认 ["ret_1d", "ret_5d", "ret_20d"]。

    Returns
    -------
    pd.DataFrame
        信号质量综合摘要。
    """
    if variants is None:
        variants = VALID_VARIANTS
    if label_cols is None:
        label_cols = ["ret_1d", "ret_5d", "ret_20d"]

    frames: list[pd.DataFrame] = []
    for label_col in label_cols:
        df = load_signal_evaluation_summaries(eval_dir, variants, label_col)
        if df.empty:
            continue
        df["label_col"] = label_col
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # 添加信号质量标签
    quality_labels = []
    for _, row in combined.iterrows():
        ic_mean = row.get("ic_mean", np.nan)
        icir = row.get("icir", np.nan)
        if pd.isna(ic_mean) or pd.isna(icir):
            quality_labels.append("unknown")
        elif abs(ic_mean) > 0.03 and abs(icir) > 0.5:
            quality_labels.append("strong")
        elif abs(ic_mean) > 0.01 and abs(icir) > 0.2:
            quality_labels.append("moderate")
        else:
            quality_labels.append("weak")
    combined["signal_quality_label"] = quality_labels

    # 选择关键列
    key_cols = [
        "alpha_variant", "label_col",
        "ic_mean", "ic_std", "icir", "ic_tstat", "ic_positive_rate",
        "rank_ic_mean", "rank_icir",
        "signal_quality_label",
    ]
    available_cols = [c for c in key_cols if c in combined.columns]
    return combined[available_cols]


def write_signal_quality_to_report(
    f,
    signal_quality: pd.DataFrame,
) -> None:
    """将信号质量摘要写入诊断报告文件句柄。

    Parameters
    ----------
    f : file object
        已打开的报告文件句柄。
    signal_quality : pd.DataFrame
        build_signal_quality_section() 的返回值。
    """
    if signal_quality.empty:
        f.write("Signal Quality: (no signal evaluation data found)\n")
        f.write("  Tip: run evaluate_alpha_signals.py before diagnosis to get IC/RankIC metrics.\n\n")
        return

    f.write("Signal Quality (from signal_evaluation):\n")
    f.write("-" * 40 + "\n")

    # 按 variant 分组
    for variant in signal_quality["alpha_variant"].unique():
        vdf = signal_quality[signal_quality["alpha_variant"] == variant]
        f.write(f"\n  {variant}:\n")
        for _, row in vdf.iterrows():
            label = row.get("label_col", "?")
            ic_mean = row.get("ic_mean", np.nan)
            rank_ic_mean = row.get("rank_ic_mean", np.nan)
            icir = row.get("icir", np.nan)
            quality = row.get("signal_quality_label", "unknown")
            f.write(
                f"    {label}: IC={ic_mean:.4f}, RankIC={rank_ic_mean:.4f}, "
                f"ICIR={icir:.4f}, quality={quality}\n"
                if not pd.isna(ic_mean)
                else f"    {label}: N/A\n"
            )

    # 总体判断
    weak_count = len(signal_quality[signal_quality["signal_quality_label"] == "weak"])
    strong_count = len(signal_quality[signal_quality["signal_quality_label"] == "strong"])
    total = len(signal_quality)

    f.write("\n  Signal Quality Summary:\n")
    f.write(f"    strong: {strong_count}/{total}, weak: {weak_count}/{total}\n")

    if weak_count == total:
        f.write("    *** ALL SIGNALS WEAK: 所有信号 IC 均低于阈值，截面排序能力不足。***\n")
        f.write("    *** 建议不要依赖回测 headline 指标做决策，信号可能只是噪声。***\n")
    elif strong_count == 0:
        f.write("    *** NO STRONG SIGNALS: 没有强信号，建议重新设计因子逻辑。***\n")
    f.write("\n")


# ---------------------------------------------------------------------------
# Diagnosis report writing
# ---------------------------------------------------------------------------

def write_recommendations(
    output_dir: Path,
    args,
    summary: pd.DataFrame,
    yearly_weakness: pd.DataFrame,
    gap_df: pd.DataFrame,
    gap_corr: pd.DataFrame,
    bad_contrib: pd.DataFrame | None,
    good_contrib: pd.DataFrame | None,
    alpha_variant_stab: pd.DataFrame | None,
    param_stab: pd.DataFrame | None,
    selected_rep: pd.DataFrame,
    table_paths: dict,
    plot_paths: dict,
    exclude_year: int | None,
    cfg: WFReportConfig,
    signal_quality: pd.DataFrame | None = None,
) -> Path:
    path = output_dir / f"{cfg.diagnosis_output_prefix}_recommendations.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{cfg.display_name} Strategy Diagnosis Recommendations{cfg.display_suffix}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Settings:\n")
        f.write(f"  analysis_dir: {args.analysis_dir}\n")
        wf_dir_val = getattr(args, "walk_forward_dir", getattr(args, "wf_dir", ""))
        f.write(f"  walk_forward_dir: {wf_dir_val}\n")
        f.write(f"  run_id: {args.run_id}\n")
        f.write(f"  exclude_year: {exclude_year}\n\n")

        # Signal Quality (from signal evaluation)
        if signal_quality is not None:
            write_signal_quality_to_report(f, signal_quality)

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
            for flag_val, count in flags.items():
                f.write(f"  {flag_val}: {count}\n")
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

        # Problem source analysis
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

        # Signal quality issues
        if signal_quality is not None and not signal_quality.empty:
            weak_signals = signal_quality[signal_quality.get("signal_quality_label", "") == "weak"]
            if len(weak_signals) == len(signal_quality):
                issues.append("signal_quality: 所有信号 IC 均弱，截面排序能力不足")
            elif len(weak_signals) > len(signal_quality) * 0.5:
                issues.append("signal_quality: 过半信号 IC 弱，因子设计可能有问题")

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
