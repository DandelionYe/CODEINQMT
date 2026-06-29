# -*- coding: utf-8 -*-
"""
analyze_alpha_v4_research_walk_forward_results.py

分析 Alpha v4 walk-forward 样本外结果。

输入：validate_alpha_v4_research_candidates.py 的输出
输出：11 个 CSV + 1 个 TXT 报告 + 可选 PNG 图表

运行示例：
python scripts\\analyze_alpha_v4_research_walk_forward_results.py --input-tag alpha_v4_ALL_stock_ts20150101_fy2021-2025_avpure_momentum_simplified_trend_momentum_mom60_120_tma120_250_vol60_brk120_bm000300SH_000905SH_000852SH_bma120_250_top20_limitALL --no-png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

logger = logging.getLogger(__name__)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR, DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.metrics import format_pct, format_float, max_drawdown_from_equity  # noqa: E402
from scripts.common.validation import resolve_path, parse_list  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "backtests" / "walk_forward_alpha_v4_research_csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backtests" / "walk_forward_alpha_v4_research_analysis"
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "data" / "qmt_export"

BENCHMARK_NAMES = {
    "000300.SH": "CSI300",
    "000905.SH": "CSI500",
    "000852.SH": "CSI1000",
    "000001.SH": "SSE",
    "399001.SZ": "SZComp",
    "399006.SZ": "ChiNext",
}



def list_candidate_files(input_dir: Path, kind: str) -> str:
    candidates = sorted(input_dir.glob(f"wf_alpha_v4_stock_*_{kind}.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not candidates:
        return "  (none)"
    return "\n".join(f"  - {p.name}" for p in candidates[:20])


def find_one_file(
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
            "必须传入 --input-tag 精确指定 alpha v4 walk-forward 文件组。\n"
            f"候选 {kind} 文件:\n{list_candidate_files(input_dir, kind)}"
        )

    files = sorted(input_dir.glob(f"wf_alpha_v4_stock_*top{portfolio_size}_{kind}.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        files = sorted(input_dir.glob(f"wf_alpha_v4_stock_*_{kind}.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"未找到匹配文件: {input_dir}/wf_alpha_v4_stock_*_{kind}.csv")
    if len(files) > 1:
        logger.warning("找到 %d 个匹配文件，使用最新: %s", len(files), files[0].name)
    return files[0]


def load_walk_forward_group(
    input_dir: Path,
    market: str,
    portfolio_size: int,
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
        path = find_one_file(input_dir, market, portfolio_size, kind, file_tag, allow_fallback)
        group["files"][key] = path
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        group[key] = df

    return group


def symbol_to_qmt_csv(symbol: str, export_root: Path) -> Path:
    csv_path, _, _, _ = ma.find_csv_for_stock(symbol, export_root)
    return csv_path


def load_benchmark_returns(
    benchmarks: list[str],
    export_root: Path,
    all_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
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


def drawdown_series(ret: pd.Series) -> pd.Series:
    equity = (1 + ret).cumprod()
    running_max = equity.cummax()
    return equity / running_max - 1.0


def calc_metrics(ret: pd.Series) -> dict:
    if ret.empty:
        return {k: np.nan for k in ["days", "total_return", "annual_return", "annual_volatility", "max_drawdown", "sharpe", "calmar"]}
    total_return = (1.0 + ret).prod() - 1.0
    equity = (1 + ret).cumprod()
    days = len(ret)
    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1.0
    annual_vol = ret.std() * SQRT_TRADING_DAYS_PER_YEAR
    sharpe = ret.mean() / ret.std() * SQRT_TRADING_DAYS_PER_YEAR if ret.std() > 0 else np.nan
    mdd = max_drawdown_from_equity(equity)
    calmar = annual_return / abs(mdd) if mdd != 0 else np.nan
    return {
        "days": days, "total_return": total_return, "annual_return": annual_return,
        "annual_volatility": annual_vol, "max_drawdown": mdd, "sharpe": sharpe, "calmar": calmar,
    }


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


def analyze_parameter_frequency(groups: dict) -> pd.DataFrame:
    selected = groups.get("selected", pd.DataFrame())
    if selected.empty:
        return pd.DataFrame()

    param_cols = [c for c in ["momentum_window", "trend_ma", "vol_window", "breakout_window"] if c in selected.columns]
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


def save_tables(output_dir: Path, **tables) -> dict[str, Path]:
    paths = {}
    prefix = "alpha_v4_wf_analysis"
    for name, df in tables.items():
        if df is not None and not df.empty:
            path = output_dir / f"{prefix}_{name}.csv"
            df.to_csv(path, encoding="utf-8-sig", index=False)
            paths[name] = path
    return paths


def plot_equity_curve(combined: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    equity = (1 + combined).cumprod()
    for col in equity.columns:
        ax.plot(equity.index, equity[col], label=col)
    ax.set_title("Alpha v4 Walk-Forward Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = output_dir / "alpha_v4_wf_analysis_equity_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_drawdown_curve(combined: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4))
    dd = drawdown_series(combined["strategy"])
    ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color="red")
    ax.set_title("Alpha v4 Strategy Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    plt.tight_layout()
    path = output_dir / "alpha_v4_wf_analysis_drawdown_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_yearly_return_bar(yearly: pd.DataFrame, output_dir: Path) -> Path:
    strategy_yearly = yearly[yearly["entity"] == "strategy"]
    if strategy_yearly.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(strategy_yearly["year"].astype(str), strategy_yearly["total_return"])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Alpha v4 Yearly Returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    plt.tight_layout()
    path = output_dir / "alpha_v4_wf_analysis_yearly_return_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_plots(output_dir: Path, combined: pd.DataFrame, yearly: pd.DataFrame,
               selected_freq: pd.DataFrame, param_freq: pd.DataFrame,
               benchmark_filter_freq: pd.DataFrame) -> dict[str, Path]:
    paths = {}
    try:
        paths["equity_curve"] = plot_equity_curve(combined, output_dir)
    except Exception:
        pass
    try:
        paths["drawdown_curve"] = plot_drawdown_curve(combined, output_dir)
    except Exception:
        pass
    try:
        p = plot_yearly_return_bar(yearly, output_dir)
        if p:
            paths["yearly_return_bar"] = p
    except Exception:
        pass
    return paths


def write_report(output_dir: Path, args, groups, benchmarks, incomplete_year,
                 overall, excess, yearly, yearly_excess, selected_freq,
                 param_freq, alpha_variant_freq, benchmark_filter_freq, ratio_stats, contribution,
                 table_paths, plot_paths) -> Path:
    path = output_dir / "alpha_v4_wf_analysis_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("Alpha v4 Walk-Forward Analysis Report\n")
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


def infer_incomplete_year(combined: pd.DataFrame, user_value: int) -> int | None:
    if user_value and user_value > 0:
        return user_value
    if combined.empty:
        return None
    latest = combined.index.max()
    if latest.month < 12 or (latest.month == 12 and latest.day < 15):
        return latest.year
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 Alpha v4 walk-forward 样本外结果")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT))
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
    param_freq = analyze_parameter_frequency(groups)
    alpha_variant_freq = analyze_alpha_variant_frequency(groups)
    benchmark_filter_freq, ratio_stats = analyze_benchmark_filter_frequency(groups)
    contribution = analyze_single_stock_contribution(groups)

    # 保存
    table_paths = save_tables(
        output_dir,
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
        plot_paths = save_plots(output_dir, combined, yearly, selected_freq, param_freq, benchmark_filter_freq)

    # 报告
    report_path = write_report(
        output_dir, args, groups, benchmarks, incomplete_year,
        overall, excess, yearly, yearly_excess, selected_freq,
        param_freq, alpha_variant_freq, benchmark_filter_freq, ratio_stats, contribution,
        table_paths, plot_paths,
    )

    logger.info("分析完成，输出到: %s", output_dir)
    logger.info("报告: %s", report_path)


if __name__ == "__main__":
    setup_cli_logging()
    main()
