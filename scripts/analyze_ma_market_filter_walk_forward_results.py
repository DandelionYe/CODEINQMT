# -*- coding: utf-8 -*-
"""
analyze_ma_market_filter_walk_forward_results.py

用途：
分析 validate_ma_market_filter_candidates.py 生成的 MA v2 walk-forward 样本外验证结果。
MA v2 = 个股均线交叉 + 大盘趋势过滤。

本脚本只做分析，不重新跑策略。

输入：
backtests/walk_forward_ma_market_filter_csv 下的：
- wf_ma_mf_stock_{market}_*_top{N}_portfolio_daily.csv
- wf_ma_mf_stock_{market}_*_top{N}_portfolio_period_summary.csv
- wf_ma_mf_stock_{market}_*_top{N}_selected_by_year.csv
- wf_ma_mf_stock_{market}_*_top{N}_test_detail.csv

基准：
默认使用 QMT 导出的指数 CSV：
- 000300.SH 沪深300
- 000905.SH 中证500
- 000852.SH 中证1000

输出：
backtests/walk_forward_ma_market_filter_analysis

运行示例：

1. 默认分析 ALL 市场的 MA v2 walk-forward 结果：
python scripts\\analyze_ma_market_filter_walk_forward_results.py

2. 不生成图片：
python scripts\\analyze_ma_market_filter_walk_forward_results.py --no-png

3. 指定更多基准：
python scripts\\analyze_ma_market_filter_walk_forward_results.py --benchmarks 000300.SH,000905.SH,000852.SH,000001.SH
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR, DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.metrics import format_pct, format_float, max_drawdown_from_equity  # noqa: E402
from scripts.common.validation import resolve_path, parse_list  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "backtests" / "walk_forward_ma_market_filter_csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backtests" / "walk_forward_ma_market_filter_analysis"
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "data" / "qmt_export"

BENCHMARK_NAMES = {
    "000300.SH": "CSI300",
    "000905.SH": "CSI500",
    "000852.SH": "CSI1000",
    "000001.SH": "SSE Composite",
    "399001.SZ": "SZ Component",
    "399006.SZ": "ChiNext",
}



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_one_file(input_dir: Path, market: str, portfolio_size: int, kind: str) -> Path:
    pattern = f"wf_ma_mf_stock_{market}_*_top{portfolio_size}_{kind}.csv"
    files = sorted(input_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"找不到 market={market}, portfolio_size={portfolio_size}, kind={kind} 的文件。\n"
            f"pattern: {pattern}\n"
            f"input_dir: {input_dir}"
        )

    if len(files) > 1:
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    return files[0]


def load_walk_forward_group(input_dir: Path, market: str, portfolio_size: int) -> dict:
    files = {
        "portfolio_daily": find_one_file(input_dir, market, portfolio_size, "portfolio_daily"),
        "portfolio_period_summary": find_one_file(input_dir, market, portfolio_size, "portfolio_period_summary"),
        "selected_by_year": find_one_file(input_dir, market, portfolio_size, "selected_by_year"),
        "test_detail": find_one_file(input_dir, market, portfolio_size, "test_detail"),
    }

    daily = pd.read_csv(files["portfolio_daily"])
    period = pd.read_csv(files["portfolio_period_summary"])
    selected = pd.read_csv(files["selected_by_year"])
    detail = pd.read_csv(files["test_detail"])

    if "date" not in daily.columns:
        raise RuntimeError(f"{files['portfolio_daily']} 缺少 date 字段。")

    if "portfolio_ret" not in daily.columns:
        raise RuntimeError(f"{files['portfolio_daily']} 缺少 portfolio_ret 字段。")

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"])
    daily = daily.sort_values("date")
    daily["portfolio_ret"] = pd.to_numeric(daily["portfolio_ret"], errors="coerce").fillna(0.0)
    daily["market_group"] = market

    if "test_year" in daily.columns:
        daily["test_year"] = pd.to_numeric(daily["test_year"], errors="coerce").astype("Int64")
    else:
        daily["test_year"] = daily["date"].dt.year

    period["market_group"] = market
    selected["market_group"] = market
    detail["market_group"] = market

    return {
        "market": market,
        "files": files,
        "daily": daily,
        "period": period,
        "selected": selected,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

def symbol_to_qmt_csv(symbol: str, export_root: Path) -> Path:
    code, market = ma.normalize_symbol(symbol)
    if market is None:
        raise ValueError(f"基准代码必须带市场后缀：{symbol}")

    csv_path = export_root / market / f"price_{code}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"基准指数文件不存在：{csv_path}")

    return csv_path


def load_benchmark_returns(
    benchmarks: list[str],
    export_root: Path,
    all_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    if all_dates.empty:
        raise RuntimeError("策略日期为空，无法加载基准。")

    start = all_dates.min().strftime("%Y%m%d")
    end = all_dates.max().strftime("%Y%m%d")

    bench_returns = pd.DataFrame(index=all_dates)

    for symbol in benchmarks:
        csv_path = symbol_to_qmt_csv(symbol, export_root)
        df = ma.load_qmt_price_csv(csv_path, start=start, end=end)

        ret = df["close"].pct_change().fillna(0.0)
        ret = ret.reindex(all_dates).fillna(0.0)

        name = BENCHMARK_NAMES.get(symbol, symbol)
        col = f"BENCH_{symbol}_{name}"
        bench_returns[col] = ret.astype(float)

    return bench_returns


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def drawdown_series(ret: pd.Series) -> pd.Series:
    equity = (1.0 + ret.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def calc_metrics(ret: pd.Series) -> dict:
    ret = ret.dropna().astype(float)

    if ret.empty:
        return {
            "days": 0,
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "calmar": np.nan,
        }

    equity = (1.0 + ret).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    days = len(ret)
    annual_return = float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / max(days, 1)) - 1.0)
    annual_volatility = float(ret.std() * SQRT_TRADING_DAYS_PER_YEAR)

    if ret.std() == 0 or np.isnan(ret.std()):
        sharpe = np.nan
    else:
        sharpe = float(ret.mean() / ret.std() * SQRT_TRADING_DAYS_PER_YEAR)

    mdd = max_drawdown_from_equity(equity)

    if mdd == 0 or np.isnan(mdd):
        calmar = np.nan
    else:
        calmar = float(annual_return / abs(mdd))

    return {
        "days": int(days),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "calmar": calmar,
    }


# ---------------------------------------------------------------------------
# Comparison builders
# ---------------------------------------------------------------------------

def build_combined_returns(groups: dict, bench_returns: pd.DataFrame) -> pd.DataFrame:
    all_dates = bench_returns.index
    combined = pd.DataFrame(index=all_dates)

    for market, obj in groups.items():
        daily = obj["daily"].copy()
        daily = daily.set_index("date").sort_index()
        combined[f"STRATEGY_{market}"] = daily["portfolio_ret"].reindex(all_dates).fillna(0.0)

    for col in bench_returns.columns:
        combined[col] = bench_returns[col].reindex(all_dates).fillna(0.0)

    combined = combined.sort_index()
    return combined


def build_overall_comparison(
    combined: pd.DataFrame,
    incomplete_year: int | None,
) -> pd.DataFrame:
    rows = []

    periods = {
        "all_years": combined.index,
    }

    if incomplete_year is not None:
        complete_index = combined.index[combined.index.year != incomplete_year]
        periods[f"exclude_incomplete_{incomplete_year}"] = complete_index

    for period_name, index in periods.items():
        sub = combined.loc[index]

        for col in sub.columns:
            metrics = calc_metrics(sub[col])
            rows.append(
                {
                    "period": period_name,
                    "entity": col,
                    "entity_type": "strategy" if col.startswith("STRATEGY_") else "benchmark",
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def build_excess_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for period, period_df in overall.groupby("period"):
        strategies = period_df[period_df["entity_type"] == "strategy"]
        benchmarks = period_df[period_df["entity_type"] == "benchmark"]

        for _, s in strategies.iterrows():
            for _, b in benchmarks.iterrows():
                rows.append(
                    {
                        "period": period,
                        "strategy": s["entity"],
                        "benchmark": b["entity"],
                        "strategy_total_return": s["total_return"],
                        "benchmark_total_return": b["total_return"],
                        "excess_total_return": s["total_return"] - b["total_return"],
                        "strategy_annual_return": s["annual_return"],
                        "benchmark_annual_return": b["annual_return"],
                        "excess_annual_return": s["annual_return"] - b["annual_return"],
                        "strategy_max_drawdown": s["max_drawdown"],
                        "benchmark_max_drawdown": b["max_drawdown"],
                        "strategy_sharpe": s["sharpe"],
                        "benchmark_sharpe": b["sharpe"],
                    }
                )

    return pd.DataFrame(rows)


def build_yearly_comparison(
    combined: pd.DataFrame,
    incomplete_year: int | None,
) -> pd.DataFrame:
    rows = []

    for year, year_df in combined.groupby(combined.index.year):
        for col in year_df.columns:
            metrics = calc_metrics(year_df[col])
            rows.append(
                {
                    "year": int(year),
                    "is_incomplete_year": bool(incomplete_year is not None and year == incomplete_year),
                    "entity": col,
                    "entity_type": "strategy" if col.startswith("STRATEGY_") else "benchmark",
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def build_yearly_excess(yearly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for year, year_df in yearly.groupby("year"):
        strategies = year_df[year_df["entity_type"] == "strategy"]
        benchmarks = year_df[year_df["entity_type"] == "benchmark"]

        for _, s in strategies.iterrows():
            for _, b in benchmarks.iterrows():
                rows.append(
                    {
                        "year": int(year),
                        "is_incomplete_year": bool(s["is_incomplete_year"]),
                        "strategy": s["entity"],
                        "benchmark": b["entity"],
                        "strategy_total_return": s["total_return"],
                        "benchmark_total_return": b["total_return"],
                        "excess_total_return": s["total_return"] - b["total_return"],
                        "strategy_annual_return": s["annual_return"],
                        "benchmark_annual_return": b["annual_return"],
                        "excess_annual_return": s["annual_return"] - b["annual_return"],
                        "strategy_max_drawdown": s["max_drawdown"],
                        "benchmark_max_drawdown": b["max_drawdown"],
                        "strategy_sharpe": s["sharpe"],
                        "benchmark_sharpe": b["sharpe"],
                        "beat_benchmark": bool(s["total_return"] > b["total_return"]),
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MA v2 specific analyses
# ---------------------------------------------------------------------------

def analyze_selected_frequency(groups: dict) -> pd.DataFrame:
    frames = []

    for market, obj in groups.items():
        selected = obj["selected"].copy()

        if selected.empty:
            continue

        selected["market_group"] = market

        if "symbol" not in selected.columns:
            continue

        if "selected_rank" in selected.columns:
            selected["selected_rank"] = pd.to_numeric(selected["selected_rank"], errors="coerce")
        else:
            selected["selected_rank"] = np.nan

        frames.append(selected)

    if not frames:
        return pd.DataFrame()

    all_selected = pd.concat(frames, ignore_index=True)

    group_cols = ["market_group", "symbol"]
    result = all_selected.groupby(group_cols).agg(
        selected_count=("symbol", "size"),
        years=("test_year", lambda x: ",".join(map(str, sorted(pd.Series(x).dropna().astype(int).unique())))),
        avg_rank=("selected_rank", "mean"),
        best_rank=("selected_rank", "min"),
        avg_train_annual_return=("train_annual_return", "mean"),
        avg_train_max_drawdown=("train_max_drawdown", "mean"),
        avg_train_sharpe=("train_sharpe", "mean"),
        avg_train_score=("train_score", "mean"),
    ).reset_index()

    result = result.sort_values(
        by=["selected_count", "avg_train_score", "avg_rank"],
        ascending=[False, False, True],
    )

    return result


def analyze_parameter_frequency(groups: dict) -> pd.DataFrame:
    frames = []

    for market, obj in groups.items():
        selected = obj["selected"].copy()

        if selected.empty:
            continue

        if "fast" not in selected.columns or "slow" not in selected.columns:
            continue

        selected["market_group"] = market
        selected["fast"] = pd.to_numeric(selected["fast"], errors="coerce").astype("Int64")
        selected["slow"] = pd.to_numeric(selected["slow"], errors="coerce").astype("Int64")
        selected["param"] = selected["fast"].astype(str) + "/" + selected["slow"].astype(str)
        frames.append(selected)

    if not frames:
        return pd.DataFrame()

    all_selected = pd.concat(frames, ignore_index=True)

    result = all_selected.groupby(["market_group", "fast", "slow", "param"]).agg(
        selected_count=("param", "size"),
        avg_rank=("selected_rank", "mean"),
        avg_train_annual_return=("train_annual_return", "mean"),
        avg_train_max_drawdown=("train_max_drawdown", "mean"),
        avg_train_sharpe=("train_sharpe", "mean"),
        avg_train_score=("train_score", "mean"),
    ).reset_index()

    result = result.sort_values(
        by=["market_group", "selected_count", "avg_train_score"],
        ascending=[True, False, False],
    )

    return result


def analyze_benchmark_filter_frequency(groups: dict) -> tuple[pd.DataFrame, dict]:
    """统计大盘过滤使用的 benchmark、benchmark_fast、benchmark_slow 频率，以及 market_filter_on_ratio 统计。"""
    frames = []

    for market, obj in groups.items():
        for source_name, df in [("selected_by_year", obj["selected"]), ("test_detail", obj["detail"])]:
            if df.empty:
                continue
            tmp = df.copy()
            tmp["_source"] = source_name
            tmp["market_group"] = market
            frames.append(tmp)

    if not frames:
        return pd.DataFrame(), {}

    all_df = pd.concat(frames, ignore_index=True)

    # Benchmark frequency (from selected_by_year only to avoid double counting)
    sel = all_df[all_df["_source"] == "selected_by_year"].copy()
    if sel.empty:
        return pd.DataFrame(), {}

    rows = []

    if "benchmark" in sel.columns:
        bench_freq = sel.groupby(["market_group", "benchmark"]).size().reset_index(name="count")
        bench_freq["kind"] = "benchmark"
        bench_freq["value"] = bench_freq["benchmark"]
        rows.append(bench_freq[["market_group", "kind", "value", "count"]])

    if "benchmark_fast" in sel.columns and "benchmark_slow" in sel.columns:
        sel["_bf_bs"] = sel["benchmark_fast"].astype(str) + "/" + sel["benchmark_slow"].astype(str)
        bs_freq = sel.groupby(["market_group", "_bf_bs"]).size().reset_index(name="count")
        bs_freq["kind"] = "benchmark_fast/slow"
        bs_freq["value"] = bs_freq["_bf_bs"]
        rows.append(bs_freq[["market_group", "kind", "value", "count"]])

    if not rows:
        return pd.DataFrame(), {}

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["kind", "count"], ascending=[True, False])

    # Market filter ratio stats (use test_detail only to avoid double counting)
    detail_df = all_df[all_df["_source"] == "test_detail"]
    ratio_stats = {}
    for col in ["train_market_filter_on_ratio", "test_market_filter_on_ratio",
                 "train_strategy_exposure_ratio", "test_strategy_exposure_ratio"]:
        if col in detail_df.columns:
            vals = pd.to_numeric(detail_df[col], errors="coerce").dropna()
            if len(vals) > 0:
                ratio_stats[col] = {
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "count": int(len(vals)),
                }

    return result, ratio_stats


def analyze_single_stock_contribution(groups: dict) -> pd.DataFrame:
    frames = []

    for market, obj in groups.items():
        detail = obj["detail"].copy()

        if detail.empty or "symbol" not in detail.columns:
            continue

        detail["market_group"] = market
        frames.append(detail)

    if not frames:
        return pd.DataFrame()

    all_detail = pd.concat(frames, ignore_index=True)

    numeric_cols = [
        "test_total_return",
        "test_annual_return",
        "test_max_drawdown",
        "test_sharpe",
        "test_trade_count",
        "test_excess_vs_buy_hold_total_return",
        "test_excess_vs_stock_only_total_return",
        "train_annual_return",
        "train_max_drawdown",
        "train_sharpe",
        "train_score",
        "selected_rank",
    ]

    for col in numeric_cols:
        if col in all_detail.columns:
            all_detail[col] = pd.to_numeric(all_detail[col], errors="coerce")

    agg_dict = {
        "selected_count": ("symbol", "size"),
        "years": ("test_year", lambda x: ",".join(map(str, sorted(pd.Series(x).dropna().astype(int).unique())))),
        "avg_selected_rank": ("selected_rank", "mean"),
        "avg_test_total_return": ("test_total_return", "mean"),
        "sum_test_total_return": ("test_total_return", "sum"),
        "median_test_total_return": ("test_total_return", "median"),
        "min_test_total_return": ("test_total_return", "min"),
        "max_test_total_return": ("test_total_return", "max"),
        "avg_test_max_drawdown": ("test_max_drawdown", "mean"),
        "avg_test_sharpe": ("test_sharpe", "mean"),
        "win_year_count": ("test_total_return", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
        "avg_train_score": ("train_score", "mean"),
    }

    # MA v2 specific: excess vs buy-hold and vs stock-only
    if "test_excess_vs_buy_hold_total_return" in all_detail.columns:
        agg_dict["avg_test_excess_vs_buy_hold"] = ("test_excess_vs_buy_hold_total_return", "mean")
        agg_dict["sum_test_excess_vs_buy_hold"] = ("test_excess_vs_buy_hold_total_return", "sum")

    if "test_excess_vs_stock_only_total_return" in all_detail.columns:
        agg_dict["avg_test_excess_vs_stock_only"] = ("test_excess_vs_stock_only_total_return", "mean")
        agg_dict["sum_test_excess_vs_stock_only"] = ("test_excess_vs_stock_only_total_return", "sum")

    result = all_detail.groupby(["market_group", "symbol"]).agg(**agg_dict).reset_index()

    result["win_rate"] = result["win_year_count"] / result["selected_count"]

    sort_cols = ["sum_test_total_return", "avg_test_total_return", "win_rate"]
    sort_asc = [False, False, False]
    result = result.sort_values(by=sort_cols, ascending=sort_asc)

    return result


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_tables(
    output_dir: Path,
    combined: pd.DataFrame,
    overall: pd.DataFrame,
    excess: pd.DataFrame,
    yearly: pd.DataFrame,
    yearly_excess: pd.DataFrame,
    selected_freq: pd.DataFrame,
    param_freq: pd.DataFrame,
    benchmark_filter_freq: pd.DataFrame,
    contribution: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "combined_daily_returns": output_dir / "mf_wf_analysis_combined_daily_returns.csv",
        "overall_comparison": output_dir / "mf_wf_analysis_overall_comparison.csv",
        "excess_comparison": output_dir / "mf_wf_analysis_excess_comparison.csv",
        "yearly_comparison": output_dir / "mf_wf_analysis_yearly_comparison.csv",
        "yearly_excess": output_dir / "mf_wf_analysis_yearly_excess.csv",
        "selected_frequency": output_dir / "mf_wf_analysis_selected_frequency.csv",
        "parameter_frequency": output_dir / "mf_wf_analysis_parameter_frequency.csv",
        "benchmark_filter_frequency": output_dir / "mf_wf_analysis_benchmark_filter_frequency.csv",
        "single_stock_contribution": output_dir / "mf_wf_analysis_single_stock_contribution.csv",
    }

    combined.reset_index(names="date").to_csv(paths["combined_daily_returns"], index=False, encoding="utf-8-sig")
    overall.to_csv(paths["overall_comparison"], index=False, encoding="utf-8-sig")
    excess.to_csv(paths["excess_comparison"], index=False, encoding="utf-8-sig")
    yearly.to_csv(paths["yearly_comparison"], index=False, encoding="utf-8-sig")
    yearly_excess.to_csv(paths["yearly_excess"], index=False, encoding="utf-8-sig")
    selected_freq.to_csv(paths["selected_frequency"], index=False, encoding="utf-8-sig")
    param_freq.to_csv(paths["parameter_frequency"], index=False, encoding="utf-8-sig")
    if not benchmark_filter_freq.empty:
        benchmark_filter_freq.to_csv(paths["benchmark_filter_frequency"], index=False, encoding="utf-8-sig")
    contribution.to_csv(paths["single_stock_contribution"], index=False, encoding="utf-8-sig")

    return paths


# ---------------------------------------------------------------------------
# PNG chart generation
# ---------------------------------------------------------------------------

def plot_equity_curve(combined: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "mf_wf_analysis_equity_curve.png"

    equity = (1.0 + combined.fillna(0.0)).cumprod()

    plt.figure(figsize=(14, 7))
    for col in equity.columns:
        plt.plot(equity.index, equity[col], label=col)

    plt.title("MA v2 Walk-forward Strategy vs Benchmarks - Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Normalized Equity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_drawdown_curve(combined: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "mf_wf_analysis_drawdown_curve.png"

    dd = pd.DataFrame(index=combined.index)
    for col in combined.columns:
        dd[col] = drawdown_series(combined[col])

    plt.figure(figsize=(14, 7))
    for col in dd.columns:
        plt.plot(dd.index, dd[col], label=col)

    plt.title("MA v2 Walk-forward Strategy vs Benchmarks - Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_yearly_return_bar(yearly: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "mf_wf_analysis_yearly_return_bar.png"

    pivot = yearly.pivot_table(
        index="year",
        columns="entity",
        values="total_return",
        aggfunc="first",
    ).sort_index()

    plt.figure(figsize=(14, 7))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.title("MA v2 Yearly Return Comparison")
    plt.xlabel("Year")
    plt.ylabel("Total Return")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_parameter_frequency(param_freq: pd.DataFrame, output_dir: Path) -> Path | None:
    if param_freq.empty:
        return None

    path = output_dir / "mf_wf_analysis_parameter_frequency.png"

    tmp = param_freq.copy()
    tmp["group_param"] = tmp["market_group"].astype(str) + " " + tmp["param"].astype(str)
    tmp = tmp.sort_values("selected_count", ascending=False).head(30)

    plt.figure(figsize=(14, 7))
    plt.bar(tmp["group_param"], tmp["selected_count"])
    plt.title("MA v2 Parameter Frequency - Top 30")
    plt.xlabel("Market / Parameter")
    plt.ylabel("Selected Count")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_selected_frequency(selected_freq: pd.DataFrame, output_dir: Path) -> Path | None:
    if selected_freq.empty:
        return None

    path = output_dir / "mf_wf_analysis_selected_frequency_top30.png"

    tmp = selected_freq.copy()
    tmp["group_symbol"] = tmp["market_group"].astype(str) + " " + tmp["symbol"].astype(str)
    tmp = tmp.sort_values("selected_count", ascending=False).head(30)

    plt.figure(figsize=(14, 7))
    plt.bar(tmp["group_symbol"], tmp["selected_count"])
    plt.title("MA v2 Selected Symbol Frequency - Top 30")
    plt.xlabel("Market / Symbol")
    plt.ylabel("Selected Count")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_benchmark_filter_frequency(benchmark_filter_freq: pd.DataFrame, output_dir: Path) -> Path | None:
    if benchmark_filter_freq.empty:
        return None

    path = output_dir / "mf_wf_analysis_benchmark_filter_frequency.png"

    tmp = benchmark_filter_freq.copy()
    tmp["label"] = tmp["kind"] + ": " + tmp["value"].astype(str)
    tmp = tmp.sort_values("count", ascending=False).head(30)

    plt.figure(figsize=(14, 7))
    plt.bar(tmp["label"], tmp["count"])
    plt.title("MA v2 Benchmark Filter Frequency")
    plt.xlabel("Benchmark / Fast-Slow")
    plt.ylabel("Count")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def save_plots(
    output_dir: Path,
    combined: pd.DataFrame,
    yearly: pd.DataFrame,
    selected_freq: pd.DataFrame,
    param_freq: pd.DataFrame,
    benchmark_filter_freq: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    paths["equity_curve"] = plot_equity_curve(combined, output_dir)
    paths["drawdown_curve"] = plot_drawdown_curve(combined, output_dir)
    paths["yearly_return_bar"] = plot_yearly_return_bar(yearly, output_dir)

    param_path = plot_parameter_frequency(param_freq, output_dir)
    if param_path is not None:
        paths["parameter_frequency"] = param_path

    selected_path = plot_selected_frequency(selected_freq, output_dir)
    if selected_path is not None:
        paths["selected_frequency_top30"] = selected_path

    bench_filter_path = plot_benchmark_filter_frequency(benchmark_filter_freq, output_dir)
    if bench_filter_path is not None:
        paths["benchmark_filter_frequency"] = bench_filter_path

    return paths


# ---------------------------------------------------------------------------
# TXT report
# ---------------------------------------------------------------------------

def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    groups: dict,
    benchmarks: list[str],
    incomplete_year: int | None,
    overall: pd.DataFrame,
    excess: pd.DataFrame,
    yearly: pd.DataFrame,
    yearly_excess: pd.DataFrame,
    selected_freq: pd.DataFrame,
    param_freq: pd.DataFrame,
    benchmark_filter_freq: pd.DataFrame,
    ratio_stats: dict,
    contribution: pd.DataFrame,
    table_paths: dict[str, Path],
    plot_paths: dict[str, Path],
) -> Path:
    path = output_dir / "mf_wf_analysis_report.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("MA v2 Walk-forward 结果分析与基准比较报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("一、分析设置\n")
        f.write("-" * 80 + "\n")
        f.write(f"input_dir: {args.input_dir}\n")
        f.write(f"output_dir: {args.output_dir}\n")
        f.write(f"markets: {args.markets}\n")
        f.write(f"portfolio_size: {args.portfolio_size}\n")
        f.write(f"benchmarks: {','.join(benchmarks)}\n")
        f.write(f"incomplete_year: {incomplete_year or 'None'}\n")
        f.write("说明：本脚本只分析已有 MA v2 walk-forward 输出，不重新跑策略。\n\n")

        f.write("二、输入文件\n")
        f.write("-" * 80 + "\n")
        for market, obj in groups.items():
            f.write(f"[{market}]\n")
            for key, file in obj["files"].items():
                f.write(f"{key}: {file}\n")
            f.write("\n")

        f.write("三、整体表现对比\n")
        f.write("-" * 80 + "\n")
        display_cols = [
            "period", "entity", "entity_type", "days", "total_return",
            "annual_return", "annual_volatility", "max_drawdown", "sharpe", "calmar",
        ]
        f.write(overall[display_cols].to_string(index=False))
        f.write("\n\n")

        f.write("四、策略相对基准超额表现\n")
        f.write("-" * 80 + "\n")
        if not excess.empty:
            f.write(excess.to_string(index=False))
        else:
            f.write("无 excess 数据。\n")
        f.write("\n\n")

        f.write("五、年度表现\n")
        f.write("-" * 80 + "\n")
        yearly_cols = [
            "year", "is_incomplete_year", "entity", "entity_type",
            "total_return", "annual_return", "max_drawdown", "sharpe",
        ]
        f.write(yearly[yearly_cols].to_string(index=False))
        f.write("\n\n")

        f.write("六、年度胜率：策略跑赢基准次数\n")
        f.write("-" * 80 + "\n")
        if not yearly_excess.empty:
            beat_stats = yearly_excess.groupby(["strategy", "benchmark"]).agg(
                years=("year", "count"),
                beat_count=("beat_benchmark", "sum"),
                avg_excess_total_return=("excess_total_return", "mean"),
                median_excess_total_return=("excess_total_return", "median"),
            ).reset_index()
            beat_stats["beat_rate"] = beat_stats["beat_count"] / beat_stats["years"]
            f.write(beat_stats.to_string(index=False))
        else:
            f.write("无 yearly_excess 数据。\n")
        f.write("\n\n")

        f.write("七、入选股票频率 Top 30\n")
        f.write("-" * 80 + "\n")
        if not selected_freq.empty:
            f.write(selected_freq.head(30).to_string(index=False))
        else:
            f.write("无 selected frequency 数据。\n")
        f.write("\n\n")

        f.write("八、个股 fast/slow 参数频率\n")
        f.write("-" * 80 + "\n")
        if not param_freq.empty:
            f.write(param_freq.to_string(index=False))
        else:
            f.write("无 parameter frequency 数据。\n")
        f.write("\n\n")

        f.write("九、大盘过滤 benchmark/benchmark_fast/benchmark_slow 使用频率\n")
        f.write("-" * 80 + "\n")
        if not benchmark_filter_freq.empty:
            f.write(benchmark_filter_freq.to_string(index=False))
        else:
            f.write("无 benchmark filter frequency 数据。\n")
        f.write("\n\n")

        # Market filter ratio stats
        f.write("九-2、大盘过滤比例统计\n")
        f.write("-" * 80 + "\n")
        if ratio_stats:
            for col_name, stats in ratio_stats.items():
                f.write(f"  {col_name}:\n")
                f.write(f"    mean={stats['mean']:.4f}, median={stats['median']:.4f}, "
                        f"min={stats['min']:.4f}, max={stats['max']:.4f}, count={stats['count']}\n")
        else:
            f.write("  未找到 train_market_filter_on_ratio / test_market_filter_on_ratio 等字段。\n")
        f.write("\n")

        f.write("十、单股样本外贡献 Top 30\n")
        f.write("-" * 80 + "\n")
        if not contribution.empty:
            f.write(contribution.head(30).to_string(index=False))
        else:
            f.write("无 contribution 数据。\n")
        f.write("\n\n")

        f.write("十一、自动诊断\n")
        f.write("-" * 80 + "\n")

        all_years = overall[overall["period"] == "all_years"].copy()
        strategies = all_years[all_years["entity_type"] == "strategy"].copy()

        if not strategies.empty:
            best_strategy = strategies.sort_values("annual_return", ascending=False).iloc[0]
            f.write(
                f"全区间年化最高的策略组合是 {best_strategy['entity']}，"
                f"年化收益 {format_pct(best_strategy['annual_return'])}，"
                f"最大回撤 {format_pct(best_strategy['max_drawdown'])}，"
                f"夏普 {format_float(best_strategy['sharpe'])}。\n"
            )

            if best_strategy["annual_return"] < 0.05:
                f.write("诊断：当前策略全区间年化偏低，单独作为交易策略的吸引力不足。\n")

            if best_strategy["sharpe"] < 0.5:
                f.write("诊断：当前策略夏普偏低，收益质量一般，可能波动与择时噪音较大。\n")

            if best_strategy["max_drawdown"] < -0.30:
                f.write("诊断：当前策略最大回撤较深，后续需要加入风险控制或市场状态过滤。\n")

        if incomplete_year is not None:
            f.write(f"说明：{incomplete_year} 年为未完整年度，整体评价应重点参考剔除该年份后的结果。\n")

        if not yearly_excess.empty:
            weak = yearly_excess[
                (yearly_excess["strategy"].str.contains("STRATEGY", na=False))
                & (yearly_excess["beat_benchmark"] == False)
            ]
            if len(weak) > 0:
                f.write("诊断：存在多个年度/基准下策略未能跑赢基准，后续应分析拖累年份和市场环境。\n")

        if not param_freq.empty:
            top_param = param_freq.sort_values("selected_count", ascending=False).iloc[0]
            f.write(
                f"参数集中度：出现最多的参数组合是 {top_param['market_group']} {top_param['param']}，"
                f"出现 {int(top_param['selected_count'])} 次。\n"
            )

        f.write("\n十二、输出文件\n")
        f.write("-" * 80 + "\n")
        for name, file in table_paths.items():
            f.write(f"{name}: {file}\n")
        for name, file in plot_paths.items():
            f.write(f"{name}: {file}\n")

    return path


# ---------------------------------------------------------------------------
# Incomplete year inference
# ---------------------------------------------------------------------------

def infer_incomplete_year(combined: pd.DataFrame, user_value: int) -> int | None:
    if user_value > 0:
        return user_value

    if combined.empty:
        return None

    max_date = combined.index.max()
    latest_year = int(max_date.year)

    if max_date.month < 12 or (max_date.month == 12 and max_date.day < 15):
        return latest_year

    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 MA v2 walk-forward 结果并与基准比较")

    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="MA v2 walk-forward 输出目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="分析结果输出目录")
    parser.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT), help="QMT 导出的行情根目录")

    parser.add_argument("--markets", default="ALL", help="要分析的市场组合，例如 ALL")
    parser.add_argument("--portfolio-size", type=int, default=20, help="walk-forward 组合数量，对应文件名里的 top20")

    parser.add_argument(
        "--benchmarks",
        default=DEFAULT_BENCHMARK_LIST,
        help="基准指数列表，例如 000300.SH,000905.SH,000852.SH",
    )

    parser.add_argument(
        "--incomplete-year",
        type=int,
        default=2026,
        help="未完整年份。例如 2026。默认 2026。",
    )

    parser.add_argument("--no-png", action="store_true", help="不生成 PNG 图，只输出 CSV 和 TXT")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    export_root = resolve_path(args.export_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    markets = parse_list(args.markets)
    benchmarks = parse_list(args.benchmarks)

    incomplete_year = args.incomplete_year if args.incomplete_year > 0 else None

    logger.info("MA v2 walk-forward 结果分析配置：")
    logger.info("input_dir: %s", input_dir)
    logger.info("output_dir: %s", output_dir)
    logger.info("export_root: %s", export_root)
    logger.info("markets: %s", markets)
    logger.info("portfolio_size: %s", args.portfolio_size)
    logger.info("benchmarks: %s", benchmarks)
    logger.info("incomplete_year: %s", incomplete_year)

    groups = {}

    for market in markets:
        groups[market] = load_walk_forward_group(
            input_dir=input_dir,
            market=market,
            portfolio_size=args.portfolio_size,
        )
        logger.info("已读取 %s MA v2 walk-forward 文件。", market)

    date_indexes = []
    for obj in groups.values():
        daily = obj["daily"]
        date_indexes.append(pd.DatetimeIndex(daily["date"]))

    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(idx) for idx in date_indexes])))

    if all_dates.empty:
        raise RuntimeError("没有读取到任何策略日期。")

    logger.info("策略日期范围：%s 至 %s，共 %d 个交易日。", all_dates.min().date(), all_dates.max().date(), len(all_dates))

    bench_returns = load_benchmark_returns(
        benchmarks=benchmarks,
        export_root=export_root,
        all_dates=all_dates,
    )

    combined = build_combined_returns(groups, bench_returns)

    if incomplete_year is None:
        incomplete_year = infer_incomplete_year(combined, 0)

    if incomplete_year is not None:
        logger.info("未完整年份标记为：%s", incomplete_year)

    overall = build_overall_comparison(combined, incomplete_year=incomplete_year)
    excess = build_excess_comparison(overall)
    yearly = build_yearly_comparison(combined, incomplete_year=incomplete_year)
    yearly_excess = build_yearly_excess(yearly)

    selected_freq = analyze_selected_frequency(groups)
    param_freq = analyze_parameter_frequency(groups)
    benchmark_filter_freq, ratio_stats = analyze_benchmark_filter_frequency(groups)
    contribution = analyze_single_stock_contribution(groups)

    table_paths = save_tables(
        output_dir=output_dir,
        combined=combined,
        overall=overall,
        excess=excess,
        yearly=yearly,
        yearly_excess=yearly_excess,
        selected_freq=selected_freq,
        param_freq=param_freq,
        benchmark_filter_freq=benchmark_filter_freq,
        contribution=contribution,
    )

    if args.no_png:
        plot_paths = {}
    else:
        plot_paths = save_plots(
            output_dir=output_dir,
            combined=combined,
            yearly=yearly,
            selected_freq=selected_freq,
            param_freq=param_freq,
            benchmark_filter_freq=benchmark_filter_freq,
        )

    report_path = write_report(
        output_dir=output_dir,
        args=args,
        groups=groups,
        benchmarks=benchmarks,
        incomplete_year=incomplete_year,
        overall=overall,
        excess=excess,
        yearly=yearly,
        yearly_excess=yearly_excess,
        selected_freq=selected_freq,
        param_freq=param_freq,
        benchmark_filter_freq=benchmark_filter_freq,
        ratio_stats=ratio_stats,
        contribution=contribution,
        table_paths=table_paths,
        plot_paths=plot_paths,
    )

    logger.info("分析完成。")
    logger.info("报告：%s", report_path)

    logger.info("主要输出文件：")
    for name, path in table_paths.items():
        logger.info("%s: %s", name, path)

    if plot_paths:
        logger.info("图片输出：")
        for name, path in plot_paths.items():
            logger.info("%s: %s", name, path)

    print("\n整体表现摘要：")
    summary_cols = [
        "period",
        "entity",
        "entity_type",
        "total_return",
        "annual_return",
        "max_drawdown",
        "sharpe",
    ]
    print(overall[summary_cols].to_string(index=False))


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception as exc:
        logger.error("程序异常：%s", repr(exc))
        raise
