# -*- coding: utf-8 -*-
"""
diagnose_ma_market_filter_strategy_results.py

用途：
诊断 MA v2（个股均线交叉 + 大盘趋势过滤）walk-forward 样本外表现弱的原因。
只做诊断，不重新跑策略、不修改回测结果、不下单、不接入 QMT 交易。

输入：
1. MA v2 分析结果：backtests/walk_forward_ma_market_filter_analysis/
2. MA v2 原始 walk-forward 文件：backtests/walk_forward_ma_market_filter_csv/

输出：
backtests/strategy_diagnosis/ma_mf_diagnosis_<run_id>/

运行示例：

1. 默认运行：
python scripts\\diagnose_ma_market_filter_strategy_results.py

2. 不生成图片：
python scripts\\diagnose_ma_market_filter_strategy_results.py --no-png

3. 指定 run id：
python scripts\\diagnose_ma_market_filter_strategy_results.py --run-id exp002_ma_mf_diagnosis
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
from scripts.common.metrics import format_pct, format_float  # noqa: E402
from scripts.common.validation import resolve_path, parse_list  # noqa: E402

DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "backtests" / "walk_forward_ma_market_filter_analysis"
DEFAULT_WF_DIR = PROJECT_ROOT / "backtests" / "walk_forward_ma_market_filter_csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "backtests" / "strategy_diagnosis"

BENCHMARK_ENTITY_MAP = {
    "000300.SH": "BENCH_000300.SH_CSI300",
    "000905.SH": "BENCH_000905.SH_CSI500",
    "000852.SH": "BENCH_000852.SH_CSI1000",
    "000001.SH": "BENCH_000001.SH_SSE Composite",
    "399001.SZ": "BENCH_399001.SZ_SZ Component",
    "399006.SZ": "BENCH_399006.SZ_ChiNext",
}

BENCHMARK_SHORT = {v: k for k, v in BENCHMARK_ENTITY_MAP.items()}




def normalize_benchmark_entities(benchmarks: list[str]) -> list[str]:
    result = []
    for b in benchmarks:
        if b.startswith("BENCH_"):
            result.append(b)
        elif b in BENCHMARK_ENTITY_MAP:
            result.append(BENCHMARK_ENTITY_MAP[b])
        else:
            result.append(b)
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_wf_file(input_dir: Path, market: str, portfolio_size: int, kind: str) -> Path:
    pattern = f"wf_ma_mf_stock_{market}_*_top{portfolio_size}_{kind}.csv"
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"找不到 market={market}, portfolio_size={portfolio_size}, kind={kind} 的文件。\n"
            f"pattern: {pattern}\ninput_dir: {input_dir}"
        )
    if len(files) > 1:
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_walk_forward_raw(
    wf_dir: Path, markets: list[str], portfolio_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames = []
    detail_frames = []

    for market in markets:
        sel_path = find_wf_file(wf_dir, market, portfolio_size, "selected_by_year")
        det_path = find_wf_file(wf_dir, market, portfolio_size, "test_detail")

        sel = pd.read_csv(sel_path)
        det = pd.read_csv(det_path)

        sel["market_group"] = market
        det["market_group"] = market
        sel["_source_file"] = str(sel_path)
        det["_source_file"] = str(det_path)

        selected_frames.append(sel)
        detail_frames.append(det)

    selected_all = pd.concat(selected_frames, ignore_index=True)
    detail_all = pd.concat(detail_frames, ignore_index=True)

    # Coerce numeric columns present in selected_by_year (comprehensive file)
    numeric_cols = [
        "test_year", "selected_rank", "fast", "slow",
        "benchmark_fast", "benchmark_slow",
        "train_annual_return", "train_max_drawdown", "train_sharpe",
        "train_total_return", "train_trade_count", "train_score",
        "train_buy_hold_total_return", "train_stock_only_total_return",
        "train_excess_vs_buy_hold_total_return", "train_excess_vs_stock_only_total_return",
        "train_market_filter_on_ratio", "train_strategy_exposure_ratio",
        "test_total_return", "test_annual_return", "test_annual_volatility",
        "test_max_drawdown", "test_sharpe", "test_trade_count",
        "test_buy_hold_total_return", "test_stock_only_total_return",
        "test_excess_vs_buy_hold_total_return", "test_excess_vs_stock_only_total_return",
        "test_market_filter_on_ratio", "test_strategy_exposure_ratio",
    ]
    selected_all = safe_to_numeric(selected_all, numeric_cols)
    detail_all = safe_to_numeric(detail_all, [c for c in numeric_cols if c in detail_all.columns])

    return selected_all, detail_all


def load_analysis_tables(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    file_map = {
        "overall": "mf_wf_analysis_overall_comparison.csv",
        "excess": "mf_wf_analysis_excess_comparison.csv",
        "yearly": "mf_wf_analysis_yearly_comparison.csv",
        "yearly_excess": "mf_wf_analysis_yearly_excess.csv",
        "selected_frequency": "mf_wf_analysis_selected_frequency.csv",
        "parameter_frequency": "mf_wf_analysis_parameter_frequency.csv",
        "benchmark_filter_frequency": "mf_wf_analysis_benchmark_filter_frequency.csv",
        "single_stock_contribution": "mf_wf_analysis_single_stock_contribution.csv",
        "combined_daily_returns": "mf_wf_analysis_combined_daily_returns.csv",
    }

    tables = {}
    for key, filename in file_map.items():
        path = analysis_dir / filename
        tables[key] = read_csv_required(path)

    # Numeric coercion per table
    if "total_return" in tables["overall"].columns:
        tables["overall"] = safe_to_numeric(tables["overall"], [
            "days", "total_return", "annual_return", "annual_volatility",
            "max_drawdown", "sharpe", "calmar",
        ])
    if "excess_total_return" in tables["excess"].columns:
        tables["excess"] = safe_to_numeric(tables["excess"], [
            "strategy_total_return", "benchmark_total_return", "excess_total_return",
            "strategy_annual_return", "benchmark_annual_return", "excess_annual_return",
            "strategy_max_drawdown", "benchmark_max_drawdown",
            "strategy_sharpe", "benchmark_sharpe",
        ])
    if "total_return" in tables["yearly"].columns:
        tables["yearly"] = safe_to_numeric(tables["yearly"], [
            "year", "days", "total_return", "annual_return", "annual_volatility",
            "max_drawdown", "sharpe", "calmar",
        ])
    if "excess_total_return" in tables["yearly_excess"].columns:
        tables["yearly_excess"] = safe_to_numeric(tables["yearly_excess"], [
            "year", "strategy_total_return", "benchmark_total_return", "excess_total_return",
            "strategy_annual_return", "benchmark_annual_return", "excess_annual_return",
            "strategy_max_drawdown", "benchmark_max_drawdown",
            "strategy_sharpe", "benchmark_sharpe",
        ])
    if "selected_count" in tables["selected_frequency"].columns:
        tables["selected_frequency"] = safe_to_numeric(tables["selected_frequency"], [
            "selected_count", "avg_rank", "best_rank",
            "avg_train_annual_return", "avg_train_max_drawdown",
            "avg_train_sharpe", "avg_train_score",
        ])
    if "selected_count" in tables["parameter_frequency"].columns:
        tables["parameter_frequency"] = safe_to_numeric(tables["parameter_frequency"], [
            "fast", "slow", "selected_count", "avg_rank",
            "avg_train_annual_return", "avg_train_max_drawdown",
            "avg_train_sharpe", "avg_train_score",
        ])
    if "avg_test_total_return" in tables["single_stock_contribution"].columns:
        tables["single_stock_contribution"] = safe_to_numeric(tables["single_stock_contribution"], [
            "selected_count", "avg_selected_rank",
            "avg_test_total_return", "sum_test_total_return",
            "median_test_total_return", "min_test_total_return", "max_test_total_return",
            "avg_test_max_drawdown", "avg_test_sharpe",
            "win_year_count", "avg_train_score",
            "avg_test_excess_vs_buy_hold", "sum_test_excess_vs_buy_hold",
            "avg_test_excess_vs_stock_only", "sum_test_excess_vs_stock_only",
            "win_rate",
        ])

    return tables


def filter_exclude_year(df: pd.DataFrame, exclude_year: int | None, year_col: str = "year") -> pd.DataFrame:
    if exclude_year is None or exclude_year == 0:
        return df.copy()
    if year_col not in df.columns:
        return df.copy()
    return df[df[year_col] != exclude_year].copy()


# ---------------------------------------------------------------------------
# Diagnosis 1: Overall summary
# ---------------------------------------------------------------------------

def build_summary(overall: pd.DataFrame, excess: pd.DataFrame, exclude_year: int | None) -> pd.DataFrame:
    # Filter all_years period only
    all_years = overall[overall["period"] == "all_years"].copy()
    strategies = all_years[all_years["entity_type"] == "strategy"].copy()

    if strategies.empty:
        return pd.DataFrame()

    rows = []
    for _, s in strategies.iterrows():
        row = {
            "strategy": s["entity"],
            "period": "all_years",
            "total_return": s["total_return"],
            "annual_return": s["annual_return"],
            "annual_volatility": s["annual_volatility"],
            "max_drawdown": s["max_drawdown"],
            "sharpe": s["sharpe"],
            "calmar": s["calmar"],
        }

        # Attach excess vs each benchmark
        exc_all = excess[(excess["period"] == "all_years") & (excess["strategy"] == s["entity"])]
        for _, e in exc_all.iterrows():
            bench = e["benchmark"]
            short = BENCHMARK_SHORT.get(bench, bench)
            row[f"benchmark_{short}"] = bench
            row[f"excess_total_return_vs_{short}"] = e["excess_total_return"]
            row[f"excess_annual_return_vs_{short}"] = e["excess_annual_return"]

        # Diagnosis flags
        flags = []
        if s["annual_return"] < 0:
            flags.append("negative_return")
        if s["sharpe"] < 0.5:
            flags.append("low_sharpe")
        if s["max_drawdown"] < -0.30:
            flags.append("high_drawdown")

        # Check underperformance
        for _, e in exc_all.iterrows():
            if e["excess_total_return"] < 0:
                short = BENCHMARK_SHORT.get(e["benchmark"], e["benchmark"])
                flags.append(f"underperform_{short}")
                break

        # Check incomplete year sensitivity
        if exclude_year is not None:
            ex_all = overall[overall["period"] == f"exclude_incomplete_{exclude_year}"]
            ex_strat = ex_all[ex_all["entity_type"] == "strategy"]
            if not ex_strat.empty:
                ex_row = ex_strat.iloc[0]
                if abs(ex_row["annual_return"] - s["annual_return"]) > 0.02:
                    flags.append("incomplete_year_sensitive")

        row["diagnosis_flags"] = "; ".join(flags) if flags else "none"
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnosis 2: Yearly weakness
# ---------------------------------------------------------------------------

def build_yearly_weakness(
    yearly: pd.DataFrame,
    yearly_excess: pd.DataFrame,
    exclude_year: int | None,
) -> pd.DataFrame:
    yearly_f = filter_exclude_year(yearly, exclude_year)
    yearly_excess_f = filter_exclude_year(yearly_excess, exclude_year)

    strat_yearly = yearly_f[yearly_f["entity_type"] == "strategy"].copy()
    if strat_yearly.empty:
        return pd.DataFrame()

    rows = []
    for _, sy in strat_yearly.iterrows():
        year = sy["year"]
        year_excess = yearly_excess_f[yearly_excess_f["year"] == year]

        for _, ye in year_excess.iterrows():
            weakness_reasons = []

            if sy["total_return"] < 0:
                weakness_reasons.append("lost_money")
            if sy["max_drawdown"] < -0.20:
                weakness_reasons.append("deep_drawdown")
            if ye["excess_total_return"] < 0:
                weakness_reasons.append("underperformed_benchmark")
            if ye["benchmark_total_return"] > 0 and sy["total_return"] < 0:
                weakness_reasons.append("benchmark_up_strategy_down")
            if sy.get("sharpe", 0) < 0:
                weakness_reasons.append("negative_sharpe")

            is_excluded = bool(exclude_year is not None and year == exclude_year)
            if is_excluded:
                weakness_reasons.append("incomplete_year")

            rows.append({
                "year": int(year),
                "is_excluded_year": is_excluded,
                "strategy_total_return": sy["total_return"],
                "strategy_max_drawdown": sy["max_drawdown"],
                "strategy_sharpe": sy.get("sharpe", np.nan),
                "benchmark": ye["benchmark"],
                "benchmark_total_return": ye["benchmark_total_return"],
                "excess_total_return": ye["excess_total_return"],
                "beat_benchmark": ye["beat_benchmark"],
                "weakness_reason": "; ".join(weakness_reasons) if weakness_reasons else "ok",
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["year", "excess_total_return"])
    return result


# ---------------------------------------------------------------------------
# Diagnosis 3: Train-test gap
# ---------------------------------------------------------------------------

def build_train_test_gap(detail: pd.DataFrame, exclude_year: int | None) -> pd.DataFrame:
    # test_detail has both train and test columns
    df = detail.copy()
    if exclude_year is not None and "test_year" in df.columns:
        df = df[df["test_year"] != exclude_year]

    if df.empty:
        return pd.DataFrame()

    required = ["train_annual_return", "test_total_return", "train_sharpe", "test_sharpe"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"test_detail 缺少字段：{missing}")

    df["train_test_gap"] = df["test_total_return"] - df["train_annual_return"]
    df["sharpe_gap"] = df["test_sharpe"] - df["train_sharpe"]

    # Gap flag
    def gap_flag(row):
        train_good = row["train_annual_return"] > 0.10 and row["train_sharpe"] > 0.5
        test_bad = row["test_total_return"] < 0
        if train_good and test_bad:
            return "train_good_test_bad"
        if train_good and not test_bad:
            return "train_good_test_ok"
        if not train_good and not test_bad:
            return "train_bad_test_good"
        return "neutral"

    df["gap_flag"] = df.apply(gap_flag, axis=1)

    out_cols = [
        "market_group", "symbol", "test_year", "selected_rank",
        "fast", "slow", "benchmark", "benchmark_fast", "benchmark_slow",
        "train_score", "train_annual_return", "train_sharpe",
        "train_excess_vs_stock_only_total_return",
        "test_total_return", "test_excess_vs_stock_only_total_return",
        "train_test_gap", "sharpe_gap", "gap_flag",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    result = df[out_cols].copy()
    result = result.sort_values("train_test_gap")
    return result


def build_train_test_correlation(gap_df: pd.DataFrame) -> pd.DataFrame:
    if gap_df.empty:
        return pd.DataFrame()

    rows = []

    pairs = [
        ("train_score", "test_total_return"),
        ("train_annual_return", "test_total_return"),
        ("train_sharpe", "test_total_return"),
        ("train_excess_vs_stock_only_total_return", "test_excess_vs_stock_only_total_return"),
        ("train_excess_vs_buy_hold_total_return", "test_excess_vs_buy_hold_total_return"),
    ]

    for train_col, test_col in pairs:
        if train_col in gap_df.columns and test_col in gap_df.columns:
            valid = gap_df[[train_col, test_col]].dropna()
            if len(valid) >= 3:
                corr = valid[train_col].corr(valid[test_col])
            else:
                corr = np.nan
            rows.append({
                "train_metric": train_col,
                "test_metric": test_col,
                "correlation": corr,
                "n_samples": len(valid),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnosis 4: Bad / Good contributors
# ---------------------------------------------------------------------------

def build_contributors(detail: pd.DataFrame, exclude_year: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = detail.copy()
    if exclude_year is not None and "test_year" in df.columns:
        df = df[df["test_year"] != exclude_year]

    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

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
        "avg_train_annual_return": ("train_annual_return", "mean"),
        "avg_train_sharpe": ("train_sharpe", "mean"),
    }

    if "test_excess_vs_buy_hold_total_return" in df.columns:
        agg_dict["avg_test_excess_vs_buy_hold"] = ("test_excess_vs_buy_hold_total_return", "mean")
    if "test_excess_vs_stock_only_total_return" in df.columns:
        agg_dict["avg_test_excess_vs_stock_only"] = ("test_excess_vs_stock_only_total_return", "mean")

    result = df.groupby(["market_group", "symbol"]).agg(**agg_dict).reset_index()
    result["win_rate"] = result["win_year_count"] / result["selected_count"]

    # Bad contributors: negative sum return or low win rate
    bad = result[
        (result["sum_test_total_return"] < 0)
        | ((result["win_rate"] < 0.5) & (result["selected_count"] >= 2))
    ].copy()

    def bad_reason(row):
        reasons = []
        if row["sum_test_total_return"] < 0:
            reasons.append("negative_total_contribution")
        if row["win_rate"] < 0.5 and row["selected_count"] >= 2:
            reasons.append("low_win_rate")
        if row["avg_test_max_drawdown"] < -0.20:
            reasons.append("deep_avg_drawdown")
        if "avg_test_excess_vs_buy_hold" in row.index and pd.notna(row.get("avg_test_excess_vs_buy_hold")) and row["avg_test_excess_vs_buy_hold"] < 0:
            reasons.append("negative_avg_excess_vs_buy_hold")
        return "; ".join(reasons) if reasons else "unknown"

    if not bad.empty:
        bad["bad_reason"] = bad.apply(bad_reason, axis=1)
    bad = bad.sort_values("sum_test_total_return")

    # Good contributors: positive sum return and decent win rate
    good = result[
        (result["sum_test_total_return"] > 0)
        & (result["win_rate"] >= 0.5)
    ].copy()
    good = good.sort_values("sum_test_total_return", ascending=False)

    return bad, good


# ---------------------------------------------------------------------------
# Diagnosis 5: Parameter stability
# ---------------------------------------------------------------------------

def build_parameter_stability(
    param_freq: pd.DataFrame,
    detail: pd.DataFrame,
    exclude_year: int | None,
) -> pd.DataFrame:
    df = detail.copy()
    if exclude_year is not None and "test_year" in df.columns:
        df = df[df["test_year"] != exclude_year]

    if df.empty:
        return pd.DataFrame()

    if "fast" not in df.columns or "slow" not in df.columns:
        return pd.DataFrame()

    df["param"] = df["fast"].astype(int).astype(str) + "/" + df["slow"].astype(int).astype(str)

    agg_dict = {
        "selected_count": ("param", "size"),
        "years": ("test_year", lambda x: ",".join(map(str, sorted(pd.Series(x).dropna().astype(int).unique())))),
        "avg_test_total_return": ("test_total_return", "mean"),
        "median_test_total_return": ("test_total_return", "median"),
        "avg_test_sharpe": ("test_sharpe", "mean"),
        "win_year_count": ("test_total_return", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
    }

    if "test_excess_vs_stock_only_total_return" in df.columns:
        agg_dict["avg_test_excess_vs_stock_only"] = ("test_excess_vs_stock_only_total_return", "mean")

    result = df.groupby(["market_group", "fast", "slow", "param"]).agg(**agg_dict).reset_index()
    result["win_rate"] = result["win_year_count"] / result["selected_count"]

    def stability_label(row):
        if row["selected_count"] >= 3 and row["win_rate"] >= 0.5 and row["avg_test_total_return"] > 0:
            return "relatively_stable"
        return "unstable_or_weak"

    result["stability_label"] = result.apply(stability_label, axis=1)
    result = result.sort_values(["selected_count", "avg_test_total_return"], ascending=[False, False])
    return result


# ---------------------------------------------------------------------------
# Diagnosis 6: Market filter effectiveness
# ---------------------------------------------------------------------------

def build_market_filter_effectiveness(
    detail: pd.DataFrame,
    exclude_year: int | None,
) -> pd.DataFrame:
    df = detail.copy()
    if exclude_year is not None and "test_year" in df.columns:
        df = df[df["test_year"] != exclude_year]

    if df.empty:
        return pd.DataFrame()

    rows = []

    # Per-benchmark effectiveness
    if "benchmark" in df.columns:
        for bench, group in df.groupby("benchmark"):
            test_ret = pd.to_numeric(group["test_total_return"], errors="coerce").dropna()
            test_excess = pd.to_numeric(group.get("test_excess_vs_stock_only_total_return", pd.Series(dtype=float)), errors="coerce").dropna()

            row = {
                "kind": "benchmark",
                "value": bench,
                "count": len(group),
                "avg_test_total_return": float(test_ret.mean()) if len(test_ret) > 0 else np.nan,
                "median_test_total_return": float(test_ret.median()) if len(test_ret) > 0 else np.nan,
                "avg_test_excess_vs_stock_only": float(test_excess.mean()) if len(test_excess) > 0 else np.nan,
                "win_rate": float((test_ret > 0).mean()) if len(test_ret) > 0 else np.nan,
            }

            # Market filter ratio stats
            for col in ["train_market_filter_on_ratio", "test_market_filter_on_ratio",
                         "train_strategy_exposure_ratio", "test_strategy_exposure_ratio"]:
                if col in group.columns:
                    vals = pd.to_numeric(group[col], errors="coerce").dropna()
                    if len(vals) > 0:
                        row[f"{col}_mean"] = float(vals.mean())
                        row[f"{col}_median"] = float(vals.median())

            rows.append(row)

    # Per benchmark_fast/slow
    if "benchmark_fast" in df.columns and "benchmark_slow" in df.columns:
        df["_bf_bs"] = df["benchmark_fast"].astype(str) + "/" + df["benchmark_slow"].astype(str)
        for bs, group in df.groupby("_bf_bs"):
            test_ret = pd.to_numeric(group["test_total_return"], errors="coerce").dropna()
            test_excess = pd.to_numeric(group.get("test_excess_vs_stock_only_total_return", pd.Series(dtype=float)), errors="coerce").dropna()

            row = {
                "kind": "benchmark_fast/slow",
                "value": str(bs),
                "count": len(group),
                "avg_test_total_return": float(test_ret.mean()) if len(test_ret) > 0 else np.nan,
                "median_test_total_return": float(test_ret.median()) if len(test_ret) > 0 else np.nan,
                "avg_test_excess_vs_stock_only": float(test_excess.mean()) if len(test_excess) > 0 else np.nan,
                "win_rate": float((test_ret > 0).mean()) if len(test_ret) > 0 else np.nan,
            }
            rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["kind", "count"], ascending=[True, False])
    return result


# ---------------------------------------------------------------------------
# Diagnosis 7: Selected repetition
# ---------------------------------------------------------------------------

def build_selected_repetition(
    selected_freq: pd.DataFrame,
    detail: pd.DataFrame,
    exclude_year: int | None,
) -> pd.DataFrame:
    if selected_freq.empty:
        return pd.DataFrame()

    result = selected_freq.copy()

    # Merge test performance from test_detail raw rows.
    df = detail.copy()
    if exclude_year is not None and "test_year" in df.columns:
        df = df[df["test_year"] != exclude_year]

    if not df.empty and "symbol" in df.columns:
        test_agg = df.groupby(["market_group", "symbol"]).agg(
            raw_selected_count=("symbol", "size"),
            avg_test_total_return=("test_total_return", "mean"),
            sum_test_total_return=("test_total_return", "sum"),
            avg_test_sharpe=("test_sharpe", "mean"),
            avg_test_excess_vs_stock_only=("test_excess_vs_stock_only_total_return", "mean"),
        ).reset_index()

        result = result.merge(test_agg, on=["market_group", "symbol"], how="left")

    result = result.sort_values("selected_count", ascending=False)
    return result


# ---------------------------------------------------------------------------
# PNG charts
# ---------------------------------------------------------------------------

def make_train_vs_test_scatter(gap_df: pd.DataFrame, output_dir: Path, top_n: int) -> Path | None:
    if gap_df.empty:
        return None

    path = output_dir / "ma_mf_diagnosis_train_vs_test_scatter.png"

    valid = gap_df.dropna(subset=["train_annual_return", "test_total_return"])
    if valid.empty:
        return None

    plt.figure(figsize=(10, 8))
    for mg, group in valid.groupby("market_group"):
        plt.scatter(group["train_annual_return"], group["test_total_return"],
                    alpha=0.5, s=20, label=mg)

    # Diagonal reference line
    lims = [
        min(valid["train_annual_return"].min(), valid["test_total_return"].min()) - 0.05,
        max(valid["train_annual_return"].max(), valid["test_total_return"].max()) + 0.05,
    ]
    plt.plot(lims, lims, "k--", alpha=0.3, label="train=test")
    plt.axhline(0, color="red", alpha=0.3, linewidth=0.5)
    plt.axvline(0, color="red", alpha=0.3, linewidth=0.5)

    plt.xlabel("Train Annual Return")
    plt.ylabel("Test Total Return")
    plt.title("MA v2 Train vs Test Return Scatter")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def make_yearly_excess_heatmap(yearly_weakness: pd.DataFrame, output_dir: Path) -> Path | None:
    if yearly_weakness.empty:
        return None

    path = output_dir / "ma_mf_diagnosis_yearly_excess_heatmap.png"

    # Pivot: rows=benchmark, cols=year, values=excess_total_return
    pivot = yearly_weakness.pivot_table(
        index="benchmark", columns="year", values="excess_total_return", aggfunc="first"
    ).sort_index()

    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(3, len(pivot.index) * 0.8)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.3, vmax=0.3)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    # Shorten benchmark labels for display
    short_labels = []
    for b in pivot.index:
        short = BENCHMARK_SHORT.get(b, b)
        short_labels.append(short)
    ax.set_yticklabels(short_labels)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > 0.15 else "black")

    plt.colorbar(im, ax=ax, label="Excess Total Return")
    ax.set_title("MA v2 Yearly Excess Return Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def make_top_draggers_chart(contrib: pd.DataFrame, output_dir: Path, top_n: int) -> Path | None:
    if contrib.empty:
        return None

    path = output_dir / "ma_mf_diagnosis_top_draggers.png"

    tmp = contrib.copy()
    tmp["label"] = tmp["market_group"].astype(str) + " " + tmp["symbol"].astype(str)
    tmp = tmp.sort_values("sum_test_total_return").head(top_n)

    if tmp.empty:
        return None

    plt.figure(figsize=(12, max(6, len(tmp) * 0.35)))
    colors = ["red" if v < 0 else "green" for v in tmp["sum_test_total_return"]]
    plt.barh(tmp["label"], tmp["sum_test_total_return"], color=colors, alpha=0.7)
    plt.xlabel("Sum Test Total Return")
    plt.title(f"MA v2 Top {top_n} Stock Contributors / Draggers")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def make_parameter_stability_chart(param_stab: pd.DataFrame, output_dir: Path) -> Path | None:
    if param_stab.empty:
        return None

    path = output_dir / "ma_mf_diagnosis_parameter_stability.png"

    tmp = param_stab.copy()
    tmp["label"] = tmp["param"].astype(str) + " (" + tmp["stability_label"].astype(str) + ")"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    colors = ["green" if s == "relatively_stable" else "red" for s in tmp["stability_label"]]

    axes[0].bar(tmp["label"], tmp["avg_test_total_return"], color=colors, alpha=0.7)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Avg Test Total Return")
    axes[0].set_title("MA v2 Parameter Stability: Avg Test Return by Fast/Slow")

    axes[1].bar(tmp["label"], tmp["win_rate"], color=colors, alpha=0.7)
    axes[1].axhline(0.5, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("Win Rate")
    axes[1].set_xlabel("Parameter (fast/slow)")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def make_market_filter_chart(filter_eff: pd.DataFrame, output_dir: Path) -> Path | None:
    if filter_eff.empty:
        return None

    path = output_dir / "ma_mf_diagnosis_market_filter_effectiveness.png"

    bench_rows = filter_eff[filter_eff["kind"] == "benchmark"].copy()
    if bench_rows.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Chart 1: avg test return by benchmark
    axes[0].bar(bench_rows["value"], bench_rows["avg_test_total_return"], alpha=0.7,
                color=["green" if v > 0 else "red" for v in bench_rows["avg_test_total_return"]])
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Avg Test Total Return")
    axes[0].set_title("Avg Test Return by Filter Benchmark")
    axes[0].tick_params(axis="x", rotation=30)

    # Chart 2: count by benchmark
    axes[1].bar(bench_rows["value"], bench_rows["count"], alpha=0.7, color="steelblue")
    axes[1].set_ylabel("Selection Count")
    axes[1].set_title("Filter Benchmark Usage Frequency")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(output_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    csv_map = {
        "summary": "ma_mf_diagnosis_summary.csv",
        "yearly_weakness": "ma_mf_diagnosis_yearly_weakness.csv",
        "train_test_gap": "ma_mf_diagnosis_train_test_gap.csv",
        "train_test_correlation": "ma_mf_diagnosis_train_test_correlation.csv",
        "bad_contributors": "ma_mf_diagnosis_bad_contributors.csv",
        "good_contributors": "ma_mf_diagnosis_good_contributors.csv",
        "parameter_stability": "ma_mf_diagnosis_parameter_stability.csv",
        "market_filter_effectiveness": "ma_mf_diagnosis_market_filter_effectiveness.csv",
        "selected_repetition": "ma_mf_diagnosis_selected_repetition.csv",
    }

    for key, filename in csv_map.items():
        if key in tables and not tables[key].empty:
            path = output_dir / filename
            tables[key].to_csv(path, index=False, encoding="utf-8-sig")
            paths[key] = path

    return paths


# ---------------------------------------------------------------------------
# Recommendations report
# ---------------------------------------------------------------------------

def write_recommendations(
    output_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    yearly_weakness: pd.DataFrame,
    gap_df: pd.DataFrame,
    gap_corr: pd.DataFrame,
    bad_contrib: pd.DataFrame,
    good_contrib: pd.DataFrame,
    param_stab: pd.DataFrame,
    filter_eff: pd.DataFrame,
    selected_rep: pd.DataFrame,
    table_paths: dict[str, Path],
    plot_paths: dict[str, Path],
    exclude_year: int | None,
) -> Path:
    path = output_dir / "ma_mf_diagnosis_recommendations.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("MA v2 Strategy Diagnosis Report\n")
        f.write("=" * 80 + "\n\n")

        # Section 1: Settings
        f.write("1. Diagnosis Settings\n")
        f.write("-" * 80 + "\n")
        f.write(f"analysis_dir: {args.analysis_dir}\n")
        f.write(f"walk_forward_dir: {args.walk_forward_dir}\n")
        f.write(f"output_dir: {output_dir}\n")
        f.write(f"markets: {args.markets}\n")
        f.write(f"portfolio_size: {args.portfolio_size}\n")
        f.write(f"benchmarks: {args.benchmarks}\n")
        f.write(f"exclude_year: {exclude_year}\n")
        f.write(f"run_id: {args.run_id}\n\n")

        # Section 2: Overall summary
        f.write("2. Overall Strategy Summary\n")
        f.write("-" * 80 + "\n")
        if not summary.empty:
            display_cols = ["strategy", "total_return", "annual_return", "max_drawdown", "sharpe", "diagnosis_flags"]
            display_cols = [c for c in display_cols if c in summary.columns]
            f.write(summary[display_cols].to_string(index=False))

            s = summary.iloc[0]
            f.write(f"\n\nKey findings:\n")
            f.write(f"  Annual return: {format_pct(s.get('annual_return', np.nan))}\n")
            f.write(f"  Max drawdown: {format_pct(s.get('max_drawdown', np.nan))}\n")
            f.write(f"  Sharpe: {format_float(s.get('sharpe', np.nan))}\n")
            f.write(f"  Diagnosis: {s.get('diagnosis_flags', 'N/A')}\n")

            # Excess
            for col in summary.columns:
                if col.startswith("excess_total_return_vs_"):
                    bench_short = col.replace("excess_total_return_vs_", "")
                    val = s[col]
                    f.write(f"  Excess vs {bench_short}: {format_pct(val)}\n")
        else:
            f.write("No summary data available.\n")
        f.write("\n")

        # Section 3: Yearly weakness
        f.write("3. Yearly Weakness Analysis\n")
        f.write("-" * 80 + "\n")
        if not yearly_weakness.empty:
            f.write(yearly_weakness.to_string(index=False))

            # Count weakness reasons
            all_reasons = "; ".join(yearly_weakness["weakness_reason"].tolist())
            f.write(f"\n\nWeakness patterns:\n")
            for reason in ["lost_money", "deep_drawdown", "underperformed_benchmark",
                           "benchmark_up_strategy_down", "negative_sharpe"]:
                count = sum(1 for r in yearly_weakness["weakness_reason"] if reason in r)
                if count > 0:
                    f.write(f"  {reason}: {count} occurrences\n")
        else:
            f.write("No yearly weakness data.\n")
        f.write("\n")

        # Section 4: Train-test gap
        f.write("4. Train-Test Gap Analysis\n")
        f.write("-" * 80 + "\n")
        if not gap_df.empty:
            gap_counts = gap_df["gap_flag"].value_counts()
            f.write("Gap flag distribution:\n")
            for flag, count in gap_counts.items():
                f.write(f"  {flag}: {count}\n")

            n_total = len(gap_df)
            n_bad = gap_counts.get("train_good_test_bad", 0)
            bad_rate = n_bad / n_total if n_total > 0 else 0
            f.write(f"\nTrain-good-test-bad rate: {bad_rate:.1%} ({n_bad}/{n_total})\n")

            if bad_rate > 0.20:
                f.write("DIAGNOSIS: Overfitting signal is strong (>20% of train-good selections fail in test).\n")
            elif bad_rate > 0.10:
                f.write("DIAGNOSIS: Moderate overfitting signal (10-20%).\n")
            else:
                f.write("DIAGNOSIS: Overfitting signal is weak (<10%). Problem may be elsewhere.\n")
        else:
            f.write("No train-test gap data.\n")

        # Correlation
        if not gap_corr.empty:
            f.write("\nTrain-test correlation:\n")
            f.write(gap_corr.to_string(index=False))
            for _, row in gap_corr.iterrows():
                corr = row["correlation"]
                if pd.notna(corr) and corr < 0.1:
                    f.write(f"\n  WARNING: {row['train_metric']} -> {row['test_metric']} correlation is {corr:.3f} (near zero). "
                            "Training score has almost no predictive power for test returns.")
        f.write("\n\n")

        # Section 5: Contributors
        f.write("5. Stock Contributors and Draggers\n")
        f.write("-" * 80 + "\n")
        if not bad_contrib.empty:
            f.write(f"Bad contributors ({len(bad_contrib)} stocks):\n")
            show_cols = ["symbol", "selected_count", "sum_test_total_return", "win_rate", "bad_reason"]
            show_cols = [c for c in show_cols if c in bad_contrib.columns]
            f.write(bad_contrib[show_cols].head(20).to_string(index=False))
        else:
            f.write("No bad contributors identified.\n")
        f.write("\n\n")
        if not good_contrib.empty:
            f.write(f"Good contributors ({len(good_contrib)} stocks):\n")
            show_cols = ["symbol", "selected_count", "sum_test_total_return", "win_rate"]
            show_cols = [c for c in show_cols if c in good_contrib.columns]
            f.write(good_contrib[show_cols].head(20).to_string(index=False))
        f.write("\n\n")

        # Section 6: Parameter stability
        f.write("6. Parameter Stability\n")
        f.write("-" * 80 + "\n")
        if not param_stab.empty:
            f.write(param_stab.to_string(index=False))
            n_stable = len(param_stab[param_stab["stability_label"] == "relatively_stable"])
            n_total = len(param_stab)
            f.write(f"\n\n{n_stable}/{n_total} parameter combinations are relatively stable.\n")
            if n_stable == 0:
                f.write("DIAGNOSIS: No parameter combination shows stable out-of-sample performance.\n")
        else:
            f.write("No parameter stability data.\n")
        f.write("\n\n")

        # Section 7: Market filter effectiveness
        f.write("7. Market Filter Effectiveness\n")
        f.write("-" * 80 + "\n")
        if not filter_eff.empty:
            f.write(filter_eff.to_string(index=False))

            # Check filter ratio stats
            for col in ["train_market_filter_on_ratio_mean", "test_market_filter_on_ratio_mean"]:
                if col in filter_eff.columns:
                    vals = filter_eff[col].dropna()
                    if len(vals) > 0:
                        f.write(f"\n  {col}: overall mean = {vals.mean():.4f}\n")
        else:
            f.write("No market filter effectiveness data.\n")
        f.write("\n\n")

        # Section 8: Selected repetition
        f.write("8. Selection Repetition\n")
        f.write("-" * 80 + "\n")
        if not selected_rep.empty:
            high_rep = selected_rep[selected_rep["selected_count"] >= 3]
            f.write(f"Stocks selected 3+ times: {len(high_rep)}\n")
            if not high_rep.empty:
                show_cols = ["symbol", "selected_count", "avg_train_score"]
                if "avg_test_total_return" in high_rep.columns:
                    show_cols.append("avg_test_total_return")
                show_cols = [c for c in show_cols if c in high_rep.columns]
                f.write(high_rep[show_cols].to_string(index=False))
        else:
            f.write("No selected repetition data.\n")
        f.write("\n\n")

        # Section 9: Recommendations
        f.write("9. Recommendations\n")
        f.write("-" * 80 + "\n")

        # Data-driven recommendations
        recommendations = []

        # Check if strategy is worth continuing
        if not summary.empty:
            s = summary.iloc[0]
            annual = s.get("annual_return", 0)
            sharpe = s.get("sharpe", 0)
            mdd = s.get("max_drawdown", -1)

            if annual < 0 and sharpe < 0:
                recommendations.append(
                    "CRITICAL: Strategy has negative annual return and negative Sharpe. "
                    "The core signal (MA crossover + market filter) is not generating alpha in this period. "
                    "Entering portfolio_backtest_csv.py is NOT recommended at this stage."
                )
            elif annual < 0.03:
                recommendations.append(
                    "CAUTION: Annual return is below 3%. Strategy may not be worth the complexity "
                    "of real-world implementation. Further alpha research needed before portfolio testing."
                )

        # Overfitting diagnosis
        if not gap_df.empty:
            n_total = len(gap_df)
            n_bad = sum(gap_df["gap_flag"] == "train_good_test_bad")
            if n_total > 0 and n_bad / n_total > 0.20:
                recommendations.append(
                    f"OVERFITTING: {n_bad}/{n_total} ({n_bad/n_total:.0%}) of train-good selections fail in test. "
                    "Consider: (a) stronger train-period filters, (b) walk-forward anchoring, "
                    "(c) reducing parameter grid to fewer, more robust combos."
                )

        # Parameter stability
        if not param_stab.empty:
            n_stable = len(param_stab[param_stab["stability_label"] == "relatively_stable"])
            if n_stable == 0:
                recommendations.append(
                    "PARAMETER INSTABILITY: No fast/slow combination is consistently profitable OOS. "
                    "The MA crossover signal itself may be weak. Consider alternative entry signals "
                    "(momentum, volume breakout, mean reversion) or adding confirming indicators."
                )

        # Market filter diagnosis
        if not filter_eff.empty:
            bench_rows = filter_eff[filter_eff["kind"] == "benchmark"]
            if not bench_rows.empty:
                worst_bench = bench_rows.loc[bench_rows["avg_test_total_return"].idxmin()]
                best_bench = bench_rows.loc[bench_rows["avg_test_total_return"].idxmax()]
                recommendations.append(
                    f"MARKET FILTER: Best filter benchmark is {best_bench['value']} "
                    f"(avg test return {best_bench['avg_test_total_return']:.2%}), "
                    f"worst is {worst_bench['value']} ({worst_bench['avg_test_total_return']:.2%}). "
                    "Consider using only the best-performing benchmark as filter."
                )

        # Contributor diagnosis
        if not bad_contrib.empty:
            n_draggers = len(bad_contrib[bad_contrib["sum_test_total_return"] < -0.1])
            if n_draggers > 5:
                recommendations.append(
                    f"STOCK SELECTION: {n_draggers} stocks contributed <-10% total return. "
                    "Consider adding a minimum test-period drawdown filter or position sizing "
                    "based on train-period volatility."
                )

        # What NOT to do
        recommendations.append("")
        recommendations.append("What NOT to do now:")
        recommendations.append("  - Do NOT add industry/sector constraints (the signal is already weak)")
        recommendations.append("  - Do NOT increase portfolio_size (more weak stocks won't help)")
        recommendations.append("  - Do NOT move to live trading or QMT integration")

        # Should we proceed to portfolio backtest?
        recommendations.append("")
        if not summary.empty:
            s = summary.iloc[0]
            if s.get("annual_return", 0) > 0.03 and s.get("sharpe", 0) > 0.3:
                recommendations.append(
                    "PROCEED to portfolio_backtest_csv.py? MAYBE - strategy shows marginal viability. "
                    "Run a few more WF experiments first."
                )
            else:
                recommendations.append(
                    "PROCEED to portfolio_backtest_csv.py? NO - strategy does not meet minimum "
                    "viability thresholds (annual > 3%, Sharpe > 0.3). Fix the alpha signal first."
                )

        for rec in recommendations:
            f.write(f"  {rec}\n")

        # Section 10: Output files
        f.write("\n\n10. Output Files\n")
        f.write("-" * 80 + "\n")
        for name, p in table_paths.items():
            f.write(f"  {name}: {p}\n")
        for name, p in plot_paths.items():
            f.write(f"  {name}: {p}\n")

    return path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断 MA v2 walk-forward 策略表现")

    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR), help="MA v2 分析结果目录")
    parser.add_argument("--walk-forward-dir", default=str(DEFAULT_WF_DIR), help="MA v2 walk-forward 原始输出目录")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="诊断输出根目录")

    parser.add_argument("--markets", default="ALL", help="市场组合")
    parser.add_argument("--portfolio-size", type=int, default=20, help="组合数量")
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARK_LIST, help="基准指数列表")
    parser.add_argument("--exclude-year", type=int, default=2026, help="排除年份（0=不排除）")
    parser.add_argument("--run-id", default="", help="运行标识（默认自动生成时间戳）")
    parser.add_argument("--top-n-chart", type=int, default=30, help="图表中显示的股票数量")
    parser.add_argument("--no-png", action="store_true", help="不生成 PNG 图")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    analysis_dir = resolve_path(args.analysis_dir)
    wf_dir = resolve_path(args.walk_forward_dir)
    output_root = resolve_path(args.output_root)

    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"ma_mf_diagnosis_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    markets = parse_list(args.markets)
    benchmarks = parse_list(args.benchmarks)
    exclude_year = args.exclude_year if args.exclude_year > 0 else None

    logger.info("MA v2 Diagnosis Configuration:")
    logger.info("  analysis_dir: %s", analysis_dir)
    logger.info("  walk_forward_dir: %s", wf_dir)
    logger.info("  output_dir: %s", output_dir)
    logger.info("  markets: %s", markets)
    logger.info("  portfolio_size: %s", args.portfolio_size)
    logger.info("  benchmarks: %s", benchmarks)
    logger.info("  exclude_year: %s", exclude_year)
    logger.info("  run_id: %s", run_id)

    # Load data
    logger.info("Loading data...")
    tables = load_analysis_tables(analysis_dir)
    selected, detail = load_walk_forward_raw(wf_dir, markets, args.portfolio_size)
    logger.info("  Loaded %d selected_by_year rows, %d test_detail rows", len(selected), len(detail))

    # Build diagnoses
    # NOTE: selected_by_year only has train columns; test_detail has both train+test.
    # Use test_detail for functions that need test data (gap, contributors, parameters, filter, repetition).
    logger.info("Building diagnoses...")

    summary = build_summary(tables["overall"], tables["excess"], exclude_year)
    logger.info("  [1] Summary: %d rows", len(summary))

    yearly_weakness = build_yearly_weakness(tables["yearly"], tables["yearly_excess"], exclude_year)
    logger.info("  [2] Yearly weakness: %d rows", len(yearly_weakness))

    gap_df = build_train_test_gap(detail, exclude_year)
    gap_corr = build_train_test_correlation(gap_df)
    logger.info("  [3] Train-test gap: %d rows, correlation: %d pairs", len(gap_df), len(gap_corr))

    bad_contrib, good_contrib = build_contributors(detail, exclude_year)
    logger.info("  [4] Bad contributors: %d, Good contributors: %d", len(bad_contrib), len(good_contrib))

    param_stab = build_parameter_stability(tables["parameter_frequency"], detail, exclude_year)
    logger.info("  [5] Parameter stability: %d rows", len(param_stab))

    filter_eff = build_market_filter_effectiveness(detail, exclude_year)
    logger.info("  [6] Market filter effectiveness: %d rows", len(filter_eff))

    selected_rep = build_selected_repetition(tables["selected_frequency"], detail, exclude_year)
    logger.info("  [7] Selected repetition: %d rows", len(selected_rep))

    # Save CSVs
    diag_tables = {
        "summary": summary,
        "yearly_weakness": yearly_weakness,
        "train_test_gap": gap_df,
        "train_test_correlation": gap_corr,
        "bad_contributors": bad_contrib,
        "good_contributors": good_contrib,
        "parameter_stability": param_stab,
        "market_filter_effectiveness": filter_eff,
        "selected_repetition": selected_rep,
    }
    table_paths = save_outputs(output_dir, diag_tables)
    logger.info("Saved %d CSV files to %s", len(table_paths), output_dir)

    # Generate charts
    plot_paths = {}
    if not args.no_png:
        for name, func, kwargs in [
            ("train_vs_test_scatter", make_train_vs_test_scatter, {"gap_df": gap_df, "output_dir": output_dir, "top_n": args.top_n_chart}),
            ("yearly_excess_heatmap", make_yearly_excess_heatmap, {"yearly_weakness": yearly_weakness, "output_dir": output_dir}),
            ("top_draggers", make_top_draggers_chart, {"contrib": bad_contrib, "output_dir": output_dir, "top_n": args.top_n_chart}),
            ("parameter_stability", make_parameter_stability_chart, {"param_stab": param_stab, "output_dir": output_dir}),
            ("market_filter_effectiveness", make_market_filter_chart, {"filter_eff": filter_eff, "output_dir": output_dir}),
        ]:
            result = func(**kwargs)
            if result is not None:
                plot_paths[name] = result
        logger.info("Generated %d PNG charts", len(plot_paths))
    else:
        logger.info("PNG generation skipped (--no-png)")

    # Write recommendations
    report_path = write_recommendations(
        output_dir=output_dir,
        args=args,
        summary=summary,
        yearly_weakness=yearly_weakness,
        gap_df=gap_df,
        gap_corr=gap_corr,
        bad_contrib=bad_contrib,
        good_contrib=good_contrib,
        param_stab=param_stab,
        filter_eff=filter_eff,
        selected_rep=selected_rep,
        table_paths=table_paths,
        plot_paths=plot_paths,
        exclude_year=exclude_year,
    )
    logger.info("Recommendations report: %s", report_path)

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY DIAGNOSTIC FINDINGS")
    print("=" * 60)

    if not summary.empty:
        s = summary.iloc[0]
        print(f"  Annual return: {format_pct(s.get('annual_return', np.nan))}")
        print(f"  Max drawdown: {format_pct(s.get('max_drawdown', np.nan))}")
        print(f"  Sharpe: {format_float(s.get('sharpe', np.nan))}")
        print(f"  Diagnosis flags: {s.get('diagnosis_flags', 'N/A')}")

    if not gap_df.empty:
        n_total = len(gap_df)
        n_bad = sum(gap_df["gap_flag"] == "train_good_test_bad")
        print(f"  Train-good-test-bad rate: {n_bad}/{n_total} = {n_bad/n_total:.1%}" if n_total > 0 else "")

    if not param_stab.empty:
        n_stable = len(param_stab[param_stab["stability_label"] == "relatively_stable"])
        print(f"  Stable parameters: {n_stable}/{len(param_stab)}")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception as exc:
        logger.error("Program error: %s", repr(exc))
        raise
