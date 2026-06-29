# -*- coding: utf-8 -*-
"""
wf_robustness_shared.py

Alpha v6/v7 稳健性验证脚本的共享逻辑。

设计原则：
- 所有版本无关的工具函数、指标计算、分析构建、gate 评估、报告生成、图表绘制集中于此。
- 每个 robustness 脚本只需定义 WFRobustnessConfig 并调用共享函数。
- 与 wf_batch_shared.py（batch/validate 共享）、wf_report_shared.py（analysis/diagnosis 共享）对应。
"""

from __future__ import annotations

import argparse
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
from scripts.common.metrics import format_pct, format_float, calc_metrics_from_daily
from scripts.common.validation import resolve_path, parse_list


# ---------------------------------------------------------------------------
# 1. WFRobustnessConfig — 稳健性验证配置
# ---------------------------------------------------------------------------

@dataclass
class WFRobustnessConfig:
    """版本特定的配置差异点。"""

    # 默认目录
    default_wf_dir: Path
    default_analysis_dir: Path
    default_output_root: Path

    # 文件前缀（load_input_files 中 glob 匹配用）
    file_prefix: str  # e.g. "wf_alpha_v6_stock_"

    # 输出 CSV 前缀
    output_prefix: str  # e.g. "alpha_v6_robustness_"

    # 报告文件名
    report_filename: str  # e.g. "alpha_v6_robustness_report.txt"

    # 参数列名
    param_cols: list[str]

    # 显示名称（图表标题等）
    display_name: str  # e.g. "Alpha v6"

    # 默认 run_id
    default_run_id: str  # e.g. "exp006_alpha_v6_full"


def make_v6_config(project_root: Path) -> WFRobustnessConfig:
    return WFRobustnessConfig(
        default_wf_dir=project_root / "backtests" / "walk_forward_alpha_v6_research_csv",
        default_analysis_dir=project_root / "backtests" / "walk_forward_alpha_v6_research_analysis",
        default_output_root=project_root / "backtests" / "strategy_robustness",
        file_prefix="wf_alpha_v6_stock_",
        output_prefix="alpha_v6_robustness_",
        report_filename="alpha_v6_robustness_report.txt",
        param_cols=["alpha_variant", "momentum_window", "trend_ma",
                     "vol_window", "breakout_window", "benchmark", "benchmark_ma"],
        display_name="Alpha v6",
        default_run_id="exp006_alpha_v6_full",
    )


def make_v7_config(project_root: Path) -> WFRobustnessConfig:
    return WFRobustnessConfig(
        default_wf_dir=project_root / "backtests" / "walk_forward_alpha_v7_research_csv",
        default_analysis_dir=project_root / "backtests" / "walk_forward_alpha_v7_research_analysis",
        default_output_root=project_root / "backtests" / "strategy_robustness",
        file_prefix="wf_alpha_v7_stock_",
        output_prefix="alpha_v7_robustness_",
        report_filename="alpha_v7_robustness_report.txt",
        param_cols=["reversal_window", "vol_window", "turnover_short",
                     "turnover_long", "divergence_window"],
        display_name="Alpha v7",
        default_run_id="exp007_alpha_v7_full",
    )


# ---------------------------------------------------------------------------
# 2. Benchmark map（v6/v7 共用）
# ---------------------------------------------------------------------------

BENCHMARK_MAP = {
    "000300.SH": ("SH", "price_000300.csv", "CSI300"),
    "000905.SH": ("SH", "price_000905.csv", "CSI500"),
    "000852.SH": ("SH", "price_000852.csv", "CSI1000"),
}


# ---------------------------------------------------------------------------
# 3. 格式化工具 — imported from scripts.common.metrics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. 数据加载
# ---------------------------------------------------------------------------

def load_input_files(wf_dir: Path, input_tag: str, file_prefix: str) -> dict[str, pd.DataFrame]:
    """加载 walk-forward 输出文件。"""
    prefix = f"{file_prefix}{input_tag}"
    required = {
        "selected_by_year": f"{prefix}_selected_by_year.csv",
        "test_detail": f"{prefix}_test_detail.csv",
        "portfolio_daily": f"{prefix}_portfolio_daily.csv",
        "portfolio_period_summary": f"{prefix}_portfolio_period_summary.csv",
    }
    missing = []
    data = {}
    for key, filename in required.items():
        path = wf_dir / filename
        if not path.exists():
            missing.append(str(path))
        else:
            data[key] = read_csv_required(path)
    if missing:
        raise FileNotFoundError(
            "缺少以下必需文件:\n" + "\n".join(f"  - {p}" for p in missing)
        )
    return data


def load_benchmark_data(benchmarks: list[str], project_root: Path) -> dict[str, pd.DataFrame]:
    """加载基准价格数据。"""
    result = {}
    for bm in benchmarks:
        if bm not in BENCHMARK_MAP:
            continue
        exchange, filename, _ = BENCHMARK_MAP[bm]
        path = project_root / "data" / "qmt_export" / exchange / filename
        if not path.exists():
            print(f"  [WARNING] 基准文件不存在: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = safe_to_numeric(df, ["close"])
        df["date"] = pd.to_datetime(df["timetag"].astype(str), format="%Y%m%d")
        df = df.sort_values("date").reset_index(drop=True)
        result[bm] = df
    return result


# ---------------------------------------------------------------------------
# 5. 指标计算（从日收益）
# ---------------------------------------------------------------------------

def compute_metrics_from_daily(daily_returns: pd.Series) -> dict:
    """从日收益率序列计算组合指标。

    委托给 scripts.common.metrics.calc_metrics_from_daily。
    """
    return calc_metrics_from_daily(daily_returns, keys_only=True)


def compute_benchmark_return(bm_df: pd.DataFrame, start_date, end_date) -> float:
    """计算基准在日期范围内的总收益。"""
    mask = (bm_df["date"] >= start_date) & (bm_df["date"] <= end_date)
    sub = bm_df.loc[mask, "close"]
    if len(sub) < 2:
        return np.nan
    return sub.iloc[-1] / sub.iloc[0] - 1


# ---------------------------------------------------------------------------
# 6. 分析构建函数
# ---------------------------------------------------------------------------

def build_scenarios(
    daily: pd.DataFrame,
    period_summary: pd.DataFrame,
    benchmarks_data: dict[str, pd.DataFrame],
    exclude_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构建 summary、leave-one-year-out、exclude-years、yearly contribution。"""

    all_years = sorted(daily["test_year"].unique())
    full_metrics = compute_metrics_from_daily(daily["portfolio_ret"])
    full_period_start = daily["date"].min()
    full_period_end = daily["date"].max()

    # --- Yearly contribution ---
    yearly_rows = []
    for year in all_years:
        yd = daily[daily["test_year"] == year]["portfolio_ret"]
        ym = compute_metrics_from_daily(yd)
        yearly_rows.append({
            "year": int(year),
            "yearly_return": ym["total_return"],
            "contribution_to_full_return": ym["total_return"] / (1 + full_metrics["total_return"]) if (1 + full_metrics["total_return"]) != 0 else np.nan,
            "is_positive": ym["total_return"] > 0,
            "is_dominant_year": abs(ym["total_return"]) > 0.7 * abs(full_metrics["total_return"]) if full_metrics["total_return"] != 0 else False,
            "note": "",
        })
    yearly_df = pd.DataFrame(yearly_rows)
    for i, row in yearly_df.iterrows():
        if row["is_dominant_year"]:
            yearly_df.at[i, "note"] = "单年贡献超过 full return 的 70%"

    # --- Summary: full + exclude years ---
    summary_rows = []

    # Full period
    n_pos = int((yearly_df["yearly_return"] > 0).sum())
    n_neg = int((yearly_df["yearly_return"] <= 0).sum())
    bm_beat = 0
    for bm, bm_df in benchmarks_data.items():
        bm_ret = compute_benchmark_return(bm_df, full_period_start, full_period_end)
        if not np.isnan(bm_ret) and full_metrics["total_return"] > bm_ret:
            bm_beat += 1

    summary_rows.append({
        "scenario": "full_period",
        "included_years": ",".join(str(y) for y in all_years),
        "excluded_years": "",
        "total_return": full_metrics["total_return"],
        "annual_return": full_metrics["annual_return"],
        "annual_volatility": full_metrics["annual_volatility"],
        "max_drawdown": full_metrics["max_drawdown"],
        "sharpe": full_metrics["sharpe"],
        "calmar": full_metrics["calmar"],
        "positive_year_count": n_pos,
        "negative_year_count": n_neg,
        "benchmark_beat_count": bm_beat,
        "robustness_label": "",
    })

    # Exclude years
    for exc in exclude_years:
        inc_years = [y for y in all_years if y != exc]
        sub = daily[daily["test_year"].isin(inc_years)]["portfolio_ret"]
        m = compute_metrics_from_daily(sub)
        sub_start = daily[daily["test_year"].isin(inc_years)]["date"].min()
        sub_end = daily[daily["test_year"].isin(inc_years)]["date"].max()
        bm_beat_exc = 0
        for bm, bm_df in benchmarks_data.items():
            bm_ret = compute_benchmark_return(bm_df, sub_start, sub_end)
            if not np.isnan(bm_ret) and m["total_return"] > bm_ret:
                bm_beat_exc += 1

        yearly_sub = yearly_df[yearly_df["year"].isin(inc_years)]
        summary_rows.append({
            "scenario": f"exclude_{exc}",
            "included_years": ",".join(str(y) for y in inc_years),
            "excluded_years": str(exc),
            "total_return": m["total_return"],
            "annual_return": m["annual_return"],
            "annual_volatility": m["annual_volatility"],
            "max_drawdown": m["max_drawdown"],
            "sharpe": m["sharpe"],
            "calmar": m["calmar"],
            "positive_year_count": int((yearly_sub["yearly_return"] > 0).sum()),
            "negative_year_count": int((yearly_sub["yearly_return"] <= 0).sum()),
            "benchmark_beat_count": bm_beat_exc,
            "robustness_label": "",
        })

    # Label
    for row in summary_rows:
        if row["total_return"] > 0 and row["sharpe"] > 0.3:
            row["robustness_label"] = "robust"
        elif row["total_return"] > 0:
            row["robustness_label"] = "marginal"
        else:
            row["robustness_label"] = "weak"

    summary_df = pd.DataFrame(summary_rows)

    # --- Leave-one-year-out ---
    loyo_rows = []
    for exc_year in all_years:
        inc_years = [y for y in all_years if y != exc_year]
        sub = daily[daily["test_year"].isin(inc_years)]["portfolio_ret"]
        m = compute_metrics_from_daily(sub)
        loyo_rows.append({
            "excluded_year": int(exc_year),
            "total_return": m["total_return"],
            "annual_return": m["annual_return"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "return_delta_vs_full": m["total_return"] - full_metrics["total_return"],
            "sharpe_delta_vs_full": m["sharpe"] - full_metrics["sharpe"] if not np.isnan(m["sharpe"]) and not np.isnan(full_metrics["sharpe"]) else np.nan,
            "pass_min_return": m["total_return"] > 0,
            "pass_min_sharpe": m["sharpe"] > 0.3 if not np.isnan(m["sharpe"]) else False,
            "robustness_flag": "",
        })
    for row in loyo_rows:
        if row["pass_min_return"] and row["pass_min_sharpe"]:
            row["robustness_flag"] = "pass"
        elif row["pass_min_return"]:
            row["robustness_flag"] = "marginal"
        else:
            row["robustness_flag"] = "fail"
    loyo_df = pd.DataFrame(loyo_rows)

    # --- Exclude years detail ---
    exclude_rows = []
    for exc in exclude_years:
        inc_years = [y for y in all_years if y != exc]
        sub = daily[daily["test_year"].isin(inc_years)]["portfolio_ret"]
        m = compute_metrics_from_daily(sub)
        conclusion = "robust" if m["total_return"] > 0 and m["sharpe"] > 0.3 else ("marginal" if m["total_return"] > 0 else "fail")
        exclude_rows.append({
            "excluded_years": str(exc),
            "total_return": m["total_return"],
            "annual_return": m["annual_return"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "conclusion": conclusion,
        })
    exclude_df = pd.DataFrame(exclude_rows)

    return summary_df, loyo_df, exclude_df, yearly_df


def build_benchmark_comparison(
    daily: pd.DataFrame,
    benchmarks_data: dict[str, pd.DataFrame],
    all_years: list[int],
    exclude_years: list[int],
) -> pd.DataFrame:
    """基准超额收益：full、exclude-years、leave-one-year-out。"""
    rows = []
    scenarios = [("full_period", all_years, "")]
    for exc in exclude_years:
        inc = [y for y in all_years if y != exc]
        scenarios.append((f"exclude_{exc}", inc, str(exc)))
    for exc_year in all_years:
        inc = [y for y in all_years if y != exc_year]
        scenarios.append((f"loyo_exclude_{exc_year}", inc, str(exc_year)))

    for scenario_name, inc_years, exc_label in scenarios:
        sub = daily[daily["test_year"].isin(inc_years)]
        strat_ret = compute_metrics_from_daily(sub["portfolio_ret"])["total_return"]
        sub_start = sub["date"].min()
        sub_end = sub["date"].max()

        for bm, bm_df in benchmarks_data.items():
            bm_ret = compute_benchmark_return(bm_df, sub_start, sub_end)
            bm_label = BENCHMARK_MAP[bm][2] if bm in BENCHMARK_MAP else bm
            rows.append({
                "scenario": scenario_name,
                "benchmark": f"{bm} ({bm_label})",
                "strategy_total_return": strat_ret,
                "benchmark_total_return": bm_ret,
                "excess_return": strat_ret - bm_ret if not np.isnan(bm_ret) else np.nan,
                "beat_benchmark": strat_ret > bm_ret if not np.isnan(bm_ret) else np.nan,
            })
    return pd.DataFrame(rows)


def build_variant_stability(detail: pd.DataFrame) -> pd.DataFrame:
    """Alpha variant 稳定性分析。"""
    if detail.empty or "alpha_variant" not in detail.columns:
        return pd.DataFrame()

    detail = safe_to_numeric(detail, ["test_total_return", "test_sharpe"])
    agg = detail.groupby("alpha_variant").agg(
        selected_count=("symbol", "count"),
        selected_year_count=("test_year", "nunique"),
        avg_test_return=("test_total_return", "mean"),
        median_test_return=("test_total_return", "median"),
        worst_year_return=("test_total_return", "min"),
        best_year_return=("test_total_return", "max"),
    ).reset_index()

    agg["win_count"] = detail.groupby("alpha_variant")["test_total_return"].apply(
        lambda x: (x > 0).sum()
    ).values
    agg["win_rate"] = agg["win_count"] / agg["selected_count"]

    def label(row):
        if (row["selected_year_count"] >= 2
                and row["avg_test_return"] > 0
                and row["win_rate"] >= 0.5):
            return "stable"
        return "unstable"

    agg["stability_label"] = agg.apply(label, axis=1)
    return agg.sort_values("avg_test_return", ascending=False)


def build_parameter_stability(detail: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    """参数组合稳定性分析。param_cols 由 config 注入。"""
    if detail.empty:
        return pd.DataFrame()

    detail = safe_to_numeric(detail, ["test_total_return"])
    effective_cols = [c for c in param_cols if c in detail.columns]
    if not effective_cols:
        return pd.DataFrame()

    agg = detail.groupby(effective_cols).agg(
        selected_count=("symbol", "count"),
        selected_year_count=("test_year", "nunique"),
        avg_test_return=("test_total_return", "mean"),
    ).reset_index()

    agg["win_count"] = detail.groupby(effective_cols)["test_total_return"].apply(
        lambda x: (x > 0).sum()
    ).values
    agg["win_rate"] = agg["win_count"] / agg["selected_count"]

    def label(row):
        if (row["selected_year_count"] >= 2
                and row["avg_test_return"] > 0
                and row["win_rate"] >= 0.5):
            return "stable"
        return "unstable"

    agg["stability_label"] = agg.apply(label, axis=1)
    return agg.sort_values("avg_test_return", ascending=False)


def build_concentration(detail: pd.DataFrame) -> pd.DataFrame:
    """股票贡献集中度分析。"""
    if detail.empty or "symbol" not in detail.columns:
        return pd.DataFrame()

    detail = safe_to_numeric(detail, ["test_total_return"])
    agg = detail.groupby("symbol").agg(
        selected_count=("symbol", "count"),
        selected_year_count=("test_year", "nunique"),
        sum_test_return=("test_total_return", "sum"),
        avg_test_return=("test_total_return", "mean"),
    ).reset_index()

    agg["win_count"] = detail.groupby("symbol")["test_total_return"].apply(
        lambda x: (x > 0).sum()
    ).values
    agg["win_rate"] = agg["win_count"] / agg["selected_count"]
    agg = agg.sort_values("sum_test_return", ascending=False).reset_index(drop=True)
    agg["contribution_rank"] = range(1, len(agg) + 1)

    total_sum = agg["sum_test_return"].sum()
    agg["concentration_note"] = ""
    top_share = agg.iloc[0]["sum_test_return"] / total_sum if total_sum != 0 and len(agg) > 0 else 0
    if top_share > 0.3:
        agg.loc[agg.index[0], "concentration_note"] = f"Top1 贡献占比 {top_share:.0%}"

    return agg


def build_train_test_stability(detail: pd.DataFrame) -> pd.DataFrame:
    """Train/test 相关性分析。"""
    if detail.empty:
        return pd.DataFrame()

    detail = safe_to_numeric(detail, ["train_score", "train_annual_return",
                                       "train_sharpe", "test_total_return"])
    pairs = [
        ("train_score", "test_total_return"),
        ("train_annual_return", "test_total_return"),
        ("train_sharpe", "test_total_return"),
    ]
    rows = []
    for train_col, test_col in pairs:
        if train_col in detail.columns and test_col in detail.columns:
            valid = detail[[train_col, test_col]].dropna()
            if len(valid) > 5:
                corr = valid[train_col].corr(valid[test_col])
                if corr > 0.5:
                    interp = "强正相关，train→test 传递良好"
                elif corr > 0.2:
                    interp = "弱正相关，有一定传递性"
                elif corr > 0:
                    interp = "极弱正相关，传递性差"
                else:
                    interp = "无正相关或负相关，train 指标无法预测 test"
                rows.append({
                    "metric_pair": f"{train_col} → {test_col}",
                    "correlation": corr,
                    "n": len(valid),
                    "interpretation": interp,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Gate 评估
# ---------------------------------------------------------------------------

def evaluate_gates(
    summary_df: pd.DataFrame,
    loyo_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    train_test_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    exclude_years: list[int],
) -> tuple[list[dict], str]:
    """评估所有硬性 gate，返回 (gate_rows, final_decision)。

    exclude_years 动态传入，不再硬编码。
    """

    gates = []

    # Gate 1: exclude year positive return (for each exclude year)
    for exc in exclude_years:
        scenario = f"exclude_{exc}"
        ex_row = summary_df[summary_df["scenario"] == scenario]
        gate_name = f"{scenario}_positive_return"
        if not ex_row.empty:
            val = ex_row.iloc[0]["total_return"]
            passed = val > 0
            gates.append({
                "gate_name": gate_name,
                "pass": passed,
                "value": val,
                "threshold": 0,
                "reason": f"剔除{exc}后 total_return={val:.4f}" + (" > 0" if passed else " <= 0，策略过度依赖该年"),
            })
        else:
            gates.append({
                "gate_name": gate_name,
                "pass": False, "value": np.nan, "threshold": 0,
                "reason": f"缺少 {scenario} 数据",
            })

    # Gate 2: exclude year sharpe > 0.3
    for exc in exclude_years:
        scenario = f"exclude_{exc}"
        ex_row = summary_df[summary_df["scenario"] == scenario]
        gate_name = f"{scenario}_sharpe_above_0.3"
        if not ex_row.empty:
            val = ex_row.iloc[0]["sharpe"]
            passed = val > 0.3 if not np.isnan(val) else False
            gates.append({
                "gate_name": gate_name,
                "pass": passed,
                "value": val,
                "threshold": 0.3,
                "reason": f"剔除{exc}后 Sharpe={format_float(val)}" + (" > 0.3" if passed else " <= 0.3"),
            })
        else:
            gates.append({
                "gate_name": gate_name,
                "pass": False, "value": np.nan, "threshold": 0.3,
                "reason": f"缺少 {scenario} 数据",
            })

    # Gate 3: leave-one-year-out <= 40% fail
    if not loyo_df.empty:
        n_fail = int((loyo_df["total_return"] <= 0).sum())
        total = len(loyo_df)
        fail_rate = n_fail / total
        passed = fail_rate <= 0.4
        gates.append({
            "gate_name": "loyo_fail_rate_below_40pct",
            "pass": passed,
            "value": fail_rate,
            "threshold": 0.4,
            "reason": f"leave-one-year-out: {n_fail}/{total} ({fail_rate:.0%}) 场景 total_return <= 0" + ("，通过" if passed else "，不通过"),
        })
    else:
        gates.append({
            "gate_name": "loyo_fail_rate_below_40pct",
            "pass": False, "value": np.nan, "threshold": 0.4,
            "reason": "缺少 leave-one-year-out 数据",
        })

    # Gate 4: stable alpha variant exists
    if not variant_df.empty:
        stable_count = int((variant_df["stability_label"] == "stable").sum())
        passed = stable_count > 0
        gates.append({
            "gate_name": "stable_alpha_variant_exists",
            "pass": passed,
            "value": stable_count,
            "threshold": 1,
            "reason": f"稳定 variant 数量={stable_count}" + ("，存在稳定 variant" if passed else "，无稳定 variant"),
        })
    else:
        gates.append({
            "gate_name": "stable_alpha_variant_exists",
            "pass": False, "value": 0, "threshold": 1,
            "reason": "variant 数据为空",
        })

    # Gate 5: train_score → test_total_return correlation >= 0.2
    if not train_test_df.empty:
        row = train_test_df[train_test_df["metric_pair"] == "train_score → test_total_return"]
        if not row.empty:
            val = row.iloc[0]["correlation"]
            passed = val >= 0.2
            gates.append({
                "gate_name": "train_test_correlation_above_0.2",
                "pass": passed,
                "value": val,
                "threshold": 0.2,
                "reason": f"train_score→test_return 相关性={val:.4f}" + (" >= 0.2" if passed else " < 0.2，train 指标无法预测 test"),
            })
        else:
            gates.append({
                "gate_name": "train_test_correlation_above_0.2",
                "pass": False, "value": np.nan, "threshold": 0.2,
                "reason": "缺少 train_score → test_total_return 相关性数据",
            })
    else:
        gates.append({
            "gate_name": "train_test_correlation_above_0.2",
            "pass": False, "value": np.nan, "threshold": 0.2,
            "reason": "train_test 数据为空",
        })

    # Gate 6: bad contributors not > 2x good contributors
    if not concentration_df.empty:
        good = concentration_df[
            (concentration_df["sum_test_return"] > 0)
            & (concentration_df["win_rate"] >= 0.5)
        ]
        bad = concentration_df[
            (concentration_df["sum_test_return"] < 0)
            | (
                (concentration_df["win_rate"] < 0.5)
                & (concentration_df["selected_count"] >= 2)
            )
        ]
        good_count = len(good)
        bad_count = len(bad)
        passed = bad_count <= 2 * good_count if good_count > 0 else bad_count == 0
        gates.append({
            "gate_name": "bad_contributors_not_dominant",
            "pass": passed,
            "value": bad_count / good_count if good_count > 0 else 0,
            "threshold": 2.0,
            "reason": f"bad_contributors={bad_count}, good_contributors={good_count}" + ("，通过" if passed else "，bad > 2x good"),
        })
    else:
        gates.append({
            "gate_name": "bad_contributors_not_dominant",
            "pass": False, "value": np.nan, "threshold": 2.0,
            "reason": "concentration 数据为空",
        })

    # Gate 7: no single year > 70% of full return
    if not yearly_df.empty and "yearly_return" in yearly_df.columns:
        full_ret = summary_df[summary_df["scenario"] == "full_period"].iloc[0]["total_return"] if not summary_df.empty else 0
        dominated = yearly_df[yearly_df["is_dominant_year"] == True] if "is_dominant_year" in yearly_df.columns else pd.DataFrame()
        passed = len(dominated) == 0
        max_year = yearly_df.loc[yearly_df["yearly_return"].abs().idxmax()] if len(yearly_df) > 0 else None
        max_val = max_year["yearly_return"] if max_year is not None else 0
        gates.append({
            "gate_name": "no_dominant_single_year",
            "pass": passed,
            "value": abs(max_val / full_ret) if full_ret != 0 else 0,
            "threshold": 0.7,
            "reason": f"最大单年贡献={format_pct(max_val)}，full return={format_pct(full_ret)}" + ("，无单年主导" if passed else "，存在单年主导"),
        })
    else:
        gates.append({
            "gate_name": "no_dominant_single_year",
            "pass": False, "value": np.nan, "threshold": 0.7,
            "reason": "yearly 数据为空",
        })

    # Final decision
    all_pass = all(g["pass"] for g in gates)
    any_critical_fail = False
    critical_gates = {f"exclude_{exc}_positive_return" for exc in exclude_years}
    critical_gates |= {f"exclude_{exc}_sharpe_above_0.3" for exc in exclude_years}
    critical_gates |= {"stable_alpha_variant_exists", "train_test_correlation_above_0.2"}
    for g in gates:
        if g["gate_name"] in critical_gates and not g["pass"]:
            any_critical_fail = True

    if all_pass:
        decision = "promote_to_portfolio_backtest"
    elif any_critical_fail:
        decision = "revise_alpha_signal"
    else:
        decision = "continue_to_robustness_validation"

    return gates, decision


# ---------------------------------------------------------------------------
# 8. 输出：报告
# ---------------------------------------------------------------------------

def write_report(
    output_dir: Path,
    cfg: WFRobustnessConfig,
    args,
    summary_df: pd.DataFrame,
    loyo_df: pd.DataFrame,
    exclude_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    param_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    train_test_df: pd.DataFrame,
    gates: list[dict],
    decision: str,
) -> Path:
    """生成 TXT 报告。"""
    path = output_dir / cfg.report_filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{cfg.display_name} Robustness Validation Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Input tag: {args.input_tag}\n")
        f.write(f"Run ID: {args.run_id}\n")
        f.write(f"Exclude years: {args.exclude_year}\n")
        f.write(f"Benchmarks: {args.benchmarks}\n\n")

        # 1. Overall summary
        f.write("1. Overall Robustness Summary\n")
        f.write("-" * 40 + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"\n  Scenario: {row['scenario']}\n")
            f.write(f"    Included years: {row['included_years']}\n")
            f.write(f"    Total return: {format_pct(row['total_return'])}\n")
            f.write(f"    Annual return: {format_pct(row['annual_return'])}\n")
            f.write(f"    Annual volatility: {format_pct(row['annual_volatility'])}\n")
            f.write(f"    Max drawdown: {format_pct(row['max_drawdown'])}\n")
            f.write(f"    Sharpe: {format_float(row['sharpe'])}\n")
            f.write(f"    Calmar: {format_float(row['calmar'])}\n")
            f.write(f"    Positive years: {int(row['positive_year_count'])}, Negative years: {int(row['negative_year_count'])}\n")
            f.write(f"    Benchmark beat count: {int(row['benchmark_beat_count'])}\n")
            f.write(f"    Label: {row['robustness_label']}\n")

        # 2. Year dependency
        f.write("\n\n2. Year Dependency Check\n")
        f.write("-" * 40 + "\n")
        for exc in [int(y) for y in args.exclude_year.split(",")]:
            ex_row = summary_df[summary_df["scenario"] == f"exclude_{exc}"]
            full = summary_df[summary_df["scenario"] == "full_period"]
            if not ex_row.empty and not full.empty:
                ret_drop = ex_row.iloc[0]["total_return"] - full.iloc[0]["total_return"]
                f.write(f"  Full period total return: {format_pct(full.iloc[0]['total_return'])}\n")
                f.write(f"  Exclude {exc} total return: {format_pct(ex_row.iloc[0]['total_return'])}\n")
                f.write(f"  Return drop: {format_pct(ret_drop)}\n")
                if ex_row.iloc[0]["total_return"] <= 0:
                    f.write(f"  ** FAIL: 剔除{exc}后收益为负，策略过度依赖{exc}年行情。**\n")
                elif ex_row.iloc[0]["sharpe"] <= 0.3:
                    f.write(f"  ** FAIL: 剔除{exc}后 Sharpe <= 0.3，风险调整后收益不足。**\n")
                else:
                    f.write(f"  PASS: 剔除{exc}后策略仍然有效。\n")

        # 3. Leave-one-year-out
        f.write("\n\n3. Leave-One-Year-Out Check\n")
        f.write("-" * 40 + "\n")
        for _, row in loyo_df.iterrows():
            f.write(f"  Exclude {int(row['excluded_year'])}: total_return={format_pct(row['total_return'])}, "
                    f"sharpe={format_float(row['sharpe'])}, flag={row['robustness_flag']}\n")
        n_fail = int((loyo_df["total_return"] <= 0).sum())
        f.write(f"\n  Fail count: {n_fail}/{len(loyo_df)}\n")
        if n_fail / len(loyo_df) > 0.4:
            f.write("  ** FAIL: 超过40%的 leave-one-year-out 场景收益为负。**\n")

        # 4. Benchmark robustness
        f.write("\n\n4. Benchmark Robustness\n")
        f.write("-" * 40 + "\n")
        for _, row in benchmark_df.iterrows():
            f.write(f"  {row['scenario']:30s} | {row['benchmark']:25s} | "
                    f"strategy={format_pct(row['strategy_total_return'])} | "
                    f"benchmark={format_pct(row['benchmark_total_return'])} | "
                    f"excess={format_pct(row['excess_return'])} | "
                    f"beat={row['beat_benchmark']}\n")

        # 5. Alpha variant stability
        f.write("\n\n5. Alpha Variant Stability\n")
        f.write("-" * 40 + "\n")
        for _, row in variant_df.iterrows():
            f.write(f"  {row['alpha_variant']:45s} | count={int(row['selected_count']):3d} | "
                    f"years={int(row['selected_year_count'])} | "
                    f"avg_ret={format_pct(row['avg_test_return'])} | "
                    f"win_rate={row['win_rate']:.2f} | "
                    f"{row['stability_label']}\n")
        stable = variant_df[variant_df["stability_label"] == "stable"]
        if len(stable) == 0:
            f.write("  ** FAIL: 无稳定 alpha variant。**\n")

        # 6. Parameter stability
        f.write("\n\n6. Parameter Stability\n")
        f.write("-" * 40 + "\n")
        stable_params = param_df[param_df["stability_label"] == "stable"] if not param_df.empty else pd.DataFrame()
        f.write(f"  Total parameter combinations: {len(param_df)}\n")
        f.write(f"  Stable parameter combinations: {len(stable_params)}\n")
        if not param_df.empty:
            for _, row in param_df.head(10).iterrows():
                parts = []
                for col in cfg.param_cols:
                    parts.append(f"{col}={row.get(col, '?')}")
                f.write(f"    {' '.join(parts):50s} | "
                        f"count={int(row['selected_count']):3d} years={int(row['selected_year_count'])} "
                        f"avg={format_pct(row['avg_test_return'])} wr={row['win_rate']:.2f} "
                        f"{row['stability_label']}\n")

        # 7. Stock concentration
        f.write("\n\n7. Stock Concentration\n")
        f.write("-" * 40 + "\n")
        if not concentration_df.empty:
            good_count = int((concentration_df["sum_test_return"] > 0).sum())
            bad_count = int((concentration_df["sum_test_return"] <= 0).sum())
            f.write(f"  Total unique stocks: {len(concentration_df)}\n")
            f.write(f"  Good contributors (sum_return > 0): {good_count}\n")
            f.write(f"  Bad contributors (sum_return <= 0): {bad_count}\n")
            f.write("\n  Top 10 contributors:\n")
            for _, row in concentration_df.head(10).iterrows():
                f.write(f"    {row['symbol']:12s} | selected={int(row['selected_count'])} | "
                        f"years={int(row['selected_year_count'])} | "
                        f"sum_ret={format_pct(row['sum_test_return'])} | "
                        f"avg_ret={format_pct(row['avg_test_return'])} | "
                        f"wr={row['win_rate']:.2f} | {row['concentration_note']}\n")
            if bad_count > 2 * good_count:
                f.write("  ** FAIL: bad contributors 数量超过 good 的 2 倍。**\n")

        # 8. Train-test stability
        f.write("\n\n8. Train-Test Stability\n")
        f.write("-" * 40 + "\n")
        for _, row in train_test_df.iterrows():
            f.write(f"  {row['metric_pair']:45s} | r={row['correlation']:.4f} (n={int(row['n'])}) | {row['interpretation']}\n")

        # 9. Final decision
        f.write("\n\n9. Final Decision\n")
        f.write("-" * 40 + "\n")
        f.write(f"\n  Decision: {decision}\n\n")
        if decision == "promote_to_portfolio_backtest":
            f.write("  所有 gate 均通过，策略稳健性验证完成，可以进入 portfolio backtest。\n")
        elif decision == "revise_alpha_signal":
            f.write("  存在 critical gate 未通过，策略信号需要修正。\n")
            f.write("  不建议进入 portfolio backtest。\n")
        else:
            f.write("  部分 gate 未通过但非 critical，建议继续做稳健性验证或 alpha 修正。\n")
            f.write("  暂不进入 portfolio backtest。\n")

        f.write("\n  Gate details:\n")
        for g in gates:
            status = "PASS" if g["pass"] else "FAIL"
            f.write(f"    [{status}] {g['gate_name']}: {g['reason']}\n")

    return path


# ---------------------------------------------------------------------------
# 9. 输出：PNG 图表
# ---------------------------------------------------------------------------

def make_equity_chart(daily: pd.DataFrame, exclude_years: list[int], output_dir: Path, display_name: str) -> Path:
    """权益曲线：full vs exclude years。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    full_eq = (1 + daily["portfolio_ret"]).cumprod()
    ax.plot(daily["date"], full_eq, label="Full period", linewidth=1.5)

    for exc in exclude_years:
        sub = daily[daily["test_year"] != exc]
        sub_eq = (1 + sub["portfolio_ret"]).cumprod()
        ax.plot(sub["date"], sub_eq, label=f"Exclude {exc}", linewidth=1, alpha=0.7)

    ax.set_title(f"{display_name}: Equity Curve (Full vs Exclude Years)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "equity_full_vs_exclude.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_loyo_chart(loyo_df: pd.DataFrame, output_dir: Path, display_name: str) -> Path:
    """Leave-one-year-out 收益柱状图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["green" if r > 0 else "red" for r in loyo_df["total_return"]]
    ax.bar(loyo_df["excluded_year"].astype(str), loyo_df["total_return"], color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{display_name}: Leave-One-Year-Out Returns")
    ax.set_xlabel("Excluded Year")
    ax.set_ylabel("Total Return")
    plt.tight_layout()
    path = output_dir / "leave_one_year_out_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_yearly_chart(yearly_df: pd.DataFrame, output_dir: Path, display_name: str) -> Path:
    """年度贡献柱状图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["green" if r > 0 else "red" for r in yearly_df["yearly_return"]]
    ax.bar(yearly_df["year"].astype(str), yearly_df["yearly_return"], color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{display_name}: Yearly Contribution")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yearly Return")
    plt.tight_layout()
    path = output_dir / "yearly_contribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_variant_chart(variant_df: pd.DataFrame, output_dir: Path, display_name: str) -> Path:
    """Variant 稳定性图表。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if l == "stable" else "orange" for l in variant_df["stability_label"]]
    ax.barh(variant_df["alpha_variant"], variant_df["avg_test_return"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(f"{display_name}: Variant Stability (Avg Test Return)")
    ax.set_xlabel("Avg Test Return")
    plt.tight_layout()
    path = output_dir / "variant_stability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_concentration_chart(concentration_df: pd.DataFrame, output_dir: Path, display_name: str) -> Path:
    """贡献者集中度图表（top 20）。"""
    top = concentration_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["green" if r > 0 else "red" for r in top["sum_test_return"]]
    ax.barh(top["symbol"][::-1], top["sum_test_return"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(f"{display_name}: Top 20 Stock Contributors")
    ax.set_xlabel("Sum Test Return")
    plt.tight_layout()
    path = output_dir / "contributor_concentration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
