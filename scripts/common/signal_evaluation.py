# -*- coding: utf-8 -*-
"""
signal_evaluation.py

SignalEvaluationRecord：信号截面质量评估模块。

借鉴 Qlib qlib/contrib/eva/alpha.py 和 SigAnaRecord 的设计，
在进入 batch/walk-forward 之前先评估信号本身的截面排序能力。

核心指标：
  - IC：Pearson 信息系数（每日截面相关）
  - RankIC：Spearman 秩信息系数
  - ICIR：IC 均值 / IC 标准差
  - IC t-stat：IC 均值的 t 检验统计量
  - IC win rate：IC > 0 的日期占比
  - Quantile long-short return：按信号分位数做多空收益
  - Long precision：top quantile 正收益比例
  - Coverage：每个交易日可用信号占比
  - Signal autocorr：信号自相关（换手压力估计）

使用方式：
  from scripts.common.signal_evaluation import (
      compute_ic_series,
      compute_rank_ic_series,
      compute_ic_summary,
      compute_quantile_long_short_return,
      compute_coverage,
      compute_signal_autocorr,
      evaluate_signal,
      save_signal_evaluation,
  )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IC / RankIC 时间序列
# ---------------------------------------------------------------------------

def compute_ic_series(
    alpha: pd.DataFrame,
    label_col: str,
    score_col: str = "alpha_score",
) -> pd.Series:
    """计算每日截面 Pearson IC。

    Parameters
    ----------
    alpha : pd.DataFrame
        MultiIndex (date, symbol) 或含 date 列的 DataFrame。
        必须包含 score_col 和 label_col。
    label_col : str
        前瞻收益列名，如 "ret_1d"。
    score_col : str
        Alpha 评分列名，默认 "alpha_score"。

    Returns
    -------
    pd.Series
        以日期为索引的每日 IC 值。
    """
    df = _ensure_date_index(alpha)
    valid = df[[score_col, label_col]].dropna()

    if valid.empty:
        return pd.Series(dtype=float)

    ic = valid.groupby(level=0).apply(
        lambda g: g[score_col].corr(g[label_col])
    )
    ic.name = "ic"
    return ic


def compute_rank_ic_series(
    alpha: pd.DataFrame,
    label_col: str,
    score_col: str = "alpha_score",
) -> pd.Series:
    """计算每日截面 Spearman RankIC（不依赖 scipy）。"""
    df = _ensure_date_index(alpha)
    valid = df[[score_col, label_col]].dropna()

    if valid.empty:
        return pd.Series(dtype=float)

    def _spearman_corr(g: pd.DataFrame) -> float:
        """Spearman = Pearson(rank(x), rank(y))。"""
        rx = g[score_col].rank()
        ry = g[label_col].rank()
        return rx.corr(ry)

    rank_ic = valid.groupby(level=0).apply(_spearman_corr)
    rank_ic.name = "rank_ic"
    return rank_ic


# ---------------------------------------------------------------------------
# IC 摘要统计
# ---------------------------------------------------------------------------

def compute_ic_summary(
    ic: pd.Series,
    rank_ic: pd.Series,
) -> pd.DataFrame:
    """将 IC / RankIC 时间序列汇总为单行摘要表。

    Returns
    -------
    pd.DataFrame
        包含列：ic_mean, ic_std, icir, ic_tstat, ic_positive_rate,
        rank_ic_mean, rank_ic_std, rank_icir, rank_ic_tstat, rank_ic_positive_rate,
        ic_days。
    """
    ic_stats = _series_stats(ic, "ic")
    rank_stats = _series_stats(rank_ic, "rank_ic")

    summary = {**ic_stats, **rank_stats}
    summary["ic_days"] = max(len(ic), len(rank_ic))

    return pd.DataFrame([summary])


def _series_stats(s: pd.Series, prefix: str) -> Dict[str, float]:
    """对一个 IC 序列计算摘要统计。"""
    if s.empty or s.isna().all():
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}ir": np.nan,
            f"{prefix}_tstat": np.nan,
            f"{prefix}_positive_rate": np.nan,
        }

    mean = s.mean()
    std = s.std()
    n = s.count()

    ir = mean / std if std > 0 else np.nan
    tstat = mean / std * np.sqrt(n) if std > 0 else np.nan
    positive_rate = (s > 0).mean()

    return {
        f"{prefix}_mean": round(float(mean), 6),
        f"{prefix}_std": round(float(std), 6),
        f"{prefix}ir": round(float(ir), 4) if not np.isnan(ir) else np.nan,
        f"{prefix}_tstat": round(float(tstat), 4) if not np.isnan(tstat) else np.nan,
        f"{prefix}_positive_rate": round(float(positive_rate), 4),
    }


# ---------------------------------------------------------------------------
# 分位数多空收益
# ---------------------------------------------------------------------------

def compute_quantile_long_short_return(
    alpha: pd.DataFrame,
    label_col: str,
    score_col: str = "alpha_score",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """按信号分位数计算每日截面平均前瞻收益。

    Parameters
    ----------
    alpha : pd.DataFrame
        MultiIndex (date, symbol) 的数据。
    label_col : str
        前瞻收益列名。
    score_col : str
        Alpha 评分列名。
    n_quantiles : int
        分位数数量，默认 5。

    Returns
    -------
    pd.DataFrame
        索引为日期，列为 Q1..Q{n} 和 long_short。
        long_short = Q_top - Q_bottom。
    """
    df = _ensure_date_index(alpha)
    valid = df[[score_col, label_col]].dropna()

    if valid.empty:
        return pd.DataFrame()

    # 为每日截面计算分位标签
    quantile_labels = valid.groupby(level=0)[score_col].transform(
        lambda s: _safe_qcut(s, n_quantiles)
    )

    # 按日期和分位组计算平均收益
    quantile_labels.name = "quantile"
    merged = valid[[label_col]].copy()
    merged["quantile"] = quantile_labels

    daily_mean = merged.groupby([merged.index.get_level_values(0), "quantile"])[label_col].mean()
    daily_mean = daily_mean.reset_index()
    daily_mean.columns = ["date", "quantile", "mean_ret"]

    # pivot: 行=日期, 列=分位编号
    result = daily_mean.pivot(index="date", columns="quantile", values="mean_ret")
    result.columns = [f"Q{int(c) + 1}" for c in result.columns]

    # long-short = 最高分位 - 最低分位
    q_cols = sorted([c for c in result.columns if c.startswith("Q")])
    if len(q_cols) >= 2:
        result["long_short"] = result[q_cols[-1]] - result[q_cols[0]]

    return result


def _safe_qcut(s: pd.Series, n_quantiles: int) -> pd.Series:
    """安全的 qcut，样本不足时退化。"""
    unique_count = s.nunique()
    if unique_count < n_quantiles:
        # 样本不足，退化为中位数分割
        try:
            return pd.qcut(s, min(unique_count, 2), labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(0, index=s.index)
    return pd.qcut(s, n_quantiles, labels=False, duplicates="drop")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def compute_coverage(
    alpha: pd.DataFrame,
    score_col: str = "alpha_score",
) -> pd.Series:
    """计算每日信号覆盖率（有效信号数 / 当日总股票数）。"""
    df = _ensure_date_index(alpha)

    if score_col not in df.columns:
        return pd.Series(dtype=float)

    total = df.groupby(level=0).size()
    valid = df.groupby(level=0)[score_col].apply(lambda s: s.notna().sum())
    coverage = valid / total
    coverage.name = "coverage"
    return coverage


# ---------------------------------------------------------------------------
# 信号自相关
# ---------------------------------------------------------------------------

def compute_signal_autocorr(
    alpha: pd.DataFrame,
    score_col: str = "alpha_score",
    max_lag: int = 5,
) -> pd.DataFrame:
    """计算信号的跨日自相关（截面排名自相关），用于估计换手压力。

    对每日截面排名取均值后，计算 lag-1..max_lag 的自相关系数。

    Returns
    -------
    pd.DataFrame
        列：lag, autocorr。
    """
    df = _ensure_date_index(alpha)

    if score_col not in df.columns:
        return pd.DataFrame(columns=["lag", "autocorr"])

    # 每日截面排名均值（反映整体信号分布变化）
    daily_mean = df.groupby(level=0)[score_col].mean()
    daily_mean = daily_mean.dropna()

    if len(daily_mean) < max_lag + 10:
        return pd.DataFrame(columns=["lag", "autocorr"])

    rows = []
    for lag in range(1, max_lag + 1):
        autocorr = daily_mean.autocorr(lag=lag)
        rows.append({"lag": lag, "autocorr": round(float(autocorr), 6) if not np.isnan(autocorr) else np.nan})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 综合评估
# ---------------------------------------------------------------------------

def evaluate_signal(
    alpha: pd.DataFrame,
    label_col: str,
    score_col: str = "alpha_score",
    n_quantiles: int = 5,
    autocorr_max_lag: int = 5,
) -> Dict[str, Any]:
    """一站式信号评估：IC、RankIC、分位收益、覆盖率、自相关。

    Parameters
    ----------
    alpha : pd.DataFrame
        MultiIndex (date, symbol) 的数据，含 score_col 和 label_col。
    label_col : str
        前瞻收益列名。
    score_col : str
        Alpha 评分列名。
    n_quantiles : int
        分位数数量。
    autocorr_max_lag : int
        自相关最大滞后阶数。

    Returns
    -------
    dict
        key: ic_daily, ic_summary, quantile_returns, coverage, autocorr。
    """
    ic = compute_ic_series(alpha, label_col, score_col)
    rank_ic = compute_rank_ic_series(alpha, label_col, score_col)
    summary = compute_ic_summary(ic, rank_ic)
    quantile = compute_quantile_long_short_return(alpha, label_col, score_col, n_quantiles)
    coverage = compute_coverage(alpha, score_col)
    autocorr = compute_signal_autocorr(alpha, score_col, autocorr_max_lag)

    return {
        "ic_daily": pd.DataFrame({"ic": ic, "rank_ic": rank_ic}),
        "ic_summary": summary,
        "quantile_returns": quantile,
        "coverage": coverage,
        "autocorr": autocorr,
    }


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------

def save_signal_evaluation(
    results: Dict[str, Any],
    output_dir: Path,
    label_col: str,
    run_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """将评估结果保存到目录。

    Parameters
    ----------
    results : dict
        evaluate_signal() 的返回值。
    output_dir : Path
        输出目录。
    label_col : str
        标签列名（用于文件命名和 manifest）。
    run_info : dict, optional
        运行元信息（experiment_id, alpha_variant, params 等）。

    Returns
    -------
    Path
        输出目录路径。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # IC daily
    ic_daily = results["ic_daily"]
    if not ic_daily.empty:
        ic_daily.to_csv(output_dir / "signal_ic_daily.csv", encoding="utf-8-sig")

    # IC summary
    summary = results["ic_summary"]
    if not summary.empty:
        summary.to_csv(output_dir / "signal_ic_summary.csv", index=False, encoding="utf-8-sig")

    # Quantile returns
    quantile = results["quantile_returns"]
    if not quantile.empty:
        quantile.to_csv(output_dir / "signal_quantile_return.csv", encoding="utf-8-sig")

    # Coverage
    coverage = results["coverage"]
    if not coverage.empty:
        coverage.to_frame("coverage").to_csv(output_dir / "signal_coverage.csv", encoding="utf-8-sig")

    # Autocorr
    autocorr = results["autocorr"]
    if not autocorr.empty:
        autocorr.to_csv(output_dir / "signal_autocorr.csv", index=False, encoding="utf-8-sig")

    # 文本报告
    report = _generate_text_report(results, label_col, run_info)
    (output_dir / "signal_evaluation_report.txt").write_text(report, encoding="utf-8")

    # Manifest
    manifest = _build_manifest(results, label_col, run_info, output_dir)
    (output_dir / "signal_evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("信号评估结果已保存到：%s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# 文本报告
# ---------------------------------------------------------------------------

def _generate_text_report(
    results: Dict[str, Any],
    label_col: str,
    run_info: Optional[Dict[str, Any]] = None,
) -> str:
    """生成人类可读的信号评估报告。"""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("Signal Evaluation Report")
    lines.append("=" * 80)
    lines.append("")

    if run_info:
        lines.append("Run Info:")
        for k, v in run_info.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines.append(f"Label: {label_col}")
    lines.append("")

    # IC Summary
    summary = results["ic_summary"]
    if not summary.empty:
        row = summary.iloc[0]
        lines.append("-" * 40)
        lines.append("IC Summary")
        lines.append("-" * 40)
        for col in summary.columns:
            val = row[col]
            if isinstance(val, float):
                lines.append(f"  {col}: {val:.6f}")
            else:
                lines.append(f"  {col}: {val}")
        lines.append("")

        # 信号质量判断
        ic_mean = row.get("ic_mean", np.nan)
        rank_ic_mean = row.get("rank_ic_mean", np.nan)
        icir = row.get("icir", np.nan)
        ic_positive_rate = row.get("ic_positive_rate", np.nan)

        lines.append("Signal Quality Assessment:")
        if not np.isnan(ic_mean):
            if abs(ic_mean) > 0.03:
                lines.append(f"  IC mean = {ic_mean:.4f}: STRONG signal (|IC| > 0.03)")
            elif abs(ic_mean) > 0.01:
                lines.append(f"  IC mean = {ic_mean:.4f}: MODERATE signal (0.01 < |IC| < 0.03)")
            else:
                lines.append(f"  IC mean = {ic_mean:.4f}: WEAK signal (|IC| < 0.01)")

        if not np.isnan(icir):
            if abs(icir) > 0.5:
                lines.append(f"  ICIR = {icir:.4f}: CONSISTENT signal (|ICIR| > 0.5)")
            elif abs(icir) > 0.2:
                lines.append(f"  ICIR = {icir:.4f}: SOMEWHAT consistent (0.2 < |ICIR| < 0.5)")
            else:
                lines.append(f"  ICIR = {icir:.4f}: INCONSISTENT signal (|ICIR| < 0.2)")

        if not np.isnan(ic_positive_rate):
            if ic_positive_rate > 0.6:
                lines.append(f"  IC positive rate = {ic_positive_rate:.1%}: STABLE direction")
            elif ic_positive_rate > 0.4:
                lines.append(f"  IC positive rate = {ic_positive_rate:.1%}: Direction somewhat stable")
            else:
                lines.append(f"  IC positive rate = {ic_positive_rate:.1%}: Direction UNSTABLE")
        lines.append("")

    # Quantile Returns
    quantile = results["quantile_returns"]
    if not quantile.empty:
        lines.append("-" * 40)
        lines.append("Quantile Returns (daily mean)")
        lines.append("-" * 40)
        means = quantile.mean()
        for col in quantile.columns:
            val = means.get(col, np.nan)
            lines.append(f"  {col}: {val:.6f}" if not np.isnan(val) else f"  {col}: N/A")
        lines.append("")

        if "long_short" in quantile.columns:
            ls = quantile["long_short"].dropna()
            if not ls.empty:
                ls_mean = ls.mean()
                ls_std = ls.std()
                ls_ir = ls_mean / ls_std if ls_std > 0 else np.nan
                ls_positive = (ls > 0).mean()
                lines.append(f"  Long-short daily mean: {ls_mean:.6f}")
                lines.append(f"  Long-short daily std:  {ls_std:.6f}")
                lines.append(f"  Long-short IR:         {ls_ir:.4f}" if not np.isnan(ls_ir) else "  Long-short IR:         N/A")
                lines.append(f"  Long-short win rate:   {ls_positive:.1%}")
                lines.append("")

    # Coverage
    coverage = results["coverage"]
    if not coverage.empty:
        lines.append("-" * 40)
        lines.append("Coverage")
        lines.append("-" * 40)
        lines.append(f"  Mean coverage: {coverage.mean():.1%}")
        lines.append(f"  Min coverage:  {coverage.min():.1%}")
        lines.append(f"  Max coverage:  {coverage.max():.1%}")
        low_cov = (coverage < 0.5).sum()
        if low_cov > 0:
            lines.append(f"  Days with coverage < 50%: {low_cov}")
        lines.append("")

    # Autocorr
    autocorr = results["autocorr"]
    if not autocorr.empty:
        lines.append("-" * 40)
        lines.append("Signal Autocorrelation")
        lines.append("-" * 40)
        for _, row in autocorr.iterrows():
            lines.append(f"  lag-{int(row['lag'])}: {row['autocorr']:.4f}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def _build_manifest(
    results: Dict[str, Any],
    label_col: str,
    run_info: Optional[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """构建 manifest JSON。"""
    summary = results["ic_summary"]
    ic_mean = float(summary.iloc[0]["ic_mean"]) if not summary.empty and "ic_mean" in summary.columns else None
    rank_ic_mean = float(summary.iloc[0]["rank_ic_mean"]) if not summary.empty and "rank_ic_mean" in summary.columns else None

    manifest: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "label_col": label_col,
        "ic_mean": ic_mean,
        "rank_ic_mean": rank_ic_mean,
    }

    if run_info:
        manifest.update(run_info)

    # 记录输出文件
    manifest["artifacts"] = [str(p.name) for p in output_dir.iterdir() if p.is_file()]

    return manifest


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _ensure_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DataFrame 有 (date, symbol) MultiIndex。"""
    if isinstance(df.index, pd.MultiIndex):
        return df

    if "date" in df.columns:
        idx_cols = ["date"]
        if "symbol" in df.columns:
            idx_cols.append("symbol")
        return df.set_index(idx_cols)

    raise ValueError("DataFrame 必须有 (date, symbol) MultiIndex 或含 date/symbol 列。")
