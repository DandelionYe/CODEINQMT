# -*- coding: utf-8 -*-
"""tests for scripts/common/signal_evaluation.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

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


def _make_synthetic_data(
    n_dates: int = 100,
    n_stocks: int = 50,
    seed: int = 42,
    signal_strength: float = 0.05,
) -> pd.DataFrame:
    """构造合成数据：alpha_score 有一定预测能力。

    alpha_score 与 ret_1d 的相关性约为 signal_strength。
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    symbols = [f"S{i:04d}" for i in range(n_stocks)]

    rows = []
    for dt in dates:
        scores = rng.randn(n_stocks)
        # ret = signal * score + noise
        noise = rng.randn(n_stocks) * 0.02
        ret_1d = signal_strength * scores + noise
        ret_5d = signal_strength * scores * 3 + rng.randn(n_stocks) * 0.05
        ret_20d = signal_strength * scores * 8 + rng.randn(n_stocks) * 0.1

        for i, sym in enumerate(symbols):
            rows.append({
                "date": dt,
                "symbol": sym,
                "alpha_score": scores[i],
                "raw_alpha_score": scores[i],
                "alpha_signal": int(scores[i] > 0),
                "ret_1d": ret_1d[i],
                "ret_5d": ret_5d[i],
                "ret_20d": ret_20d[i],
            })

    df = pd.DataFrame(rows)
    df = df.set_index(["date", "symbol"])
    return df


@pytest.fixture
def synthetic_data():
    return _make_synthetic_data()


@pytest.fixture
def weak_signal_data():
    """信号很弱的数据（纯噪声）。"""
    return _make_synthetic_data(signal_strength=0.0, seed=99)


# --- IC 计算 ---

class TestICSeries:
    def test_returns_series(self, synthetic_data):
        ic = compute_ic_series(synthetic_data, "ret_1d")
        assert isinstance(ic, pd.Series)
        assert len(ic) > 0
        assert ic.name == "ic"

    def test_ic_positive_for_strong_signal(self, synthetic_data):
        ic = compute_ic_series(synthetic_data, "ret_1d")
        assert ic.mean() > 0, "有信号时 IC 均值应为正"

    def test_ic_near_zero_for_noise(self, weak_signal_data):
        ic = compute_ic_series(weak_signal_data, "ret_1d")
        assert abs(ic.mean()) < 0.05, "纯噪声时 IC 应接近 0"

    def test_ic_with_missing_data(self, synthetic_data):
        # 注入一些 NaN
        df = synthetic_data.copy()
        mask = np.random.RandomState(0).rand(len(df)) < 0.1
        df.loc[mask, "alpha_score"] = np.nan
        ic = compute_ic_series(df, "ret_1d")
        assert len(ic) > 0

    def test_empty_data(self):
        df = pd.DataFrame(columns=["alpha_score", "ret_1d"]).set_index(
            pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), pd.Index([])], names=["date", "symbol"])
        )
        ic = compute_ic_series(df, "ret_1d")
        assert len(ic) == 0


class TestRankICSeries:
    def test_returns_series(self, synthetic_data):
        rank_ic = compute_rank_ic_series(synthetic_data, "ret_1d")
        assert isinstance(rank_ic, pd.Series)
        assert len(rank_ic) > 0
        assert rank_ic.name == "rank_ic"

    def test_rank_ic_positive_for_strong_signal(self, synthetic_data):
        rank_ic = compute_rank_ic_series(synthetic_data, "ret_1d")
        assert rank_ic.mean() > 0


# --- IC Summary ---

class TestICSummary:
    def test_summary_structure(self, synthetic_data):
        ic = compute_ic_series(synthetic_data, "ret_1d")
        rank_ic = compute_rank_ic_series(synthetic_data, "ret_1d")
        summary = compute_ic_summary(ic, rank_ic)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1

        expected_cols = [
            "ic_mean", "ic_std", "icir", "ic_tstat", "ic_positive_rate",
            "rank_ic_mean", "rank_ic_std", "rank_icir", "rank_ic_tstat", "rank_ic_positive_rate",
            "ic_days",
        ]
        for col in expected_cols:
            assert col in summary.columns, f"缺少列: {col}"

    def test_summary_values_reasonable(self, synthetic_data):
        ic = compute_ic_series(synthetic_data, "ret_1d")
        rank_ic = compute_rank_ic_series(synthetic_data, "ret_1d")
        summary = compute_ic_summary(ic, rank_ic).iloc[0]

        assert summary["ic_days"] == len(ic)
        assert 0 <= summary["ic_positive_rate"] <= 1.0
        assert summary["ic_std"] >= 0

    def test_empty_ic(self):
        ic = pd.Series(dtype=float)
        rank_ic = pd.Series(dtype=float)
        summary = compute_ic_summary(ic, rank_ic)
        assert len(summary) == 1
        row = summary.iloc[0]
        # ic_days is 0, others are NaN
        assert row["ic_days"] == 0
        for col in ["ic_mean", "ic_std", "icir", "rank_ic_mean", "rank_ic_std", "rank_icir"]:
            assert np.isnan(row[col])


# --- Quantile Long-Short ---

class TestQuantileLongShort:
    def test_returns_dataframe(self, synthetic_data):
        result = compute_quantile_long_short_return(synthetic_data, "ret_1d", n_quantiles=5)
        assert isinstance(result, pd.DataFrame)
        assert "Q1" in result.columns
        assert "Q5" in result.columns
        assert "long_short" in result.columns

    def test_long_short_sign(self, synthetic_data):
        """有信号时，long-short 应大致为正。"""
        result = compute_quantile_long_short_return(synthetic_data, "ret_1d", n_quantiles=5)
        ls = result["long_short"].dropna()
        assert ls.mean() > 0, "有信号时 Q5-Q1 应为正"

    def test_custom_quantiles(self, synthetic_data):
        result = compute_quantile_long_short_return(synthetic_data, "ret_1d", n_quantiles=10)
        assert "Q10" in result.columns

    def test_different_horizons(self, synthetic_data):
        for horizon in ["ret_1d", "ret_5d", "ret_20d"]:
            result = compute_quantile_long_short_return(synthetic_data, horizon)
            assert not result.empty


# --- Coverage ---

class TestCoverage:
    def test_returns_series(self, synthetic_data):
        cov = compute_coverage(synthetic_data)
        assert isinstance(cov, pd.Series)
        assert len(cov) > 0
        assert cov.mean() == 1.0  # 无 NaN 时全覆盖

    def test_with_missing_scores(self, synthetic_data):
        df = synthetic_data.copy()
        # 让 50% 的分数为 NaN
        mask = np.random.RandomState(0).rand(len(df)) < 0.5
        df.loc[mask, "alpha_score"] = np.nan
        cov = compute_coverage(df)
        assert 0.4 < cov.mean() < 0.6


# --- Signal Autocorr ---

class TestSignalAutocorr:
    def test_returns_dataframe(self, synthetic_data):
        result = compute_signal_autocorr(synthetic_data, max_lag=3)
        assert isinstance(result, pd.DataFrame)
        assert "lag" in result.columns
        assert "autocorr" in result.columns
        assert len(result) == 3

    def test_autocorr_values_bounded(self, synthetic_data):
        result = compute_signal_autocorr(synthetic_data, max_lag=5)
        for _, row in result.iterrows():
            assert -1.0 <= row["autocorr"] <= 1.0


# --- 综合评估 ---

class TestEvaluateSignal:
    def test_returns_all_keys(self, synthetic_data):
        result = evaluate_signal(synthetic_data, "ret_1d")
        expected_keys = {"ic_daily", "ic_summary", "quantile_returns", "coverage", "autocorr"}
        assert set(result.keys()) == expected_keys

    def test_with_different_params(self, synthetic_data):
        result = evaluate_signal(
            synthetic_data, "ret_1d",
            n_quantiles=10,
            autocorr_max_lag=3,
        )
        assert len(result["autocorr"]) == 3
        assert "Q10" in result["quantile_returns"].columns


# --- 保存 ---

class TestSaveEvaluation:
    def test_save_creates_files(self, synthetic_data):
        result = evaluate_signal(synthetic_data, "ret_1d")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            save_signal_evaluation(result, output_dir, "ret_1d", {"test": "info"})

            assert (output_dir / "signal_ic_daily.csv").exists()
            assert (output_dir / "signal_ic_summary.csv").exists()
            assert (output_dir / "signal_quantile_return.csv").exists()
            assert (output_dir / "signal_coverage.csv").exists()
            assert (output_dir / "signal_autocorr.csv").exists()
            assert (output_dir / "signal_evaluation_report.txt").exists()
            assert (output_dir / "signal_evaluation_manifest.json").exists()

    def test_report_content(self, synthetic_data):
        result = evaluate_signal(synthetic_data, "ret_1d")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            save_signal_evaluation(result, output_dir, "ret_1d")

            report = (output_dir / "signal_evaluation_report.txt").read_text(encoding="utf-8")
            assert "IC Summary" in report
            assert "Quantile Returns" in report
            assert "Coverage" in report
