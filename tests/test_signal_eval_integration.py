# -*- coding: utf-8 -*-
"""
tests/test_signal_eval_integration.py

测试信号评估与诊断报告集成（wf_report_shared.py 中的信号质量函数）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.wf_report_shared import (
    load_signal_evaluation_summaries,
    build_signal_quality_section,
    write_signal_quality_to_report,
    VALID_VARIANTS,
)
from scripts.common.signal_evaluation import (
    compute_ic_series,
    compute_rank_ic_series,
    compute_ic_summary,
    evaluate_signal,
    save_signal_evaluation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cross_section():
    """构造一个小型截面数据，含 date/symbol MultiIndex。"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "601318.SH"]
    rows = []
    for d in dates:
        for s in symbols:
            rows.append({
                "date": d,
                "symbol": s,
                "alpha_score": np.random.randn(),
                "ret_1d": np.random.randn() * 0.02,
                "ret_5d": np.random.randn() * 0.05,
                "ret_20d": np.random.randn() * 0.1,
            })
    df = pd.DataFrame(rows)
    return df.set_index(["date", "symbol"])


@pytest.fixture
def signal_eval_dir(tmp_path, sample_cross_section):
    """生成一个临时 signal_evaluation 输出目录。"""
    eval_root = tmp_path / "signal_evaluation"
    for variant in ["short_term_reversal", "low_volatility"]:
        results = evaluate_signal(
            sample_cross_section, "ret_1d", "alpha_score", n_quantiles=3,
        )
        out_dir = eval_root / variant / "ret_1d"
        save_signal_evaluation(results, out_dir, "ret_1d", run_info={"alpha_variant": variant})

        # 也保存 ret_5d
        results_5d = evaluate_signal(
            sample_cross_section, "ret_5d", "alpha_score", n_quantiles=3,
        )
        out_dir_5d = eval_root / variant / "ret_5d"
        save_signal_evaluation(results_5d, out_dir_5d, "ret_5d", run_info={"alpha_variant": variant})

    return eval_root


# ---------------------------------------------------------------------------
# load_signal_evaluation_summaries
# ---------------------------------------------------------------------------

class TestLoadSignalEvaluationSummaries:

    def test_loads_existing_variants(self, signal_eval_dir):
        df = load_signal_evaluation_summaries(signal_eval_dir, ["short_term_reversal"], "ret_1d")
        assert not df.empty
        assert "alpha_variant" in df.columns
        assert df["alpha_variant"].iloc[0] == "short_term_reversal"

    def test_skips_missing_variants(self, signal_eval_dir):
        df = load_signal_evaluation_summaries(signal_eval_dir, ["nonexistent_variant"], "ret_1d")
        assert df.empty

    def test_loads_multiple_variants(self, signal_eval_dir):
        df = load_signal_evaluation_summaries(signal_eval_dir, ["short_term_reversal", "low_volatility"], "ret_1d")
        assert len(df) == 2
        assert set(df["alpha_variant"]) == {"short_term_reversal", "low_volatility"}

    def test_empty_dir_returns_empty(self, tmp_path):
        df = load_signal_evaluation_summaries(tmp_path / "nonexistent", ["short_term_reversal"], "ret_1d")
        assert df.empty


# ---------------------------------------------------------------------------
# build_signal_quality_section
# ---------------------------------------------------------------------------

class TestBuildSignalQualitySection:

    def test_returns_combined_table(self, signal_eval_dir):
        df = build_signal_quality_section(
            signal_eval_dir,
            variants=["short_term_reversal", "low_volatility"],
            label_cols=["ret_1d", "ret_5d"],
        )
        assert not df.empty
        assert "alpha_variant" in df.columns
        assert "label_col" in df.columns
        assert "signal_quality_label" in df.columns
        # 2 variants x 2 labels = 4 rows
        assert len(df) == 4

    def test_quality_labels_are_valid(self, signal_eval_dir):
        df = build_signal_quality_section(signal_eval_dir, label_cols=["ret_1d"])
        valid_labels = {"strong", "moderate", "weak", "unknown"}
        for label in df["signal_quality_label"]:
            assert label in valid_labels

    def test_empty_when_no_data(self, tmp_path):
        df = build_signal_quality_section(tmp_path / "nonexistent")
        assert df.empty


# ---------------------------------------------------------------------------
# write_signal_quality_to_report
# ---------------------------------------------------------------------------

class TestWriteSignalQualityToReport:

    def test_writes_quality_section(self, signal_eval_dir):
        df = build_signal_quality_section(signal_eval_dir, label_cols=["ret_1d"])

        import io
        buf = io.StringIO()
        write_signal_quality_to_report(buf, df)
        text = buf.getvalue()

        assert "Signal Quality" in text
        assert "short_term_reversal" in text

    def test_writes_empty_notice(self):
        import io
        buf = io.StringIO()
        write_signal_quality_to_report(buf, pd.DataFrame())
        text = buf.getvalue()

        assert "no signal evaluation data" in text


# ---------------------------------------------------------------------------
# IC 系列计算基础测试
# ---------------------------------------------------------------------------

class TestICSeries:

    def test_compute_ic_series_returns_series(self, sample_cross_section):
        ic = compute_ic_series(sample_cross_section, "ret_1d", "alpha_score")
        assert isinstance(ic, pd.Series)
        assert len(ic) > 0

    def test_compute_rank_ic_series_returns_series(self, sample_cross_section):
        rank_ic = compute_rank_ic_series(sample_cross_section, "ret_1d", "alpha_score")
        assert isinstance(rank_ic, pd.Series)
        assert len(rank_ic) > 0

    def test_ic_summary_has_expected_columns(self, sample_cross_section):
        ic = compute_ic_series(sample_cross_section, "ret_1d", "alpha_score")
        rank_ic = compute_rank_ic_series(sample_cross_section, "ret_1d", "alpha_score")
        summary = compute_ic_summary(ic, rank_ic)
        assert "ic_mean" in summary.columns
        assert "rank_ic_mean" in summary.columns
        assert "icir" in summary.columns


# ---------------------------------------------------------------------------
# 信号评估端到端
# ---------------------------------------------------------------------------

class TestEvaluateSignalE2E:

    def test_evaluate_signal_returns_expected_keys(self, sample_cross_section):
        result = evaluate_signal(sample_cross_section, "ret_1d", "alpha_score", n_quantiles=3)
        assert "ic_daily" in result
        assert "ic_summary" in result
        assert "quantile_returns" in result
        assert "coverage" in result
        assert "autocorr" in result

    def test_save_and_load_roundtrip(self, tmp_path, sample_cross_section):
        result = evaluate_signal(sample_cross_section, "ret_1d", "alpha_score", n_quantiles=3)
        out_dir = tmp_path / "eval_output"
        save_signal_evaluation(result, out_dir, "ret_1d", run_info={"test": True})

        # 读回 IC summary
        summary_path = out_dir / "signal_ic_summary.csv"
        assert summary_path.exists()
        df = pd.read_csv(summary_path, encoding="utf-8-sig")
        assert "ic_mean" in df.columns
        assert "rank_ic_mean" in df.columns
