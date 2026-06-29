# -*- coding: utf-8 -*-
"""
tests/test_wf_robustness_shared.py

测试 scripts/common/wf_robustness_shared.py 共享模块。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.wf_robustness_shared import (
    WFRobustnessConfig,
    make_v6_config,
    make_v7_config,
    format_pct,
    format_float,
    compute_metrics_from_daily,
    compute_benchmark_return,
    build_scenarios,
    build_benchmark_comparison,
    build_variant_stability,
    build_parameter_stability,
    build_concentration,
    build_train_test_stability,
    evaluate_gates,
    load_input_files,
)


# ---------------------------------------------------------------------------
# Config factory tests
# ---------------------------------------------------------------------------

class TestMakeV6Config:
    def test_returns_config(self):
        cfg = make_v6_config(PROJECT_ROOT)
        assert isinstance(cfg, WFRobustnessConfig)
        assert cfg.display_name == "Alpha v6"
        assert "alpha_v6" in cfg.file_prefix
        assert cfg.default_run_id == "exp006_alpha_v6_full"

    def test_param_cols_match_v6(self):
        cfg = make_v6_config(PROJECT_ROOT)
        assert "momentum_window" in cfg.param_cols
        assert "trend_ma" in cfg.param_cols
        assert "breakout_window" in cfg.param_cols


class TestMakeV7Config:
    def test_returns_config(self):
        cfg = make_v7_config(PROJECT_ROOT)
        assert isinstance(cfg, WFRobustnessConfig)
        assert cfg.display_name == "Alpha v7"
        assert "alpha_v7" in cfg.file_prefix
        assert cfg.default_run_id == "exp007_alpha_v7_full"

    def test_param_cols_match_v7(self):
        cfg = make_v7_config(PROJECT_ROOT)
        assert cfg.param_cols == ["reversal_window", "vol_window", "turnover_short",
                                   "turnover_long", "divergence_window"]


# ---------------------------------------------------------------------------
# Format utilities
# ---------------------------------------------------------------------------

class TestFormatUtils:
    def test_format_pct_normal(self):
        assert format_pct(0.1234) == "12.34%"

    def test_format_pct_nan(self):
        assert format_pct(np.nan) == "N/A"

    def test_format_pct_none(self):
        assert format_pct(None) == "N/A"

    def test_format_float_normal(self):
        assert format_float(1.23456) == "1.2346"

    def test_format_float_nan(self):
        assert format_float(np.nan) == "N/A"


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_synthetic_daily(n_years: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_years * 252)
    years = [d.year for d in dates]
    rets = rng.normal(0.0004, 0.015, len(dates))
    return pd.DataFrame({
        "date": dates,
        "portfolio_ret": rets,
        "equity": (1 + pd.Series(rets)).cumprod().values,
        "test_year": years,
    })


def _make_synthetic_detail(n_stocks: int = 10, n_years: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    variants = ["short_term_reversal", "low_volatility", "turnover_reversal", "volume_price_divergence"]
    rows = []
    for year in range(2021, 2021 + n_years):
        for i in range(n_stocks):
            rows.append({
                "symbol": f"{600000 + i:06d}.SH",
                "alpha_variant": rng.choice(variants),
                "reversal_window": rng.choice([5, 10, 20]),
                "vol_window": rng.choice([20, 60, 120]),
                "turnover_short": 10,
                "turnover_long": 60,
                "divergence_window": rng.choice([10, 20, 60]),
                "benchmark": "000300.SH",
                "benchmark_ma": 120,
                "selected_rank": i + 1,
                "train_score": rng.uniform(0.1, 0.8),
                "train_annual_return": rng.uniform(-0.05, 0.3),
                "train_sharpe": rng.uniform(-0.5, 2.0),
                "train_max_drawdown": rng.uniform(-0.5, -0.05),
                "train_annual_volatility": rng.uniform(0.1, 0.3),
                "train_excess_vs_buy_hold": rng.uniform(-0.1, 0.2),
                "test_total_return": rng.normal(0.05, 0.15),
                "test_annual_return": rng.normal(0.05, 0.15),
                "test_sharpe": rng.normal(0.3, 0.8),
                "test_max_drawdown": rng.uniform(-0.5, -0.05),
                "test_annual_volatility": rng.uniform(0.1, 0.3),
                "test_excess_vs_buy_hold": rng.uniform(-0.1, 0.2),
                "test_days": 252,
                "test_trade_count": rng.randint(5, 30),
                "test_year": year,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_metrics_from_daily
# ---------------------------------------------------------------------------

class TestComputeMetricsFromDaily:
    def test_normal_series(self):
        rng = np.random.RandomState(99)
        rets = pd.Series(rng.normal(0.0004, 0.015, 500))
        m = compute_metrics_from_daily(rets)
        expected_total = (1 + rets).prod() - 1
        assert abs(m["total_return"] - expected_total) < 1e-10
        assert not np.isnan(m["sharpe"])
        assert not np.isnan(m["max_drawdown"])

    def test_empty_series(self):
        m = compute_metrics_from_daily(pd.Series([], dtype=float))
        assert np.isnan(m["total_return"])
        assert np.isnan(m["sharpe"])

    def test_positive_drift(self):
        rets = pd.Series([0.001] * 252)
        m = compute_metrics_from_daily(rets)
        assert m["total_return"] > 0
        assert m["sharpe"] > 0


# ---------------------------------------------------------------------------
# compute_benchmark_return
# ---------------------------------------------------------------------------

class TestComputeBenchmarkReturn:
    def test_no_leakage(self):
        dates = pd.bdate_range("2020-01-01", periods=1000)
        prices = np.linspace(100, 200, 1000)
        bm = pd.DataFrame({"date": dates, "close": prices})
        ret = compute_benchmark_return(bm, "2021-01-01", "2021-12-31")
        sub = bm[(bm["date"] >= "2021-01-01") & (bm["date"] <= "2021-12-31")]
        expected = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1
        assert abs(ret - expected) < 1e-10

    def test_short_range_returns_nan(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        bm = pd.DataFrame({"date": dates, "close": range(10)})
        ret = compute_benchmark_return(bm, "2025-01-01", "2025-12-31")
        assert np.isnan(ret)


# ---------------------------------------------------------------------------
# build_variant_stability
# ---------------------------------------------------------------------------

class TestBuildVariantStability:
    def test_returns_dataframe(self):
        result = build_variant_stability(_make_synthetic_detail())
        assert not result.empty
        assert "stability_label" in result.columns

    def test_empty_returns_empty(self):
        result = build_variant_stability(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# build_parameter_stability
# ---------------------------------------------------------------------------

class TestBuildParameterStability:
    def test_v7_param_cols(self):
        detail = _make_synthetic_detail()
        result = build_parameter_stability(detail, ["reversal_window", "vol_window", "turnover_short", "turnover_long", "divergence_window"])
        assert not result.empty
        for col in ["reversal_window", "vol_window"]:
            assert col in result.columns

    def test_empty_returns_empty(self):
        result = build_parameter_stability(pd.DataFrame(), ["reversal_window"])
        assert result.empty


# ---------------------------------------------------------------------------
# build_concentration
# ---------------------------------------------------------------------------

class TestBuildConcentration:
    def test_returns_dataframe(self):
        result = build_concentration(_make_synthetic_detail())
        assert "contribution_rank" in result.columns

    def test_empty_returns_empty(self):
        result = build_concentration(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# build_train_test_stability
# ---------------------------------------------------------------------------

class TestBuildTrainTestStability:
    def test_returns_dataframe(self):
        result = build_train_test_stability(_make_synthetic_detail())
        if not result.empty:
            assert "metric_pair" in result.columns

    def test_empty_returns_empty(self):
        result = build_train_test_stability(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# build_scenarios
# ---------------------------------------------------------------------------

class TestBuildScenarios:
    def test_returns_four_dataframes(self):
        daily = _make_synthetic_daily()
        period = pd.DataFrame({"test_year": [2021, 2022, 2023, 2024, 2025]})
        summary, loyo, exclude, yearly = build_scenarios(daily, period, {}, [2025])
        assert not summary.empty
        assert not loyo.empty
        assert not exclude.empty
        assert not yearly.empty

    def test_robustness_labels(self):
        daily = _make_synthetic_daily()
        period = pd.DataFrame({"test_year": [2021, 2022, 2023, 2024, 2025]})
        summary, _, _, _ = build_scenarios(daily, period, {}, [2025])
        valid = {"robust", "marginal", "weak"}
        assert set(summary["robustness_label"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# build_benchmark_comparison
# ---------------------------------------------------------------------------

class TestBuildBenchmarkComparison:
    def test_returns_dataframe_with_benchmark(self):
        daily = _make_synthetic_daily()
        all_years = sorted(daily["test_year"].unique())
        dates = pd.bdate_range("2020-01-01", periods=1500)
        bm_df = pd.DataFrame({"date": dates, "close": np.linspace(100, 200, 1500)})
        result = build_benchmark_comparison(daily, {"000300.SH": bm_df}, all_years, [2025])
        assert not result.empty
        assert "excess_return" in result.columns

    def test_empty_benchmarks(self):
        daily = _make_synthetic_daily()
        all_years = sorted(daily["test_year"].unique())
        result = build_benchmark_comparison(daily, {}, all_years, [2025])
        assert result.empty


# ---------------------------------------------------------------------------
# evaluate_gates
# ---------------------------------------------------------------------------

class TestEvaluateGates:
    def test_returns_gates_and_decision(self):
        daily = _make_synthetic_daily()
        detail = _make_synthetic_detail()
        period = pd.DataFrame({"test_year": [2021, 2022, 2023, 2024, 2025]})
        summary, loyo, _, yearly = build_scenarios(daily, period, {}, [2025])
        variant = build_variant_stability(detail)
        tt = build_train_test_stability(detail)
        conc = build_concentration(detail)

        gates, decision = evaluate_gates(summary, loyo, variant, tt, conc, yearly, [2025])
        assert isinstance(gates, list)
        assert len(gates) > 0
        assert decision in ("promote_to_portfolio_backtest", "revise_alpha_signal",
                            "continue_to_robustness_validation")
        for g in gates:
            assert "gate_name" in g
            assert "pass" in g

    def test_dynamic_exclude_years(self):
        daily = _make_synthetic_daily()
        detail = _make_synthetic_detail()
        period = pd.DataFrame({"test_year": [2021, 2022, 2023, 2024, 2025]})
        summary, loyo, _, yearly = build_scenarios(daily, period, {}, [2024, 2025])
        variant = build_variant_stability(detail)
        tt = build_train_test_stability(detail)
        conc = build_concentration(detail)

        gates, _ = evaluate_gates(summary, loyo, variant, tt, conc, yearly, [2024, 2025])
        gate_names = {g["gate_name"] for g in gates}
        assert "exclude_2024_positive_return" in gate_names
        assert "exclude_2025_positive_return" in gate_names


# ---------------------------------------------------------------------------
# load_input_files error handling
# ---------------------------------------------------------------------------

class TestLoadInputFiles:
    def test_raises_on_missing_files(self):
        with pytest.raises(FileNotFoundError, match="缺少以下必需文件"):
            load_input_files(Path("/nonexistent"), "fake_tag", "wf_alpha_v6_stock_")
