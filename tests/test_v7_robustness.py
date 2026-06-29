# -*- coding: utf-8 -*-
"""
tests/test_v7_robustness.py

验证 validate_alpha_v7_robustness.py 的导入、配置、核心构建函数和金融正确性。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestV7RobustnessImport:
    def test_import_main_module(self):
        import scripts.validate_alpha_v7_robustness as mod
        assert hasattr(mod, "main")
        assert hasattr(mod, "PARAM_COLS")

    def test_param_cols_match_v7(self):
        from scripts.validate_alpha_v7_robustness import PARAM_COLS
        assert PARAM_COLS == ["reversal_window", "vol_window", "turnover_short",
                              "turnover_long", "divergence_window"]

    def test_default_dirs_are_v7(self):
        from scripts.common.wf_robustness_shared import make_v7_config
        cfg = make_v7_config(PROJECT_ROOT)
        assert "alpha_v7" in str(cfg.default_wf_dir)
        assert "alpha_v7" in str(cfg.default_analysis_dir)
        assert "strategy_robustness" in str(cfg.default_output_root)

    def test_prefix_is_v7(self):
        from scripts.common.wf_robustness_shared import make_v7_config
        cfg = make_v7_config(PROJECT_ROOT)
        assert "wf_alpha_v7_stock_" in cfg.file_prefix


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_synthetic_daily(n_years: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create synthetic portfolio_daily DataFrame for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_years * 252)
    years = []
    for d in dates:
        years.append(d.year)
    rets = rng.normal(0.0004, 0.015, len(dates))
    df = pd.DataFrame({
        "date": dates,
        "portfolio_ret": rets,
        "equity": (1 + pd.Series(rets)).cumprod().values,
        "test_year": years,
    })
    return df


def _make_synthetic_detail(n_stocks: int = 10, n_years: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create synthetic test_detail DataFrame for testing."""
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
# Build function tests
# ---------------------------------------------------------------------------

class TestBuildVariantStability:
    def test_returns_dataframe(self):
        from scripts.common.wf_robustness_shared import build_variant_stability
        detail = _make_synthetic_detail()
        result = build_variant_stability(detail)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "alpha_variant" in result.columns
        assert "stability_label" in result.columns

    def test_empty_detail_returns_empty(self):
        from scripts.common.wf_robustness_shared import build_variant_stability
        result = build_variant_stability(pd.DataFrame())
        assert result.empty

    def test_stability_labels(self):
        from scripts.common.wf_robustness_shared import build_variant_stability
        detail = _make_synthetic_detail()
        result = build_variant_stability(detail)
        valid_labels = {"stable", "unstable"}
        assert set(result["stability_label"].unique()).issubset(valid_labels)


class TestBuildParameterStability:
    def test_returns_dataframe(self):
        from scripts.common.wf_robustness_shared import build_parameter_stability
        detail = _make_synthetic_detail()
        param_cols = ["reversal_window", "vol_window", "turnover_short",
                      "turnover_long", "divergence_window"]
        result = build_parameter_stability(detail, param_cols)
        assert isinstance(result, pd.DataFrame)
        # Should have v7 param columns
        for col in ["reversal_window", "vol_window"]:
            assert col in result.columns

    def test_empty_detail_returns_empty(self):
        from scripts.common.wf_robustness_shared import build_parameter_stability
        result = build_parameter_stability(pd.DataFrame(), ["reversal_window"])
        assert result.empty


class TestBuildConcentration:
    def test_returns_dataframe(self):
        from scripts.common.wf_robustness_shared import build_concentration
        detail = _make_synthetic_detail()
        result = build_concentration(detail)
        assert isinstance(result, pd.DataFrame)
        assert "symbol" in result.columns
        assert "contribution_rank" in result.columns

    def test_empty_returns_empty(self):
        from scripts.common.wf_robustness_shared import build_concentration
        result = build_concentration(pd.DataFrame())
        assert result.empty


class TestBuildTrainTestStability:
    def test_returns_dataframe(self):
        from scripts.common.wf_robustness_shared import build_train_test_stability
        detail = _make_synthetic_detail()
        result = build_train_test_stability(detail)
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "metric_pair" in result.columns
            assert "correlation" in result.columns

    def test_empty_returns_empty(self):
        from scripts.common.wf_robustness_shared import build_train_test_stability
        result = build_train_test_stability(pd.DataFrame())
        assert result.empty


class TestBuildBenchmarkComparison:
    def test_returns_dataframe_with_benchmark(self):
        from scripts.common.wf_robustness_shared import build_benchmark_comparison
        daily = _make_synthetic_daily()
        all_years = sorted(daily["test_year"].unique())
        # Create a synthetic benchmark
        dates = pd.bdate_range("2020-01-01", periods=1500)
        bm_df = pd.DataFrame({"date": dates, "close": np.linspace(100, 200, 1500)})
        benchmarks_data = {"000300.SH": bm_df}
        result = build_benchmark_comparison(daily, benchmarks_data, all_years, [2025])
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "scenario" in result.columns
        assert "excess_return" in result.columns

    def test_empty_benchmarks_returns_empty(self):
        from scripts.common.wf_robustness_shared import build_benchmark_comparison
        daily = _make_synthetic_daily()
        all_years = sorted(daily["test_year"].unique())
        result = build_benchmark_comparison(daily, {}, all_years, [2025])
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestBuildScenarios:
    def test_returns_four_dataframes(self):
        from scripts.common.wf_robustness_shared import build_scenarios
        daily = _make_synthetic_daily()
        period = pd.DataFrame({"test_year": [2021, 2022, 2023, 2024, 2025]})
        summary, loyo, exclude, yearly = build_scenarios(daily, period, {}, [2025])
        assert not summary.empty
        assert not loyo.empty
        assert not exclude.empty
        assert not yearly.empty

    def test_robustness_labels(self):
        from scripts.common.wf_robustness_shared import build_scenarios
        daily = _make_synthetic_daily()
        period = pd.DataFrame({"test_year": [2021, 2022, 2023, 2024, 2025]})
        summary, _, _, _ = build_scenarios(daily, period, {}, [2025])
        valid = {"robust", "marginal", "weak"}
        assert set(summary["robustness_label"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# Gate evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluateGates:
    def test_returns_gates_and_decision(self):
        from scripts.common.wf_robustness_shared import (
            build_scenarios, build_variant_stability, build_train_test_stability,
            build_concentration, evaluate_gates)
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
            assert isinstance(g["pass"], (bool, np.bool_))


# ---------------------------------------------------------------------------
# Financial correctness tests
# ---------------------------------------------------------------------------

class TestFinancialCorrectness:
    def test_compute_metrics_no_future_data(self):
        """compute_metrics_from_daily uses only the input series, no external data."""
        from scripts.common.wf_robustness_shared import compute_metrics_from_daily
        rng = np.random.RandomState(99)
        rets = pd.Series(rng.normal(0.0004, 0.015, 500))
        m = compute_metrics_from_daily(rets)
        # Verify total_return = (1+r).prod() - 1
        expected_total = (1 + rets).prod() - 1
        assert abs(m["total_return"] - expected_total) < 1e-10

    def test_metrics_empty_series(self):
        from scripts.common.wf_robustness_shared import compute_metrics_from_daily
        m = compute_metrics_from_daily(pd.Series([], dtype=float))
        assert np.isnan(m["total_return"])
        assert np.isnan(m["sharpe"])

    def test_benchmark_return_no_leakage(self):
        """compute_benchmark_return only uses data within the date range."""
        from scripts.common.wf_robustness_shared import compute_benchmark_return
        dates = pd.bdate_range("2020-01-01", periods=1000)
        prices = np.linspace(100, 200, 1000)
        bm = pd.DataFrame({"date": dates, "close": prices})
        # Only use middle range
        ret = compute_benchmark_return(bm, "2021-01-01", "2021-12-31")
        sub = bm[(bm["date"] >= "2021-01-01") & (bm["date"] <= "2021-12-31")]
        expected = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1
        assert abs(ret - expected) < 1e-10


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_exp007_has_robustness_command(self):
        import json
        with open(PROJECT_ROOT / "configs" / "research_experiments.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        exp = [e for e in data["experiments"]
               if e["experiment_id"] == "exp_007_alpha_v7_expression_layer"][0]
        assert "robustness" in exp["commands"]
        assert "alpha_v7_robustness" in exp["commands"]["robustness"]

    def test_exp007_has_robustness_dir(self):
        import json
        with open(PROJECT_ROOT / "configs" / "research_experiments.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        exp = [e for e in data["experiments"]
               if e["experiment_id"] == "exp_007_alpha_v7_expression_layer"][0]
        assert exp["outputs"]["robustness_dir"] is not None
        assert "alpha_v7_robustness" in exp["outputs"]["robustness_dir"]
