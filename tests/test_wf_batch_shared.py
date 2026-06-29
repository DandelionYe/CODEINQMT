# -*- coding: utf-8 -*-
"""
tests/test_wf_batch_shared.py

测试 scripts/common/wf_batch_shared.py 共享模块。
"""

from __future__ import annotations

import sys
import unittest.mock
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.wf_batch_shared import (
    WFConfig,
    WFValidateConfig,
    build_batch_tag,
    build_catalog,
    build_param_signature,
    build_variant_param_combos,
    build_validate_tag,
    calc_portfolio_period_metrics,
    calc_score,
    calc_train_score,
    compact_benchmark_list,
    compact_int_list,
    compact_variant_list,
    get_required_rows,
    get_test_years,
    load_benchmark_cache,
    load_symbol_data,
    pass_train_filters_simple,
    process_one_stock,
    run_alpha_frame,
    run_one_backtest,
    save_validate_outputs,
    train_one_stock_all_years,
)
# Import test_selected_for_period via module reference to avoid pytest collecting it as a test
import scripts.common.wf_batch_shared as _wf_batch_mod


# ---------------------------------------------------------------------------
# build_variant_param_combos
# ---------------------------------------------------------------------------

class TestBuildVariantParamCombos:
    def test_single_variant_short_term_reversal(self):
        combos = build_variant_param_combos(
            ["short_term_reversal"], [5, 10], [60], [10], [60], [20],
        )
        assert len(combos) == 2
        assert combos[0] == ("short_term_reversal", 5, 60, 10, 60, 20)
        assert combos[1] == ("short_term_reversal", 10, 60, 10, 60, 20)

    def test_single_variant_low_volatility(self):
        combos = build_variant_param_combos(
            ["low_volatility"], [10], [20, 60], [10], [60], [20],
        )
        assert len(combos) == 2
        assert combos[0] == ("low_volatility", 10, 20, 10, 60, 20)
        assert combos[1] == ("low_volatility", 10, 60, 10, 60, 20)

    def test_single_variant_turnover_reversal(self):
        combos = build_variant_param_combos(
            ["turnover_reversal"], [10], [60], [5, 10], [30, 60], [20],
        )
        assert len(combos) == 4  # 2 x 2

    def test_single_variant_volume_price_divergence(self):
        combos = build_variant_param_combos(
            ["volume_price_divergence"], [10], [60], [10], [60], [10, 20],
        )
        assert len(combos) == 2

    def test_all_variants(self):
        combos = build_variant_param_combos(
            ["short_term_reversal", "low_volatility", "turnover_reversal", "volume_price_divergence"],
            [5, 10], [20, 60], [10], [60], [10, 20],
        )
        # str: 2, lv: 2, tr: 1, vpd: 2 = 7
        assert len(combos) == 7

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="未知的 alpha_variant"):
            build_variant_param_combos(["unknown_variant"], [10], [60], [10], [60], [20])


# ---------------------------------------------------------------------------
# compact_int_list
# ---------------------------------------------------------------------------

class TestCompactIntList:
    def test_single(self):
        assert compact_int_list([10]) == "10"

    def test_range(self):
        assert compact_int_list([5, 10, 20]) == "5-20x3"

    def test_duplicates(self):
        assert compact_int_list([10, 10, 10]) == "10"

    def test_empty(self):
        assert compact_int_list([]) == "none"


# ---------------------------------------------------------------------------
# compact_variant_list
# ---------------------------------------------------------------------------

class TestCompactVariantList:
    TAGS = {
        "short_term_reversal": "str",
        "low_volatility": "lv",
        "turnover_reversal": "tr",
        "volume_price_divergence": "vpd",
    }

    def test_all4(self):
        result = compact_variant_list(
            ["short_term_reversal", "low_volatility", "turnover_reversal", "volume_price_divergence"],
            self.TAGS,
        )
        assert result == "all4"

    def test_subset(self):
        result = compact_variant_list(["short_term_reversal", "low_volatility"], self.TAGS)
        assert result == "str-lv"


# ---------------------------------------------------------------------------
# compact_benchmark_list
# ---------------------------------------------------------------------------

class TestCompactBenchmarkList:
    def test_single(self):
        result = compact_benchmark_list(["000300.SH"])
        assert "000300" in result or result == "000300.SH"

    def test_multiple(self):
        result = compact_benchmark_list(["000300.SH", "000905.SH"])
        assert result == "bm2"


# ---------------------------------------------------------------------------
# build_batch_tag
# ---------------------------------------------------------------------------

class TestBuildBatchTag:
    def test_basic(self):
        cfg = WFConfig(
            alpha_version="alpha_v6",
            output_dir_name="test",
            compute_signals_fn=lambda *a: None,
            prepare_benchmark_regime_fn=lambda *a: None,
            valid_variants=["short_term_reversal"],
        )
        tag = build_batch_tag(
            cfg, ["short_term_reversal"], [10], [60], [10], [60], [20],
            ["000300.SH"], [120], "short",
        )
        assert "alpha_v6" in tag
        assert "short" in tag


# ---------------------------------------------------------------------------
# calc_score
# ---------------------------------------------------------------------------

class TestCalcScore:
    def test_basic(self):
        metrics = {
            "strategy_annual_return": 0.10,
            "strategy_sharpe": 1.0,
            "excess_vs_buy_hold_total_return": 0.05,
            "strategy_max_drawdown": -0.20,
        }
        score = calc_score(metrics)
        assert score == 0.10 + 0.20 * 1.0 + 0.30 * 0.05 + (-0.20)


# ---------------------------------------------------------------------------
# calc_train_score
# ---------------------------------------------------------------------------

class TestCalcTrainScore:
    def test_basic(self):
        metrics = {
            "annual_return": 0.10,
            "sharpe": 1.0,
            "excess_vs_buy_hold_total_return": 0.05,
            "max_drawdown": -0.20,
        }
        score = calc_train_score(metrics)
        assert score == 0.10 + 0.25 * 1.0 + 0.20 * 0.05 + 0.80 * (-0.20)


# ---------------------------------------------------------------------------
# get_required_rows
# ---------------------------------------------------------------------------

class TestGetRequiredRows:
    def test_short_mode(self):
        result = get_required_rows("short", 10, 60, 60, 20, 120, 10, 1500, 0)
        # max(10, 60, 60, 20, 300) + 10 = 310
        assert result == 310

    def test_long_mode(self):
        result = get_required_rows("long", 10, 60, 60, 20, 120, 10, 1500, 0)
        assert result == max(1500, 310)

    def test_custom_mode(self):
        result = get_required_rows("custom", 10, 60, 60, 20, 120, 10, 1500, 500)
        assert result == max(500, 310)


# ---------------------------------------------------------------------------
# pass_train_filters_simple
# ---------------------------------------------------------------------------

class TestPassTrainFiltersSimple:
    def test_pass(self):
        metrics = {
            "annual_return": 0.10,
            "max_drawdown": -0.30,
            "sharpe": 0.5,
            "days": 1500,
            "trade_count": 10,
            "annual_volatility": 0.20,
        }
        assert pass_train_filters_simple(
            metrics, 1000, 4, -0.55, 0.2, 0.02, 0.0, 0.0,
        )

    def test_fail_low_rows(self):
        metrics = {
            "annual_return": 0.10,
            "max_drawdown": -0.30,
            "sharpe": 0.5,
            "days": 500,
            "trade_count": 10,
            "annual_volatility": 0.20,
        }
        assert not pass_train_filters_simple(
            metrics, 1000, 4, -0.55, 0.2, 0.02, 0.0, 0.0,
        )

    def test_fail_nan(self):
        metrics = {
            "annual_return": np.nan,
            "max_drawdown": -0.30,
            "sharpe": 0.5,
            "days": 1500,
            "trade_count": 10,
            "annual_volatility": 0.20,
        }
        assert not pass_train_filters_simple(
            metrics, 1000, 4, -0.55, 0.2, 0.02, 0.0, 0.0,
        )


# ---------------------------------------------------------------------------
# calc_portfolio_period_metrics
# ---------------------------------------------------------------------------

class TestCalcPortfolioPeriodMetrics:
    def test_empty(self):
        result = calc_portfolio_period_metrics(pd.DataFrame(), 2024, 1_000_000)
        assert result["test_year"] == 2024
        assert result["portfolio_size_actual"] == 0

    def test_basic(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        returns_df = pd.DataFrame({
            "A": np.random.randn(100) * 0.01,
            "B": np.random.randn(100) * 0.01,
        }, index=dates)
        result = calc_portfolio_period_metrics(returns_df, 2024, 1_000_000)
        assert result["test_year"] == 2024
        assert result["portfolio_size_actual"] == 2
        assert "total_return" in result
        assert "sharpe" in result


# ---------------------------------------------------------------------------
# WFConfig / WFValidateConfig
# ---------------------------------------------------------------------------

class TestWFConfig:
    def test_creation(self):
        cfg = WFConfig(
            alpha_version="v6",
            output_dir_name="test",
            compute_signals_fn=lambda: None,
            prepare_benchmark_regime_fn=lambda: None,
            valid_variants=["a", "b"],
        )
        assert cfg.alpha_version == "v6"
        assert len(cfg.variant_tags) == 4


class TestWFValidateConfig:
    def test_creation(self):
        cfg = WFValidateConfig(
            alpha_version="v7",
            output_dir_name="test",
            file_prefix="wf_v7",
            report_title="Test Report",
            compute_signals_fn=lambda: None,
            prepare_benchmark_regime_fn=lambda: None,
            valid_variants=["a"],
        )
        assert cfg.alpha_version == "v7"
        assert cfg.file_prefix == "wf_v7"


# ---------------------------------------------------------------------------
# build_param_signature
# ---------------------------------------------------------------------------

class TestBuildParamSignature:
    def test_basic(self):
        sig = build_param_signature(
            ["short_term_reversal"], [10], [60], [10], [60], [20],
            ["000300.SH"], [120],
        )
        assert sig == "short_term_reversal|10|60|10|60|20|000300.SH|120"

    def test_multiple_values(self):
        sig = build_param_signature(
            ["short_term_reversal", "low_volatility"],
            [5, 10], [20, 60], [10], [60], [10, 20],
            ["000300.SH", "000905.SH"], [120, 250],
        )
        parts = sig.split("|")
        assert len(parts) == 8
        assert parts[0] == "short_term_reversal,low_volatility"
        assert parts[1] == "5,10"
        assert parts[6] == "000300.SH,000905.SH"
        assert parts[7] == "120,250"

    def test_deterministic(self):
        args = (
            ["short_term_reversal"], [10], [60], [10], [60], [20],
            ["000300.SH"], [120],
        )
        assert build_param_signature(*args) == build_param_signature(*args)

    def test_different_params_different_sig(self):
        sig1 = build_param_signature(
            ["short_term_reversal"], [10], [60], [10], [60], [20],
            ["000300.SH"], [120],
        )
        sig2 = build_param_signature(
            ["short_term_reversal"], [5], [60], [10], [60], [20],
            ["000300.SH"], [120],
        )
        assert sig1 != sig2


# ---------------------------------------------------------------------------
# get_test_years
# ---------------------------------------------------------------------------

class TestGetTestYears:
    def _make_args(self, first, last):
        import argparse
        return argparse.Namespace(first_test_year=first, last_test_year=last)

    def test_basic_range(self):
        args = self._make_args(2021, 2025)
        assert get_test_years(args) == [2021, 2022, 2023, 2024, 2025]

    def test_single_year(self):
        args = self._make_args(2024, 2024)
        assert get_test_years(args) == [2024]

    def test_last_zero_uses_current_year(self):
        args = self._make_args(2023, 0)
        result = get_test_years(args)
        assert result[0] == 2023
        assert result[-1] == datetime.now().year
        assert len(result) >= 3

    def test_last_negative_uses_current_year(self):
        args = self._make_args(2024, -1)
        result = get_test_years(args)
        assert result[0] == 2024
        assert result[-1] == datetime.now().year


# ---------------------------------------------------------------------------
# build_validate_tag
# ---------------------------------------------------------------------------

class TestBuildValidateTag:
    def _make_args(self):
        import argparse
        return argparse.Namespace(
            alpha_variant_list="short_term_reversal",
            reversal_window_list="10",
            vol_window_list="60",
            turnover_short_list="10",
            turnover_long_list="60",
            divergence_window_list="20",
            benchmark_list="000300.SH",
            benchmark_ma_list="120",
            first_test_year=2021,
            last_test_year=2025,
            market="SH",
            security_type="stock",
            train_start="20150101",
            portfolio_size=20,
            limit=0,
        )

    def test_contains_version(self):
        cfg = WFValidateConfig(
            alpha_version="v7", output_dir_name="test",
            file_prefix="wf_alpha_v7", report_title="Test",
            compute_signals_fn=lambda: None,
            prepare_benchmark_regime_fn=lambda: None,
            valid_variants=["short_term_reversal"],
        )
        tag = build_validate_tag(cfg, self._make_args())
        assert tag.startswith("v7_")

    def test_contains_market_and_security(self):
        cfg = WFValidateConfig(
            alpha_version="v6", output_dir_name="test",
            file_prefix="wf_alpha_v6", report_title="Test",
            compute_signals_fn=lambda: None,
            prepare_benchmark_regime_fn=lambda: None,
            valid_variants=["short_term_reversal"],
        )
        tag = build_validate_tag(cfg, self._make_args())
        assert "SH" in tag
        assert "stock" in tag

    def test_contains_year_range(self):
        cfg = WFValidateConfig(
            alpha_version="v7", output_dir_name="test",
            file_prefix="wf_alpha_v7", report_title="Test",
            compute_signals_fn=lambda: None,
            prepare_benchmark_regime_fn=lambda: None,
            valid_variants=["short_term_reversal"],
        )
        tag = build_validate_tag(cfg, self._make_args())
        assert "2021-2025" in tag

    def test_last_year_zero_shows_latest(self):
        cfg = WFValidateConfig(
            alpha_version="v7", output_dir_name="test",
            file_prefix="wf_alpha_v7", report_title="Test",
            compute_signals_fn=lambda: None,
            prepare_benchmark_regime_fn=lambda: None,
            valid_variants=["short_term_reversal"],
        )
        args = self._make_args()
        args.last_test_year = 0
        tag = build_validate_tag(cfg, args)
        assert "latest" in tag

    def test_limit_tag(self):
        cfg = WFValidateConfig(
            alpha_version="v7", output_dir_name="test",
            file_prefix="wf_alpha_v7", report_title="Test",
            compute_signals_fn=lambda: None,
            prepare_benchmark_regime_fn=lambda: None,
            valid_variants=["short_term_reversal"],
        )
        args = self._make_args()
        args.limit = 50
        tag = build_validate_tag(cfg, args)
        assert "l50" in tag

    def test_limit_zero_tag_all(self):
        cfg = WFValidateConfig(
            alpha_version="v7", output_dir_name="test",
            file_prefix="wf_alpha_v7", report_title="Test",
            compute_signals_fn=lambda: None,
            prepare_benchmark_regime_fn=lambda: None,
            valid_variants=["short_term_reversal"],
        )
        tag = build_validate_tag(cfg, self._make_args())
        assert "all" in tag

    def test_deterministic(self):
        cfg = WFValidateConfig(
            alpha_version="v7", output_dir_name="test",
            file_prefix="wf_alpha_v7", report_title="Test",
            compute_signals_fn=lambda: None,
            prepare_benchmark_regime_fn=lambda: None,
            valid_variants=["short_term_reversal"],
        )
        args = self._make_args()
        assert build_validate_tag(cfg, args) == build_validate_tag(cfg, args)


# ---------------------------------------------------------------------------
# save_validate_outputs
# ---------------------------------------------------------------------------

class TestSaveValidateOutputs:
    def _make_cfg(self, version="v7"):
        return WFValidateConfig(
            alpha_version=version, output_dir_name="test",
            file_prefix=f"wf_alpha_{version}", report_title="Test Report",
            compute_signals_fn=lambda: None,
            prepare_benchmark_regime_fn=lambda: None,
            valid_variants=["short_term_reversal"],
        )

    def _make_args(self):
        import argparse
        return argparse.Namespace(
            alpha_variant_list="short_term_reversal",
            reversal_window_list="10",
            vol_window_list="60",
            turnover_short_list="10",
            turnover_long_list="60",
            divergence_window_list="20",
            benchmark_list="000300.SH",
            benchmark_ma_list="120",
            first_test_year=2021,
            last_test_year=2025,
            market="SH",
            security_type="stock",
            train_start="20150101",
            portfolio_size=20,
            limit=0,
        )

    def test_creates_files(self, tmp_path):
        cfg = self._make_cfg()
        args = self._make_args()
        selected = pd.DataFrame({"symbol": ["A"], "score": [1.0]})
        detail = pd.DataFrame({"symbol": ["A"], "test_return": [0.05]})
        daily = pd.DataFrame({"date": ["2024-01-01"], "ret": [0.01]})
        periods = [{"test_year": 2024, "total_return": 0.05}]

        save_validate_outputs(cfg, selected, detail, daily, periods, args, output_dir=tmp_path)

        files = list(tmp_path.iterdir())
        file_names = [f.name for f in files]
        assert any("selected_by_year" in n for n in file_names)
        assert any("test_detail" in n for n in file_names)
        assert any("portfolio_daily" in n for n in file_names)
        assert any("portfolio_period_summary" in n for n in file_names)
        assert any("report.txt" in n for n in file_names)

    def test_report_contains_title(self, tmp_path):
        cfg = self._make_cfg()
        args = self._make_args()
        save_validate_outputs(cfg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], args, output_dir=tmp_path)

        report_files = [f for f in tmp_path.iterdir() if f.name.endswith("report.txt")]
        assert len(report_files) == 1
        content = report_files[0].read_text(encoding="utf-8")
        assert "Test Report" in content

    def test_report_v7_signal_source(self, tmp_path):
        cfg = self._make_cfg("v7")
        args = self._make_args()
        save_validate_outputs(cfg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], args, output_dir=tmp_path)

        report_files = [f for f in tmp_path.iterdir() if f.name.endswith("report.txt")]
        content = report_files[0].read_text(encoding="utf-8")
        assert "expression_layer" in content

    def test_report_v6_no_signal_source(self, tmp_path):
        cfg = self._make_cfg("v6")
        args = self._make_args()
        save_validate_outputs(cfg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], args, output_dir=tmp_path)

        report_files = [f for f in tmp_path.iterdir() if f.name.endswith("report.txt")]
        content = report_files[0].read_text(encoding="utf-8")
        assert "expression_layer" not in content

    def test_report_contains_params(self, tmp_path):
        cfg = self._make_cfg()
        args = self._make_args()
        save_validate_outputs(cfg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], args, output_dir=tmp_path)

        report_files = [f for f in tmp_path.iterdir() if f.name.endswith("report.txt")]
        content = report_files[0].read_text(encoding="utf-8")
        assert "20150101" in content
        assert "2021" in content
        assert "short_term_reversal" in content

    def test_empty_dataframes_no_error(self, tmp_path):
        cfg = self._make_cfg()
        args = self._make_args()
        save_validate_outputs(cfg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], args, output_dir=tmp_path)
        # Should not raise, report still created
        report_files = [f for f in tmp_path.iterdir() if f.name.endswith("report.txt")]
        assert len(report_files) == 1

    def test_portfolio_periods_in_report(self, tmp_path):
        cfg = self._make_cfg()
        args = self._make_args()
        periods = [
            {"test_year": 2023, "total_return": 0.10, "sharpe": 1.5, "max_drawdown": -0.05, "portfolio_size_actual": 20},
            {"test_year": 2024, "total_return": 0.05, "sharpe": 0.8, "max_drawdown": -0.10, "portfolio_size_actual": 18},
        ]
        save_validate_outputs(cfg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), periods, args, output_dir=tmp_path)

        report_files = [f for f in tmp_path.iterdir() if f.name.endswith("report.txt")]
        content = report_files[0].read_text(encoding="utf-8")
        assert "2023" in content
        assert "2024" in content
        assert "Overall" in content


# ---------------------------------------------------------------------------
# load_symbol_data (mocked)
# ---------------------------------------------------------------------------

class TestLoadSymbolData:
    def test_returns_dataframe(self, tmp_path):
        csv_path = tmp_path / "000001.SZ.csv"
        csv_path.write_text("date,open,high,low,close,volume\n2024-01-02,10,11,9,10.5,1000\n")

        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.load_qmt_price_csv",
            return_value=pd.DataFrame({
                "date": pd.to_datetime(["2024-01-02"]),
                "open": [10.0], "high": [11.0], "low": [9.0],
                "close": [10.5], "volume": [1000],
            }),
        ):
            result = load_symbol_data(csv_path, "20240101", "20240131")
            assert result is not None
            assert len(result) == 1

    def test_returns_none_on_error(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("")

        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.load_qmt_price_csv",
            side_effect=RuntimeError("No data"),
        ):
            result = load_symbol_data(csv_path, "20240101", "20240131")
            assert result is None


# ---------------------------------------------------------------------------
# get_test_years edge cases
# ---------------------------------------------------------------------------

class TestGetTestYearsEdge:
    def test_wide_range(self):
        import argparse
        args = argparse.Namespace(first_test_year=2015, last_test_year=2025)
        result = get_test_years(args)
        assert len(result) == 11
        assert result == list(range(2015, 2026))

    def test_order_ascending(self):
        import argparse
        args = argparse.Namespace(first_test_year=2020, last_test_year=2025)
        result = get_test_years(args)
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]


# ---------------------------------------------------------------------------
# process_one_stock (mocked)
# ---------------------------------------------------------------------------

class TestProcessOneStock:
    def _make_args_tuple(self, csv_path, combos, benchmark_cache):
        return (
            {"symbol": "000001.SZ", "market": "SZ", "security_type": "stock", "csv_path": str(csv_path)},
            combos,
            benchmark_cache,
            "20200101", "20241231",
            1_000_000, 0.0001, 0.0005, 0.0,
            "short", 10, 1500, 500,
            lambda *a, **kw: None,  # compute_signals_fn
        )

    def test_empty_data_returns_skipped(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        combos = [("short_term_reversal", 10, 60, 10, 60, 20)]
        bm_cache = {("000300.SH", 120): {"benchmark": "000300.SH", "benchmark_ma": 120, "csv_path": "x", "filter_df": pd.DataFrame()}}

        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.load_qmt_price_csv",
            side_effect=RuntimeError("No data"),
        ):
            rows, skipped, errors = process_one_stock(self._make_args_tuple(csv_path, combos, bm_cache))
            assert len(rows) == 0
            assert len(skipped) == 1
            assert "数据为空" in skipped[0]["reason"]

    def test_insufficient_rows_skipped(self, tmp_path):
        csv_path = tmp_path / "short.csv"
        csv_path.write_text("")
        combos = [("short_term_reversal", 10, 60, 10, 60, 20)]
        bm_cache = {("000300.SH", 120): {"benchmark": "000300.SH", "benchmark_ma": 120, "csv_path": "x", "filter_df": pd.DataFrame()}}
        short_df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "open": [10]*5, "high": [11]*5, "low": [9]*5, "close": [10.5]*5, "volume": [1000]*5,
        })

        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.load_qmt_price_csv",
            return_value=short_df,
        ):
            rows, skipped, errors = process_one_stock(self._make_args_tuple(csv_path, combos, bm_cache))
            assert len(rows) == 0
            assert len(skipped) >= 1
            assert "rows=" in skipped[0]["reason"]

    def test_success_adds_row(self, tmp_path):
        csv_path = tmp_path / "good.csv"
        csv_path.write_text("")
        combos = [("short_term_reversal", 10, 60, 10, 60, 20)]
        dates = pd.date_range("2020-01-01", periods=2000, freq="B")
        good_df = pd.DataFrame({
            "date": dates, "open": np.ones(2000)*10, "high": np.ones(2000)*11,
            "low": np.ones(2000)*9, "close": np.ones(2000)*10.5, "volume": np.ones(2000)*1000,
        })
        filter_df = pd.DataFrame({"date": dates, "market_filter": np.ones(2000)})
        bm_cache = {("000300.SH", 120): {"benchmark": "000300.SH", "benchmark_ma": 120, "csv_path": "x", "filter_df": filter_df}}

        mock_metrics = {
            "strategy_annual_return": 0.1, "strategy_sharpe": 1.0,
            "excess_vs_buy_hold_total_return": 0.05, "strategy_max_drawdown": -0.1,
            "total_return": 0.5, "annual_return": 0.1, "annual_volatility": 0.2,
            "max_drawdown": -0.1, "sharpe": 1.0, "days": 2000, "trade_count": 50,
            "buy_hold_total_return": 0.3,
        }
        mock_result_df = pd.DataFrame({"date": dates, "position": np.ones(2000), "strategy_ret": np.ones(2000)*0.001, "stock_ret": np.ones(2000)*0.0005, "market_filter": np.ones(2000)})

        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.load_qmt_price_csv",
            return_value=good_df,
        ), unittest.mock.patch(
            "scripts.common.wf_batch_shared.run_one_backtest",
            return_value=(mock_result_df, mock_metrics),
        ):
            rows, skipped, errors = process_one_stock(self._make_args_tuple(csv_path, combos, bm_cache))
            assert len(rows) == 1
            assert rows[0]["symbol"] == "000001.SZ"
            assert rows[0]["alpha_variant"] == "short_term_reversal"
            assert "score" in rows[0]


# ---------------------------------------------------------------------------
# train_one_stock_all_years (mocked)
# ---------------------------------------------------------------------------

class TestTrainOneStockAllYears:
    def _make_args_tuple(self):
        return (
            {"symbol": "000001.SZ", "market": "SZ", "security_type": "stock", "csv_path": "/fake/path.csv"},
            [2023, 2024],
            "20200101",
            "20241231",
            [("short_term_reversal", 10, 60, 10, 60, 20)],
            {("000300.SH", 120): {"benchmark": "000300.SH", "benchmark_ma": 120, "csv_path": "/fake/bm.csv", "filter_df": pd.DataFrame({"date": pd.date_range("2020-01-01", periods=1500, freq="B"), "market_filter": np.ones(1500)})}},
            0.0001, 0.0005, 0.0,
            1_000_000, 10, 500, 4, -0.55, 0.2, 0.02, 0.0, 0.0,
        )

    def test_empty_data_returns_empty(self):
        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.load_qmt_price_csv",
            side_effect=RuntimeError("No data"),
        ):
            result = train_one_stock_all_years(self._make_args_tuple(), lambda *a, **kw: None)
            assert result == {}

    def test_no_passing_filters_returns_empty(self):
        dates = pd.date_range("2020-01-01", periods=1500, freq="B")
        df = pd.DataFrame({
            "date": dates, "open": np.ones(1500)*10, "high": np.ones(1500)*11,
            "low": np.ones(1500)*9, "close": np.ones(1500)*10.5, "volume": np.ones(1500)*1000,
        })
        bad_metrics = {
            "total_return": -0.5, "annual_return": -0.2, "annual_volatility": 0.5,
            "max_drawdown": -0.8, "sharpe": -2.0, "days": 500, "trade_count": 2,
            "buy_hold_total_return": 0.1, "excess_total_return": -0.6,
        }
        mock_result = pd.DataFrame({
            "date": dates, "position": np.ones(1500), "strategy_ret": np.ones(1500)*(-0.001),
            "stock_ret": np.ones(1500)*0.0005, "market_filter": np.ones(1500),
        })

        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.load_qmt_price_csv",
            return_value=df,
        ), unittest.mock.patch(
            "scripts.common.wf_batch_shared.run_alpha_frame",
            return_value=mock_result,
        ), unittest.mock.patch(
            "scripts.common.wf_batch_shared.calc_metrics_from_returns",
            return_value=bad_metrics,
        ):
            result = train_one_stock_all_years(self._make_args_tuple(), lambda *a, **kw: None)
            assert result == {}

    def test_success_returns_by_year(self):
        dates = pd.date_range("2020-01-01", periods=1500, freq="B")
        df = pd.DataFrame({
            "date": dates, "open": np.ones(1500)*10, "high": np.ones(1500)*11,
            "low": np.ones(1500)*9, "close": np.ones(1500)*10.5, "volume": np.ones(1500)*1000,
        })
        good_metrics = {
            "total_return": 0.3, "annual_return": 0.1, "annual_volatility": 0.15,
            "max_drawdown": -0.1, "sharpe": 1.2, "days": 1000, "trade_count": 50,
            "buy_hold_total_return": 0.2, "excess_total_return": 0.1,
        }
        mock_result = pd.DataFrame({
            "date": dates, "position": np.ones(1500), "strategy_ret": np.ones(1500)*0.001,
            "stock_ret": np.ones(1500)*0.0005, "market_filter": np.ones(1500),
        })

        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.load_qmt_price_csv",
            return_value=df,
        ), unittest.mock.patch(
            "scripts.common.wf_batch_shared.run_alpha_frame",
            return_value=mock_result,
        ), unittest.mock.patch(
            "scripts.common.wf_batch_shared.calc_metrics_from_returns",
            return_value=good_metrics,
        ):
            result = train_one_stock_all_years(self._make_args_tuple(), lambda *a, **kw: None)
            assert 2023 in result
            assert 2024 in result
            assert result[2023]["symbol"] == "000001.SZ"
            assert "train_score" in result[2023]


# ---------------------------------------------------------------------------
# test_selected_for_period (mocked)
# ---------------------------------------------------------------------------

class TestTestSelectedForPeriod:
    """Tests for test_selected_for_period (imported via module ref to avoid pytest collection)."""

    def _make_selected(self):
        return pd.DataFrame([{
            "symbol": "000001.SZ", "csv_path": "/fake/path.csv",
            "alpha_variant": "short_term_reversal", "reversal_window": 10,
            "vol_window": 60, "turnover_short": 10, "turnover_long": 60,
            "divergence_window": 20, "benchmark": "000300.SH", "benchmark_ma": 120,
            "selected_rank": 1, "train_score": 0.5,
            "train_annual_return": 0.1, "train_sharpe": 1.0,
            "train_max_drawdown": -0.1, "train_annual_volatility": 0.15,
            "train_excess_vs_buy_hold": 0.05,
        }])

    def _call(self, *a, **kw):
        return _wf_batch_mod.test_selected_for_period(*a, **kw)

    def test_empty_selected(self):
        import argparse
        args = argparse.Namespace(commission=0.0001, sell_tax=0.0005, slippage=0.0)
        detail_df, returns_df = self._call(
            pd.DataFrame(), {}, {}, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
            args, lambda *a, **kw: None,
        )
        assert detail_df.empty
        assert returns_df.empty

    def test_missing_data_skips(self):
        import argparse
        args = argparse.Namespace(commission=0.0001, sell_tax=0.0005, slippage=0.0)
        detail_df, returns_df = self._call(
            self._make_selected(), {}, {},
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
            args, lambda *a, **kw: None,
        )
        assert detail_df.empty

    def test_success_returns_detail_and_returns(self):
        import argparse
        args = argparse.Namespace(commission=0.0001, sell_tax=0.0005, slippage=0.0)
        dates = pd.date_range("2024-01-01", periods=250, freq="B")
        full_df = pd.DataFrame({
            "date": dates, "open": np.ones(250)*10, "high": np.ones(250)*11,
            "low": np.ones(250)*9, "close": np.ones(250)*10.5, "volume": np.ones(250)*1000,
        })
        data_cache = {"/fake/path.csv": full_df}
        filter_df = pd.DataFrame({"date": dates, "market_filter": np.ones(250)})
        benchmark_cache = {("000300.SH", 120): {"benchmark": "000300.SH", "benchmark_ma": 120, "csv_path": "/fake/bm.csv", "filter_df": filter_df}}

        mock_result = pd.DataFrame({
            "date": dates, "position": np.ones(250), "strategy_ret": np.ones(250)*0.001,
            "stock_ret": np.ones(250)*0.0005, "market_filter": np.ones(250),
        })
        mock_metrics = {
            "total_return": 0.2, "annual_return": 0.1, "annual_volatility": 0.15,
            "max_drawdown": -0.05, "sharpe": 1.5, "days": 250, "trade_count": 30,
            "buy_hold_total_return": 0.15, "excess_total_return": 0.05,
        }

        with unittest.mock.patch(
            "scripts.common.wf_batch_shared.run_alpha_frame",
            return_value=mock_result,
        ), unittest.mock.patch(
            "scripts.common.wf_batch_shared.calc_metrics_from_returns",
            return_value=mock_metrics,
        ):
            detail_df, returns_df = self._call(
                self._make_selected(), data_cache, benchmark_cache,
                pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
                args, lambda *a, **kw: None,
            )
            assert len(detail_df) == 1
            assert detail_df.iloc[0]["symbol"] == "000001.SZ"
            assert "test_total_return" in detail_df.columns
            assert not returns_df.empty

    def test_missing_benchmark_skips(self):
        import argparse
        args = argparse.Namespace(commission=0.0001, sell_tax=0.0005, slippage=0.0)
        dates = pd.date_range("2024-01-01", periods=250, freq="B")
        full_df = pd.DataFrame({
            "date": dates, "open": np.ones(250)*10, "high": np.ones(250)*11,
            "low": np.ones(250)*9, "close": np.ones(250)*10.5, "volume": np.ones(250)*1000,
        })
        data_cache = {"/fake/path.csv": full_df}
        # Empty benchmark cache — missing (000300.SH, 120)
        detail_df, returns_df = self._call(
            self._make_selected(), data_cache, {},
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
            args, lambda *a, **kw: None,
        )
        assert detail_df.empty


# ---------------------------------------------------------------------------
# build_catalog / load_benchmark_cache (mocked)
# ---------------------------------------------------------------------------

class TestBuildCatalog:
    def test_basic(self, tmp_path):
        import argparse
        mock_catalog = pd.DataFrame({
            "symbol": ["000001.SZ", "600000.SH"],
            "market": ["SZ", "SH"],
            "security_type": ["stock", "stock"],
            "csv_path": ["/a.csv", "/b.csv"],
        })
        args = argparse.Namespace(
            export_root=str(tmp_path), market="ALL", security_type="ALL", limit=0,
        )
        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.scan_qmt_export",
            return_value=mock_catalog,
        ):
            result = build_catalog(args)
            assert len(result) == 2

    def test_market_filter(self, tmp_path):
        import argparse
        mock_catalog = pd.DataFrame({
            "symbol": ["000001.SZ", "600000.SH"],
            "market": ["SZ", "SH"],
            "security_type": ["stock", "stock"],
            "csv_path": ["/a.csv", "/b.csv"],
        })
        args = argparse.Namespace(
            export_root=str(tmp_path), market="SZ", security_type="ALL", limit=0,
        )
        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.scan_qmt_export",
            return_value=mock_catalog,
        ):
            result = build_catalog(args)
            assert len(result) == 1
            assert result.iloc[0]["market"] == "SZ"

    def test_limit(self, tmp_path):
        import argparse
        mock_catalog = pd.DataFrame({
            "symbol": [f"{i:06d}.SZ" for i in range(100)],
            "market": ["SZ"] * 100,
            "security_type": ["stock"] * 100,
            "csv_path": [f"/{i}.csv" for i in range(100)],
        })
        args = argparse.Namespace(
            export_root=str(tmp_path), market="ALL", security_type="ALL", limit=10,
        )
        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.scan_qmt_export",
            return_value=mock_catalog,
        ):
            result = build_catalog(args)
            assert len(result) == 10


class TestLoadBenchmarkCache:
    def test_basic(self, tmp_path):
        dates = pd.date_range("2020-01-01", periods=1500, freq="B")
        bm_df = pd.DataFrame({
            "date": dates, "open": np.ones(1500)*10, "high": np.ones(1500)*11,
            "low": np.ones(1500)*9, "close": np.ones(1500)*10.5, "volume": np.ones(1500)*1000,
        })
        filter_df = pd.DataFrame({"date": dates, "market_filter": np.ones(1500)})

        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.find_csv_for_stock",
            return_value=(tmp_path / "000300.SH.csv", None, None, None),
        ), unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.load_qmt_price_csv",
            return_value=bm_df,
        ), unittest.mock.patch(
            "scripts.common.wf_batch_shared.prepare_benchmark_regime_fn_placeholder",
            create=True,
        ):
            mock_prep = lambda df, ma: filter_df
            cache = load_benchmark_cache(
                mock_prep, "000300.SH", "120", "20200101", "20241231", tmp_path,
            )
            assert ("000300.SH", 120) in cache
            assert cache[("000300.SH", 120)]["benchmark"] == "000300.SH"

    def test_multiple_benchmarks(self, tmp_path):
        dates = pd.date_range("2020-01-01", periods=1500, freq="B")
        bm_df = pd.DataFrame({
            "date": dates, "open": np.ones(1500)*10, "high": np.ones(1500)*11,
            "low": np.ones(1500)*9, "close": np.ones(1500)*10.5, "volume": np.ones(1500)*1000,
        })
        filter_df = pd.DataFrame({"date": dates, "market_filter": np.ones(1500)})

        with unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.find_csv_for_stock",
            return_value=(tmp_path / "bm.csv", None, None, None),
        ), unittest.mock.patch(
            "strategies.ma_demo_strategy_csv.load_qmt_price_csv",
            return_value=bm_df,
        ):
            mock_prep = lambda df, ma: filter_df
            cache = load_benchmark_cache(
                mock_prep, "000300.SH,000905.SH", "120,250", "20200101", "20241231", tmp_path,
            )
            assert ("000300.SH", 120) in cache
            assert ("000300.SH", 250) in cache
            assert ("000905.SH", 120) in cache
            assert ("000905.SH", 250) in cache
            assert len(cache) == 4
