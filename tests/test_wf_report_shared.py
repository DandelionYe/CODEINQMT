# -*- coding: utf-8 -*-
"""
tests/test_wf_report_shared.py

覆盖 scripts/common/wf_report_shared.py 中此前无测试的函数。
聚焦纯逻辑函数和 I/O 函数，跳过需要真实 QMT 数据的函数。
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def v7_cfg():
    """最小 WFReportConfig 实例（v7）。"""
    from scripts.common.wf_report_shared import WFReportConfig
    return WFReportConfig(
        default_input_dir=Path("/tmp/in"),
        default_output_dir=Path("/tmp/out"),
        default_analysis_dir=Path("/tmp/analysis"),
        default_wf_dir=Path("/tmp/wf"),
        default_output_root=Path("/tmp/root"),
        file_prefix="wf_alpha_v7_stock_",
        analysis_output_prefix="alpha_v7_wf_analysis",
        diagnosis_output_prefix="alpha_v7_diagnosis",
        param_cols=["reversal_window", "vol_window"],
        display_name="Alpha v7",
        display_suffix=" (expression layer)",
        diagnosis_dir_prefix="alpha_v7_research_diagnosis_",
        analyze_description="test analyze",
        diagnose_description="test diagnose",
    )


@pytest.fixture
def daily_df():
    """构造模拟的 portfolio_daily DataFrame。"""
    dates = pd.date_range("2021-01-01", periods=252, freq="B")
    rng = np.random.RandomState(42)
    ret = rng.normal(0.0005, 0.01, len(dates))
    return pd.DataFrame({
        "date": dates,
        "portfolio_ret": ret,
    })


@pytest.fixture
def combined_df(daily_df):
    """构造模拟的 combined returns DataFrame（strategy + benchmark）。"""
    df = daily_df.set_index("date").sort_index()
    rng = np.random.RandomState(99)
    combined = pd.DataFrame(index=df.index)
    combined["strategy"] = df["portfolio_ret"]
    combined["BENCH_000300.SH_CSI300"] = rng.normal(0.0003, 0.012, len(df))
    return combined


@pytest.fixture
def yearly_df():
    """构造模拟的 yearly comparison DataFrame。"""
    return pd.DataFrame([
        {"entity": "strategy", "year": 2021, "total_return": 0.15, "annual_return": 0.15,
         "max_drawdown": -0.10, "sharpe": 1.2, "annual_volatility": 0.12},
        {"entity": "strategy", "year": 2022, "total_return": -0.05, "annual_return": -0.05,
         "max_drawdown": -0.20, "sharpe": -0.4, "annual_volatility": 0.15},
        {"entity": "strategy", "year": 2023, "total_return": 0.25, "annual_return": 0.25,
         "max_drawdown": -0.08, "sharpe": 2.0, "annual_volatility": 0.10},
        {"entity": "BENCH_000300.SH_CSI300", "year": 2021, "total_return": 0.10,
         "annual_return": 0.10, "max_drawdown": -0.12, "sharpe": 0.8, "annual_volatility": 0.14},
        {"entity": "BENCH_000300.SH_CSI300", "year": 2022, "total_return": -0.15,
         "annual_return": -0.15, "max_drawdown": -0.25, "sharpe": -1.0, "annual_volatility": 0.18},
        {"entity": "BENCH_000300.SH_CSI300", "year": 2023, "total_return": 0.08,
         "annual_return": 0.08, "max_drawdown": -0.10, "sharpe": 0.6, "annual_volatility": 0.13},
    ])


@pytest.fixture
def detail_df():
    """构造模拟的 test_detail DataFrame。"""
    return pd.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ", "600000.SH"],
        "alpha_variant": ["short_term_reversal", "low_volatility", "short_term_reversal",
                          "turnover_reversal", "short_term_reversal"],
        "reversal_window": [5, 10, 5, 20, 10],
        "vol_window": [20, 60, 20, 60, 20],
        "train_annual_return": [0.10, 0.08, -0.02, 0.15, 0.06],
        "train_sharpe": [0.8, 0.6, -0.1, 1.2, 0.5],
        "train_score": [0.05, 0.04, -0.01, 0.08, 0.03],
        "test_total_return": [0.05, -0.03, 0.02, -0.10, 0.01],
        "test_sharpe": [0.4, -0.2, 0.1, -0.8, 0.05],
        "year": [2021, 2022, 2021, 2022, 2023],
    })


@pytest.fixture
def selected_df():
    """构造模拟的 selected_by_year DataFrame。"""
    return pd.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ", "000001.SZ", "600000.SH", "000002.SZ"],
        "selected_rank": [1, 2, 3, 1, 2],
        "train_score": [0.05, 0.04, 0.03, 0.06, 0.02],
        "alpha_variant": ["short_term_reversal", "low_volatility", "short_term_reversal",
                          "turnover_reversal", "short_term_reversal"],
        "reversal_window": [5, 10, 5, 20, 5],
        "vol_window": [20, 60, 20, 60, 20],
        "benchmark": ["000300.SH", "000300.SH", "000905.SH", "000300.SH", "000300.SH"],
        "benchmark_ma": [120, 120, 60, 120, 120],
        "year": [2021, 2021, 2022, 2022, 2023],
    })


# ---------------------------------------------------------------------------
# infer_incomplete_year
# ---------------------------------------------------------------------------

class TestInferIncompleteYear:
    def test_user_value_provided(self):
        from scripts.common.wf_report_shared import infer_incomplete_year
        combined = pd.DataFrame({"val": range(5)}, index=pd.date_range("2023-06-01", periods=5))
        assert infer_incomplete_year(combined, 2023) == 2023

    def test_user_value_zero(self):
        from scripts.common.wf_report_shared import infer_incomplete_year
        dates = pd.date_range("2023-11-01", periods=30, freq="B")
        combined = pd.DataFrame({"val": range(len(dates))}, index=dates)
        assert infer_incomplete_year(combined, 0) == 2023

    def test_empty_combined(self):
        from scripts.common.wf_report_shared import infer_incomplete_year
        combined = pd.DataFrame()
        assert infer_incomplete_year(combined, 0) is None

    def test_december_before_15th(self):
        from scripts.common.wf_report_shared import infer_incomplete_year
        dates = pd.date_range("2023-12-01", periods=10, freq="B")
        combined = pd.DataFrame({"val": range(len(dates))}, index=dates)
        assert infer_incomplete_year(combined, 0) == 2023

    def test_december_after_15th(self):
        from scripts.common.wf_report_shared import infer_incomplete_year
        dates = pd.date_range("2023-12-16", periods=10, freq="B")
        combined = pd.DataFrame({"val": range(len(dates))}, index=dates)
        assert infer_incomplete_year(combined, 0) is None

    def test_november(self):
        from scripts.common.wf_report_shared import infer_incomplete_year
        dates = pd.date_range("2023-11-15", periods=10, freq="B")
        combined = pd.DataFrame({"val": range(len(dates))}, index=dates)
        assert infer_incomplete_year(combined, 0) == 2023

    def test_full_year(self):
        from scripts.common.wf_report_shared import infer_incomplete_year
        dates = pd.date_range("2022-01-01", periods=252, freq="B")
        combined = pd.DataFrame({"val": range(len(dates))}, index=dates)
        # 2022-12-30 is before Dec 15 → still incomplete
        result = infer_incomplete_year(combined, 0)
        # The latest date in 2022 business days is around Dec 30
        if combined.index.max().month == 12 and combined.index.max().day >= 15:
            assert result is None
        else:
            assert result == 2022


# ---------------------------------------------------------------------------
# drawdown_series
# ---------------------------------------------------------------------------

class TestDrawdownSeries:
    def test_monotonic_increase(self):
        from scripts.common.wf_report_shared import drawdown_series
        ret = pd.Series([0.01, 0.02, 0.01, 0.03])
        dd = drawdown_series(ret)
        assert (dd >= -1e-10).all()  # no drawdown
        assert dd.iloc[0] == pytest.approx(0.0)

    def test_known_drawdown(self):
        from scripts.common.wf_report_shared import drawdown_series
        ret = pd.Series([0.10, -0.20, 0.05])
        dd = drawdown_series(ret)
        # After 0.10: equity=1.1, max=1.1 → dd=0
        # After -0.20: equity=0.88, max=1.1 → dd=-0.2
        assert dd.iloc[1] == pytest.approx(-0.2, abs=0.001)

    def test_single_element(self):
        from scripts.common.wf_report_shared import drawdown_series
        ret = pd.Series([0.05])
        dd = drawdown_series(ret)
        assert len(dd) == 1
        assert dd.iloc[0] == pytest.approx(0.0)

    def test_all_negative(self):
        from scripts.common.wf_report_shared import drawdown_series
        ret = pd.Series([-0.01, -0.02, -0.01])
        dd = drawdown_series(ret)
        assert (dd <= 0).all()


# ---------------------------------------------------------------------------
# build_combined_returns
# ---------------------------------------------------------------------------

class TestBuildCombinedReturns:
    def test_basic(self, daily_df):
        from scripts.common.wf_report_shared import build_combined_returns
        bench = pd.DataFrame(
            {"BENCH_000300.SH_CSI300": [0.001] * len(daily_df)},
            index=pd.DatetimeIndex(daily_df["date"]),
        )
        groups = {"daily": daily_df}
        result = build_combined_returns(groups, bench)
        assert "strategy" in result.columns
        assert "BENCH_000300.SH_CSI300" in result.columns
        assert len(result) == len(daily_df)

    def test_empty_daily(self):
        from scripts.common.wf_report_shared import build_combined_returns
        groups = {"daily": pd.DataFrame()}
        result = build_combined_returns(groups, pd.DataFrame())
        assert result.empty

    def test_missing_daily_key(self):
        from scripts.common.wf_report_shared import build_combined_returns
        groups = {}
        result = build_combined_returns(groups, pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# build_yearly_excess
# ---------------------------------------------------------------------------

class TestBuildYearlyExcess:
    def test_basic(self, yearly_df):
        from scripts.common.wf_report_shared import build_yearly_excess
        result = build_yearly_excess(yearly_df)
        assert not result.empty
        assert "excess_return" in result.columns
        assert "beat_benchmark" in result.columns
        # 2021: strategy 0.15 vs bench 0.10 → beat
        row_2021 = result[result["year"] == 2021].iloc[0]
        assert row_2021["beat_benchmark"] is True or row_2021["beat_benchmark"] == True

    def test_empty_yearly(self):
        from scripts.common.wf_report_shared import build_yearly_excess
        # build_yearly_excess doesn't guard empty columns, pass schema-compatible empty
        empty = pd.DataFrame(columns=["entity", "year", "total_return"])
        result = build_yearly_excess(empty)
        assert result.empty


# ---------------------------------------------------------------------------
# filter_exclude_year
# ---------------------------------------------------------------------------

class TestFilterExcludeYear:
    def test_exclude(self):
        from scripts.common.wf_report_shared import filter_exclude_year
        df = pd.DataFrame({"year": [2021, 2022, 2023], "val": [1, 2, 3]})
        result = filter_exclude_year(df, 2022)
        assert list(result["year"]) == [2021, 2023]

    def test_no_exclude(self):
        from scripts.common.wf_report_shared import filter_exclude_year
        df = pd.DataFrame({"year": [2021, 2022], "val": [1, 2]})
        result = filter_exclude_year(df, None)
        assert len(result) == 2

    def test_custom_year_col(self):
        from scripts.common.wf_report_shared import filter_exclude_year
        df = pd.DataFrame({"test_year": [2021, 2022], "val": [1, 2]})
        result = filter_exclude_year(df, 2021, year_col="test_year")
        assert len(result) == 1

    def test_exclude_value_zero(self):
        from scripts.common.wf_report_shared import filter_exclude_year
        df = pd.DataFrame({"year": [2021, 2022], "val": [1, 2]})
        result = filter_exclude_year(df, 0)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# build_yearly_weakness
# ---------------------------------------------------------------------------

class TestBuildYearlyWeakness:
    def test_basic(self, yearly_df):
        from scripts.common.wf_report_shared import build_yearly_weakness
        yearly_excess = pd.DataFrame({
            "year": [2021, 2022, 2023],
            "excess_return": [0.05, 0.10, 0.17],
            "benchmark_return": [0.10, -0.15, 0.08],
            "strategy_return": [0.15, -0.05, 0.25],
        })
        result = build_yearly_weakness(yearly_df, yearly_excess, exclude_year=None)
        assert not result.empty
        assert "lost_money" in result.columns
        assert "deep_drawdown" in result.columns

    def test_empty_yearly(self):
        from scripts.common.wf_report_shared import build_yearly_weakness
        # build_yearly_weakness doesn't guard empty columns, pass schema-compatible empty
        empty = pd.DataFrame(columns=["entity", "year", "total_return", "max_drawdown", "sharpe"])
        result = build_yearly_weakness(empty, pd.DataFrame(), None)
        assert result.empty

    def test_exclude_year(self, yearly_df):
        from scripts.common.wf_report_shared import build_yearly_weakness
        result = build_yearly_weakness(yearly_df, pd.DataFrame(), exclude_year=2022)
        assert 2022 not in result["year"].values


# ---------------------------------------------------------------------------
# build_train_test_correlation
# ---------------------------------------------------------------------------

class TestBuildTrainTestCorrelation:
    def test_basic(self, detail_df):
        from scripts.common.wf_report_shared import build_train_test_correlation
        result = build_train_test_correlation(detail_df)
        # detail_df has 5 rows, need > 5 for correlation
        # With 5 rows, the condition len(valid) > 5 fails, so result is empty
        # Let's create a larger fixture
        big = pd.concat([detail_df] * 3, ignore_index=True)
        result = build_train_test_correlation(big)
        assert not result.empty
        assert "correlation" in result.columns

    def test_empty(self):
        from scripts.common.wf_report_shared import build_train_test_correlation
        result = build_train_test_correlation(pd.DataFrame())
        assert result.empty

    def test_missing_columns(self):
        from scripts.common.wf_report_shared import build_train_test_correlation
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = build_train_test_correlation(df)
        assert result.empty


# ---------------------------------------------------------------------------
# build_contributors
# ---------------------------------------------------------------------------

class TestBuildContributors:
    def test_basic(self, detail_df):
        from scripts.common.wf_report_shared import build_contributors
        bad, good = build_contributors(detail_df, exclude_year=None)
        # detail_df: 000001.SZ has mixed, 000002.SZ mixed, 600000.SH positive
        assert isinstance(bad, pd.DataFrame)
        assert isinstance(good, pd.DataFrame)

    def test_empty(self):
        from scripts.common.wf_report_shared import build_contributors
        bad, good = build_contributors(pd.DataFrame(), None)
        assert bad.empty and good.empty

    def test_no_symbol_column(self):
        from scripts.common.wf_report_shared import build_contributors
        df = pd.DataFrame({"val": [1, 2, 3]})
        bad, good = build_contributors(df, None)
        assert bad.empty and good.empty


# ---------------------------------------------------------------------------
# build_selected_repetition
# ---------------------------------------------------------------------------

class TestBuildSelectedRepetition:
    def test_basic(self, selected_df, detail_df):
        from scripts.common.wf_report_shared import build_selected_repetition
        # Build a simple selected_freq
        selected_freq = selected_df.groupby("symbol").size().reset_index(name="selected_count")
        result = build_selected_repetition(selected_freq, detail_df, exclude_year=None)
        assert not result.empty
        assert "selected_count" in result.columns

    def test_empty_selected_freq(self, detail_df):
        from scripts.common.wf_report_shared import build_selected_repetition
        result = build_selected_repetition(pd.DataFrame(), detail_df, None)
        assert result.empty

    def test_empty_detail(self, selected_df):
        from scripts.common.wf_report_shared import build_selected_repetition
        selected_freq = selected_df.groupby("symbol").size().reset_index(name="selected_count")
        result = build_selected_repetition(selected_freq, pd.DataFrame(), None)
        assert result.empty


# ---------------------------------------------------------------------------
# analyze_selected_frequency
# ---------------------------------------------------------------------------

class TestAnalyzeSelectedFrequency:
    def test_basic(self, selected_df):
        from scripts.common.wf_report_shared import analyze_selected_frequency
        groups = {"selected": selected_df}
        result = analyze_selected_frequency(groups)
        assert not result.empty
        assert "selected_count" in result.columns
        assert "symbol" in result.columns
        # 000001.SZ appears twice
        row = result[result["symbol"] == "000001.SZ"].iloc[0]
        assert row["selected_count"] == 2

    def test_empty(self):
        from scripts.common.wf_report_shared import analyze_selected_frequency
        result = analyze_selected_frequency({"selected": pd.DataFrame()})
        assert result.empty

    def test_missing_key(self):
        from scripts.common.wf_report_shared import analyze_selected_frequency
        result = analyze_selected_frequency({})
        assert result.empty


# ---------------------------------------------------------------------------
# analyze_alpha_variant_frequency
# ---------------------------------------------------------------------------

class TestAnalyzeAlphaVariantFrequency:
    def test_basic(self, selected_df):
        from scripts.common.wf_report_shared import analyze_alpha_variant_frequency
        groups = {"selected": selected_df}
        result = analyze_alpha_variant_frequency(groups)
        assert not result.empty
        assert "alpha_variant" in result.columns

    def test_no_alpha_variant_column(self):
        from scripts.common.wf_report_shared import analyze_alpha_variant_frequency
        df = pd.DataFrame({"symbol": ["A", "B"]})
        result = analyze_alpha_variant_frequency({"selected": df})
        assert result.empty

    def test_empty(self):
        from scripts.common.wf_report_shared import analyze_alpha_variant_frequency
        result = analyze_alpha_variant_frequency({"selected": pd.DataFrame()})
        assert result.empty


# ---------------------------------------------------------------------------
# analyze_benchmark_filter_frequency
# ---------------------------------------------------------------------------

class TestAnalyzeBenchmarkFilterFrequency:
    def test_basic(self, selected_df):
        from scripts.common.wf_report_shared import analyze_benchmark_filter_frequency
        detail = pd.DataFrame({
            "test_annual_volatility": [0.10, 0.12, 0.15],
            "test_max_drawdown": [-0.05, -0.10, -0.08],
        })
        groups = {"selected": selected_df, "detail": detail}
        freq, stats = analyze_benchmark_filter_frequency(groups)
        assert not freq.empty
        assert "benchmark" in freq.columns
        assert "test_annual_volatility" in stats

    def test_no_benchmark_columns(self):
        from scripts.common.wf_report_shared import analyze_benchmark_filter_frequency
        df = pd.DataFrame({"symbol": ["A", "B"]})
        freq, stats = analyze_benchmark_filter_frequency({"selected": df})
        assert freq.empty

    def test_empty(self):
        from scripts.common.wf_report_shared import analyze_benchmark_filter_frequency
        freq, stats = analyze_benchmark_filter_frequency({"selected": pd.DataFrame()})
        assert freq.empty


# ---------------------------------------------------------------------------
# analyze_single_stock_contribution
# ---------------------------------------------------------------------------

class TestAnalyzeSingleStockContribution:
    def test_basic(self, detail_df):
        from scripts.common.wf_report_shared import analyze_single_stock_contribution
        groups = {"detail": detail_df}
        result = analyze_single_stock_contribution(groups)
        assert not result.empty
        assert "symbol" in result.columns
        assert "win_count" in result.columns
        assert "win_rate" in result.columns

    def test_empty(self):
        from scripts.common.wf_report_shared import analyze_single_stock_contribution
        result = analyze_single_stock_contribution({"detail": pd.DataFrame()})
        assert result.empty

    def test_no_symbol_column(self):
        from scripts.common.wf_report_shared import analyze_single_stock_contribution
        df = pd.DataFrame({"val": [1, 2, 3]})
        result = analyze_single_stock_contribution({"detail": df})
        assert result.empty


# ---------------------------------------------------------------------------
# save_tables
# ---------------------------------------------------------------------------

class TestSaveTables:
    def test_basic(self, tmp_path, v7_cfg):
        from scripts.common.wf_report_shared import save_tables
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        paths = save_tables(tmp_path, v7_cfg, table1=df1, table2=df2)
        assert len(paths) == 2
        assert paths["table1"].exists()
        assert paths["table2"].exists()

    def test_empty_df_skipped(self, tmp_path, v7_cfg):
        from scripts.common.wf_report_shared import save_tables
        paths = save_tables(tmp_path, v7_cfg, empty=pd.DataFrame())
        assert len(paths) == 0

    def test_none_skipped(self, tmp_path, v7_cfg):
        from scripts.common.wf_report_shared import save_tables
        paths = save_tables(tmp_path, v7_cfg, none=None)
        assert len(paths) == 0


# ---------------------------------------------------------------------------
# save_diagnosis_outputs
# ---------------------------------------------------------------------------

class TestSaveDiagnosisOutputs:
    def test_basic(self, tmp_path, v7_cfg):
        from scripts.common.wf_report_shared import save_diagnosis_outputs
        tables = {
            "summary": pd.DataFrame({"val": [1]}),
            "gap": pd.DataFrame({"val": [2]}),
        }
        paths = save_diagnosis_outputs(tmp_path, tables, v7_cfg)
        assert len(paths) == 2
        for p in paths.values():
            assert p.exists()
            assert p.name.startswith("alpha_v7_diagnosis_")

    def test_empty_skipped(self, tmp_path, v7_cfg):
        from scripts.common.wf_report_shared import save_diagnosis_outputs
        paths = save_diagnosis_outputs(tmp_path, {"empty": pd.DataFrame()}, v7_cfg)
        assert len(paths) == 0


# ---------------------------------------------------------------------------
# make_v6_config
# ---------------------------------------------------------------------------

class TestMakeV6Config:
    def test_returns_config(self):
        from scripts.common.wf_report_shared import make_v6_config, WFReportConfig
        cfg = make_v6_config(Path("/project"))
        assert isinstance(cfg, WFReportConfig)
        assert cfg.file_prefix == "wf_alpha_v6_stock_"
        assert cfg.display_name == "Alpha v6"
        assert "momentum_window" in cfg.param_cols
        assert cfg.default_input_dir == Path("/project/backtests/walk_forward_alpha_v6_research_csv")

    def test_differs_from_v7(self):
        from scripts.common.wf_report_shared import make_v6_config, make_v7_config
        v6 = make_v6_config(Path("/p"))
        v7 = make_v7_config(Path("/p"))
        assert v6.file_prefix != v7.file_prefix
        assert v6.param_cols != v7.param_cols


# ---------------------------------------------------------------------------
# Plotting functions (basic smoke tests with real data)
# ---------------------------------------------------------------------------

class TestPlotEquityCurve:
    def test_creates_png(self, tmp_path, combined_df, v7_cfg):
        from scripts.common.wf_report_shared import plot_equity_curve
        path = plot_equity_curve(combined_df, tmp_path, v7_cfg)
        assert path.exists()
        assert path.suffix == ".png"


class TestPlotDrawdownCurve:
    def test_creates_png(self, tmp_path, combined_df, v7_cfg):
        from scripts.common.wf_report_shared import plot_drawdown_curve
        path = plot_drawdown_curve(combined_df, tmp_path, v7_cfg)
        assert path.exists()
        assert path.suffix == ".png"


class TestPlotYearlyReturnBar:
    def test_creates_png(self, tmp_path, yearly_df, v7_cfg):
        from scripts.common.wf_report_shared import plot_yearly_return_bar
        path = plot_yearly_return_bar(yearly_df, tmp_path, v7_cfg)
        assert path is not None
        assert path.exists()

    def test_no_strategy_rows(self, tmp_path, v7_cfg):
        from scripts.common.wf_report_shared import plot_yearly_return_bar
        df = pd.DataFrame({"entity": ["BENCH"], "year": [2021], "total_return": [0.1]})
        path = plot_yearly_return_bar(df, tmp_path, v7_cfg)
        assert path is None


class TestSavePlots:
    def test_basic(self, tmp_path, combined_df, yearly_df, v7_cfg):
        from scripts.common.wf_report_shared import save_plots
        empty = pd.DataFrame()
        paths = save_plots(tmp_path, combined_df, yearly_df, empty, empty, empty, v7_cfg)
        assert "equity_curve" in paths
        assert "drawdown_curve" in paths


# ---------------------------------------------------------------------------
# Diagnosis plotting smoke tests
# ---------------------------------------------------------------------------

class TestMakeTrainVsTestScatter:
    def test_creates_png(self, tmp_path, detail_df, v7_cfg):
        from scripts.common.wf_report_shared import make_train_vs_test_scatter
        path = make_train_vs_test_scatter(detail_df, tmp_path, v7_cfg)
        assert path is not None
        assert path.exists()

    def test_empty_returns_none(self, tmp_path, v7_cfg):
        from scripts.common.wf_report_shared import make_train_vs_test_scatter
        path = make_train_vs_test_scatter(pd.DataFrame(), tmp_path, v7_cfg)
        assert path is None


class TestMakeYearlyExcessHeatmap:
    def test_creates_png(self, tmp_path, v7_cfg):
        from scripts.common.wf_report_shared import make_yearly_excess_heatmap
        df = pd.DataFrame({"year": [2021, 2022], "total_return": [0.1, -0.05]})
        path = make_yearly_excess_heatmap(df, tmp_path, v7_cfg)
        assert path is not None
        assert path.exists()

    def test_empty_returns_none(self, tmp_path, v7_cfg):
        from scripts.common.wf_report_shared import make_yearly_excess_heatmap
        path = make_yearly_excess_heatmap(pd.DataFrame(), tmp_path, v7_cfg)
        assert path is None


class TestMakeTopDraggersChart:
    def test_creates_png(self, tmp_path, v7_cfg):
        from scripts.common.wf_report_shared import make_top_draggers_chart
        df = pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "sum_return": [-0.10, -0.05, -0.02],
        })
        path = make_top_draggers_chart(df, tmp_path, 3, v7_cfg)
        assert path is not None
        assert path.exists()

    def test_empty_returns_none(self, tmp_path, v7_cfg):
        from scripts.common.wf_report_shared import make_top_draggers_chart
        path = make_top_draggers_chart(pd.DataFrame(), tmp_path, 3, v7_cfg)
        assert path is None


# ---------------------------------------------------------------------------
# write_analysis_report
# ---------------------------------------------------------------------------

class TestWriteAnalysisReport:
    def test_creates_report(self, tmp_path, combined_df, yearly_df, v7_cfg):
        from scripts.common.wf_report_shared import (
            write_analysis_report, build_overall_comparison, build_excess_comparison,
            build_yearly_comparison, build_yearly_excess,
        )
        overall = build_overall_comparison(combined_df, None)
        excess = build_excess_comparison(overall)
        yearly = build_yearly_comparison(combined_df, None)
        yearly_excess = build_yearly_excess(yearly)

        args = MagicMock()
        args.input_dir = str(tmp_path)
        args.output_dir = str(tmp_path)
        args.markets = ["ALL"]
        args.portfolio_size = 20
        args.benchmarks = ["000300.SH"]
        args.incomplete_year = 0

        path = write_analysis_report(
            tmp_path, args, {}, ["000300.SH"], None,
            overall, excess, yearly, yearly_excess,
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), {}, pd.DataFrame(),
            {}, {}, v7_cfg,
        )
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Alpha v7" in content
        assert "Walk-Forward Analysis Report" in content


# ---------------------------------------------------------------------------
# write_recommendations
# ---------------------------------------------------------------------------

class TestWriteRecommendations:
    def test_creates_report(self, tmp_path, detail_df, v7_cfg):
        from scripts.common.wf_report_shared import (
            write_recommendations, build_summary, build_overall_comparison,
            build_excess_comparison, build_train_test_gap, build_train_test_correlation,
            build_contributors, build_alpha_variant_stability, build_parameter_stability,
            build_selected_repetition, build_yearly_weakness, build_yearly_excess,
        )
        # Build minimal inputs
        overall = pd.DataFrame([{
            "entity": "strategy", "period": "all_years",
            "total_return": 0.10, "annual_return": 0.10,
            "max_drawdown": -0.15, "sharpe": 0.8, "annual_volatility": 0.12,
        }])
        excess = pd.DataFrame([{
            "period": "all_years", "benchmark": "BENCH_000300.SH_CSI300",
            "excess_return": 0.05, "strategy_return": 0.10, "benchmark_return": 0.05,
            "strategy_annual": 0.10, "benchmark_annual": 0.05, "excess_annual": 0.05,
            "strategy_sharpe": 0.8, "benchmark_sharpe": 0.5,
            "strategy_max_drawdown": -0.15, "benchmark_max_drawdown": -0.20,
        }])
        summary = build_summary(overall, excess, None)
        gap_df = build_train_test_gap(detail_df, None)
        big_detail = pd.concat([detail_df] * 3, ignore_index=True)
        gap_corr = build_train_test_correlation(big_detail)
        bad, good = build_contributors(detail_df, None)
        av_stab = build_alpha_variant_stability(detail_df, None)
        param_stab = build_parameter_stability(detail_df, None, v7_cfg)

        selected_freq = detail_df.groupby("symbol").size().reset_index(name="selected_count")
        selected_rep = build_selected_repetition(selected_freq, detail_df, None)

        yearly = pd.DataFrame([
            {"entity": "strategy", "year": 2021, "total_return": 0.15, "max_drawdown": -0.10, "sharpe": 1.2},
        ])
        yearly_excess = pd.DataFrame([{
            "year": 2021, "benchmark": "BENCH", "excess_return": 0.05,
            "benchmark_return": 0.10, "strategy_return": 0.15, "beat_benchmark": True,
        }])
        yearly_weakness = build_yearly_weakness(yearly, yearly_excess, None)

        args = MagicMock()
        args.analysis_dir = str(tmp_path)
        args.walk_forward_dir = str(tmp_path)
        args.run_id = "test_run"

        path = write_recommendations(
            tmp_path, args, summary, yearly_weakness, gap_df, gap_corr,
            bad, good, av_stab, param_stab, selected_rep, {}, {}, None, v7_cfg,
        )
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Alpha v7" in content
        assert "Diagnosis Recommendations" in content
        assert "Decision:" in content


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_benchmark_names(self):
        from scripts.common.wf_report_shared import BENCHMARK_NAMES
        assert "000300.SH" in BENCHMARK_NAMES
        assert BENCHMARK_NAMES["000300.SH"] == "CSI300"

    def test_overfitting_blocker_rate(self):
        from scripts.common.wf_report_shared import OVERFITTING_BLOCKER_RATE
        assert OVERFITTING_BLOCKER_RATE == 0.20

    def test_bad_to_good_ratio(self):
        from scripts.common.wf_report_shared import BAD_TO_GOOD_CONTRIBUTOR_BLOCKER_RATIO
        assert BAD_TO_GOOD_CONTRIBUTOR_BLOCKER_RATIO == 2.0


# ---------------------------------------------------------------------------
# build_alpha_variant_stability
# ---------------------------------------------------------------------------

class TestBuildAlphaVariantStability:
    """直接测试 build_alpha_variant_stability 函数。"""

    def test_basic(self):
        from scripts.common.wf_report_shared import build_alpha_variant_stability
        detail = pd.DataFrame({
            "alpha_variant": ["short_term_reversal", "short_term_reversal", "low_volatility", "low_volatility", "low_volatility"],
            "symbol": ["A", "B", "C", "D", "E"],
            "test_total_return": [0.10, 0.05, -0.02, 0.08, 0.12],
        })
        result = build_alpha_variant_stability(detail, exclude_year=None)
        assert len(result) == 2
        assert "alpha_variant" in result.columns
        assert "selected_count" in result.columns
        assert "avg_test_return" in result.columns
        assert "win_count" in result.columns
        assert "win_rate" in result.columns
        assert "stability_label" in result.columns

    def test_sorted_by_avg_return_desc(self):
        from scripts.common.wf_report_shared import build_alpha_variant_stability
        detail = pd.DataFrame({
            "alpha_variant": ["A", "A", "B", "B"],
            "symbol": ["s1", "s2", "s3", "s4"],
            "test_total_return": [0.10, 0.20, 0.05, 0.03],
        })
        result = build_alpha_variant_stability(detail, exclude_year=None)
        assert result.iloc[0]["alpha_variant"] == "A"  # avg 0.15 > 0.04

    def test_stability_label_logic(self):
        from scripts.common.wf_report_shared import build_alpha_variant_stability
        # A: count>=3, win_rate>=0.5, avg>0 → relatively_stable
        # B: count<3 → unstable_or_weak
        detail = pd.DataFrame({
            "alpha_variant": ["A", "A", "A", "B"],
            "symbol": ["s1", "s2", "s3", "s4"],
            "test_total_return": [0.10, 0.05, 0.02, -0.10],
        })
        result = build_alpha_variant_stability(detail, exclude_year=None)
        labels = dict(zip(result["alpha_variant"], result["stability_label"]))
        assert labels["A"] == "relatively_stable"
        assert labels["B"] == "unstable_or_weak"

    def test_exclude_year(self):
        from scripts.common.wf_report_shared import build_alpha_variant_stability
        detail = pd.DataFrame({
            "alpha_variant": ["A", "A", "A", "A"],
            "symbol": ["s1", "s2", "s3", "s4"],
            "test_total_return": [0.10, 0.05, -0.02, 0.08],
            "year": [2021, 2022, 2023, 2024],
        })
        result = build_alpha_variant_stability(detail, exclude_year=2024)
        assert result.iloc[0]["selected_count"] == 3

    def test_empty_detail(self):
        from scripts.common.wf_report_shared import build_alpha_variant_stability
        result = build_alpha_variant_stability(pd.DataFrame(), exclude_year=None)
        assert result.empty

    def test_no_alpha_variant_column(self):
        from scripts.common.wf_report_shared import build_alpha_variant_stability
        detail = pd.DataFrame({"symbol": ["A"], "test_total_return": [0.1]})
        result = build_alpha_variant_stability(detail, exclude_year=None)
        assert result.empty

    def test_no_test_total_return_column(self):
        from scripts.common.wf_report_shared import build_alpha_variant_stability
        detail = pd.DataFrame({
            "alpha_variant": ["A", "B"],
            "symbol": ["s1", "s2"],
        })
        result = build_alpha_variant_stability(detail, exclude_year=None)
        assert len(result) == 2
        assert "selected_count" in result.columns
        # avg_test_return falls back to count, win_count/win_rate absent
        assert "win_count" not in result.columns

    def test_all_negative_returns(self):
        from scripts.common.wf_report_shared import build_alpha_variant_stability
        detail = pd.DataFrame({
            "alpha_variant": ["A", "A", "A"],
            "symbol": ["s1", "s2", "s3"],
            "test_total_return": [-0.10, -0.05, -0.02],
        })
        result = build_alpha_variant_stability(detail, exclude_year=None)
        assert result.iloc[0]["stability_label"] == "unstable_or_weak"


# ---------------------------------------------------------------------------
# symbol_to_qmt_csv / load_benchmark_returns / load_walk_forward_group / load_walk_forward_raw
# ---------------------------------------------------------------------------

class TestSymbolToQmtCsv:
    """symbol_to_qmt_csv 委托给 ma.find_csv_for_stock。"""

    def test_returns_csv_path(self, tmp_path):
        from unittest.mock import patch
        from scripts.common.wf_report_shared import symbol_to_qmt_csv
        expected = tmp_path / "000001.SZ.csv"
        with patch("strategies.ma_demo_strategy_csv.find_csv_for_stock") as mock_fn:
            mock_fn.return_value = (expected, "SZ", "stock", "000001")
            result = symbol_to_qmt_csv("000001.SZ", tmp_path)
        assert result == expected

    def test_delegates_symbol_and_root(self, tmp_path):
        from unittest.mock import patch
        from scripts.common.wf_report_shared import symbol_to_qmt_csv
        with patch("strategies.ma_demo_strategy_csv.find_csv_for_stock") as mock_fn:
            mock_fn.return_value = (tmp_path / "x.csv", "", "", "")
            symbol_to_qmt_csv("600000.SH", tmp_path)
            mock_fn.assert_called_once_with("600000.SH", tmp_path)


class TestLoadBenchmarkReturns:
    """load_benchmark_returns 加载基准收益率。"""

    def _make_mock_ma(self, dates, close_prices):
        """构造 mock 函数，返回指定日期和收盘价。"""
        from unittest.mock import MagicMock
        csv_df = pd.DataFrame({"date": dates, "close": close_prices})
        return MagicMock(
            find_csv_for_stock=MagicMock(return_value=(Path("/fake.csv"), "", "", "")),
            load_qmt_price_csv=MagicMock(return_value=csv_df),
        )

    def test_single_benchmark(self, tmp_path):
        from unittest.mock import patch
        from scripts.common.wf_report_shared import load_benchmark_returns
        dates = pd.date_range("2021-01-01", periods=5, freq="B")
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        mock_ma = self._make_mock_ma(dates, prices)
        all_dates = pd.date_range("2021-01-01", periods=5, freq="B")
        with patch("strategies.ma_demo_strategy_csv.find_csv_for_stock", mock_ma.find_csv_for_stock), \
             patch("strategies.ma_demo_strategy_csv.load_qmt_price_csv", mock_ma.load_qmt_price_csv):
            result = load_benchmark_returns(["000300.SH"], tmp_path, all_dates)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        # 列名格式: BENCH_{symbol}_{name}
        assert "BENCH_000300.SH_CSI300" in result.columns

    def test_multiple_benchmarks(self, tmp_path):
        from unittest.mock import patch
        from scripts.common.wf_report_shared import load_benchmark_returns
        dates = pd.date_range("2021-01-01", periods=3, freq="B")
        prices = [100.0, 101.0, 102.0]
        mock_ma = self._make_mock_ma(dates, prices)
        all_dates = pd.date_range("2021-01-01", periods=3, freq="B")
        with patch("strategies.ma_demo_strategy_csv.find_csv_for_stock", mock_ma.find_csv_for_stock), \
             patch("strategies.ma_demo_strategy_csv.load_qmt_price_csv", mock_ma.load_qmt_price_csv):
            result = load_benchmark_returns(
                ["000300.SH", "000905.SH"], tmp_path, all_dates
            )
        assert "BENCH_000300.SH_CSI300" in result.columns
        assert "BENCH_000905.SH_CSI500" in result.columns

    def test_unknown_benchmark_uses_symbol_as_name(self, tmp_path):
        from unittest.mock import patch
        from scripts.common.wf_report_shared import load_benchmark_returns
        dates = pd.date_range("2021-01-01", periods=3, freq="B")
        prices = [100.0, 101.0, 102.0]
        mock_ma = self._make_mock_ma(dates, prices)
        all_dates = pd.date_range("2021-01-01", periods=3, freq="B")
        with patch("strategies.ma_demo_strategy_csv.find_csv_for_stock", mock_ma.find_csv_for_stock), \
             patch("strategies.ma_demo_strategy_csv.load_qmt_price_csv", mock_ma.load_qmt_price_csv):
            result = load_benchmark_returns(["UNKNOWN.XX"], tmp_path, all_dates)
        # 未知基准: BENCHMARK_NAMES.get(bm, bm) 返回 bm 本身
        assert "BENCH_UNKNOWN.XX_UNKNOWN.XX" in result.columns

    def test_failed_benchmark_skipped(self, tmp_path):
        from unittest.mock import patch
        from scripts.common.wf_report_shared import load_benchmark_returns
        all_dates = pd.date_range("2021-01-01", periods=3, freq="B")
        with patch("strategies.ma_demo_strategy_csv.find_csv_for_stock",
                    side_effect=FileNotFoundError("not found")):
            result = load_benchmark_returns(["000300.SH"], tmp_path, all_dates)
        # 加载失败的基准被跳过，返回空 DataFrame
        assert result.empty

    def test_ffill_missing_dates(self, tmp_path):
        from unittest.mock import patch
        from scripts.common.wf_report_shared import load_benchmark_returns
        # 基准数据只有 3 天，但 all_dates 有 5 天
        dates = pd.date_range("2021-01-01", periods=3, freq="B")
        prices = [100.0, 101.0, 102.0]
        mock_ma = self._make_mock_ma(dates, prices)
        all_dates = pd.date_range("2021-01-01", periods=5, freq="B")
        with patch("strategies.ma_demo_strategy_csv.find_csv_for_stock", mock_ma.find_csv_for_stock), \
             patch("strategies.ma_demo_strategy_csv.load_qmt_price_csv", mock_ma.load_qmt_price_csv):
            result = load_benchmark_returns(["000300.SH"], tmp_path, all_dates)
        assert len(result) == 5
        col = "BENCH_000300.SH_CSI300"
        # 第 4、5 天应被 ffill
        assert not result[col].iloc[:3].isna().all()
        assert result[col].iloc[3] == result[col].iloc[2]


class TestLoadWalkForwardGroup:
    """load_walk_forward_group 加载 walk-forward 输出文件组。"""

    def _create_wf_files(self, tmp_path, cfg, file_tag, market="ALL", portfolio_size=20):
        """在 tmp_path 中创建模拟 walk-forward 文件。"""
        prefix = cfg.file_prefix
        suffixes = {
            "portfolio_daily": f"{prefix}{file_tag}_portfolio_daily.csv",
            "portfolio_period_summary": f"{prefix}{file_tag}_portfolio_period_summary.csv",
            "selected_by_year": f"{prefix}{file_tag}_selected_by_year.csv",
            "test_detail": f"{prefix}{file_tag}_test_detail.csv",
        }
        created = {}
        for kind, fname in suffixes.items():
            fpath = tmp_path / fname
            if kind == "portfolio_daily":
                pd.DataFrame({
                    "date": pd.date_range("2021-01-01", periods=3, freq="B"),
                    "strategy_ret": [0.01, -0.005, 0.003],
                }).to_csv(fpath, index=False)
            elif kind == "portfolio_period_summary":
                pd.DataFrame({
                    "year": [2021],
                    "total_return": [0.05],
                }).to_csv(fpath, index=False)
            elif kind == "selected_by_year":
                pd.DataFrame({
                    "year": [2021],
                    "symbol": ["000001.SZ"],
                }).to_csv(fpath, index=False)
            elif kind == "test_detail":
                pd.DataFrame({
                    "symbol": ["000001.SZ"],
                    "test_total_return": [0.08],
                }).to_csv(fpath, index=False)
            created[kind] = fpath
        return created

    def test_loads_all_four_kinds(self, v7_cfg):
        from scripts.common.wf_report_shared import load_walk_forward_group
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._create_wf_files(tmp_path, v7_cfg, "ALL_top20")
            result = load_walk_forward_group(
                tmp_path, "ALL", 20, v7_cfg, file_tag="ALL_top20"
            )
        assert "daily" in result
        assert "period" in result
        assert "selected" in result
        assert "detail" in result
        assert isinstance(result["daily"], pd.DataFrame)
        assert len(result["daily"]) == 3

    def test_date_column_converted(self, v7_cfg):
        from scripts.common.wf_report_shared import load_walk_forward_group
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._create_wf_files(tmp_path, v7_cfg, "ALL_top20")
            result = load_walk_forward_group(
                tmp_path, "ALL", 20, v7_cfg, file_tag="ALL_top20"
            )
        assert pd.api.types.is_datetime64_any_dtype(result["daily"]["date"])

    def test_missing_file_raises(self, v7_cfg):
        from scripts.common.wf_report_shared import load_walk_forward_group
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # 不创建任何文件
            with pytest.raises(FileNotFoundError):
                load_walk_forward_group(
                    tmp_path, "ALL", 20, v7_cfg, file_tag="nonexistent"
                )

    def test_group_metadata(self, v7_cfg):
        from scripts.common.wf_report_shared import load_walk_forward_group
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._create_wf_files(tmp_path, v7_cfg, "ALL_top20")
            result = load_walk_forward_group(
                tmp_path, "ALL", 20, v7_cfg, file_tag="ALL_top20"
            )
        assert result["market"] == "ALL"


class TestLoadWalkForwardRaw:
    """load_walk_forward_raw 加载 selected 和 detail 数据。"""

    def _create_raw_files(self, tmp_path, cfg, file_tag):
        prefix = cfg.file_prefix
        selected_path = tmp_path / f"{prefix}{file_tag}_selected_by_year.csv"
        detail_path = tmp_path / f"{prefix}{file_tag}_test_detail.csv"
        pd.DataFrame({
            "year": [2021, 2022],
            "symbol": ["000001.SZ", "600000.SH"],
        }).to_csv(selected_path, index=False)
        pd.DataFrame({
            "symbol": ["000001.SZ", "600000.SH"],
            "train_annual_return": [0.10, 0.05],
            "test_total_return": [0.08, 0.03],
        }).to_csv(detail_path, index=False)
        return selected_path, detail_path

    def test_returns_two_dataframes(self, v7_cfg):
        from scripts.common.wf_report_shared import load_walk_forward_raw
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._create_raw_files(tmp_path, v7_cfg, "ALL_top20")
            selected, detail = load_walk_forward_raw(
                tmp_path, ["ALL"], 20, v7_cfg, file_tag="ALL_top20"
            )
        assert isinstance(selected, pd.DataFrame)
        assert isinstance(detail, pd.DataFrame)
        assert len(selected) == 2
        assert len(detail) == 2

    def test_numeric_columns_converted(self, v7_cfg):
        from scripts.common.wf_report_shared import load_walk_forward_raw
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._create_raw_files(tmp_path, v7_cfg, "ALL_top20")
            _, detail = load_walk_forward_raw(
                tmp_path, ["ALL"], 20, v7_cfg, file_tag="ALL_top20"
            )
        assert pd.api.types.is_numeric_dtype(detail["train_annual_return"])
        assert pd.api.types.is_numeric_dtype(detail["test_total_return"])

    def test_single_market_uses_market_as_prefix(self, v7_cfg):
        from scripts.common.wf_report_shared import load_walk_forward_raw
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # 当 markets 只有 1 个时，用 market 名而非 "ALL"
            prefix = v7_cfg.file_prefix
            for kind in ["selected_by_year", "test_detail"]:
                fpath = tmp_path / f"{prefix}SH_top20_{kind}.csv"
                pd.DataFrame({"x": [1]}).to_csv(fpath, index=False)
            selected, detail = load_walk_forward_raw(
                tmp_path, ["SH"], 20, v7_cfg, file_tag="SH_top20"
            )
        assert len(selected) == 1
        assert len(detail) == 1

    def test_missing_file_raises(self, v7_cfg):
        from scripts.common.wf_report_shared import load_walk_forward_raw
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with pytest.raises(FileNotFoundError):
                load_walk_forward_raw(
                    tmp_path, ["ALL"], 20, v7_cfg, file_tag="nonexistent"
                )
