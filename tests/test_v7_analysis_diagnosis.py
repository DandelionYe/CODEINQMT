# -*- coding: utf-8 -*-
"""
Tests for Alpha v7 analysis and diagnosis scripts.

Verifies:
- Import correctness
- File glob pattern matching (wf_alpha_v7_stock_*)
- Output prefix consistency (alpha_v7_wf_analysis, alpha_v7_diagnosis)
- Parameter compatibility (--input-dir for diagnosis)
- Data flow: analysis output prefix matches diagnosis input prefix
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np


# --- Import tests ---

class TestV7AnalysisImport:
    def test_import_analysis_module(self):
        """v7 analysis script should be importable."""
        import scripts.analyze_alpha_v7_research_walk_forward_results as mod
        assert hasattr(mod, "main")
        assert hasattr(mod, "CFG")

    def test_config_has_v7_prefix(self):
        import scripts.analyze_alpha_v7_research_walk_forward_results as mod
        assert mod.CFG.file_prefix == "wf_alpha_v7_stock_"

    def test_config_has_v7_output_prefix(self):
        import scripts.analyze_alpha_v7_research_walk_forward_results as mod
        assert mod.CFG.analysis_output_prefix == "alpha_v7_wf_analysis"


class TestV7DiagnosisImport:
    def test_import_diagnosis_module(self):
        """v7 diagnosis script should be importable."""
        import scripts.diagnose_alpha_v7_research_strategy_results as mod
        assert hasattr(mod, "main")
        assert hasattr(mod, "CFG")

    def test_config_has_v7_prefix(self):
        import scripts.diagnose_alpha_v7_research_strategy_results as mod
        assert mod.CFG.file_prefix == "wf_alpha_v7_stock_"

    def test_analysis_prefix_matches_analysis_output(self):
        """Diagnosis must read the same prefix that analysis writes."""
        import scripts.analyze_alpha_v7_research_walk_forward_results as analysis_mod
        import scripts.diagnose_alpha_v7_research_strategy_results as diagnosis_mod
        assert diagnosis_mod.CFG.analysis_output_prefix == analysis_mod.CFG.analysis_output_prefix

    def test_diagnosis_prefix_is_v7(self):
        import scripts.diagnose_alpha_v7_research_strategy_results as mod
        assert mod.CFG.diagnosis_output_prefix == "alpha_v7_diagnosis"


# --- Glob pattern tests ---

class TestV7AnalysisGlobPattern:
    def test_find_one_file_with_tag(self, tmp_path):
        """find_one_file should match wf_alpha_v7_stock_* files."""
        from scripts.common.wf_report_shared import find_one_file, make_v7_config
        from pathlib import Path

        cfg = make_v7_config(Path(__file__).resolve().parents[1])

        # Create a fake v7 walk-forward file
        tag = "alpha_v7_ALL_stock_ts20150101_test"
        fname = f"wf_alpha_v7_stock_{tag}_portfolio_daily.csv"
        (tmp_path / fname).write_text("date,portfolio_ret\n2024-01-01,0.01\n", encoding="utf-8")

        result = find_one_file(tmp_path, "ALL", 20, "portfolio_daily", cfg, file_tag=tag)
        assert result.name == fname

    def test_find_one_file_no_match_raises(self, tmp_path):
        """Should raise FileNotFoundError when no v7 files exist."""
        from scripts.common.wf_report_shared import find_one_file, make_v7_config
        from pathlib import Path

        cfg = make_v7_config(Path(__file__).resolve().parents[1])

        with pytest.raises(FileNotFoundError):
            find_one_file(tmp_path, "ALL", 20, "portfolio_daily", cfg, file_tag="nonexistent")

    def test_list_candidate_files_empty(self, tmp_path):
        """list_candidate_files should return placeholder when no files exist."""
        from scripts.common.wf_report_shared import list_candidate_files, make_v7_config
        from pathlib import Path

        cfg = make_v7_config(Path(__file__).resolve().parents[1])
        result = list_candidate_files(tmp_path, "portfolio_daily", cfg)
        assert "(none)" in result


class TestV7DiagnosisGlobPattern:
    def test_find_wf_file_with_tag(self, tmp_path):
        """find_one_file should match wf_alpha_v7_stock_* files."""
        from scripts.common.wf_report_shared import find_one_file, make_v7_config
        from pathlib import Path

        cfg = make_v7_config(Path(__file__).resolve().parents[1])

        tag = "alpha_v7_ALL_test"
        fname = f"wf_alpha_v7_stock_{tag}_selected_by_year.csv"
        (tmp_path / fname).write_text("symbol,year\n000001.SZ,2024\n", encoding="utf-8")

        result = find_one_file(tmp_path, "ALL", 20, "selected_by_year", cfg, file_tag=tag)
        assert result.name == fname


# --- Analysis table I/O tests ---

class TestV7DiagnosisReadsAnalysisTables:
    def test_load_analysis_tables_reads_v7_prefix(self, tmp_path):
        """load_analysis_tables should find files with alpha_v7_wf_analysis_ prefix."""
        from scripts.common.wf_report_shared import load_analysis_tables, make_v7_config
        from pathlib import Path

        cfg = make_v7_config(Path(__file__).resolve().parents[1])

        # Create fake analysis output files
        (tmp_path / "alpha_v7_wf_analysis_overall_comparison.csv").write_text(
            "entity,period,total_return\nstrategy,all_years,0.15\n", encoding="utf-8"
        )
        (tmp_path / "alpha_v7_wf_analysis_excess_comparison.csv").write_text(
            "period,benchmark,excess_return\nall_years,BENCH_CSI300,0.05\n", encoding="utf-8"
        )

        tables = load_analysis_tables(tmp_path, cfg)
        assert not tables["overall_comparison"].empty
        assert tables["overall_comparison"].iloc[0]["entity"] == "strategy"
        assert not tables["excess_comparison"].empty


# --- Build function tests with synthetic data ---

class TestV7AnalysisBuildFunctions:
    def test_build_overall_comparison(self):
        from scripts.common.wf_report_shared import build_overall_comparison

        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        combined = pd.DataFrame({
            "strategy": np.random.randn(100) * 0.01,
            "BENCH_CSI300": np.random.randn(100) * 0.01,
        }, index=dates)

        result = build_overall_comparison(combined, incomplete_year=None)
        assert not result.empty
        assert "strategy" in result["entity"].values
        assert "total_return" in result.columns
        assert "sharpe" in result.columns

    def test_build_excess_comparison(self):
        from scripts.common.wf_report_shared import build_excess_comparison

        overall = pd.DataFrame([
            {"entity": "strategy", "period": "all_years", "total_return": 0.15, "annual_return": 0.10,
             "sharpe": 0.8, "max_drawdown": -0.15, "annual_volatility": 0.12},
            {"entity": "BENCH_CSI300", "period": "all_years", "total_return": 0.08, "annual_return": 0.06,
             "sharpe": 0.5, "max_drawdown": -0.20, "annual_volatility": 0.15},
        ])
        excess = build_excess_comparison(overall)
        assert not excess.empty
        assert excess.iloc[0]["excess_return"] == pytest.approx(0.07)

    def test_build_yearly_comparison(self):
        from scripts.common.wf_report_shared import build_yearly_comparison

        dates = pd.date_range("2022-01-01", periods=500, freq="B")
        combined = pd.DataFrame({
            "strategy": np.random.randn(500) * 0.01,
        }, index=dates)

        result = build_yearly_comparison(combined, incomplete_year=None)
        assert not result.empty
        assert "year" in result.columns


class TestV7DiagnosisBuildFunctions:
    def test_build_summary(self):
        from scripts.common.wf_report_shared import build_summary

        overall = pd.DataFrame([{
            "entity": "strategy", "period": "all_years",
            "total_return": 0.20, "annual_return": 0.12,
            "max_drawdown": -0.15, "sharpe": 0.9, "annual_volatility": 0.10,
        }])
        excess = pd.DataFrame([{
            "period": "all_years", "benchmark": "BENCH_CSI300",
            "excess_return": 0.08,
        }])

        summary = build_summary(overall, excess, exclude_year=None)
        assert not summary.empty
        assert summary.iloc[0]["total_return"] == pytest.approx(0.20)

    def test_build_train_test_gap(self):
        from scripts.common.wf_report_shared import build_train_test_gap

        detail = pd.DataFrame({
            "symbol": ["000001.SZ", "000002.SZ", "600000.SH"],
            "year": [2023, 2023, 2024],
            "train_annual_return": [0.10, 0.08, -0.02],
            "test_total_return": [-0.05, 0.06, 0.03],
            "train_sharpe": [1.2, 0.8, -0.1],
            "test_sharpe": [-0.3, 0.5, 0.2],
        })

        gap = build_train_test_gap(detail, exclude_year=None)
        assert not gap.empty
        assert "gap_flag" in gap.columns
        assert gap.iloc[0]["gap_flag"] == "train_good_test_bad"

    def test_build_alpha_variant_stability(self):
        from scripts.common.wf_report_shared import build_alpha_variant_stability

        detail = pd.DataFrame({
            "symbol": ["A", "B", "C", "D", "E"],
            "alpha_variant": ["short_term_reversal"] * 3 + ["low_volatility"] * 2,
            "test_total_return": [0.05, 0.03, 0.08, -0.02, -0.01],
            "year": [2023, 2024, 2025, 2023, 2024],
        })

        stab = build_alpha_variant_stability(detail, exclude_year=None)
        assert not stab.empty
        assert "stability_label" in stab.columns

    def test_collect_decision_blockers(self):
        from scripts.common.wf_report_shared import collect_decision_blockers

        # Overfitting scenario: many train_good_test_bad
        gap_df = pd.DataFrame({
            "gap_flag": ["train_good_test_bad"] * 5 + ["train_good_test_ok"] * 3,
        })
        blockers = collect_decision_blockers(gap_df, None, None, None)
        assert any("severe_overfitting" in b for b in blockers)


# --- Argument parsing tests ---

class TestV7DiagnosisArgParsing:
    def test_input_dir_takes_precedence(self, tmp_path):
        """--input-dir should take precedence over --walk-forward-dir."""
        import scripts.diagnose_alpha_v7_research_strategy_results as mod
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--input-dir", default="")
        parser.add_argument("--walk-forward-dir", default=str(mod.CFG.default_wf_dir))

        args = parser.parse_args(["--input-dir", str(tmp_path / "custom")])
        wf_dir = args.input_dir if args.input_dir else args.walk_forward_dir
        assert wf_dir == str(tmp_path / "custom")

    def test_walk_forward_dir_as_fallback(self):
        """Without --input-dir, --walk-forward-dir should be used."""
        import scripts.diagnose_alpha_v7_research_strategy_results as mod
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--input-dir", default="")
        parser.add_argument("--walk-forward-dir", default=str(mod.CFG.default_wf_dir))

        args = parser.parse_args([])
        wf_dir = args.input_dir if args.input_dir else args.walk_forward_dir
        assert wf_dir == str(mod.CFG.default_wf_dir)


# --- Parameter column compatibility ---

class TestV7ParameterColumns:
    def test_analysis_param_frequency_uses_v7_columns(self):
        """v7 analysis should look for v7-specific parameter columns, not v6."""
        import scripts.analyze_alpha_v7_research_walk_forward_results as mod

        selected = pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "reversal_window": [5, 10, 20],
            "vol_window": [60, 60, 120],
            "alpha_variant": ["short_term_reversal", "low_volatility", "turnover_reversal"],
        })
        groups = {"selected": selected}

        from scripts.common.wf_report_shared import analyze_parameter_frequency
        freq = analyze_parameter_frequency(groups, mod.CFG)
        assert not freq.empty
        assert "reversal_window" in freq.columns

    def test_diagnosis_param_stability_uses_v7_columns(self):
        """v7 diagnosis should look for v7-specific parameter columns."""
        import scripts.diagnose_alpha_v7_research_strategy_results as mod

        detail = pd.DataFrame({
            "symbol": ["A", "B", "C", "D"],
            "alpha_variant": ["short_term_reversal"] * 4,
            "reversal_window": [5, 10, 20, 5],
            "vol_window": [60, 60, 120, 60],
            "test_total_return": [0.05, 0.03, -0.01, 0.08],
            "year": [2023, 2024, 2025, 2024],
        })

        from scripts.common.wf_report_shared import build_parameter_stability
        stab = build_parameter_stability(detail, exclude_year=None, cfg=mod.CFG)
        assert not stab.empty
        assert "reversal_window" in stab.columns
