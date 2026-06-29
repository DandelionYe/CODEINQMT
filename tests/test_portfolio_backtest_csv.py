# -*- coding: utf-8 -*-
"""Tests for portfolio_backtest_csv module: infer_file_prefix, load_input_files."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.portfolio_backtest_csv import infer_file_prefix, load_input_files


class TestInferFilePrefix:
    """Tests for infer_file_prefix()."""

    def test_infer_file_prefix_found(self, tmp_path):
        """匹配到文件时应正确提取前缀。"""
        (tmp_path / "wf_alpha_v7_stock_ALL_selected_by_year.csv").touch()
        result = infer_file_prefix(tmp_path, "ALL")
        assert result == "wf_alpha_v7_stock"

    def test_infer_file_prefix_fallback(self, tmp_path):
        """空目录应回退到默认前缀。"""
        result = infer_file_prefix(tmp_path, "ALL")
        assert result == "wf_alpha_v4_stock"

    def test_infer_file_prefix_multiple_matches(self, tmp_path):
        """多个匹配时取最后一个（按文件名排序）。"""
        (tmp_path / "wf_alpha_v4_stock_ALL_selected_by_year.csv").touch()
        (tmp_path / "wf_alpha_v7_stock_ALL_selected_by_year.csv").touch()
        result = infer_file_prefix(tmp_path, "ALL")
        assert result == "wf_alpha_v7_stock"

    def test_infer_file_prefix_partial_suffix_no_match(self, tmp_path):
        """文件后缀不完整时不匹配。"""
        (tmp_path / "wf_alpha_v7_stock_ALL_selected_by_year_extra.csv").touch()
        result = infer_file_prefix(tmp_path, "ALL")
        assert result == "wf_alpha_v4_stock"


class TestLoadInputFilesCustomPrefix:
    """Tests for load_input_files() with custom file_prefix."""

    def test_load_input_files_custom_prefix(self, tmp_path):
        """使用自定义前缀加载文件。"""
        prefix = "wf_alpha_v7_stock"
        tag = "ALL"
        # Create required files
        selected = pd.DataFrame({
            "symbol": ["000001.SZ"], "test_year": [2024], "selected_rank": [1],
        })
        daily = pd.DataFrame({
            "date": ["2024-01-01"], "equity": [1000000], "daily_return": [0.0],
        })
        selected.to_csv(tmp_path / f"{prefix}_{tag}_selected_by_year.csv", index=False)
        daily.to_csv(tmp_path / f"{prefix}_{tag}_portfolio_daily.csv", index=False)

        data = load_input_files(tmp_path, tag, prefix)
        assert "selected_by_year" in data
        assert "portfolio_daily" in data
        assert len(data["selected_by_year"]) == 1

    def test_load_input_files_default_prefix(self, tmp_path):
        """默认前缀 wf_alpha_v4_stock。"""
        prefix = "wf_alpha_v4_stock"
        tag = "ALL"
        selected = pd.DataFrame({
            "symbol": ["000001.SZ"], "test_year": [2024], "selected_rank": [1],
        })
        daily = pd.DataFrame({
            "date": ["2024-01-01"], "equity": [1000000], "daily_return": [0.0],
        })
        selected.to_csv(tmp_path / f"{prefix}_{tag}_selected_by_year.csv", index=False)
        daily.to_csv(tmp_path / f"{prefix}_{tag}_portfolio_daily.csv", index=False)

        data = load_input_files(tmp_path, tag)
        assert "selected_by_year" in data

    def test_load_input_files_missing_required(self, tmp_path):
        """缺少必需文件应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="缺少以下必需文件"):
            load_input_files(tmp_path, "ALL", "wf_alpha_v7_stock")
