# -*- coding: utf-8 -*-
"""
tests/test_factor_data.py

覆盖 scripts/common/factor_data.py 的测试：
- load_board_membership / load_boards
- get_stock_boards（PIT-safe 日期过滤、去重）
- get_board_stocks（板块类型过滤、最新日期取值）
- get_board_coverage_stats
- 缺失文件守卫

使用 tmp_path 创建临时 parquet 文件，不依赖真实数据。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _patch_factors_dir(tmp_path):
    """将 FACTORS_DIR / PARQUET_DIR / BOARDS_FILE / MEMBERS_FILE 指向临时目录。"""
    parquet_dir = tmp_path / "factors" / "concept_industry" / "parquet"
    parquet_dir.mkdir(parents=True)
    boards_file = parquet_dir / "ths_boards.parquet"
    members_file = parquet_dir / "ths_board_members.parquet"

    boards_df = pd.DataFrame({
        "board_code": ["300008", "885001", "BK0001"],
        "board_name": ["锂电池", "半导体", "银行"],
        "board_type": ["concept", "concept", "industry"],
    })
    boards_df.to_parquet(boards_file, index=False)

    members_df = pd.DataFrame({
        "board_code": ["300008", "300008", "885001", "885001", "BK0001"],
        "board_name": ["锂电池", "锂电池", "半导体", "半导体", "银行"],
        "board_type": ["concept", "concept", "concept", "concept", "industry"],
        "stock_code": ["000001.SZ", "000002.SZ", "000001.SZ", "600000.SH", "601398.SH"],
        "stock_name": ["平安银行", "万科A", "平安银行", "浦发银行", "工商银行"],
        "fetch_date": ["2026-05-01", "2026-05-01", "2026-05-15", "2026-05-15", "2026-05-10"],
        "fetch_time": ["10:00", "10:00", "11:00", "11:00", "09:30"],
        "source": ["ths", "ths", "ths", "ths", "ths"],
    })
    members_df.to_parquet(members_file, index=False)

    with patch("scripts.common.factor_data.BOARDS_FILE", boards_file), \
         patch("scripts.common.factor_data.MEMBERS_FILE", members_file):
        yield {
            "boards_file": boards_file,
            "members_file": members_file,
            "boards_df": boards_df,
            "members_df": members_df,
        }


# ---------------------------------------------------------------------------
# load_board_membership / load_boards
# ---------------------------------------------------------------------------

class TestLoadBoardMembership:
    """load_board_membership 的测试。"""

    def test_returns_dataframe(self, _patch_factors_dir):
        from scripts.common.factor_data import load_board_membership
        df = load_board_membership()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_columns_present(self, _patch_factors_dir):
        from scripts.common.factor_data import load_board_membership
        df = load_board_membership()
        for col in ["board_code", "board_name", "board_type", "stock_code",
                     "stock_name", "fetch_date", "fetch_time", "source"]:
            assert col in df.columns

    def test_missing_file_returns_empty(self, tmp_path):
        from scripts.common.factor_data import load_board_membership
        missing = tmp_path / "nonexistent.parquet"
        with patch("scripts.common.factor_data.MEMBERS_FILE", missing):
            df = load_board_membership()
        assert df.empty


class TestLoadBoards:
    """load_boards 的测试。"""

    def test_returns_dataframe(self, _patch_factors_dir):
        from scripts.common.factor_data import load_boards
        df = load_boards()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_missing_file_returns_empty(self, tmp_path):
        from scripts.common.factor_data import load_boards
        missing = tmp_path / "nonexistent.parquet"
        with patch("scripts.common.factor_data.BOARDS_FILE", missing):
            df = load_boards()
        assert df.empty


# ---------------------------------------------------------------------------
# get_stock_boards
# ---------------------------------------------------------------------------

class TestGetStockBoards:
    """get_stock_boards 的测试。"""

    def test_basic_lookup(self, _patch_factors_dir):
        from scripts.common.factor_data import get_stock_boards
        result = get_stock_boards("000001.SZ")
        assert len(result) == 2  # 锂电池 + 半导体
        board_codes = set(result["board_code"])
        assert "300008" in board_codes
        assert "885001" in board_codes

    def test_pit_safe_date_filter(self, _patch_factors_dir):
        """query_date 应只保留 fetch_date <= query_date 的记录。"""
        from scripts.common.factor_data import get_stock_boards
        # 000001.SZ 在 300008 fetch_date=2026-05-01 和 885001 fetch_date=2026-05-15
        result = get_stock_boards("000001.SZ", query_date="2026-05-10")
        assert len(result) == 1
        assert result.iloc[0]["board_code"] == "300008"

    def test_pit_safe_date_filter_all(self, _patch_factors_dir):
        """query_date 早于所有 fetch_date 时返回空。"""
        from scripts.common.factor_data import get_stock_boards
        result = get_stock_boards("000001.SZ", query_date="2026-04-01")
        assert len(result) == 0

    def test_unknown_stock_returns_empty(self, _patch_factors_dir):
        from scripts.common.factor_data import get_stock_boards
        result = get_stock_boards("999999.SZ")
        assert len(result) == 0

    def test_empty_members_returns_empty(self, tmp_path):
        from scripts.common.factor_data import get_stock_boards
        empty_file = tmp_path / "empty.parquet"
        # 需要保留列 schema，否则 load_board_membership 的 logger 调用会 KeyError
        pd.DataFrame(columns=[
            "board_code", "board_name", "board_type",
            "stock_code", "stock_name", "fetch_date", "fetch_time", "source",
        ]).to_parquet(empty_file, index=False)
        with patch("scripts.common.factor_data.MEMBERS_FILE", empty_file):
            result = get_stock_boards("000001.SZ")
        assert result.empty

    def test_preloaded_members_df(self, _patch_factors_dir):
        """传入 members_df 时不应读文件。"""
        from scripts.common.factor_data import get_stock_boards, load_board_membership
        members_df = load_board_membership()
        result = get_stock_boards("000001.SZ", members_df=members_df)
        assert len(result) == 2

    def test_dedup_by_board_code(self, _patch_factors_dir):
        """同一板块多次 fetch 时只保留最新 fetch_date。"""
        from scripts.common.factor_data import get_stock_boards
        # 000001.SZ 在 300008 只有一条记录，应直接返回
        result = get_stock_boards("000001.SZ")
        # 每个 board_code 只出现一次
        assert len(result) == result["board_code"].nunique()


# ---------------------------------------------------------------------------
# get_board_stocks
# ---------------------------------------------------------------------------

class TestGetBoardStocks:
    """get_board_stocks 的测试。"""

    def test_basic_lookup(self, _patch_factors_dir):
        from scripts.common.factor_data import get_board_stocks
        result = get_board_stocks("300008")
        assert len(result) == 2
        stock_codes = set(result["stock_code"])
        assert "000001.SZ" in stock_codes
        assert "000002.SZ" in stock_codes

    def test_board_type_filter(self, _patch_factors_dir):
        from scripts.common.factor_data import get_board_stocks
        result = get_board_stocks("300008", board_type="concept")
        assert len(result) == 2
        result_ind = get_board_stocks("300008", board_type="industry")
        assert len(result_ind) == 0

    def test_pit_safe_date_filter(self, _patch_factors_dir):
        """query_date 应只保留 fetch_date <= query_date 的记录。"""
        from scripts.common.factor_data import get_board_stocks
        # 300008 fetch_date=2026-05-01
        result = get_board_stocks("300008", query_date="2026-04-01")
        assert len(result) == 0

    def test_unknown_board_returns_empty(self, _patch_factors_dir):
        from scripts.common.factor_data import get_board_stocks
        result = get_board_stocks("999999")
        assert len(result) == 0

    def test_empty_members_returns_empty(self, tmp_path):
        from scripts.common.factor_data import get_board_stocks
        empty_file = tmp_path / "empty.parquet"
        pd.DataFrame(columns=[
            "board_code", "board_name", "board_type",
            "stock_code", "stock_name", "fetch_date", "fetch_time", "source",
        ]).to_parquet(empty_file, index=False)
        with patch("scripts.common.factor_data.MEMBERS_FILE", empty_file):
            result = get_board_stocks("300008")
        assert result.empty

    def test_latest_date_per_board(self, _patch_factors_dir):
        """同一板块多日期时只取最新 fetch_date。"""
        from scripts.common.factor_data import get_board_stocks
        # 885001 的 fetch_date 都是 2026-05-15
        result = get_board_stocks("885001", query_date="2026-06-01")
        assert len(result) == 2
        assert (result["fetch_date"] == "2026-05-15").all()

    def test_preloaded_members_df(self, _patch_factors_dir):
        from scripts.common.factor_data import get_board_stocks, load_board_membership
        members_df = load_board_membership()
        result = get_board_stocks("300008", members_df=members_df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# get_board_coverage_stats
# ---------------------------------------------------------------------------

class TestGetBoardCoverageStats:
    """get_board_coverage_stats 的测试。"""

    def test_returns_dict(self, _patch_factors_dir):
        from scripts.common.factor_data import get_board_coverage_stats
        stats = get_board_coverage_stats()
        assert isinstance(stats, dict)

    def test_keys_present(self, _patch_factors_dir):
        from scripts.common.factor_data import get_board_coverage_stats
        stats = get_board_coverage_stats()
        for key in ["total_boards", "covered_boards", "total_stocks",
                     "coverage_rate", "total_member_rows"]:
            assert key in stats

    def test_values_reasonable(self, _patch_factors_dir):
        from scripts.common.factor_data import get_board_coverage_stats
        stats = get_board_coverage_stats()
        assert stats["total_boards"] == 3
        assert stats["covered_boards"] == 3
        assert stats["total_stocks"] == 4  # 000001, 000002, 600000, 601398
        assert stats["total_member_rows"] == 5
        assert 0.0 <= stats["coverage_rate"] <= 1.0

    def test_empty_members(self, tmp_path):
        from scripts.common.factor_data import get_board_coverage_stats
        empty_file = tmp_path / "empty.parquet"
        pd.DataFrame(columns=[
            "board_code", "board_name", "board_type",
            "stock_code", "stock_name", "fetch_date", "fetch_time", "source",
        ]).to_parquet(empty_file, index=False)
        boards_file = tmp_path / "boards.parquet"
        pd.DataFrame({"board_code": ["A"], "board_name": ["X"], "board_type": ["concept"]}).to_parquet(boards_file, index=False)
        with patch("scripts.common.factor_data.MEMBERS_FILE", empty_file), \
             patch("scripts.common.factor_data.BOARDS_FILE", boards_file):
            stats = get_board_coverage_stats()
        assert stats["total_boards"] == 1
        assert stats["covered_boards"] == 0
        assert stats["coverage_rate"] == 0.0
        assert stats["total_stocks"] == 0


# ---------------------------------------------------------------------------
# 金融正确性检查
# ---------------------------------------------------------------------------

class TestFinancialCorrectness:
    """PIT-safe 和数据完整性检查。"""

    def test_pit_safe_no_future_leakage(self, _patch_factors_dir):
        """query_date 应严格过滤，不包含未来数据。"""
        from scripts.common.factor_data import get_stock_boards
        # 885001 fetch_date=2026-05-15
        # query_date=2026-05-14 不应包含 885001
        result = get_stock_boards("000001.SZ", query_date="2026-05-14")
        board_codes = set(result["board_code"])
        assert "885001" not in board_codes

    def test_board_stocks_pit_safe(self, _patch_factors_dir):
        """get_board_stocks 的 query_date 也应严格过滤。"""
        from scripts.common.factor_data import get_board_stocks
        # 885001 fetch_date=2026-05-15
        result = get_board_stocks("885001", query_date="2026-05-14")
        assert len(result) == 0

    def test_date_boundary_inclusive(self, _patch_factors_dir):
        """query_date == fetch_date 时应包含该记录。"""
        from scripts.common.factor_data import get_stock_boards
        result = get_stock_boards("000001.SZ", query_date="2026-05-15")
        board_codes = set(result["board_code"])
        assert "885001" in board_codes
