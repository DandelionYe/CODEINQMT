# -*- coding: utf-8 -*-
"""
tests/test_common_utils.py

覆盖 scripts/common/ 下三个共享工具模块的测试：
- constants.py：常量和路径计算
- data_io.py：safe_to_numeric、read_csv_required
- validation.py：parse_date_yyyymmdd、resolve_path、parse_list、parse_int_list、
  parse_workers、safe_symbol_tag
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# =========================================================================
# constants.py
# =========================================================================

class TestConstants:
    """scripts.common.constants 的基本验证。"""

    def test_trading_days_value(self):
        from scripts.common.constants import TRADING_DAYS_PER_YEAR
        assert TRADING_DAYS_PER_YEAR == 252

    def test_sqrt_trading_days(self):
        from scripts.common.constants import SQRT_TRADING_DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR
        assert SQRT_TRADING_DAYS_PER_YEAR == math.sqrt(TRADING_DAYS_PER_YEAR)

    def test_default_benchmark(self):
        from scripts.common.constants import DEFAULT_BENCHMARK
        assert DEFAULT_BENCHMARK == "000300.SH"

    def test_default_benchmark_list(self):
        from scripts.common.constants import DEFAULT_BENCHMARK_LIST
        items = DEFAULT_BENCHMARK_LIST.split(",")
        assert "000300.SH" in items
        assert "000905.SH" in items
        assert "000852.SH" in items

    def test_project_root_is_path(self):
        from scripts.common.constants import PROJECT_ROOT
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()

    def test_project_root_ends_with_codeinqmt(self):
        from scripts.common.constants import PROJECT_ROOT
        assert PROJECT_ROOT.name.startswith("CODEINQMT")

    def test_default_export_root_relative(self):
        from scripts.common.constants import DEFAULT_EXPORT_ROOT, PROJECT_ROOT
        assert DEFAULT_EXPORT_ROOT == PROJECT_ROOT / "data" / "qmt_export"

    def test_default_parquet_root_relative(self):
        from scripts.common.constants import DEFAULT_PARQUET_ROOT, PROJECT_ROOT
        assert DEFAULT_PARQUET_ROOT == PROJECT_ROOT / "data" / "qmt_parquet"


# =========================================================================
# data_io.py
# =========================================================================

class TestSafeToNumeric:
    """scripts.common.data_io.safe_to_numeric 的测试。"""

    def test_converts_numeric_strings(self):
        from scripts.common.data_io import safe_to_numeric
        df = pd.DataFrame({"a": ["1.5", "2.0", "abc"], "b": ["10", "20", "30"]})
        result = safe_to_numeric(df, columns=["a"])
        assert result["a"].dtype.kind == "f"
        assert result["a"].iloc[0] == pytest.approx(1.5)
        assert pd.isna(result["a"].iloc[2])

    def test_converts_all_columns_when_none(self):
        from scripts.common.data_io import safe_to_numeric
        df = pd.DataFrame({"x": ["1", "2"], "y": ["3.0", "4.5"]})
        result = safe_to_numeric(df)
        # integer-like strings become int dtype, float-like become float
        assert result["x"].dtype.kind in ("f", "i", "u")
        assert result["y"].dtype.kind in ("f", "i", "u")

    def test_skips_missing_column(self):
        from scripts.common.data_io import safe_to_numeric
        df = pd.DataFrame({"a": ["1", "2"]})
        result = safe_to_numeric(df, columns=["a", "nonexistent"])
        assert result["a"].iloc[0] == 1.0

    def test_returns_same_dataframe(self):
        from scripts.common.data_io import safe_to_numeric
        df = pd.DataFrame({"a": ["1"]})
        result = safe_to_numeric(df)
        assert result is df  # 原地修改

    def test_already_numeric_column(self):
        from scripts.common.data_io import safe_to_numeric
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = safe_to_numeric(df, columns=["a"])
        assert result["a"].iloc[0] == 1.0

    def test_empty_dataframe(self):
        from scripts.common.data_io import safe_to_numeric
        df = pd.DataFrame()
        result = safe_to_numeric(df)
        assert len(result) == 0

    def test_mixed_types(self):
        from scripts.common.data_io import safe_to_numeric
        df = pd.DataFrame({"a": [1, "2", 3.0, None]})
        result = safe_to_numeric(df, columns=["a"])
        assert pd.isna(result["a"].iloc[3])


class TestReadCsvRequired:
    """scripts.common.data_io.read_csv_required 的测试。"""

    def test_reads_valid_csv(self, tmp_path):
        from scripts.common.data_io import read_csv_required
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        df = read_csv_required(csv_file)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_raises_on_missing_file(self, tmp_path):
        from scripts.common.data_io import read_csv_required
        with pytest.raises(FileNotFoundError):
            read_csv_required(tmp_path / "nonexistent.csv")

    def test_raises_with_label(self, tmp_path):
        from scripts.common.data_io import read_csv_required
        with pytest.raises(FileNotFoundError, match="walk-forward"):
            read_csv_required(tmp_path / "nonexistent.csv", label="walk-forward 结果")

    def test_accepts_string_path(self, tmp_path):
        from scripts.common.data_io import read_csv_required
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n", encoding="utf-8")
        df = read_csv_required(str(csv_file))
        assert len(df) == 1

    def test_utf8_sig_bom_handling(self, tmp_path):
        from scripts.common.data_io import read_csv_required
        csv_file = tmp_path / "bom.csv"
        csv_file.write_bytes(b"\xef\xbb\xbfa,b\n1,2\n")
        df = read_csv_required(csv_file, encoding="utf-8-sig")
        assert list(df.columns) == ["a", "b"]

    def test_custom_encoding(self, tmp_path):
        from scripts.common.data_io import read_csv_required
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a\n1\n", encoding="utf-8")
        df = read_csv_required(csv_file, encoding="utf-8")
        assert len(df) == 1


# =========================================================================
# validation.py
# =========================================================================

class TestParseDateYyyymmdd:
    """scripts.common.validation.parse_date_yyyymmdd 的测试。"""

    def test_valid_date(self):
        from scripts.common.validation import parse_date_yyyymmdd
        ts = parse_date_yyyymmdd("20240115")
        assert ts == pd.Timestamp("2024-01-15")

    def test_start_date(self):
        from scripts.common.validation import parse_date_yyyymmdd
        ts = parse_date_yyyymmdd("20150101")
        assert ts.year == 2015
        assert ts.month == 1
        assert ts.day == 1

    def test_end_date(self):
        from scripts.common.validation import parse_date_yyyymmdd
        ts = parse_date_yyyymmdd("20241231")
        assert ts.year == 2024
        assert ts.month == 12
        assert ts.day == 31

    def test_invalid_format_raises(self):
        from scripts.common.validation import parse_date_yyyymmdd
        with pytest.raises(ValueError, match="无效日期格式"):
            parse_date_yyyymmdd("2024-01-15")

    def test_invalid_date_raises(self):
        from scripts.common.validation import parse_date_yyyymmdd
        with pytest.raises(ValueError):
            parse_date_yyyymmdd("20241301")  # 月份 13

    def test_empty_string_raises(self):
        from scripts.common.validation import parse_date_yyyymmdd
        with pytest.raises(ValueError):
            parse_date_yyyymmdd("")

    def test_returns_timestamp(self):
        from scripts.common.validation import parse_date_yyyymmdd
        ts = parse_date_yyyymmdd("20200601")
        assert isinstance(ts, pd.Timestamp)


class TestResolvePath:
    """scripts.common.validation.resolve_path 的测试。"""

    def test_absolute_path_not_prefixed(self):
        from scripts.common.validation import resolve_path
        from scripts.common.constants import PROJECT_ROOT
        # Use a real absolute path on the current drive to avoid Windows drive-letter issues
        p = Path("C:/absolute/path/to/file.txt") if Path("C:/").exists() else Path("/absolute/path/to/file.txt")
        result = resolve_path(p)
        assert result.is_absolute()
        assert not str(result).startswith(str(PROJECT_ROOT))

    def test_relative_path_prefixed(self):
        from scripts.common.validation import resolve_path
        from scripts.common.constants import PROJECT_ROOT
        result = resolve_path("data/qmt_export/file.csv")
        assert result == PROJECT_ROOT / "data" / "qmt_export" / "file.csv"

    def test_string_input(self):
        from scripts.common.validation import resolve_path
        from scripts.common.constants import PROJECT_ROOT
        result = resolve_path("backtests/test")
        assert result == PROJECT_ROOT / "backtests" / "test"

    def test_path_object_input(self):
        from scripts.common.validation import resolve_path
        from scripts.common.constants import PROJECT_ROOT
        result = resolve_path(Path("scripts/test.py"))
        assert result == PROJECT_ROOT / "scripts" / "test.py"

    def test_absolute_string_not_prefixed(self):
        from scripts.common.validation import resolve_path
        from scripts.common.constants import PROJECT_ROOT
        p = "C:/absolute/path" if Path("C:/").exists() else "/absolute/path"
        result = resolve_path(p)
        assert result.is_absolute()
        assert not str(result).startswith(str(PROJECT_ROOT))


class TestParseList:
    """scripts.common.validation.parse_list 的测试。"""

    def test_basic_list(self):
        from scripts.common.validation import parse_list
        result = parse_list("000001.SZ,600000.SH,000002.SZ")
        assert result == ["000001.SZ", "600000.SH", "000002.SZ"]

    def test_upper_default(self):
        from scripts.common.validation import parse_list
        result = parse_list("abc,def")
        assert result == ["ABC", "DEF"]

    def test_upper_false(self):
        from scripts.common.validation import parse_list
        result = parse_list("abc,def", upper=False)
        assert result == ["abc", "def"]

    def test_strips_whitespace(self):
        from scripts.common.validation import parse_list
        result = parse_list(" a , b , c ")
        assert result == ["A", "B", "C"]

    def test_empty_items_filtered(self):
        from scripts.common.validation import parse_list
        result = parse_list("a,,b,,,c")
        assert result == ["A", "B", "C"]

    def test_single_item(self):
        from scripts.common.validation import parse_list
        result = parse_list("000300.SH")
        assert result == ["000300.SH"]

    def test_empty_string(self):
        from scripts.common.validation import parse_list
        result = parse_list("")
        assert result == []


class TestParseIntList:
    """scripts.common.validation.parse_int_list 的测试。"""

    def test_basic_list(self):
        from scripts.common.validation import parse_int_list
        result = parse_int_list("5,10,20,60")
        assert result == [5, 10, 20, 60]

    def test_single_value(self):
        from scripts.common.validation import parse_int_list
        result = parse_int_list("120")
        assert result == [120]

    def test_strips_whitespace(self):
        from scripts.common.validation import parse_int_list
        result = parse_int_list(" 5 , 10 , 20 ")
        assert result == [5, 10, 20]

    def test_invalid_raises(self):
        from scripts.common.validation import parse_int_list
        with pytest.raises(ValueError):
            parse_int_list("5,abc,20")

    def test_empty_string(self):
        from scripts.common.validation import parse_int_list
        result = parse_int_list("")
        assert result == []


class TestParseWorkers:
    """scripts.common.validation.parse_workers 的测试。"""

    def test_auto(self):
        from scripts.common.validation import parse_workers
        result = parse_workers("auto")
        assert result == os.cpu_count() or 1

    def test_positive_int(self):
        from scripts.common.validation import parse_workers
        assert parse_workers("4") == 4

    def test_one(self):
        from scripts.common.validation import parse_workers
        assert parse_workers("1") == 1

    def test_zero_raises(self):
        from scripts.common.validation import parse_workers
        with pytest.raises(ValueError):
            parse_workers("0")

    def test_negative_raises(self):
        from scripts.common.validation import parse_workers
        with pytest.raises(ValueError):
            parse_workers("-1")

    def test_non_numeric_raises(self):
        from scripts.common.validation import parse_workers
        with pytest.raises(ValueError):
            parse_workers("abc")

    def test_auto_case_insensitive(self):
        from scripts.common.validation import parse_workers
        assert parse_workers("Auto") == os.cpu_count() or 1
        assert parse_workers("AUTO") == os.cpu_count() or 1


class TestSafeSymbolTag:
    """scripts.common.validation.safe_symbol_tag 的测试。"""

    def test_removes_dot(self):
        from scripts.common.validation import safe_symbol_tag
        assert safe_symbol_tag("000001.SZ") == "000001SZ"

    def test_no_dot_unchanged(self):
        from scripts.common.validation import safe_symbol_tag
        assert safe_symbol_tag("000001SZ") == "000001SZ"

    def test_multiple_dots(self):
        from scripts.common.validation import safe_symbol_tag
        assert safe_symbol_tag("a.b.c") == "abc"

    def test_empty_string(self):
        from scripts.common.validation import safe_symbol_tag
        assert safe_symbol_tag("") == ""

    def test_sh_suffix(self):
        from scripts.common.validation import safe_symbol_tag
        assert safe_symbol_tag("600000.SH") == "600000SH"


# =========================================================================
# logging_setup.py
# =========================================================================

class TestLoggingSetup:
    """scripts.common.logging_setup 的基本验证。"""

    def test_setup_cli_logging_runs_without_error(self):
        from scripts.common.logging_setup import setup_cli_logging
        # 不应抛异常
        setup_cli_logging()

    def test_setup_cli_logging_accepts_custom_level(self):
        import logging
        from scripts.common.logging_setup import setup_cli_logging
        # basicConfig 在 pytest 环境下可能不修改已有 handler，但不应抛异常
        setup_cli_logging(level=logging.DEBUG)

    def test_setup_cli_logging_is_callable(self):
        """验证 setup_cli_logging 是可调用的且接受 level 参数。"""
        import logging
        from scripts.common.logging_setup import setup_cli_logging
        assert callable(setup_cli_logging)
        # 接受 int level 参数
        setup_cli_logging(level=logging.WARNING)
