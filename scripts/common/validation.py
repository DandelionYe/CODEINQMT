# -*- coding: utf-8 -*-
"""
scripts/common/validation.py

共享验证与解析工具。从 14+ 个文件中抽取的重复逻辑。

使用方式：
    from scripts.common.validation import (
        parse_date_yyyymmdd, resolve_path, parse_list,
        parse_int_list, parse_workers, safe_symbol_tag,
    )
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scripts.common.constants import PROJECT_ROOT


# ---------------------------------------------------------------------------
# 日期解析
# ---------------------------------------------------------------------------

def parse_date_yyyymmdd(text: str) -> pd.Timestamp:
    """将 'YYYYMMDD' 字符串解析为 pd.Timestamp。

    严格格式校验，非法输入抛出 ValueError。
    原分散在 validate_ma_candidates / validate_ma_v3_momentum_candidates /
    validate_alpha_v4_research_candidates / validate_ma_market_filter_candidates。
    """
    dt = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"无效日期格式：'{text}'，期望 YYYYMMDD")
    return pd.Timestamp(dt)


# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------

def resolve_path(path_text: str | Path) -> Path:
    """将相对路径解析为基于 PROJECT_ROOT 的绝对路径。

    原分散在 12 个 scripts/ 文件中，逻辑完全相同。
    """
    p = Path(path_text)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


# ---------------------------------------------------------------------------
# 列表解析
# ---------------------------------------------------------------------------

def parse_list(text: str, *, upper: bool = True) -> list[str]:
    """将逗号分隔的字符串解析为字符串列表。

    原分散在 15 个文件中（名 parse_list 或 parse_symbol_list）。
    upper=True 时对每项调用 .upper()，适合 symbol/benchmark 列表。
    """
    items = [x.strip() for x in text.split(",") if x.strip()]
    if upper:
        items = [x.upper() for x in items]
    return items


def parse_int_list(text: str) -> list[int]:
    """将逗号分隔的字符串解析为整数列表。

    原分散在 7 个文件中，逻辑完全相同。
    """
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# workers 解析
# ---------------------------------------------------------------------------

def parse_workers(text: str) -> int:
    """解析 --workers 参数。'auto' 返回 CPU 数，否则转正整数。

    原分散在 7 个文件中，逻辑完全相同。
    非法输入或非正整数抛出 ValueError（argparse 会自动捕获）。
    """
    if text.lower() == "auto":
        return os.cpu_count() or 1
    try:
        n = int(text)
        if n < 1:
            raise ValueError
        return n
    except ValueError:
        raise ValueError(f"无效的 workers 值：{text}。请使用正整数或 'auto'。")


# ---------------------------------------------------------------------------
# Symbol 工具
# ---------------------------------------------------------------------------

def safe_symbol_tag(symbol: str) -> str:
    """将 '000001.SZ' 转为 '000001SZ'（去掉点号）。

    原分散在 6 个文件中，逻辑完全相同。
    """
    return symbol.replace(".", "")
