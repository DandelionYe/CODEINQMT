# -*- coding: utf-8 -*-
"""
scripts/common/factor_data.py

因子数据加载工具。将 factors/ 目录中的因子数据接入 D's_Flow pipeline。

当前支持：
- ths_board_membership: 同花顺概念/行业板块成分股数据

使用方式：
    from scripts.common.factor_data import load_board_membership, get_stock_boards

    # 加载全部成分股数据
    members = load_board_membership()

    # 查询某只股票所属的板块（PIT-safe）
    boards = get_stock_boards("000001.SZ", query_date="2026-05-19")

    # 获取某个板块的所有成分股
    stocks = get_board_stocks("300008")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from scripts.common.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)

# 因子数据根目录
FACTORS_DIR = PROJECT_ROOT / "factors" / "concept_industry"
PARQUET_DIR = FACTORS_DIR / "parquet"

# 文件路径
BOARDS_FILE = PARQUET_DIR / "ths_boards.parquet"
MEMBERS_FILE = PARQUET_DIR / "ths_board_members.parquet"


def load_board_membership() -> pd.DataFrame:
    """加载全部板块成分股数据。

    Returns
    -------
    pd.DataFrame
        columns: board_code, board_name, board_type, stock_code, stock_name,
                 fetch_date, fetch_time, source
    """
    if not MEMBERS_FILE.exists():
        logger.warning("板块成分股数据不存在: %s", MEMBERS_FILE)
        return pd.DataFrame()

    df = pd.read_parquet(MEMBERS_FILE)
    logger.info("已加载板块成分股数据: %d 行, %d 个板块, %d 只股票",
                len(df), df["board_code"].nunique(), df["stock_code"].nunique())
    return df


def load_boards() -> pd.DataFrame:
    """加载板块元数据。

    Returns
    -------
    pd.DataFrame
        columns: board_code, board_name, board_type, href, ...
    """
    if not BOARDS_FILE.exists():
        logger.warning("板块元数据不存在: %s", BOARDS_FILE)
        return pd.DataFrame()

    return pd.read_parquet(BOARDS_FILE)


def get_stock_boards(
    stock_code: str,
    query_date: str | None = None,
    members_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """查询某只股票所属的板块（PIT-safe）。

    Parameters
    ----------
    stock_code : str
        股票代码，如 "000001.SZ"
    query_date : str, optional
        查询日期 (YYYY-MM-DD)，仅返回 fetch_date <= query_date 的记录。
        为 None 时返回最新数据。
    members_df : pd.DataFrame, optional
        预加载的成分股数据，避免重复读取文件。

    Returns
    -------
    pd.DataFrame
        该股票所属的板块列表。
    """
    if members_df is None:
        members_df = load_board_membership()

    if members_df.empty:
        return members_df

    stock_data = members_df[members_df["stock_code"] == stock_code].copy()

    if stock_data.empty:
        return stock_data

    if query_date is not None:
        stock_data = stock_data[stock_data["fetch_date"] <= query_date]

    # 每个板块取最新的 fetch_date
    stock_data = (
        stock_data.sort_values("fetch_date", ascending=False)
        .drop_duplicates(subset=["board_code", "board_type"], keep="first")
    )

    return stock_data


def get_board_stocks(
    board_code: str,
    board_type: str | None = None,
    query_date: str | None = None,
    members_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """获取某个板块的所有成分股。

    Parameters
    ----------
    board_code : str
        板块代码，如 "300008"
    board_type : str, optional
        板块类型 ("concept" 或 "industry")。为 None 时返回所有类型的成分股。
    query_date : str, optional
        查询日期 (YYYY-MM-DD)，仅返回 fetch_date <= query_date 的记录。
    members_df : pd.DataFrame, optional
        预加载的成分股数据。

    Returns
    -------
    pd.DataFrame
        该板块的成分股列表。
    """
    if members_df is None:
        members_df = load_board_membership()

    if members_df.empty:
        return members_df

    mask = members_df["board_code"] == board_code
    if board_type is not None:
        mask = mask & (members_df["board_type"] == board_type)
    board_data = members_df[mask].copy()

    if board_data.empty:
        return board_data

    if query_date is not None:
        board_data = board_data[board_data["fetch_date"] <= query_date]

    if board_data.empty:
        return board_data

    # 取最新的 fetch_date
    latest_date = board_data["fetch_date"].max()
    board_data = board_data[board_data["fetch_date"] == latest_date]

    return board_data


def get_board_coverage_stats(members_df: pd.DataFrame | None = None) -> dict:
    """获取板块数据覆盖统计。

    Returns
    -------
    dict
        total_boards, covered_boards, total_stocks, coverage_rate
    """
    if members_df is None:
        members_df = load_board_membership()

    boards_df = load_boards()

    total_boards = len(boards_df)
    covered_boards = members_df["board_code"].nunique() if not members_df.empty else 0
    total_stocks = members_df["stock_code"].nunique() if not members_df.empty else 0

    return {
        "total_boards": total_boards,
        "covered_boards": covered_boards,
        "coverage_rate": covered_boards / total_boards if total_boards > 0 else 0.0,
        "total_stocks": total_stocks,
        "total_member_rows": len(members_df),
    }
