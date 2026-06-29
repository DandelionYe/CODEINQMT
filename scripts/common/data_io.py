# -*- coding: utf-8 -*-
"""
scripts/common/data_io.py

共享数据加载与 I/O 工具。从 12+ 个文件中抽取的重复逻辑。

使用方式：
    from scripts.common.data_io import (
        safe_to_numeric, read_csv_required,
        PROJECT_ROOT, DEFAULT_EXPORT_ROOT, DEFAULT_PARQUET_ROOT,
    )
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.common.constants import PROJECT_ROOT, DEFAULT_EXPORT_ROOT, DEFAULT_PARQUET_ROOT


# ---------------------------------------------------------------------------
# 数值列强制转换
# ---------------------------------------------------------------------------

def safe_to_numeric(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """将指定列安全转换为数值类型，无法转换的变为 NaN。

    原分散在 6 个 diagnose_*/portfolio_*/validate_* 文件中，逻辑完全相同。

    Parameters
    ----------
    df : pd.DataFrame
        待转换的 DataFrame。
    columns : list[str], optional
        要转换的列名列表。为 None 时转换所有列。

    Returns
    -------
    pd.DataFrame
        转换后的 DataFrame（原地修改）。
    """
    cols = columns if columns is not None else df.columns
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# CSV 加载（带存在性检查）
# ---------------------------------------------------------------------------

def read_csv_required(
    path: str | Path,
    *,
    encoding: str = "utf-8-sig",
    label: str = "",
) -> pd.DataFrame:
    """加载 CSV 文件，文件不存在时抛出 FileNotFoundError。

    原分散在 6 个 diagnose_*/portfolio_*/validate_* 文件中。
    统一使用 utf-8-sig 编码（兼容 Excel 导出的 BOM 头）。

    Parameters
    ----------
    path : str | Path
        CSV 文件路径。
    encoding : str
        文件编码，默认 utf-8-sig。
    label : str
        用于错误消息的标签（如 "walk-forward 结果"）。

    Returns
    -------
    pd.DataFrame
        加载的数据。

    Raises
    ------
    FileNotFoundError
        文件不存在时。
    """
    p = Path(path)
    if not p.exists():
        label_str = f"（{label}）" if label else ""
        raise FileNotFoundError(f"文件不存在{label_str}：{p}")
    return pd.read_csv(p, encoding=encoding)
