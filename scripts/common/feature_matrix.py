# -*- coding: utf-8 -*-
"""
feature_matrix.py

Feature Matrix Builder：将多股票的日频数据组装为统一的 (date, symbol) MultiIndex 特征矩阵。

借鉴 Qlib DatasetH.prepare(segment, col_set=["feature", "label"]) 的设计，
为规则策略、信号评价和未来 ML 模型提供统一数据接口。

核心功能：
  - load_stock_data()：加载 parquet 股票数据
  - compute_factor_on_universe()：对股票池计算因子表达式
  - compute_forward_returns()：计算前瞻收益标签
  - build_feature_matrix()：一站式构建特征矩阵
  - save_feature_matrix()：保存 parquet + manifest

输出结构：
  index: (date, symbol)
  columns:
    feature/<factor_id>
    label/ret_1d, label/ret_5d, label/ret_20d
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.common.constants import PROJECT_ROOT, DEFAULT_PARQUET_ROOT

logger = logging.getLogger(__name__)

# 默认 parquet 数据根目录
_DEFAULT_PARQUET_ROOT = DEFAULT_PARQUET_ROOT


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_stock_data(
    symbol: str,
    parquet_root: Path = _DEFAULT_PARQUET_ROOT,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """加载单只股票的 parquet 数据。

    Parameters
    ----------
    symbol : str
        股票代码，如 "000001.SZ" 或 "600000.SH"。
    parquet_root : Path
        parquet 数据根目录。
    start, end : str, optional
        日期范围过滤，格式 "YYYYMMDD"。

    Returns
    -------
    pd.DataFrame
        含 date(int)、symbol、open、high、low、close、volume、amount 列。
    """
    code, market = symbol.split(".")
    path = parquet_root / market / f"price_{code}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"股票数据文件不存在：{path}")

    df = pd.read_parquet(path)

    # 标准化列名：timetag -> date, volumn -> volume
    if "timetag" in df.columns:
        df = df.rename(columns={"timetag": "date"})
    if "volumn" in df.columns:
        df = df.rename(columns={"volumn": "volume"})

    # 添加 symbol 列
    df["symbol"] = symbol

    # 日期过滤
    if start is not None:
        df = df[df["date"] >= int(start)]
    if end is not None:
        df = df[df["date"] <= int(end)]

    # 确保 date 为 int
    df["date"] = df["date"].astype(int)

    return df[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


def load_universe_data(
    symbols: Sequence[str],
    parquet_root: Path = _DEFAULT_PARQUET_ROOT,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """加载股票池所有股票的数据并拼接。

    Parameters
    ----------
    symbols : Sequence[str]
        股票代码列表。
    parquet_root : Path
        parquet 数据根目录。
    start, end : str, optional
        日期范围。

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, symbol) 的拼接数据。
    """
    frames = []
    failed = []
    for sym in symbols:
        try:
            df = load_stock_data(sym, parquet_root, start, end)
            frames.append(df)
        except FileNotFoundError:
            failed.append(sym)
            logger.warning("跳过缺失股票：%s", sym)

    if failed:
        logger.warning("共 %d 只股票数据缺失", len(failed))

    if not frames:
        raise ValueError("无有效股票数据")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index(["date", "symbol"]).sort_index()
    return combined


# ---------------------------------------------------------------------------
# 因子计算
# ---------------------------------------------------------------------------

def compute_factor_on_universe(
    universe_data: pd.DataFrame,
    factor_expr: Any,
    factor_id: str,
) -> pd.Series:
    """对股票池数据计算因子表达式。

    对每只股票独立计算表达式，然后拼接回 MultiIndex。

    Parameters
    ----------
    universe_data : pd.DataFrame
        MultiIndex (date, symbol) 的股票数据。
    factor_expr : Expression
        因子表达式对象（来自 feature_expression 模块）。
    factor_id : str
        因子 ID，用于命名输出列。

    Returns
    -------
    pd.Series
        以 (date, symbol) 为索引的因子值。
    """
    results = []
    symbols = universe_data.index.get_level_values(1).unique()

    for sym in symbols:
        try:
            stock_df = universe_data.xs(sym, level=1)
            vals = factor_expr.eval(stock_df)
            vals = vals.rename(factor_id)
            # 恢复 symbol 索引
            vals.index = pd.MultiIndex.from_arrays(
                [vals.index, [sym] * len(vals)],
                names=["date", "symbol"],
            )
            results.append(vals)
        except Exception as e:
            logger.warning("计算因子 %s 失败（%s）：%s", factor_id, sym, e)

    if not results:
        return pd.Series(dtype=float, name=factor_id)

    combined = pd.concat(results).sort_index()
    combined.name = factor_id
    return combined


# ---------------------------------------------------------------------------
# 前瞻收益标签
# ---------------------------------------------------------------------------

def compute_forward_returns(
    universe_data: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 20),
) -> pd.DataFrame:
    """计算前瞻收益标签。

    对每只股票计算 ret_Nd = close.shift(-N) / close - 1。

    **金融正确性**：
    - 使用 shift(-N) 获取未来收益，这是 label 构造的标准做法。
    - 在回测时，信号必须 shift(1) 后才能使用，避免未来函数。
    - ret_Nd 在 T 日可用时，实际代表 T+N 日的收益，应在 T+1 日才能交易。

    Parameters
    ----------
    universe_data : pd.DataFrame
        MultiIndex (date, symbol) 的股票数据，必须含 close 列。
    horizons : Sequence[int]
        前瞻期列表，默认 (1, 5, 20)。

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, symbol)，列为 label/ret_1d, label/ret_5d, ...
    """
    label_frames = []
    symbols = universe_data.index.get_level_values(1).unique()

    for sym in symbols:
        try:
            stock_df = universe_data.xs(sym, level=1)
            close = stock_df["close"]

            sym_labels = pd.DataFrame(index=stock_df.index)
            for h in horizons:
                # 前瞻收益：未来第 h 日的收盘价 / 当日收盘价 - 1
                future_close = close.shift(-h)
                sym_labels[f"label/ret_{h}d"] = future_close / close - 1

            # 恢复 symbol 索引
            sym_labels.index = pd.MultiIndex.from_arrays(
                [sym_labels.index, [sym] * len(sym_labels)],
                names=["date", "symbol"],
            )
            label_frames.append(sym_labels)
        except Exception as e:
            logger.warning("计算前瞻收益失败（%s）：%s", sym, e)

    if not label_frames:
        return pd.DataFrame()

    combined = pd.concat(label_frames).sort_index()
    return combined


# ---------------------------------------------------------------------------
# 构建特征矩阵
# ---------------------------------------------------------------------------

def build_feature_matrix(
    symbols: Sequence[str],
    factor_expressions: Dict[str, Any],
    parquet_root: Path = _DEFAULT_PARQUET_ROOT,
    start: Optional[str] = None,
    end: Optional[str] = None,
    label_horizons: Sequence[int] = (1, 5, 20),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """一站式构建特征矩阵。

    Parameters
    ----------
    symbols : Sequence[str]
        股票代码列表。
    factor_expressions : Dict[str, Any]
        {factor_id: Expression} 映射。
    parquet_root : Path
        parquet 数据根目录。
    start, end : str, optional
        日期范围。
    label_horizons : Sequence[int]
        前瞻收益期数。

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (feature_matrix, manifest)
    """
    # 1. 加载数据
    logger.info("加载 %d 只股票数据...", len(symbols))
    universe_data = load_universe_data(symbols, parquet_root, start, end)
    logger.info("数据加载完成：shape=%s", universe_data.shape)

    # 2. 计算因子
    factor_results = {}
    for factor_id, expr in factor_expressions.items():
        logger.info("计算因子：%s", factor_id)
        factor_results[f"feature/{factor_id}"] = compute_factor_on_universe(
            universe_data, expr, f"feature/{factor_id}"
        )

    # 3. 计算前瞻收益标签
    logger.info("计算前瞻收益标签：horizons=%s", label_horizons)
    labels = compute_forward_returns(universe_data, label_horizons)

    # 4. 拼接
    parts = list(factor_results.values())
    if not labels.empty:
        parts.append(labels)

    feature_matrix = pd.concat(parts, axis=1)

    # 5. 构建 manifest
    manifest = _build_manifest(
        symbols, factor_expressions, parquet_root, start, end,
        label_horizons, universe_data, feature_matrix,
    )

    logger.info("特征矩阵构建完成：shape=%s", feature_matrix.shape)
    return feature_matrix, manifest


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------

def save_feature_matrix(
    feature_matrix: pd.DataFrame,
    manifest: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """保存特征矩阵和 manifest。

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        特征矩阵。
    manifest : Dict[str, Any]
        manifest 字典。
    output_dir : Path
        输出目录。

    Returns
    -------
    Path
        输出目录路径。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 parquet
    parquet_path = output_dir / "feature_matrix.parquet"
    feature_matrix.to_parquet(parquet_path, engine="pyarrow")

    # 保存 manifest
    manifest_path = output_dir / "feature_matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 保存 CSV 快照（方便人工检查）
    csv_path = output_dir / "feature_matrix_head.csv"
    feature_matrix.head(200).to_csv(csv_path, encoding="utf-8-sig")

    logger.info("特征矩阵已保存到：%s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _build_manifest(
    symbols: Sequence[str],
    factor_expressions: Dict[str, Any],
    parquet_root: Path,
    start: Optional[str],
    end: Optional[str],
    label_horizons: Sequence[int],
    universe_data: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> Dict[str, Any]:
    """构建 manifest JSON。"""
    dates = feature_matrix.index.get_level_values(0)
    date_range = [int(dates.min()), int(dates.max())] if len(dates) > 0 else []

    # 数据哈希（基于文件路径和大小的简易哈希）
    data_hash = hashlib.md5(
        f"{sorted(symbols)}_{date_range}_{len(factor_expressions)}".encode()
    ).hexdigest()[:12]

    return {
        "generated_at": datetime.now().isoformat(),
        "factor_ids": list(factor_expressions.keys()),
        "params": {
            fid: _expr_to_dict(expr) for fid, expr in factor_expressions.items()
        },
        "source_data_root": str(parquet_root),
        "date_range": date_range,
        "universe": sorted(symbols),
        "universe_size": len(symbols),
        "label_horizons": list(label_horizons),
        "data_hash": data_hash,
        "pit_rules_summary": "行情滚动因子默认 PIT-safe；不使用未来数据。",
        "shape": list(feature_matrix.shape),
        "columns": list(feature_matrix.columns),
    }


def _expr_to_dict(expr: Any) -> str:
    """将表达式对象转为字符串表示。"""
    return repr(expr)


# ---------------------------------------------------------------------------
# 从 factor_registry.json 构建表达式
# ---------------------------------------------------------------------------

def build_expressions_from_registry(
    registry_path: Path,
    variant_ids: Optional[Sequence[str]] = None,
    params_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """从 factor_registry.json 构建表达式对象。

    目前只支持登记在 registry 中的因子，表达式通过 eval() 还原。
    后续可改为从 expression 字段解析。

    Parameters
    ----------
    registry_path : Path
        factor_registry.json 路径。
    variant_ids : Sequence[str], optional
        只构建指定的因子。为 None 时构建全部。
    params_override : Dict[str, Any], optional
        参数覆盖，如 {"reversal_window": 20}。

    Returns
    -------
    Dict[str, Any]
        {factor_id: Expression} 映射。
    """
    from scripts.common.feature_expression import (
        Field, PctChange, Shift, Neg, Abs, RollingMean, RollingStd,
        RollingMax, RollingMin, ZScore, CSRank, Where,
        Add, Sub, Mul, Div, Gt, Ge, Lt, Le, Eq, And, Or, Not,
        Const, AsInt, normalize_zscore,
    )
    import math

    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    factors = registry.get("factors", [])

    if variant_ids is not None:
        factors = [f for f in factors if f.get("variant") in variant_ids]

    result = {}
    for factor_def in factors:
        factor_id = factor_def["factor_id"]
        expr_str = factor_def.get("expression", "")

        # 合并参数
        params = {}
        for pname, pdef in factor_def.get("params", {}).items():
            params[pname] = pdef.get("default", 0)
        if params_override:
            params.update(params_override)

        try:
            # 安全 eval：只允许已知的表达式类和参数
            safe_ns = {
                "Field": Field, "PctChange": PctChange, "Shift": Shift,
                "Neg": Neg, "Abs": Abs, "RollingMean": RollingMean,
                "RollingStd": RollingStd, "RollingMax": RollingMax,
                "RollingMin": RollingMin, "ZScore": ZScore, "CSRank": CSRank,
                "Where": Where, "Add": Add, "Sub": Sub, "Mul": Mul,
                "Div": Div, "Gt": Gt, "Ge": Ge, "Lt": Lt, "Le": Le,
                "Eq": Eq, "And": And, "Or": Or, "Not": Not,
                "Const": Const, "AsInt": AsInt,
                "normalize_zscore": normalize_zscore,
                "sqrt": math.sqrt,
                **params,
            }
            expr = eval(expr_str, {"__builtins__": {}}, safe_ns)
            result[factor_id] = expr
            logger.info("从 registry 构建表达式：%s", factor_id)
        except Exception as e:
            logger.warning("构建表达式失败（%s）：%s", factor_id, e)

    return result


# ---------------------------------------------------------------------------
# 便捷函数：从 registry 构建完整特征矩阵
# ---------------------------------------------------------------------------

def build_feature_matrix_from_registry(
    symbols: Sequence[str],
    registry_path: Path,
    variant_ids: Optional[Sequence[str]] = None,
    params_override: Optional[Dict[str, Any]] = None,
    parquet_root: Path = _DEFAULT_PARQUET_ROOT,
    start: Optional[str] = None,
    end: Optional[str] = None,
    label_horizons: Sequence[int] = (1, 5, 20),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """从 factor_registry.json 构建完整特征矩阵。

    便捷函数，组合了 build_expressions_from_registry() 和 build_feature_matrix()。
    """
    expressions = build_expressions_from_registry(
        registry_path, variant_ids, params_override
    )
    if not expressions:
        raise ValueError("未能从 registry 构建任何表达式")

    return build_feature_matrix(
        symbols, expressions, parquet_root, start, end, label_horizons,
    )
