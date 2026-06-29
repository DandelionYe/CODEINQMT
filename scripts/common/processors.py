# -*- coding: utf-8 -*-
"""
processors.py

轻量 Processor 链，借鉴 Qlib DataHandlerLP / Processor 的设计。

提供数据预处理组件：缺失值填充、截面标准化、winsorize、训练期拟合标准化等。
所有 Processor 遵循 fit/process 二阶段接口，fit 只在训练期调用，避免未来函数泄漏。

核心设计：
  - fit(data, segment_mask)：在训练期数据上学习参数（如 mean、std）。
  - process(data)：将学习到的参数应用到任意数据。
  - fit_transform(data, segment_mask)：fit + process 的便捷组合。

Processor 列表：
  FillNa          - 填充 NaN
  ProcessInf      - 将 inf 替换为 NaN
  DropNaFeature   - 删除特征列为 NaN 的行
  DropNaLabel     - 删除标签列为 NaN 的行
  CSZScoreNorm    - 截面 z-score 标准化（每日 groupby）
  CSRankNorm      - 截面排名标准化（每日 groupby）
  Winsorize       - 极值截断
  TrainFitZScore  - 训练期拟合的 z-score 标准化（时间序列维度）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class Processor:
    """Processor 基类。所有 Processor 必须实现 _fit 和 _process。"""

    def __init__(self) -> None:
        self._fitted = False
        self._fit_params: Dict[str, Any] = {}

    def fit(self, data: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "Processor":
        """在训练期数据上学习参数。

        Parameters
        ----------
        data : pd.DataFrame
            训练期数据，(date, symbol) MultiIndex。
        feature_cols : list of str, optional
            要处理的列名列表。为 None 时自动选取 feature/ 开头的列。
        """
        cols = feature_cols or _auto_feature_cols(data)
        self._fit_internal(data, cols)
        self._fitted = True
        return self

    def process(self, data: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """将学习到的参数应用到数据。

        Parameters
        ----------
        data : pd.DataFrame
            待处理数据。
        feature_cols : list of str, optional
            要处理的列名列表。

        Returns
        -------
        pd.DataFrame
            处理后的数据（副本）。
        """
        cols = feature_cols or _auto_feature_cols(data)
        return self._process_internal(data.copy(), cols)

    def fit_transform(
        self, data: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """fit + process 的便捷组合。"""
        self.fit(data, feature_cols)
        return self.process(data, feature_cols)

    def _fit_internal(self, data: pd.DataFrame, cols: List[str]) -> None:
        """子类实现：学习参数。"""
        raise NotImplementedError

    def _process_internal(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """子类实现：应用参数。"""
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        """返回学习到的参数（用于序列化/调试）。"""
        return {
            "type": type(self).__name__,
            "fitted": self._fitted,
            "fit_params": self._fit_params,
        }


def _auto_feature_cols(data: pd.DataFrame) -> List[str]:
    """自动选取 feature/ 开头的列。"""
    return [c for c in data.columns if c.startswith("feature/")]


# ---------------------------------------------------------------------------
# FillNa
# ---------------------------------------------------------------------------

class FillNa(Processor):
    """用指定值填充 NaN。

    Parameters
    ----------
    fill_value : float
        填充值，默认 0.0。
    """

    def __init__(self, fill_value: float = 0.0) -> None:
        super().__init__()
        self.fill_value = fill_value

    def _fit_internal(self, data: pd.DataFrame, cols: List[str]) -> None:
        self._fit_params["fill_value"] = self.fill_value

    def _process_internal(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in data.columns:
                data[c] = data[c].fillna(self.fill_value)
        return data


# ---------------------------------------------------------------------------
# ProcessInf
# ---------------------------------------------------------------------------

class ProcessInf(Processor):
    """将 inf/-inf 替换为 NaN。"""

    def _fit_internal(self, data: pd.DataFrame, cols: List[str]) -> None:
        pass

    def _process_internal(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in data.columns:
                data[c] = data[c].replace([np.inf, -np.inf], np.nan)
        return data


# ---------------------------------------------------------------------------
# DropNaFeature
# ---------------------------------------------------------------------------

class DropNaFeature(Processor):
    """删除特征列中含 NaN 的行。

    Parameters
    ----------
    how : str
        'any'（任一 NaN 即删）或 'all'（全部 NaN 才删）。默认 'any'。
    """

    def __init__(self, how: str = "any") -> None:
        super().__init__()
        self.how = how

    def _fit_internal(self, data: pd.DataFrame, cols: List[str]) -> None:
        self._fit_params["how"] = self.how

    def _process_internal(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        existing = [c for c in cols if c in data.columns]
        if not existing:
            return data
        return data.dropna(subset=existing, how=self.how)


# ---------------------------------------------------------------------------
# DropNaLabel
# ---------------------------------------------------------------------------

class DropNaLabel(Processor):
    """删除标签列中含 NaN 的行。

    Parameters
    ----------
    label_cols : list of str
        标签列名列表。默认 ['label/ret_1d', 'label/ret_5d', 'label/ret_20d']。
    how : str
        'any' 或 'all'。默认 'any'。
    """

    def __init__(
        self,
        label_cols: Optional[List[str]] = None,
        how: str = "any",
    ) -> None:
        super().__init__()
        self.label_cols = label_cols or ["label/ret_1d", "label/ret_5d", "label/ret_20d"]
        self.how = how

    def _fit_internal(self, data: pd.DataFrame, cols: List[str]) -> None:
        self._fit_params["label_cols"] = self.label_cols
        self._fit_params["how"] = self.how

    def _process_internal(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        existing = [c for c in self.label_cols if c in data.columns]
        if not existing:
            return data
        return data.dropna(subset=existing, how=self.how)


# ---------------------------------------------------------------------------
# CSZScoreNorm
# ---------------------------------------------------------------------------

class CSZScoreNorm(Processor):
    """截面 z-score 标准化。

    每个交易日独立计算 mean 和 std，然后标准化。
    适用于 MultiIndex (date, symbol) 数据。

    **金融正确性**：截面标准化按交易日 groupby，不跨日期泄漏。
    """

    def _fit_internal(self, data: pd.DataFrame, cols: List[str]) -> None:
        # 截面标准化不需要全局 fit 参数，每日独立计算
        self._fit_params["note"] = "CSZScoreNorm 每日独立计算，无需全局 fit"

    def _process_internal(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in data.columns:
                data[c] = _cs_zscore(data[c])
        return data


def _cs_zscore(s: pd.Series) -> pd.Series:
    """截面 z-score：按 level-0 (date) groupby 标准化。"""
    if isinstance(s.index, pd.MultiIndex) and len(s.index.names) >= 2:
        grouped = s.groupby(level=0)
        mean = grouped.transform("mean")
        std = grouped.transform("std")
    else:
        mean = s.mean()
        std = s.std()

    # std=0 或 NaN 时返回 0
    result = (s - mean) / std.replace(0, np.nan)
    return result.fillna(0.0)


# ---------------------------------------------------------------------------
# CSRankNorm
# ---------------------------------------------------------------------------

class CSRankNorm(Processor):
    """截面排名标准化。

    每个交易日独立排名，归一化到 [0, 1]。
    适用于 MultiIndex (date, symbol) 数据。

    **金融正确性**：截面排名按交易日 groupby，不跨日期泄漏。
    """

    def _fit_internal(self, data: pd.DataFrame, cols: List[str]) -> None:
        self._fit_params["note"] = "CSRankNorm 每日独立计算，无需全局 fit"

    def _process_internal(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in data.columns:
                data[c] = _cs_rank(data[c])
        return data


def _cs_rank(s: pd.Series) -> pd.Series:
    """截面排名：按 level-0 (date) groupby 排名归一化。"""
    if isinstance(s.index, pd.MultiIndex) and len(s.index.names) >= 2:
        return s.groupby(level=0).rank(pct=True)
    return s.rank(pct=True)


# ---------------------------------------------------------------------------
# Winsorize
# ---------------------------------------------------------------------------

class Winsorize(Processor):
    """极值截断（winsorize）。

    将超出 [lower_quantile, upper_quantile] 范围的值截断到边界。
    quantile 参数在 fit 时根据训练数据确定边界，process 时应用。

    Parameters
    ----------
    lower_quantile : float
        下分位数，默认 0.01。
    upper_quantile : float
        上分位数，默认 0.99。
    """

    def __init__(
        self,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        super().__init__()
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def _fit_internal(self, data: pd.DataFrame, cols: List[str]) -> None:
        bounds = {}
        for c in cols:
            if c in data.columns:
                s = data[c].dropna()
                if len(s) > 0:
                    bounds[c] = (
                        float(s.quantile(self.lower_quantile)),
                        float(s.quantile(self.upper_quantile)),
                    )
        self._fit_params["bounds"] = bounds

    def _process_internal(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        bounds = self._fit_params.get("bounds", {})
        for c in cols:
            if c in data.columns and c in bounds:
                lo, hi = bounds[c]
                data[c] = data[c].clip(lower=lo, upper=hi)
        return data


# ---------------------------------------------------------------------------
# TrainFitZScore
# ---------------------------------------------------------------------------

class TrainFitZScore(Processor):
    """训练期拟合的 z-score 标准化。

    在训练期计算 mean 和 std，然后用这些参数标准化所有数据。
    适用于时间序列维度的标准化，不是截面标准化。

    **金融正确性**：
    - fit 只使用训练期数据，不包含测试期。
    - process 使用训练期学习到的参数，确保不泄漏。
    """

    def _fit_internal(self, data: pd.DataFrame, cols: List[str]) -> None:
        stats = {}
        for c in cols:
            if c in data.columns:
                s = data[c].dropna()
                if len(s) > 0:
                    stats[c] = {
                        "mean": float(s.mean()),
                        "std": float(s.std()),
                    }
        self._fit_params["stats"] = stats

    def _process_internal(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        stats = self._fit_params.get("stats", {})
        for c in cols:
            if c in data.columns and c in stats:
                mean = stats[c]["mean"]
                std = stats[c]["std"]
                if std == 0 or np.isnan(std):
                    data[c] = 0.0
                else:
                    data[c] = (data[c] - mean) / std
        return data


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def make_default_infer_processors() -> List[Processor]:
    """构建默认的推理处理器链（DK_I）。

    处理顺序：
    1. ProcessInf：替换 inf
    2. FillNa(0)：填充 NaN
    3. CSZScoreNorm：截面 z-score
    """
    return [
        ProcessInf(),
        FillNa(0.0),
        CSZScoreNorm(),
    ]


def make_default_learn_processors() -> List[Processor]:
    """构建默认的学习处理器链（DK_L）。

    处理顺序：
    1. ProcessInf：替换 inf
    2. FillNa(0)：填充 NaN
    3. DropNaLabel：删除标签缺失的行
    4. CSZScoreNorm：截面 z-score
    """
    return [
        ProcessInf(),
        FillNa(0.0),
        DropNaLabel(),
        CSZScoreNorm(),
    ]


def apply_processor_chain(
    data: pd.DataFrame,
    processors: List[Processor],
    feature_cols: Optional[List[str]] = None,
    fit_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """依次应用处理器链。

    Parameters
    ----------
    data : pd.DataFrame
        待处理数据。
    processors : list of Processor
        处理器列表。
    feature_cols : list of str, optional
        要处理的列。
    fit_data : pd.DataFrame, optional
        用于 fit 的数据（通常为训练期数据）。为 None 时用 data 本身 fit。

    Returns
    -------
    pd.DataFrame
        处理后的数据。
    """
    result = data
    fit_df = fit_data if fit_data is not None else data

    for proc in processors:
        proc.fit(fit_df, feature_cols)
        result = proc.process(result, feature_cols)

    return result
