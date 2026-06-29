# -*- coding: utf-8 -*-
"""
rule_model.py

SimpleRuleModel — 基于特征表达式的规则模型。

将 feature_expression.py 的表达式层包装为 AlphaModel 接口，
使规则信号和 ML 模型共用同一套 walk-forward 评价和回测流程。

设计：
  - fit() 为空操作（规则模型不需要训练）。
  - predict() 对输入数据评估表达式，输出 alpha_score。
  - 支持预计算列（直接使用已有的 alpha_score 列）。

使用方式：

  # 方式 1：使用表达式
  from scripts.common.feature_expression import build_expression
  expr = build_expression("short_term_reversal", reversal_window=10)
  model = SimpleRuleModel(expression=expr)

  # 方式 2：使用预计算列
  model = SimpleRuleModel(score_col="feature/reversal_10d")

  # 统一接口
  model.fit(train_data)
  pred = model.predict(test_data)  # -> pd.Series
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from scripts.common.feature_expression import Expression, normalize_zscore

logger = logging.getLogger(__name__)


class SimpleRuleModel:
    """基于特征表达式的规则模型。

    Parameters
    ----------
    expression : Expression, optional
        特征表达式。评估后作为 raw score，再经 ZScore 标准化。
    score_col : str, optional
        预计算的 score 列名（如 "feature/reversal_10d"）。
        如果提供，直接使用该列作为 prediction score。
        expression 和 score_col 必须提供其一。
    zscore : bool, default True
        是否对 expression 输出做 ZScore 标准化。
    signal_threshold : float, default 0.0
        信号阈值。alpha_score > threshold 视为看多。
    """

    def __init__(
        self,
        expression: Optional[Expression] = None,
        score_col: Optional[str] = None,
        zscore: bool = True,
        signal_threshold: float = 0.0,
    ) -> None:
        if expression is None and score_col is None:
            raise ValueError("必须提供 expression 或 score_col 之一")
        if expression is not None and score_col is not None:
            raise ValueError("expression 和 score_col 不能同时提供")

        self._expression = expression
        self._score_col = score_col
        self._zscore = zscore
        self._signal_threshold = signal_threshold
        self._fitted = False

    def fit(
        self,
        train_data: pd.DataFrame,
        label_col: Optional[str] = None,
    ) -> None:
        """规则模型 fit 为空操作。

        规则模型不需要训练，但记录 fit 状态以保持接口一致性。
        label_col 参数被忽略。
        """
        self._fitted = True
        logger.debug("SimpleRuleModel.fit: 规则模型无需训练，标记为 fitted")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """生成 prediction score。

        Parameters
        ----------
        data : pd.DataFrame
            输入数据。expression 模式下需要包含表达式依赖的列；
            score_col 模式下需要包含 score_col 列。

        Returns
        -------
        pd.Series
            prediction score，索引与输入数据一致。
        """
        if self._score_col is not None:
            return self._predict_from_col(data)
        else:
            return self._predict_from_expression(data)

    def predict_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成二值信号（1=看多，0=空仓）。

        Returns
        -------
        pd.Series
            二值信号。
        """
        score = self.predict(data)
        return (score > self._signal_threshold).astype(int)

    def get_params(self) -> Dict[str, Any]:
        """返回模型参数。"""
        params: Dict[str, Any] = {
            "model_type": "SimpleRuleModel",
            "zscore": self._zscore,
            "signal_threshold": self._signal_threshold,
        }
        if self._expression is not None:
            params["expression"] = repr(self._expression)
        if self._score_col is not None:
            params["score_col"] = self._score_col
        return params

    # -----------------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------------

    def _predict_from_col(self, data: pd.DataFrame) -> pd.Series:
        """从预计算列读取 score。"""
        col = self._score_col
        if col not in data.columns:
            raise KeyError(
                f"列 '{col}' 不在数据中。可用列：{list(data.columns[:10])}..."
            )
        score = data[col].copy()
        if self._zscore:
            score = normalize_zscore(score)
        return score

    def _predict_from_expression(self, data: pd.DataFrame) -> pd.Series:
        """评估表达式生成 score。"""
        raw = self._expression.eval(data)
        if self._zscore:
            return normalize_zscore(raw)
        else:
            return raw

    def __repr__(self) -> str:
        if self._expression is not None:
            return f"SimpleRuleModel(expression={self._expression!r}, zscore={self._zscore})"
        return f"SimpleRuleModel(score_col={self._score_col!r}, zscore={self._zscore})"
