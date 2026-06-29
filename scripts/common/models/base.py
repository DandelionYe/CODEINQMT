# -*- coding: utf-8 -*-
"""
base.py

AlphaModel 抽象基类，借鉴 Qlib qlib/model/base.py 设计。

提供统一的 fit / predict 接口，让规则模型和 ML 模型共用同一套
walk-forward 评价和回测流程。

设计原则：
  - fit(segment) 只在训练期数据上学习参数，不泄漏测试期。
  - predict(segment) 输出 prediction score，格式与规则信号的 alpha_score 一致。
  - get_params() 记录模型参数，用于 run manifest 和复现。
  - 不绑定特定数据格式，通过 DataFrame 传入传出。

使用方式：
  model = SimpleRuleModel(expression=expr, score_col="alpha_score")
  model.fit(train_df, label_col="label/ret_1d")
  pred = model.predict(test_df)
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AlphaModel(abc.ABC):
    """Alpha 模型抽象基类。

    所有模型（规则模型、LightGBM 等）必须实现此接口。
    prediction score 的语义与 alpha_score 一致：值越高，预期收益越高。
    """

    @abc.abstractmethod
    def fit(
        self,
        train_data: pd.DataFrame,
        label_col: Optional[str] = None,
    ) -> None:
        """在训练期数据上学习参数。

        **金融正确性**：fit 只使用 train_data，不访问测试期数据。
        处理器参数（如 mean、std）只在训练期学习。

        Parameters
        ----------
        train_data : pd.DataFrame
            训练期数据。可以是 feature matrix（含 feature/ 列）或
            已处理的 DataFrame。模型应自行提取所需列。
        label_col : str, optional
            标签列名。ML 模型需要标签训练，规则模型可忽略。
        """
        ...

    @abc.abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """对输入数据生成 prediction score。

        **金融正确性**：predict 不能使用 label 列（推理时标签不可用）。
        输出的 score 应与 alpha_score 语义一致：值越高，预期收益越高。

        Parameters
        ----------
        data : pd.DataFrame
            输入数据。可以是 feature matrix 或已处理的 DataFrame。

        Returns
        -------
        pd.Series
            prediction score，索引与输入数据一致。
        """
        ...

    @abc.abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """返回模型参数字典，用于记录和复现。

        Returns
        -------
        dict
            模型参数。至少包含 model_type 和关键超参数。
        """
        ...

    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"
