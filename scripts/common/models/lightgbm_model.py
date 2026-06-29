# -*- coding: utf-8 -*-
"""
lightgbm_model.py

LightGBMModel — 基于 LightGBM 的 ML alpha 模型。

将 LightGBM 梯度提升树包装为 AlphaModel 接口，
使 ML 模型和规则模型共用同一套 walk-forward 评价和回测流程。

依赖：lightgbm（可选，不在最小 requirements 中）。
当 lightgbm 未安装时，实例化会抛出 ImportError 并给出明确提示。

使用方式：

  from scripts.common.models import LightGBMModel

  model = LightGBMModel(
      label_col="label/ret_1d",
      feature_cols=["feature/reversal_10d", "feature/low_vol_60d"],
      n_estimators=100,
      learning_rate=0.05,
  )
  model.fit(train_df)
  pred = model.predict(test_df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.common.models.base import AlphaModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 延迟导入 lightgbm，未安装时给出明确错误
# ---------------------------------------------------------------------------

def _import_lightgbm():
    """延迟导入 lightgbm，未安装时抛出 ImportError。"""
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        raise ImportError(
            "LightGBM 未安装。请先在 research-env 环境中安装：\n"
            "  conda activate research-env\n"
            "  pip install lightgbm\n"
            "或：\n"
            "  conda install -c conda-forge lightgbm"
        )


# ---------------------------------------------------------------------------
# LightGBMModel
# ---------------------------------------------------------------------------

class LightGBMModel(AlphaModel):
    """基于 LightGBM 的 alpha 模型。

    Parameters
    ----------
    label_col : str
        标签列名（如 "label/ret_1d"）。fit 时从此列读取训练标签。
    feature_cols : list of str, optional
        特征列名列表。为 None 时自动检测 "feature/" 前缀列。
    n_estimators : int, default 100
        树的数量。
    learning_rate : float, default 0.05
        学习率。
    max_depth : int, default 6
        树的最大深度。
    num_leaves : int, default 31
        叶子节点数。
    subsample : float, default 0.8
        行采样比例。
    colsample_bytree : float, default 0.8
        列采样比例。
    random_state : int, default 42
        随机种子。
    verbose : int, default -1
        LightGBM 日志级别。-1 为静默。
    """

    def __init__(
        self,
        label_col: str = "label/ret_1d",
        feature_cols: Optional[List[str]] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbose: int = -1,
    ) -> None:
        # 验证 lightgbm 可用（在构造时立即检查，而非 fit/predict 时）
        _import_lightgbm()

        self._label_col = label_col
        self._feature_cols = feature_cols
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._num_leaves = num_leaves
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._random_state = random_state
        self._verbose = verbose

        self._model = None
        self._fitted_feature_cols: Optional[List[str]] = None

    # -----------------------------------------------------------------------
    # AlphaModel 接口
    # -----------------------------------------------------------------------

    def fit(
        self,
        train_data: pd.DataFrame,
        label_col: Optional[str] = None,
    ) -> None:
        """在训练期数据上拟合 LightGBM 模型。

        **金融正确性**：fit 只使用 train_data，不访问测试期数据。

        Parameters
        ----------
        train_data : pd.DataFrame
            训练期 feature matrix（含 feature/ 列和 label 列）。
        label_col : str, optional
            标签列名。若提供则覆盖构造时的 label_col。
        """
        lgb = _import_lightgbm()

        effective_label_col = label_col or self._label_col

        # 提取特征列（排除标签列）
        feature_cols = self._resolve_feature_cols(train_data, exclude_col=effective_label_col)
        self._fitted_feature_cols = feature_cols

        # 提取训练数据
        X = train_data[feature_cols]
        y = train_data[effective_label_col]

        # 删除标签缺失行
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) == 0:
            logger.warning("LightGBMModel.fit: 训练数据为空（标签全为 NaN）")
            self._model = None
            return

        # 处理特征中的 NaN：LightGBM 原生支持 NaN，无需填充
        # 处理 Inf：替换为 NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        # 构建 LightGBM 参数
        params = {
            "objective": "regression",
            "metric": "mse",
            "n_estimators": self._n_estimators,
            "learning_rate": self._learning_rate,
            "max_depth": self._max_depth,
            "num_leaves": self._num_leaves,
            "subsample": self._subsample,
            "colsample_bytree": self._colsample_bytree,
            "random_state": self._random_state,
            "verbose": self._verbose,
        }

        logger.info(
            "LightGBMModel.fit: n_samples=%d, n_features=%d, params=%s",
            len(X), len(feature_cols), {k: v for k, v in params.items() if k != "verbose"},
        )

        self._model = lgb.LGBMRegressor(**params)
        self._model.fit(X, y)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """对输入数据生成 prediction score。

        **金融正确性**：predict 不使用 label 列。

        Parameters
        ----------
        data : pd.DataFrame
            输入 feature matrix。

        Returns
        -------
        pd.Series
            prediction score，索引与输入数据一致。
        """
        if self._model is None:
            logger.warning("LightGBMModel.predict: 模型未训练，返回 NaN")
            return pd.Series(np.nan, index=data.index, name="prediction")

        feature_cols = self._fitted_feature_cols or self._resolve_feature_cols(data)
        X = data[feature_cols].replace([np.inf, -np.inf], np.nan)

        pred = self._model.predict(X)
        return pd.Series(pred, index=data.index, name="prediction")

    def get_params(self) -> Dict[str, Any]:
        """返回模型参数。"""
        params: Dict[str, Any] = {
            "model_type": "LightGBMModel",
            "label_col": self._label_col,
            "n_estimators": self._n_estimators,
            "learning_rate": self._learning_rate,
            "max_depth": self._max_depth,
            "num_leaves": self._num_leaves,
            "subsample": self._subsample,
            "colsample_bytree": self._colsample_bytree,
            "random_state": self._random_state,
        }
        if self._feature_cols is not None:
            params["feature_cols"] = self._feature_cols
        if self._fitted_feature_cols is not None:
            params["fitted_n_features"] = len(self._fitted_feature_cols)
        return params

    # -----------------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------------

    def _resolve_feature_cols(self, data: pd.DataFrame, exclude_col: Optional[str] = None) -> List[str]:
        """解析特征列名。

        Parameters
        ----------
        data : pd.DataFrame
            输入数据。
        exclude_col : str, optional
            需要排除的列名（如标签列），fit 时自动排除以避免标签泄漏。

        Raises
        ------
        ValueError
            当无 feature/ 前缀列且未显式指定 feature_cols 时。
        """
        if self._feature_cols is not None:
            cols = self._feature_cols
            if exclude_col is not None and exclude_col in cols:
                cols = [c for c in cols if c != exclude_col]
            return cols

        # 自动检测 "feature/" 前缀列
        feature_cols = [c for c in data.columns if c.startswith("feature/")]

        if not feature_cols:
            raise ValueError(
                "无法自动检测特征列（未找到 'feature/' 前缀列）。"
                "请通过 feature_cols 参数显式指定。"
            )

        logger.info("自动检测到 %d 个特征列", len(feature_cols))
        return feature_cols

    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"LightGBMModel({param_str})"
