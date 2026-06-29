# -*- coding: utf-8 -*-
"""
data_handler.py

轻量 DataHandler，借鉴 Qlib DataHandlerLP / DatasetH 的设计。

为规则策略、信号评价和未来 ML 模型提供统一的数据加载、特征构建、标签构建和
train/test 切片接口。

核心设计：
  - QMTDataHandler：从 feature matrix 出发，提供 DK_I（推理数据）和 DK_L（学习数据）。
  - 按年度切分 train/test segment，服务现有 walk-forward。
  - 处理器链在训练期 fit，不泄漏测试期。

数据约定：
  - 输入：feature matrix（(date, symbol) MultiIndex parquet）
  - 输出：DK_I（features + meta，不含 label）和 DK_L（features + label）
  - segment 划分：按年度切分，fit 只在训练期执行

使用方式：
  handler = QMTDataHandler(feature_matrix_path)
  dk_i = handler.process_infer(segment="test", test_year=2024)
  dk_l = handler.process_learn(segment="train", test_year=2024)
  train, test = handler.prepare_walk_forward(test_year=2024)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from scripts.common.processors import (
    CSRankNorm,
    CSZScoreNorm,
    DropNaFeature,
    DropNaLabel,
    FillNa,
    Processor,
    ProcessInf,
    Winsorize,
    apply_processor_chain,
    make_default_infer_processors,
    make_default_learn_processors,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QMTDataHandler
# ---------------------------------------------------------------------------

class QMTDataHandler:
    """轻量 DataHandler，管理 feature matrix 的加载、处理和切片。

    Parameters
    ----------
    feature_matrix : pd.DataFrame or Path
        特征矩阵，(date, symbol) MultiIndex。或 parquet 文件路径。
    infer_processors : list of Processor, optional
        推理处理器链。默认使用 make_default_infer_processors()。
    learn_processors : list of Processor, optional
        学习处理器链。默认使用 make_default_learn_processors()。
    """

    def __init__(
        self,
        feature_matrix: Union[pd.DataFrame, Path],
        infer_processors: Optional[List[Processor]] = None,
        learn_processors: Optional[List[Processor]] = None,
    ) -> None:
        if isinstance(feature_matrix, (str, Path)):
            self._path = Path(feature_matrix)
            self._raw: Optional[pd.DataFrame] = None
        else:
            self._path = None
            self._raw = feature_matrix

        self._infer_processors = infer_processors
        self._learn_processors = learn_processors

        # 缓存
        self._feature_cols: Optional[List[str]] = None
        self._label_cols: Optional[List[str]] = None

    # -----------------------------------------------------------------------
    # 数据加载
    # -----------------------------------------------------------------------

    def load_raw(self) -> pd.DataFrame:
        """加载原始特征矩阵。

        Returns
        -------
        pd.DataFrame
            (date, symbol) MultiIndex 的原始特征矩阵。
        """
        if self._raw is None:
            if self._path is None:
                raise ValueError("未指定数据源")
            logger.info("加载特征矩阵：%s", self._path)
            self._raw = pd.read_parquet(self._path)
            logger.info("加载完成：shape=%s", self._raw.shape)
        return self._raw

    # -----------------------------------------------------------------------
    # 列分类
    # -----------------------------------------------------------------------

    def feature_cols(self) -> List[str]:
        """获取特征列名列表（feature/ 前缀）。"""
        if self._feature_cols is None:
            raw = self.load_raw()
            self._feature_cols = [c for c in raw.columns if c.startswith("feature/")]
        return self._feature_cols

    def label_cols(self) -> List[str]:
        """获取标签列名列表（label/ 前缀）。"""
        if self._label_cols is None:
            raw = self.load_raw()
            self._label_cols = [c for c in raw.columns if c.startswith("label/")]
        return self._label_cols

    def meta_cols(self) -> List[str]:
        """获取元数据列名列表（非 feature/ 非 label/）。"""
        raw = self.load_raw()
        return [c for c in raw.columns if not c.startswith("feature/") and not c.startswith("label/")]

    # -----------------------------------------------------------------------
    # 年度切分
    # -----------------------------------------------------------------------

    def get_year_mask(self, year: int) -> pd.Series:
        """获取指定年度的布尔掩码。

        Parameters
        ----------
        year : int
            年份，如 2024。

        Returns
        -------
        pd.Series
            布尔索引，True 表示该行属于指定年度。
        """
        raw = self.load_raw()
        dates = raw.index.get_level_values(0)
        return (dates // 10000) == year

    def get_segment_data(
        self,
        segment: str,
        test_year: int,
    ) -> pd.DataFrame:
        """按 segment 获取数据。

        Parameters
        ----------
        segment : str
            'train'、'test'、'all'。
        test_year : int
            测试年份。train = 年份 < test_year，test = 年份 == test_year。

        Returns
        -------
        pd.DataFrame
            切片后的数据。
        """
        raw = self.load_raw()

        if segment == "all":
            return raw
        elif segment == "train":
            mask = self.get_year_mask(test_year)
            # 训练期：test_year 之前的所有数据
            dates = raw.index.get_level_values(0)
            train_mask = (dates // 10000) < test_year
            return raw.loc[train_mask]
        elif segment == "test":
            mask = self.get_year_mask(test_year)
            return raw.loc[mask]
        else:
            raise ValueError(f"未知 segment: {segment}，应为 'train'、'test' 或 'all'")

    # -----------------------------------------------------------------------
    # 推理数据 (DK_I)
    # -----------------------------------------------------------------------

    def process_infer(
        self,
        segment: str = "test",
        test_year: int = 2024,
        processors: Optional[List[Processor]] = None,
    ) -> pd.DataFrame:
        """生成推理数据（DK_I）。

        DK_I 包含特征列，不包含标签列。
        处理器在训练期 fit，然后应用到目标 segment。

        **金融正确性**：
        - 推理数据不依赖 label，避免标签泄漏。
        - 处理器参数在训练期学习，不使用测试期数据。

        Parameters
        ----------
        segment : str
            目标 segment，默认 'test'。
        test_year : int
            测试年份。
        processors : list of Processor, optional
            自定义处理器链。

        Returns
        -------
        pd.DataFrame
            处理后的推理数据。
        """
        procs = processors or self._infer_processors or make_default_infer_processors()
        feat_cols = self.feature_cols()

        # 获取训练期数据用于 fit
        train_data = self.get_segment_data("train", test_year)
        target_data = self.get_segment_data(segment, test_year)

        if train_data.empty:
            logger.warning("训练期数据为空（test_year=%d），跳过 fit", test_year)
            return target_data[feat_cols] if feat_cols else target_data

        # 只保留特征列
        train_feats = train_data[feat_cols] if feat_cols else train_data
        target_feats = target_data[feat_cols] if feat_cols else target_data

        # 在训练期 fit，然后处理目标数据
        result = apply_processor_chain(target_feats, procs, feat_cols, fit_data=train_feats)
        return result

    # -----------------------------------------------------------------------
    # 学习数据 (DK_L)
    # -----------------------------------------------------------------------

    def process_learn(
        self,
        segment: str = "train",
        test_year: int = 2024,
        processors: Optional[List[Processor]] = None,
    ) -> pd.DataFrame:
        """生成学习数据（DK_L）。

        DK_L 包含特征列和标签列。
        处理器在训练期 fit，然后应用到目标 segment。
        学习数据会删除标签缺失的行。

        **金融正确性**：
        - 学习数据包含 label，但只用于训练。
        - DropNaLabel 删除标签缺失的行。
        - 处理器参数只在训练期学习。

        Parameters
        ----------
        segment : str
            目标 segment，只允许 'train'。传入其他值会抛出 ValueError，
            因为学习数据的处理器必须在训练期 fit，不能在测试期 fit（数据泄漏）。
        test_year : int
            测试年份。
        processors : list of Processor, optional
            自定义处理器链。

        Returns
        -------
        pd.DataFrame
            处理后的学习数据（含 label）。
        """
        if segment != "train":
            raise ValueError(
                f"process_learn() 只允许 segment='train'，收到 segment='{segment}'。"
                f"学习数据必须在训练期 fit，不能在测试期 fit（会导致数据泄漏）。"
                f"如需测试期推理数据，请使用 process_infer()。"
            )

        procs = processors or self._learn_processors or make_default_learn_processors()
        feat_cols = self.feature_cols()
        label_cols = self.label_cols()
        all_cols = feat_cols + label_cols

        # 获取数据
        data = self.get_segment_data(segment, test_year)

        if data.empty:
            return data

        # 保留特征和标签列
        available = [c for c in all_cols if c in data.columns]
        subset = data[available]

        # 学习数据在自身 fit（因为是训练期）
        result = apply_processor_chain(subset, procs, feat_cols)
        return result

    # -----------------------------------------------------------------------
    # Walk-Forward 便捷接口
    # -----------------------------------------------------------------------

    def prepare_walk_forward(
        self,
        test_year: int,
        infer_processors: Optional[List[Processor]] = None,
        learn_processors: Optional[List[Processor]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """准备 walk-forward 的训练和测试数据。

        Returns
        -------
        Tuple[train_dk_l, test_dk_i, train_features, test_features]
            - train_dk_l：训练期学习数据（含 label）
            - test_dk_i：测试期推理数据（不含 label）
            - train_features：训练期特征（仅 feature/ 列）
            - test_features：测试期特征（仅 feature/ 列）
        """
        train_dk_l = self.process_learn("train", test_year, learn_processors)
        test_dk_i = self.process_infer("test", test_year, infer_processors)

        feat_cols = self.feature_cols()
        train_features = train_dk_l[feat_cols] if feat_cols else train_dk_l
        test_features = test_dk_i

        return train_dk_l, test_dk_i, train_features, test_features

    # -----------------------------------------------------------------------
    # 信息
    # -----------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """返回数据集的基本描述。"""
        raw = self.load_raw()
        dates = raw.index.get_level_values(0)
        symbols = raw.index.get_level_values(1)

        return {
            "shape": list(raw.shape),
            "feature_cols": len(self.feature_cols()),
            "label_cols": len(self.label_cols()),
            "date_range": [int(dates.min()), int(dates.max())] if len(dates) > 0 else [],
            "n_symbols": symbols.nunique(),
            "n_dates": dates.nunique(),
            "columns": list(raw.columns),
        }
