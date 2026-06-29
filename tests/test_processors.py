# -*- coding: utf-8 -*-
"""
test_processors.py

processors.py 和 data_handler.py 的单元测试。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.processors import (
    CSRankNorm,
    CSZScoreNorm,
    DropNaFeature,
    DropNaLabel,
    FillNa,
    Processor,
    ProcessInf,
    TrainFitZScore,
    Winsorize,
    apply_processor_chain,
    make_default_infer_processors,
    make_default_learn_processors,
    _auto_feature_cols,
    _cs_rank,
    _cs_zscore,
)

from scripts.common.data_handler import QMTDataHandler


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_multi_index_data(
    n_dates: int = 10,
    n_symbols: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """构造测试用的 (date, symbol) MultiIndex DataFrame。"""
    rng = np.random.RandomState(seed)
    dates = [20240101 + i for i in range(n_dates)]
    symbols = [f"S{i:06d}.SZ" for i in range(n_symbols)]

    idx = pd.MultiIndex.from_product(
        [dates, symbols], names=["date", "symbol"]
    )
    n = len(idx)

    df = pd.DataFrame(
        {
            "feature/f1": rng.randn(n),
            "feature/f2": rng.randn(n) * 10 + 5,
            "label/ret_1d": rng.randn(n) * 0.02,
            "label/ret_5d": rng.randn(n) * 0.05,
        },
        index=idx,
    )
    return df


def _make_data_with_nan_inf() -> pd.DataFrame:
    """构造含 NaN 和 inf 的测试数据。"""
    df = _make_multi_index_data(n_dates=5, n_symbols=2)
    df.iloc[0, 0] = np.nan
    df.iloc[1, 1] = np.inf
    df.iloc[2, 0] = -np.inf
    df.iloc[3, 2] = np.nan  # label NaN
    return df


# ---------------------------------------------------------------------------
# _auto_feature_cols
# ---------------------------------------------------------------------------

class TestAutoFeatureCols:
    def test_basic(self):
        df = _make_multi_index_data()
        cols = _auto_feature_cols(df)
        assert cols == ["feature/f1", "feature/f2"]

    def test_no_features(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert _auto_feature_cols(df) == []


# ---------------------------------------------------------------------------
# FillNa
# ---------------------------------------------------------------------------

class TestFillNa:
    def test_fill_zero(self):
        df = _make_multi_index_data()
        df.iloc[0, 0] = np.nan
        df.iloc[1, 1] = np.nan

        proc = FillNa(0.0)
        result = proc.fit_transform(df, ["feature/f1", "feature/f2"])

        assert result["feature/f1"].iloc[0] == 0.0
        assert result["feature/f2"].iloc[1] == 0.0

    def test_fill_custom_value(self):
        df = _make_multi_index_data()
        df.iloc[0, 0] = np.nan

        proc = FillNa(-1.0)
        result = proc.fit_transform(df, ["feature/f1"])
        assert result["feature/f1"].iloc[0] == -1.0

    def test_no_nan(self):
        df = _make_multi_index_data()
        proc = FillNa(0.0)
        result = proc.fit_transform(df, ["feature/f1"])
        assert result["feature/f1"].equals(df["feature/f1"])


# ---------------------------------------------------------------------------
# ProcessInf
# ---------------------------------------------------------------------------

class TestProcessInf:
    def test_replace_inf(self):
        df = _make_multi_index_data()
        df.iloc[0, 0] = np.inf
        df.iloc[1, 0] = -np.inf

        proc = ProcessInf()
        result = proc.fit_transform(df, ["feature/f1"])

        assert np.isnan(result["feature/f1"].iloc[0])
        assert np.isnan(result["feature/f1"].iloc[1])

    def test_no_inf(self):
        df = _make_multi_index_data()
        proc = ProcessInf()
        result = proc.fit_transform(df, ["feature/f1"])
        assert result["feature/f1"].equals(df["feature/f1"])


# ---------------------------------------------------------------------------
# DropNaFeature
# ---------------------------------------------------------------------------

class TestDropNaFeature:
    def test_drop_any(self):
        df = _make_multi_index_data()
        df.iloc[0, 0] = np.nan  # feature/f1
        df.iloc[1, 1] = np.nan  # feature/f2

        proc = DropNaFeature(how="any")
        result = proc.fit_transform(df, ["feature/f1", "feature/f2"])

        # 两行被删除
        assert len(result) == len(df) - 2

    def test_drop_all(self):
        df = _make_multi_index_data()
        df.iloc[0, 0] = np.nan  # only f1
        df.iloc[1, 0] = np.nan
        df.iloc[1, 1] = np.nan  # both f1 and f2

        proc = DropNaFeature(how="all")
        result = proc.fit_transform(df, ["feature/f1", "feature/f2"])

        # 只有第 1 行（index=1）被删除
        assert len(result) == len(df) - 1


# ---------------------------------------------------------------------------
# DropNaLabel
# ---------------------------------------------------------------------------

class TestDropNaLabel:
    def test_drop_label_nan(self):
        df = _make_multi_index_data()
        df.iloc[0, 2] = np.nan  # label/ret_1d

        proc = DropNaLabel(label_cols=["label/ret_1d"])
        result = proc.fit_transform(df, ["feature/f1"])

        assert len(result) == len(df) - 1

    def test_no_drop(self):
        df = _make_multi_index_data()
        proc = DropNaLabel(label_cols=["label/ret_1d"])
        result = proc.fit_transform(df, ["feature/f1"])
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# CSZScoreNorm
# ---------------------------------------------------------------------------

class TestCSZScoreNorm:
    def test_basic_normalization(self):
        df = _make_multi_index_data(n_dates=5, n_symbols=3)
        proc = CSZScoreNorm()
        result = proc.fit_transform(df, ["feature/f1"])

        # 每日截面应有 mean≈0
        daily_mean = result["feature/f1"].groupby(level=0).mean()
        assert all(abs(daily_mean) < 1e-10)

    def test_constant_column(self):
        df = _make_multi_index_data(n_dates=3, n_symbols=2)
        df["feature/f1"] = 5.0  # 常数列

        proc = CSZScoreNorm()
        result = proc.fit_transform(df, ["feature/f1"])

        # std=0 → 全 0
        assert all(result["feature/f1"] == 0.0)

    def test_no_cross_date_leakage(self):
        """验证截面标准化不跨日期泄漏。"""
        df = _make_multi_index_data(n_dates=10, n_symbols=3)

        proc = CSZScoreNorm()
        result = proc.fit_transform(df, ["feature/f1"])

        # 每日的 z-score 应该独立于其他日期
        dates = result.index.get_level_values(0).unique()
        for d in dates:
            day_data = result.loc[d, "feature/f1"]
            assert abs(day_data.mean()) < 1e-10


# ---------------------------------------------------------------------------
# CSRankNorm
# ---------------------------------------------------------------------------

class TestCSRankNorm:
    def test_basic_ranking(self):
        df = _make_multi_index_data(n_dates=5, n_symbols=3)
        proc = CSRankNorm()
        result = proc.fit_transform(df, ["feature/f1"])

        # 排名应在 [0, 1] 之间
        assert result["feature/f1"].min() >= 0.0
        assert result["feature/f1"].max() <= 1.0

    def test_rank_per_date(self):
        """每日截面独立排名。"""
        df = _make_multi_index_data(n_dates=5, n_symbols=3)
        proc = CSRankNorm()
        result = proc.fit_transform(df, ["feature/f1"])

        dates = result.index.get_level_values(0).unique()
        for d in dates:
            day_ranks = result.loc[d, "feature/f1"]
            # 3 只股票的排名应为 1/3, 2/3, 1.0
            assert set(day_ranks) == {1 / 3, 2 / 3, 1.0}


# ---------------------------------------------------------------------------
# Winsorize
# ---------------------------------------------------------------------------

class TestWinsorize:
    def test_clip_extremes(self):
        df = _make_multi_index_data(n_dates=100, n_symbols=10, seed=42)
        proc = Winsorize(lower_quantile=0.05, upper_quantile=0.95)
        result = proc.fit_transform(df, ["feature/f1"])

        # 原始数据应有超出范围的值
        orig_min = df["feature/f1"].min()
        orig_max = df["feature/f1"].max()
        proc_min = result["feature/f1"].min()
        proc_max = result["feature/f1"].max()

        # 处理后范围应缩小
        assert proc_min >= orig_min
        assert proc_max <= orig_max

    def test_fit_on_train_process_on_test(self):
        """验证 fit 在训练期，process 应用到测试期。"""
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)

        # 训练期：前 10 天
        train_dates = df.index.get_level_values(0).unique()[:10]
        train = df.loc[train_dates]

        proc = Winsorize(lower_quantile=0.1, upper_quantile=0.9)
        proc.fit(train, ["feature/f1"])

        # 处理测试期
        test_dates = df.index.get_level_values(0).unique()[10:]
        test = df.loc[test_dates]
        result = proc.process(test, ["feature/f1"])

        # 应该不报错，且结果有效
        assert len(result) == len(test)
        assert not result["feature/f1"].isna().all()


# ---------------------------------------------------------------------------
# TrainFitZScore
# ---------------------------------------------------------------------------

class TestTrainFitZScore:
    def test_fit_on_train_transform_test(self):
        """训练期 fit 的参数应用到测试期。"""
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        dates = df.index.get_level_values(0).unique()
        train = df.loc[dates[:10]]
        test = df.loc[dates[10:]]

        proc = TrainFitZScore()
        proc.fit(train, ["feature/f1"])

        # 训练期参数
        params = proc.get_params()
        train_mean = params["fit_params"]["stats"]["feature/f1"]["mean"]
        train_std = params["fit_params"]["stats"]["feature/f1"]["std"]

        # 应用到测试期
        result = proc.process(test, ["feature/f1"])

        # 手动验证
        expected = (test["feature/f1"] - train_mean) / train_std
        pd.testing.assert_series_equal(
            result["feature/f1"], expected, check_names=False, atol=1e-10
        )

    def test_no_test_data_leakage(self):
        """确保 test 数据不影响 fit 参数。"""
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        dates = df.index.get_level_values(0).unique()
        train = df.loc[dates[:10]]

        proc = TrainFitZScore()
        proc.fit(train, ["feature/f1"])

        params1 = proc.get_params()

        # 用不同的 test 数据应该不影响参数
        test = df.loc[dates[10:]]
        proc.process(test, ["feature/f1"])
        params2 = proc.get_params()

        assert params1 == params2

    def test_constant_column(self):
        """常数列 std=0 → 输出全 0。"""
        df = _make_multi_index_data(n_dates=10, n_symbols=3)
        df["feature/f1"] = 5.0

        proc = TrainFitZScore()
        result = proc.fit_transform(df, ["feature/f1"])

        assert all(result["feature/f1"] == 0.0)


# ---------------------------------------------------------------------------
# apply_processor_chain
# ---------------------------------------------------------------------------

class TestApplyProcessorChain:
    def test_chain(self):
        df = _make_data_with_nan_inf()
        procs = [ProcessInf(), FillNa(0.0), CSZScoreNorm()]
        feat_cols = ["feature/f1", "feature/f2"]

        result = apply_processor_chain(df, procs, feat_cols)

        # 无 NaN、无 inf
        assert not result[feat_cols].isna().any().any()
        assert not np.isinf(result[feat_cols]).any().any()

    def test_fit_data_separate(self):
        """fit_data 与 data 不同时，用 fit_data 的参数处理 data。"""
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        dates = df.index.get_level_values(0).unique()
        train = df.loc[dates[:10]]
        test = df.loc[dates[10:]]

        procs = [TrainFitZScore()]
        result = apply_processor_chain(
            test, procs, ["feature/f1"], fit_data=train
        )

        # 应该使用训练期参数
        assert len(result) == len(test)


# ---------------------------------------------------------------------------
# make_default_infer_processors / make_default_learn_processors
# ---------------------------------------------------------------------------

class TestDefaultProcessors:
    def test_infer_chain(self):
        procs = make_default_infer_processors()
        assert len(procs) == 3
        assert isinstance(procs[0], ProcessInf)
        assert isinstance(procs[1], FillNa)
        assert isinstance(procs[2], CSZScoreNorm)

    def test_learn_chain(self):
        procs = make_default_learn_processors()
        assert len(procs) == 4
        assert isinstance(procs[0], ProcessInf)
        assert isinstance(procs[1], FillNa)
        assert isinstance(procs[2], DropNaLabel)
        assert isinstance(procs[3], CSZScoreNorm)


# ---------------------------------------------------------------------------
# QMTDataHandler
# ---------------------------------------------------------------------------

class TestQMTDataHandler:
    def test_from_dataframe(self):
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        handler = QMTDataHandler(df)

        raw = handler.load_raw()
        assert raw.shape == df.shape

    def test_feature_label_cols(self):
        df = _make_multi_index_data()
        handler = QMTDataHandler(df)

        assert handler.feature_cols() == ["feature/f1", "feature/f2"]
        assert handler.label_cols() == ["label/ret_1d", "label/ret_5d"]

    def test_get_year_mask(self):
        df = _make_multi_index_data(n_dates=10, n_symbols=2)
        # 修改日期为跨年
        dates = df.index.get_level_values(0)
        new_dates = [20230101 + i if i < 5 else 20240101 + i for i in range(10)]
        df = df.reset_index()
        df["date"] = new_dates * 2
        df = df.set_index(["date", "symbol"])

        handler = QMTDataHandler(df)

        mask_2023 = handler.get_year_mask(2023)
        assert mask_2023.sum() == 10  # 5 dates * 2 symbols

        mask_2024 = handler.get_year_mask(2024)
        assert mask_2024.sum() == 10

    def test_get_segment_data(self):
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        # 设置日期为 2023 和 2024
        dates = df.index.get_level_values(0).unique()
        new_dates = [20230101 + i for i in range(10)] + [20240101 + i for i in range(10)]
        mapping = dict(zip(dates, new_dates))
        df = df.reset_index()
        df["date"] = df["date"].map(mapping)
        df = df.set_index(["date", "symbol"])

        handler = QMTDataHandler(df)

        train = handler.get_segment_data("train", 2024)
        test = handler.get_segment_data("test", 2024)

        # 训练期：2023 年，10 dates * 3 symbols = 30
        assert len(train) == 30
        # 测试期：2024 年，10 dates * 3 symbols = 30
        assert len(test) == 30

    def test_process_infer(self):
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        dates = df.index.get_level_values(0).unique()
        new_dates = [20230101 + i for i in range(10)] + [20240101 + i for i in range(10)]
        mapping = dict(zip(dates, new_dates))
        df = df.reset_index()
        df["date"] = df["date"].map(mapping)
        df = df.set_index(["date", "symbol"])

        handler = QMTDataHandler(df)

        # process_infer 应返回特征列，不含标签
        dk_i = handler.process_infer("test", 2024)
        assert list(dk_i.columns) == ["feature/f1", "feature/f2"]
        assert len(dk_i) == 30

    def test_process_learn(self):
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        # 添加一些 NaN label
        df.iloc[0, 2] = np.nan  # label/ret_1d
        df.iloc[1, 3] = np.nan  # label/ret_5d

        dates = df.index.get_level_values(0).unique()
        new_dates = [20230101 + i for i in range(10)] + [20240101 + i for i in range(10)]
        mapping = dict(zip(dates, new_dates))
        df = df.reset_index()
        df["date"] = df["date"].map(mapping)
        df = df.set_index(["date", "symbol"])

        handler = QMTDataHandler(df)

        # process_learn 应返回特征+标签，删除标签 NaN 的行
        dk_l = handler.process_learn("train", 2024)
        assert "label/ret_1d" in dk_l.columns
        assert "label/ret_5d" in dk_l.columns
        # 标签 NaN 的行应被删除
        assert not dk_l[["label/ret_1d", "label/ret_5d"]].isna().any(axis=1).any()

    def test_process_learn_rejects_test_segment(self):
        """process_learn(segment='test') 应抛出 ValueError。"""
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        dates = df.index.get_level_values(0).unique()
        new_dates = [20230101 + i for i in range(10)] + [20240101 + i for i in range(10)]
        mapping = dict(zip(dates, new_dates))
        df = df.reset_index()
        df["date"] = df["date"].map(mapping)
        df = df.set_index(["date", "symbol"])

        handler = QMTDataHandler(df)

        with pytest.raises(ValueError, match="只允许 segment='train'"):
            handler.process_learn("test", 2024)

    def test_process_learn_rejects_all_segment(self):
        """process_learn(segment='all') 应抛出 ValueError。"""
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        dates = df.index.get_level_values(0).unique()
        new_dates = [20230101 + i for i in range(10)] + [20240101 + i for i in range(10)]
        mapping = dict(zip(dates, new_dates))
        df = df.reset_index()
        df["date"] = df["date"].map(mapping)
        df = df.set_index(["date", "symbol"])

        handler = QMTDataHandler(df)

        with pytest.raises(ValueError, match="只允许 segment='train'"):
            handler.process_learn("all", 2024)

    def test_process_learn_accepts_train_segment(self):
        """process_learn(segment='train') 不应抛出错误。"""
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        dates = df.index.get_level_values(0).unique()
        new_dates = [20230101 + i for i in range(10)] + [20240101 + i for i in range(10)]
        mapping = dict(zip(dates, new_dates))
        df = df.reset_index()
        df["date"] = df["date"].map(mapping)
        df = df.set_index(["date", "symbol"])

        handler = QMTDataHandler(df)

        # 不应抛出错误
        dk_l = handler.process_learn("train", 2024)
        assert len(dk_l) > 0

    def test_prepare_walk_forward(self):
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        dates = df.index.get_level_values(0).unique()
        new_dates = [20230101 + i for i in range(10)] + [20240101 + i for i in range(10)]
        mapping = dict(zip(dates, new_dates))
        df = df.reset_index()
        df["date"] = df["date"].map(mapping)
        df = df.set_index(["date", "symbol"])

        handler = QMTDataHandler(df)
        train_dk_l, test_dk_i, train_feat, test_feat = handler.prepare_walk_forward(2024)

        # 训练数据含 label
        assert "label/ret_1d" in train_dk_l.columns
        # 测试数据不含 label
        assert not any(c.startswith("label/") for c in test_dk_i.columns)
        # 特征只有 feature/ 列
        assert all(c.startswith("feature/") for c in train_feat.columns)
        assert all(c.startswith("feature/") for c in test_feat.columns)

    def test_meta_cols(self):
        """meta_cols 返回非 feature/ 非 label/ 的列名。"""
        df = _make_multi_index_data(n_dates=10, n_symbols=3)
        # 添加一列 meta 数据
        df["meta/market"] = "SH"

        handler = QMTDataHandler(df)
        meta = handler.meta_cols()

        assert "meta/market" in meta
        # feature 和 label 列不应出现在 meta 中
        for c in meta:
            assert not c.startswith("feature/")
            assert not c.startswith("label/")

    def test_meta_cols_empty(self):
        """无 meta 列时返回空列表。"""
        df = _make_multi_index_data(n_dates=10, n_symbols=3)
        handler = QMTDataHandler(df)

        # 默认数据只有 feature/ 和 label/ 列
        assert handler.meta_cols() == []

    def test_describe(self):
        df = _make_multi_index_data(n_dates=10, n_symbols=3)
        handler = QMTDataHandler(df)
        desc = handler.describe()

        assert desc["shape"] == [30, 4]
        assert desc["feature_cols"] == 2
        assert desc["label_cols"] == 2
        assert desc["n_symbols"] == 3
        assert desc["n_dates"] == 10


# ---------------------------------------------------------------------------
# 金融正确性检查
# ---------------------------------------------------------------------------

class TestFinancialCorrectness:
    """显式检查金融正确性约束。"""

    def test_no_future_function_in_train_fit(self):
        """Processor fit 只使用训练期数据，不泄漏测试期。"""
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        dates = df.index.get_level_values(0).unique()
        train = df.loc[dates[:10]]
        test = df.loc[dates[10:]]

        # TrainFitZScore
        proc = TrainFitZScore()
        proc.fit(train, ["feature/f1"])
        params_before = proc.get_params()

        # 处理测试期
        proc.process(test, ["feature/f1"])
        params_after = proc.get_params()

        # 参数不应变化
        assert params_before == params_after

    def test_cs_no_cross_date_leakage(self):
        """截面标准化不跨日期。"""
        df = _make_multi_index_data(n_dates=10, n_symbols=3, seed=42)

        proc = CSZScoreNorm()
        result = proc.fit_transform(df, ["feature/f1"])

        dates = result.index.get_level_values(0).unique()
        for d in dates:
            day_vals = result.loc[d, "feature/f1"]
            # 每日截面 mean ≈ 0
            assert abs(day_vals.mean()) < 1e-10

    def test_cs_rank_independent_per_date(self):
        """截面排名每日独立。"""
        df = _make_multi_index_data(n_dates=10, n_symbols=3, seed=42)

        proc = CSRankNorm()
        result = proc.fit_transform(df, ["feature/f1"])

        dates = result.index.get_level_values(0).unique()
        for d in dates:
            day_ranks = result.loc[d, "feature/f1"]
            # 3 只股票排名应为 1/3, 2/3, 1.0
            assert set(day_ranks) == {1 / 3, 2 / 3, 1.0}

    def test_handler_no_label_in_infer(self):
        """DK_I 不包含标签列。"""
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        dates = df.index.get_level_values(0).unique()
        new_dates = [20230101 + i for i in range(10)] + [20240101 + i for i in range(10)]
        mapping = dict(zip(dates, new_dates))
        df = df.reset_index()
        df["date"] = df["date"].map(mapping)
        df = df.set_index(["date", "symbol"])

        handler = QMTDataHandler(df)
        dk_i = handler.process_infer("test", 2024)

        assert not any(c.startswith("label/") for c in dk_i.columns)

    def test_handler_drop_label_nan_in_learn(self):
        """DK_L 删除标签缺失的行。"""
        df = _make_multi_index_data(n_dates=20, n_symbols=3, seed=42)
        # 设置一些标签 NaN
        df.iloc[0, 2] = np.nan  # label/ret_1d
        df.iloc[5, 2] = np.nan
        df.iloc[10, 3] = np.nan  # label/ret_5d

        dates = df.index.get_level_values(0).unique()
        new_dates = [20230101 + i for i in range(10)] + [20240101 + i for i in range(10)]
        mapping = dict(zip(dates, new_dates))
        df = df.reset_index()
        df["date"] = df["date"].map(mapping)
        df = df.set_index(["date", "symbol"])

        handler = QMTDataHandler(df)
        dk_l = handler.process_learn("train", 2024)

        # 标签 NaN 的行应被删除
        assert not dk_l[["label/ret_1d", "label/ret_5d"]].isna().any(axis=1).any()

    def test_rolling_uses_only_history(self):
        """滚动窗口只使用历史数据（由 feature_expression 层保证）。"""
        # 这个测试验证 rolling 的行为
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        rolling_mean = s.rolling(3).mean()

        # 第 0、1 个值应为 NaN（窗口不足）
        assert np.isnan(rolling_mean.iloc[0])
        assert np.isnan(rolling_mean.iloc[1])
        # 第 2 个值应为 (1+2+3)/3 = 2.0
        assert rolling_mean.iloc[2] == 2.0
        # 第 3 个值应为 (2+3+4)/3 = 3.0
        assert rolling_mean.iloc[3] == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
