# -*- coding: utf-8 -*-
"""
test_alpha_models.py

tests for scripts.common.models (AlphaModel base, SimpleRuleModel).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.common.models.base import AlphaModel
from scripts.common.models.rule_model import SimpleRuleModel
from scripts.common.feature_expression import (
    Field,
    PctChange,
    RollingMean,
    ZScore,
    normalize_zscore,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """构造测试用价格 DataFrame。"""
    rng = np.random.RandomState(seed)
    dates = [20240101 + i for i in range(n)]
    close = 10.0 + np.cumsum(rng.randn(n) * 0.1)
    volume = rng.randint(1000, 10000, size=n).astype(float)
    return pd.DataFrame(
        {"close": close, "volume": volume},
        index=pd.Index(dates, name="date"),
    )


def _make_feature_matrix(n_dates: int = 60, n_symbols: int = 3, seed: int = 42) -> pd.DataFrame:
    """构造 MultiIndex feature matrix。"""
    rng = np.random.RandomState(seed)
    dates = [20240101 + i for i in range(n_dates)]
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    n = len(idx)
    return pd.DataFrame(
        {
            "feature/close": 10.0 + np.cumsum(rng.randn(n) * 0.1),
            "feature/volume": rng.randint(1000, 10000, size=n).astype(float),
            "feature/reversal_10d": rng.randn(n),
            "label/ret_1d": rng.randn(n) * 0.01,
            "label/ret_5d": rng.randn(n) * 0.02,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# TestAlphaModelBase
# ---------------------------------------------------------------------------

class TestAlphaModelBase:
    """AlphaModel 抽象基类测试。"""

    def test_cannot_instantiate(self):
        """抽象类不能直接实例化。"""
        with pytest.raises(TypeError):
            AlphaModel()

    def test_subclass_must_implement_all(self):
        """子类必须实现所有抽象方法。"""

        class IncompleteModel(AlphaModel):
            def fit(self, train_data, label_col=None):
                pass

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_subclass_complete(self):
        """完整子类可以实例化。"""

        class CompleteModel(AlphaModel):
            def fit(self, train_data, label_col=None):
                pass

            def predict(self, data):
                return pd.Series(0.0, index=data.index)

            def get_params(self):
                return {"model_type": "CompleteModel"}

        model = CompleteModel()
        assert model.get_params()["model_type"] == "CompleteModel"

    def test_repr(self):
        """repr 包含类名和参数。"""

        class MyModel(AlphaModel):
            def fit(self, train_data, label_col=None):
                pass

            def predict(self, data):
                return pd.Series(0.0, index=data.index)

            def get_params(self):
                return {"model_type": "MyModel", "alpha": 0.5}

        model = MyModel()
        r = repr(model)
        assert "MyModel" in r
        assert "alpha" in r


# ---------------------------------------------------------------------------
# TestSimpleRuleModel
# ---------------------------------------------------------------------------

class TestSimpleRuleModel:
    """SimpleRuleModel 测试。"""

    def test_expression_mode(self):
        """表达式模式：评估表达式输出 score。"""
        expr = Field("close")
        model = SimpleRuleModel(expression=expr, zscore=False)
        df = _make_price_df(50)
        pred = model.predict(df)
        assert isinstance(pred, pd.Series)
        assert len(pred) == 50
        pd.testing.assert_series_equal(pred, df["close"], check_names=False)

    def test_expression_zscore(self):
        """表达式模式 + ZScore 标准化。"""
        expr = Field("close")
        model = SimpleRuleModel(expression=expr, zscore=True)
        df = _make_price_df(50)
        pred = model.predict(df)
        expected = normalize_zscore(df["close"])
        pd.testing.assert_series_equal(pred, expected, check_names=False)

    def test_score_col_mode(self):
        """预计算列模式。"""
        model = SimpleRuleModel(score_col="feature/reversal_10d", zscore=False)
        fm = _make_feature_matrix()
        pred = model.predict(fm)
        expected = fm["feature/reversal_10d"]
        pd.testing.assert_series_equal(pred, expected, check_names=False)

    def test_score_col_zscore(self):
        """预计算列模式 + ZScore。"""
        model = SimpleRuleModel(score_col="feature/reversal_10d", zscore=True)
        fm = _make_feature_matrix()
        pred = model.predict(fm)
        expected = normalize_zscore(fm["feature/reversal_10d"])
        pd.testing.assert_series_equal(pred, expected, check_names=False)

    def test_score_col_missing_raises(self):
        """预计算列不存在时抛出 KeyError。"""
        model = SimpleRuleModel(score_col="feature/nonexistent")
        fm = _make_feature_matrix()
        with pytest.raises(KeyError, match="feature/nonexistent"):
            model.predict(fm)

    def test_must_provide_one(self):
        """expression 和 score_col 都不提供时报错。"""
        with pytest.raises(ValueError, match="必须提供"):
            SimpleRuleModel()

    def test_cannot_provide_both(self):
        """同时提供 expression 和 score_col 报错。"""
        with pytest.raises(ValueError, match="不能同时提供"):
            SimpleRuleModel(expression=Field("close"), score_col="x")

    def test_fit_is_noop(self):
        """fit 不改变 predict 结果。"""
        expr = Field("close")
        model = SimpleRuleModel(expression=expr, zscore=False)
        df = _make_price_df(50)
        before = model.predict(df)
        model.fit(df)
        after = model.predict(df)
        pd.testing.assert_series_equal(before, after)

    def test_fit_sets_fitted(self):
        """fit 设置 _fitted 标志。"""
        model = SimpleRuleModel(score_col="close")
        assert not model._fitted
        model.fit(_make_price_df())
        assert model._fitted

    def test_predict_signal(self):
        """predict_signal 返回二值信号。"""
        model = SimpleRuleModel(score_col="close", zscore=False)
        df = _make_price_df(50)
        signal = model.predict_signal(df)
        assert set(signal.unique()).issubset({0, 1})

    def test_predict_signal_threshold(self):
        """signal_threshold 控制信号阈值。"""
        model = SimpleRuleModel(score_col="close", zscore=False, signal_threshold=15.0)
        df = _make_price_df(50)
        signal = model.predict_signal(df)
        score = df["close"]
        assert (signal == (score > 15.0).astype(int)).all()

    def test_get_params_expression(self):
        """get_params 包含 expression 信息。"""
        expr = Field("close")
        model = SimpleRuleModel(expression=expr, zscore=True)
        params = model.get_params()
        assert params["model_type"] == "SimpleRuleModel"
        assert params["zscore"] is True
        assert "expression" in params
        assert "close" in params["expression"]

    def test_get_params_score_col(self):
        """get_params 包含 score_col。"""
        model = SimpleRuleModel(score_col="feature/x")
        params = model.get_params()
        assert params["score_col"] == "feature/x"
        assert "expression" not in params

    def test_repr_expression(self):
        """repr 表达式模式。"""
        model = SimpleRuleModel(expression=Field("close"))
        r = repr(model)
        assert "SimpleRuleModel" in r
        assert "close" in r

    def test_repr_score_col(self):
        """repr 预计算列模式。"""
        model = SimpleRuleModel(score_col="feature/x")
        r = repr(model)
        assert "SimpleRuleModel" in r
        assert "feature/x" in r

    def test_complex_expression(self):
        """复合表达式评估。"""
        expr = ZScore(-PctChange(Field("close"), 10))
        model = SimpleRuleModel(expression=expr, zscore=False)
        df = _make_price_df(50)
        pred = model.predict(df)
        assert isinstance(pred, pd.Series)
        assert len(pred) == 50
        # ZScore 输出应近似零均值（忽略 NaN）
        valid = pred.dropna()
        if len(valid) > 5:
            assert abs(valid.mean()) < 1.0

    def test_expression_with_feature_matrix(self):
        """表达式在 MultiIndex feature matrix 上评估。"""
        expr = Field("feature/close")
        model = SimpleRuleModel(expression=expr, zscore=False)
        fm = _make_feature_matrix()
        pred = model.predict(fm)
        assert len(pred) == len(fm)
        pd.testing.assert_series_equal(pred, fm["feature/close"], check_names=False)


# ---------------------------------------------------------------------------
# TestSimpleRuleModelIntegration
# ---------------------------------------------------------------------------

class TestSimpleRuleModelIntegration:
    """SimpleRuleModel 与 Alpha v6 信号一致性测试。"""

    def test_short_term_reversal_consistency(self):
        """short_term_reversal 表达式模型与手动计算一致。"""
        df = _make_price_df(100)
        reversal_window = 10
        expr = ZScore(-PctChange(Field("close"), reversal_window))
        model = SimpleRuleModel(expression=expr, zscore=False)
        pred = model.predict(df)

        # 手动计算
        ret = df["close"].pct_change(reversal_window)
        expected = normalize_zscore(-ret)

        pd.testing.assert_series_equal(pred, expected, check_names=False, atol=1e-10)

    def test_model_walk_forward_interface(self):
        """模拟 walk-forward 流程：fit on train, predict on test。"""
        # 构造跨两年的数据
        rng = np.random.RandomState(42)
        dates_train = [20230101 + i for i in range(50)]
        dates_test = [20240101 + i for i in range(20)]
        symbols = ["SYM000", "SYM001"]
        idx_train = pd.MultiIndex.from_product([dates_train, symbols], names=["date", "symbol"])
        idx_test = pd.MultiIndex.from_product([dates_test, symbols], names=["date", "symbol"])
        idx = idx_train.append(idx_test)
        n = len(idx)
        fm = pd.DataFrame(
            {"feature/reversal_10d": rng.randn(n)},
            index=idx,
        )

        train = fm.loc[fm.index.get_level_values(0) < 20240101]
        test = fm.loc[fm.index.get_level_values(0) >= 20240101]

        model = SimpleRuleModel(score_col="feature/reversal_10d", zscore=True)
        model.fit(train)
        pred = model.predict(test)

        assert len(pred) == len(test)
        assert not pred.isna().all()

    def test_model_with_label(self):
        """fit 接受 label_col 参数但不报错。"""
        fm = _make_feature_matrix()
        model = SimpleRuleModel(score_col="feature/reversal_10d")
        model.fit(fm, label_col="label/ret_1d")  # 不应报错
        pred = model.predict(fm)
        assert len(pred) == len(fm)
