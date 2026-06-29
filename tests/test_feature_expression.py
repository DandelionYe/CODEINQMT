# -*- coding: utf-8 -*-
"""tests for scripts/common/feature_expression.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.common.feature_expression import (
    Abs,
    Add,
    And,
    AsInt,
    Const,
    CSRank,
    Div,
    Eq,
    Field,
    Ge,
    Gt,
    Le,
    Lt,
    Mul,
    Neg,
    Not,
    Or,
    PctChange,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingStd,
    Shift,
    Sub,
    Where,
    ZScore,
    normalize_zscore,
)


@pytest.fixture
def sample_df():
    """5 行示例 DataFrame。"""
    return pd.DataFrame({
        "close": [10.0, 11.0, 12.0, 11.5, 13.0],
        "volume": [100, 200, 150, 180, 220],
        "flag": [1, 0, 1, 1, 0],
    })


# --- normalize_zscore ---

class TestNormalizeZscore:
    def test_basic(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_zscore(s)
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 1e-10

    def test_constant_returns_zero(self):
        s = pd.Series([3.0, 3.0, 3.0])
        result = normalize_zscore(s)
        assert (result == 0.0).all()

    def test_single_value(self):
        s = pd.Series([5.0])
        result = normalize_zscore(s)
        assert (result == 0.0).all()


# --- Field / Const ---

class TestLeafNodes:
    def test_field(self, sample_df):
        result = Field("close").eval(sample_df)
        pd.testing.assert_series_equal(result, sample_df["close"])

    def test_const_scalar(self, sample_df):
        result = Const(42).eval(sample_df)
        assert (result == 42).all()
        assert len(result) == len(sample_df)

    def test_const_series(self, sample_df):
        s = pd.Series([1, 2, 3, 4, 5])
        result = Const(s).eval(sample_df)
        assert list(result) == [1, 2, 3, 4, 5]


# --- 单目操作 ---

class TestUnaryOps:
    def test_neg(self, sample_df):
        result = Neg(Field("close")).eval(sample_df)
        expected = -sample_df["close"]
        pd.testing.assert_series_equal(result, expected)

    def test_abs(self, sample_df):
        neg = pd.Series([-1, -2, 3, -4, 5])
        result = Abs(Const(neg)).eval(sample_df)
        assert list(result) == [1, 2, 3, 4, 5]

    def test_not(self, sample_df):
        result = Not(Field("flag")).eval(sample_df)
        assert list(result) == [0, 1, 0, 0, 1]

    def test_asint(self, sample_df):
        bool_series = pd.Series([True, False, True, False, True])
        result = AsInt(Const(bool_series)).eval(sample_df)
        assert list(result) == [1, 0, 1, 0, 1]


# --- 时序操作 ---

class TestTimeOps:
    def test_shift(self, sample_df):
        result = Shift(Field("close"), 1).eval(sample_df)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == 10.0

    def test_pct_change(self, sample_df):
        result = PctChange(Field("close"), 1).eval(sample_df)
        assert np.isnan(result.iloc[0])
        assert abs(result.iloc[1] - 0.1) < 1e-10  # (11-10)/10


# --- 滚动操作 ---

class TestRollingOps:
    def test_rolling_mean(self, sample_df):
        result = RollingMean(Field("close"), 3).eval(sample_df)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert abs(result.iloc[2] - (10 + 11 + 12) / 3) < 1e-10

    def test_rolling_std(self, sample_df):
        result = RollingStd(Field("close"), 3).eval(sample_df)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] > 0

    def test_rolling_max(self, sample_df):
        result = RollingMax(Field("close"), 3).eval(sample_df)
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 12.0

    def test_rolling_min(self, sample_df):
        result = RollingMin(Field("close"), 3).eval(sample_df)
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 10.0


# --- 标准化 ---

class TestNormalization:
    def test_zscore(self, sample_df):
        result = ZScore(Field("close")).eval(sample_df)
        assert abs(result.mean()) < 1e-10

    def test_csrank_basic(self, sample_df):
        result = CSRank(Field("close")).eval(sample_df)
        assert result.min() > 0
        assert result.max() <= 1.0


# --- 双目算术 ---

class TestBinaryArithmetic:
    def test_add(self, sample_df):
        expr = Field("close") + Field("volume")
        result = expr.eval(sample_df)
        expected = sample_df["close"] + sample_df["volume"]
        pd.testing.assert_series_equal(result, expected)

    def test_sub(self, sample_df):
        expr = Field("close") - Const(10)
        result = expr.eval(sample_df)
        assert list(result) == [0.0, 1.0, 2.0, 1.5, 3.0]

    def test_mul(self, sample_df):
        expr = Field("close") * Const(2)
        result = expr.eval(sample_df)
        assert list(result) == [20.0, 22.0, 24.0, 23.0, 26.0]

    def test_div(self, sample_df):
        expr = Field("close") / Const(2)
        result = expr.eval(sample_df)
        assert list(result) == [5.0, 5.5, 6.0, 5.75, 6.5]

    def test_radd(self, sample_df):
        expr = Const(100) + Field("close")
        result = expr.eval(sample_df)
        assert list(result) == [110.0, 111.0, 112.0, 111.5, 113.0]

    def test_neg_operator(self, sample_df):
        expr = -Field("close")
        result = expr.eval(sample_df)
        assert list(result) == [-10.0, -11.0, -12.0, -11.5, -13.0]


# --- 比较操作 ---

class TestComparison:
    def test_gt(self, sample_df):
        expr = Field("close") > Const(11.5)
        result = expr.eval(sample_df)
        assert list(result) == [0, 0, 1, 0, 1]

    def test_ge(self, sample_df):
        expr = Field("close") >= Const(12.0)
        result = expr.eval(sample_df)
        assert list(result) == [0, 0, 1, 0, 1]

    def test_lt(self, sample_df):
        expr = Field("close") < Const(11.5)
        result = expr.eval(sample_df)
        assert list(result) == [1, 1, 0, 0, 0]

    def test_le(self, sample_df):
        expr = Field("close") <= Const(11.0)
        result = expr.eval(sample_df)
        assert list(result) == [1, 1, 0, 0, 0]

    def test_eq(self, sample_df):
        expr = Field("close") == Const(12.0)
        result = expr.eval(sample_df)
        assert list(result) == [0, 0, 1, 0, 0]


# --- 逻辑操作 ---

class TestLogical:
    def test_and(self, sample_df):
        # close=[10,11,12,11.5,13]; >10=[F,T,T,T,T]; <12.5=[T,T,T,T,F]; AND=[F,T,T,T,F]
        expr = (Field("close") > Const(10)) & (Field("close") < Const(12.5))
        result = expr.eval(sample_df)
        assert list(result) == [0, 1, 1, 1, 0]

    def test_or(self, sample_df):
        expr = (Field("close") < Const(10.5)) | (Field("close") > Const(12.5))
        result = expr.eval(sample_df)
        assert list(result) == [1, 0, 0, 0, 1]

    def test_invert(self, sample_df):
        expr = ~(Field("flag"))
        result = expr.eval(sample_df)
        assert list(result) == [0, 1, 0, 0, 1]


# --- Where ---

class TestWhere:
    def test_where(self, sample_df):
        expr = Where(Field("flag"), Field("close"), Const(0))
        result = expr.eval(sample_df)
        assert list(result) == [10.0, 0, 12.0, 11.5, 0]


# --- 组合表达式 ---

class TestComposite:
    def test_alpha_v6_short_term_reversal(self, sample_df):
        """复现 alpha v6 short_term_reversal 信号。"""
        reversal_window = 2
        reversal_return = PctChange(Field("close"), reversal_window)
        raw_alpha_score = -reversal_return
        alpha_signal = AsInt(reversal_return < 0)
        alpha_score = ZScore(raw_alpha_score)

        raw = raw_alpha_score.eval(sample_df)
        signal = alpha_signal.eval(sample_df)
        score = alpha_score.eval(sample_df)

        # 前 reversal_window 个值为 NaN
        assert raw.iloc[:reversal_window].isna().all()
        # alpha_signal 是 0 或 1
        assert set(signal.dropna().unique()).issubset({0, 1})
        # alpha_score 已标准化
        valid = score.dropna()
        if len(valid) > 1:
            assert abs(valid.mean()) < 1e-10

    def test_operator_chaining(self, sample_df):
        """测试表达式链式组合。"""
        expr = ZScore(Field("close") - RollingMean(Field("close"), 3))
        result = expr.eval(sample_df)
        # 第 0,1 行是 NaN（rolling 3 不够）
        assert result.iloc[:2].isna().all()
        valid = result.iloc[2:].dropna()
        if len(valid) > 1:
            assert abs(valid.mean()) < 1e-10


# ---------------------------------------------------------------------------
# 测试：normalize_zscore 统一导入
# ---------------------------------------------------------------------------

class TestNormalizeZscoreUnifiedImport:
    """验证所有策略脚本的 normalize_zscore 均从 feature_expression 导入。"""

    @pytest.mark.parametrize("module_path", [
        "strategies.alpha_v4_research_strategy_csv",
        "strategies.alpha_v5_research_strategy_csv",
        "strategies.alpha_v6_research_strategy_csv",
        "strategies.alpha_v7_research_strategy_csv",
        "strategies.ma_v3_momentum_strategy_csv",
        "scripts.batch_ma_v3_momentum_backtest_csv",
        "scripts.validate_ma_v3_momentum_candidates",
    ])
    def test_normalize_zscore_from_canonical(self, module_path):
        """normalize_zscore 应来自 scripts.common.feature_expression。"""
        import importlib
        mod = importlib.import_module(module_path)
        fn = getattr(mod, "normalize_zscore")
        assert fn.__module__ == "scripts.common.feature_expression", \
            f"{module_path}.normalize_zscore 来自 {fn.__module__}，应来自 scripts.common.feature_expression"
