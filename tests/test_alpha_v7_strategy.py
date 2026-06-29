# -*- coding: utf-8 -*-
"""
test_alpha_v7_strategy.py

Alpha v7 策略测试：验证表达式层信号与旧 Alpha v6 硬编码逻辑一致。

核心验证：
1. build_expression() 构建的表达式产出与 compute_alpha_v6_signals() 一致
2. compute_alpha_v7_signals() 输出字段完整
3. 金融正确性：次日持仓、无未来函数
4. 信号评估集成
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.constants import SQRT_TRADING_DAYS_PER_YEAR  # noqa: E402
from scripts.common.feature_expression import (  # noqa: E402
    Const,
    Field,
    Mul,
    Neg,
    PctChange,
    RollingMean,
    RollingStd,
    Sub,
    ZScore,
    normalize_zscore,
)
from strategies.alpha_v7_research_strategy_csv import (  # noqa: E402
    build_expression,
    compute_alpha_v7_signals,
    prepare_benchmark_regime,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_stock_df() -> pd.DataFrame:
    """生成模拟股票数据。"""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    df = pd.DataFrame({
        "date": dates,
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": volume,
        "amount": close * volume,
    })
    df = df.set_index("date", drop=False)
    return df


@pytest.fixture
def sample_benchmark_df() -> pd.DataFrame:
    """生成模拟基准数据。"""
    np.random.seed(123)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 3000 * np.cumprod(1 + np.random.randn(n) * 0.01)

    df = pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.randint(100_000_000, 500_000_000, n).astype(float),
        "amount": np.random.randn(n),
    })
    return df


# ---------------------------------------------------------------------------
# 辅助：旧 Alpha v6 信号计算（用于对照）
# ---------------------------------------------------------------------------

def _old_compute_alpha_v6_signals(
    stock_df: pd.DataFrame,
    alpha_variant: str,
    reversal_window: int,
    vol_window: int,
    turnover_short: int,
    turnover_long: int,
    divergence_window: int,
) -> pd.DataFrame:
    """复制 Alpha v6 的硬编码逻辑，用于对照测试。"""
    result = stock_df.copy()

    result["reversal_return"] = result["close"].pct_change(reversal_window)
    result["realized_vol"] = result["close"].pct_change().rolling(vol_window).std() * SQRT_TRADING_DAYS_PER_YEAR
    result["turnover_ratio"] = (
        result["volume"].rolling(turnover_short).mean()
        / result["volume"].rolling(turnover_long).mean().replace(0, np.nan)
    )
    result["price_trend"] = result["close"].pct_change(divergence_window)
    result["volume_trend"] = result["volume"].pct_change(divergence_window)

    if alpha_variant == "short_term_reversal":
        result["raw_alpha_score"] = -result["reversal_return"]
        result["alpha_signal"] = (result["reversal_return"] < 0).astype(int)
    elif alpha_variant == "low_volatility":
        result["raw_alpha_score"] = -result["realized_vol"]
        result["alpha_signal"] = 1
    elif alpha_variant == "turnover_reversal":
        result["raw_alpha_score"] = -(result["turnover_ratio"] - 1)
        result["alpha_signal"] = (result["turnover_ratio"] < 1).astype(int)
    elif alpha_variant == "volume_price_divergence":
        def _zscore(s):
            std = s.std()
            if std == 0 or np.isnan(std):
                return s * 0.0
            return (s - s.mean()) / std
        result["raw_alpha_score"] = _zscore(result["price_trend"]) - _zscore(result["volume_trend"])
        result["alpha_signal"] = (
            (result["price_trend"] > 0) & (result["volume_trend"] < 0)
        ).astype(int)
    else:
        raise ValueError(f"未知 variant: {alpha_variant}")

    std = result["raw_alpha_score"].std()
    if std == 0 or np.isnan(std):
        result["alpha_score"] = result["raw_alpha_score"] * 0.0
    else:
        result["alpha_score"] = (result["raw_alpha_score"] - result["raw_alpha_score"].mean()) / std

    return result


# ---------------------------------------------------------------------------
# 测试：表达式与旧逻辑一致性
# ---------------------------------------------------------------------------

class TestExpressionConsistency:
    """验证 Alpha v7 表达式层与 Alpha v6 硬编码逻辑一致。"""

    @pytest.mark.parametrize("variant", [
        "short_term_reversal",
        "low_volatility",
        "turnover_reversal",
        "volume_price_divergence",
    ])
    def test_raw_alpha_score_matches(self, sample_stock_df, variant):
        """raw_alpha_score 应与旧逻辑一致。"""
        params = dict(
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
        )

        old = _old_compute_alpha_v6_signals(sample_stock_df, variant, **params)
        new = compute_alpha_v7_signals(sample_stock_df, variant, **params)

        # 比较 raw_alpha_score（允许 NaN 位置一致）
        old_raw = old["raw_alpha_score"]
        new_raw = new["raw_alpha_score"]

        # NaN 位置应一致
        assert old_raw.isna().sum() == new_raw.isna().sum(), \
            f"{variant}: NaN 位置不一致"

        # 非 NaN 值应接近
        valid_mask = old_raw.notna() & new_raw.notna()
        if valid_mask.sum() > 0:
            np.testing.assert_allclose(
                old_raw[valid_mask].values,
                new_raw[valid_mask].values,
                rtol=1e-10, atol=1e-12,
                err_msg=f"{variant}: raw_alpha_score 值不一致",
            )

    @pytest.mark.parametrize("variant", [
        "short_term_reversal",
        "low_volatility",
        "turnover_reversal",
        "volume_price_divergence",
    ])
    def test_alpha_signal_matches(self, sample_stock_df, variant):
        """alpha_signal 应与旧逻辑一致。"""
        params = dict(
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
        )

        old = _old_compute_alpha_v6_signals(sample_stock_df, variant, **params)
        new = compute_alpha_v7_signals(sample_stock_df, variant, **params)

        # NaN 位置一致
        old_sig = old["alpha_signal"]
        new_sig = new["alpha_signal"]

        # 对于 low_volatility，alpha_signal 始终为 1
        if variant == "low_volatility":
            assert (new_sig == 1).all(), f"{variant}: alpha_signal 应始终为 1"
            return

        # 其他 variant：比较非 NaN 部分
        valid_mask = old_sig.notna() & new_sig.notna()
        if valid_mask.sum() > 0:
            np.testing.assert_array_equal(
                old_sig[valid_mask].values.astype(int),
                new_sig[valid_mask].values.astype(int),
                err_msg=f"{variant}: alpha_signal 不一致",
            )

    @pytest.mark.parametrize("variant", [
        "short_term_reversal",
        "low_volatility",
        "turnover_reversal",
        "volume_price_divergence",
    ])
    def test_alpha_score_matches(self, sample_stock_df, variant):
        """alpha_score（z-score 标准化）应与旧逻辑一致。"""
        params = dict(
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
        )

        old = _old_compute_alpha_v6_signals(sample_stock_df, variant, **params)
        new = compute_alpha_v7_signals(sample_stock_df, variant, **params)

        old_score = old["alpha_score"]
        new_score = new["alpha_score"]

        valid_mask = old_score.notna() & new_score.notna()
        if valid_mask.sum() > 0:
            np.testing.assert_allclose(
                old_score[valid_mask].values,
                new_score[valid_mask].values,
                rtol=1e-10, atol=1e-12,
                err_msg=f"{variant}: alpha_score 不一致",
            )


# ---------------------------------------------------------------------------
# 测试：表达式构建
# ---------------------------------------------------------------------------

class TestBuildExpression:
    """测试 build_expression() 返回有效的表达式对象。"""

    def test_short_term_reversal_expressions(self):
        raw_expr, signal_expr = build_expression("short_term_reversal", reversal_window=10)
        assert raw_expr is not None
        assert signal_expr is not None

    def test_low_volatility_expressions(self):
        raw_expr, signal_expr = build_expression("low_volatility", vol_window=60)
        assert raw_expr is not None
        assert signal_expr is not None

    def test_turnover_reversal_expressions(self):
        raw_expr, signal_expr = build_expression("turnover_reversal", turnover_short=10, turnover_long=60)
        assert raw_expr is not None
        assert signal_expr is not None

    def test_volume_price_divergence_expressions(self):
        raw_expr, signal_expr = build_expression("volume_price_divergence", divergence_window=20)
        assert raw_expr is not None
        assert signal_expr is not None

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="未知"):
            build_expression("unknown_variant")

    def test_expressions_eval_to_series(self, sample_stock_df):
        """表达式 eval 应返回与输入等长的 Series。"""
        for variant in ["short_term_reversal", "low_volatility", "turnover_reversal", "volume_price_divergence"]:
            raw_expr, signal_expr = build_expression(variant)
            raw = raw_expr.eval(sample_stock_df)
            sig = signal_expr.eval(sample_stock_df)
            assert len(raw) == len(sample_stock_df), f"{variant}: raw 长度不一致"
            assert len(sig) == len(sample_stock_df), f"{variant}: signal 长度不一致"


# ---------------------------------------------------------------------------
# 测试：compute_alpha_v7_signals 输出
# ---------------------------------------------------------------------------

class TestComputeAlphaV7Signals:
    """测试 compute_alpha_v7_signals() 输出完整性。"""

    def test_output_columns(self, sample_stock_df):
        """输出应包含 raw_alpha_score, alpha_score, alpha_signal。"""
        result = compute_alpha_v7_signals(
            sample_stock_df, "short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
        )
        assert "raw_alpha_score" in result.columns
        assert "alpha_score" in result.columns
        assert "alpha_signal" in result.columns

    def test_output_length_unchanged(self, sample_stock_df):
        """输出长度应与输入一致。"""
        result = compute_alpha_v7_signals(
            sample_stock_df, "short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
        )
        assert len(result) == len(sample_stock_df)

    def test_alpha_signal_is_binary(self, sample_stock_df):
        """alpha_signal 应为 0 或 1。"""
        result = compute_alpha_v7_signals(
            sample_stock_df, "short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
        )
        valid = result["alpha_signal"].dropna()
        assert set(valid.unique()).issubset({0, 1}), \
            f"alpha_signal 应为 0/1，实际: {valid.unique()}"

    def test_alpha_score_is_standardized(self, sample_stock_df):
        """alpha_score 应近似均值 0、标准差 1。"""
        result = compute_alpha_v7_signals(
            sample_stock_df, "short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
        )
        score = result["alpha_score"].dropna()
        if len(score) > 10:
            assert abs(score.mean()) < 0.1, f"alpha_score 均值应接近 0: {score.mean()}"
            assert abs(score.std() - 1.0) < 0.2, f"alpha_score 标准差应接近 1: {score.std()}"


# ---------------------------------------------------------------------------
# 测试：参数化窗口
# ---------------------------------------------------------------------------

class TestParameterizedWindows:
    """测试不同窗口参数。"""

    def test_different_reversal_windows(self, sample_stock_df):
        """不同 reversal_window 应产出不同结果。"""
        results = {}
        for w in [5, 10, 20]:
            r = compute_alpha_v7_signals(
                sample_stock_df, "short_term_reversal",
                reversal_window=w, vol_window=60,
                turnover_short=10, turnover_long=60,
                divergence_window=20,
            )
            results[w] = r["raw_alpha_score"].dropna()

        # 不同窗口的结果应该不同
        assert not results[5].equals(results[10])
        assert not results[10].equals(results[20])

    def test_different_vol_windows(self, sample_stock_df):
        """不同 vol_window 应产出不同结果。"""
        results = {}
        for w in [20, 60, 120]:
            r = compute_alpha_v7_signals(
                sample_stock_df, "low_volatility",
                reversal_window=10, vol_window=w,
                turnover_short=10, turnover_long=60,
                divergence_window=20,
            )
            results[w] = r["raw_alpha_score"].dropna()

        assert not results[20].equals(results[60])
        assert not results[60].equals(results[120])


# ---------------------------------------------------------------------------
# 测试：金融正确性
# ---------------------------------------------------------------------------

class TestFinancialCorrectness:
    """金融正确性检查。"""

    def test_no_future_data_in_signal(self, sample_stock_df):
        """信号计算不应使用未来数据。

        验证方式：修改 T+1 日之后的数据，T 日信号不应改变。
        """
        original = compute_alpha_v7_signals(
            sample_stock_df, "short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
        )

        # 修改第 150 天之后的数据
        modified = sample_stock_df.copy()
        modified.loc[modified.index[150:], "close"] = 99999.0

        modified_result = compute_alpha_v7_signals(
            modified, "short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
        )

        # 第 140 天的信号不应改变（140 + reversal_window=10 = 150，刚好边界）
        # 取第 120 天（远离边界）的信号应完全一致
        check_idx = 120
        assert original["raw_alpha_score"].iloc[check_idx] == modified_result["raw_alpha_score"].iloc[check_idx], \
            "修改未来数据不应影响历史信号"

    def test_benchmark_regime_no_future(self, sample_benchmark_df):
        """大盘 regime filter 不应使用未来数据。"""
        result = prepare_benchmark_regime(sample_benchmark_df, benchmark_ma=120)

        # 修改第 150 天之后的数据
        modified = sample_benchmark_df.copy()
        modified.loc[modified.index[150:], "close"] = 99999.0

        modified_result = prepare_benchmark_regime(modified, benchmark_ma=120)

        # 第 100 天的 market_filter 不应改变
        check_idx = 100
        assert result["market_filter"].iloc[check_idx] == modified_result["market_filter"].iloc[check_idx], \
            "修改未来数据不应影响历史 regime filter"


# ---------------------------------------------------------------------------
# 测试：边界情况
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """边界情况测试。"""

    def test_short_data(self):
        """数据很短时不应崩溃。"""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5, freq="B"),
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [1e6, 1e6, 1e6, 1e6, 1e6],
            "amount": [1e8, 1e8, 1e8, 1e8, 1e8],
        }).set_index("date", drop=False)

        # 不应崩溃，但可能有大量 NaN
        result = compute_alpha_v7_signals(
            df, "short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
        )
        assert len(result) == 5

    def test_constant_price(self):
        """价格不变时信号应为 0 或 NaN。"""
        n = 100
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=n, freq="B"),
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1e6] * n,
            "amount": [1e8] * n,
        }).set_index("date", drop=False)

        result = compute_alpha_v7_signals(
            df, "short_term_reversal",
            reversal_window=10, vol_window=60,
            turnover_short=10, turnover_long=60,
            divergence_window=20,
        )

        # 价格不变，pct_change = 0，raw_alpha_score = -0 = 0
        raw = result["raw_alpha_score"].dropna()
        if len(raw) > 0:
            assert (raw == 0).all(), "价格不变时 raw_alpha_score 应为 0"
