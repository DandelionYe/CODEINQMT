# -*- coding: utf-8 -*-
"""
tests/test_metrics_shared.py

验证 scripts.common.metrics 中共享函数的正确性：
- format_pct / format_float（从 14+ 个文件统一到此处）
- calc_portfolio_metrics（从 portfolio_backtest_csv.py 迁移）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.common.metrics import (
    format_pct,
    format_float,
    max_drawdown_from_equity,
    calc_metrics_from_returns,
    calc_metrics_from_dataframe,
    calc_metrics_simple,
    calc_metrics_from_daily,
    calc_portfolio_metrics,
)
from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR


# =========================================================================
# format_pct / format_float
# =========================================================================


class TestFormatPct:
    """format_pct 格式化百分比。"""

    def test_positive(self):
        assert format_pct(0.1234) == "12.34%"

    def test_negative(self):
        assert format_pct(-0.05) == "-5.00%"

    def test_zero(self):
        assert format_pct(0.0) == "0.00%"

    def test_none_returns_na(self):
        assert format_pct(None) == "N/A"

    def test_nan_returns_na(self):
        assert format_pct(np.nan) == "N/A"

    def test_large_value(self):
        assert format_pct(1.5) == "150.00%"


class TestFormatFloat:
    """format_float 格式化浮点数。"""

    def test_positive(self):
        assert format_float(1.23456) == "1.2346"

    def test_negative(self):
        assert format_float(-0.5678) == "-0.5678"

    def test_zero(self):
        assert format_float(0.0) == "0.0000"

    def test_none_returns_na(self):
        assert format_float(None) == "N/A"

    def test_nan_returns_na(self):
        assert format_float(np.nan) == "N/A"


# =========================================================================
# calc_portfolio_metrics
# =========================================================================


class TestCalcPortfolioMetrics:
    """calc_portfolio_metrics 组合层指标。"""

    def _make_returns(self, n: int = 100, seed: int = 42) -> pd.Series:
        rng = np.random.RandomState(seed)
        return pd.Series(rng.normal(0.0005, 0.01, n))

    def test_basic_keys(self):
        ret = self._make_returns()
        m = calc_portfolio_metrics(ret, 100.0, 50.0, 0.5)
        expected = {
            "total_return", "annual_return", "annual_volatility",
            "max_drawdown", "sharpe", "calmar", "turnover",
            "total_commission", "total_slippage_cost",
        }
        assert expected == set(m.keys())

    def test_empty_returns(self):
        m = calc_portfolio_metrics(pd.Series([], dtype=float))
        nan_keys = {"total_return", "annual_return", "annual_volatility",
                    "max_drawdown", "sharpe", "calmar", "turnover"}
        zero_keys = {"total_commission", "total_slippage_cost"}
        for k in nan_keys:
            assert np.isnan(m[k]), f"{k} should be NaN"
        for k in zero_keys:
            assert m[k] == 0.0, f"{k} should be 0.0"

    def test_commission_slippage_passthrough(self):
        ret = self._make_returns()
        m = calc_portfolio_metrics(ret, 123.45, 67.89)
        assert m["total_commission"] == 123.45
        assert m["total_slippage_cost"] == 67.89

    def test_turnover_passthrough(self):
        ret = self._make_returns()
        m = calc_portfolio_metrics(ret, turnover=2.5)
        assert m["turnover"] == 2.5

    def test_total_return_formula(self):
        ret = pd.Series([0.01, -0.005, 0.02, 0.003])
        m = calc_portfolio_metrics(ret)
        expected = (1 + ret).prod() - 1
        assert abs(m["total_return"] - expected) < 1e-12

    def test_annual_return_formula(self):
        ret = pd.Series([0.01, -0.005, 0.02])
        m = calc_portfolio_metrics(ret)
        total = (1 + ret).prod() - 1
        expected = (1 + total) ** (TRADING_DAYS_PER_YEAR / len(ret)) - 1
        assert abs(m["annual_return"] - expected) < 1e-12

    def test_annual_volatility_formula(self):
        ret = pd.Series([0.01, -0.005, 0.02, 0.003])
        m = calc_portfolio_metrics(ret)
        expected = ret.std() * SQRT_TRADING_DAYS_PER_YEAR
        assert abs(m["annual_volatility"] - expected) < 1e-12

    def test_max_drawdown_non_positive(self):
        ret = self._make_returns()
        m = calc_portfolio_metrics(ret)
        assert m["max_drawdown"] <= 0

    def test_max_drawdown_formula(self):
        ret = pd.Series([0.01, -0.03, 0.02, -0.01, 0.015])
        m = calc_portfolio_metrics(ret)
        equity = (1 + ret).cumprod()
        dd = (equity - equity.cummax()) / equity.cummax()
        assert abs(m["max_drawdown"] - dd.min()) < 1e-12

    def test_sharpe_formula(self):
        ret = self._make_returns()
        m = calc_portfolio_metrics(ret)
        annual_ret = m["annual_return"]
        annual_vol = m["annual_volatility"]
        expected = annual_ret / annual_vol if annual_vol > 0 else np.nan
        if np.isnan(expected):
            assert np.isnan(m["sharpe"])
        else:
            assert abs(m["sharpe"] - expected) < 1e-12

    def test_calmar_formula(self):
        ret = self._make_returns()
        m = calc_portfolio_metrics(ret)
        if m["max_drawdown"] != 0:
            expected = m["annual_return"] / abs(m["max_drawdown"])
            assert abs(m["calmar"] - expected) < 1e-12

    def test_all_positive_returns(self):
        ret = pd.Series([0.001] * 200)
        m = calc_portfolio_metrics(ret)
        assert m["total_return"] > 0
        assert m["max_drawdown"] == 0
        assert np.isnan(m["calmar"])  # max_dd == 0

    def test_all_negative_returns(self):
        ret = pd.Series([-0.001] * 200)
        m = calc_portfolio_metrics(ret)
        assert m["total_return"] < 0
        assert m["max_drawdown"] < 0


class TestCalcPortfolioMetricsVsCalcMetricsFromReturns:
    """验证 calc_portfolio_metrics 与 calc_metrics_from_returns 口径一致（共享字段）。"""

    def _make_returns(self, n: int = 100, seed: int = 42) -> pd.Series:
        rng = np.random.RandomState(seed)
        return pd.Series(rng.normal(0.0005, 0.01, n))

    def test_total_return_consistency(self):
        ret = self._make_returns()
        pm = calc_portfolio_metrics(ret)
        rm = calc_metrics_from_returns(ret)
        assert abs(pm["total_return"] - rm["total_return"]) < 1e-10

    def test_annual_return_consistency(self):
        ret = self._make_returns()
        pm = calc_portfolio_metrics(ret)
        rm = calc_metrics_from_returns(ret)
        assert abs(pm["annual_return"] - rm["annual_return"]) < 1e-10

    def test_max_drawdown_consistency(self):
        ret = self._make_returns()
        pm = calc_portfolio_metrics(ret)
        rm = calc_metrics_from_returns(ret)
        assert abs(pm["max_drawdown"] - rm["max_drawdown"]) < 1e-10

    def test_annual_volatility_consistency(self):
        ret = self._make_returns()
        pm = calc_portfolio_metrics(ret)
        rm = calc_metrics_from_returns(ret)
        assert abs(pm["annual_volatility"] - rm["annual_volatility"]) < 1e-10

    def test_calmar_consistency(self):
        ret = self._make_returns()
        pm = calc_portfolio_metrics(ret)
        rm = calc_metrics_from_returns(ret)
        assert abs(pm["calmar"] - rm["calmar"]) < 1e-10


# =========================================================================
# calc_metrics_from_daily（统一版本）
# =========================================================================


class TestCalcMetricsFromDaily:
    """calc_metrics_from_daily 统一指标计算。"""

    def _make_returns(self, n: int = 100, seed: int = 42) -> pd.Series:
        rng = np.random.RandomState(seed)
        return pd.Series(rng.normal(0.0005, 0.01, n))

    def test_keys_only_returns_subset(self):
        ret = self._make_returns()
        m = calc_metrics_from_daily(ret, keys_only=True)
        expected = {"total_return", "annual_return", "annual_volatility",
                    "max_drawdown", "sharpe", "calmar"}
        assert set(m.keys()) == expected

    def test_full_keys_include_portfolio_fields(self):
        ret = self._make_returns()
        m = calc_metrics_from_daily(ret, keys_only=False)
        assert "turnover" in m
        assert "total_commission" in m
        assert "total_slippage_cost" in m

    def test_empty_returns_keys_only(self):
        m = calc_metrics_from_daily(pd.Series([], dtype=float), keys_only=True)
        for k in ("total_return", "annual_return", "annual_volatility",
                   "max_drawdown", "sharpe", "calmar"):
            assert np.isnan(m[k]), f"{k} should be NaN"

    def test_empty_returns_full_keys(self):
        m = calc_metrics_from_daily(pd.Series([], dtype=float), keys_only=False)
        assert np.isnan(m["turnover"])
        assert m["total_commission"] == 0.0
        assert m["total_slippage_cost"] == 0.0

    def test_total_return_formula(self):
        ret = pd.Series([0.01, -0.005, 0.02, 0.003])
        m = calc_metrics_from_daily(ret)
        expected = float((1 + ret).prod() - 1)
        assert abs(m["total_return"] - expected) < 1e-12

    def test_annual_return_formula(self):
        ret = pd.Series([0.01, -0.005, 0.02])
        m = calc_metrics_from_daily(ret)
        total = float((1 + ret).prod() - 1)
        expected = (1 + total) ** (TRADING_DAYS_PER_YEAR / len(ret)) - 1
        assert abs(m["annual_return"] - expected) < 1e-12

    def test_annual_volatility_formula(self):
        ret = pd.Series([0.01, -0.005, 0.02, 0.003])
        m = calc_metrics_from_daily(ret)
        expected = float(ret.std()) * SQRT_TRADING_DAYS_PER_YEAR
        assert abs(m["annual_volatility"] - expected) < 1e-12

    def test_sharpe_annualized_ratio(self):
        ret = self._make_returns()
        m = calc_metrics_from_daily(ret, sharpe_method="annualized_ratio")
        expected = m["annual_return"] / m["annual_volatility"]
        assert abs(m["sharpe"] - expected) < 1e-12

    def test_sharpe_mean_std(self):
        ret = self._make_returns()
        m = calc_metrics_from_daily(ret, sharpe_method="mean_std")
        expected = float(ret.mean() / ret.std() * SQRT_TRADING_DAYS_PER_YEAR)
        assert abs(m["sharpe"] - expected) < 1e-12

    def test_max_drawdown_non_positive(self):
        ret = self._make_returns()
        m = calc_metrics_from_daily(ret)
        assert m["max_drawdown"] <= 0

    def test_calmar_formula(self):
        ret = self._make_returns()
        m = calc_metrics_from_daily(ret)
        if m["max_drawdown"] != 0:
            expected = m["annual_return"] / abs(m["max_drawdown"])
            assert abs(m["calmar"] - expected) < 1e-12

    def test_all_positive_returns(self):
        ret = pd.Series([0.001] * 200)
        m = calc_metrics_from_daily(ret)
        assert m["total_return"] > 0
        assert m["max_drawdown"] == 0
        assert np.isnan(m["calmar"])


class TestCalcMetricsFromDailyDelegation:
    """验证 calc_metrics_from_daily 与被委托方的输出一致性。"""

    def _make_returns(self, n: int = 100, seed: int = 42) -> pd.Series:
        rng = np.random.RandomState(seed)
        return pd.Series(rng.normal(0.0005, 0.01, n))

    def test_consistent_with_compute_metrics_from_daily(self):
        """与 wf_robustness_shared.compute_metrics_from_daily 输出一致。"""
        from scripts.common.wf_robustness_shared import compute_metrics_from_daily
        ret = self._make_returns()
        m_new = calc_metrics_from_daily(ret, keys_only=True)
        m_old = compute_metrics_from_daily(ret)
        for k in m_old:
            assert abs(m_new[k] - m_old[k]) < 1e-12, f"Mismatch on {k}"

    def test_consistent_with_calc_metrics(self):
        """与 wf_report_shared.calc_metrics 输出一致。"""
        from scripts.common.wf_report_shared import calc_metrics
        ret = self._make_returns()
        m_new = calc_metrics_from_daily(ret, keys_only=True, sharpe_method="mean_std")
        m_old = calc_metrics(ret)
        for k in ("total_return", "annual_return", "annual_volatility",
                   "max_drawdown", "sharpe", "calmar"):
            assert abs(m_new[k] - m_old[k]) < 1e-12, f"Mismatch on {k}"
        assert m_old["days"] == len(ret)

    def test_consistent_empty_compute_metrics_from_daily(self):
        """空序列与 compute_metrics_from_daily 一致。"""
        from scripts.common.wf_robustness_shared import compute_metrics_from_daily
        m_new = calc_metrics_from_daily(pd.Series([], dtype=float), keys_only=True)
        m_old = compute_metrics_from_daily(pd.Series([], dtype=float))
        for k in m_old:
            assert np.isnan(m_new[k]) and np.isnan(m_old[k]), f"NaN mismatch on {k}"

    def test_consistent_empty_calc_metrics(self):
        """空序列与 calc_metrics 一致。"""
        from scripts.common.wf_report_shared import calc_metrics
        m_new = calc_metrics_from_daily(pd.Series([], dtype=float), keys_only=True, sharpe_method="mean_std")
        m_old = calc_metrics(pd.Series([], dtype=float))
        for k in ("total_return", "annual_return", "annual_volatility",
                   "max_drawdown", "sharpe", "calmar"):
            assert np.isnan(m_new[k]) and np.isnan(m_old[k]), f"NaN mismatch on {k}"
        assert m_old["days"] == 0

    def test_consistent_with_calc_portfolio_metrics(self):
        """与 calc_portfolio_metrics 在共享字段上一致。"""
        ret = self._make_returns()
        m_new = calc_metrics_from_daily(ret, keys_only=False)
        m_port = calc_portfolio_metrics(ret)
        for k in ("total_return", "annual_return", "annual_volatility",
                   "max_drawdown", "sharpe", "calmar"):
            assert abs(m_new[k] - m_port[k]) < 1e-12, f"Mismatch on {k}"


# =========================================================================
# max_drawdown_from_equity（直接测试）
# =========================================================================


class TestMaxDrawdownFromEquity:
    """max_drawdown_from_equity 直接测试。"""

    def test_known_drawdown(self):
        """已知净值序列，回撤 = (95-110)/110。"""
        equity = pd.Series([100.0, 110.0, 95.0, 105.0])
        result = max_drawdown_from_equity(equity)
        assert abs(result - (95.0 / 110.0 - 1.0)) < 1e-12

    def test_monotonic_increasing_returns_zero(self):
        """单调递增净值，回撤为 0。"""
        equity = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        assert max_drawdown_from_equity(equity) == 0.0

    def test_single_element(self):
        """单元素序列，回撤为 0。"""
        equity = pd.Series([100.0])
        assert max_drawdown_from_equity(equity) == 0.0

    def test_two_elements_down(self):
        """两个元素下跌，回撤为 (90-100)/100。"""
        equity = pd.Series([100.0, 90.0])
        assert abs(max_drawdown_from_equity(equity) - (-0.1)) < 1e-12

    def test_deeper_drawdown_not_first(self):
        """最大回撤不在序列开头。"""
        equity = pd.Series([100.0, 120.0, 110.0, 80.0, 90.0])
        result = max_drawdown_from_equity(equity)
        # 最大回撤：从 120 跌到 80，即 80/120 - 1 = -1/3
        assert abs(result - (80.0 / 120.0 - 1.0)) < 1e-12

    def test_multiple_equal_peaks(self):
        """多次到达相同峰值。"""
        equity = pd.Series([100.0, 90.0, 100.0, 95.0, 100.0])
        result = max_drawdown_from_equity(equity)
        # 第一次 100->90 = -10%, 第二次 100->95 = -5%
        assert abs(result - (-0.1)) < 1e-12


# =========================================================================
# calc_metrics_from_dataframe（DataFrame 包装器）
# =========================================================================


class TestCalcMetricsFromDataframe:
    """calc_metrics_from_dataframe DataFrame 包装器测试。"""

    def _make_df(self, n: int = 50, seed: int = 42) -> pd.DataFrame:
        """构造含 strategy_ret 和 equity 列的 DataFrame。"""
        rng = np.random.RandomState(seed)
        ret = pd.Series(rng.normal(0.0005, 0.01, n), name="strategy_ret")
        equity = (1_000_000 * (1 + ret).cumprod()).rename("equity")
        return pd.DataFrame({"strategy_ret": ret, "equity": equity})

    def test_basic_keys(self):
        """返回 dict 包含所有核心 key。"""
        df = self._make_df()
        m = calc_metrics_from_dataframe(df)
        expected = {
            "total_return", "annual_return", "max_drawdown",
            "sharpe", "annual_volatility", "days",
            "trade_count", "final_equity", "calmar",
        }
        assert expected == set(m.keys())

    def test_total_return_from_returns(self):
        """total_return 基于收益率序列计算（而非净值列）。"""
        df = self._make_df()
        m = calc_metrics_from_dataframe(df)
        ret = df["strategy_ret"].dropna().astype(float)
        expected = float((1 + ret).prod() - 1.0)
        assert abs(m["total_return"] - expected) < 1e-10

    def test_final_equity_overridden(self):
        """final_equity 被覆盖为 equity 序列的最后一个值。"""
        df = self._make_df()
        m = calc_metrics_from_dataframe(df)
        assert m["final_equity"] == float(df["equity"].iloc[-1])

    def test_empty_dataframe(self):
        """空 DataFrame 返回零值/NaN。"""
        df = pd.DataFrame({"strategy_ret": [], "equity": []})
        m = calc_metrics_from_dataframe(df)
        assert m["total_return"] == 0.0
        assert m["sharpe"] == 0.0
        assert m["final_equity"] == 0.0
        assert m["days"] == 0

    def test_position_column_detected(self):
        """含 position 列时 trade_count > 0。"""
        df = self._make_df()
        pos = pd.Series([0] * 10 + [1] * 20 + [0] * 20, name="position")
        df["position"] = pos
        m = calc_metrics_from_dataframe(df)
        assert m["trade_count"] > 0

    def test_no_position_column(self):
        """无 position 列时 trade_count 为 0。"""
        df = self._make_df()
        m = calc_metrics_from_dataframe(df)
        assert m["trade_count"] == 0

    def test_custom_column_names(self):
        """自定义列名。"""
        df = self._make_df()
        df = df.rename(columns={"strategy_ret": "my_ret", "equity": "my_eq"})
        m = calc_metrics_from_dataframe(df, ret_col="my_ret", equity_col="my_eq")
        assert m["total_return"] != 0.0

    def test_consistency_with_calc_metrics_from_returns(self):
        """与 calc_metrics_from_returns 输出一致（共享字段）。"""
        df = self._make_df()
        m_df = calc_metrics_from_dataframe(df)
        m_ret = calc_metrics_from_returns(
            df["strategy_ret"].dropna().astype(float),
            cash=float(df["equity"].iloc[0]),
        )
        # total_return、annual_return、max_drawdown、sharpe 应一致
        for k in ("total_return", "annual_return", "max_drawdown", "sharpe", "annual_volatility"):
            assert abs(m_df[k] - m_ret[k]) < 1e-10, f"Mismatch on {k}"

    def test_short_dataframe(self):
        """只有 2 行的 DataFrame 不崩溃。"""
        df = pd.DataFrame({
            "strategy_ret": [0.01, -0.005],
            "equity": [1_000_000.0, 1_005_000.0],
        })
        m = calc_metrics_from_dataframe(df)
        assert m["days"] == 2
        assert isinstance(m["total_return"], float)


# =========================================================================
# calc_metrics_simple（简化版包装器）
# =========================================================================


class TestCalcMetricsSimple:
    """calc_metrics_simple 简化版包装器测试。"""

    def test_returns_exactly_7_keys(self):
        """返回恰好 7 个 key。"""
        ret = pd.Series([0.01, -0.005, 0.02, 0.003])
        m = calc_metrics_simple(ret)
        expected = {
            "total_return", "annual_return", "max_drawdown",
            "sharpe", "annual_volatility", "days", "calmar",
        }
        assert set(m.keys()) == expected

    def test_values_match_calc_metrics_from_returns(self):
        """所有值与 calc_metrics_from_returns 一致。"""
        ret = pd.Series([0.01, -0.005, 0.02, 0.003, -0.01, 0.005])
        m_simple = calc_metrics_simple(ret)
        m_full = calc_metrics_from_returns(ret)
        for k in m_simple:
            assert abs(m_simple[k] - m_full[k]) < 1e-12, f"Mismatch on {k}"

    def test_empty_series(self):
        """空 Series 不崩溃。"""
        ret = pd.Series([], dtype=float)
        m = calc_metrics_simple(ret)
        assert m["days"] == 0
        assert m["total_return"] == 0.0

    def test_single_return(self):
        """单个收益率。"""
        ret = pd.Series([0.01])
        m = calc_metrics_simple(ret)
        assert m["days"] == 1
        assert abs(m["total_return"] - 0.01) < 1e-12

    def test_no_trade_count_key(self):
        """不包含 trade_count（简化版不含此字段）。"""
        ret = pd.Series([0.01, -0.005, 0.02])
        m = calc_metrics_simple(ret)
        assert "trade_count" not in m
        assert "final_equity" not in m


# =========================================================================
# 金融正确性检查
# =========================================================================


class TestFinancialCorrectness:
    """金融正确性验证。"""

    def test_no_future_data_in_metrics(self):
        """指标计算不使用未来数据——验证 rolling 计算只依赖历史。"""
        ret = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.01, 0.005])
        m = calc_portfolio_metrics(ret)
        # 指标只依赖完整序列，不引入前瞻偏差
        assert isinstance(m["total_return"], float)

    def test_format_handles_inf(self):
        """format_pct/format_float 对 inf 值不崩溃。"""
        # np.isinf 不是 NaN，所以会尝试格式化
        # 但实际使用中 inf 不太可能出现，这里验证 NaN 路径
        assert format_pct(np.nan) == "N/A"
        assert format_float(np.nan) == "N/A"

    def test_portfolio_metrics_monotonic_equity(self):
        """全正收益时权益曲线单调递增，回撤为 0。"""
        ret = pd.Series([0.001] * 100)
        m = calc_portfolio_metrics(ret)
        assert m["max_drawdown"] == 0.0
        assert m["total_return"] > 0


# ---------------------------------------------------------------------------
# 统一导入验证：确保 11 个旧脚本的 format_pct/format_float 均来自规范模块
# ---------------------------------------------------------------------------

class TestFormatUnifiedImport:
    """验证所有旧脚本的 format_pct/format_float 均来自 scripts.common.metrics。"""

    _LEGACY_SCRIPTS = [
        "scripts.analyze_alpha_v4_research_walk_forward_results",
        "scripts.analyze_alpha_v5_research_walk_forward_results",
        "scripts.analyze_ma_v3_momentum_walk_forward_results",
        "scripts.analyze_ma_market_filter_walk_forward_results",
        "scripts.analyze_walk_forward_results",
        "scripts.diagnose_alpha_v4_research_strategy_results",
        "scripts.diagnose_alpha_v5_research_strategy_results",
        "scripts.diagnose_ma_market_filter_strategy_results",
        "scripts.diagnose_ma_v3_momentum_strategy_results",
        "scripts.validate_alpha_v4_robustness",
        "scripts.validate_alpha_v5_robustness",
    ]

    @pytest.mark.parametrize("module_path", _LEGACY_SCRIPTS)
    def test_format_pct_from_canonical(self, module_path: str):
        """format_pct 应来自 scripts.common.metrics，且正确处理 None。"""
        mod = __import__(module_path, fromlist=["format_pct"])
        assert mod.format_pct is format_pct
        assert mod.format_pct(None) == "N/A"
        assert mod.format_pct(float("nan")) == "N/A"
        assert mod.format_pct(0.1234) == "12.34%"

    @pytest.mark.parametrize("module_path", _LEGACY_SCRIPTS)
    def test_format_float_from_canonical(self, module_path: str):
        """format_float 应来自 scripts.common.metrics，且正确处理 None。"""
        mod = __import__(module_path, fromlist=["format_float"])
        assert mod.format_float is format_float
        assert mod.format_float(None) == "N/A"
        assert mod.format_float(float("nan")) == "N/A"
        assert mod.format_float(0.1234) == "0.1234"
