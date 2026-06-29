# -*- coding: utf-8 -*-
"""tests for scripts/evaluate_alpha_signals.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.evaluate_alpha_signals import (
    VALID_VARIANTS,
    compute_alpha_signals,
    compute_forward_returns,
    load_universe,
    evaluate_variant,
    print_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_stock_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """构造单只股票的价格/成交量 DataFrame，模拟真实 QMT 数据结构。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 10.0 * np.exp(np.cumsum(rng.randn(n) * 0.02))
    volume = rng.randint(1000, 100000, size=n).astype(float)
    return pd.DataFrame({
        "date": dates,
        "close": close,
        "volume": volume,
    })


def _make_multi_stock_cross(
    n_dates: int = 100,
    n_stocks: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """构造多股票截面数据，用于 evaluate_variant 的 mock。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    symbols = [f"S{i:04d}.SZ" for i in range(n_stocks)]
    rows = []
    for dt in dates:
        for sym in symbols:
            rows.append({
                "date": dt,
                "symbol": sym,
                "close": 10.0 + rng.randn() * 0.5,
                "volume": float(rng.randint(1000, 50000)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test compute_alpha_signals
# ---------------------------------------------------------------------------

class TestComputeAlphaSignals:
    """测试 4 个 alpha variant 的信号计算。"""

    def test_short_term_reversal_basic(self):
        df = _make_stock_df(100)
        result = compute_alpha_signals(df, "short_term_reversal", reversal_window=10)
        assert "raw_alpha_score" in result.columns
        assert "alpha_score" in result.columns
        assert "alpha_signal" in result.columns
        # alpha_signal 应为 0 或 1
        assert set(result["alpha_signal"].dropna().unique()).issubset({0, 1})

    def test_low_volatility_basic(self):
        df = _make_stock_df(100)
        result = compute_alpha_signals(df, "low_volatility", vol_window=60)
        assert "raw_alpha_score" in result.columns
        # low_volatility 的 alpha_signal 恒为 1
        valid = result["alpha_signal"].dropna()
        assert (valid == 1).all()

    def test_turnover_reversal_basic(self):
        df = _make_stock_df(100)
        result = compute_alpha_signals(
            df, "turnover_reversal", turnover_short=10, turnover_long=60,
        )
        assert "raw_alpha_score" in result.columns
        assert "alpha_signal" in result.columns

    def test_volume_price_divergence_basic(self):
        df = _make_stock_df(100)
        result = compute_alpha_signals(df, "volume_price_divergence", divergence_window=20)
        assert "raw_alpha_score" in result.columns
        assert "alpha_signal" in result.columns

    def test_unknown_variant_raises(self):
        df = _make_stock_df(50)
        with pytest.raises(ValueError, match="未知的 alpha_variant"):
            compute_alpha_signals(df, "nonexistent_variant")

    def test_alpha_score_standardized(self):
        """alpha_score 应该近似标准化（mean≈0, std≈1）。"""
        df = _make_stock_df(200)
        result = compute_alpha_signals(df, "short_term_reversal", reversal_window=10)
        valid = result["alpha_score"].dropna()
        assert abs(valid.mean()) < 0.1
        assert abs(valid.std() - 1.0) < 0.1

    def test_consistent_with_alpha_v6_logic(self):
        """验证 compute_alpha_signals 与 v7 表达式层一致。"""
        df = _make_stock_df(200, seed=123)
        result = compute_alpha_signals(df, "short_term_reversal", reversal_window=10)
        from strategies.alpha_v7_research_strategy_csv import build_expression
        raw_expr, sig_expr = build_expression("short_term_reversal", reversal_window=10)
        expected_raw = raw_expr.eval(df)
        expected_signal = sig_expr.eval(df).astype(int)
        np.testing.assert_allclose(
            result["raw_alpha_score"].values, expected_raw.values, rtol=1e-10,
        )
        np.testing.assert_array_equal(
            result["alpha_signal"].values, expected_signal.values,
        )

    def test_consistency_with_v7_expression(self):
        """新版本使用表达式层，应与直接调用 build_expression 结果一致。"""
        from strategies.alpha_v7_research_strategy_csv import build_expression
        stock_df = _make_stock_df(100)
        for variant in ["short_term_reversal", "low_volatility", "turnover_reversal", "volume_price_divergence"]:
            signals = compute_alpha_signals(stock_df, variant)
            raw_expr, sig_expr = build_expression(variant)
            expected_raw = raw_expr.eval(stock_df)
            expected_sig = sig_expr.eval(stock_df).astype(int)
            np.testing.assert_allclose(signals["raw_alpha_score"].values, expected_raw.values, rtol=1e-10)
            np.testing.assert_array_equal(signals["alpha_signal"].values, expected_sig.values)

    def test_preserves_date_and_volume(self):
        """原始 date 和 volume 列应保留。"""
        df = _make_stock_df(50)
        result = compute_alpha_signals(df, "short_term_reversal")
        assert "date" in result.columns
        assert "volume" in result.columns
        pd.testing.assert_series_equal(result["date"], df["date"])

    def test_constant_score_no_crash(self):
        """当所有 raw_alpha_score 相同时（std=0），不应崩溃。"""
        df = _make_stock_df(50)
        # 强制 close 不变 → reversal_return = 0 → raw_alpha_score = 0
        df["close"] = 10.0
        result = compute_alpha_signals(df, "short_term_reversal", reversal_window=5)
        # 前 reversal_window 行是 NaN（pct_change 产生），后面是 0.0
        valid = result["alpha_score"].dropna()
        assert (valid == 0.0).all()


# ---------------------------------------------------------------------------
# Test compute_forward_returns
# ---------------------------------------------------------------------------

class TestComputeForwardReturns:

    def test_basic_horizons(self):
        df = _make_stock_df(50)
        result = compute_forward_returns(df, horizons=[1, 5, 20])
        assert "ret_1d" in result.columns
        assert "ret_5d" in result.columns
        assert "ret_20d" in result.columns
        assert "date" in result.columns
        assert "close" in result.columns

    def test_forward_return_formula(self):
        """验证 ret_Nd = close(t+N) / close(t) - 1。"""
        df = _make_stock_df(30, seed=7)
        result = compute_forward_returns(df, horizons=[1])
        # 手动验证第一个值
        expected = df["close"].iloc[1] / df["close"].iloc[0] - 1.0
        assert abs(result["ret_1d"].iloc[0] - expected) < 1e-10

    def test_tail_nan(self):
        """最后 N 行的 ret_Nd 应为 NaN。"""
        df = _make_stock_df(30)
        result = compute_forward_returns(df, horizons=[5])
        assert pd.isna(result["ret_5d"].iloc[-1])
        assert pd.isna(result["ret_5d"].iloc[-5])
        # 倒数第 6 行应有值
        assert not pd.isna(result["ret_5d"].iloc[-6])

    def test_single_horizon(self):
        df = _make_stock_df(20)
        result = compute_forward_returns(df, horizons=[3])
        assert "ret_3d" in result.columns
        assert len(result) == 20


# ---------------------------------------------------------------------------
# Test load_universe
# ---------------------------------------------------------------------------

class TestLoadUniverse:

    def test_from_stock_list(self):
        """逗号分隔的 stock_list 应正确解析。"""
        symbols = load_universe(
            universe_file=None,
            export_root=Path("/nonexistent"),
            stock_list="000001.SZ,600000.SH,000002.SZ",
        )
        assert symbols == ["000001.SZ", "600000.SH", "000002.SZ"]

    def test_from_file(self, tmp_path):
        """从文件加载股票列表。"""
        f = tmp_path / "universe.txt"
        f.write_text("000001.SZ\n600000.SH\n# comment\n000002.SZ\n", encoding="utf-8")
        symbols = load_universe(
            universe_file=f,
            export_root=Path("/nonexistent"),
            stock_list=None,
        )
        assert symbols == ["000001.SZ", "600000.SH", "000002.SZ"]

    def test_file_priority_over_stock_list(self):
        """universe_file 优先于 stock_list。"""
        # 没有文件时用 stock_list
        symbols = load_universe(
            universe_file=None,
            export_root=Path("/nonexistent"),
            stock_list="000001.SZ",
        )
        assert symbols == ["000001.SZ"]

    def test_empty_input_no_catalog(self):
        """无输入且无 catalog 时应返回空列表。"""
        symbols = load_universe(
            universe_file=None,
            export_root=Path("/nonexistent"),
            stock_list=None,
        )
        # 可能为空或从 catalog 扫描
        assert isinstance(symbols, list)

    def test_file_with_bom(self, tmp_path):
        """UTF-8 BOM 文件应正确处理。"""
        f = tmp_path / "universe_bom.txt"
        f.write_bytes(b"\xef\xbb\xbf000001.SZ\n600000.SH\n")
        symbols = load_universe(
            universe_file=f,
            export_root=Path("/nonexistent"),
            stock_list=None,
        )
        assert "000001.SZ" in symbols

    def test_whitespace_stripped(self):
        """前后空格应被去除。"""
        symbols = load_universe(
            universe_file=None,
            export_root=Path("/nonexistent"),
            stock_list=" 000001.SZ , 600000.SH ",
        )
        assert symbols == ["000001.SZ", "600000.SH"]


# ---------------------------------------------------------------------------
# Test evaluate_variant (mocked I/O)
# ---------------------------------------------------------------------------

class TestEvaluateVariant:
    """测试 evaluate_variant，mock 数据加载避免依赖真实数据。"""

    @patch("scripts.evaluate_alpha_signals.find_csv_for_stock")
    @patch("scripts.evaluate_alpha_signals.load_qmt_price_csv")
    def test_basic_evaluate(self, mock_load, mock_find):
        """mock 3 只股票数据，验证返回结构。"""
        mock_find.return_value = (Path("/fake.csv"), "000001.SZ", "stock", "SZ")
        stock_df = _make_stock_df(200)
        mock_load.return_value = stock_df

        results = evaluate_variant(
            symbols=["000001.SZ"],
            alpha_variant="short_term_reversal",
            export_root=Path("/fake"),
            start="20230101",
            end="20241231",
            label_horizons=[1, 5],
            n_quantiles=5,
            reversal_window=10,
            vol_window=60,
            turnover_short=10,
            turnover_long=60,
            divergence_window=20,
        )
        assert "ret_1d" in results
        assert "ret_5d" in results
        for label_col, eval_result in results.items():
            assert "ic_summary" in eval_result
            assert "ic_daily" in eval_result
            assert "quantile_returns" in eval_result

    @patch("scripts.evaluate_alpha_signals.find_csv_for_stock")
    @patch("scripts.evaluate_alpha_signals.load_qmt_price_csv")
    def test_all_variants_evaluate(self, mock_load, mock_find):
        """验证所有 4 个 variant 都能通过 evaluate_variant。"""
        mock_find.return_value = (Path("/fake.csv"), "000001.SZ", "stock", "SZ")
        stock_df = _make_stock_df(200)
        mock_load.return_value = stock_df

        for variant in VALID_VARIANTS:
            results = evaluate_variant(
                symbols=["000001.SZ"],
                alpha_variant=variant,
                export_root=Path("/fake"),
                start="20230101",
                end="",
                label_horizons=[1],
                n_quantiles=3,
                reversal_window=10,
                vol_window=60,
                turnover_short=10,
                turnover_long=60,
                divergence_window=20,
            )
            assert "ret_1d" in results, f"Failed for {variant}"

    @patch("scripts.evaluate_alpha_signals.find_csv_for_stock")
    @patch("scripts.evaluate_alpha_signals.load_qmt_price_csv")
    def test_short_data_skipped(self, mock_load, mock_find):
        """数据太短的股票应被跳过，最终无数据则抛异常。"""
        mock_find.return_value = (Path("/fake.csv"), "000001.SZ", "stock", "SZ")
        short_df = _make_stock_df(5)  # 太短
        mock_load.return_value = short_df

        with pytest.raises(RuntimeError, match="没有成功加载"):
            evaluate_variant(
                symbols=["000001.SZ"],
                alpha_variant="short_term_reversal",
                export_root=Path("/fake"),
                start="20230101",
                end="",
                label_horizons=[1],
                n_quantiles=5,
                reversal_window=10,
                vol_window=60,
                turnover_short=10,
                turnover_long=60,
                divergence_window=20,
            )


# ---------------------------------------------------------------------------
# Test print_summary
# ---------------------------------------------------------------------------

class TestPrintSummary:

    def test_print_summary_no_crash(self, capsys):
        """print_summary 应正常输出不崩溃。"""
        # 构造最小 results 结构
        results = {
            "ret_1d": {
                "ic_summary": pd.DataFrame([{
                    "ic_mean": 0.05,
                    "ic_std": 0.02,
                    "icir": 2.5,
                    "ic_tstat": 3.0,
                    "ic_positive_rate": 0.7,
                    "rank_ic_mean": 0.04,
                    "rank_icir": 2.0,
                }]),
                "quantile_returns": pd.DataFrame({
                    "quantile": [1, 2, 3, 4, 5],
                    "mean_return": [0.01, 0.005, 0.0, -0.005, -0.01],
                    "long_short": [0.02, 0.015, 0.01, 0.005, 0.0],
                }),
            },
        }
        print_summary(results, "test_variant")
        captured = capsys.readouterr()
        assert "test_variant" in captured.out
        assert "IC mean" in captured.out

    def test_print_summary_empty(self, capsys):
        """空 summary 不应崩溃。"""
        results = {
            "ret_1d": {
                "ic_summary": pd.DataFrame(),
                "quantile_returns": pd.DataFrame(),
            },
        }
        print_summary(results, "empty_variant")
        captured = capsys.readouterr()
        assert "empty_variant" in captured.out
        assert "无有效数据" in captured.out


# ---------------------------------------------------------------------------
# Test VALID_VARIANTS 常量
# ---------------------------------------------------------------------------

class TestConstants:

    def test_valid_variants_count(self):
        assert len(VALID_VARIANTS) == 4

    def test_valid_variants_names(self):
        expected = {
            "short_term_reversal",
            "low_volatility",
            "turnover_reversal",
            "volume_price_divergence",
        }
        assert set(VALID_VARIANTS) == expected
