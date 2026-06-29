# -*- coding: utf-8 -*-
"""
test_feature_matrix.py

测试 Feature Matrix Builder。

覆盖：
  - load_stock_data
  - load_universe_data
  - compute_factor_on_universe
  - compute_forward_returns
  - build_feature_matrix
  - save_feature_matrix
  - build_expressions_from_registry
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 确保项目根在 sys.path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.common.feature_expression import (
    Field, PctChange, Neg, RollingMean, RollingStd, ZScore,
)
from scripts.common.feature_matrix import (
    load_stock_data,
    load_universe_data,
    compute_factor_on_universe,
    compute_forward_returns,
    build_feature_matrix,
    save_feature_matrix,
    build_expressions_from_registry,
    build_feature_matrix_from_registry,
    _build_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures：模拟 parquet 数据
# ---------------------------------------------------------------------------

def _make_stock_df(symbol: str, n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """生成模拟股票数据。"""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("20200101", periods=n_days, freq="B").strftime("%Y%m%d").astype(int)
    close = 10.0 + np.cumsum(rng.randn(n_days) * 0.1)
    close = np.maximum(close, 1.0)  # 确保为正

    return pd.DataFrame({
        "timetag": dates,
        "open": close * (1 + rng.randn(n_days) * 0.005),
        "high": close * (1 + abs(rng.randn(n_days) * 0.01)),
        "low": close * (1 - abs(rng.randn(n_days) * 0.01)),
        "close": close,
        "volumn": rng.randint(100000, 1000000, n_days).astype(float),
        "amount": rng.uniform(1e8, 1e9, n_days),
    })


@pytest.fixture
def tmp_parquet_dir(tmp_path):
    """创建临时 parquet 目录，包含两只股票。"""
    parquet_root = tmp_path / "parquet"
    for market in ["SZ", "SH"]:
        (parquet_root / market).mkdir(parents=True, exist_ok=True)

    df1 = _make_stock_df("000001.SZ", seed=42)
    df1.to_parquet(parquet_root / "SZ" / "price_000001.parquet", engine="pyarrow")

    df2 = _make_stock_df("000002.SZ", seed=99)
    df2.to_parquet(parquet_root / "SZ" / "price_000002.parquet", engine="pyarrow")

    df3 = _make_stock_df("600000.SH", seed=7)
    df3.to_parquet(parquet_root / "SH" / "price_600000.parquet", engine="pyarrow")

    return parquet_root


@pytest.fixture
def registry_path(tmp_path):
    """创建临时 factor_registry.json。"""
    registry = {
        "version": "1.0",
        "factors": [
            {
                "factor_id": "alpha_v6_short_term_reversal",
                "display_name": "短期反转",
                "expression": "Neg(PctChange(Field('close'), reversal_window))",
                "inputs": ["close"],
                "params": {
                    "reversal_window": {"default": 10, "candidates": [5, 10, 20]}
                },
                "variant": "short_term_reversal",
                "pit_safe": True,
            },
            {
                "factor_id": "alpha_v6_low_volatility",
                "display_name": "低波动异象",
                "expression": "Neg(RollingStd(PctChange(Field('close'), 1), vol_window) * sqrt(252))",
                "inputs": ["close"],
                "params": {
                    "vol_window": {"default": 60, "candidates": [20, 60, 120]}
                },
                "variant": "low_volatility",
                "pit_safe": True,
            },
        ],
    }
    path = tmp_path / "factor_registry.json"
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_stock_data
# ---------------------------------------------------------------------------

class TestLoadStockData:
    def test_basic(self, tmp_parquet_dir):
        df = load_stock_data("000001.SZ", tmp_parquet_dir)
        assert "date" in df.columns
        assert "symbol" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert (df["symbol"] == "000001.SZ").all()
        assert df["date"].dtype in [int, np.int64, np.int32]

    def test_column_rename(self, tmp_parquet_dir):
        """timetag -> date, volumn -> volume"""
        df = load_stock_data("000001.SZ", tmp_parquet_dir)
        assert "timetag" not in df.columns
        assert "volumn" not in df.columns
        assert "date" in df.columns
        assert "volume" in df.columns

    def test_date_filter(self, tmp_parquet_dir):
        df_all = load_stock_data("000001.SZ", tmp_parquet_dir)
        df_filtered = load_stock_data("000001.SZ", tmp_parquet_dir, start="20200301")
        assert len(df_filtered) <= len(df_all)
        assert (df_filtered["date"] >= 20200301).all()

    def test_date_filter_end(self, tmp_parquet_dir):
        df = load_stock_data("000001.SZ", tmp_parquet_dir, end="20200301")
        assert (df["date"] <= 20200301).all()

    def test_missing_file(self, tmp_parquet_dir):
        with pytest.raises(FileNotFoundError):
            load_stock_data("999999.SZ", tmp_parquet_dir)


# ---------------------------------------------------------------------------
# load_universe_data
# ---------------------------------------------------------------------------

class TestLoadUniverseData:
    def test_basic(self, tmp_parquet_dir):
        symbols = ["000001.SZ", "000002.SZ"]
        df = load_universe_data(symbols, tmp_parquet_dir)
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["date", "symbol"]
        unique_symbols = df.index.get_level_values(1).unique().tolist()
        assert set(unique_symbols) == set(symbols)

    def test_missing_symbol_warns(self, tmp_parquet_dir, caplog):
        symbols = ["000001.SZ", "999999.SZ"]
        df = load_universe_data(symbols, tmp_parquet_dir)
        # 应该只加载 000001.SZ
        unique_symbols = df.index.get_level_values(1).unique().tolist()
        assert unique_symbols == ["000001.SZ"]

    def test_all_missing_raises(self, tmp_parquet_dir):
        with pytest.raises(ValueError, match="无有效股票数据"):
            load_universe_data(["999999.SZ"], tmp_parquet_dir)


# ---------------------------------------------------------------------------
# compute_factor_on_universe
# ---------------------------------------------------------------------------

class TestComputeFactorOnUniverse:
    def test_basic(self, tmp_parquet_dir):
        universe = load_universe_data(["000001.SZ", "000002.SZ"], tmp_parquet_dir)
        expr = Neg(PctChange(Field("close"), 10))
        result = compute_factor_on_universe(universe, expr, "test_factor")

        assert isinstance(result, pd.Series)
        assert result.name == "test_factor"
        assert isinstance(result.index, pd.MultiIndex)
        assert len(result) > 0

    def test_single_stock(self, tmp_parquet_dir):
        universe = load_universe_data(["000001.SZ"], tmp_parquet_dir)
        expr = RollingMean(Field("close"), 5)
        result = compute_factor_on_universe(universe, expr, "ma5")

        assert len(result) > 0
        # 前 4 个值应为 NaN（rolling(5) 的窗口）
        stock_vals = result.xs("000001.SZ", level=1)
        assert stock_vals.iloc[:4].isna().all()


# ---------------------------------------------------------------------------
# compute_forward_returns
# ---------------------------------------------------------------------------

class TestComputeForwardReturns:
    def test_basic(self, tmp_parquet_dir):
        universe = load_universe_data(["000001.SZ"], tmp_parquet_dir)
        labels = compute_forward_returns(universe, horizons=[1, 5, 20])

        assert "label/ret_1d" in labels.columns
        assert "label/ret_5d" in labels.columns
        assert "label/ret_20d" in labels.columns
        assert isinstance(labels.index, pd.MultiIndex)

    def test_no_future_leakage(self, tmp_parquet_dir):
        """验证 ret_1d 的第一行不是 NaN（因为 shift(-1)），最后一行是 NaN。"""
        universe = load_universe_data(["000001.SZ"], tmp_parquet_dir)
        labels = compute_forward_returns(universe, horizons=[1])

        stock_labels = labels.xs("000001.SZ", level=1)
        # 最后一行应为 NaN（没有未来数据）
        assert pd.isna(stock_labels.iloc[-1]["label/ret_1d"])
        # 第一行应有值（有未来数据）
        assert not pd.isna(stock_labels.iloc[0]["label/ret_1d"])

    def test_multi_stock(self, tmp_parquet_dir):
        universe = load_universe_data(["000001.SZ", "000002.SZ"], tmp_parquet_dir)
        labels = compute_forward_returns(universe, horizons=[1, 5])

        symbols = labels.index.get_level_values(1).unique().tolist()
        assert set(symbols) == {"000001.SZ", "000002.SZ"}


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_basic(self, tmp_parquet_dir):
        symbols = ["000001.SZ", "000002.SZ"]
        expressions = {
            "reversal_10d": Neg(PctChange(Field("close"), 10)),
            "ma_20d": RollingMean(Field("close"), 20),
        }
        fm, manifest = build_feature_matrix(
            symbols, expressions, tmp_parquet_dir,
            start="20200301", label_horizons=[1, 5],
        )

        assert "feature/reversal_10d" in fm.columns
        assert "feature/ma_20d" in fm.columns
        assert "label/ret_1d" in fm.columns
        assert "label/ret_5d" in fm.columns
        assert isinstance(fm.index, pd.MultiIndex)
        assert fm.index.names == ["date", "symbol"]

        # manifest 检查
        assert manifest["universe_size"] == 2
        assert len(manifest["date_range"]) == 2
        assert manifest["date_range"][0] >= 20200301

    def test_with_zscore(self, tmp_parquet_dir):
        """测试 ZScore 因子。"""
        symbols = ["000001.SZ"]
        expressions = {
            "zscore_reversal": ZScore(Neg(PctChange(Field("close"), 5))),
        }
        fm, manifest = build_feature_matrix(
            symbols, expressions, tmp_parquet_dir,
            start="20200601", label_horizons=[1],
        )

        col = "feature/zscore_reversal"
        assert col in fm.columns
        # ZScore 后均值应接近 0（排除 NaN）
        valid = fm[col].dropna()
        if len(valid) > 10:
            assert abs(valid.mean()) < 0.5  # 宽松检查


# ---------------------------------------------------------------------------
# save_feature_matrix
# ---------------------------------------------------------------------------

class TestSaveFeatureMatrix:
    def test_basic(self, tmp_parquet_dir, tmp_path):
        symbols = ["000001.SZ"]
        expressions = {"reversal": Neg(PctChange(Field("close"), 5))}
        fm, manifest = build_feature_matrix(
            symbols, expressions, tmp_parquet_dir, label_horizons=[1],
        )

        output_dir = tmp_path / "output"
        save_feature_matrix(fm, manifest, output_dir)

        assert (output_dir / "feature_matrix.parquet").exists()
        assert (output_dir / "feature_matrix_manifest.json").exists()
        assert (output_dir / "feature_matrix_head.csv").exists()

        # 验证 parquet 可读回
        loaded = pd.read_parquet(output_dir / "feature_matrix.parquet")
        assert loaded.shape == fm.shape
        assert list(loaded.columns) == list(fm.columns)

    def test_manifest_content(self, tmp_parquet_dir, tmp_path):
        symbols = ["000001.SZ", "000002.SZ"]
        expressions = {"reversal": Neg(PctChange(Field("close"), 5))}
        fm, manifest = build_feature_matrix(
            symbols, expressions, tmp_parquet_dir, label_horizons=[1, 5],
        )

        output_dir = tmp_path / "output"
        save_feature_matrix(fm, manifest, output_dir)

        loaded_manifest = json.loads((output_dir / "feature_matrix_manifest.json").read_text(encoding="utf-8"))
        assert loaded_manifest["universe_size"] == 2
        assert "feature/reversal" in loaded_manifest["columns"]
        assert "label/ret_1d" in loaded_manifest["columns"]


# ---------------------------------------------------------------------------
# build_expressions_from_registry
# ---------------------------------------------------------------------------

class TestBuildExpressionsFromRegistry:
    def test_basic(self, registry_path):
        expressions = build_expressions_from_registry(registry_path)
        assert len(expressions) == 2
        assert "alpha_v6_short_term_reversal" in expressions
        assert "alpha_v6_low_volatility" in expressions

    def test_filter_by_variant(self, registry_path):
        expressions = build_expressions_from_registry(
            registry_path, variant_ids=["short_term_reversal"]
        )
        assert len(expressions) == 1
        assert "alpha_v6_short_term_reversal" in expressions

    def test_params_override(self, registry_path):
        expressions = build_expressions_from_registry(
            registry_path,
            variant_ids=["short_term_reversal"],
            params_override={"reversal_window": 20},
        )
        assert "alpha_v6_short_term_reversal" in expressions


# ---------------------------------------------------------------------------
# 端到端：从 registry 构建特征矩阵
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_from_registry(self, tmp_parquet_dir, registry_path):
        """端到端：从 registry 构建特征矩阵并保存。"""
        symbols = ["000001.SZ", "000002.SZ"]

        expressions = build_expressions_from_registry(registry_path)
        fm, manifest = build_feature_matrix(
            symbols, expressions, tmp_parquet_dir,
            start="20200301", label_horizons=[1, 5],
        )

        assert fm.shape[0] > 0
        assert fm.shape[1] >= 4  # 2 features + 2 labels

        # 保存并读回
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td) / "fm_output"
            save_feature_matrix(fm, manifest, output_dir)
            loaded = pd.read_parquet(output_dir / "feature_matrix.parquet")
            assert loaded.shape == fm.shape


# ---------------------------------------------------------------------------
# build_feature_matrix_from_registry
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrixFromRegistry:
    """测试 build_feature_matrix_from_registry 便捷函数。"""

    def test_basic(self, tmp_parquet_dir, registry_path):
        """基本调用：从 registry 构建特征矩阵。"""
        symbols = ["000001.SZ", "000002.SZ"]
        fm, manifest = build_feature_matrix_from_registry(
            symbols, registry_path, parquet_root=tmp_parquet_dir,
            start="20200301", label_horizons=[1, 5],
        )

        assert fm.shape[0] > 0
        # 至少有 2 个 feature + 2 个 label
        assert fm.shape[1] >= 4
        assert "universe_size" in manifest

    def test_variant_filter(self, tmp_parquet_dir, registry_path):
        """variant_ids 过滤只构建指定 variant。"""
        symbols = ["000001.SZ"]
        fm, manifest = build_feature_matrix_from_registry(
            symbols, registry_path,
            variant_ids=["short_term_reversal"],
            parquet_root=tmp_parquet_dir,
            start="20200301", label_horizons=[1],
        )

        feature_cols = [c for c in fm.columns if c.startswith("feature/")]
        assert len(feature_cols) == 1
        assert "feature/alpha_v6_short_term_reversal" in feature_cols

    def test_params_override(self, tmp_parquet_dir, registry_path):
        """params_override 覆盖 registry 中的默认参数。"""
        symbols = ["000001.SZ"]
        fm_default, _ = build_feature_matrix_from_registry(
            symbols, registry_path,
            variant_ids=["short_term_reversal"],
            parquet_root=tmp_parquet_dir,
            start="20200301", label_horizons=[1],
        )
        fm_override, _ = build_feature_matrix_from_registry(
            symbols, registry_path,
            variant_ids=["short_term_reversal"],
            params_override={"reversal_window": 20},
            parquet_root=tmp_parquet_dir,
            start="20200301", label_horizons=[1],
        )

        # 不同参数应产生不同结果（除非数据太短）
        col = "feature/alpha_v6_short_term_reversal"
        if len(fm_default) > 20 and len(fm_override) > 20:
            # 仅比较非 NaN 部分
            valid_idx = fm_default[col].dropna().index.intersection(
                fm_override[col].dropna().index
            )
            if len(valid_idx) > 20:
                # 不要求完全不同，但至少不应全部相等
                assert not fm_default.loc[valid_idx, col].equals(
                    fm_override.loc[valid_idx, col]
                )

    def test_empty_registry_raises(self, tmp_parquet_dir, tmp_path):
        """空 registry 应抛出 ValueError。"""
        empty_registry = tmp_path / "empty_registry.json"
        empty_registry.write_text(
            json.dumps({"factors": []}),
            encoding="utf-8",
        )

        symbols = ["000001.SZ"]
        with pytest.raises(ValueError, match="未能从 registry 构建任何表达式"):
            build_feature_matrix_from_registry(
                symbols, empty_registry, parquet_root=tmp_parquet_dir,
            )

    def test_label_horizons_passthrough(self, tmp_parquet_dir, registry_path):
        """label_horizons 参数正确透传。"""
        symbols = ["000001.SZ"]
        fm, manifest = build_feature_matrix_from_registry(
            symbols, registry_path,
            variant_ids=["short_term_reversal"],
            parquet_root=tmp_parquet_dir,
            start="20200301", label_horizons=[1, 10],
        )

        label_cols = [c for c in fm.columns if c.startswith("label/")]
        assert "label/ret_1d" in label_cols
        assert "label/ret_10d" in label_cols

    def test_consistency_with_manual_call(self, tmp_parquet_dir, registry_path):
        """build_feature_matrix_from_registry 结果与手动调用一致。"""
        symbols = ["000001.SZ", "000002.SZ"]

        # 便捷函数
        fm_auto, manifest_auto = build_feature_matrix_from_registry(
            symbols, registry_path, parquet_root=tmp_parquet_dir,
            start="20200301", label_horizons=[1, 5],
        )

        # 手动调用
        expressions = build_expressions_from_registry(registry_path)
        fm_manual, manifest_manual = build_feature_matrix(
            symbols, expressions, tmp_parquet_dir,
            start="20200301", label_horizons=[1, 5],
        )

        assert fm_auto.shape == fm_manual.shape
        assert list(fm_auto.columns) == list(fm_manual.columns)
        pd.testing.assert_frame_equal(fm_auto, fm_manual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
