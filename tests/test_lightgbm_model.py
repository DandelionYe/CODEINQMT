# -*- coding: utf-8 -*-
"""
test_lightgbm_model.py

Tests for scripts.common.models.lightgbm_model (LightGBMModel).

测试策略：
  - 当 lightgbm 已安装时：运行完整功能测试。
  - 当 lightgbm 未安装时：只测试导入错误处理和 __init__.py 条件导出。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 检测 lightgbm 是否可用
# ---------------------------------------------------------------------------

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_feature_matrix(
    n_dates: int = 60,
    n_symbols: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """构造 MultiIndex feature matrix。"""
    rng = np.random.RandomState(seed)
    dates = [20240101 + i for i in range(n_dates)]
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    n = len(idx)
    return pd.DataFrame(
        {
            "feature/reversal_10d": rng.randn(n),
            "feature/low_vol_60d": rng.randn(n),
            "feature/turnover_10d": rng.randn(n),
            "label/ret_1d": rng.randn(n) * 0.01,
            "label/ret_5d": rng.randn(n) * 0.02,
        },
        index=idx,
    )


def _make_simple_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """构造简单 DataFrame（非 MultiIndex）。"""
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "feature/f1": rng.randn(n),
            "feature/f2": rng.randn(n),
            "label/y": rng.randn(n) * 0.01,
        },
        index=pd.Index(range(n), name="idx"),
    )


# ---------------------------------------------------------------------------
# TestLightGBMModelImportGuard — 不依赖 lightgbm 安装
# ---------------------------------------------------------------------------

class TestLightGBMModelImportGuard:
    """测试导入错误处理（不依赖 lightgbm 安装）。"""

    def test_import_guard_error_message(self):
        """_import_lightgbm 未安装时抛出 ImportError 并包含安装提示。"""
        from scripts.common.models.lightgbm_model import _import_lightgbm

        if HAS_LIGHTGBM:
            # 已安装时直接返回模块
            mod = _import_lightgbm()
            assert hasattr(mod, "LGBMRegressor")
        else:
            # 未安装时抛出 ImportError
            with pytest.raises(ImportError, match="LightGBM 未安装"):
                _import_lightgbm()

    def test_init_checks_dependency(self):
        """构造 LightGBMModel 时检查 lightgbm 可用性。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        if HAS_LIGHTGBM:
            model = LightGBMModel()
            assert model.get_params()["model_type"] == "LightGBMModel"
        else:
            with pytest.raises(ImportError, match="LightGBM 未安装"):
                LightGBMModel()

    def test_init_export_in_models(self):
        """__init__.py 导出 LightGBMModel（类定义本身可导入，实例化时才检查依赖）。"""
        from scripts.common import models

        # LightGBMModel 类始终可导入（延迟依赖检查在 __init__ 中）
        assert hasattr(models, "LightGBMModel")
        assert "LightGBMModel" in models.__all__


# ---------------------------------------------------------------------------
# TestLightGBMModel — 需要 lightgbm 安装
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm 未安装")
class TestLightGBMModel:
    """LightGBMModel 完整功能测试。"""

    def test_basic_fit_predict(self):
        """基本 fit + predict 流程。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(
            label_col="label/ret_1d",
            n_estimators=10,
            verbose=-1,
        )
        model.fit(fm)
        pred = model.predict(fm)
        assert isinstance(pred, pd.Series)
        assert len(pred) == len(fm)
        assert not pred.isna().all()

    def test_auto_detect_feature_cols(self):
        """自动检测 feature/ 前缀列。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5)
        model.fit(fm)
        params = model.get_params()
        assert params["fitted_n_features"] == 3  # 3 个 feature/ 列

    def test_explicit_feature_cols(self):
        """显式指定特征列。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(
            label_col="label/ret_1d",
            feature_cols=["feature/reversal_10d"],
            n_estimators=5,
        )
        model.fit(fm)
        pred = model.predict(fm)
        assert len(pred) == len(fm)

    def test_get_params(self):
        """get_params 返回完整参数。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        model = LightGBMModel(
            label_col="label/ret_1d",
            n_estimators=50,
            learning_rate=0.1,
        )
        params = model.get_params()
        assert params["model_type"] == "LightGBMModel"
        assert params["n_estimators"] == 50
        assert params["learning_rate"] == 0.1
        assert params["label_col"] == "label/ret_1d"

    def test_repr(self):
        """repr 包含类名和关键参数。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        model = LightGBMModel(label_col="label/ret_1d", n_estimators=10)
        r = repr(model)
        assert "LightGBMModel" in r
        assert "label/ret_1d" in r

    def test_predict_unfitted_returns_nan(self):
        """未 fit 时 predict 返回 NaN。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5)
        # 不调用 fit
        pred = model.predict(fm)
        assert pred.isna().all()

    def test_fit_with_all_nan_labels(self):
        """标签全为 NaN 时 fit 不报错。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        fm["label/ret_1d"] = np.nan
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5)
        model.fit(fm)
        pred = model.predict(fm)
        assert pred.isna().all()  # 模型未训练，返回 NaN

    def test_predict_preserves_index(self):
        """predict 输出索引与输入一致。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5)
        model.fit(fm)
        pred = model.predict(fm)
        pd.testing.assert_index_equal(pred.index, fm.index)

    def test_fit_with_inf_values(self):
        """特征含 Inf 时 fit 不报错（替换为 NaN）。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        fm.iloc[0, 0] = np.inf
        fm.iloc[1, 0] = -np.inf
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5)
        model.fit(fm)
        pred = model.predict(fm)
        assert len(pred) == len(fm)

    def test_label_col_override(self):
        """fit 时 label_col 参数覆盖构造时的值。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5)
        model.fit(fm, label_col="label/ret_5d")
        pred = model.predict(fm)
        assert not pred.isna().all()

    def test_no_feature_cols_raises(self):
        """无特征列时抛出 ValueError。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        model = LightGBMModel(label_col="a", n_estimators=5)
        with pytest.raises(ValueError, match="无法自动检测特征列"):
            model.fit(df)

    def test_alphamodel_interface(self):
        """LightGBMModel 是 AlphaModel 的子类。"""
        from scripts.common.models.base import AlphaModel
        from scripts.common.models.lightgbm_model import LightGBMModel

        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5)
        assert isinstance(model, AlphaModel)

    def test_exclude_col_filters_explicit_feature_cols(self):
        """exclude_col 从显式 feature_cols 中排除指定列。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(
            label_col="label/ret_1d",
            feature_cols=["feature/reversal_10d", "feature/low_vol_60d", "label/ret_1d"],
            n_estimators=5,
        )
        # fit 内部会用 exclude_col=effective_label_col 排除标签列
        model.fit(fm)
        # 如果 exclude_col 未生效，label/ret_1d 会被当作特征列（共 3 个）
        # exclude_col 生效后，fitted 应为 2 个特征列
        params = model.get_params()
        assert params["fitted_n_features"] == 2

    def test_exclude_col_with_auto_detect(self):
        """exclude_col 在自动检测模式下不影响 feature/ 前缀列。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5)
        model.fit(fm)
        # 自动检测到 3 个 feature/ 列（reversal_10d, low_vol_60d, turnover_10d）
        params = model.get_params()
        assert params["fitted_n_features"] == 3

    def test_resolve_feature_cols_with_exclude_col(self):
        """_resolve_feature_cols 直接调用时 exclude_col 正确排除。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(
            label_col="label/ret_1d",
            feature_cols=["feature/f1", "label/ret_1d", "feature/f2"],
            n_estimators=5,
        )
        cols = model._resolve_feature_cols(fm, exclude_col="label/ret_1d")
        assert "label/ret_1d" not in cols
        assert cols == ["feature/f1", "feature/f2"]

    def test_resolve_feature_cols_exclude_col_not_present(self):
        """_resolve_feature_cols exclude_col 不在列表中时不影响结果。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(
            label_col="label/ret_1d",
            feature_cols=["feature/f1", "feature/f2"],
            n_estimators=5,
        )
        cols = model._resolve_feature_cols(fm, exclude_col="label/nonexistent")
        assert cols == ["feature/f1", "feature/f2"]


# ---------------------------------------------------------------------------
# TestLightGBMModelFinancialCorrectness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm 未安装")
class TestLightGBMModelFinancialCorrectness:
    """LightGBMModel 金融正确性检查。"""

    def test_fit_does_not_use_test_data(self):
        """fit 不使用测试期数据（通过独立训练验证）。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        rng = np.random.RandomState(42)
        # 构造 train/test 分离数据
        train = pd.DataFrame({
            "feature/f1": rng.randn(100),
            "feature/f2": rng.randn(100),
            "label/y": rng.randn(100) * 0.01,
        })
        test = pd.DataFrame({
            "feature/f1": rng.randn(50),
            "feature/f2": rng.randn(50),
            "label/y": rng.randn(50) * 0.01,
        })

        model = LightGBMModel(label_col="label/y", n_estimators=10)
        model.fit(train)
        pred_train = model.predict(train)
        pred_test = model.predict(test)
        # 两组预测应该不同（不同数据）
        assert not np.allclose(pred_train.values[:50], pred_test.values, equal_nan=True)

    def test_predict_does_not_use_label(self):
        """predict 不使用 label 列（删除 label 后结果一致）。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5)
        model.fit(fm)

        pred_with_label = model.predict(fm)
        fm_no_label = fm.drop(columns=["label/ret_1d", "label/ret_5d"])
        pred_no_label = model.predict(fm_no_label)

        pd.testing.assert_series_equal(pred_with_label, pred_no_label, check_names=False)

    def test_prediction_is_real_valued(self):
        """prediction 输出为有限实数（无 Inf）。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix()
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=10)
        model.fit(fm)
        pred = model.predict(fm)
        valid = pred.dropna()
        assert np.all(np.isfinite(valid.values))
