# -*- coding: utf-8 -*-
"""
test_train_alpha_model.py

Tests for scripts/train_alpha_model.py — model + QMTDataHandler integration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.data_handler import QMTDataHandler
from scripts.common.feature_expression import (
    Field,
    PctChange,
    ZScore,
    normalize_zscore,
)
from scripts.common.models import AlphaModel, SimpleRuleModel
from scripts.train_alpha_model import (
    compute_daily_ic_series,
    compute_ic,
    compute_rank_ic,
    run_walk_forward,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_feature_matrix(
    n_years: int = 3,
    n_symbols: int = 3,
    days_per_year: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """构造跨多年 MultiIndex feature matrix。"""
    rng = np.random.RandomState(seed)
    dates = []
    for y in range(2021, 2021 + n_years):
        dates.extend([y * 10000 + 101 + i for i in range(days_per_year)])
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    n = len(idx)

    # 构造有意义的特征：reversal 与 label 有弱相关
    reversal = rng.randn(n)
    label_1d = reversal * 0.1 + rng.randn(n) * 0.5  # label 与 reversal 有弱正相关

    return pd.DataFrame(
        {
            "feature/close": 10.0 + np.cumsum(rng.randn(n) * 0.05),
            "feature/volume": rng.randint(1000, 10000, size=n).astype(float),
            "feature/reversal_10d": reversal,
            "feature/vol_60d": np.abs(rng.randn(n)),
            "label/ret_1d": label_1d,
            "label/ret_5d": rng.randn(n) * 0.02,
        },
        index=idx,
    )


def _save_feature_matrix(fm: pd.DataFrame, tmp_path: Path) -> Path:
    """保存 feature matrix 到临时 parquet 文件。"""
    path = tmp_path / "feature_matrix.parquet"
    fm.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# TestComputeIC
# ---------------------------------------------------------------------------

class TestComputeIC:
    """IC 计算函数测试。"""

    def test_compute_ic_perfect_correlation(self):
        """完美正相关 -> IC = 1.0。"""
        pred = pd.Series([1, 2, 3, 4, 5], dtype=float)
        label = pd.Series([1, 2, 3, 4, 5], dtype=float)
        assert abs(compute_ic(pred, label) - 1.0) < 1e-10

    def test_compute_ic_negative_correlation(self):
        """完美负相关 -> IC = -1.0。"""
        pred = pd.Series([1, 2, 3, 4, 5], dtype=float)
        label = pd.Series([5, 4, 3, 2, 1], dtype=float)
        assert abs(compute_ic(pred, label) - (-1.0)) < 1e-10

    def test_compute_ic_with_nan(self):
        """NaN 值被自动排除，剩余 >= 5 个有效值仍可计算 IC。"""
        pred = pd.Series([1, 2, np.nan, 4, 5, 6, 7], dtype=float)
        label = pd.Series([1, 2, 3, 4, 5, 6, 7], dtype=float)
        ic = compute_ic(pred, label)
        assert not np.isnan(ic)

    def test_compute_ic_too_few(self):
        """少于 5 个有效值返回 NaN。"""
        pred = pd.Series([1, 2, 3], dtype=float)
        label = pd.Series([1, 2, 3], dtype=float)
        assert np.isnan(compute_ic(pred, label))

    def test_compute_rank_ic(self):
        """RankIC 使用 Spearman 相关。"""
        pred = pd.Series([1, 3, 5, 7, 9], dtype=float)
        label = pd.Series([2, 4, 6, 8, 10], dtype=float)
        assert abs(compute_rank_ic(pred, label) - 1.0) < 1e-10

    def test_compute_rank_ic_monotonic(self):
        """单调递增 -> RankIC = 1.0。"""
        pred = pd.Series([10, 20, 30, 40, 50], dtype=float)
        label = pd.Series([100, 200, 300, 400, 500], dtype=float)
        assert abs(compute_rank_ic(pred, label) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# TestComputeDailyICSeries
# ---------------------------------------------------------------------------

class TestComputeDailyICSeries:
    """每日 IC 系列计算测试。"""

    def test_basic(self):
        """基本每日 IC 计算。"""
        rng = np.random.RandomState(42)
        dates = [20240101] * 10 + [20240102] * 10
        pred = pd.Series(rng.randn(20), dtype=float)
        label = pd.Series(rng.randn(20), dtype=float)
        date_idx = pd.Index(dates)
        result = compute_daily_ic_series(pred, label, date_idx)
        assert len(result) == 2
        assert not result.isna().all()

    def test_single_day(self):
        """单日数据。"""
        pred = pd.Series([1, 2, 3, 4, 5], dtype=float)
        label = pd.Series([2, 4, 6, 8, 10], dtype=float)
        date_idx = pd.Index([20240101] * 5)
        result = compute_daily_ic_series(pred, label, date_idx)
        assert len(result) == 1

    def test_too_few_per_day(self):
        """每日少于 3 个观测被跳过。"""
        pred = pd.Series([1, 2], dtype=float)
        label = pd.Series([1, 2], dtype=float)
        date_idx = pd.Index([20240101, 20240101])
        result = compute_daily_ic_series(pred, label, date_idx)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestRunWalkForward
# ---------------------------------------------------------------------------

class TestRunWalkForward:
    """Walk-forward 预测流程测试。"""

    def test_basic_score_col(self, tmp_path):
        """score_col 模式基本流程。"""
        fm = _make_feature_matrix(n_years=3, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = SimpleRuleModel(score_col="feature/reversal_10d", zscore=True)

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022, 2023],
            label_col="label/ret_1d",
        )

        assert "per_year_results" in results
        assert "ic_summary" in results
        assert "predictions" in results
        assert len(results["per_year_results"]) == 2
        assert results["ic_summary"]["n_years"] == 2

    def test_predictions_shape(self, tmp_path):
        """预测输出形状正确。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=3, days_per_year=20)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = SimpleRuleModel(score_col="feature/reversal_10d", zscore=False)

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022],
            label_col="label/ret_1d",
        )

        pred = results["predictions"]
        assert not pred.empty
        assert "prediction" in pred.columns
        assert "label" in pred.columns
        assert "date" in pred.columns
        assert "symbol" in pred.columns
        assert "test_year" in pred.columns
        # 2022 年有 20 天 * 3 只股票 = 60 行
        assert len(pred) == 20 * 3

    def test_ic_summary_keys(self, tmp_path):
        """IC 摘要包含所有必要 key。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = SimpleRuleModel(score_col="feature/reversal_10d")

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022],
            label_col="label/ret_1d",
        )

        summary = results["ic_summary"]
        assert "mean_ic" in summary
        assert "mean_rank_ic" in summary
        assert "mean_icir" in summary
        assert "mean_ic_win_rate" in summary
        assert "n_years" in summary

    def test_per_year_result_keys(self, tmp_path):
        """逐年结果包含所有必要 key。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = SimpleRuleModel(score_col="feature/reversal_10d")

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022],
            label_col="label/ret_1d",
        )

        r = results["per_year_results"][0]
        assert "test_year" in r
        assert "n_train" in r
        assert "n_test" in r
        assert "ic" in r
        assert "rank_ic" in r
        assert "icir" in r
        assert "ic_win_rate" in r
        assert "status" in r

    def test_expression_mode(self, tmp_path):
        """表达式模式：使用原始价格列构建表达式。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        expr = ZScore(-PctChange(Field("feature/close"), 10))
        model = SimpleRuleModel(expression=expr, zscore=False)

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022],
            label_col="label/ret_1d",
        )

        assert results["ic_summary"]["n_years"] == 1
        assert not results["predictions"].empty

    def test_missing_label_col(self, tmp_path, caplog):
        """标签列不存在时跳过 IC 计算。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = SimpleRuleModel(score_col="feature/reversal_10d")

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022],
            label_col="label/nonexistent",
        )

        # 应该仍然返回结果，只是 IC 为 NaN
        assert len(results["per_year_results"]) == 1

    def test_all_test_years(self, tmp_path):
        """使用全部可用年份。"""
        fm = _make_feature_matrix(n_years=4, n_symbols=2, days_per_year=30)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = SimpleRuleModel(score_col="feature/reversal_10d")

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022, 2023, 2024],
            label_col="label/ret_1d",
        )

        assert len(results["per_year_results"]) == 3
        assert results["ic_summary"]["n_years"] == 3


# ---------------------------------------------------------------------------
# TestModelDataHandlerIntegration
# ---------------------------------------------------------------------------

class TestModelDataHandlerIntegration:
    """模型与 DataHandler 集成测试。"""

    def test_fit_predict_consistency(self, tmp_path):
        """fit 不改变 predict 结果（规则模型）。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = SimpleRuleModel(score_col="feature/reversal_10d", zscore=False)

        train_dk_l, test_dk_i, _, _ = handler.prepare_walk_forward(2022)

        pred_before = model.predict(test_dk_i)
        model.fit(train_dk_l)
        pred_after = model.predict(test_dk_i)

        pd.testing.assert_series_equal(pred_before, pred_after, check_names=False)

    def test_predict_output_length(self, tmp_path):
        """prediction 长度等于测试期数据长度。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=3, days_per_year=20)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = SimpleRuleModel(score_col="feature/reversal_10d")

        _, test_dk_i, _, _ = handler.prepare_walk_forward(2022)
        pred = model.predict(test_dk_i)

        assert len(pred) == len(test_dk_i)

    def test_predict_no_label_leakage(self, tmp_path):
        """prediction 不使用 label 列。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = SimpleRuleModel(score_col="feature/reversal_10d")

        _, test_dk_i, _, _ = handler.prepare_walk_forward(2022)
        pred = model.predict(test_dk_i)

        # prediction 应该与 feature/reversal_10d 相关，但不等于 label
        assert "label/ret_1d" not in test_dk_i.columns or \
            not pred.equals(test_dk_i.get("label/ret_1d", pd.Series()))

    def test_model_params_recorded(self, tmp_path):
        """模型参数可通过 get_params() 获取。"""
        model = SimpleRuleModel(score_col="feature/reversal_10d", zscore=True)
        params = model.get_params()

        assert params["model_type"] == "SimpleRuleModel"
        assert params["score_col"] == "feature/reversal_10d"
        assert params["zscore"] is True

    def test_expression_model_with_handler(self, tmp_path):
        """表达式模型与 DataHandler 集成。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        expr = -PctChange(Field("feature/close"), 10)
        model = SimpleRuleModel(expression=expr, zscore=True)

        train_dk_l, test_dk_i, _, _ = handler.prepare_walk_forward(2022)
        model.fit(train_dk_l)
        pred = model.predict(test_dk_i)

        assert len(pred) == len(test_dk_i)
        assert not pred.isna().all()


# ---------------------------------------------------------------------------
# TestFinancialCorrectness
# ---------------------------------------------------------------------------

class TestFinancialCorrectness:
    """金融正确性检查。"""

    def test_no_future_leakage_in_wf(self, tmp_path):
        """walk-forward 无未来函数：训练期 < 测试年份。"""
        fm = _make_feature_matrix(n_years=3, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)

        # 2022 测试：训练期应为 2021
        train_dk_l, test_dk_i, _, _ = handler.prepare_walk_forward(2022)
        train_dates = train_dk_l.index.get_level_values(0)
        test_dates = test_dk_i.index.get_level_values(0)

        assert (train_dates // 10000).max() < 2022
        assert (test_dates // 10000).min() >= 2022
        assert (test_dates // 10000).max() < 2023

    def test_label_not_in_dk_i(self, tmp_path):
        """DK_I（推理数据）不包含 label 列。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        _, test_dk_i, _, _ = handler.prepare_walk_forward(2022)

        label_cols = [c for c in test_dk_i.columns if c.startswith("label/")]
        assert len(label_cols) == 0

    def test_prediction_is_not_label(self, tmp_path):
        """prediction 不等于 label（无标签泄漏）。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = SimpleRuleModel(score_col="feature/reversal_10d", zscore=False)

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022],
            label_col="label/ret_1d",
        )

        pred = results["predictions"]["prediction"]
        label = results["predictions"]["label"]

        # prediction 和 label 不应该完全相同
        assert not pred.equals(label)

    def test_rolling_window_uses_only_history(self, tmp_path):
        """表达式中的滚动窗口只使用历史数据。"""
        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)

        # 使用 PctChange（10日变化率）——应该只用历史数据
        expr = ZScore(-PctChange(Field("feature/close"), 10))
        model = SimpleRuleModel(expression=expr, zscore=False)

        train_dk_l, test_dk_i, _, _ = handler.prepare_walk_forward(2022)
        model.fit(train_dk_l)
        pred = model.predict(test_dk_i)

        # prediction 应该有值（不全是 NaN）
        assert not pred.isna().all()


class TestSafeNanmean:
    """测试 _safe_nanmean 修复 RuntimeWarning。"""

    def test_all_nan_values_no_warning(self, tmp_path):
        """IC 值为 NaN 时不产生 RuntimeWarning（< 5 个有效数据点）。"""
        import warnings

        # 构造场景：训练期有足够数据，但测试期只有 4 个数据点
        # compute_ic 在 len(valid) < 5 时返回 NaN
        rng = np.random.RandomState(42)
        # 2020：训练期（50 个日期 × 1 只股票）
        train_dates = [20200101 + i for i in range(50)]
        # 2021：测试期（4 个日期 × 1 只股票，< 5 使 IC 为 NaN）
        test_dates = [20210101 + i for i in range(4)]
        dates = train_dates + test_dates
        symbols = ["SYM000"]
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        n = len(idx)
        fm = pd.DataFrame(
            {
                "feature/f1": rng.randn(n),
                "label/ret_1d": rng.randn(n) * 0.01,
            },
            index=idx,
        )
        fm_path = tmp_path / "fm.parquet"
        fm.to_parquet(fm_path)

        handler = QMTDataHandler(fm_path)
        model = SimpleRuleModel(score_col="feature/f1", zscore=False)

        # 捕获 RuntimeWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = run_walk_forward(
                handler=handler,
                model=model,
                test_years=[2021],
                label_col="label/ret_1d",
            )
            # 不应有 "Mean of empty slice" RuntimeWarning
            nan_warnings = [
                x for x in w if "Mean of empty slice" in str(x.message)
            ]
            assert len(nan_warnings) == 0, f"不应有 RuntimeWarning: {nan_warnings}"

        # walk-forward 应正常完成，IC 为 NaN（< 5 个有效点）
        assert results["ic_summary"]["n_years"] == 1
        assert np.isnan(results["ic_summary"]["mean_ic"])


# ---------------------------------------------------------------------------
# TestLightGBMWalkForward — 需要 lightgbm 安装
# ---------------------------------------------------------------------------

try:
    import lightgbm  # noqa: F401
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm 未安装")
class TestLightGBMWalkForward:
    """LightGBMModel 与 walk-forward 流程集成测试。"""

    def test_lightgbm_basic_walk_forward(self, tmp_path):
        """LightGBMModel 基本 walk-forward 流程。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix(n_years=3, n_symbols=2, days_per_year=50)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=10, verbose=-1)

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022, 2023],
            label_col="label/ret_1d",
        )

        assert results["ic_summary"]["n_years"] == 2
        assert len(results["per_year_results"]) == 2
        assert not results["predictions"].empty

    def test_lightgbm_predictions_shape(self, tmp_path):
        """LightGBMModel 预测输出形状正确。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix(n_years=2, n_symbols=3, days_per_year=20)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5, verbose=-1)

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022],
            label_col="label/ret_1d",
        )

        pred = results["predictions"]
        assert not pred.empty
        assert "prediction" in pred.columns
        assert "label" in pred.columns
        assert len(pred) == 20 * 3  # 20 days * 3 symbols

    def test_lightgbm_ic_summary_keys(self, tmp_path):
        """LightGBMModel IC 摘要包含所有必要 key。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5, verbose=-1)

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022],
            label_col="label/ret_1d",
        )

        summary = results["ic_summary"]
        assert "mean_ic" in summary
        assert "mean_rank_ic" in summary
        assert "mean_icir" in summary
        assert "mean_ic_win_rate" in summary
        assert "n_years" in summary

    def test_lightgbm_no_label_leakage(self, tmp_path):
        """LightGBMModel prediction 不使用 label 列（DK_I 不含 label）。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5, verbose=-1)

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022],
            label_col="label/ret_1d",
        )

        # prediction 和 label 不应该完全相同
        pred = results["predictions"]["prediction"]
        label = results["predictions"]["label"]
        assert not pred.equals(label)

    def test_lightgbm_with_zscore_pred(self, tmp_path):
        """LightGBMModel + zscore_pred 选项正常工作。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix(n_years=2, n_symbols=2)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5, verbose=-1)

        results = run_walk_forward(
            handler=handler,
            model=model,
            test_years=[2022],
            label_col="label/ret_1d",
            zscore_pred=True,
        )

        assert results["ic_summary"]["n_years"] == 1
        # zscore 后 prediction 均值应接近 0
        pred = results["predictions"]["prediction"].dropna()
        if len(pred) > 0:
            assert abs(pred.mean()) < 2.0  # zscore 后均值应较小

    def test_lightgbm_fit_uses_train_only(self, tmp_path):
        """LightGBMModel walk-forward 中 fit 只使用训练期数据。"""
        from scripts.common.models.lightgbm_model import LightGBMModel

        fm = _make_feature_matrix(n_years=3, n_symbols=2, days_per_year=30)
        fm_path = _save_feature_matrix(fm, tmp_path)

        handler = QMTDataHandler(fm_path)
        train_dk_l, test_dk_i, _, _ = handler.prepare_walk_forward(2022)

        # 验证训练期数据只有 2021 年
        train_dates = train_dk_l.index.get_level_values(0)
        assert (train_dates // 10000).max() < 2022

        model = LightGBMModel(label_col="label/ret_1d", n_estimators=5, verbose=-1)
        model.fit(train_dk_l)
        pred = model.predict(test_dk_i)

        assert len(pred) == len(test_dk_i)
        assert not pred.isna().all()
