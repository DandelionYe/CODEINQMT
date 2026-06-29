# -*- coding: utf-8 -*-
"""
tests/test_records.py

Record Template 模块单元测试。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.records import (
    BaseRecord,
    SignalEvaluationRecord,
    WalkForwardRecord,
    DiagnosisRecord,
    RobustnessRecord,
    _safe_float,
    _safe_int,
    _classify_signal_quality,
    _json_default,
)


# ---------------------------------------------------------------------------
# 辅助函数测试
# ---------------------------------------------------------------------------


class TestSafeHelpers:
    """辅助函数测试。"""

    def test_safe_float_valid(self):
        assert _safe_float(1.5) == 1.5
        assert _safe_float(0) == 0.0
        assert _safe_float("3.14") == 3.14

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_nan(self):
        assert _safe_float(np.nan) is None

    def test_safe_float_invalid(self):
        assert _safe_float("abc") is None

    def test_safe_int_valid(self):
        assert _safe_int(5) == 5
        assert _safe_int(3.7) == 3
        assert _safe_int("10") == 10

    def test_safe_int_none(self):
        assert _safe_int(None) is None

    def test_safe_int_invalid(self):
        assert _safe_int("abc") is None

    def test_classify_signal_quality_strong(self):
        assert _classify_signal_quality(0.05, 0.8) == "STRONG"
        assert _classify_signal_quality(-0.04, 0.5) == "STRONG"

    def test_classify_signal_quality_moderate(self):
        assert _classify_signal_quality(0.02, 0.3) == "MODERATE"
        assert _classify_signal_quality(0.015, 0.6) == "MODERATE+"

    def test_classify_signal_quality_weak(self):
        assert _classify_signal_quality(0.005, 0.1) == "WEAK"

    def test_classify_signal_quality_unknown(self):
        assert _classify_signal_quality(None, None) == "UNKNOWN"


# ---------------------------------------------------------------------------
# BaseRecord 测试
# ---------------------------------------------------------------------------


class TestBaseRecord:
    """BaseRecord 基类测试。"""

    def test_to_dict(self):
        rec = BaseRecord(record_type="test", variant="v1")
        d = rec.to_dict()
        assert d["record_type"] == "test"
        assert d["variant"] == "v1"

    def test_save_and_load(self, tmp_path):
        rec = BaseRecord(record_type="test", variant="v1")
        path = tmp_path / "record.json"
        rec.save(path)
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["record_type"] == "test"
        assert data["variant"] == "v1"

    def test_save_creates_parent_dirs(self, tmp_path):
        rec = BaseRecord(record_type="test")
        path = tmp_path / "deep" / "nested" / "record.json"
        rec.save(path)
        assert path.exists()

    def test_apply_to_recorder_noop(self, tmp_path):
        """基类 apply_to_recorder 不应报错。"""
        rec = BaseRecord(record_type="test")
        rec.apply_to_recorder(None)  # 不应抛异常


# ---------------------------------------------------------------------------
# SignalEvaluationRecord 测试
# ---------------------------------------------------------------------------


class TestSignalEvaluationRecord:
    """SignalEvaluationRecord 测试。"""

    def _make_eval_dir(self, tmp_path: Path) -> Path:
        """创建模拟的 signal_evaluation 输出目录。"""
        eval_dir = tmp_path / "eval_output"
        eval_dir.mkdir()

        # IC summary
        ic_summary = pd.DataFrame([{
            "label_col": "ret_1d",
            "ic_mean": 0.035,
            "ic_std": 0.08,
            "rank_ic_mean": 0.04,
            "rank_ic_std": 0.09,
            "icir": 0.44,
            "ic_t_stat": 2.5,
            "ic_positive_rate": 0.62,
            "n_days": 500,
        }])
        ic_summary.to_csv(eval_dir / "signal_ic_summary.csv", index=False, encoding="utf-8-sig")

        # Quantile returns
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        quantile = pd.DataFrame({
            "Q1": np.random.randn(100) * 0.001,
            "Q2": np.random.randn(100) * 0.001,
            "Q3": np.random.randn(100) * 0.001,
            "Q4": np.random.randn(100) * 0.001,
            "Q5": np.random.randn(100) * 0.001 + 0.002,
        }, index=dates)
        quantile["long_short"] = quantile["Q5"] - quantile["Q1"]
        quantile.to_csv(eval_dir / "signal_quantile_return.csv", encoding="utf-8-sig")

        # Coverage
        coverage = pd.DataFrame({"coverage": np.random.uniform(0.8, 1.0, 100)}, index=dates)
        coverage.to_csv(eval_dir / "signal_coverage.csv", encoding="utf-8-sig")

        # Autocorr
        autocorr = pd.DataFrame({"lag": [1, 2, 3], "autocorr": [0.85, 0.72, 0.60]})
        autocorr.to_csv(eval_dir / "signal_autocorr.csv", index=False, encoding="utf-8-sig")

        return eval_dir

    def test_from_dir_full(self, tmp_path):
        eval_dir = self._make_eval_dir(tmp_path)
        rec = SignalEvaluationRecord.from_dir(eval_dir, variant="short_term_reversal", label_col="ret_1d")

        assert rec.record_type == "signal_evaluation"
        assert rec.variant == "short_term_reversal"
        assert rec.label_col == "ret_1d"
        assert rec.ic_mean == pytest.approx(0.035)
        assert rec.rank_ic_mean == pytest.approx(0.04)
        assert rec.icir == pytest.approx(0.44)
        assert rec.ic_positive_rate == pytest.approx(0.62)
        assert rec.n_days == 500
        assert rec.autocorr_lag1 == pytest.approx(0.85)
        assert rec.signal_quality == "STRONG"
        assert rec.coverage_mean is not None
        assert rec.long_short_daily_mean is not None

    def test_from_dir_missing_files(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        rec = SignalEvaluationRecord.from_dir(empty_dir, variant="test")
        assert rec.ic_mean is None
        assert rec.signal_quality == "UNKNOWN"

    def test_from_dir_nonexistent(self, tmp_path):
        rec = SignalEvaluationRecord.from_dir(tmp_path / "nonexistent", variant="test")
        assert rec.ic_mean is None

    def test_to_dict(self, tmp_path):
        eval_dir = self._make_eval_dir(tmp_path)
        rec = SignalEvaluationRecord.from_dir(eval_dir, variant="test")
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert d["record_type"] == "signal_evaluation"
        assert "ic_mean" in d
        assert "signal_quality" in d

    def test_save_and_load(self, tmp_path):
        eval_dir = self._make_eval_dir(tmp_path)
        rec = SignalEvaluationRecord.from_dir(eval_dir, variant="test")
        path = tmp_path / "signal_eval_record.json"
        rec.save(path)
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["record_type"] == "signal_evaluation"
        assert data["ic_mean"] == pytest.approx(0.035)

    def test_apply_to_recorder(self, tmp_path):
        eval_dir = self._make_eval_dir(tmp_path)
        rec = SignalEvaluationRecord.from_dir(eval_dir, variant="test")

        from scripts.common.recorder import RunRecorder
        recorder = RunRecorder("exp_test", run_id="run_001", base_dir=tmp_path)
        rec.apply_to_recorder(recorder)

        assert recorder.manifest.metrics["signal_ic_mean"] == pytest.approx(0.035)
        assert recorder.manifest.metrics["signal_rank_ic_mean"] == pytest.approx(0.04)
        assert recorder.manifest.metrics["signal_icir"] == pytest.approx(0.44)
        assert recorder.manifest.metrics["signal_quality"] == "STRONG"


# ---------------------------------------------------------------------------
# WalkForwardRecord 测试
# ---------------------------------------------------------------------------


class TestWalkForwardRecord:
    """WalkForwardRecord 测试。"""

    def _make_wf_dir(self, tmp_path: Path) -> Path:
        """创建模拟的 walk-forward 输出目录。"""
        wf_dir = tmp_path / "wf_output"
        wf_dir.mkdir()

        # 模拟 3 只股票的 summary
        for stock in ["000001", "000002", "600000"]:
            df = pd.DataFrame({
                "test_year": [2021, 2022, 2023, 2024],
                "excess_return": [0.05, -0.02, 0.08, 0.03],
                "sharpe": [1.2, -0.3, 1.5, 0.6],
                "max_drawdown": [-0.15, -0.25, -0.10, -0.12],
            })
            df.to_csv(wf_dir / f"wf_alpha_v7_stock_{stock}_summary.csv", index=False, encoding="utf-8-sig")

        return wf_dir

    def test_from_dir(self, tmp_path):
        wf_dir = self._make_wf_dir(tmp_path)
        rec = WalkForwardRecord.from_dir(wf_dir, variant="short_term_reversal")

        assert rec.record_type == "walk_forward"
        assert rec.variant == "short_term_reversal"
        assert rec.n_stocks == 3
        assert rec.n_test_years == 4
        assert rec.test_years == [2021, 2022, 2023, 2024]
        assert rec.mean_excess_return is not None
        assert rec.win_rate is not None
        assert rec.mean_sharpe is not None
        assert rec.mean_max_drawdown is not None

    def test_from_dir_empty(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        rec = WalkForwardRecord.from_dir(empty_dir, variant="test")
        assert rec.n_stocks is None

    def test_from_dir_nonexistent(self, tmp_path):
        rec = WalkForwardRecord.from_dir(tmp_path / "nonexistent", variant="test")
        assert rec.n_stocks is None

    def test_apply_to_recorder(self, tmp_path):
        wf_dir = self._make_wf_dir(tmp_path)
        rec = WalkForwardRecord.from_dir(wf_dir, variant="test")

        from scripts.common.recorder import RunRecorder
        recorder = RunRecorder("exp_test", run_id="run_001", base_dir=tmp_path)
        rec.apply_to_recorder(recorder)

        assert "wf_n_stocks" in recorder.manifest.metrics
        assert "wf_mean_excess_return" in recorder.manifest.metrics
        assert "wf_win_rate" in recorder.manifest.metrics


# ---------------------------------------------------------------------------
# DiagnosisRecord 测试
# ---------------------------------------------------------------------------


class TestDiagnosisRecord:
    """DiagnosisRecord 测试。"""

    def _make_diag_dir(self, tmp_path: Path) -> Path:
        """创建模拟的 diagnosis 输出目录。"""
        diag_dir = tmp_path / "diag_output"
        diag_dir.mkdir()

        # Train-test gap
        gap = pd.DataFrame({
            "test_year": [2021, 2022, 2023, 2024],
            "train_annual_return": [0.15, 0.12, 0.18, 0.10],
            "test_total_return": [0.05, -0.02, 0.08, 0.03],
            "gap": [0.10, 0.14, 0.10, 0.07],
        })
        gap.to_csv(diag_dir / "alpha_v7_diagnosis_train_test_gap.csv", index=False, encoding="utf-8-sig")

        return diag_dir

    def test_from_dir(self, tmp_path):
        diag_dir = self._make_diag_dir(tmp_path)
        rec = DiagnosisRecord.from_dir(diag_dir, variant="short_term_reversal")

        assert rec.record_type == "diagnosis"
        assert rec.variant == "short_term_reversal"
        assert rec.mean_train_return is not None
        assert rec.mean_test_return is not None
        assert rec.mean_gap is not None
        assert rec.yearly_consistency == pytest.approx(0.75)  # 3/4 年正收益

    def test_from_dir_empty(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        rec = DiagnosisRecord.from_dir(empty_dir, variant="test")
        assert rec.mean_train_return is None

    def test_apply_to_recorder(self, tmp_path):
        diag_dir = self._make_diag_dir(tmp_path)
        rec = DiagnosisRecord.from_dir(diag_dir, variant="test")

        from scripts.common.recorder import RunRecorder
        recorder = RunRecorder("exp_test", run_id="run_001", base_dir=tmp_path)
        rec.apply_to_recorder(recorder)

        assert "diag_mean_train_return" in recorder.manifest.metrics
        assert "diag_mean_test_return" in recorder.manifest.metrics
        assert "diag_yearly_consistency" in recorder.manifest.metrics


# ---------------------------------------------------------------------------
# RobustnessRecord 测试
# ---------------------------------------------------------------------------


class TestRobustnessRecord:
    """RobustnessRecord 测试。"""

    def _make_robust_dir(self, tmp_path: Path) -> Path:
        """创建模拟的 robustness 输出目录。"""
        robust_dir = tmp_path / "robust_output"
        robust_dir.mkdir()

        # Manifest
        manifest = {
            "overall_pass": True,
            "gates": {
                "min_sharpe": {"pass": True, "value": 0.85, "threshold": 0.5},
                "max_drawdown": {"pass": True, "value": -0.15, "threshold": -0.25},
                "win_rate": {"pass": False, "value": 0.55, "threshold": 0.60},
            },
        }
        (robust_dir / "robustness_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # LOYO scenarios
        scenarios = pd.DataFrame({
            "scenario": ["leave_2021", "leave_2022", "leave_2023", "leave_2024"],
            "excess_return": [0.05, -0.03, 0.08, 0.02],
        })
        scenarios.to_csv(robust_dir / "robustness_scenarios.csv", index=False, encoding="utf-8-sig")

        # Benchmark comparison
        bench = pd.DataFrame({
            "benchmark": ["000300.SH", "000905.SH"],
            "excess_return": [0.06, 0.04],
        })
        bench.to_csv(robust_dir / "robustness_benchmark_comparison.csv", index=False, encoding="utf-8-sig")

        return robust_dir

    def test_from_dir(self, tmp_path):
        robust_dir = self._make_robust_dir(tmp_path)
        rec = RobustnessRecord.from_dir(robust_dir, variant="short_term_reversal")

        assert rec.record_type == "robustness"
        assert rec.variant == "short_term_reversal"
        assert rec.gates_total == 3
        assert rec.gates_passed == 2
        assert rec.gates_failed == ["win_rate"]
        assert rec.overall_pass is True
        assert rec.loyo_worst_year == "leave_2022"
        assert rec.loyo_worst_excess == pytest.approx(-0.03)
        assert rec.benchmark_excess == pytest.approx(0.05)

    def test_from_dir_empty(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        rec = RobustnessRecord.from_dir(empty_dir, variant="test")
        assert rec.gates_total == 0
        assert rec.overall_pass is False

    def test_from_dir_nonexistent(self, tmp_path):
        rec = RobustnessRecord.from_dir(tmp_path / "nonexistent", variant="test")
        assert rec.gates_total == 0

    def test_apply_to_recorder(self, tmp_path):
        robust_dir = self._make_robust_dir(tmp_path)
        rec = RobustnessRecord.from_dir(robust_dir, variant="test")

        from scripts.common.recorder import RunRecorder
        recorder = RunRecorder("exp_test", run_id="run_001", base_dir=tmp_path)
        rec.apply_to_recorder(recorder)

        assert recorder.manifest.metrics["robust_gates_passed"] == 2
        assert recorder.manifest.metrics["robust_gates_total"] == 3
        assert recorder.manifest.metrics["robust_overall_pass"] is True
        assert recorder.manifest.metrics["robust_loyo_worst_excess"] == pytest.approx(-0.03)


# ---------------------------------------------------------------------------
# 集成测试：Record + RunRecorder 完整工作流
# ---------------------------------------------------------------------------


class TestRecordIntegration:
    """Record Template 与 RunRecorder 集成测试。"""

    def test_full_pipeline_records(self, tmp_path):
        """模拟完整 pipeline：创建 RunRecorder，依次应用各阶段 Record。"""
        from scripts.common.recorder import RunRecorder

        # 创建 RunRecorder
        recorder = RunRecorder("exp_007", run_id="run_e2e", base_dir=tmp_path, stage="full_pipeline")
        recorder.set_params({
            "alpha_variant": "short_term_reversal",
            "reversal_window": 10,
        })

        # 1. Signal Evaluation Record
        eval_dir = tmp_path / "signal_eval"
        eval_dir.mkdir()
        ic_summary = pd.DataFrame([{
            "label_col": "ret_1d", "ic_mean": 0.04, "ic_std": 0.08,
            "rank_ic_mean": 0.045, "rank_ic_std": 0.09, "icir": 0.5,
            "ic_t_stat": 3.0, "ic_positive_rate": 0.65, "n_days": 500,
        }])
        ic_summary.to_csv(eval_dir / "signal_ic_summary.csv", index=False, encoding="utf-8-sig")
        (eval_dir / "signal_quantile_return.csv").write_text("Q1,Q5,long_short\n0.001,0.003,0.002\n", encoding="utf-8-sig")
        (eval_dir / "signal_coverage.csv").write_text("coverage\n0.95\n", encoding="utf-8-sig")

        sig_rec = SignalEvaluationRecord.from_dir(eval_dir, variant="short_term_reversal", label_col="ret_1d")
        sig_rec.apply_to_recorder(recorder)
        sig_rec.save(tmp_path / "signal_eval_record.json")

        recorder.mark_stage("signal_evaluation", "completed")

        # 2. WalkForward Record
        wf_dir = tmp_path / "wf"
        wf_dir.mkdir()
        pd.DataFrame({
            "test_year": [2023, 2024],
            "excess_return": [0.08, 0.03],
            "sharpe": [1.5, 0.6],
            "max_drawdown": [-0.10, -0.12],
        }).to_csv(wf_dir / "wf_alpha_v7_stock_000001_summary.csv", index=False, encoding="utf-8-sig")

        wf_rec = WalkForwardRecord.from_dir(wf_dir, variant="short_term_reversal")
        wf_rec.apply_to_recorder(recorder)
        wf_rec.save(tmp_path / "wf_record.json")

        recorder.mark_stage("walk_forward", "completed")

        # 3. Robustness Record
        robust_dir = tmp_path / "robust"
        robust_dir.mkdir()
        (robust_dir / "robustness_manifest.json").write_text(
            json.dumps({"overall_pass": True, "gates": {"sharpe": {"pass": True}}}),
            encoding="utf-8",
        )

        rob_rec = RobustnessRecord.from_dir(robust_dir, variant="short_term_reversal")
        rob_rec.apply_to_recorder(recorder)
        rob_rec.save(tmp_path / "robust_record.json")

        recorder.mark_stage("robustness", "completed")

        # 完成
        recorder.finish()

        # 验证 manifest 中包含所有阶段的指标
        loaded = RunRecorder.load_manifest("exp_007", "run_e2e", base_dir=tmp_path)
        assert loaded.stage_status["signal_evaluation"] == "completed"
        assert loaded.stage_status["walk_forward"] == "completed"
        assert loaded.stage_status["robustness"] == "completed"
        assert loaded.metrics["signal_ic_mean"] == pytest.approx(0.04)
        assert loaded.metrics["signal_quality"] == "STRONG"
        assert loaded.metrics["wf_n_stocks"] == 1
        assert loaded.metrics["robust_overall_pass"] is True

        # 验证 Record JSON 文件
        assert (tmp_path / "signal_eval_record.json").exists()
        assert (tmp_path / "wf_record.json").exists()
        assert (tmp_path / "robust_record.json").exists()


# ---------------------------------------------------------------------------
# 测试：_json_default
# ---------------------------------------------------------------------------

class TestJsonDefault:
    """_json_default 的 numpy/pandas 序列化分支。"""

    def test_np_int64(self):
        """np.integer 应转为 Python int。"""
        val = np.int64(42)
        result = _json_default(val)
        assert result == 42
        assert isinstance(result, int)
        assert not isinstance(result, np.integer)

    def test_np_int32(self):
        """np.int32 应转为 Python int。"""
        val = np.int32(-7)
        result = _json_default(val)
        assert result == -7
        assert isinstance(result, int)

    def test_np_float64(self):
        """np.floating 应转为 Python float。"""
        val = np.float64(3.14)
        result = _json_default(val)
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)
        assert not isinstance(result, np.floating)

    def test_np_float32(self):
        """np.float32 应转为 Python float。"""
        val = np.float32(2.72)
        result = _json_default(val)
        assert result == pytest.approx(2.72, rel=1e-5)
        assert isinstance(result, float)

    def test_np_float_nan(self):
        """np.float64 NaN 应转为 Python float(nan)。"""
        val = np.float64("nan")
        result = _json_default(val)
        assert isinstance(result, float)
        assert np.isnan(result)

    def test_np_array(self):
        """np.ndarray 应转为 Python list。"""
        val = np.array([1.0, 2.0, 3.0])
        result = _json_default(val)
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_np_array_2d(self):
        """二维 np.ndarray 应转为嵌套 list。"""
        val = np.array([[1, 2], [3, 4]])
        result = _json_default(val)
        assert isinstance(result, list)
        assert result == [[1, 2], [3, 4]]

    def test_pd_timestamp(self):
        """pd.Timestamp 应转为 ISO 格式字符串。"""
        val = pd.Timestamp("2026-06-08 10:30:00")
        result = _json_default(val)
        assert isinstance(result, str)
        assert "2026-06-08" in result

    def test_unsupported_type_raises(self):
        """不支持的类型应抛出 TypeError。"""
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default(object())

    def test_save_with_numpy_values(self, tmp_path):
        """BaseRecord.save() 含 numpy 值时应通过 _json_default 正确序列化。
        使用 np.int32/float32 确保触发 _json_default（它们不继承 Python int/float）。
        """
        from dataclasses import dataclass as _dc

        @_dc
        class NumpyRecord(BaseRecord):
            ic_mean: float = 0.0
            n_stocks: int = 0

        rec = NumpyRecord(
            record_type="numpy_test",
            ic_mean=np.float32(0.042),
            n_stocks=np.int32(150),
        )
        out_path = tmp_path / "rec.json"
        rec.save(out_path)
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["ic_mean"] == pytest.approx(0.042, rel=1e-4)
        assert loaded["n_stocks"] == 150
