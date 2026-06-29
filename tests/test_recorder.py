# -*- coding: utf-8 -*-
"""
tests/test_recorder.py

RunRecorder 单元测试。
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.recorder import RunManifest, RunRecorder, _generate_run_id, _utc_now_iso


# ---------------------------------------------------------------------------
# RunManifest 测试
# ---------------------------------------------------------------------------


class TestRunManifest:
    """RunManifest 数据类测试。"""

    def test_default_values(self):
        m = RunManifest(experiment_id="exp_001", run_id="run_001")
        assert m.experiment_id == "exp_001"
        assert m.run_id == "run_001"
        assert m.stage == ""
        assert m.status == "running"
        assert m.params == {}
        assert m.metrics == {}
        assert m.artifacts == {}
        assert m.stage_status == {}

    def test_to_dict(self):
        m = RunManifest(
            experiment_id="exp_001",
            run_id="run_001",
            params={"window": 5},
            metrics={"sharpe": 1.2},
        )
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["experiment_id"] == "exp_001"
        assert d["params"]["window"] == 5
        assert d["metrics"]["sharpe"] == 1.2

    def test_save_and_load(self, tmp_path):
        m = RunManifest(
            experiment_id="exp_001",
            run_id="run_001",
            params={"alpha": 0.05},
            metrics={"ic_mean": 0.03},
            artifacts={"report": "/path/to/report.csv"},
        )
        path = tmp_path / "run_manifest.json"
        m.save(path)

        assert path.exists()
        loaded = RunManifest.load(path)
        assert loaded.experiment_id == "exp_001"
        assert loaded.run_id == "run_001"
        assert loaded.params["alpha"] == 0.05
        assert loaded.metrics["ic_mean"] == 0.03
        assert loaded.artifacts["report"] == "/path/to/report.csv"

    def test_save_creates_parent_dirs(self, tmp_path):
        m = RunManifest(experiment_id="exp_001", run_id="run_001")
        path = tmp_path / "deep" / "nested" / "run_manifest.json"
        m.save(path)
        assert path.exists()

    def test_save_json_format(self, tmp_path):
        """验证 JSON 格式正确（可读、有缩进）。"""
        m = RunManifest(experiment_id="exp_001", run_id="run_001")
        path = tmp_path / "run_manifest.json"
        m.save(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["experiment_id"] == "exp_001"


# ---------------------------------------------------------------------------
# RunRecorder 测试
# ---------------------------------------------------------------------------


class TestRunRecorder:
    """RunRecorder 类测试。"""

    def test_init_with_defaults(self, tmp_path):
        rec = RunRecorder("exp_001", base_dir=tmp_path)
        assert rec.experiment_id == "exp_001"
        assert rec.run_id.startswith("run_")
        assert len(rec.run_id) == 12  # "run_" + 8 hex chars

    def test_init_with_explicit_run_id(self, tmp_path):
        rec = RunRecorder("exp_001", run_id="run_abc12345", base_dir=tmp_path)
        assert rec.run_id == "run_abc12345"
        assert rec.run_dir == tmp_path / "exp_001" / "runs" / "run_abc12345"

    def test_init_with_provenance(self, tmp_path):
        """验证 config_hash 和 code_version 构造参数写入 manifest。"""
        rec = RunRecorder(
            "exp_001", run_id="run_prov", base_dir=tmp_path,
            config_hash="abc123", code_version="deadbeef",
        )
        assert rec.manifest.config_hash == "abc123"
        assert rec.manifest.code_version == "deadbeef"

    def test_init_provenance_defaults_empty(self, tmp_path):
        """不传 provenance 参数时默认为空字符串。"""
        rec = RunRecorder("exp_001", run_id="run_noprov", base_dir=tmp_path)
        assert rec.manifest.config_hash == ""
        assert rec.manifest.code_version == ""

    def test_provenance_survives_save_load(self, tmp_path):
        """验证 provenance 字段在 save/load 后保持不变。"""
        rec = RunRecorder(
            "exp_001", run_id="roundtrip", base_dir=tmp_path,
            config_hash="hash_abc", code_version="commit_xyz",
        )
        rec.finish()
        loaded = RunManifest.load(rec.manifest_path)
        assert loaded.config_hash == "hash_abc"
        assert loaded.code_version == "commit_xyz"

    def test_set_params(self, tmp_path):
        rec = RunRecorder("exp_001", run_id="run_001", base_dir=tmp_path)
        rec.set_params({"window": 5, "variant": "short_term_reversal"})
        assert rec.manifest.params["window"] == 5
        assert rec.manifest.params["variant"] == "short_term_reversal"

    def test_set_params_merges(self, tmp_path):
        rec = RunRecorder("exp_001", run_id="run_001", base_dir=tmp_path)
        rec.set_params({"window": 5})
        rec.set_params({"variant": "str"})
        assert rec.manifest.params == {"window": 5, "variant": "str"}

    def test_record_metric(self, tmp_path):
        rec = RunRecorder("exp_001", run_id="run_001", base_dir=tmp_path)
        rec.record_metric("sharpe", 1.23)
        rec.record_metric("ic_mean", 0.05)
        assert rec.manifest.metrics["sharpe"] == 1.23
        assert rec.manifest.metrics["ic_mean"] == 0.05

    def test_record_metrics(self, tmp_path):
        rec = RunRecorder("exp_001", run_id="run_001", base_dir=tmp_path)
        rec.record_metrics({"sharpe": 1.0, "max_dd": -0.15})
        assert rec.manifest.metrics == {"sharpe": 1.0, "max_dd": -0.15}

    def test_record_artifact(self, tmp_path):
        rec = RunRecorder("exp_001", run_id="run_001", base_dir=tmp_path)
        rec.record_artifact("report", "backtests/analysis/report.csv")
        rec.record_artifact("chart", "backtests/analysis/chart.png")
        assert rec.manifest.artifacts["report"] == "backtests/analysis/report.csv"
        assert rec.manifest.artifacts["chart"] == "backtests/analysis/chart.png"

    def test_mark_stage(self, tmp_path):
        rec = RunRecorder("exp_001", run_id="run_001", base_dir=tmp_path)
        rec.mark_stage("walk_forward", "completed")
        rec.mark_stage("analysis", "running")
        assert rec.manifest.stage_status["walk_forward"] == "completed"
        assert rec.manifest.stage_status["analysis"] == "running"

    def test_set_status(self, tmp_path):
        rec = RunRecorder("exp_001", run_id="run_001", base_dir=tmp_path)
        rec.set_status("failed")
        assert rec.manifest.status == "failed"

    def test_finish_saves_manifest(self, tmp_path):
        rec = RunRecorder("exp_001", run_id="run_001", base_dir=tmp_path)
        rec.set_params({"window": 5})
        rec.record_metric("sharpe", 1.5)
        rec.finish()

        assert rec.manifest_path.exists()
        loaded = RunManifest.load(rec.manifest_path)
        assert loaded.status == "completed"
        assert loaded.finished_at != ""
        assert loaded.params["window"] == 5
        assert loaded.metrics["sharpe"] == 1.5

    def test_finish_with_failed_status(self, tmp_path):
        rec = RunRecorder("exp_001", run_id="run_001", base_dir=tmp_path)
        rec.finish(status="failed")
        loaded = RunManifest.load(rec.manifest_path)
        assert loaded.status == "failed"

    def test_save_without_finish(self, tmp_path):
        """save() 不改变 status。"""
        rec = RunRecorder("exp_001", run_id="run_001", base_dir=tmp_path)
        rec.record_metric("partial", 42)
        rec.save()

        loaded = RunManifest.load(rec.manifest_path)
        assert loaded.status == "running"
        assert loaded.metrics["partial"] == 42
        assert loaded.finished_at == ""

    def test_load_manifest(self, tmp_path):
        rec = RunRecorder("exp_001", run_id="run_001", base_dir=tmp_path)
        rec.set_params({"k": "v"})
        rec.finish()

        loaded = RunRecorder.load_manifest("exp_001", "run_001", base_dir=tmp_path)
        assert loaded.experiment_id == "exp_001"
        assert loaded.params["k"] == "v"

    def test_list_runs_empty(self, tmp_path):
        runs = RunRecorder.list_runs("exp_nonexistent", base_dir=tmp_path)
        assert runs == []

    def test_list_runs(self, tmp_path):
        for rid in ["run_c", "run_a", "run_b"]:
            rec = RunRecorder("exp_001", run_id=rid, base_dir=tmp_path)
            rec.finish()
        runs = RunRecorder.list_runs("exp_001", base_dir=tmp_path)
        assert runs == ["run_a", "run_b", "run_c"]  # sorted

    def test_list_runs_ignores_incomplete(self, tmp_path):
        """没有 run_manifest.json 的目录不应出现在列表中。"""
        rec = RunRecorder("exp_001", run_id="run_complete", base_dir=tmp_path)
        rec.finish()
        # 创建一个空目录
        (tmp_path / "exp_001" / "runs" / "run_incomplete").mkdir(parents=True)
        runs = RunRecorder.list_runs("exp_001", base_dir=tmp_path)
        assert runs == ["run_complete"]

    def test_full_workflow(self, tmp_path):
        """完整工作流：创建 -> 设置 -> 记录 -> 完成 -> 读取。"""
        rec = RunRecorder("exp_007", run_id="run_e2e", base_dir=tmp_path, stage="walk_forward")
        rec.set_params({
            "alpha_variant": "short_term_reversal",
            "reversal_window": 5,
            "start": "20200101",
        })
        rec.mark_stage("walk_forward", "completed")
        rec.mark_stage("analysis", "completed")
        rec.record_metric("sharpe", 0.85)
        rec.record_metric("ic_mean", 0.04)
        rec.record_metric("rank_ic_mean", 0.03)
        rec.record_artifact("wf_summary", "backtests/walk_forward_alpha_v7/summary.csv")
        rec.record_artifact("analysis_report", "backtests/analysis/report.csv")
        rec.finish()

        # 读回验证
        loaded = RunRecorder.load_manifest("exp_007", "run_e2e", base_dir=tmp_path)
        assert loaded.experiment_id == "exp_007"
        assert loaded.run_id == "run_e2e"
        assert loaded.stage == "walk_forward"
        assert loaded.status == "completed"
        assert loaded.params["alpha_variant"] == "short_term_reversal"
        assert loaded.params["reversal_window"] == 5
        assert loaded.metrics["sharpe"] == 0.85
        assert loaded.stage_status["walk_forward"] == "completed"
        assert loaded.stage_status["analysis"] == "completed"
        assert "wf_summary" in loaded.artifacts
        assert loaded.started_at <= loaded.finished_at


# ---------------------------------------------------------------------------
# 工具函数测试
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """工具函数测试。"""

    def test_generate_run_id_format(self):
        rid = _generate_run_id()
        assert rid.startswith("run_")
        assert len(rid) == 12  # "run_" + 8 hex

    def test_generate_run_id_unique(self):
        ids = {_generate_run_id() for _ in range(100)}
        assert len(ids) == 100  # 无重复

    def test_utc_now_iso_format(self):
        ts = _utc_now_iso()
        assert ts.endswith("Z")
        assert "T" in ts
        # 基本格式验证：YYYY-MM-DDTHH:MM:SSZ
        parts = ts.replace("Z", "").split("T")
        assert len(parts) == 2
        date_part, time_part = parts
        assert len(date_part) == 10  # YYYY-MM-DD
        assert len(time_part) == 8   # HH:MM:SS
