# -*- coding: utf-8 -*-
"""
tests/test_pipeline_recorder_integration.py

RunRecorder 与 run_research_pipeline.py 集成测试。
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.recorder import RunRecorder, RunManifest


# ---------------------------------------------------------------------------
# _try_collect_records 测试
# ---------------------------------------------------------------------------


class TestTryCollectRecords:
    """_try_collect_records 辅助函数测试。"""

    def _make_walk_forward_dir(self, base: Path) -> Path:
        """创建模拟的 walk-forward 输出目录。"""
        wf_dir = base / "walk_forward"
        wf_dir.mkdir(parents=True, exist_ok=True)
        for stock in ["000001", "000002"]:
            df = pd.DataFrame({
                "test_year": [2023, 2024],
                "excess_return": [0.05, 0.03],
                "sharpe": [1.2, 0.6],
                "max_drawdown": [-0.10, -0.12],
            })
            df.to_csv(wf_dir / f"wf_alpha_v7_stock_{stock}_summary.csv", index=False, encoding="utf-8-sig")
        return wf_dir

    def _make_robustness_dir(self, base: Path) -> Path:
        """创建模拟的 robustness 输出目录。"""
        robust_dir = base / "robustness"
        robust_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "overall_pass": True,
            "gates": {
                "sharpe": {"pass": True, "value": 0.85, "threshold": 0.5},
                "win_rate": {"pass": False, "value": 0.55, "threshold": 0.60},
            },
        }
        (robust_dir / "robustness_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return robust_dir

    def _make_diagnosis_dir(self, base: Path) -> Path:
        """创建模拟的 diagnosis 输出目录。"""
        diag_dir = base / "diagnosis"
        diag_dir.mkdir(parents=True, exist_ok=True)
        gap = pd.DataFrame({
            "test_year": [2023, 2024],
            "train_annual_return": [0.15, 0.12],
            "test_total_return": [0.05, -0.02],
            "gap": [0.10, 0.14],
        })
        gap.to_csv(diag_dir / "alpha_v7_diagnosis_train_test_gap.csv", index=False, encoding="utf-8-sig")
        return diag_dir

    def test_collect_walk_forward_record(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        self._make_walk_forward_dir(tmp_path)
        experiment = {
            "experiment_id": "exp_007",
            "outputs": {"walk_forward_dir": str(tmp_path / "walk_forward")},
        }
        records = _try_collect_records(experiment, ["walk_forward"], tmp_path)
        assert len(records) == 1
        assert records[0].record_type == "walk_forward"
        assert records[0].n_stocks == 2

    def test_collect_robustness_record(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        self._make_robustness_dir(tmp_path)
        experiment = {
            "experiment_id": "exp_007",
            "outputs": {"robustness_dir": str(tmp_path / "robustness")},
        }
        records = _try_collect_records(experiment, ["robustness"], tmp_path)
        assert len(records) == 1
        assert records[0].record_type == "robustness"
        assert records[0].overall_pass is True

    def test_collect_diagnosis_record(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        self._make_diagnosis_dir(tmp_path)
        experiment = {
            "experiment_id": "exp_007",
            "outputs": {"diagnosis_dir": str(tmp_path / "diagnosis")},
        }
        records = _try_collect_records(experiment, ["diagnosis"], tmp_path)
        assert len(records) == 1
        assert records[0].record_type == "diagnosis"

    def test_collect_multiple_stages(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        self._make_walk_forward_dir(tmp_path)
        self._make_robustness_dir(tmp_path)
        self._make_diagnosis_dir(tmp_path)
        experiment = {
            "experiment_id": "exp_007",
            "outputs": {
                "walk_forward_dir": str(tmp_path / "walk_forward"),
                "diagnosis_dir": str(tmp_path / "diagnosis"),
                "robustness_dir": str(tmp_path / "robustness"),
            },
        }
        records = _try_collect_records(experiment, ["walk_forward", "diagnosis", "robustness"], tmp_path)
        types = {r.record_type for r in records}
        assert types == {"walk_forward", "diagnosis", "robustness"}

    def test_collect_skips_missing_output_dir(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        experiment = {
            "experiment_id": "exp_007",
            "outputs": {"walk_forward_dir": str(tmp_path / "nonexistent")},
        }
        records = _try_collect_records(experiment, ["walk_forward"], tmp_path)
        assert records == []

    def test_collect_skips_unmapped_stage(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        experiment = {
            "experiment_id": "exp_007",
            "outputs": {},
        }
        # single_symbol_check 没有 Record Template 映射
        records = _try_collect_records(experiment, ["single_symbol_check"], tmp_path)
        assert records == []

    def test_collect_skips_empty_output(self, tmp_path):
        """空输出目录不应产生 Record。"""
        from scripts.run_research_pipeline import _try_collect_records

        wf_dir = tmp_path / "walk_forward"
        wf_dir.mkdir()
        # 空目录，无 summary 文件
        experiment = {
            "experiment_id": "exp_007",
            "outputs": {"walk_forward_dir": str(wf_dir)},
        }
        records = _try_collect_records(experiment, ["walk_forward"], tmp_path)
        assert records == []

    def test_collect_no_outputs_key(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        experiment = {"experiment_id": "exp_007"}
        records = _try_collect_records(experiment, ["walk_forward"], tmp_path)
        assert records == []

    def _make_signal_eval_dir(self, base: Path, variant: str = "short_term_reversal",
                              label_col: str = "ret_1d") -> Path:
        """创建模拟的 signal_evaluation 输出目录。"""
        se_dir = base / "signal_evaluation" / variant / label_col
        se_dir.mkdir(parents=True, exist_ok=True)
        # IC summary
        ic_df = pd.DataFrame([{
            "ic_mean": 0.035, "ic_std": 0.08, "rank_ic_mean": 0.03,
            "rank_ic_std": 0.07, "icir": 0.44, "ic_t_stat": 2.1,
            "ic_positive_rate": 0.62, "n_days": 200,
        }])
        ic_df.to_csv(se_dir / "signal_ic_summary.csv", index=False, encoding="utf-8-sig")
        # Quantile return
        qr_df = pd.DataFrame({"long_short": [0.001, 0.002, -0.001, 0.003, 0.001]})
        qr_df.to_csv(se_dir / "signal_quantile_return.csv", index=True, encoding="utf-8-sig")
        # Coverage
        cov_df = pd.DataFrame({"coverage": [0.95, 0.92, 0.98]})
        cov_df.to_csv(se_dir / "signal_coverage.csv", index=False, encoding="utf-8-sig")
        return base / "signal_evaluation"

    def test_collect_signal_evaluation_single_variant(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        self._make_signal_eval_dir(tmp_path, variant="short_term_reversal", label_col="ret_1d")
        experiment = {
            "experiment_id": "exp_007",
            "outputs": {"signal_evaluation_dir": str(tmp_path / "signal_evaluation")},
        }
        records = _try_collect_records(experiment, ["signal_evaluation"], tmp_path)
        assert len(records) == 1
        assert records[0].record_type == "signal_evaluation"
        assert records[0].variant == "short_term_reversal"
        assert records[0].label_col == "ret_1d"
        assert records[0].ic_mean == 0.035

    def test_collect_signal_evaluation_multiple_variants(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        for variant in ["short_term_reversal", "low_volatility"]:
            self._make_signal_eval_dir(tmp_path, variant=variant, label_col="ret_5d")
        experiment = {
            "experiment_id": "exp_007",
            "outputs": {"signal_evaluation_dir": str(tmp_path / "signal_evaluation")},
        }
        records = _try_collect_records(experiment, ["signal_evaluation"], tmp_path)
        assert len(records) == 2
        variants = {r.variant for r in records}
        assert variants == {"short_term_reversal", "low_volatility"}
        assert all(r.label_col == "ret_5d" for r in records)

    def test_collect_signal_evaluation_multiple_labels(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        for label in ["ret_1d", "ret_5d", "ret_20d"]:
            self._make_signal_eval_dir(tmp_path, variant="short_term_reversal", label_col=label)
        experiment = {
            "experiment_id": "exp_007",
            "outputs": {"signal_evaluation_dir": str(tmp_path / "signal_evaluation")},
        }
        records = _try_collect_records(experiment, ["signal_evaluation"], tmp_path)
        assert len(records) == 3
        labels = {r.label_col for r in records}
        assert labels == {"ret_1d", "ret_5d", "ret_20d"}

    def test_collect_signal_evaluation_skips_dir_without_ic_summary(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        # 创建有 IC summary 的目录
        self._make_signal_eval_dir(tmp_path, variant="short_term_reversal", label_col="ret_1d")
        # 创建没有 IC summary 的目录
        empty_dir = tmp_path / "signal_evaluation" / "low_volatility" / "ret_1d"
        empty_dir.mkdir(parents=True, exist_ok=True)

        experiment = {
            "experiment_id": "exp_007",
            "outputs": {"signal_evaluation_dir": str(tmp_path / "signal_evaluation")},
        }
        records = _try_collect_records(experiment, ["signal_evaluation"], tmp_path)
        assert len(records) == 1
        assert records[0].variant == "short_term_reversal"

    def test_collect_signal_evaluation_missing_dir(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        experiment = {
            "experiment_id": "exp_007",
            "outputs": {"signal_evaluation_dir": str(tmp_path / "nonexistent")},
        }
        records = _try_collect_records(experiment, ["signal_evaluation"], tmp_path)
        assert records == []

    def test_collect_signal_evaluation_no_outputs_key(self, tmp_path):
        from scripts.run_research_pipeline import _try_collect_records

        experiment = {"experiment_id": "exp_007"}
        records = _try_collect_records(experiment, ["signal_evaluation"], tmp_path)
        assert records == []


# ---------------------------------------------------------------------------
# RunRecorder 集成到 run_execute 的测试
# ---------------------------------------------------------------------------


def _mock_run_command_streaming_success(cmd_parts, cwd, stdout_path, stderr_path):
    """模拟命令执行成功。"""
    stdout_path.write_text("mock stdout\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return 0, "mock stdout", ""


def _mock_run_command_streaming_fail(cmd_parts, cwd, stdout_path, stderr_path):
    """模拟命令执行失败。"""
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("mock error\n", encoding="utf-8")
    return 1, "", "mock error"


class TestRunExecuteWithRecorder:
    """run_execute 中 RunRecorder 集成测试。"""

    @patch("scripts.run_research_pipeline.run_command_streaming", side_effect=_mock_run_command_streaming_success)
    def test_run_execute_creates_recorder(self, mock_cmd, tmp_path):
        """验证 --use-recorder 创建 RunRecorder 并保存 manifest。"""
        from scripts.run_research_pipeline import run_execute

        experiment = {
            "experiment_id": "exp_test",
            "strategy_name": "test_strategy",
            "strategy_version": "v1",
            "hypothesis": "test",
            "parameters": {},
            "cost_model": {},
            "commands": {
                "single_symbol_check": "python -c pass",
            },
            "outputs": {},
        }
        env_info = {
            "conda_env": "research-env",
            "conda_env_target": "research-env",
            "python_executable": sys.executable,
        }

        # run_execute 正常返回（不调用 sys.exit）当 success=True
        run_execute(
            experiment, ["single_symbol_check"], env_info, tmp_path,
            skip_existing=False, use_recorder=True,
        )

        # 验证 RunRecorder 目录被创建
        runs_dir = tmp_path / "experiments" / "ds_flow" / "exp_test" / "runs"
        assert runs_dir.exists()
        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 1
        manifest_path = run_dirs[0] / "run_manifest.json"
        assert manifest_path.exists()
        loaded = RunManifest.load(manifest_path)
        assert loaded.experiment_id == "exp_test"
        assert loaded.status == "completed"
        assert loaded.params["strategy_name"] == "test_strategy"

    @patch("scripts.run_research_pipeline.run_command_streaming", side_effect=_mock_run_command_streaming_success)
    def test_run_execute_recorder_marks_stages(self, mock_cmd, tmp_path):
        """验证 stage 被正确标记到 recorder。"""
        from scripts.run_research_pipeline import run_execute

        experiment = {
            "experiment_id": "exp_test",
            "strategy_name": "test",
            "strategy_version": "v1",
            "hypothesis": "test",
            "parameters": {},
            "cost_model": {},
            "commands": {
                "single_symbol_check": "python -c pass",
            },
            "outputs": {},
        }
        env_info = {
            "conda_env": "research-env",
            "conda_env_target": "research-env",
            "python_executable": sys.executable,
        }

        run_execute(
            experiment, ["single_symbol_check"], env_info, tmp_path,
            skip_existing=False, use_recorder=True,
        )

        # 验证 stage 被标记
        runs_dir = tmp_path / "experiments" / "ds_flow" / "exp_test" / "runs"
        run_dirs = list(runs_dir.iterdir())
        manifest_path = run_dirs[0] / "run_manifest.json"
        loaded = RunManifest.load(manifest_path)
        assert loaded.stage_status.get("single_symbol_check") == "completed"

    @patch("scripts.run_research_pipeline.run_command_streaming", side_effect=_mock_run_command_streaming_success)
    def test_run_execute_without_recorder(self, mock_cmd, tmp_path):
        """不传 use_recorder 时不应创建 recorder 目录。"""
        from scripts.run_research_pipeline import run_execute

        experiment = {
            "experiment_id": "exp_test",
            "strategy_name": "test",
            "strategy_version": "v1",
            "hypothesis": "test",
            "parameters": {},
            "cost_model": {},
            "commands": {
                "single_symbol_check": "python -c pass",
            },
            "outputs": {},
        }
        env_info = {
            "conda_env": "research-env",
            "conda_env_target": "research-env",
            "python_executable": sys.executable,
        }

        run_execute(
            experiment, ["single_symbol_check"], env_info, tmp_path,
            skip_existing=False, use_recorder=False,
        )

        # 验证没有 recorder 目录
        runs_dir = tmp_path / "experiments" / "ds_flow" / "exp_test" / "runs"
        assert not runs_dir.exists()

    @patch("scripts.run_research_pipeline.run_command_streaming", side_effect=_mock_run_command_streaming_success)
    def test_run_execute_recorder_with_record_templates(self, mock_cmd, tmp_path):
        """验证 Record Template 被自动加载到 recorder。"""
        from scripts.run_research_pipeline import run_execute

        # 创建模拟的 robustness 输出目录
        robust_dir = tmp_path / "robustness_output"
        robust_dir.mkdir()
        manifest = {
            "overall_pass": True,
            "gates": {"sharpe": {"pass": True, "value": 0.85, "threshold": 0.5}},
        }
        (robust_dir / "robustness_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        experiment = {
            "experiment_id": "exp_test",
            "strategy_name": "test",
            "strategy_version": "v1",
            "hypothesis": "test",
            "parameters": {},
            "cost_model": {},
            "commands": {},  # 无命令
            "outputs": {
                "robustness_dir": str(robust_dir),
            },
        }
        env_info = {
            "conda_env": "research-env",
            "conda_env_target": "research-env",
            "python_executable": sys.executable,
        }

        run_execute(
            experiment, ["robustness"], env_info, tmp_path,
            skip_existing=False, use_recorder=True,
        )

        # 验证 Record Template 被保存
        runs_dir = tmp_path / "experiments" / "ds_flow" / "exp_test" / "runs"
        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 1
        robustness_record_path = run_dirs[0] / "robustness_record.json"
        assert robustness_record_path.exists()
        with open(robustness_record_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["record_type"] == "robustness"
        assert data["overall_pass"] is True

        # 验证 recorder manifest 中有 robustness 指标
        manifest_path = run_dirs[0] / "run_manifest.json"
        loaded = RunManifest.load(manifest_path)
        assert loaded.metrics.get("robust_overall_pass") is True

    @patch("scripts.run_research_pipeline.run_command_streaming", side_effect=_mock_run_command_streaming_fail)
    def test_run_execute_recorder_failed_stage(self, mock_cmd, tmp_path):
        """验证失败 stage 被标记为 failed。"""
        from scripts.run_research_pipeline import run_execute

        experiment = {
            "experiment_id": "exp_test",
            "strategy_name": "test",
            "strategy_version": "v1",
            "hypothesis": "test",
            "parameters": {},
            "cost_model": {},
            "commands": {
                "single_symbol_check": "python -c fail",
            },
            "outputs": {},
        }
        env_info = {
            "conda_env": "research-env",
            "conda_env_target": "research-env",
            "python_executable": sys.executable,
        }

        with pytest.raises(SystemExit) as exc_info:
            run_execute(
                experiment, ["single_symbol_check"], env_info, tmp_path,
                skip_existing=False, use_recorder=True,
            )
        assert exc_info.value.code == 1

        # 验证 recorder 记录了失败
        runs_dir = tmp_path / "experiments" / "ds_flow" / "exp_test" / "runs"
        run_dirs = list(runs_dir.iterdir())
        manifest_path = run_dirs[0] / "run_manifest.json"
        loaded = RunManifest.load(manifest_path)
        assert loaded.status == "failed"
        assert loaded.stage_status.get("single_symbol_check") == "failed"
        assert loaded.metrics.get("pipeline_success") is False

    @patch("scripts.run_research_pipeline.run_command_streaming", side_effect=_mock_run_command_streaming_success)
    def test_run_execute_recorder_multiple_stages(self, mock_cmd, tmp_path):
        """验证多 stage 全部被标记。"""
        from scripts.run_research_pipeline import run_execute

        experiment = {
            "experiment_id": "exp_test",
            "strategy_name": "test",
            "strategy_version": "v1",
            "hypothesis": "test",
            "parameters": {},
            "cost_model": {},
            "commands": {
                "single_symbol_check": "python -c pass",
                "batch_backtest": "python -c pass",
            },
            "outputs": {},
        }
        env_info = {
            "conda_env": "research-env",
            "conda_env_target": "research-env",
            "python_executable": sys.executable,
        }

        run_execute(
            experiment, ["single_symbol_check", "batch_backtest"], env_info, tmp_path,
            skip_existing=False, use_recorder=True,
        )

        runs_dir = tmp_path / "experiments" / "ds_flow" / "exp_test" / "runs"
        run_dirs = list(runs_dir.iterdir())
        manifest_path = run_dirs[0] / "run_manifest.json"
        loaded = RunManifest.load(manifest_path)
        assert loaded.stage_status.get("single_symbol_check") == "completed"
        assert loaded.stage_status.get("batch_backtest") == "completed"

    @patch("scripts.run_research_pipeline.run_command_streaming", side_effect=_mock_run_command_streaming_success)
    def test_run_execute_recorder_saves_params(self, mock_cmd, tmp_path):
        """验证实验参数被保存到 recorder。"""
        from scripts.run_research_pipeline import run_execute

        experiment = {
            "experiment_id": "exp_test",
            "strategy_name": "alpha_v7",
            "strategy_version": "v7",
            "hypothesis": "short-term reversal",
            "parameters": {"reversal_window": 5},
            "cost_model": {"commission": 0.0003},
            "commands": {
                "single_symbol_check": "python -c pass",
            },
            "outputs": {},
        }
        env_info = {
            "conda_env": "research-env",
            "conda_env_target": "research-env",
            "python_executable": sys.executable,
        }

        run_execute(
            experiment, ["single_symbol_check"], env_info, tmp_path,
            skip_existing=False, use_recorder=True,
        )

        runs_dir = tmp_path / "experiments" / "ds_flow" / "exp_test" / "runs"
        run_dirs = list(runs_dir.iterdir())
        manifest_path = run_dirs[0] / "run_manifest.json"
        loaded = RunManifest.load(manifest_path)
        assert loaded.params["strategy_name"] == "alpha_v7"
        assert loaded.params["hypothesis"] == "short-term reversal"
        assert loaded.params["parameters"] == {"reversal_window": 5}
        assert loaded.params["cost_model"] == {"commission": 0.0003}

    @patch("scripts.run_research_pipeline.run_command_streaming", side_effect=_mock_run_command_streaming_success)
    @patch("scripts.run_research_pipeline._get_git_provenance", return_value=("abc123deadbeef", False))
    def test_run_execute_recorder_auto_populates_provenance(self, mock_git, mock_cmd, tmp_path):
        """验证 config_hash 和 code_version 被自动填充到 recorder manifest。"""
        from scripts.run_research_pipeline import run_execute

        experiment = {
            "experiment_id": "exp_test",
            "strategy_name": "alpha_v7",
            "strategy_version": "v7",
            "hypothesis": "test",
            "parameters": {},
            "cost_model": {},
            "commands": {
                "single_symbol_check": "python -c pass",
            },
            "outputs": {},
        }
        env_info = {
            "conda_env": "research-env",
            "conda_env_target": "research-env",
            "python_executable": sys.executable,
        }

        run_execute(
            experiment, ["single_symbol_check"], env_info, tmp_path,
            skip_existing=False, use_recorder=True,
        )

        runs_dir = tmp_path / "experiments" / "ds_flow" / "exp_test" / "runs"
        run_dirs = list(runs_dir.iterdir())
        manifest_path = run_dirs[0] / "run_manifest.json"
        loaded = RunManifest.load(manifest_path)

        # config_hash 应该是 experiment dict 的 SHA-256（非空）
        assert loaded.config_hash != ""
        assert len(loaded.config_hash) == 64  # SHA-256 hex

        # code_version 应该是 git commit hash
        assert loaded.code_version == "abc123deadbeef"

        # git_dirty 应该在 params 中
        assert loaded.params["git_dirty"] is False


class TestCLIUseRecorder:
    """--use-recorder CLI 参数测试。"""

    def test_parse_args_use_recorder(self):
        from scripts.run_research_pipeline import parse_args

        with patch("sys.argv", ["run_research_pipeline.py", "--experiment-id", "exp_001", "--use-recorder"]):
            args = parse_args()
        assert args.use_recorder is True

    def test_parse_args_no_use_recorder(self):
        from scripts.run_research_pipeline import parse_args

        with patch("sys.argv", ["run_research_pipeline.py", "--experiment-id", "exp_001"]):
            args = parse_args()
        assert args.use_recorder is False
