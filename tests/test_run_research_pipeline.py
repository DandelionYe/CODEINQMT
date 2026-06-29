"""Unit tests for scripts/run_research_pipeline.py"""
import json
import pytest
from pathlib import Path

from scripts.run_research_pipeline import (
    STAGE_ORDER,
    parse_stages_arg,
    validate_schema,
    validate_stage_commands,
    find_experiment,
    resolve_commands,
    should_skip_stage,
    load_config,
)


# --- resolve_commands (trivial pure function) ---

class TestResolveCommands:
    def test_string_input(self):
        assert resolve_commands("echo hello") == ["echo hello"]

    def test_list_input(self):
        assert resolve_commands(["cmd1", "cmd2"]) == ["cmd1", "cmd2"]

    def test_list_with_non_strings(self):
        assert resolve_commands([123, True]) == ["123", "True"]

    def test_none_input(self):
        assert resolve_commands(None) == []

    def test_empty_string(self):
        assert resolve_commands("") == []


# --- parse_stages_arg ---

class TestParseStagesArg:
    def test_none_returns_full_order(self):
        assert parse_stages_arg(None) == list(STAGE_ORDER)

    @pytest.mark.parametrize("stage", ["analysis", "single_symbol_check", "portfolio_backtest"])
    def test_single_stage(self, stage):
        result = parse_stages_arg(stage)
        assert result == [stage]

    def test_multiple_stages_reordered(self):
        result = parse_stages_arg("diagnosis,analysis,batch_backtest")
        assert result == ["batch_backtest", "analysis", "diagnosis"]

    def test_reverse_order(self):
        """Reversed STAGE_ORDER should be reordered to canonical order."""
        result = parse_stages_arg(",".join(reversed(STAGE_ORDER)))
        assert result == list(STAGE_ORDER)

    def test_all_stages_at_once(self):
        result = parse_stages_arg(",".join(STAGE_ORDER))
        assert result == list(STAGE_ORDER)

    def test_duplicates_deduped(self):
        result = parse_stages_arg("analysis,analysis,diagnosis")
        assert result == ["analysis", "diagnosis"]
        assert len(result) == 2

    def test_unknown_stage_exits(self):
        with pytest.raises(SystemExit):
            parse_stages_arg("analysis,fake_stage")

    def test_multiple_unknown_stages_exits(self):
        with pytest.raises(SystemExit):
            parse_stages_arg("analysis,fake1,fake2")

    def test_mixed_valid_and_invalid_exits(self):
        with pytest.raises(SystemExit):
            parse_stages_arg("analysis,invalid_stage,diagnosis")

    def test_empty_string_returns_full_order(self):
        result = parse_stages_arg("")
        assert result == list(STAGE_ORDER)

    def test_whitespace_handling(self):
        result = parse_stages_arg("  analysis , diagnosis  ")
        assert result == ["analysis", "diagnosis"]

    @pytest.mark.parametrize("input_str,expected", [
        ("analysis,", ["analysis"]),
        (",analysis", ["analysis"]),
        ("analysis,,diagnosis", ["analysis", "diagnosis"]),
    ])
    def test_empty_elements_filtered(self, input_str, expected):
        result = parse_stages_arg(input_str)
        assert result == expected


# --- find_experiment ---

class TestFindExperiment:
    def test_found(self, sample_config):
        exp = find_experiment(sample_config, "exp_test_001")
        assert exp["experiment_id"] == "exp_test_001"

    def test_not_found_exits(self, sample_config):
        with pytest.raises(SystemExit):
            find_experiment(sample_config, "nonexistent_id")

    def test_empty_experiments_list(self):
        config = {"experiments": []}
        with pytest.raises(SystemExit):
            find_experiment(config, "any_id")


# --- validate_stage_commands ---

class TestValidateStageCommands:
    def test_all_commands_present(self, sample_experiment):
        stages = ["analysis", "diagnosis"]
        missing = validate_stage_commands(stages, sample_experiment, explicit_stages=True)
        assert missing == []

    def test_missing_command_explicit_exits(self, sample_experiment):
        sample_experiment["commands"]["robustness"] = None
        stages = ["robustness"]
        with pytest.raises(SystemExit):
            validate_stage_commands(stages, sample_experiment, explicit_stages=True)

    def test_missing_command_implicit_warns(self, sample_experiment, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            stages = ["analysis", "robustness"]
            missing = validate_stage_commands(stages, sample_experiment, explicit_stages=False)
        assert "robustness" in missing
        assert "没有定义 command" in caplog.text

    def test_empty_commands_dict(self):
        exp = {"commands": {}, "experiment_id": "test"}
        missing = validate_stage_commands(["analysis"], exp, explicit_stages=False)
        assert missing == ["analysis"]


# --- validate_schema ---

class TestValidateSchema:
    def test_valid_config_passes(self, sample_config, schema_path):
        errors = validate_schema(sample_config, schema_path)
        assert errors == []

    def test_missing_schema_file(self, sample_config):
        errors = validate_schema(sample_config, "/nonexistent/schema.json")
        assert len(errors) == 1
        assert "Schema 文件不存在" in errors[0]

    def test_invalid_config_fails(self, schema_path):
        bad_config = {"schema_version": "1.0", "experiments": "not_a_list"}
        errors = validate_schema(bad_config, schema_path)
        assert len(errors) > 0

    def test_degraded_mode_without_jsonschema(self, sample_config, monkeypatch):
        """Simulate jsonschema not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "jsonschema":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        errors = validate_schema(sample_config, "any_path", allow_degraded=False)
        assert any("jsonschema" in e for e in errors)

    def test_degraded_mode_valid_config(self, sample_config, monkeypatch):
        """Degraded mode with valid config passes."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "jsonschema":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        errors = validate_schema(sample_config, "any_path", allow_degraded=True)
        assert errors == []

    def test_degraded_mode_missing_experiment_id(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "jsonschema":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        bad_config = {
            "schema_version": "1.0",
            "experiments": [{"commands": {}}],
        }
        errors = validate_schema(bad_config, "any_path", allow_degraded=True)
        assert any("experiment_id" in e for e in errors)


# --- should_skip_stage ---

class TestShouldSkipStage:
    def test_no_output_mapping(self):
        assert should_skip_stage("unknown_stage", {}, skip_existing=True) is False

    def test_no_output_dir_in_experiment(self, sample_experiment):
        assert should_skip_stage("robustness", sample_experiment, skip_existing=True) is False

    def test_output_dir_empty(self, sample_experiment, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        sample_experiment["outputs"]["analysis_dir"] = str(empty_dir)
        result = should_skip_stage("analysis", sample_experiment, skip_existing=True)
        assert result is False

    def test_output_dir_nonempty_skip(self, sample_experiment, tmp_path):
        nonempty = tmp_path / "nonempty"
        nonempty.mkdir()
        (nonempty / "file.txt").write_text("data")
        sample_experiment["outputs"]["analysis_dir"] = str(nonempty)
        assert should_skip_stage("analysis", sample_experiment, skip_existing=True) is True

    def test_output_dir_nonempty_no_skip(self, sample_experiment, tmp_path):
        nonempty = tmp_path / "nonempty2"
        nonempty.mkdir()
        (nonempty / "file.txt").write_text("data")
        sample_experiment["outputs"]["analysis_dir"] = str(nonempty)
        assert should_skip_stage("analysis", sample_experiment, skip_existing=False) is False


# --- F1-2: Schema validation regression tests ---

class TestSchemaRegression:
    """Verify the real config passes schema validation (regression safety net)."""

    def test_real_config_passes_schema(self, schema_path):
        """The actual research_experiments.json must pass schema validation."""
        config = load_config(str(Path(__file__).resolve().parent.parent / "configs" / "research_experiments.json"))
        errors = validate_schema(config, schema_path)
        assert errors == [], f"Schema validation failed: {errors}"

    def test_real_config_has_all_experiments(self):
        """Verify the expected experiment IDs exist."""
        config = load_config(str(Path(__file__).resolve().parent.parent / "configs" / "research_experiments.json"))
        ids = [e["experiment_id"] for e in config["experiments"]]
        assert "exp_002_ma_cross_with_market_filter" in ids
        assert "exp_003_ma_v3_momentum_trend_filter" in ids
        assert "exp_004_next_alpha_research" in ids
        assert "exp_007_alpha_v7_expression_layer" in ids

    def test_exp_007_alpha_v7_entry_structure(self):
        """Verify exp_007 Alpha v7 entry has correct structure."""
        config = load_config(str(Path(__file__).resolve().parent.parent / "configs" / "research_experiments.json"))
        exp_007 = [e for e in config["experiments"] if e["experiment_id"] == "exp_007_alpha_v7_expression_layer"]
        assert len(exp_007) == 1
        exp = exp_007[0]
        assert exp["strategy_version"] == "v7"
        assert exp["strategy_family"] == "alpha_research"
        assert "single_symbol_check" in exp["commands"]
        assert "alpha_v7_research_strategy_csv.py" in exp["commands"]["single_symbol_check"]
        assert exp["outputs"]["single_symbol_dir"] == "backtests/alpha_v7_research_strategy_csv"
        assert len(exp["parameters"]["alpha_variant_list"]) == 4

    def test_exp_007_analysis_diagnosis_use_v7_scripts(self):
        """Verify exp_007 analysis and diagnosis commands point to v7 scripts, not v6."""
        config = load_config(str(Path(__file__).resolve().parent.parent / "configs" / "research_experiments.json"))
        exp_007 = [e for e in config["experiments"] if e["experiment_id"] == "exp_007_alpha_v7_expression_layer"][0]
        assert "analyze_alpha_v7" in exp_007["commands"]["analysis"]
        assert "diagnose_alpha_v7" in exp_007["commands"]["diagnosis"]
        # Must NOT reference v6 scripts
        assert "alpha_v6" not in exp_007["commands"]["analysis"]
        assert "alpha_v6" not in exp_007["commands"]["diagnosis"]

    def test_exp_007_has_signal_evaluation(self):
        """Verify exp_007 has signal_evaluation command and output directory."""
        config = load_config(str(Path(__file__).resolve().parent.parent / "configs" / "research_experiments.json"))
        exp_007 = [e for e in config["experiments"] if e["experiment_id"] == "exp_007_alpha_v7_expression_layer"][0]
        assert "signal_evaluation" in exp_007["commands"]
        assert "evaluate_alpha_signals" in exp_007["commands"]["signal_evaluation"]
        assert "signal_evaluation_dir" in exp_007["outputs"]
        assert "signal_evaluation" in exp_007["outputs"]["signal_evaluation_dir"]

    def test_exp_007_has_portfolio_backtest(self):
        """Verify exp_007 has portfolio_backtest command and output directory."""
        config = load_config(str(Path(__file__).resolve().parent.parent / "configs" / "research_experiments.json"))
        exp_007 = [e for e in config["experiments"] if e["experiment_id"] == "exp_007_alpha_v7_expression_layer"][0]
        assert "portfolio_backtest" in exp_007["commands"]
        assert "portfolio_backtest_csv.py" in exp_007["commands"]["portfolio_backtest"]
        assert "portfolio_backtest_dir" in exp_007["outputs"]
        assert exp_007["outputs"]["portfolio_backtest_dir"] is not None

    def test_missing_required_experiment_id_fails(self, sample_config, schema_path):
        """Removing a required experiment field must fail schema validation."""
        del sample_config["experiments"][0]["experiment_id"]
        errors = validate_schema(sample_config, schema_path)
        assert len(errors) > 0
        assert "experiment_id" in errors[0]

    def test_missing_required_command_fails(self, sample_config, schema_path):
        """Removing a required command field must fail schema validation."""
        del sample_config["experiments"][0]["commands"]["diagnosis"]
        errors = validate_schema(sample_config, schema_path)
        assert len(errors) > 0
        assert "diagnosis" in errors[0]

    def test_missing_required_output_fails(self, sample_config, schema_path):
        """Removing a required output field must fail schema validation."""
        del sample_config["experiments"][0]["outputs"]["diagnosis_dir"]
        errors = validate_schema(sample_config, schema_path)
        assert len(errors) > 0
        assert "diagnosis_dir" in errors[0]


# --- load_config ---

class TestLoadConfig:
    def test_valid_file(self, tmp_path):
        cfg = {"schema_version": "1.0", "experiments": []}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        result = load_config(str(p))
        assert result == cfg

    def test_missing_file_exits(self):
        with pytest.raises(SystemExit):
            load_config("/nonexistent/path/config.json")

    def test_invalid_json_exits(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(SystemExit):
            load_config(str(p))


# --- task fallback in validate_stage_commands ---

class TestTaskFallback:
    """验证 validate_stage_commands 在 commands 缺失时尝试 task 生成。"""

    def test_task_generates_signal_evaluation(self):
        """当 commands 缺 signal_evaluation 但 task 有 evaluation 时，不报缺失。"""
        exp = {
            "experiment_id": "exp_task_test",
            "commands": {"analysis": "echo ok"},
            "task": {
                "signal": {"variants": ["v1", "v2"]},
                "evaluation": {"signal_ic": True},
                "data": {"start": "20200101"},
            },
        }
        missing = validate_stage_commands(["signal_evaluation"], exp, explicit_stages=True)
        assert missing == []

    def test_task_generates_walk_forward(self):
        """当 commands 缺 walk_forward 但 task 有 walk_forward 时，不报缺失。"""
        exp = {
            "experiment_id": "exp_task_alpha_v7_test",
            "commands": {},
            "task": {
                "signal": {"variants": ["v1"]},
                "walk_forward": {"first_test_year": 2021, "last_test_year": 2025, "portfolio_size": 20},
                "data": {"universe": "ALL"},
            },
        }
        missing = validate_stage_commands(["walk_forward"], exp, explicit_stages=True)
        assert missing == []

    def test_task_does_not_help_unsupported_stage(self):
        """task 不支持的 stage 仍报缺失。"""
        exp = {
            "experiment_id": "exp_task_test",
            "commands": {},
            "task": {
                "signal": {"variants": ["v1"]},
                "evaluation": {"signal_ic": True},
            },
        }
        missing = validate_stage_commands(["batch_backtest"], exp, explicit_stages=False)
        assert "batch_backtest" in missing

    def test_commands_takes_priority_over_task(self):
        """当 commands 已有 stage 时，不使用 task 生成。"""
        exp = {
            "experiment_id": "exp_task_test",
            "commands": {"signal_evaluation": "echo my_command"},
            "task": {
                "signal": {"variants": ["v1"]},
                "evaluation": {"signal_ic": True},
            },
        }
        missing = validate_stage_commands(["signal_evaluation"], exp, explicit_stages=True)
        assert missing == []

    def test_exp007_signal_evaluation_not_missing(self):
        """exp_007 的 signal_evaluation 应通过 task 生成命令。"""
        config = load_config(str(Path(__file__).resolve().parent.parent / "configs" / "research_experiments.json"))
        exp = next(e for e in config["experiments"] if e["experiment_id"] == "exp_007_alpha_v7_expression_layer")
        # signal_evaluation 在 commands 中已有，但验证 task 也能生成
        from scripts.common.config_loader import generate_task_command
        cmd = generate_task_command(exp, "signal_evaluation")
        assert cmd is not None
        assert "evaluate_alpha_signals.py" in cmd
