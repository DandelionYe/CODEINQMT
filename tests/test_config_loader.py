# -*- coding: utf-8 -*-
"""
test_config_loader.py

验证配置继承与结构化 Task 支持。

核心验证：
1. load_base_config 正确加载基线配置
2. load_base_config 支持 _inherits 递归合并
3. load_base_config 检测循环继承
4. resolve_experiment_config 合并 base_config 与实验字段
5. 嵌套字段（parameters/cost_model/universe/date_range）深度合并
6. task 配置辅助函数
7. 基线配置链验证
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.config_loader import (  # noqa: E402
    load_base_config,
    resolve_experiment_config,
    get_task_config,
    task_has_stage,
    merge_task_defaults,
    generate_task_command,
    validate_base_config_chain,
    list_base_configs,
    clear_base_config_cache,
    BASE_CONFIG_DIR,
    _BASE_CONFIG_CACHE,
    _deep_merge,
    _extract_alpha_version,
)


# ---------------------------------------------------------------------------
# 测试：load_base_config
# ---------------------------------------------------------------------------

class TestLoadBaseConfig:
    """基线配置加载。"""

    def test_load_cost_model(self):
        """加载 cost_model_a_share_default。"""
        config = load_base_config("cost_model_a_share_default")
        assert config["cost_model"]["cash"] == 1000000
        assert config["cost_model"]["commission"] == 0.0001
        assert config["cost_model"]["sell_tax"] == 0.0005
        assert config["cost_model"]["slippage"] == 0.0
        # 元数据字段应被移除
        assert "_description" not in config
        assert "_updated_at" not in config

    def test_load_walk_forward(self):
        """加载 walk_forward_default。"""
        config = load_base_config("walk_forward_default")
        assert config["date_range"]["first_test_year"] == 2021
        assert config["date_range"]["last_test_year"] == 2025
        assert config["parameters"]["portfolio_size"] == 20
        assert config["parameters"]["min_train_rows"] == 1000
        assert config["parameters"]["train_excess_mode"] == "stock_only"

    def test_load_alpha_research_base(self):
        """加载 alpha_research_base（含 _inherits）。"""
        config = load_base_config("alpha_research_base")
        # 应包含继承的 cost_model 字段
        assert config["cost_model"]["cash"] == 1000000
        assert config["cost_model"]["commission"] == 0.0001
        # 应包含继承的 walk_forward 字段
        assert config["date_range"]["first_test_year"] == 2021
        assert config["parameters"]["portfolio_size"] == 20
        # 应包含自身的 universe 和 date_range
        assert config["universe"]["market"] == "ALL"
        assert config["date_range"]["train_start"] == "20150101"
        assert config["parameters"]["benchmark_list"] == ["000300.SH", "000905.SH", "000852.SH"]
        # _inherits 应被移除
        assert "_inherits" not in config

    def test_nonexistent_config_raises(self):
        """不存在的基线配置应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_base_config("nonexistent_config_xyz")


# ---------------------------------------------------------------------------
# 测试：循环继承检测
# ---------------------------------------------------------------------------

class TestCircularInheritance:
    """循环继承检测。"""

    def test_circular_inheritance_detected(self, tmp_path):
        """检测 A -> B -> A 循环。"""
        # 创建临时循环继承文件
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        (base_dir / "a.json").write_text(
            json.dumps({"_inherits": ["b"], "value_a": 1}),
            encoding="utf-8",
        )
        (base_dir / "b.json").write_text(
            json.dumps({"_inherits": ["a"], "value_b": 2}),
            encoding="utf-8",
        )

        import scripts.common.config_loader as cl
        old_dir = cl.BASE_CONFIG_DIR
        cl.BASE_CONFIG_DIR = base_dir
        try:
            with pytest.raises(ValueError, match="循环继承"):
                load_base_config("a")
        finally:
            cl.BASE_CONFIG_DIR = old_dir


# ---------------------------------------------------------------------------
# 测试：resolve_experiment_config
# ---------------------------------------------------------------------------

class TestResolveExperimentConfig:
    """实验配置解析与合并。"""

    def test_no_base_config(self):
        """没有 base_config 时返回原始实验。"""
        experiment = {"experiment_id": "test_001", "parameters": {"x": 1}}
        result = resolve_experiment_config(experiment)
        assert result["experiment_id"] == "test_001"
        assert result["parameters"]["x"] == 1

    def test_with_base_config(self):
        """有 base_config 时合并基线字段。"""
        experiment = {
            "experiment_id": "test_002",
            "base_config": "cost_model_a_share_default",
            "cost_model": {"commission": 0.0003},  # 覆盖基线
        }
        result = resolve_experiment_config(experiment)
        # 深度合并：实验字段覆盖基线
        assert result["cost_model"]["commission"] == 0.0003
        # 基线字段保留
        assert result["cost_model"]["sell_tax"] == 0.0005
        assert result["cost_model"]["cash"] == 1000000
        # base_config 引用被移除
        assert "base_config" not in result

    def test_nested_merge_parameters(self):
        """parameters 字段深度合并。"""
        experiment = {
            "experiment_id": "test_003",
            "base_config": "walk_forward_default",
            "parameters": {
                "portfolio_size": 30,  # 覆盖基线
                "custom_param": "value",  # 新增
            },
        }
        result = resolve_experiment_config(experiment)
        # date_range 从基线保留
        assert result["date_range"]["first_test_year"] == 2021
        # parameters 深度合并
        assert result["parameters"]["portfolio_size"] == 30  # 实验覆盖
        assert result["parameters"]["custom_param"] == "value"  # 实验新增
        assert result["parameters"]["min_train_rows"] == 1000  # 基线保留

    def test_nested_merge_cost_model(self):
        """cost_model 字段深度合并。"""
        experiment = {
            "experiment_id": "test_004",
            "base_config": "cost_model_a_share_default",
            "cost_model": {
                "slippage": 0.001,  # 覆盖基线
            },
        }
        result = resolve_experiment_config(experiment)
        assert result["cost_model"]["slippage"] == 0.001  # 实验覆盖
        assert result["cost_model"]["commission"] == 0.0001  # 基线保留
        assert result["cost_model"]["cash"] == 1000000  # 基线保留

    def test_nested_merge_universe(self):
        """universe 字段深度合并。"""
        experiment = {
            "experiment_id": "test_005",
            "base_config": "alpha_research_base",
            "universe": {
                "market": "SH",  # 覆盖基线
            },
        }
        result = resolve_experiment_config(experiment)
        assert result["universe"]["market"] == "SH"  # 实验覆盖
        assert result["universe"]["security_type"] == "stock"  # 基线保留

    def test_nested_merge_date_range(self):
        """date_range 字段深度合并。"""
        experiment = {
            "experiment_id": "test_006",
            "base_config": "alpha_research_base",
            "date_range": {
                "train_start": "20100101",  # 覆盖基线
            },
        }
        result = resolve_experiment_config(experiment)
        assert result["date_range"]["train_start"] == "20100101"  # 实验覆盖
        assert result["date_range"]["first_test_year"] == 2021  # 继承自 walk_forward_default
        assert result["date_range"]["incomplete_year"] == 2026  # 继承自 alpha_research_base


# ---------------------------------------------------------------------------
# 测试：task 配置辅助函数
# ---------------------------------------------------------------------------

class TestTaskConfig:
    """task 配置辅助函数。"""

    def test_get_task_config_present(self):
        """有 task 字段时返回 task dict。"""
        experiment = {
            "experiment_id": "test",
            "task": {"signal": {"type": "rule"}},
        }
        task = get_task_config(experiment)
        assert task is not None
        assert task["signal"]["type"] == "rule"

    def test_get_task_config_absent(self):
        """没有 task 字段时返回 None。"""
        experiment = {"experiment_id": "test"}
        task = get_task_config(experiment)
        assert task is None

    def test_task_has_stage_true(self):
        """task 中有对应 stage 时返回 True。"""
        experiment = {
            "task": {
                "evaluation": {"signal_ic": True},
                "walk_forward": {"first_test_year": 2021},
                "signal": {"variants": ["v1"]},
            }
        }
        assert task_has_stage(experiment, "signal_evaluation") is True
        assert task_has_stage(experiment, "walk_forward") is True
        assert task_has_stage(experiment, "portfolio_backtest") is True

    def test_task_has_stage_false(self):
        """task 中没有对应 stage 时返回 False。"""
        experiment = {
            "task": {"evaluation": {"signal_ic": True}},
        }
        assert task_has_stage(experiment, "walk_forward") is False

    def test_task_has_stage_no_task(self):
        """没有 task 时返回 False。"""
        experiment = {"experiment_id": "test"}
        assert task_has_stage(experiment, "signal_evaluation") is False

    def test_merge_task_defaults(self):
        """合并 task 与基线默认值。"""
        task = {
            "signal": {"type": "rule", "variants": ["v1"]},
            "walk_forward": {"portfolio_size": 30},
        }
        base_task = {
            "signal": {"type": "ml"},
            "walk_forward": {"first_test_year": 2021, "portfolio_size": 20},
            "evaluation": {"signal_ic": True},
        }
        merged = merge_task_defaults(task, base_task)
        # task 覆盖基线
        assert merged["signal"]["type"] == "rule"
        assert merged["signal"]["variants"] == ["v1"]
        assert merged["walk_forward"]["portfolio_size"] == 30
        # 基线保留
        assert merged["walk_forward"]["first_test_year"] == 2021
        assert merged["evaluation"]["signal_ic"] is True


# ---------------------------------------------------------------------------
# 测试：基线配置链验证
# ---------------------------------------------------------------------------

class TestValidateBaseConfigChain:
    """基线配置链验证。"""

    def test_current_configs_valid(self):
        """当前所有基线配置应合法。"""
        errors = validate_base_config_chain()
        assert errors == [], f"基线配置验证失败: {errors}"

    def test_list_base_configs(self):
        """列出所有基线配置。"""
        configs = list_base_configs()
        assert "cost_model_a_share_default" in configs
        assert "walk_forward_default" in configs
        assert "alpha_research_base" in configs


# ---------------------------------------------------------------------------
# 测试：exp_007 集成验证
# ---------------------------------------------------------------------------

class TestExp007Integration:
    """验证 exp_007 的 base_config 和 task 配置。"""

    def test_exp007_has_base_config(self):
        """exp_007 应有 base_config 字段。"""
        config_path = PROJECT_ROOT / "configs" / "research_experiments.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        exp007 = next(e for e in data["experiments"]
                      if e["experiment_id"] == "exp_007_alpha_v7_expression_layer")
        assert "base_config" in exp007
        assert exp007["base_config"] == "alpha_research_base"

    def test_exp007_has_task(self):
        """exp_007 应有 task 字段。"""
        config_path = PROJECT_ROOT / "configs" / "research_experiments.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        exp007 = next(e for e in data["experiments"]
                      if e["experiment_id"] == "exp_007_alpha_v7_expression_layer")
        assert "task" in exp007
        task = exp007["task"]
        assert task["signal"]["type"] == "rule"
        assert len(task["signal"]["variants"]) == 4
        assert task["evaluation"]["signal_ic"] is True
        assert "WalkForwardRecord" in task["records"]

    def test_exp007_resolve_config(self):
        """exp_007 的 base_config 应正确合并。"""
        config_path = PROJECT_ROOT / "configs" / "research_experiments.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        exp007 = next(e for e in data["experiments"]
                      if e["experiment_id"] == "exp_007_alpha_v7_expression_layer")
        resolved = resolve_experiment_config(exp007)

        # 继承的 cost_model
        assert resolved["cost_model"]["cash"] == 1000000
        assert resolved["cost_model"]["commission"] == 0.0001
        # 继承的 universe
        assert resolved["universe"]["market"] == "ALL"
        # 继承的 date_range
        assert resolved["date_range"]["train_start"] == "20150101"
        # task 字段保留
        assert "task" in resolved

    def test_exp007_schema_valid(self):
        """exp_007 应通过 schema 验证。"""
        config_path = PROJECT_ROOT / "configs" / "research_experiments.json"
        schema_path = PROJECT_ROOT / "configs" / "research_experiments.schema.json"

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        try:
            from jsonschema import validate
            validate(instance=data, schema=schema)
        except ImportError:
            pytest.skip("jsonschema not installed")


# ---------------------------------------------------------------------------
# 测试：exp_002-exp_006 base_config 继承
# ---------------------------------------------------------------------------

class TestExp002To006BaseConfig:
    """验证 exp_002-exp_006 的 base_config 继承。"""

    def _load_experiments(self):
        config_path = PROJECT_ROOT / "configs" / "research_experiments.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["experiments"]

    def test_all_experiments_have_base_config(self):
        """所有实验都应有 base_config 字段。"""
        experiments = self._load_experiments()
        for exp in experiments:
            assert "base_config" in exp, f"{exp['experiment_id']} missing base_config"
            assert exp["base_config"] == "alpha_research_base"

    def test_exp002_resolves_correctly(self):
        """exp_002 应正确继承 base_config，保留 last_test_year='latest'。"""
        experiments = self._load_experiments()
        exp = next(e for e in experiments if e["experiment_id"] == "exp_002_ma_cross_with_market_filter")
        resolved = resolve_experiment_config(exp)

        # 继承的 cost_model
        assert resolved["cost_model"]["cash"] == 1000000
        assert resolved["cost_model"]["commission"] == 0.0001
        # 继承的 universe
        assert resolved["universe"]["market"] == "ALL"
        assert resolved["universe"]["security_type"] == "stock"
        # date_range: last_test_year 被实验覆盖为 'latest'
        assert resolved["date_range"]["last_test_year"] == "latest"
        assert resolved["date_range"]["train_start"] == "20150101"
        assert resolved["date_range"]["first_test_year"] == 2021
        assert resolved["date_range"]["incomplete_year"] == 2026
        # parameters: 实验特有参数保留
        assert resolved["parameters"]["fast_list"] == [5, 10, 20]
        assert resolved["parameters"]["slow_list"] == [60, 120, 250]
        assert resolved["parameters"]["benchmark_fast_list"] == [20]
        assert resolved["parameters"]["benchmark_slow_list"] == [120]
        # parameters: 继承的公共参数
        assert resolved["parameters"]["portfolio_size"] == 20
        assert resolved["parameters"]["benchmark_list"] == ["000300.SH", "000905.SH", "000852.SH"]
        assert resolved["parameters"]["min_train_rows"] == 1000

    def test_exp003_resolves_correctly(self):
        """exp_003 应正确继承 base_config，保留实验特有参数。"""
        experiments = self._load_experiments()
        exp = next(e for e in experiments if e["experiment_id"] == "exp_003_ma_v3_momentum_trend_filter")
        resolved = resolve_experiment_config(exp)

        # 继承的 cost_model
        assert resolved["cost_model"]["cash"] == 1000000
        # 继承的 universe
        assert resolved["universe"]["market"] == "ALL"
        # date_range: 完全继承
        assert resolved["date_range"]["train_start"] == "20150101"
        assert resolved["date_range"]["first_test_year"] == 2021
        assert resolved["date_range"]["last_test_year"] == 2025
        # parameters: 实验特有参数
        assert resolved["parameters"]["ma_mid_list"] == [20, 60]
        assert resolved["parameters"]["ma_long_list"] == [120, 250]
        assert resolved["parameters"]["momentum_window_list"] == [60, 120]
        assert resolved["parameters"]["max_train_volatility"] == 0.0
        assert resolved["parameters"]["min_train_calmar"] == 0.0
        # parameters: 继承的公共参数
        assert resolved["parameters"]["portfolio_size"] == 20
        assert resolved["parameters"]["benchmark_list"] == ["000300.SH", "000905.SH", "000852.SH"]
        assert resolved["parameters"]["benchmark_ma_list"] == [120, 250]

    def test_exp004_resolves_correctly(self):
        """exp_004 应正确继承 base_config。"""
        experiments = self._load_experiments()
        exp = next(e for e in experiments if e["experiment_id"] == "exp_004_next_alpha_research")
        resolved = resolve_experiment_config(exp)

        assert resolved["cost_model"]["cash"] == 1000000
        assert resolved["universe"]["market"] == "ALL"
        assert resolved["date_range"]["train_start"] == "20150101"
        # 实验特有参数
        assert "pure_momentum" in resolved["parameters"]["alpha_variant_list"]
        assert resolved["parameters"]["momentum_window_list"] == [60, 120, 250]
        # 继承的公共参数
        assert resolved["parameters"]["portfolio_size"] == 20

    def test_exp005_resolves_correctly(self):
        """exp_005 应正确继承 base_config。"""
        experiments = self._load_experiments()
        exp = next(e for e in experiments if e["experiment_id"] == "exp_005_alpha_v5_signal_diversification")
        resolved = resolve_experiment_config(exp)

        assert resolved["cost_model"]["cash"] == 1000000
        assert resolved["universe"]["market"] == "ALL"
        # 实验特有参数
        assert "momentum_reversion_blend" in resolved["parameters"]["alpha_variant_list"]
        assert resolved["parameters"]["reversion_window_list"] == [10, 20, 40]
        # 继承的公共参数
        assert resolved["parameters"]["portfolio_size"] == 20
        assert resolved["parameters"]["benchmark_list"] == ["000300.SH", "000905.SH", "000852.SH"]

    def test_exp006_resolves_correctly(self):
        """exp_006 应正确继承 base_config。"""
        experiments = self._load_experiments()
        exp = next(e for e in experiments if e["experiment_id"] == "exp_006_alpha_v6_non_momentum_signals")
        resolved = resolve_experiment_config(exp)

        assert resolved["cost_model"]["cash"] == 1000000
        assert resolved["universe"]["market"] == "ALL"
        # 实验特有参数
        assert "short_term_reversal" in resolved["parameters"]["alpha_variant_list"]
        assert resolved["parameters"]["reversal_window_list"] == [5, 10, 20]
        # 继承的公共参数
        assert resolved["parameters"]["portfolio_size"] == 20
        assert resolved["parameters"]["benchmark_ma_list"] == [120, 250]

    def test_all_experiments_schema_valid(self):
        """所有实验应通过 schema 验证。"""
        config_path = PROJECT_ROOT / "configs" / "research_experiments.json"
        schema_path = PROJECT_ROOT / "configs" / "research_experiments.schema.json"

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        try:
            from jsonschema import validate
            validate(instance=data, schema=schema)
        except ImportError:
            pytest.skip("jsonschema not installed")


# ---------------------------------------------------------------------------
# 测试：Task → Command 生成
# ---------------------------------------------------------------------------

class TestGenerateTaskCommand:
    """generate_task_command 从 task 配置生成 CLI 命令。"""

    def test_signal_evaluation_basic(self):
        """signal_evaluation: 基本命令生成。"""
        experiment = {
            "experiment_id": "exp_007_test",
            "task": {
                "signal": {"type": "rule", "variants": ["short_term_reversal", "low_volatility"]},
                "evaluation": {"signal_ic": True},
                "data": {"start": "20150101"},
            },
        }
        cmd = generate_task_command(experiment, "signal_evaluation")
        assert cmd is not None
        assert "evaluate_alpha_signals.py" in cmd
        assert "--experiment-id exp_007_test" in cmd
        assert "--alpha-variant short_term_reversal,low_volatility" in cmd
        assert "--start 20150101" in cmd

    def test_signal_evaluation_no_start(self):
        """signal_evaluation: 无 start 时不报错。"""
        experiment = {
            "experiment_id": "exp_test",
            "task": {
                "signal": {"variants": ["v1"]},
                "evaluation": {"signal_ic": True},
            },
        }
        cmd = generate_task_command(experiment, "signal_evaluation")
        assert cmd is not None
        assert "--start" not in cmd

    def test_signal_evaluation_no_ic(self):
        """signal_evaluation: evaluation.signal_ic 为 False 时返回 None。"""
        experiment = {
            "task": {
                "signal": {"variants": ["v1"]},
                "evaluation": {"signal_ic": False},
            },
        }
        assert generate_task_command(experiment, "signal_evaluation") is None

    def test_signal_evaluation_no_variants(self):
        """signal_evaluation: 无 variants 时返回 None。"""
        experiment = {
            "task": {
                "signal": {"type": "rule"},
                "evaluation": {"signal_ic": True},
            },
        }
        assert generate_task_command(experiment, "signal_evaluation") is None

    def test_signal_evaluation_single_variant(self):
        """signal_evaluation: 单个 variant 不加逗号。"""
        experiment = {
            "experiment_id": "exp_test",
            "task": {
                "signal": {"variants": ["short_term_reversal"]},
                "evaluation": {"signal_ic": True},
            },
        }
        cmd = generate_task_command(experiment, "signal_evaluation")
        assert "--alpha-variant short_term_reversal" in cmd
        assert "," not in cmd.split("--alpha-variant ")[1].split()[0]

    def test_walk_forward_basic(self):
        """walk_forward: 基本命令生成（使用 alpha_ver 动态脚本路径）。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_expression_layer",
            "task": {
                "signal": {"variants": ["short_term_reversal"]},
                "walk_forward": {
                    "first_test_year": 2021,
                    "last_test_year": 2025,
                    "portfolio_size": 20,
                },
                "data": {"universe": "ALL"},
            },
        }
        cmd = generate_task_command(experiment, "walk_forward")
        assert cmd is not None
        assert "validate_alpha_v7_research_candidates.py" in cmd
        assert "--market ALL" in cmd
        assert "--alpha-variant-list short_term_reversal" in cmd
        assert "--first-test-year 2021" in cmd
        assert "--last-test-year 2025" in cmd
        assert "--portfolio-size 20" in cmd

    def test_walk_forward_dynamic_version(self):
        """walk_forward: 脚本路径根据 experiment_id 中的 alpha 版本动态生成。"""
        experiment = {
            "experiment_id": "exp_008_alpha_v8_new_strategy",
            "task": {
                "signal": {"variants": ["my_variant"]},
                "walk_forward": {"first_test_year": 2022},
                "data": {"universe": "ALL"},
            },
        }
        cmd = generate_task_command(experiment, "walk_forward")
        assert cmd is not None
        assert "validate_alpha_v8_research_candidates.py" in cmd

    def test_walk_forward_no_alpha_version(self):
        """walk_forward: experiment_id 不含 alpha 版本时返回 None。"""
        experiment = {
            "experiment_id": "exp_007_test",
            "task": {
                "signal": {"variants": ["v1"]},
                "walk_forward": {"first_test_year": 2021},
                "data": {"universe": "ALL"},
            },
        }
        assert generate_task_command(experiment, "walk_forward") is None

    def test_walk_forward_no_wf(self):
        """walk_forward: 无 walk_forward 配置时返回 None。"""
        experiment = {
            "task": {"signal": {"variants": ["v1"]}},
        }
        assert generate_task_command(experiment, "walk_forward") is None

    def test_walk_forward_no_variants(self):
        """walk_forward: 无 variants 时返回 None。"""
        experiment = {
            "task": {
                "signal": {},
                "walk_forward": {"first_test_year": 2021},
            },
        }
        assert generate_task_command(experiment, "walk_forward") is None

    def test_unsupported_stage(self):
        """不支持的 stage 返回 None。"""
        experiment = {"task": {"signal": {"variants": ["v1"]}}}
        assert generate_task_command(experiment, "single_symbol_check") is None

    def test_no_task(self):
        """无 task 时返回 None。"""
        experiment = {"experiment_id": "test"}
        assert generate_task_command(experiment, "signal_evaluation") is None

    def test_exp007_task_generates_signal_evaluation(self):
        """exp_007 的 task 应能生成 signal_evaluation 命令。"""
        config_path = PROJECT_ROOT / "configs" / "research_experiments.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        exp = next(e for e in data["experiments"] if e["experiment_id"] == "exp_007_alpha_v7_expression_layer")
        cmd = generate_task_command(exp, "signal_evaluation")
        assert cmd is not None
        assert "evaluate_alpha_signals.py" in cmd
        assert "short_term_reversal" in cmd
        assert "low_volatility" in cmd
        assert "turnover_reversal" in cmd
        assert "volume_price_divergence" in cmd

    # --- batch_backtest ---

    def test_batch_backtest_basic(self):
        """batch_backtest: 基本命令生成。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_expression_layer",
            "task": {
                "signal": {"variants": ["short_term_reversal", "low_volatility"]},
                "data": {"universe": "ALL", "start": "20150101"},
            },
        }
        cmd = generate_task_command(experiment, "batch_backtest")
        assert cmd is not None
        assert "batch_alpha_v7_research_backtest_csv.py" in cmd
        assert "--market ALL" in cmd
        assert "--alpha-variant-list short_term_reversal,low_volatility" in cmd
        assert "--start 20150101" in cmd

    def test_batch_backtest_no_variants(self):
        """batch_backtest: 无 variants 时返回 None。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_test",
            "task": {"signal": {}, "data": {"universe": "ALL"}},
        }
        assert generate_task_command(experiment, "batch_backtest") is None

    def test_batch_backtest_with_param_grids(self):
        """batch_backtest: 参数网格从 task.parameters 读取。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_test",
            "task": {
                "signal": {"variants": ["short_term_reversal"]},
                "data": {"universe": "ALL"},
                "parameters": {
                    "reversal_window_list": [5, 10, 20],
                    "vol_window_list": [20, 60],
                    "benchmark_list": ["000300.SH", "000905.SH"],
                },
            },
        }
        cmd = generate_task_command(experiment, "batch_backtest")
        assert cmd is not None
        assert "--reversal-window-list 5,10,20" in cmd
        assert "--vol-window-list 20,60" in cmd
        assert "--benchmark-list 000300.SH,000905.SH" in cmd

    def test_batch_backtest_no_alpha_version(self):
        """batch_backtest: experiment_id 无 alpha 版本时返回 None。"""
        experiment = {
            "experiment_id": "exp_007_test",
            "task": {"signal": {"variants": ["v1"]}, "data": {}},
        }
        assert generate_task_command(experiment, "batch_backtest") is None

    # --- analysis ---

    def test_analysis_basic(self):
        """analysis: 基本命令生成。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_expression_layer",
            "task": {"signal": {"variants": ["v1"]}},
        }
        cmd = generate_task_command(experiment, "analysis")
        assert cmd is not None
        assert "analyze_alpha_v7_research_walk_forward_results.py" in cmd
        assert "--input-dir backtests/walk_forward_alpha_v7_research_csv" in cmd
        assert "--no-png" in cmd

    def test_analysis_with_portfolio_size(self):
        """analysis: walk_forward.portfolio_size 传递到 --portfolio-size。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_expression_layer",
            "task": {
                "signal": {"variants": ["v1"]},
                "walk_forward": {"portfolio_size": 30},
            },
        }
        cmd = generate_task_command(experiment, "analysis")
        assert cmd is not None
        assert "--portfolio-size 30" in cmd

    def test_analysis_no_alpha_version(self):
        """analysis: experiment_id 无 alpha 版本时返回 None。"""
        experiment = {
            "experiment_id": "exp_007_test",
            "task": {"signal": {"variants": ["v1"]}},
        }
        assert generate_task_command(experiment, "analysis") is None

    # --- diagnosis ---

    def test_diagnosis_basic(self):
        """diagnosis: 基本命令生成。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_expression_layer",
            "task": {"signal": {"variants": ["v1"]}},
        }
        cmd = generate_task_command(experiment, "diagnosis")
        assert cmd is not None
        assert "diagnose_alpha_v7_research_strategy_results.py" in cmd
        assert "--input-dir backtests/walk_forward_alpha_v7_research_csv" in cmd
        assert "--analysis-dir backtests/walk_forward_alpha_v7_research_analysis" in cmd
        assert "--no-png" in cmd

    def test_diagnosis_no_alpha_version(self):
        """diagnosis: experiment_id 无 alpha 版本时返回 None。"""
        experiment = {
            "experiment_id": "exp_007_test",
            "task": {"signal": {"variants": ["v1"]}},
        }
        assert generate_task_command(experiment, "diagnosis") is None

    # --- robustness ---

    def test_robustness_basic(self):
        """robustness: 基本命令生成。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_expression_layer",
            "task": {
                "signal": {"variants": ["v1"]},
                "data": {"universe": "ALL"},
            },
        }
        cmd = generate_task_command(experiment, "robustness")
        assert cmd is not None
        assert "validate_alpha_v7_robustness.py" in cmd
        assert "--input-tag ALL" in cmd
        assert "--walk-forward-dir backtests/walk_forward_alpha_v7_research_csv" in cmd
        assert "--no-png" in cmd

    def test_robustness_no_alpha_version(self):
        """robustness: experiment_id 无 alpha 版本时返回 None。"""
        experiment = {
            "experiment_id": "exp_007_test",
            "task": {"signal": {"variants": ["v1"]}, "data": {}},
        }
        assert generate_task_command(experiment, "robustness") is None

    # --- portfolio_backtest ---

    def test_portfolio_backtest_basic(self):
        """portfolio_backtest: 基本命令生成。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_expression_layer",
            "task": {
                "signal": {"variants": ["short_term_reversal"]},
                "data": {"universe": "ALL"},
            },
        }
        cmd = generate_task_command(experiment, "portfolio_backtest")
        assert cmd is not None
        assert "portfolio_backtest_csv.py" in cmd
        assert "--input-tag ALL" in cmd
        assert "--walk-forward-dir backtests/walk_forward_alpha_v7_research_csv" in cmd
        assert "--run-id exp_007_alpha_v7_expression_layer" in cmd
        assert "--no-png" in cmd

    def test_portfolio_backtest_with_backtest_params(self):
        """portfolio_backtest: 从 task.backtest 读取可选参数。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_expression_layer",
            "task": {
                "signal": {"variants": ["v1"]},
                "data": {"universe": "ALL"},
                "backtest": {
                    "max_positions": 30,
                    "max_weight": 0.1,
                    "lot_size": 100,
                    "initial_cash": 2000000,
                },
            },
        }
        cmd = generate_task_command(experiment, "portfolio_backtest")
        assert cmd is not None
        assert "--max-positions 30" in cmd
        assert "--max-weight 0.1" in cmd
        assert "--lot-size 100" in cmd
        assert "--initial-cash 2000000" in cmd

    def test_portfolio_backtest_no_alpha_version(self):
        """portfolio_backtest: experiment_id 无 alpha 版本时返回 None。"""
        experiment = {
            "experiment_id": "exp_007_test",
            "task": {"signal": {"variants": ["v1"]}, "data": {}},
        }
        assert generate_task_command(experiment, "portfolio_backtest") is None

    def test_portfolio_backtest_no_variants(self):
        """portfolio_backtest: 无 variants 时仍可生成（不依赖 variants）。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_expression_layer",
            "task": {"signal": {}, "data": {"universe": "ALL"}},
        }
        cmd = generate_task_command(experiment, "portfolio_backtest")
        # portfolio_backtest 不需要 variants，只需 alpha_version 和 universe
        assert cmd is not None
        assert "portfolio_backtest_csv.py" in cmd

    # --- model_training ---

    def test_model_training_score_col_basic(self):
        """model_training: score_col 模式基本命令生成。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_expression_layer",
            "task": {
                "model_training": {
                    "score_col": "feature/reversal_10d",
                    "feature_matrix": "factors/processed/feature_matrix/run_001/feature_matrix.parquet",
                },
            },
        }
        cmd = generate_task_command(experiment, "model_training")
        assert cmd is not None
        assert "train_alpha_model.py" in cmd
        assert "--feature-matrix factors/processed/feature_matrix/run_001/feature_matrix.parquet" in cmd
        assert "--score-col feature/reversal_10d" in cmd
        assert "--output-dir backtests/model_prediction/exp_007_alpha_v7_expression_layer" in cmd

    def test_model_training_alpha_variant_basic(self):
        """model_training: alpha_variant 模式基本命令生成。"""
        experiment = {
            "experiment_id": "exp_007_alpha_v7_expression_layer",
            "task": {
                "model_training": {
                    "alpha_variant": "short_term_reversal",
                    "feature_matrix": "factors/processed/feature_matrix/run_001/feature_matrix.parquet",
                },
            },
        }
        cmd = generate_task_command(experiment, "model_training")
        assert cmd is not None
        assert "--alpha-variant short_term_reversal" in cmd
        assert "--score-col" not in cmd

    def test_model_training_with_label_col(self):
        """model_training: 自定义 label_col。"""
        experiment = {
            "experiment_id": "exp_test",
            "task": {
                "model_training": {
                    "score_col": "feature/x",
                    "feature_matrix": "data.parquet",
                    "label_col": "label/ret_5d",
                },
            },
        }
        cmd = generate_task_command(experiment, "model_training")
        assert cmd is not None
        assert "--label-col label/ret_5d" in cmd

    def test_model_training_with_test_years_list(self):
        """model_training: test_years 列表转逗号分隔。"""
        experiment = {
            "experiment_id": "exp_test",
            "task": {
                "model_training": {
                    "score_col": "feature/x",
                    "feature_matrix": "data.parquet",
                    "test_years": [2022, 2023, 2024],
                },
            },
        }
        cmd = generate_task_command(experiment, "model_training")
        assert cmd is not None
        assert "--test-years 2022,2023,2024" in cmd

    def test_model_training_with_test_years_string(self):
        """model_training: test_years 字符串直接传递。"""
        experiment = {
            "experiment_id": "exp_test",
            "task": {
                "model_training": {
                    "score_col": "feature/x",
                    "feature_matrix": "data.parquet",
                    "test_years": "2022,2023,2024",
                },
            },
        }
        cmd = generate_task_command(experiment, "model_training")
        assert cmd is not None
        assert "--test-years 2022,2023,2024" in cmd

    def test_model_training_no_zscore(self):
        """model_training: no_zscore 选项。"""
        experiment = {
            "experiment_id": "exp_test",
            "task": {
                "model_training": {
                    "score_col": "feature/x",
                    "feature_matrix": "data.parquet",
                    "no_zscore": True,
                },
            },
        }
        cmd = generate_task_command(experiment, "model_training")
        assert cmd is not None
        assert "--no-zscore" in cmd

    def test_model_training_zscore_pred(self):
        """model_training: zscore_pred 选项。"""
        experiment = {
            "experiment_id": "exp_test",
            "task": {
                "model_training": {
                    "score_col": "feature/x",
                    "feature_matrix": "data.parquet",
                    "zscore_pred": True,
                },
            },
        }
        cmd = generate_task_command(experiment, "model_training")
        assert cmd is not None
        assert "--zscore-pred" in cmd

    def test_model_training_signal_threshold(self):
        """model_training: signal_threshold 选项。"""
        experiment = {
            "experiment_id": "exp_test",
            "task": {
                "model_training": {
                    "score_col": "feature/x",
                    "feature_matrix": "data.parquet",
                    "signal_threshold": 0.5,
                },
            },
        }
        cmd = generate_task_command(experiment, "model_training")
        assert cmd is not None
        assert "--signal-threshold 0.5" in cmd

    def test_model_training_no_model_training(self):
        """model_training: 无 model_training 配置时返回 None。"""
        experiment = {"task": {"signal": {"variants": ["v1"]}}}
        assert generate_task_command(experiment, "model_training") is None

    def test_model_training_no_feature_matrix(self):
        """model_training: 无 feature_matrix 路径时返回 None。"""
        experiment = {
            "task": {
                "model_training": {
                    "score_col": "feature/x",
                },
            },
        }
        assert generate_task_command(experiment, "model_training") is None

    def test_model_training_no_score_or_variant(self):
        """model_training: 无 score_col 和 alpha_variant 时返回 None。"""
        experiment = {
            "task": {
                "model_training": {
                    "feature_matrix": "data.parquet",
                },
            },
        }
        assert generate_task_command(experiment, "model_training") is None

    def test_model_training_all_options(self):
        """model_training: 全部选项组合。"""
        experiment = {
            "experiment_id": "exp_008_alpha_v8_test",
            "task": {
                "model_training": {
                    "alpha_variant": "low_volatility",
                    "feature_matrix": "factors/processed/feature_matrix/run_002/feature_matrix.parquet",
                    "label_col": "label/ret_20d",
                    "test_years": [2023, 2024],
                    "zscore_pred": True,
                    "signal_threshold": 0.1,
                },
            },
        }
        cmd = generate_task_command(experiment, "model_training")
        assert cmd is not None
        assert "train_alpha_model.py" in cmd
        assert "--alpha-variant low_volatility" in cmd
        assert "--label-col label/ret_20d" in cmd
        assert "--test-years 2023,2024" in cmd
        assert "--zscore-pred" in cmd
        assert "--signal-threshold 0.1" in cmd
        assert "--output-dir backtests/model_prediction/exp_008_alpha_v8_test" in cmd

    def test_task_has_stage_model_training(self):
        """task_has_stage: model_training 阶段在 task 中存在时返回 True。"""
        experiment = {
            "task": {
                "model_training": {
                    "score_col": "feature/x",
                    "feature_matrix": "data.parquet",
                },
            },
        }
        assert task_has_stage(experiment, "model_training") is True

    def test_task_has_stage_model_training_absent(self):
        """task_has_stage: model_training 阶段不在 task 中时返回 False。"""
        experiment = {"task": {"signal": {"variants": ["v1"]}}}
        assert task_has_stage(experiment, "model_training") is False

    # --- exp_007 集成 ---

    def test_exp007_generates_all_stages(self):
        """exp_007 的 task 应能生成全部 8 个已支持 stage 的命令。"""
        config_path = PROJECT_ROOT / "configs" / "research_experiments.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        exp = next(e for e in data["experiments"] if e["experiment_id"] == "exp_007_alpha_v7_expression_layer")
        for stage in ["signal_evaluation", "walk_forward", "batch_backtest", "analysis", "diagnosis", "robustness", "portfolio_backtest"]:
            cmd = generate_task_command(exp, stage)
            assert cmd is not None, f"stage={stage} 应生成命令但返回 None"


# ---------------------------------------------------------------------------
# 测试：基线配置缓存
# ---------------------------------------------------------------------------

class TestBaseConfigCache:
    """load_base_config 缓存行为。"""

    def setup_method(self):
        """每个测试前清空缓存。"""
        clear_base_config_cache()

    def teardown_method(self):
        """每个测试后清空缓存。"""
        clear_base_config_cache()

    def test_cache_populated_on_first_load(self):
        """首次加载后缓存应包含该配置。"""
        assert "cost_model_a_share_default" not in _BASE_CONFIG_CACHE
        load_base_config("cost_model_a_share_default")
        assert "cost_model_a_share_default" in _BASE_CONFIG_CACHE

    def test_cache_hit_returns_same_values(self):
        """缓存命中应返回相同的值。"""
        config1 = load_base_config("cost_model_a_share_default")
        config2 = load_base_config("cost_model_a_share_default")
        assert config1 == config2

    def test_cache_returns_deep_copy(self):
        """缓存返回深拷贝，修改不影响缓存。"""
        config1 = load_base_config("cost_model_a_share_default")
        config1["cost_model"]["cash"] = 999
        config2 = load_base_config("cost_model_a_share_default")
        assert config2["cost_model"]["cash"] == 1000000

    def test_inherited_configs_independently_cached(self):
        """继承链中的父配置应独立缓存。"""
        load_base_config("alpha_research_base")
        # alpha_research_base 继承 cost_model_a_share_default 和 walk_forward_default
        assert "cost_model_a_share_default" in _BASE_CONFIG_CACHE
        assert "walk_forward_default" in _BASE_CONFIG_CACHE
        assert "alpha_research_base" in _BASE_CONFIG_CACHE

    def test_clear_cache(self):
        """clear_base_config_cache 应清空所有缓存。"""
        load_base_config("cost_model_a_share_default")
        assert len(_BASE_CONFIG_CACHE) > 0
        clear_base_config_cache()
        assert len(_BASE_CONFIG_CACHE) == 0

    def test_cycle_detection_still_works_with_cache(self, tmp_path):
        """缓存不影响循环继承检测。"""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        (base_dir / "a.json").write_text(
            json.dumps({"_inherits": ["b"], "value_a": 1}),
            encoding="utf-8",
        )
        (base_dir / "b.json").write_text(
            json.dumps({"_inherits": ["a"], "value_b": 2}),
            encoding="utf-8",
        )

        import scripts.common.config_loader as cl
        old_dir = cl.BASE_CONFIG_DIR
        cl.BASE_CONFIG_DIR = base_dir
        cl._BASE_CONFIG_CACHE.clear()
        try:
            with pytest.raises(ValueError, match="循环继承"):
                load_base_config("a")
        finally:
            cl.BASE_CONFIG_DIR = old_dir
            cl._BASE_CONFIG_CACHE.clear()

    def test_nonexistent_still_raises_with_cache(self):
        """不存在的配置仍应抛出异常。"""
        with pytest.raises(FileNotFoundError):
            load_base_config("nonexistent_xyz")

    def test_resolve_uses_cache(self):
        """resolve_experiment_config 应受益于缓存。"""
        config_path = PROJECT_ROOT / "configs" / "research_experiments.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        experiments = data["experiments"]

        # 第一次 resolve 会填充缓存
        exp006 = next(e for e in experiments if e["experiment_id"] == "exp_006_alpha_v6_non_momentum_signals")
        resolve_experiment_config(exp006)
        assert "alpha_research_base" in _BASE_CONFIG_CACHE

        # 第二次 resolve 应命中缓存
        exp007 = next(e for e in experiments if e["experiment_id"] == "exp_007_alpha_v7_expression_layer")
        result = resolve_experiment_config(exp007)
        assert result["cost_model"]["cash"] == 1000000

    def test_cache_returns_different_objects(self):
        """多次缓存命中应返回不同对象（深拷贝）。"""
        config1 = load_base_config("cost_model_a_share_default")
        config2 = load_base_config("cost_model_a_share_default")
        assert config1 is not config2
        assert config1["cost_model"] is not config2["cost_model"]


# ---------------------------------------------------------------------------
# 测试：_deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    """_deep_merge 的递归合并逻辑。"""

    def test_flat_override(self):
        """平铺 dict 覆盖。"""
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}
        assert result is base  # 修改并返回 base

    def test_nested_merge(self):
        """嵌套 dict 递归合并。"""
        base = {"a": {"x": 1, "y": 2}, "b": 10}
        override = {"a": {"y": 99, "z": 3}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 3}, "b": 10}

    def test_deep_nested_merge(self):
        """三层嵌套递归合并。"""
        base = {"l1": {"l2": {"l3": {"v1": 1, "v2": 2}}}}
        override = {"l1": {"l2": {"l3": {"v2": 99, "v3": 3}}}}
        result = _deep_merge(base, override)
        assert result == {"l1": {"l2": {"l3": {"v1": 1, "v2": 99, "v3": 3}}}}

    def test_override_dict_replaces_non_dict(self):
        """override 的 dict 类型覆盖 base 的非 dict 类型。"""
        base = {"a": 42}
        override = {"a": {"nested": True}}
        result = _deep_merge(base, override)
        assert result == {"a": {"nested": True}}

    def test_override_non_dict_replaces_dict(self):
        """override 的非 dict 类型覆盖 base 的 dict 类型。"""
        base = {"a": {"nested": True}}
        override = {"a": 42}
        result = _deep_merge(base, override)
        assert result == {"a": 42}

    def test_empty_override(self):
        """空 override 不改变 base。"""
        base = {"a": 1, "b": {"c": 2}}
        result = _deep_merge(base, {})
        assert result == {"a": 1, "b": {"c": 2}}

    def test_empty_base(self):
        """空 base 被 override 填充。"""
        base = {}
        override = {"a": 1, "b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_list_values_not_recursed(self):
        """列表类型值直接覆盖，不递归。"""
        base = {"a": [1, 2]}
        override = {"a": [3, 4, 5]}
        result = _deep_merge(base, override)
        assert result == {"a": [3, 4, 5]}

    def test_mixed_types(self):
        """混合类型场景。"""
        base = {
            "str": "hello",
            "num": 10,
            "nested": {"a": 1, "b": [1, 2]},
            "list": [10, 20],
        }
        override = {
            "num": 99,
            "nested": {"b": [3, 4], "c": "new"},
            "list": [30],
            "extra": True,
        }
        result = _deep_merge(base, override)
        assert result["str"] == "hello"
        assert result["num"] == 99
        assert result["nested"] == {"a": 1, "b": [3, 4], "c": "new"}
        assert result["list"] == [30]
        assert result["extra"] is True


# ---------------------------------------------------------------------------
# 测试：_extract_alpha_version
# ---------------------------------------------------------------------------

class TestExtractAlphaVersion:
    """_extract_alpha_version 的版本提取逻辑。"""

    def test_v4(self):
        """exp_004 应提取 v4。"""
        assert _extract_alpha_version("exp_004_alpha_v4_research") == "v4"

    def test_v5(self):
        """exp_005 应提取 v5。"""
        assert _extract_alpha_version("exp_005_alpha_v5_research") == "v5"

    def test_v6(self):
        """exp_006 应提取 v6。"""
        assert _extract_alpha_version("exp_006_alpha_v6_non_momentum_signals") == "v6"

    def test_v7(self):
        """exp_007 应提取 v7。"""
        assert _extract_alpha_version("exp_007_alpha_v7_expression_layer") == "v7"

    def test_no_alpha_returns_none(self):
        """不含 alpha 版本的 experiment_id 应返回 None。"""
        assert _extract_alpha_version("exp_001_ma_momentum") is None

    def test_empty_string(self):
        """空字符串应返回 None。"""
        assert _extract_alpha_version("") is None

    def test_v10_multidigit(self):
        """多位数版本号（如 v10）应正确提取。"""
        assert _extract_alpha_version("exp_010_alpha_v10_future") == "v10"
