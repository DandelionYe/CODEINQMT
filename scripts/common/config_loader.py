# -*- coding: utf-8 -*-
"""
scripts/common/config_loader.py

配置继承与结构化 Task 支持。

设计原则：
- 基线配置存放在 configs/base/ 目录，以 JSON 文件形式管理。
- 实验配置可通过 base_config 字段引用基线，实现配置继承。
- 继承链支持多级（base 可以 _inherits 其他 base），有环检测。
- 实验特定字段覆盖基线字段（浅合并：顶层 key 级别覆盖）。
- task 字段提供结构化实验定义，与 commands 字段并存。
- pipeline 优先使用 commands；当某个 stage 支持 task 后再切换。

金融正确性检查：
- 本模块只处理配置元数据，不涉及价格/收益计算。
- 配置继承不会改变回测、信号计算或数据处理的金融逻辑。
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_DIR = PROJECT_ROOT / "configs" / "base"

# 模块级缓存：避免每次 resolve_experiment_config 都从磁盘重新加载基线配置。
# 键为配置名，值为合并后的 dict（原始副本，不含元数据字段）。
_BASE_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个 dict。override 中的值覆盖 base。

    对于两个 dict 类型的值，递归合并；其他类型直接覆盖。
    修改 base 并返回。
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# 1. 基线配置加载
# ---------------------------------------------------------------------------

def load_base_config(name: str, _visited: set[str] | None = None) -> dict[str, Any]:
    """加载基线配置文件，支持 _inherits 递归合并。

    参数：
        name: 基线配置名（不含 .json 后缀），对应 configs/base/<name>.json。
        _visited: 内部用于环检测。

    返回：
        合并后的配置 dict（深拷贝）。_inherits、_description、_updated_at 字段被移除。

    异常：
        FileNotFoundError: 基线配置文件不存在。
        ValueError: 检测到循环继承。

    缓存：
        使用模块级 _BASE_CONFIG_CACHE 缓存已加载的基线配置，避免重复磁盘读取。
        返回深拷贝，防止调用方修改污染缓存。
    """
    # 检查缓存（递归调用也会命中，因为父配置已独立缓存）
    if name in _BASE_CONFIG_CACHE:
        return copy.deepcopy(_BASE_CONFIG_CACHE[name])

    if _visited is None:
        _visited = set()

    if name in _visited:
        raise ValueError(f"检测到循环继承: {' -> '.join(_visited)} -> {name}")

    _visited.add(name)

    config_path = BASE_CONFIG_DIR / f"{name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"基线配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 递归合并父配置
    parents = config.pop("_inherits", [])
    merged: dict[str, Any] = {}
    for parent_name in parents:
        parent_config = load_base_config(parent_name, _visited.copy())
        _deep_merge(merged, parent_config)

    # 移除元数据字段
    config.pop("_description", None)
    config.pop("_updated_at", None)

    # 当前配置深度覆盖父配置
    _deep_merge(merged, config)

    # 写入缓存
    _BASE_CONFIG_CACHE[name] = copy.deepcopy(merged)

    return merged


def clear_base_config_cache() -> None:
    """清空基线配置缓存。

    用于测试或当配置文件在运行期间被修改时需要重新加载。
    """
    _BASE_CONFIG_CACHE.clear()


def resolve_experiment_config(experiment: dict[str, Any]) -> dict[str, Any]:
    """解析实验配置，合并 base_config 继承。

    参数：
        experiment: 实验 dict（来自 research_experiments.json 的一个条目）。

    返回：
        合并后的实验配置 dict。base_config 字段被解析并合并，
        实验特定字段覆盖基线字段。

    用法：
        experiment = resolve_experiment_config(raw_experiment)
        cost_model = experiment.get("cost_model", {})
        walk_forward_params = {
            k: experiment.get("parameters", {}).get(k)
            for k in ["first_test_year", "last_test_year", "portfolio_size"]
        }
    """
    base_config_name = experiment.get("base_config")
    if not base_config_name:
        return experiment

    base = load_base_config(base_config_name)

    # 浅合并：实验字段覆盖基线字段
    merged = {**base, **experiment}

    # 特殊处理嵌套字段：parameters 和 cost_model 需要深度合并
    if "parameters" in base and "parameters" in experiment:
        merged_parameters = {**base["parameters"], **experiment["parameters"]}
        merged["parameters"] = merged_parameters

    if "cost_model" in base and "cost_model" in experiment:
        merged_cost = {**base["cost_model"], **experiment["cost_model"]}
        merged["cost_model"] = merged_cost

    if "universe" in base and "universe" in experiment:
        merged_universe = {**base["universe"], **experiment["universe"]}
        merged["universe"] = merged_universe

    if "date_range" in base and "date_range" in experiment:
        merged_date = {**base["date_range"], **experiment["date_range"]}
        merged["date_range"] = merged_date

    # base_config 已被合并，移除原始引用
    merged.pop("base_config", None)

    return merged


# ---------------------------------------------------------------------------
# 2. Task 配置辅助
# ---------------------------------------------------------------------------

def get_task_config(experiment: dict[str, Any]) -> dict[str, Any] | None:
    """获取实验的结构化 task 配置。

    参数：
        experiment: 实验 dict。

    返回：
        task dict，如果实验没有定义 task 则返回 None。
    """
    return experiment.get("task")


def task_has_stage(experiment: dict[str, Any], stage: str) -> bool:
    """检查实验的 task 配置是否支持指定 stage。

    参数：
        experiment: 实验 dict。
        stage: pipeline stage 名称。

    返回：
        True 如果 task 中定义了该 stage 的配置。
    """
    task = get_task_config(experiment)
    if task is None:
        return False

    # task 中的 stage 对应关系
    stage_task_map = {
        "signal_evaluation": "evaluation",
        "walk_forward": "walk_forward",
        "batch_backtest": "signal",     # 需要 task.signal.variants
        "analysis": "signal",           # 从 experiment_id 推导路径
        "diagnosis": "signal",          # 从 experiment_id 推导路径
        "robustness": "signal",         # 需要 task.data.universe
        "portfolio_backtest": "signal", # 需要 task.data.universe + alpha_version
        "model_training": "model_training",
        "backtest": "backtest",
        "data": "data",
    }
    task_key = stage_task_map.get(stage, stage)
    return task_key in task


def merge_task_defaults(task: dict[str, Any], base_task: dict[str, Any]) -> dict[str, Any]:
    """合并 task 配置与基线默认值。

    参数：
        task: 实验特定的 task 配置。
        base_task: 基线 task 默认值。

    返回：
        合并后的 task dict。
    """
    merged = {**base_task, **task}

    # 深合并嵌套字段
    for key in ["data", "features", "labels", "signal", "walk_forward",
                "evaluation", "backtest", "records"]:
        if key in base_task and key in task:
            if isinstance(base_task[key], dict) and isinstance(task[key], dict):
                merged[key] = {**base_task[key], **task[key]}

    return merged


# ---------------------------------------------------------------------------
# 3. Task → Command 生成
# ---------------------------------------------------------------------------

# 每个 stage 的 task-to-command 生成器。
# 签名: (task: dict, experiment_id: str) -> str | None
_TASK_COMMAND_BUILDERS: dict[str, Any] = {}


def _build_signal_evaluation_command(task: dict[str, Any], experiment_id: str) -> str | None:
    """从 task 配置生成 signal_evaluation stage 的 CLI 命令。"""
    evaluation = task.get("evaluation", {})
    if not evaluation.get("signal_ic"):
        return None

    signal = task.get("signal", {})
    variants = signal.get("variants")
    if not variants:
        return None

    data = task.get("data", {})
    start = data.get("start")

    parts = ["python scripts/evaluate_alpha_signals.py"]
    parts.append(f"--experiment-id {experiment_id}")
    if isinstance(variants, list):
        parts.append(f"--alpha-variant {','.join(variants)}")
    else:
        parts.append(f"--alpha-variant {variants}")
    if start:
        parts.append(f"--start {start}")

    return " ".join(parts)


def _build_walk_forward_command(task: dict[str, Any], experiment_id: str) -> str | None:
    """从 task 配置生成 walk_forward stage 的 CLI 命令。"""
    wf = task.get("walk_forward", {})
    if not wf:
        return None

    signal = task.get("signal", {})
    variants = signal.get("variants")
    if not variants:
        return None

    alpha_ver = _extract_alpha_version(experiment_id)
    if alpha_ver is None:
        return None

    data = task.get("data", {})
    universe = data.get("universe", "ALL")

    parts = [f"python scripts/validate_alpha_{alpha_ver}_research_candidates.py"]
    parts.append(f"--market {universe}")

    if isinstance(variants, list):
        parts.append(f"--alpha-variant-list {','.join(variants)}")
    else:
        parts.append(f"--alpha-variant-list {variants}")

    if "first_test_year" in wf:
        parts.append(f"--first-test-year {wf['first_test_year']}")
    if "last_test_year" in wf:
        parts.append(f"--last-test-year {wf['last_test_year']}")
    if "portfolio_size" in wf:
        parts.append(f"--portfolio-size {wf['portfolio_size']}")

    return " ".join(parts)


def _extract_alpha_version(experiment_id: str) -> str | None:
    """从 experiment_id 中提取 alpha 版本标识（如 'v7'）。

    示例：'exp_007_alpha_v7_expression_layer' -> 'v7'
    """
    parts = experiment_id.split("_")
    for i, part in enumerate(parts):
        if part == "alpha" and i + 1 < len(parts):
            return parts[i + 1]
    return None


# 参数网格映射：task.parameters 中的 key → CLI 参数名
_PARAM_GRID_MAP: list[tuple[str, str]] = [
    ("reversal_window_list", "--reversal-window-list"),
    ("vol_window_list", "--vol-window-list"),
    ("turnover_short_list", "--turnover-short-list"),
    ("turnover_long_list", "--turnover-long-list"),
    ("divergence_window_list", "--divergence-window-list"),
    ("benchmark_list", "--benchmark-list"),
    ("benchmark_ma_list", "--benchmark-ma-list"),
]


def _build_batch_backtest_command(task: dict[str, Any], experiment_id: str) -> str | None:
    """从 task 配置生成 batch_backtest stage 的 CLI 命令。"""
    signal = task.get("signal", {})
    variants = signal.get("variants")
    if not variants:
        return None

    alpha_ver = _extract_alpha_version(experiment_id)
    if alpha_ver is None:
        return None

    data = task.get("data", {})
    universe = data.get("universe", "ALL")
    start = data.get("start")

    parts = [f"python scripts/batch_alpha_{alpha_ver}_research_backtest_csv.py"]
    parts.append(f"--market {universe}")
    if isinstance(variants, list):
        parts.append(f"--alpha-variant-list {','.join(variants)}")
    else:
        parts.append(f"--alpha-variant-list {variants}")
    if start:
        parts.append(f"--start {start}")

    params = task.get("parameters", {})
    for param_key, cli_flag in _PARAM_GRID_MAP:
        val = params.get(param_key)
        if val is not None:
            if isinstance(val, list):
                parts.append(f"{cli_flag} {','.join(str(v) for v in val)}")
            else:
                parts.append(f"{cli_flag} {val}")

    return " ".join(parts)


def _build_analysis_command(task: dict[str, Any], experiment_id: str) -> str | None:
    """从 task 配置生成 analysis stage 的 CLI 命令。"""
    alpha_ver = _extract_alpha_version(experiment_id)
    if alpha_ver is None:
        return None

    wf_input_dir = f"backtests/walk_forward_alpha_{alpha_ver}_research_csv"
    output_dir = f"backtests/walk_forward_alpha_{alpha_ver}_research_analysis"

    parts = [f"python scripts/analyze_alpha_{alpha_ver}_research_walk_forward_results.py"]
    parts.append(f"--input-dir {wf_input_dir}")
    parts.append("--no-png")

    wf = task.get("walk_forward", {})
    if "portfolio_size" in wf:
        parts.append(f"--portfolio-size {wf['portfolio_size']}")

    return " ".join(parts)


def _build_diagnosis_command(task: dict[str, Any], experiment_id: str) -> str | None:
    """从 task 配置生成 diagnosis stage 的 CLI 命令。"""
    alpha_ver = _extract_alpha_version(experiment_id)
    if alpha_ver is None:
        return None

    wf_input_dir = f"backtests/walk_forward_alpha_{alpha_ver}_research_csv"
    analysis_dir = f"backtests/walk_forward_alpha_{alpha_ver}_research_analysis"

    parts = [f"python scripts/diagnose_alpha_{alpha_ver}_research_strategy_results.py"]
    parts.append(f"--input-dir {wf_input_dir}")
    parts.append(f"--analysis-dir {analysis_dir}")
    parts.append("--no-png")

    wf = task.get("walk_forward", {})
    if "portfolio_size" in wf:
        parts.append(f"--portfolio-size {wf['portfolio_size']}")

    return " ".join(parts)


def _build_robustness_command(task: dict[str, Any], experiment_id: str) -> str | None:
    """从 task 配置生成 robustness stage 的 CLI 命令。"""
    alpha_ver = _extract_alpha_version(experiment_id)
    if alpha_ver is None:
        return None

    data = task.get("data", {})
    universe = data.get("universe", "ALL")
    wf_input_dir = f"backtests/walk_forward_alpha_{alpha_ver}_research_csv"

    parts = [f"python scripts/validate_alpha_{alpha_ver}_robustness.py"]
    parts.append(f"--input-tag {universe}")
    parts.append(f"--walk-forward-dir {wf_input_dir}")
    parts.append("--no-png")

    return " ".join(parts)


def _build_portfolio_backtest_command(task: dict[str, Any], experiment_id: str) -> str | None:
    """从 task 配置生成 portfolio_backtest stage 的 CLI 命令。"""
    alpha_ver = _extract_alpha_version(experiment_id)
    if alpha_ver is None:
        return None

    data = task.get("data", {})
    universe = data.get("universe", "ALL")
    wf_input_dir = f"backtests/walk_forward_alpha_{alpha_ver}_research_csv"

    parts = ["python scripts/portfolio_backtest_csv.py"]
    parts.append(f"--input-tag {universe}")
    parts.append(f"--walk-forward-dir {wf_input_dir}")
    parts.append(f"--run-id {experiment_id}")
    file_prefix = f"wf_{alpha_ver}_stock"
    parts.append(f"--file-prefix {file_prefix}")

    # 可选参数从 task.backtest 读取
    backtest = task.get("backtest", {})
    if "max_positions" in backtest:
        parts.append(f"--max-positions {backtest['max_positions']}")
    if "max_weight" in backtest:
        parts.append(f"--max-weight {backtest['max_weight']}")
    if "lot_size" in backtest:
        parts.append(f"--lot-size {backtest['lot_size']}")
    if "initial_cash" in backtest:
        parts.append(f"--initial-cash {backtest['initial_cash']}")

    parts.append("--no-png")

    return " ".join(parts)


def _build_model_training_command(task: dict[str, Any], experiment_id: str) -> str | None:
    """从 task 配置生成 model_training stage 的 CLI 命令。

    task.model_training 字段支持两种模式：
      - score_col 模式：直接使用 feature matrix 中的预计算列。
      - alpha_variant 模式：使用表达式构建（需要原始价格列在 feature matrix 中）。

    至少需要 score_col 或 alpha_variant 之一，以及 feature_matrix 路径。
    """
    mt = task.get("model_training", {})
    if not mt:
        return None

    # 必须有 feature_matrix 路径
    feature_matrix = mt.get("feature_matrix")
    if not feature_matrix:
        return None

    parts = ["python scripts/train_alpha_model.py"]
    parts.append(f"--feature-matrix {feature_matrix}")

    # score_col 或 alpha_variant（二选一）
    score_col = mt.get("score_col")
    alpha_variant = mt.get("alpha_variant")
    if score_col:
        parts.append(f"--score-col {score_col}")
    elif alpha_variant:
        parts.append(f"--alpha-variant {alpha_variant}")
    else:
        return None

    # 可选参数
    if "label_col" in mt:
        parts.append(f"--label-col {mt['label_col']}")
    if "test_years" in mt:
        years = mt["test_years"]
        if isinstance(years, list):
            parts.append(f"--test-years {','.join(str(y) for y in years)}")
        else:
            parts.append(f"--test-years {years}")
    if mt.get("no_zscore"):
        parts.append("--no-zscore")
    if mt.get("zscore_pred"):
        parts.append("--zscore-pred")
    if "signal_threshold" in mt:
        parts.append(f"--signal-threshold {mt['signal_threshold']}")

    # 输出目录：使用 experiment_id 作为子目录
    output_dir = f"backtests/model_prediction/{experiment_id}"
    parts.append(f"--output-dir {output_dir}")

    return " ".join(parts)


_TASK_COMMAND_BUILDERS["signal_evaluation"] = _build_signal_evaluation_command
_TASK_COMMAND_BUILDERS["walk_forward"] = _build_walk_forward_command
_TASK_COMMAND_BUILDERS["batch_backtest"] = _build_batch_backtest_command
_TASK_COMMAND_BUILDERS["analysis"] = _build_analysis_command
_TASK_COMMAND_BUILDERS["diagnosis"] = _build_diagnosis_command
_TASK_COMMAND_BUILDERS["robustness"] = _build_robustness_command
_TASK_COMMAND_BUILDERS["portfolio_backtest"] = _build_portfolio_backtest_command
_TASK_COMMAND_BUILDERS["model_training"] = _build_model_training_command


def generate_task_command(experiment: dict[str, Any], stage: str) -> str | None:
    """从实验的 task 配置生成指定 stage 的 CLI 命令。

    当 experiment.commands 中没有某个 stage 的命令时，
    可以尝试从 task 字段自动生成。
    目前支持 signal_evaluation、walk_forward、batch_backtest、analysis、diagnosis、robustness、portfolio_backtest、model_training。

    参数：
        experiment: 已解析的实验 dict（应已调用 resolve_experiment_config）。
        stage: pipeline stage 名称。

    返回：
        命令字符串，如果 task 不支持该 stage 则返回 None。
    """
    task = get_task_config(experiment)
    if task is None:
        return None

    builder = _TASK_COMMAND_BUILDERS.get(stage)
    if builder is None:
        return None

    experiment_id = experiment.get("experiment_id", "unknown")
    return builder(task, experiment_id)


# ---------------------------------------------------------------------------
# 4. 配置验证
# ---------------------------------------------------------------------------

def validate_base_config_chain() -> list[str]:
    """验证所有基线配置的继承链是否合法。

    返回：
        错误消息列表，空列表表示全部合法。
    """
    errors: list[str] = []

    if not BASE_CONFIG_DIR.exists():
        return [f"基线配置目录不存在: {BASE_CONFIG_DIR}"]

    for config_file in sorted(BASE_CONFIG_DIR.glob("*.json")):
        name = config_file.stem
        try:
            load_base_config(name)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            errors.append(f"基线配置 {name}: {e}")

    return errors


def list_base_configs() -> list[str]:
    """列出所有可用的基线配置名。

    返回：
        配置名列表（不含 .json 后缀）。
    """
    if not BASE_CONFIG_DIR.exists():
        return []
    return sorted(f.stem for f in BASE_CONFIG_DIR.glob("*.json"))
