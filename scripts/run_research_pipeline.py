# -*- coding: utf-8 -*-
"""
configs 驱动的研究流水线入口。

从 configs/research_experiments.json 读取实验配置，
根据 experiment_id 找到实验，按 commands 中定义的阶段命令进行 dry-run 或 execute。

用法：
    python scripts/run_research_pipeline.py --list-experiments
    python scripts/run_research_pipeline.py --experiment-id exp_002_ma_cross_with_market_filter --show-experiment
    python scripts/run_research_pipeline.py --experiment-id exp_002_ma_cross_with_market_filter --dry-run
    python scripts/run_research_pipeline.py --experiment-id exp_002_ma_cross_with_market_filter --stages analysis,diagnosis --execute
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.config_loader import generate_task_command  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    "single_symbol_check",
    "signal_evaluation",
    "batch_backtest",
    "walk_forward",
    "model_training",
    "analysis",
    "diagnosis",
    "robustness",
    "portfolio_backtest",
]

STAGE_INDEX = {s: i for i, s in enumerate(STAGE_ORDER)}

STAGE_OUTPUT_MAP = {
    "single_symbol_check": "single_symbol_dir",
    "signal_evaluation": "signal_evaluation_dir",
    "batch_backtest": "batch_dir",
    "walk_forward": "walk_forward_dir",
    "model_training": "model_training_dir",
    "analysis": "analysis_dir",
    "diagnosis": "diagnosis_dir",
    "robustness": "robustness_dir",
    "portfolio_backtest": "portfolio_backtest_dir",
}

DEFAULT_CONFIG_PATH = Path("configs") / "research_experiments.json"
DEFAULT_SCHEMA_PATH = Path("configs") / "research_experiments.schema.json"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="configs 驱动的研究流水线入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config paths
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="实验配置 JSON 路径")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="JSON Schema 路径")

    # Experiment selection
    parser.add_argument("--experiment-id", help="要运行的实验 ID")
    parser.add_argument("--stages", help="逗号分隔的 stage 列表，如 analysis,diagnosis")

    # Mode
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不执行命令")
    parser.add_argument("--execute", action="store_true", help="执行模式，实际运行命令")

    # Environment
    parser.add_argument("--require-conda-env", default="research-env", help="要求的 conda 环境名")
    parser.add_argument("--allow-env-mismatch", action="store_true", help="调试用：环境不匹配时只警告不阻断")

    # Info commands
    parser.add_argument("--list-experiments", action="store_true", help="列出所有实验")
    parser.add_argument("--show-experiment", action="store_true", help="展示指定实验详情")

    # Skip
    parser.add_argument("--skip-existing", action="store_true", help="跳过已有输出的 stage")

    # RunRecorder
    parser.add_argument(
        "--use-recorder", action="store_true",
        help="使用 RunRecorder 记录运行参数、阶段状态和 Record Template 摘要到 experiments/ds_flow/",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """读取并解析 research_experiments.json。"""
    p = Path(config_path)
    if not p.exists():
        logger.error(f"配置文件不存在: {p.resolve()}")
        sys.exit(1)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"配置文件 JSON 格式错误: {p.resolve()}")
        print(f"  {e}")
        sys.exit(1)


def validate_schema(config: dict, schema_path: str, allow_degraded: bool = False) -> list[str]:
    """校验配置。优先 jsonschema，否则根据 allow_degraded 决定是否退化。返回错误列表（空=通过）。"""
    try:
        import jsonschema

        sp = Path(schema_path)
        if not sp.exists():
            return [f"Schema 文件不存在: {sp.resolve()}"]
        with open(sp, encoding="utf-8") as f:
            schema = json.load(f)
        try:
            jsonschema.validate(instance=config, schema=schema)
            return []
        except jsonschema.ValidationError as e:
            return [f"Schema 校验失败: {e.message} (路径: {'/'.join(str(p) for p in e.absolute_path)})"]
    except ImportError:
        if not allow_degraded:
            return [
                "缺少 jsonschema 依赖，无法执行完整 schema 校验。",
                "请安装: pip install jsonschema",
                "或使用 --allow-env-mismatch 以使用退化校验（仅用于调试）。",
            ]
        # 退化校验（仅在显式允许时）
        logger.warning("jsonschema 未安装，退化为基础校验。请安装 jsonschema 以获得完整校验。")
        errors = []
        if "schema_version" not in config:
            errors.append("缺少 schema_version 字段")
        if "experiments" not in config or not isinstance(config["experiments"], list):
            errors.append("experiments 字段不存在或不是列表")
        else:
            for i, exp in enumerate(config["experiments"]):
                if "experiment_id" not in exp:
                    errors.append(f"experiments[{i}] 缺少 experiment_id")
                if "commands" not in exp:
                    errors.append(f"experiments[{i}].{exp.get('experiment_id', '?')} 缺少 commands")
        return errors


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------


def check_research_env(require_env: str, allow_mismatch: bool) -> dict:
    """检查当前是否在 research-env conda 环境中。返回环境信息 dict。"""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    python_exe = sys.executable
    python_ver = sys.version
    cwd = os.getcwd()

    env_info = {
        "conda_env": conda_env,
        "python_executable": python_exe,
        "python_version": python_ver,
        "cwd": cwd,
        "is_target_env": conda_env == require_env,
    }

    # 项目根目录检查
    config_in_cwd = Path(cwd) / "configs" / "research_experiments.json"
    env_info["config_found_in_cwd"] = config_in_cwd.exists()

    if not env_info["is_target_env"]:
        msg = (
            f"[ERROR] 当前 conda 环境为 '{conda_env}'，要求 '{require_env}'。\n"
            f"  请先激活环境:\n"
            f"    conda activate {require_env}\n"
            f"  然后重新运行命令。"
        )
        if allow_mismatch:
            logger.warning(msg.replace("[ERROR] ", ""))
            logger.warning("  --allow-env-mismatch 已传入，继续执行。")
        else:
            logger.error(msg.replace("[ERROR] ", ""))
            sys.exit(1)

    if not env_info["config_found_in_cwd"]:
        logger.warning(f"当前工作目录 {cwd} 下未找到 configs/research_experiments.json")

    return env_info


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------


def find_experiment(config: dict, experiment_id: str) -> dict:
    """根据 experiment_id 查找实验，找不到则报错并列出可用 ID。"""
    for exp in config.get("experiments", []):
        if exp.get("experiment_id") == experiment_id:
            return exp
    available = [e.get("experiment_id", "?") for e in config.get("experiments", [])]
    logger.error(f"未找到 experiment_id: {experiment_id}")
    print(f"  可用的 experiment_id:")
    for aid in available:
        print(f"    - {aid}")
    sys.exit(1)


def list_experiments(config: dict) -> None:
    """打印所有实验的摘要信息。"""
    experiments = config.get("experiments", [])
    if not experiments:
        print("没有登记的实验。")
        return
    print(f"共 {len(experiments)} 个实验:\n")
    for exp in experiments:
        decision = exp.get("decision", {})
        print(f"  experiment_id:     {exp.get('experiment_id', '?')}")
        print(f"  status:            {exp.get('status', '?')}")
        print(f"  strategy_version:  {exp.get('strategy_version', '?')}")
        print(f"  strategy_name:     {exp.get('strategy_name', '?')}")
        print(f"  decision:          {decision.get('decision', '?')}")
        print()


def _find_latest_manifest(experiment_id: str, project_root: Path) -> dict | None:
    """查找给定 experiment_id 的最近一次 run_manifest.json 或 pipeline_run_summary.json。"""
    runs_dir = project_root / "logs" / "pipeline_runs"
    if not runs_dir.is_dir():
        return None
    # 按目录名倒序（时间戳在前），取最新的匹配项
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and experiment_id in d.name],
        reverse=True,
    )
    for run_dir in candidates:
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        summary_path = run_dir / "pipeline_run_summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
    return None


def show_experiment(exp: dict, project_root: Path | None = None) -> None:
    """打印实验的核心配置。"""
    fields = [
        "strategy_name", "hypothesis", "status",
        "parameters", "outputs", "decision", "commands",
    ]
    print(f"experiment_id: {exp.get('experiment_id', '?')}\n")
    for field in fields:
        val = exp.get(field)
        if val is None:
            print(f"  {field}: null")
        elif isinstance(val, dict):
            print(f"  {field}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {field}: {val}")
        print()

    # 展示最近一次运行摘要
    if project_root is not None:
        experiment_id = exp.get("experiment_id", "")
        manifest = _find_latest_manifest(experiment_id, project_root)
        if manifest:
            print("最近一次运行:")
            print(f"  run_id: {manifest.get('run_id', '?')}")
            print(f"  success: {manifest.get('success', '?')}")
            print(f"  started_at: {manifest.get('started_at', '?')}")
            print(f"  git_commit: {manifest.get('git_commit', '?')[:12]}")
            print(f"  git_dirty: {manifest.get('git_dirty', '?')}")
            env = manifest.get("environment", {})
            print(f"  conda_env: {env.get('actual_conda_env', manifest.get('actual_conda_env', '?'))}")
            stages = manifest.get("stages", [])
            results = manifest.get("stage_results", [])
            if results:
                print(f"  stages executed: {len(results)}")
                for r in results:
                    rc = r.get("return_code", "?")
                    dur = r.get("duration_seconds", 0)
                    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                    print(f"    {r.get('stage', '?')}: {status} ({dur:.1f}s)")
            artifacts = manifest.get("output_artifacts", {})
            if artifacts:
                total_files = sum(a.get("file_count", 0) for a in artifacts.values())
                total_bytes = sum(a.get("total_bytes", 0) for a in artifacts.values())
                print(f"  output_artifacts: {total_files} files, {total_bytes / 1024:.0f} KB across {len(artifacts)} stages")
        else:
            print("最近一次运行: 无记录")


def resolve_commands(stage_value) -> list[str]:
    """将 stage 命令值统一处理为 list[str]，过滤空命令。"""
    if isinstance(stage_value, str):
        return [stage_value] if stage_value.strip() else []
    if isinstance(stage_value, list):
        return [str(c) for c in stage_value if str(c).strip()]
    return []


def parse_stages_arg(stages_arg: str | None) -> list[str]:
    """解析 --stages 参数，返回按 STAGE_ORDER 排序的合法 stage 列表。"""
    if not stages_arg:
        return list(STAGE_ORDER)
    requested = [s.strip() for s in stages_arg.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGE_ORDER]
    if unknown:
        logger.error(f"未知 stage: {', '.join(unknown)}")
        print(f"  可用 stage: {', '.join(STAGE_ORDER)}")
        sys.exit(1)
    # 按 STAGE_ORDER 重排，而不是使用用户输入顺序
    deduped = sorted(set(requested), key=lambda s: STAGE_INDEX[s])
    if len(deduped) < len(requested):
        seen, dupes = set(), set()
        for s in requested:
            if s in seen:
                dupes.add(s)
            seen.add(s)
        logger.warning(f"--stages 中有重复项，已自动去重: {', '.join(sorted(dupes))}")
    return deduped


def validate_stage_commands(stages: list[str], experiment: dict, explicit_stages: bool) -> list[str]:
    """校验所选 stage 是否都有 command 定义（commands 或 task 生成）。返回缺失的 stage 列表。"""
    commands = experiment.get("commands", {})
    missing = []
    for stage in stages:
        stage_cmds = resolve_commands(commands.get(stage))
        if not stage_cmds:
            # 尝试从 task 生成命令
            task_cmd = generate_task_command(experiment, stage)
            if not task_cmd:
                missing.append(stage)
    if missing and explicit_stages:
        # 显式请求的 stage 缺 command 时必须失败
        logger.error(f"以下 stage 没有定义 command: {', '.join(missing)}")
        print(f"  实验 '{experiment.get('experiment_id', '?')}' 的 commands 中缺少这些 stage。")
        sys.exit(1)
    elif missing:
        # 默认全量流程中缺 command 只警告
        logger.warning(f"以下 stage 没有定义 command，将跳过: {', '.join(missing)}")
    return missing


def should_skip_stage(stage: str, experiment: dict, skip_existing: bool) -> bool:
    """检查是否应跳过该 stage。"""
    output_key = STAGE_OUTPUT_MAP.get(stage)
    if not output_key:
        return False
    output_dir = experiment.get("outputs", {}).get(output_key)
    if not output_dir:
        return False
    p = Path(output_dir)
    if p.exists() and any(p.iterdir()):
        if skip_existing:
            print(f"  [SKIP] stage '{stage}' — 输出目录已存在且非空: {output_dir}")
            return True
        else:
            logger.info(f"stage '{stage}' — 输出目录已存在且非空: {output_dir} (传入 --skip-existing 可跳过)")
    return False


# ---------------------------------------------------------------------------
# Mode: dry-run
# ---------------------------------------------------------------------------


def run_dry_run(experiment: dict, stages: list[str], env_info: dict, skip_existing: bool) -> None:
    """预览模式：打印将要执行的信息，不执行任何命令。"""
    print("=" * 60)
    print("MODE: dry-run")
    print("=" * 60)
    print()
    print(f"  experiment_id:     {experiment.get('experiment_id', '?')}")
    print(f"  strategy_name:     {experiment.get('strategy_name', '?')}")
    print(f"  status:            {experiment.get('status', '?')}")
    print()
    print(f"  conda_env:         {env_info['conda_env']}")
    print(f"  python_executable: {env_info['python_executable']}")
    print(f"  python_version:    {env_info['python_version'].split()[0]}")
    print()
    print(f"  stages: {', '.join(stages)}")
    print()

    commands = experiment.get("commands", {})
    for stage in stages:
        if should_skip_stage(stage, experiment, skip_existing):
            continue
        stage_cmds = resolve_commands(commands.get(stage))
        if not stage_cmds:
            # 尝试从 task 生成命令
            task_cmd = generate_task_command(experiment, stage)
            if task_cmd:
                stage_cmds = [task_cmd]
                print(f"  [{stage}] $ {task_cmd}  (from task)")
                continue
        for cmd in stage_cmds:
            print(f"  [{stage}] $ {cmd}")
    print()
    print("dry-run 完成，未执行任何命令。")


# ---------------------------------------------------------------------------
# Mode: execute
# ---------------------------------------------------------------------------


def run_command_streaming(
    cmd_parts: list[str],
    cwd: str,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, str, str]:
    """实时流式执行子命令，stdout/stderr 同时输出到控制台和日志文件。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd_parts,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=env,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    stdout_f = open(stdout_path, "w", encoding="utf-8")
    stderr_f = open(stderr_path, "w", encoding="utf-8")

    def _reader(pipe, lines, file_obj, write_to_stdout):
        try:
            for raw_line in iter(pipe.readline, b""):
                if not raw_line:
                    break
                try:
                    line = raw_line.decode("utf-8", errors="replace")
                except Exception:
                    line = raw_line.decode("latin-1", errors="replace")
                line = line.rstrip("\n\r")
                lines.append(line)
                print(line, file=sys.stdout if write_to_stdout else sys.stderr, flush=True)
                file_obj.write(line + "\n")
                file_obj.flush()
        except OSError:
            pass
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    t_out = threading.Thread(
        target=_reader, args=(proc.stdout, stdout_lines, stdout_f, True), daemon=True,
    )
    t_err = threading.Thread(
        target=_reader, args=(proc.stderr, stderr_lines, stderr_f, False), daemon=True,
    )
    t_out.start()
    t_err.start()

    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError:
            proc.kill()
        raise

    t_out.join(timeout=10)
    t_err.join(timeout=10)

    stdout_f.close()
    stderr_f.close()

    return proc.returncode, "\n".join(stdout_lines), "\n".join(stderr_lines)


def _get_git_provenance(project_root: Path) -> tuple[str, bool]:
    """Return (commit_hash, is_dirty) for the working tree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True, text=True, timeout=10,
        )
        commit = result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning(f"无法获取 git commit: {e}")
        commit = "unknown"

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            capture_output=True, text=True, timeout=10,
        )
        dirty = bool(result.stdout.strip()) if result.returncode == 0 else True
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning(f"无法获取 git 状态: {e}")
        dirty = True

    return commit, dirty


def _compute_config_hash(experiment: dict) -> str:
    """SHA-256 hex digest of the experiment config dict (canonical JSON)."""
    canonical = json.dumps(experiment, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scan_dir(directory: Path, *, base: Path | None = None) -> list[tuple[str, int]]:
    """扫描目录，返回 [(relative_path, size), ...] 列表。TOCTOU 安全。"""
    if base is None:
        base = directory
    result = []
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            try:
                result.append((str(p.relative_to(base)), p.stat().st_size))
            except OSError:
                continue  # file deleted between rglob and stat
    return result


def _compute_data_snapshot(project_root: Path) -> dict:
    """Lightweight fingerprint of the data/qmt_export directory."""
    data_dir = project_root / "data" / "qmt_export"
    if not data_dir.is_dir():
        return {"error": f"directory not found: {data_dir}", "file_count": 0, "total_bytes": 0, "hash": ""}

    entries = _scan_dir(data_dir)
    hasher = hashlib.sha256()
    for rel_path, size in entries:
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(str(size).encode("utf-8"))

    return {
        "file_count": len(entries),
        "total_bytes": sum(s for _, s in entries),
        "hash": hasher.hexdigest(),
    }


def _compute_schema_hash(project_root: Path) -> str:
    """SHA-256 hex digest of the JSON Schema file."""
    schema_path = project_root / DEFAULT_SCHEMA_PATH
    if not schema_path.exists():
        return "unknown"
    return hashlib.sha256(schema_path.read_bytes()).hexdigest()


def _collect_output_artifacts(
    experiment: dict,
    stages: list[str],
    project_root: Path,
) -> dict[str, dict]:
    """为每个已执行的 stage 收集输出目录的文件清单。

    Returns: {stage: {"output_dir": str, "file_count": int, "total_bytes": int, "files": [str, ...]}}
    """
    artifacts = {}
    for stage in stages:
        output_key = STAGE_OUTPUT_MAP.get(stage)
        if not output_key:
            continue
        output_dir_rel = experiment.get("outputs", {}).get(output_key)
        if not output_dir_rel:
            continue
        output_dir = project_root / output_dir_rel
        if not output_dir.is_dir():
            artifacts[stage] = {"output_dir": output_dir_rel, "file_count": 0, "total_bytes": 0, "files": []}
            continue
        entries = _scan_dir(output_dir, base=project_root)
        artifacts[stage] = {
            "output_dir": output_dir_rel,
            "file_count": len(entries),
            "total_bytes": sum(s for _, s in entries),
            "files": [p for p, _ in entries[:50]],  # cap at 50 to keep manifest readable
        }
    return artifacts


def _try_collect_records(
    experiment: dict,
    stages: list[str],
    project_root: Path,
) -> list:
    """尝试从已完成 stage 的输出目录加载 Record Template。

    Returns: list of BaseRecord 实例（可能为空）。
    """
    from scripts.common.records import (
        SignalEvaluationRecord,
        WalkForwardRecord,
        DiagnosisRecord,
        RobustnessRecord,
    )

    # stage -> (output_key, RecordClass, extra_kwargs_factory)
    # kwargs_factory: None -> 单记录（variant=exp_id）；callable -> 返回 dict 或 None
    #   返回 None 表示需要扫描子目录发现多个记录（如 signal_evaluation）
    _STAGE_RECORD_MAP = {
        "walk_forward": ("walk_forward_dir", WalkForwardRecord, None),
        "diagnosis": ("diagnosis_dir", DiagnosisRecord, None),
        "robustness": ("robustness_dir", RobustnessRecord, None),
        "signal_evaluation": ("signal_evaluation_dir", SignalEvaluationRecord, "scan"),
    }

    records = []
    outputs = experiment.get("outputs", {})
    exp_id = experiment.get("experiment_id", "")

    for stage in stages:
        entry = _STAGE_RECORD_MAP.get(stage)
        if entry is None:
            continue
        output_key, record_cls, kwargs_factory = entry
        output_dir_rel = outputs.get(output_key)
        if not output_dir_rel:
            continue
        output_dir = project_root / output_dir_rel
        if not output_dir.is_dir():
            continue

        if kwargs_factory == "scan":
            # signal_evaluation 输出目录结构: <variant>/<label_col>/
            # 需要扫描子目录为每个 variant/label_col 组合创建记录
            records.extend(
                _collect_signal_eval_records(output_dir, record_cls)
            )
            continue

        try:
            kwargs = {"variant": exp_id}
            if callable(kwargs_factory):
                extra = kwargs_factory(experiment)
                if extra is None:
                    continue
                kwargs.update(extra)
            rec = record_cls.from_dir(output_dir, **kwargs)
            # 只在有实际数据时添加
            d = rec.to_dict()
            has_data = any(
                v is not None and v != "" and v != [] and v != 0
                for k, v in d.items()
                if k not in ("record_type", "variant", "generated_at")
            )
            if has_data:
                records.append(rec)
        except Exception as e:
            logger.warning("加载 %s Record Template 失败: %s", stage, e)

    return records


def _collect_signal_eval_records(
    output_dir: Path,
    record_cls: type,
) -> list:
    """扫描 signal_evaluation 输出目录，为每个 variant/label_col 组合创建记录。

    目录结构：
        <output_dir>/<variant>/<label_col>/signal_ic_summary.csv
    """
    records = []
    if not output_dir.is_dir():
        return records

    for variant_dir in sorted(output_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        for label_dir in sorted(variant_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label_col = label_dir.name
            # 跳过没有 IC summary 的目录
            ic_path = label_dir / "signal_ic_summary.csv"
            if not ic_path.exists():
                continue
            try:
                rec = record_cls.from_dir(
                    label_dir, variant=variant, label_col=label_col
                )
                d = rec.to_dict()
                has_data = any(
                    v is not None and v != "" and v != [] and v != 0
                    for k, v in d.items()
                    if k not in ("record_type", "variant", "generated_at", "label_col")
                )
                if has_data:
                    records.append(rec)
            except Exception as e:
                logger.warning(
                    "加载 signal_evaluation Record 失败 (%s/%s): %s",
                    variant, label_col, e,
                )
    return records


def run_execute(
    experiment: dict,
    stages: list[str],
    env_info: dict,
    project_root: Path,
    skip_existing: bool,
    use_recorder: bool = False,
) -> None:
    """执行模式：按 stage 顺序执行命令，记录日志。"""
    experiment_id = experiment.get("experiment_id", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{timestamp}_{experiment_id}"
    log_dir = project_root / "logs" / "pipeline_runs" / run_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODE: execute")
    print("=" * 60)
    print()
    print(f"  experiment_id:     {experiment_id}")
    print(f"  strategy_name:     {experiment.get('strategy_name', '?')}")
    print(f"  conda_env:         {env_info['conda_env']}")
    print(f"  python_executable: {env_info['python_executable']}")
    print(f"  log_dir:           {log_dir.relative_to(project_root)}")
    print()

    started_at = datetime.now().isoformat()
    commands = experiment.get("commands", {})
    stage_results = []
    success = True

    # --- Provenance (compute early for RunRecorder) ---
    git_commit, git_dirty = _get_git_provenance(project_root)
    config_hash = _compute_config_hash(experiment)

    # --- RunRecorder (optional) ---
    recorder = None
    if use_recorder:
        try:
            from scripts.common.recorder import RunRecorder
            recorder = RunRecorder(
                experiment_id,
                run_id=run_dir_name,
                base_dir=project_root / "experiments" / "ds_flow",
                config_hash=config_hash,
                code_version=git_commit,
            )
            recorder.set_params({
                "strategy_name": experiment.get("strategy_name", ""),
                "strategy_version": experiment.get("strategy_version", ""),
                "hypothesis": experiment.get("hypothesis", ""),
                "parameters": experiment.get("parameters", {}),
                "cost_model": experiment.get("cost_model", {}),
                "stages_requested": stages,
                "conda_env": env_info.get("conda_env", ""),
                "git_dirty": git_dirty,
            })
            print(f"  RunRecorder 已创建: {recorder.run_dir.relative_to(project_root)}")
        except Exception as e:
            logger.warning("RunRecorder 创建失败，继续执行: %s", e)
            recorder = None

    for idx, stage in enumerate(stages, 1):
        if should_skip_stage(stage, experiment, skip_existing):
            stage_results.append({
                "stage": stage,
                "command": None,
                "return_code": None,
                "duration_seconds": 0,
                "stdout_log": None,
                "stderr_log": None,
                "skipped": True,
            })
            continue

        stage_cmds = resolve_commands(commands.get(stage))
        if not stage_cmds:
            # 尝试从 task 生成命令
            task_cmd = generate_task_command(experiment, stage)
            if task_cmd:
                stage_cmds = [task_cmd]
                logger.info("stage '%s' 从 task 生成命令: %s", stage, task_cmd)
        if not stage_cmds:
            # 无命令的 stage 标记为 completed（用于 Record Template 收集）
            if recorder is not None:
                recorder.mark_stage(stage, "completed")
            stage_results.append({
                "stage": stage,
                "command": None,
                "return_code": 0,
                "duration_seconds": 0,
                "stdout_log": None,
                "stderr_log": None,
                "no_commands": True,
            })
            continue
        for cmd_idx, cmd in enumerate(stage_cmds, 1):
            print(f"  [{idx:02d}/{len(stages):02d}] {stage}: {cmd}")
            cmd_parts = shlex.split(cmd, posix=False)

            stage_start = datetime.now()

            safe_name = stage.replace(" ", "_")
            suffix = f"_cmd{cmd_idx:02d}" if len(stage_cmds) > 1 else ""
            stdout_log = log_dir / f"stage_{idx:02d}_{safe_name}{suffix}_stdout.txt"
            stderr_log = log_dir / f"stage_{idx:02d}_{safe_name}{suffix}_stderr.txt"

            try:
                return_code, stdout_text, stderr_text = run_command_streaming(
                    cmd_parts, str(project_root), stdout_log, stderr_log,
                )
            except KeyboardInterrupt:
                print("\n  [INTERRUPTED] 用户中断 (Ctrl+C)")
                sys.exit(130)
            except Exception as e:
                return_code = -1
                stdout_text = ""
                stderr_text = str(e)
            stage_end = datetime.now()
            duration = (stage_end - stage_start).total_seconds()

            stage_result = {
                "stage": stage,
                "command": cmd,
                "return_code": return_code,
                "duration_seconds": round(duration, 2),
                "stdout_log": str(stdout_log.relative_to(project_root)),
                "stderr_log": str(stderr_log.relative_to(project_root)),
            }
            stage_results.append(stage_result)

            if return_code == 0:
                print(f"        OK ({duration:.1f}s)")
                if recorder is not None:
                    recorder.mark_stage(stage, "completed")
            else:
                print(f"        FAILED (return_code={return_code}, {duration:.1f}s)")
                print(f"        stderr: {stderr_text[:200] if stderr_text else '(empty)'}")
                success = False
                break
        if not success:
            break

    finished_at = datetime.now().isoformat()

    # Provenance (git_commit, git_dirty, config_hash already computed above)
    schema_hash = _compute_schema_hash(project_root)
    data_snapshot = _compute_data_snapshot(project_root)
    output_artifacts = _collect_output_artifacts(experiment, stages, project_root)

    # Run manifest (extended from summary)
    manifest = {
        "manifest_version": "1.0",
        "run_id": run_dir_name,
        "experiment_id": experiment_id,
        "strategy_family": experiment.get("strategy_family", ""),
        "strategy_version": experiment.get("strategy_version", ""),
        "strategy_name": experiment.get("strategy_name", ""),
        "status": experiment.get("status", ""),
        "started_at": started_at,
        "finished_at": finished_at,
        "success": success,
        # Git provenance
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        # Config & schema hashes
        "config_hash": config_hash,
        "schema_hash": schema_hash,
        # Data snapshot
        "data_snapshot": data_snapshot,
        # Environment
        "environment": {
            "require_conda_env": env_info.get("conda_env_target", ""),
            "actual_conda_env": env_info["conda_env"],
            "python_executable": env_info["python_executable"],
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        # Execution
        "stages": stages,
        "stage_results": stage_results,
        # Output artifacts
        "output_artifacts": output_artifacts,
        # Experiment config snapshot (for reproducibility)
        "experiment_config": {
            "hypothesis": experiment.get("hypothesis", ""),
            "parameters": experiment.get("parameters", {}),
            "cost_model": experiment.get("cost_model", {}),
            "commands": experiment.get("commands", {}),
            "outputs": experiment.get("outputs", {}),
        },
    }
    manifest_path = log_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Backward-compatible summary (flat subset of manifest)
    summary = {
        **{k: manifest[k] for k in [
            "run_id", "experiment_id", "strategy_name", "status",
            "started_at", "finished_at", "success",
            "git_commit", "git_dirty", "config_hash", "data_snapshot",
        ]},
        "require_conda_env": manifest["environment"]["require_conda_env"],
        "actual_conda_env": manifest["environment"]["actual_conda_env"],
        "python_executable": manifest["environment"]["python_executable"],
        "stages": manifest["stages"],
        "stage_results": manifest["stage_results"],
    }
    summary_path = log_dir / "pipeline_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- RunRecorder: apply Record Templates and finish ---
    if recorder is not None:
        try:
            # 从 pipeline manifest 提取关键指标到 recorder
            recorder.record_metric("pipeline_success", success)
            recorder.record_metric("pipeline_run_dir", str(log_dir.relative_to(project_root)))
            if git_commit:
                recorder.record_metric("git_commit", git_commit)
            recorder.record_metric("git_dirty", git_dirty)

            # 记录产物
            for stage_name, art in output_artifacts.items():
                if art.get("file_count", 0) > 0:
                    recorder.record_artifact(f"pipeline_{stage_name}", art["output_dir"])

            # 自动加载 Record Template
            completed_stages = [s for s in stages if recorder.manifest.stage_status.get(s) == "completed"]
            records = _try_collect_records(experiment, completed_stages, project_root)
            for rec in records:
                rec.apply_to_recorder(recorder)
                rec.save(recorder.run_dir / f"{rec.record_type}_record.json")
            if records:
                print(f"  RunRecorder: 已加载 {len(records)} 个 Record Template")

            # 标记失败的 stage
            for sr in stage_results:
                if sr.get("return_code") and sr["return_code"] != 0:
                    recorder.mark_stage(sr["stage"], "failed")
                elif sr.get("skipped"):
                    recorder.mark_stage(sr["stage"], "skipped")

            recorder.finish(status="completed" if success else "failed")
            print(f"  RunRecorder manifest: {recorder.manifest_path.relative_to(project_root)}")
        except Exception as e:
            logger.warning("RunRecorder 完成记录失败: %s", e)

    print()
    if success:
        print(f"  所有 stage 执行成功。日志: {log_dir.relative_to(project_root)}")
    else:
        failed = [r for r in stage_results if r.get("return_code") and r["return_code"] != 0]
        if failed:
            f = failed[0]
            print(f"  执行失败于 stage '{f['stage']}' (return_code={f['return_code']})")
            print(f"  日志: {log_dir.relative_to(project_root)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    """配置 logging：控制台输出 WARNING+，verbose 模式输出 DEBUG+。"""
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    _setup_logging(verbose=getattr(args, "verbose", False))

    # --- List experiments (no env check needed) ---
    if args.list_experiments:
        config = load_config(args.config)
        list_experiments(config)
        return

    # --- Show experiment (no env check needed) ---
    if args.show_experiment:
        if not args.experiment_id:
            logger.error("--show-experiment 需要配合 --experiment-id")
            sys.exit(1)
        config = load_config(args.config)
        exp = find_experiment(config, args.experiment_id)
        show_experiment(exp, project_root=project_root)
        return

    # --- Validate mode ---
    if args.dry_run and args.execute:
        logger.error("--dry-run 和 --execute 不能同时传入。")
        sys.exit(1)

    # --- Load & validate config ---
    config = load_config(args.config)
    errors = validate_schema(config, args.schema, allow_degraded=args.allow_env_mismatch)
    if errors:
        for err in errors:
            logger.error(err)
        sys.exit(1)

    # --- Find experiment ---
    if not args.experiment_id:
        logger.error("需要 --experiment-id（或使用 --list-experiments）")
        sys.exit(1)
    experiment = find_experiment(config, args.experiment_id)

    # --- Parse stages ---
    stages = parse_stages_arg(args.stages)

    # --- Validate stage commands ---
    explicit_stages = args.stages is not None
    missing_stages = validate_stage_commands(stages, experiment, explicit_stages)
    if missing_stages:
        stages = [s for s in stages if s not in missing_stages]

    # --- Environment check ---
    env_info = check_research_env(args.require_conda_env, args.allow_env_mismatch)
    env_info["conda_env_target"] = args.require_conda_env

    # --- Default to dry-run ---
    if not args.dry_run and not args.execute:
        logger.info("未指定 --dry-run 或 --execute，默认使用 dry-run 模式。")
        args.dry_run = True

    # --- Print header ---
    print()
    print(f"  project_root:      {project_root}")
    print(f"  config:            {args.config}")
    print(f"  experiment_id:     {args.experiment_id}")
    print()

    # --- Run ---
    if args.dry_run:
        run_dry_run(experiment, stages, env_info, args.skip_existing)
    else:
        run_execute(experiment, stages, env_info, project_root, args.skip_existing, use_recorder=args.use_recorder)


if __name__ == "__main__":
    main()
