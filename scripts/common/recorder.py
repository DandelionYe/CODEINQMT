# -*- coding: utf-8 -*-
"""
scripts/common/recorder.py

轻量 RunRecorder，用于将每次 pipeline 运行的参数、指标、产物和阶段状态
持久化为结构化的 run_manifest.json。

借鉴 Qlib 的 Recorder / Record Template 思路，但保持本地文件系统实现，
不依赖 MLflow 或外部服务。

目录结构：
    experiments/ds_flow/<experiment_id>/runs/<run_id>/
        run_manifest.json       # 运行主清单
        params.json             # 参数快照（可选，从 manifest 中拆出）
        metrics.json            # 指标快照（可选，从 manifest 中拆出）
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.common.constants import PROJECT_ROOT


def _utc_now_iso() -> str:
    """返回 UTC 当前时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_run_id() -> str:
    """生成短 run_id，格式：run_<8位hex>。"""
    return "run_" + uuid.uuid4().hex[:8]


@dataclass
class RunManifest:
    """运行清单数据类。"""

    experiment_id: str
    run_id: str
    stage: str = ""
    started_at: str = ""
    finished_at: str = ""
    status: str = "running"  # running / completed / failed
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path
    stage_status: Dict[str, str] = field(default_factory=dict)  # stage -> status
    config_hash: str = ""
    code_version: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化的 dict。"""
        return asdict(self)

    def save(self, path: Path) -> None:
        """保存为 JSON 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        """从 JSON 文件加载。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class RunRecorder:
    """
    轻量运行记录器。

    用法：
        recorder = RunRecorder("exp_007_alpha_v7_expression_layer")
        recorder.set_params({"alpha_variant": "short_term_reversal", "reversal_window": 5})
        recorder.record_metric("sharpe", 1.23)
        recorder.record_artifact("report", "backtests/analysis/xxx.csv")
        recorder.mark_stage("walk_forward", "completed")
        recorder.finish()
    """

    def __init__(
        self,
        experiment_id: str,
        run_id: Optional[str] = None,
        base_dir: Optional[Path] = None,
        stage: str = "",
        config_hash: str = "",
        code_version: str = "",
    ) -> None:
        self.experiment_id = experiment_id
        self.run_id = run_id or _generate_run_id()
        self.base_dir = base_dir or PROJECT_ROOT / "experiments" / "ds_flow"
        self.run_dir = self.base_dir / experiment_id / "runs" / self.run_id
        self.manifest_path = self.run_dir / "run_manifest.json"

        self._manifest = RunManifest(
            experiment_id=experiment_id,
            run_id=self.run_id,
            stage=stage,
            started_at=_utc_now_iso(),
            config_hash=config_hash,
            code_version=code_version,
        )

    @property
    def manifest(self) -> RunManifest:
        return self._manifest

    def set_params(self, params: Dict[str, Any]) -> None:
        """设置运行参数。"""
        self._manifest.params.update(params)

    def record_metric(self, name: str, value: Any) -> None:
        """记录单个指标。"""
        self._manifest.metrics[name] = value

    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """批量记录指标。"""
        self._manifest.metrics.update(metrics)

    def record_artifact(self, name: str, path: str) -> None:
        """记录产物路径。"""
        self._manifest.artifacts[name] = path

    def mark_stage(self, stage: str, status: str = "completed") -> None:
        """标记某个 stage 的完成状态。"""
        self._manifest.stage_status[stage] = status

    def set_status(self, status: str) -> None:
        """设置整体状态。"""
        self._manifest.status = status

    def finish(self, status: str = "completed") -> None:
        """完成运行并保存 manifest。"""
        self._manifest.finished_at = _utc_now_iso()
        self._manifest.status = status
        self._manifest.save(self.manifest_path)

    def save(self) -> None:
        """保存当前 manifest（不改变状态）。"""
        self._manifest.save(self.manifest_path)

    @staticmethod
    def load_manifest(experiment_id: str, run_id: str, base_dir: Optional[Path] = None) -> RunManifest:
        """加载已有 manifest。"""
        bd = base_dir or PROJECT_ROOT / "experiments" / "ds_flow"
        path = bd / experiment_id / "runs" / run_id / "run_manifest.json"
        return RunManifest.load(path)

    @staticmethod
    def list_runs(experiment_id: str, base_dir: Optional[Path] = None) -> List[str]:
        """列出某个实验的所有 run_id。"""
        bd = base_dir or PROJECT_ROOT / "experiments" / "ds_flow"
        runs_dir = bd / experiment_id / "runs"
        if not runs_dir.exists():
            return []
        return sorted(
            d.name for d in runs_dir.iterdir()
            if d.is_dir() and (d / "run_manifest.json").exists()
        )
