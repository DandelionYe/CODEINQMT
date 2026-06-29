# -*- coding: utf-8 -*-
"""
scripts/common/records.py

Record Template 模块：将 D's_Flow 各阶段的输出标准化为结构化记录，
与 RunRecorder 配合，为每次 pipeline 运行生成可索引、可比较的记录。

借鉴 Qlib 的 SignalRecord / SigAnaRecord / PortAnaRecord 设计，
但保持本地文件系统实现，不依赖 MLflow。

Record Template 类型：
  - SignalEvaluationRecord：信号评估阶段的 IC/RankIC/long-short 摘要
  - WalkForwardRecord：walk-forward 验证阶段的样本外表现摘要
  - DiagnosisRecord：诊断阶段的参数稳定性/年度一致性摘要
  - RobustnessRecord：稳健性 gate 阶段的通过/拒绝结果

用法：
    from scripts.common.records import SignalEvaluationRecord

    rec = SignalEvaluationRecord.from_dir(
        output_dir=Path("backtests/signal_evaluation/exp_007/ret_1d"),
        variant="short_term_reversal",
        label_col="ret_1d",
    )
    rec.save(Path("experiments/ds_flow/exp_007/runs/run_xxx/signal_eval_record.json"))
    rec.apply_to_recorder(run_recorder)  # 自动写入 RunRecorder
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------


@dataclass
class BaseRecord:
    """Record Template 基类。"""

    record_type: str = ""
    variant: str = ""
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化的 dict。"""
        return asdict(self)

    def save(self, path: Path) -> None:
        """保存为 JSON 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=_json_default)
        logger.info("Record 已保存到：%s", path)

    def apply_to_recorder(self, recorder: Any) -> None:
        """将关键指标写入 RunRecorder。子类应重写此方法。"""
        pass


def _json_default(obj: Any) -> Any:
    """JSON 序列化默认处理：numpy 类型转 Python 原生类型。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# SignalEvaluationRecord
# ---------------------------------------------------------------------------


@dataclass
class SignalEvaluationRecord(BaseRecord):
    """信号评估记录：从 signal_evaluation 输出目录加载 IC/RankIC/long-short 摘要。"""

    record_type: str = "signal_evaluation"
    label_col: str = ""

    # IC 摘要
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    rank_ic_mean: Optional[float] = None
    rank_ic_std: Optional[float] = None
    icir: Optional[float] = None
    ic_t_stat: Optional[float] = None
    ic_positive_rate: Optional[float] = None
    n_days: Optional[int] = None

    # 分位多空收益
    long_short_daily_mean: Optional[float] = None
    long_short_daily_std: Optional[float] = None
    long_short_ir: Optional[float] = None
    long_short_win_rate: Optional[float] = None

    # 覆盖率
    coverage_mean: Optional[float] = None
    coverage_min: Optional[float] = None

    # 信号自相关（lag-1）
    autocorr_lag1: Optional[float] = None

    # 信号质量标签
    signal_quality: str = ""  # STRONG / MODERATE / WEAK / UNKNOWN

    @classmethod
    def from_dir(
        cls,
        output_dir: Path,
        variant: str = "",
        label_col: str = "",
    ) -> "SignalEvaluationRecord":
        """从 signal_evaluation 输出目录构建记录。

        Parameters
        ----------
        output_dir : Path
            signal_evaluation 输出目录，含 signal_ic_summary.csv 等文件。
        variant : str
            Alpha variant 名称。
        label_col : str
            标签列名（如 ret_1d）。
        """
        from datetime import datetime

        rec = cls(variant=variant, label_col=label_col, generated_at=datetime.now().isoformat())
        output_dir = Path(output_dir)

        # IC summary
        ic_summary_path = output_dir / "signal_ic_summary.csv"
        if ic_summary_path.exists():
            try:
                df = pd.read_csv(ic_summary_path, encoding="utf-8-sig")
                if not df.empty:
                    row = df.iloc[0]
                    rec.ic_mean = _safe_float(row.get("ic_mean"))
                    rec.ic_std = _safe_float(row.get("ic_std"))
                    rec.rank_ic_mean = _safe_float(row.get("rank_ic_mean"))
                    rec.rank_ic_std = _safe_float(row.get("rank_ic_std"))
                    rec.icir = _safe_float(row.get("icir"))
                    rec.ic_t_stat = _safe_float(row.get("ic_t_stat"))
                    rec.ic_positive_rate = _safe_float(row.get("ic_positive_rate"))
                    rec.n_days = _safe_int(row.get("n_days"))
            except Exception as e:
                logger.warning("读取 IC summary 失败：%s", e)

        # Quantile returns
        quantile_path = output_dir / "signal_quantile_return.csv"
        if quantile_path.exists():
            try:
                df = pd.read_csv(quantile_path, encoding="utf-8-sig", index_col=0)
                if "long_short" in df.columns:
                    ls = df["long_short"].dropna()
                    if not ls.empty:
                        rec.long_short_daily_mean = float(ls.mean())
                        rec.long_short_daily_std = float(ls.std())
                        rec.long_short_ir = (
                            float(ls.mean() / ls.std()) if ls.std() > 0 else None
                        )
                        rec.long_short_win_rate = float((ls > 0).mean())
            except Exception as e:
                logger.warning("读取 quantile returns 失败：%s", e)

        # Coverage
        coverage_path = output_dir / "signal_coverage.csv"
        if coverage_path.exists():
            try:
                df = pd.read_csv(coverage_path, encoding="utf-8-sig")
                if "coverage" in df.columns:
                    cov = df["coverage"].dropna()
                    if not cov.empty:
                        rec.coverage_mean = float(cov.mean())
                        rec.coverage_min = float(cov.min())
            except Exception as e:
                logger.warning("读取 coverage 失败：%s", e)

        # Autocorr
        autocorr_path = output_dir / "signal_autocorr.csv"
        if autocorr_path.exists():
            try:
                df = pd.read_csv(autocorr_path, encoding="utf-8-sig")
                if not df.empty and "lag" in df.columns and "autocorr" in df.columns:
                    lag1 = df[df["lag"] == 1]
                    if not lag1.empty:
                        rec.autocorr_lag1 = _safe_float(lag1.iloc[0]["autocorr"])
            except Exception as e:
                logger.warning("读取 autocorr 失败：%s", e)

        # 信号质量标签
        rec.signal_quality = _classify_signal_quality(rec.ic_mean, rec.icir)

        return rec

    def apply_to_recorder(self, recorder: Any) -> None:
        """将信号评估指标写入 RunRecorder。"""
        metrics = {}
        if self.ic_mean is not None:
            metrics["signal_ic_mean"] = self.ic_mean
        if self.rank_ic_mean is not None:
            metrics["signal_rank_ic_mean"] = self.rank_ic_mean
        if self.icir is not None:
            metrics["signal_icir"] = self.icir
        if self.long_short_daily_mean is not None:
            metrics["signal_long_short_mean"] = self.long_short_daily_mean
        if self.long_short_ir is not None:
            metrics["signal_long_short_ir"] = self.long_short_ir
        if self.coverage_mean is not None:
            metrics["signal_coverage_mean"] = self.coverage_mean
        if self.signal_quality:
            metrics["signal_quality"] = self.signal_quality
        if metrics:
            recorder.record_metrics(metrics)


# ---------------------------------------------------------------------------
# WalkForwardRecord
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardRecord(BaseRecord):
    """Walk-forward 验证记录：从 walk-forward 输出目录加载样本外表现摘要。"""

    record_type: str = "walk_forward"
    variant: str = ""
    n_stocks: Optional[int] = None
    n_test_years: Optional[int] = None
    test_years: List[int] = field(default_factory=list)

    # 汇总指标（跨所有股票和年份）
    mean_excess_return: Optional[float] = None
    median_excess_return: Optional[float] = None
    win_rate: Optional[float] = None  # excess_return > 0 的股票占比
    mean_sharpe: Optional[float] = None
    mean_max_drawdown: Optional[float] = None

    @classmethod
    def from_dir(
        cls,
        output_dir: Path,
        variant: str = "",
    ) -> "WalkForwardRecord":
        """从 walk-forward 输出目录构建记录。

        读取 <file_prefix>stock_*_summary.csv 文件，提取样本外表现。
        """
        from datetime import datetime

        rec = cls(variant=variant, generated_at=datetime.now().isoformat())
        output_dir = Path(output_dir)

        if not output_dir.exists():
            logger.warning("Walk-forward 输出目录不存在：%s", output_dir)
            return rec

        # 查找所有 summary CSV
        summary_files = sorted(output_dir.glob("*_summary.csv"))
        if not summary_files:
            logger.warning("未找到 walk-forward summary 文件：%s", output_dir)
            return rec

        all_excess = []
        all_sharpe = []
        all_mdd = []
        test_years_set = set()

        for sf in summary_files:
            try:
                df = pd.read_csv(sf, encoding="utf-8-sig")
                if df.empty:
                    continue
                if "test_year" in df.columns:
                    test_years_set.update(df["test_year"].dropna().astype(int).tolist())
                if "excess_return" in df.columns:
                    all_excess.extend(df["excess_return"].dropna().tolist())
                if "sharpe" in df.columns:
                    all_sharpe.extend(df["sharpe"].dropna().tolist())
                if "max_drawdown" in df.columns:
                    all_mdd.extend(df["max_drawdown"].dropna().tolist())
            except Exception as e:
                logger.warning("读取 %s 失败：%s", sf.name, e)

        rec.n_stocks = len(summary_files)
        rec.test_years = sorted(test_years_set)
        rec.n_test_years = len(test_years_set)

        if all_excess:
            arr = np.array(all_excess, dtype=float)
            rec.mean_excess_return = float(np.nanmean(arr))
            rec.median_excess_return = float(np.nanmedian(arr))
            rec.win_rate = float(np.nanmean(arr > 0))

        if all_sharpe:
            rec.mean_sharpe = float(np.nanmean(all_sharpe))

        if all_mdd:
            rec.mean_max_drawdown = float(np.nanmean(all_mdd))

        return rec

    def apply_to_recorder(self, recorder: Any) -> None:
        """将 walk-forward 指标写入 RunRecorder。"""
        metrics = {}
        if self.n_stocks is not None:
            metrics["wf_n_stocks"] = self.n_stocks
        if self.n_test_years is not None:
            metrics["wf_n_test_years"] = self.n_test_years
        if self.mean_excess_return is not None:
            metrics["wf_mean_excess_return"] = self.mean_excess_return
        if self.win_rate is not None:
            metrics["wf_win_rate"] = self.win_rate
        if self.mean_sharpe is not None:
            metrics["wf_mean_sharpe"] = self.mean_sharpe
        if self.mean_max_drawdown is not None:
            metrics["wf_mean_max_drawdown"] = self.mean_max_drawdown
        if metrics:
            recorder.record_metrics(metrics)


# ---------------------------------------------------------------------------
# DiagnosisRecord
# ---------------------------------------------------------------------------


@dataclass
class DiagnosisRecord(BaseRecord):
    """诊断记录：从 diagnosis 输出目录加载参数稳定性/年度一致性摘要。"""

    record_type: str = "diagnosis"
    variant: str = ""

    # 训练-测试 gap
    mean_train_return: Optional[float] = None
    mean_test_return: Optional[float] = None
    mean_gap: Optional[float] = None

    # 年度一致性
    yearly_consistency: Optional[float] = None  # test 正收益年份占比

    # 参数稳定性
    parameter_stability: str = ""  # STABLE / MODERATE / UNSTABLE / UNKNOWN

    # 推荐
    recommendation: str = ""

    @classmethod
    def from_dir(
        cls,
        output_dir: Path,
        variant: str = "",
    ) -> "DiagnosisRecord":
        """从 diagnosis 输出目录构建记录。"""
        from datetime import datetime

        rec = cls(variant=variant, generated_at=datetime.now().isoformat())
        output_dir = Path(output_dir)

        if not output_dir.exists():
            logger.warning("Diagnosis 输出目录不存在：%s", output_dir)
            return rec

        # 读取 diagnosis manifest（如果存在）
        manifest_path = output_dir / "diagnosis_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                rec.recommendation = manifest.get("recommendation", "")
            except Exception as e:
                logger.warning("读取 diagnosis manifest 失败：%s", e)

        # 读取 train-test gap CSV
        gap_files = sorted(output_dir.glob("*train_test_gap*.csv"))
        if gap_files:
            try:
                df = pd.read_csv(gap_files[0], encoding="utf-8-sig")
                if not df.empty:
                    if "train_annual_return" in df.columns:
                        rec.mean_train_return = float(df["train_annual_return"].mean())
                    if "test_total_return" in df.columns:
                        rec.mean_test_return = float(df["test_total_return"].mean())
                    if "gap" in df.columns:
                        rec.mean_gap = float(df["gap"].mean())
                    # 年度一致性：test 正收益年份占比
                    if "test_total_return" in df.columns:
                        valid = df["test_total_return"].dropna()
                        if not valid.empty:
                            rec.yearly_consistency = float((valid > 0).mean())
            except Exception as e:
                logger.warning("读取 train-test gap 失败：%s", e)

        return rec

    def apply_to_recorder(self, recorder: Any) -> None:
        """将诊断指标写入 RunRecorder。"""
        metrics = {}
        if self.mean_train_return is not None:
            metrics["diag_mean_train_return"] = self.mean_train_return
        if self.mean_test_return is not None:
            metrics["diag_mean_test_return"] = self.mean_test_return
        if self.mean_gap is not None:
            metrics["diag_mean_gap"] = self.mean_gap
        if self.yearly_consistency is not None:
            metrics["diag_yearly_consistency"] = self.yearly_consistency
        if self.recommendation:
            metrics["diag_recommendation"] = self.recommendation
        if metrics:
            recorder.record_metrics(metrics)


# ---------------------------------------------------------------------------
# RobustnessRecord
# ---------------------------------------------------------------------------


@dataclass
class RobustnessRecord(BaseRecord):
    """稳健性 gate 记录：从 robustness 输出目录加载 gate 通过/拒绝结果。"""

    record_type: str = "robustness"
    variant: str = ""

    # Gate 结果
    gates_passed: int = 0
    gates_total: int = 0
    gates_failed: List[str] = field(default_factory=list)
    overall_pass: bool = False

    # 关键指标
    loyo_worst_year: Optional[str] = None
    loyo_worst_excess: Optional[float] = None
    benchmark_excess: Optional[float] = None

    @classmethod
    def from_dir(
        cls,
        output_dir: Path,
        variant: str = "",
    ) -> "RobustnessRecord":
        """从 robustness 输出目录构建记录。"""
        from datetime import datetime

        rec = cls(variant=variant, generated_at=datetime.now().isoformat())
        output_dir = Path(output_dir)

        if not output_dir.exists():
            logger.warning("Robustness 输出目录不存在：%s", output_dir)
            return rec

        # 读取 robustness manifest
        manifest_path = output_dir / "robustness_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                gates = manifest.get("gates", {})
                rec.gates_total = len(gates)
                rec.gates_passed = sum(1 for v in gates.values() if v.get("pass", False))
                rec.gates_failed = [k for k, v in gates.items() if not v.get("pass", False)]
                rec.overall_pass = manifest.get("overall_pass", False)
            except Exception as e:
                logger.warning("读取 robustness manifest 失败：%s", e)

        # 读取 LOYO 结果
        loyo_path = output_dir / "robustness_scenarios.csv"
        if loyo_path.exists():
            try:
                df = pd.read_csv(loyo_path, encoding="utf-8-sig")
                if not df.empty and "excess_return" in df.columns:
                    worst_idx = df["excess_return"].idxmin()
                    rec.loyo_worst_year = str(df.loc[worst_idx, "scenario"]) if "scenario" in df.columns else ""
                    rec.loyo_worst_excess = float(df.loc[worst_idx, "excess_return"])
            except Exception as e:
                logger.warning("读取 LOYO 结果失败：%s", e)

        # 读取基准对比
        bench_path = output_dir / "robustness_benchmark_comparison.csv"
        if bench_path.exists():
            try:
                df = pd.read_csv(bench_path, encoding="utf-8-sig")
                if not df.empty and "excess_return" in df.columns:
                    rec.benchmark_excess = float(df["excess_return"].mean())
            except Exception as e:
                logger.warning("读取基准对比失败：%s", e)

        return rec

    def apply_to_recorder(self, recorder: Any) -> None:
        """将稳健性指标写入 RunRecorder。"""
        metrics = {
            "robust_gates_passed": self.gates_passed,
            "robust_gates_total": self.gates_total,
            "robust_overall_pass": self.overall_pass,
        }
        if self.loyo_worst_excess is not None:
            metrics["robust_loyo_worst_excess"] = self.loyo_worst_excess
        if self.benchmark_excess is not None:
            metrics["robust_benchmark_excess"] = self.benchmark_excess
        recorder.record_metrics(metrics)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> Optional[float]:
    """安全转 float，NaN 或无效值返回 None。"""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """安全转 int。"""
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _classify_signal_quality(ic_mean: Optional[float], icir: Optional[float]) -> str:
    """根据 IC 均值和 ICIR 分类信号质量。"""
    if ic_mean is None:
        return "UNKNOWN"
    if abs(ic_mean) > 0.03:
        return "STRONG"
    if abs(ic_mean) > 0.01:
        if icir is not None and abs(icir) > 0.5:
            return "MODERATE+"
        return "MODERATE"
    return "WEAK"
