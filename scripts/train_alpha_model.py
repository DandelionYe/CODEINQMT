# -*- coding: utf-8 -*-
"""
train_alpha_model.py

CLI：使用 AlphaModel 接口对 feature matrix 执行 walk-forward 预测。

将 SimpleRuleModel（或未来 ML 模型）与 QMTDataHandler 集成，
按年度 walk-forward 生成 prediction score 并计算 IC/RankIC。

支持两种模型模式：
  1. score_col 模式：直接使用 feature matrix 中的预计算列（如 feature/reversal_10d）
  2. expression 模式：使用 Alpha v7 的表达式构建（需要原始价格列在 feature matrix 中）

运行示例：
  # score_col 模式
  python scripts/train_alpha_model.py \\
    --feature-matrix factors/processed/feature_matrix/run_xxx/feature_matrix.parquet \\
    --score-col feature/reversal_10d \\
    --label-col label/ret_1d \\
    --test-years 2023,2024

  # expression 模式
  python scripts/train_alpha_model.py \\
    --feature-matrix factors/processed/feature_matrix/run_xxx/feature_matrix.parquet \\
    --alpha-variant short_term_reversal \\
    --label-col label/ret_1d \\
    --test-years 2022,2023,2024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.constants import PROJECT_ROOT as _PR
from scripts.common.data_handler import QMTDataHandler
from scripts.common.logging_setup import setup_cli_logging
from scripts.common.models import AlphaModel, SimpleRuleModel

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = _PR / "backtests" / "model_prediction"


# ---------------------------------------------------------------------------
# IC 计算
# ---------------------------------------------------------------------------

def compute_ic(pred: pd.Series, label: pd.Series) -> float:
    """计算 Pearson IC。"""
    valid = pd.DataFrame({"pred": pred, "label": label}).dropna()
    if len(valid) < 5:
        return float("nan")
    return float(valid["pred"].corr(valid["label"]))


def compute_rank_ic(pred: pd.Series, label: pd.Series) -> float:
    """计算 Spearman Rank IC（不依赖 scipy，使用 rank + Pearson）。"""
    valid = pd.DataFrame({"pred": pred, "label": label}).dropna()
    if len(valid) < 5:
        return float("nan")
    # Spearman = Pearson(rank(x), rank(y))
    rp = valid["pred"].rank()
    rl = valid["label"].rank()
    return float(rp.corr(rl))


def compute_daily_ic_series(
    pred: pd.Series,
    label: pd.Series,
    dates: pd.Index,
) -> pd.Series:
    """按日期计算每日 IC。"""
    df = pd.DataFrame({"pred": pred.values, "label": label.values}, index=dates)
    df.index.name = "_date_key_"
    daily_ics = []
    for dt, group in df.groupby(level=0):
        if len(group) < 3:
            continue
        ic = group["pred"].corr(group["label"])
        daily_ics.append({"_date_key_": dt, "ic": ic})
    if not daily_ics:
        return pd.Series(dtype=float)
    result = pd.DataFrame(daily_ics).set_index("_date_key_")["ic"]
    result.index.name = None
    return result


# ---------------------------------------------------------------------------
# Walk-Forward 预测
# ---------------------------------------------------------------------------

def run_walk_forward(
    handler: QMTDataHandler,
    model: AlphaModel,
    test_years: List[int],
    label_col: str,
    zscore_pred: bool = False,
) -> Dict[str, Any]:
    """执行 walk-forward 预测。

    Parameters
    ----------
    handler : QMTDataHandler
        数据处理器。
    model : AlphaModel
        模型实例（SimpleRuleModel 或 LightGBMModel 等）。
    test_years : list of int
        测试年份列表。
    label_col : str
        标签列名。
    zscore_pred : bool
        是否对 prediction 做 ZScore 标准化。

    Returns
    -------
    dict
        包含 per_year_results、ic_summary、predictions。
    """
    from scripts.common.feature_expression import normalize_zscore

    per_year_results = []
    all_predictions = []

    for test_year in test_years:
        logger.info("Walk-forward: test_year=%d", test_year)

        # 准备数据
        train_dk_l, test_dk_i, train_features, test_features = \
            handler.prepare_walk_forward(test_year)

        if train_dk_l.empty:
            logger.warning("训练期数据为空（test_year=%d），跳过", test_year)
            per_year_results.append({
                "test_year": test_year,
                "n_train": 0,
                "n_test": 0,
                "ic": float("nan"),
                "rank_ic": float("nan"),
                "icir": float("nan"),
                "ic_win_rate": float("nan"),
                "status": "skipped_empty_train",
            })
            continue

        if test_dk_i.empty:
            logger.warning("测试期数据为空（test_year=%d），跳过", test_year)
            per_year_results.append({
                "test_year": test_year,
                "n_train": len(train_dk_l),
                "n_test": 0,
                "ic": float("nan"),
                "rank_ic": float("nan"),
                "icir": float("nan"),
                "ic_win_rate": float("nan"),
                "status": "skipped_empty_test",
            })
            continue

        # fit on train
        model.fit(train_dk_l, label_col=label_col)

        # predict on test
        pred = model.predict(test_dk_i)
        if zscore_pred:
            pred = normalize_zscore(pred)

        # 获取测试期 label
        if label_col in test_dk_i.columns:
            label = test_dk_i[label_col]
        elif label_col in handler.label_cols():
            # label 可能在原始数据中但不在 DK_I 中（DK_I 不含 label）
            # 需要从原始数据中获取
            raw = handler.load_raw()
            test_mask = handler.get_year_mask(test_year)
            label = raw.loc[test_mask, label_col]
        else:
            logger.warning("标签列 '%s' 不在数据中，跳过 IC 计算", label_col)
            label = pd.Series(np.nan, index=test_dk_i.index)

        # 计算 IC
        ic = compute_ic(pred, label)
        rank_ic = compute_rank_ic(pred, label)

        # 计算每日 IC 系列（用于 ICIR 和 win rate）
        dates = test_dk_i.index.get_level_values(0)
        daily_ics = compute_daily_ic_series(pred, label, dates)
        if len(daily_ics) > 0:
            ic_mean = daily_ics.mean()
            ic_std = daily_ics.std()
            icir = ic_mean / ic_std if ic_std > 0 else float("nan")
            ic_win_rate = (daily_ics > 0).mean()
        else:
            icir = float("nan")
            ic_win_rate = float("nan")

        # 保存预测
        pred_df = pd.DataFrame({
            "date": test_dk_i.index.get_level_values(0),
            "symbol": test_dk_i.index.get_level_values(1),
            "prediction": pred.values,
            "label": label.values,
        })
        pred_df["test_year"] = test_year
        all_predictions.append(pred_df)

        per_year_results.append({
            "test_year": test_year,
            "n_train": len(train_dk_l),
            "n_test": len(test_dk_i),
            "ic": ic,
            "rank_ic": rank_ic,
            "icir": icir,
            "ic_win_rate": ic_win_rate,
            "status": "ok",
        })

        logger.info(
            "  test_year=%d: n_train=%d, n_test=%d, IC=%.4f, RankIC=%.4f, ICIR=%.4f",
            test_year, len(train_dk_l), len(test_dk_i), ic, rank_ic, icir,
        )

    # 汇总
    valid_results = [r for r in per_year_results if r["status"] == "ok"]
    if valid_results:

        def _safe_nanmean(values):
            """对全 NaN 列表返回 NaN，避免 RuntimeWarning。"""
            arr = np.array(values, dtype=float)
            if np.all(np.isnan(arr)):
                return float("nan")
            return float(np.nanmean(arr))

        ic_summary = {
            "mean_ic": _safe_nanmean([r["ic"] for r in valid_results]),
            "mean_rank_ic": _safe_nanmean([r["rank_ic"] for r in valid_results]),
            "mean_icir": _safe_nanmean([r["icir"] for r in valid_results]),
            "mean_ic_win_rate": _safe_nanmean([r["ic_win_rate"] for r in valid_results]),
            "n_years": len(valid_results),
        }
    else:
        ic_summary = {
            "mean_ic": float("nan"),
            "mean_rank_ic": float("nan"),
            "mean_icir": float("nan"),
            "mean_ic_win_rate": float("nan"),
            "n_years": 0,
        }

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    return {
        "per_year_results": per_year_results,
        "ic_summary": ic_summary,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 AlphaModel 接口对 feature matrix 执行 walk-forward 预测。",
    )

    # 数据
    parser.add_argument(
        "--feature-matrix",
        type=str,
        required=True,
        help="Feature matrix parquet 文件路径",
    )

    # 模型配置
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--score-col",
        type=str,
        default=None,
        help="预计算的 score 列名（如 feature/reversal_10d）",
    )
    model_group.add_argument(
        "--alpha-variant",
        type=str,
        default=None,
        help="Alpha variant 名称（如 short_term_reversal），使用表达式模式",
    )

    # 表达式模式参数
    parser.add_argument("--reversal-window", type=int, default=10)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--turnover-short", type=int, default=10)
    parser.add_argument("--turnover-long", type=int, default=60)
    parser.add_argument("--divergence-window", type=int, default=20)

    # 标签和评估
    parser.add_argument(
        "--label-col",
        type=str,
        default="label/ret_1d",
        help="标签列名，默认 label/ret_1d",
    )
    parser.add_argument(
        "--test-years",
        type=str,
        default=None,
        help="测试年份，逗号分隔，如 2022,2023,2024。为空时自动检测。",
    )

    # 模型选项
    parser.add_argument(
        "--no-zscore",
        action="store_true",
        help="不对 expression 输出做 ZScore 标准化",
    )
    parser.add_argument(
        "--zscore-pred",
        action="store_true",
        help="对 prediction 做 ZScore 标准化",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.0,
        help="信号阈值，默认 0.0",
    )

    # 输出
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认 backtests/model_prediction/<timestamp>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划，不实际执行",
    )

    return parser.parse_args()


def _auto_detect_test_years(handler: QMTDataHandler) -> List[int]:
    """自动检测可用的测试年份。"""
    raw = handler.load_raw()
    dates = raw.index.get_level_values(0)
    years = sorted(set(dates // 10000))
    # 至少需要 1 年训练数据，所以从第 2 年开始
    if len(years) < 2:
        return years
    return years[1:]


def main() -> None:
    args = parse_args()
    setup_cli_logging()

    # 加载 feature matrix
    fm_path = Path(args.feature_matrix)
    if not fm_path.exists():
        logger.error("Feature matrix 文件不存在：%s", fm_path)
        sys.exit(1)

    logger.info("加载 feature matrix：%s", fm_path)
    handler = QMTDataHandler(fm_path)

    # 检测测试年份
    if args.test_years:
        test_years = [int(y.strip()) for y in args.test_years.split(",")]
    else:
        test_years = _auto_detect_test_years(handler)
        logger.info("自动检测测试年份：%s", test_years)

    if not test_years:
        logger.error("无可用测试年份")
        sys.exit(1)

    # 构建模型
    zscore = not args.no_zscore
    if args.score_col:
        model = SimpleRuleModel(
            score_col=args.score_col,
            zscore=zscore,
            signal_threshold=args.signal_threshold,
        )
        model_desc = f"score_col={args.score_col}"
    else:
        # expression 模式：需要从 alpha_v7 导入 build_expression
        from strategies.alpha_v7_research_strategy_csv import build_expression

        raw_score_expr, _ = build_expression(
            alpha_variant=args.alpha_variant,
            reversal_window=args.reversal_window,
            vol_window=args.vol_window,
            turnover_short=args.turnover_short,
            turnover_long=args.turnover_long,
            divergence_window=args.divergence_window,
        )
        model = SimpleRuleModel(
            expression=raw_score_expr,
            zscore=zscore,
            signal_threshold=args.signal_threshold,
        )
        model_desc = f"expression={args.alpha_variant}"

    # 输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_DIR / f"run_{ts}"

    # Dry-run
    if args.dry_run:
        print("=" * 60)
        print("Walk-Forward 模型预测计划（dry-run）")
        print("=" * 60)
        print(f"  Feature matrix：{fm_path}")
        print(f"  模型：{model_desc}")
        print(f"  标签列：{args.label_col}")
        print(f"  测试年份：{test_years}")
        print(f"  ZScore：{zscore}")
        print(f"  输出目录：{output_dir}")
        print("=" * 60)
        return

    # 执行 walk-forward
    logger.info("开始 walk-forward 预测：model=%s, test_years=%s", model_desc, test_years)
    results = run_walk_forward(
        handler=handler,
        model=model,
        test_years=test_years,
        label_col=args.label_col,
        zscore_pred=args.zscore_pred,
    )

    # 保存输出
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存预测
    if not results["predictions"].empty:
        pred_path = output_dir / "predictions.parquet"
        results["predictions"].to_parquet(pred_path, index=False)
        logger.info("预测结果已保存：%s", pred_path)

    # 保存 IC 摘要
    summary = {
        "model": model.get_params(),
        "label_col": args.label_col,
        "test_years": test_years,
        "ic_summary": results["ic_summary"],
        "per_year_results": results["per_year_results"],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    summary_path = output_dir / "model_prediction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("摘要已保存：%s", summary_path)

    # 保存每日 IC 系列
    if not results["predictions"].empty:
        dates = results["predictions"]["date"]
        pred = results["predictions"]["prediction"]
        label = results["predictions"]["label"]
        daily_ics = compute_daily_ic_series(pred, label, dates)
        if not daily_ics.empty:
            ic_path = output_dir / "daily_ic.csv"
            daily_ics.to_csv(ic_path, header=["ic"])
            logger.info("每日 IC 已保存：%s", ic_path)

    # 打印摘要
    ic_sum = results["ic_summary"]
    print(f"\nWalk-Forward 模型预测完成：")
    print(f"  模型：{model_desc}")
    print(f"  标签：{args.label_col}")
    print(f"  测试年份：{test_years}")
    print(f"  年份数：{ic_sum['n_years']}")
    print(f"  平均 IC：{ic_sum['mean_ic']:.4f}")
    print(f"  平均 RankIC：{ic_sum['mean_rank_ic']:.4f}")
    print(f"  平均 ICIR：{ic_sum['mean_icir']:.4f}")
    print(f"  平均 IC 胜率：{ic_sum['mean_ic_win_rate']:.2%}")
    print(f"  输出目录：{output_dir}")

    # 逐年详情
    print(f"\n  逐年结果：")
    for r in results["per_year_results"]:
        status = r["status"]
        if status == "ok":
            print(
                f"    {r['test_year']}: "
                f"IC={r['ic']:.4f}, RankIC={r['rank_ic']:.4f}, "
                f"ICIR={r['icir']:.4f}, n={r['n_test']}"
            )
        else:
            print(f"    {r['test_year']}: {status}")


if __name__ == "__main__":
    main()
