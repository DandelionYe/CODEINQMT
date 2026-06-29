# -*- coding: utf-8 -*-
"""
evaluate_alpha_signals.py

SignalEvaluationRecord CLI：批量评估 Alpha 信号的截面排序能力。

在进入 batch_backtest / walk_forward 之前，先评估信号本身是否有截面排序能力。
输出 IC/RankIC/ICIR、分位数多空收益、覆盖率、信号自相关等。

运行示例：

  # 评估 short_term_reversal 在沪深 300 成分股上的信号质量
  python scripts/evaluate_alpha_signals.py \\
      --alpha-variant short_term_reversal \\
      --reversal-window 10 \\
      --universe-file data/universe_hs300.txt \\
      --start 20150101

  # 评估所有 4 个 variant
  python scripts/evaluate_alpha_signals.py \\
      --alpha-variant all \\
      --start 20150101 --end 20251231

  # 指定实验 ID（输出到 backtests/signal_evaluation/<experiment_id>/）
  python scripts/evaluate_alpha_signals.py \\
      --experiment-id exp_006_alpha_v6_non_momentum_signals \\
      --alpha-variant short_term_reversal \\
      --reversal-window 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.constants import (  # noqa: E402
    DEFAULT_BENCHMARK,
    SQRT_TRADING_DAYS_PER_YEAR,
)
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.signal_evaluation import (  # noqa: E402
    evaluate_signal,
    save_signal_evaluation,
)
from strategies.ma_demo_strategy_csv import (  # noqa: E402
    DEFAULT_EXPORT_ROOT,
    find_csv_for_stock,
    load_qmt_price_csv,
    scan_qmt_export,
)
from strategies.alpha_v7_research_strategy_csv import build_expression as build_v7_expression  # noqa: E402

logger = logging.getLogger(__name__)

VALID_VARIANTS = [
    "short_term_reversal",
    "low_volatility",
    "turnover_reversal",
    "volume_price_divergence",
]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backtests" / "signal_evaluation"


# ---------------------------------------------------------------------------
# Alpha 信号计算（独立版本，不依赖回测逻辑）
# ---------------------------------------------------------------------------

def compute_alpha_signals(
    stock_df: pd.DataFrame,
    alpha_variant: str,
    reversal_window: int = 10,
    vol_window: int = 60,
    turnover_short: int = 10,
    turnover_long: int = 60,
    divergence_window: int = 20,
) -> pd.DataFrame:
    """计算 alpha 信号，委托给表达式层 build_expression()。

    与 alpha_v7_research_strategy_csv.build_expression() 共享同一表达式逻辑，
    避免硬编码副本导致的漂移风险。
    """
    result = stock_df[["date", "close", "volume"]].copy()

    raw_score_expr, signal_expr = build_v7_expression(
        alpha_variant,
        reversal_window=reversal_window,
        vol_window=vol_window,
        turnover_short=turnover_short,
        turnover_long=turnover_long,
        divergence_window=divergence_window,
    )

    result["raw_alpha_score"] = raw_score_expr.eval(stock_df)
    result["alpha_signal"] = signal_expr.eval(stock_df).astype(int)

    # 标准化评分
    std = result["raw_alpha_score"].std()
    if std == 0 or np.isnan(std):
        result["alpha_score"] = result["raw_alpha_score"] * 0.0
    else:
        result["alpha_score"] = (
            result["raw_alpha_score"] - result["raw_alpha_score"].mean()
        ) / std

    return result


# ---------------------------------------------------------------------------
# 前瞻收益
# ---------------------------------------------------------------------------

def compute_forward_returns(
    stock_df: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 20),
) -> pd.DataFrame:
    """计算前瞻收益列 ret_1d, ret_5d, ret_20d。

    ret_Nd = close(t+N) / close(t) - 1
    """
    result = stock_df[["date", "close"]].copy()
    for h in horizons:
        result[f"ret_{h}d"] = result["close"].shift(-h) / result["close"] - 1.0
    return result


# ---------------------------------------------------------------------------
# Universe 加载
# ---------------------------------------------------------------------------

def load_universe(
    universe_file: Optional[Path],
    export_root: Path,
    stock_list: Optional[str],
) -> List[str]:
    """从文件或逗号分隔列表加载股票代码列表。

    支持格式：
    - 每行一个代码的文本文件
    - 逗号分隔的字符串
    - 自动过滤非股票（指数等）
    """
    symbols: List[str] = []

    if universe_file and universe_file.exists():
        text = universe_file.read_text(encoding="utf-8-sig").strip()
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line)
    elif stock_list:
        symbols = [s.strip() for s in stock_list.split(",") if s.strip()]

    if not symbols:
        # 扫描所有可用股票
        catalog_path = PROJECT_ROOT / "data" / "qmt_export_catalog.csv"
        if catalog_path.exists():
            catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
            stocks = catalog[catalog["security_type"] == "stock"]
            symbols = stocks["symbol"].tolist()
        else:
            # 直接扫描目录
            logger.info("未找到 catalog，扫描 qmt_export 目录...")
            try:
                catalog = scan_qmt_export(export_root)
                stocks = catalog[catalog["security_type"] == "stock"]
                symbols = stocks["symbol"].tolist()
            except FileNotFoundError:
                logger.warning("导出目录不存在: %s，返回空 universe", export_root)
                symbols = []

    logger.info("Universe 股票数: %d", len(symbols))
    return symbols


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def evaluate_variant(
    symbols: List[str],
    alpha_variant: str,
    export_root: Path,
    start: str,
    end: str,
    label_horizons: Sequence[int],
    n_quantiles: int,
    reversal_window: int,
    vol_window: int,
    turnover_short: int,
    turnover_long: int,
    divergence_window: int,
) -> Dict[str, Any]:
    """对单个 alpha variant 进行信号评估。

    1. 加载每只股票的价格数据
    2. 计算 alpha 信号和前瞻收益
    3. 合并为截面 DataFrame
    4. 运行信号评估
    """
    all_frames: List[pd.DataFrame] = []
    loaded = 0
    failed = 0

    for symbol in symbols:
        try:
            csv_path, sym, _, _ = find_csv_for_stock(symbol, export_root)
            stock_df = load_qmt_price_csv(csv_path, start, end)

            if len(stock_df) < max(reversal_window, vol_window, turnover_long, divergence_window) + 10:
                continue

            # 计算 alpha 信号
            signals = compute_alpha_signals(
                stock_df, alpha_variant,
                reversal_window, vol_window,
                turnover_short, turnover_long,
                divergence_window,
            )

            # 计算前瞻收益
            fwd = compute_forward_returns(stock_df, label_horizons)

            # 合并
            merged = signals[["date", "raw_alpha_score", "alpha_score", "alpha_signal"]].copy()
            for h in label_horizons:
                merged[f"ret_{h}d"] = fwd[f"ret_{h}d"].values
            merged["symbol"] = sym

            all_frames.append(merged)
            loaded += 1

        except Exception as e:
            failed += 1
            logger.debug("跳过 %s: %s", symbol, e)

    logger.info("加载成功: %d, 失败/跳过: %d", loaded, failed)

    if not all_frames:
        raise RuntimeError("没有成功加载任何股票数据。")

    # 合并所有股票
    cross = pd.concat(all_frames, ignore_index=True)
    cross["date"] = pd.to_datetime(cross["date"])
    cross = cross.set_index(["date", "symbol"]).sort_index()

    # 对每个 horizon 运行评估
    results: Dict[str, Any] = {}
    for h in label_horizons:
        label_col = f"ret_{h}d"
        logger.info("评估 %s / %s ...", alpha_variant, label_col)
        eval_result = evaluate_signal(
            cross, label_col,
            score_col="alpha_score",
            n_quantiles=n_quantiles,
        )
        results[label_col] = eval_result

    return results


def print_summary(all_results: Dict[str, Dict[str, Any]], alpha_variant: str) -> None:
    """打印信号评估摘要到终端。"""
    print("\n" + "=" * 80)
    print(f"Signal Evaluation Summary: {alpha_variant}")
    print("=" * 80)

    for label_col, result in all_results.items():
        summary = result["ic_summary"]
        if summary.empty:
            print(f"\n  {label_col}: 无有效数据")
            continue

        row = summary.iloc[0]
        print(f"\n  {label_col}:")
        print(f"    IC mean:        {row.get('ic_mean', np.nan):.6f}")
        print(f"    IC std:         {row.get('ic_std', np.nan):.6f}")
        print(f"    ICIR:           {row.get('icir', np.nan):.4f}")
        print(f"    IC t-stat:      {row.get('ic_tstat', np.nan):.4f}")
        print(f"    IC positive %:  {row.get('ic_positive_rate', np.nan):.1%}")
        print(f"    RankIC mean:    {row.get('rank_ic_mean', np.nan):.6f}")
        print(f"    RankICIR:       {row.get('rank_icir', np.nan):.4f}")

        quantile = result["quantile_returns"]
        if not quantile.empty and "long_short" in quantile.columns:
            ls = quantile["long_short"].dropna()
            if not ls.empty:
                print(f"    Long-short IR:  {ls.mean() / ls.std():.4f}" if ls.std() > 0 else "    Long-short IR:  N/A")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignalEvaluationRecord：批量评估 Alpha 信号的截面排序能力",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 信号参数
    parser.add_argument("--alpha-variant", default="short_term_reversal",
                        choices=VALID_VARIANTS + ["all"],
                        help="Alpha variant 或 'all' 评估所有 4 个")
    parser.add_argument("--reversal-window", type=int, default=10)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--turnover-short", type=int, default=10)
    parser.add_argument("--turnover-long", type=int, default=60)
    parser.add_argument("--divergence-window", type=int, default=20)

    # 数据参数
    parser.add_argument("--universe-file", type=str, default="",
                        help="股票列表文件（每行一个代码），为空则使用全市场")
    parser.add_argument("--stock-list", type=str, default="",
                        help="逗号分隔的股票列表，如 000001.SZ,600000.SH")
    parser.add_argument("--start", default="20150101", help="开始日期")
    parser.add_argument("--end", default="", help="结束日期")
    parser.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT))

    # 评估参数
    parser.add_argument("--label-horizons", default="1,5,20",
                        help="前瞻收益期限（逗号分隔天数）")
    parser.add_argument("--n-quantiles", type=int, default=5, help="分位数数量")

    # 输出参数
    parser.add_argument("--experiment-id", default="",
                        help="实验 ID（用于输出目录命名）")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="输出目录")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root

    # 加载 universe
    universe_file = Path(args.universe_file) if args.universe_file else None
    symbols = load_universe(universe_file, export_root, args.stock_list)

    if not symbols:
        logger.error("没有可用的股票。请指定 --universe-file 或 --stock-list。")
        sys.exit(1)

    label_horizons = [int(h.strip()) for h in args.label_horizons.split(",")]

    # 确定要评估的 variants
    variants = VALID_VARIANTS if args.alpha_variant == "all" else [args.alpha_variant]

    # 输出目录
    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = PROJECT_ROOT / output_base

    if args.experiment_id:
        output_base = output_base / args.experiment_id

    # 运行评估
    for variant in variants:
        t0 = time.time()
        logger.info("开始评估 %s ...", variant)

        results = evaluate_variant(
            symbols=symbols,
            alpha_variant=variant,
            export_root=export_root,
            start=args.start,
            end=args.end,
            label_horizons=label_horizons,
            n_quantiles=args.n_quantiles,
            reversal_window=args.reversal_window,
            vol_window=args.vol_window,
            turnover_short=args.turnover_short,
            turnover_long=args.turnover_long,
            divergence_window=args.divergence_window,
        )

        elapsed = time.time() - t0
        logger.info("%s 评估完成，耗时 %.1f 秒", variant, elapsed)

        # 保存每个 horizon 的结果
        for label_col, eval_result in results.items():
            run_info = {
                "experiment_id": args.experiment_id or "standalone",
                "alpha_variant": variant,
                "params": {
                    "reversal_window": args.reversal_window,
                    "vol_window": args.vol_window,
                    "turnover_short": args.turnover_short,
                    "turnover_long": args.turnover_long,
                    "divergence_window": args.divergence_window,
                },
                "universe_size": len(symbols),
                "date_range": f"{args.start}_{args.end or 'latest'}",
                "elapsed_seconds": round(elapsed, 1),
            }

            output_dir = output_base / variant / label_col
            save_signal_evaluation(eval_result, output_dir, label_col, run_info)

        # 终端摘要
        print_summary(results, variant)


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception:
        logger.error("程序异常：", exc_info=True)
        sys.exit(1)
