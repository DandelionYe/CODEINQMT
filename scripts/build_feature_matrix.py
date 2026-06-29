# -*- coding: utf-8 -*-
"""
build_feature_matrix.py

CLI：构建 Feature Matrix。

读取股票池数据，使用 factor_registry.json 中登记的因子表达式计算特征，
并计算前瞻收益标签，输出 (date, symbol) MultiIndex 的 parquet 文件。

运行示例：
  python scripts/build_feature_matrix.py --symbols 000001.SZ,000002.SZ,600000.SH --start 20200101
  python scripts/build_feature_matrix.py --universe-file configs/universe.txt --variant-ids short_term_reversal,low_volatility
  python scripts/build_feature_matrix.py --symbols 000001.SZ --start 20200101 --end 20251231 --output-dir factors/processed/feature_matrix/test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.constants import PROJECT_ROOT as _PR, DEFAULT_PARQUET_ROOT
from scripts.common.feature_matrix import (
    build_feature_matrix_from_registry,
    save_feature_matrix,
)
from scripts.common.logging_setup import setup_cli_logging

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = _PR / "configs" / "factor_registry.json"
DEFAULT_OUTPUT_DIR = _PR / "factors" / "processed" / "feature_matrix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 Feature Matrix：从 parquet 数据 + factor_registry 生成统一特征矩阵。",
    )

    # 股票池
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="逗号分隔的股票代码列表，如 000001.SZ,000002.SZ",
    )
    parser.add_argument(
        "--universe-file",
        type=str,
        default=None,
        help="股票池文件路径（每行一个股票代码）",
    )

    # 日期范围
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYYMMDD")

    # 因子
    parser.add_argument(
        "--registry-path",
        type=str,
        default=str(DEFAULT_REGISTRY_PATH),
        help="factor_registry.json 路径",
    )
    parser.add_argument(
        "--variant-ids",
        type=str,
        default=None,
        help="逗号分隔的 variant ID 列表，如 short_term_reversal,low_volatility。为空时使用全部。",
    )
    parser.add_argument(
        "--label-horizons",
        type=str,
        default="1,5,20",
        help="前瞻收益期数，逗号分隔，默认 1,5,20",
    )

    # 参数覆盖
    parser.add_argument(
        "--reversal-window", type=int, default=None, help="覆盖 reversal_window 参数"
    )
    parser.add_argument(
        "--vol-window", type=int, default=None, help="覆盖 vol_window 参数"
    )
    parser.add_argument(
        "--turnover-short", type=int, default=None, help="覆盖 turnover_short 参数"
    )
    parser.add_argument(
        "--turnover-long", type=int, default=None, help="覆盖 turnover_long 参数"
    )
    parser.add_argument(
        "--divergence-window", type=int, default=None, help="覆盖 divergence_window 参数"
    )

    # 输出
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认 factors/processed/feature_matrix/<timestamp>",
    )
    parser.add_argument(
        "--parquet-root",
        type=str,
        default=str(DEFAULT_PARQUET_ROOT),
        help="parquet 数据根目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划，不实际执行",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_cli_logging()

    # 解析股票池
    symbols = _resolve_symbols(args)
    if not symbols:
        logger.error("未指定股票池。使用 --symbols 或 --universe-file。")
        sys.exit(1)

    # 解析 variant IDs
    variant_ids = None
    if args.variant_ids:
        variant_ids = [v.strip() for v in args.variant_ids.split(",") if v.strip()]

    # 解析 label horizons
    label_horizons = [int(h) for h in args.label_horizons.split(",")]

    # 参数覆盖
    params_override = {}
    if args.reversal_window is not None:
        params_override["reversal_window"] = args.reversal_window
    if args.vol_window is not None:
        params_override["vol_window"] = args.vol_window
    if args.turnover_short is not None:
        params_override["turnover_short"] = args.turnover_short
    if args.turnover_long is not None:
        params_override["turnover_long"] = args.turnover_long
    if args.divergence_window is not None:
        params_override["divergence_window"] = args.divergence_window

    # 输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_DIR / f"run_{ts}"

    # Dry-run
    if args.dry_run:
        print("=" * 60)
        print("Feature Matrix 构建计划（dry-run）")
        print("=" * 60)
        print(f"  股票数量：{len(symbols)}")
        print(f"  日期范围：{args.start or '全部'} ~ {args.end or '全部'}")
        print(f"  Variant IDs：{variant_ids or '全部'}")
        print(f"  Label horizons：{label_horizons}")
        print(f"  参数覆盖：{params_override or '无'}")
        print(f"  输出目录：{output_dir}")
        print("=" * 60)
        return

    # 构建
    registry_path = Path(args.registry_path)
    parquet_root = Path(args.parquet_root)

    logger.info("开始构建 Feature Matrix...")
    feature_matrix, manifest = build_feature_matrix_from_registry(
        symbols=symbols,
        registry_path=registry_path,
        variant_ids=variant_ids,
        params_override=params_override or None,
        parquet_root=parquet_root,
        start=args.start,
        end=args.end,
        label_horizons=label_horizons,
    )

    # 保存
    save_feature_matrix(feature_matrix, manifest, output_dir)

    print(f"\nFeature Matrix 构建完成：")
    print(f"  Shape: {feature_matrix.shape}")
    print(f"  列：{list(feature_matrix.columns)}")
    print(f"  日期范围：{manifest['date_range']}")
    print(f"  输出目录：{output_dir}")


def _resolve_symbols(args: argparse.Namespace) -> list:
    """从 --symbols 或 --universe-file 解析股票列表。"""
    if args.symbols:
        return [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.universe_file:
        p = Path(args.universe_file)
        if not p.exists():
            logger.error("universe 文件不存在：%s", p)
            return []
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        return [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    return []


if __name__ == "__main__":
    main()
