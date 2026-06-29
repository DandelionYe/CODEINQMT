# -*- coding: utf-8 -*-
"""
convert_qmt_csv_to_parquet.py

用途：
将 QMT 导出的 CSV 文件批量转换为 Parquet 格式，加速后续回测的数据读取。

运行示例：
python scripts\\convert_qmt_csv_to_parquet.py
python scripts\\convert_qmt_csv_to_parquet.py --force
python scripts\\convert_qmt_csv_to_parquet.py --limit 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402

logger = logging.getLogger(__name__)


def convert_one_csv(csv_path: Path, parquet_path: Path) -> bool:
    """读取 CSV，列名 lowercase，写入 Parquet。不做日期转换。"""
    raw = ma.read_csv_auto_encoding(csv_path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(parquet_path, engine="pyarrow", index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="将 QMT CSV 批量转换为 Parquet")
    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT), help="QMT CSV 导出根目录")
    parser.add_argument("--parquet-root", default=str(ma.DEFAULT_EXPORT_ROOT.parent / "qmt_parquet"), help="Parquet 输出根目录")
    parser.add_argument("--force", action="store_true", help="强制重新转换已存在的文件")
    parser.add_argument("--limit", type=int, default=0, help="只转换前 N 个文件；0 表示不限制")
    args = parser.parse_args()

    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root

    parquet_root = Path(args.parquet_root)
    if not parquet_root.is_absolute():
        parquet_root = PROJECT_ROOT / parquet_root

    if not export_root.exists():
        logger.error("导出目录不存在：%s", export_root)
        return

    csv_files = []
    for market in ["SH", "SZ"]:
        market_dir = export_root / market
        if market_dir.exists():
            csv_files.extend(sorted(market_dir.glob("price_*.csv")))

    if not csv_files:
        logger.warning("没有找到 CSV 文件。")
        return

    if args.limit > 0:
        csv_files = csv_files[:args.limit]

    total = len(csv_files)
    converted = 0
    skipped = 0
    errors = 0

    logger.info("找到 %d 个 CSV 文件，开始转换...", total)
    logger.info("输出目录：%s", parquet_root)

    start_time = time.time()

    for i, csv_path in enumerate(csv_files, 1):
        market = csv_path.parent.name
        parquet_path = parquet_root / market / csv_path.with_suffix(".parquet").name

        if parquet_path.exists() and not args.force:
            skipped += 1
            if i % 500 == 0:
                logger.info("进度：%d/%d（已转换 %d，跳过 %d，错误 %d）", i, total, converted, skipped, errors)
            continue

        try:
            convert_one_csv(csv_path, parquet_path)
            converted += 1
        except Exception as exc:
            errors += 1
            logger.error("%s: %r", csv_path.name, exc)

        if i % 500 == 0:
            logger.info("进度：%d/%d（已转换 %d，跳过 %d，错误 %d）", i, total, converted, skipped, errors)

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("转换完成。")
    print(f"总文件数：{total}")
    print(f"已转换：{converted}")
    print(f"已跳过：{skipped}")
    print(f"错误：{errors}")
    print(f"耗时：{elapsed:.1f} 秒")
    print("=" * 60)


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception:
        logger.error("程序异常：", exc_info=True)
