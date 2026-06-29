# -*- coding: utf-8 -*-
"""
filter_ma_results.py

用途：
筛选 batch_ma_backtest_csv.py 生成的批量均线回测结果。

主要剔除：
1. 短样本：rows 太少
2. 交易次数太少：trade_count 太低
3. 回撤过大：max_drawdown 太低
4. 跑输买入持有：excess_total_return <= 0
5. 夏普过低、年化过低的结果

运行示例：

1. 自动读取 backtests\\batch_ma_csv 下所有 batch_ma_summary_*.csv，并按默认标准筛选：
python scripts\\filter_ma_results.py

2. 只筛选深市短样本结果：
python scripts\\filter_ma_results.py --input "backtests\\batch_ma_csv\\batch_ma_summary_stock_SZ_20150101_latest_f5-10-20_s60-120-250_short_warmup10.csv"

3. 同时筛选沪深两个结果：
python scripts\\filter_ma_results.py --input "backtests\\batch_ma_csv\\batch_ma_summary_stock_SZ_20150101_latest_f5-10-20_s60-120-250_short_warmup10.csv" --input "backtests\\batch_ma_csv\\batch_ma_summary_stock_SH_20150101_latest_f5-10-20_s60-120-250_short_warmup10.csv"

4. 放宽筛选条件：
python scripts\\filter_ma_results.py --min-rows 800 --min-trade-count 3 --max-drawdown-limit -0.60 --min-sharpe 0.3 --min-annual-return 0.03

5. 稳健筛选，适合最终候选：
python scripts\\filter_ma_results.py --min-rows 1500 --min-trade-count 8 --max-drawdown-limit -0.45 --min-sharpe 0.6 --min-annual-return 0.08

6. 不要求跑赢买入持有：
python scripts\\filter_ma_results.py --allow-negative-excess
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.validation import resolve_path  # noqa: E402

DEFAULT_INPUT_GLOB = PROJECT_ROOT / "backtests" / "batch_ma_csv" / "batch_ma_summary_*.csv"
OUTPUT_DIR = PROJECT_ROOT / "backtests" / "filtered_ma_csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


REQUIRED_COLUMNS = {
    "symbol",
    "market",
    "security_type",
    "fast",
    "slow",
    "rows",
    "annual_return",
    "max_drawdown",
    "sharpe",
    "total_return",
    "buy_hold_total_return",
    "excess_total_return",
    "trade_count",
    "score",
}



def read_inputs(inputs: list[str], input_glob: str) -> tuple[pd.DataFrame, list[Path]]:
    files: list[Path] = []

    if inputs:
        for item in inputs:
            path = resolve_path(item)
            if not path.exists():
                raise FileNotFoundError(f"输入文件不存在：{path}")
            files.append(path)
    else:
        glob_path = resolve_path(input_glob)
        files = sorted(glob_path.parent.glob(glob_path.name))

    # 避免把 top50、errors、skipped 文件误读进来
    files = [
        f for f in files
        if f.name.startswith("batch_ma_summary_") and f.suffix.lower() == ".csv"
    ]

    if not files:
        raise RuntimeError("没有找到可读取的 batch_ma_summary_*.csv 文件。")

    frames = []

    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = str(file)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.warning("[跳过] %s 缺少字段：%s", file, missing)
            continue

        frames.append(df)

    if not frames:
        raise RuntimeError("没有任何输入文件包含完整字段，无法筛选。")

    merged = pd.concat(frames, ignore_index=True)

    # 统一数值字段
    numeric_cols = [
        "fast",
        "slow",
        "rows",
        "annual_return",
        "max_drawdown",
        "sharpe",
        "total_return",
        "buy_hold_total_return",
        "excess_total_return",
        "trade_count",
        "score",
    ]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # 去重：如果同一个 symbol + fast + slow 在多个文件里重复，保留 score 更高的一条
    merged = merged.sort_values("score", ascending=False)
    merged = merged.drop_duplicates(subset=["symbol", "fast", "slow"], keep="first")

    return merged, files


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()

    conditions = {
        "pass_rows": work["rows"] >= args.min_rows,
        "pass_trade_count": work["trade_count"] >= args.min_trade_count,
        "pass_drawdown": work["max_drawdown"] >= args.max_drawdown_limit,
        "pass_sharpe": work["sharpe"] >= args.min_sharpe,
        "pass_annual_return": work["annual_return"] >= args.min_annual_return,
    }

    if args.allow_negative_excess:
        conditions["pass_excess"] = True
    else:
        conditions["pass_excess"] = work["excess_total_return"] > args.min_excess_total_return

    if args.market != "ALL":
        conditions["pass_market"] = work["market"] == args.market
    else:
        conditions["pass_market"] = True

    if args.security_type != "ALL":
        conditions["pass_security_type"] = work["security_type"] == args.security_type
    else:
        conditions["pass_security_type"] = True

    for name, cond in conditions.items():
        work[name] = cond

    pass_cols = list(conditions.keys())
    work["passed"] = work[pass_cols].all(axis=1)

    filtered = work[work["passed"]].copy()

    # 重新计算一个更保守的筛选分数
    # 解释：
    # - 年化越高越好
    # - 夏普越高越好
    # - 超额收益越高越好
    # - max_drawdown 是负数，所以越接近 0 越好
    # - 交易次数过多可能是噪音，这里暂时不惩罚，只在筛选条件里控制最低次数
    filtered["filter_score"] = (
        filtered["annual_return"]
        + 0.30 * filtered["sharpe"].fillna(0)
        + 0.40 * filtered["excess_total_return"]
        + 0.80 * filtered["max_drawdown"]
    )

    filtered = filtered.sort_values(
        by=["filter_score", "sharpe", "annual_return", "max_drawdown"],
        ascending=[False, False, False, False],
    )

    return filtered, work


def make_best_per_symbol(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
        return filtered.copy()

    best = filtered.sort_values(
        by=["filter_score", "sharpe", "annual_return"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["symbol"], keep="first")

    return best.reset_index(drop=True)


def save_outputs(
    filtered: pd.DataFrame,
    audited: pd.DataFrame,
    best: pd.DataFrame,
    args: argparse.Namespace,
    input_files: list[Path],
) -> None:
    tag = (
        f"{args.output_name}_"
        f"{args.market}_{args.security_type}_"
        f"rows{args.min_rows}_"
        f"trades{args.min_trade_count}_"
        f"dd{str(args.max_drawdown_limit).replace('-', 'm').replace('.', 'p')}_"
        f"sharpe{str(args.min_sharpe).replace('.', 'p')}_"
        f"ann{str(args.min_annual_return).replace('.', 'p')}"
    )

    filtered_path = OUTPUT_DIR / f"{tag}_filtered.csv"
    best_path = OUTPUT_DIR / f"{tag}_best_per_symbol.csv"
    audit_path = OUTPUT_DIR / f"{tag}_audit_all.csv"
    report_path = OUTPUT_DIR / f"{tag}_report.txt"

    filtered.to_csv(filtered_path, index=False, encoding="utf-8-sig")
    best.to_csv(best_path, index=False, encoding="utf-8-sig")
    audited.to_csv(audit_path, index=False, encoding="utf-8-sig")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("filter_ma_results.py 筛选报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("输入文件：\n")
        for file in input_files:
            f.write(f"- {file}\n")

        f.write("\n筛选条件：\n")
        f.write(f"market: {args.market}\n")
        f.write(f"security_type: {args.security_type}\n")
        f.write(f"min_rows: {args.min_rows}\n")
        f.write(f"min_trade_count: {args.min_trade_count}\n")
        f.write(f"max_drawdown_limit: {args.max_drawdown_limit}\n")
        f.write(f"min_sharpe: {args.min_sharpe}\n")
        f.write(f"min_annual_return: {args.min_annual_return}\n")
        f.write(f"allow_negative_excess: {args.allow_negative_excess}\n")
        f.write(f"min_excess_total_return: {args.min_excess_total_return}\n")

        f.write("\n数量统计：\n")
        f.write(f"原始结果数量: {len(audited)}\n")
        f.write(f"通过筛选数量: {len(filtered)}\n")
        f.write(f"每只股票最佳组合数量: {len(best)}\n")

        f.write("\n输出文件：\n")
        f.write(f"filtered: {filtered_path}\n")
        f.write(f"best_per_symbol: {best_path}\n")
        f.write(f"audit_all: {audit_path}\n")

    logger.info("筛选完成。")
    logger.info("原始结果数量：%d", len(audited))
    logger.info("通过筛选数量：%d", len(filtered))
    logger.info("每只股票最佳组合数量：%d", len(best))
    logger.info("筛选结果：%s", filtered_path)
    logger.info("每只股票最佳组合：%s", best_path)
    logger.info("审计明细：%s", audit_path)
    logger.info("报告：%s", report_path)

    if not filtered.empty:
        print("\nTop 20 filtered:")
        cols = [
            "symbol",
            "market",
            "fast",
            "slow",
            "rows",
            "annual_return",
            "max_drawdown",
            "sharpe",
            "total_return",
            "buy_hold_total_return",
            "excess_total_return",
            "trade_count",
            "filter_score",
        ]
        print(filtered[cols].head(20).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛选批量均线回测结果")

    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="指定输入 summary CSV。可重复传入。不传则自动扫描 batch_ma_summary_*.csv。",
    )
    parser.add_argument(
        "--input-glob",
        default=str(DEFAULT_INPUT_GLOB),
        help="输入文件通配符。默认读取 backtests\\batch_ma_csv\\batch_ma_summary_*.csv。",
    )
    parser.add_argument("--output-name", default="ma_filter", help="输出文件名前缀")

    parser.add_argument("--market", default="ALL", choices=["ALL", "SH", "SZ"], help="筛选市场")
    parser.add_argument("--security-type", default="stock", choices=["stock", "index", "other", "ALL"], help="筛选证券类型")

    parser.add_argument("--min-rows", type=int, default=1500, help="最少样本行数，用于剔除短样本。默认 1500")
    parser.add_argument("--min-trade-count", type=int, default=8, help="最少换仓次数。默认 8")
    parser.add_argument("--max-drawdown-limit", type=float, default=-0.45, help="最大回撤下限。默认 -0.45，表示保留 max_drawdown >= -45%")
    parser.add_argument("--min-sharpe", type=float, default=0.6, help="最低夏普比率。默认 0.6")
    parser.add_argument("--min-annual-return", type=float, default=0.08, help="最低年化收益。默认 8%")
    parser.add_argument("--min-excess-total-return", type=float, default=0.0, help="最低总超额收益。默认 > 0")
    parser.add_argument("--allow-negative-excess", action="store_true", help="允许跑输买入持有")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    merged, input_files = read_inputs(args.input, args.input_glob)
    filtered, audited = apply_filters(merged, args)
    best = make_best_per_symbol(filtered)

    save_outputs(
        filtered=filtered,
        audited=audited,
        best=best,
        args=args,
        input_files=input_files,
    )


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception as exc:
        logger.error("程序异常：%s", repr(exc))
        raise