# -*- coding: utf-8 -*-
"""
batch_ma_backtest_csv.py

用途：
批量读取 QMT 导出的 CSV 日线数据，对全市场/指定市场/指定类型证券做均线策略批量回测。

依赖：
strategies/ma_demo_strategy_csv.py 中已有的：
- scan_qmt_export
- load_qmt_price_csv
- calc_metrics

运行示例：

1. 小规模测试，只跑前 50 个标的：
python scripts\\batch_ma_backtest_csv.py --limit 50

2. 短样本模式：纳入新股，只要求数据行数满足 slow + warmup-buffer。
   适合“先尽量纳入更多股票”，但结果里要重点看 rows 和 trade_count。
python scripts\\batch_ma_backtest_csv.py --market SZ --security-type stock --fast-list 5,10,20 --slow-list 60,120,250 --start 20150101 --sample-mode short

3. 长样本模式：默认要求至少 1500 行数据。
   适合更稳健的长期样本筛选，会排除很多上市较晚的股票。
python scripts\\batch_ma_backtest_csv.py --market SZ --security-type stock --fast-list 5,10,20 --slow-list 60,120,250 --start 20150101 --sample-mode long

4. 自定义样本模式：自己指定最少行数。
python scripts\\batch_ma_backtest_csv.py --market SZ --security-type stock --fast-list 5,10,20 --slow-list 60,120,250 --start 20150101 --sample-mode custom --min-rows 1000

5. 全市场股票，短样本模式：
python scripts\\batch_ma_backtest_csv.py --security-type stock --fast-list 5,10,20 --slow-list 60,120,250 --start 20150101 --sample-mode short

6. 全市场股票，长样本模式：
python scripts\\batch_ma_backtest_csv.py --security-type stock --fast-list 5,10,20 --slow-list 60,120,250 --start 20150101 --sample-mode long
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402


BACKTEST_DIR = PROJECT_ROOT / "backtests" / "batch_ma_csv"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.validation import parse_int_list, parse_workers  # noqa: E402

logger = logging.getLogger(__name__)


def get_required_rows_simple(sample_mode: str, slow: int, warmup_buffer: int, long_min_rows: int, min_rows_arg: int | None) -> int:
    """根据 sample-mode 计算最少数据行数（纯函数版本，供 worker 使用）。"""
    warmup_required = slow + warmup_buffer

    if sample_mode == "short":
        return warmup_required

    if sample_mode == "long":
        return max(long_min_rows, warmup_required)

    if sample_mode == "custom":
        if min_rows_arg is None:
            raise ValueError("--sample-mode custom 时必须提供 --min-rows。")
        return max(min_rows_arg, warmup_required)

    raise ValueError(f"未知 sample_mode: {sample_mode}")


def get_required_rows(args: argparse.Namespace, slow: int) -> int:
    """根据 sample-mode 计算每组参数需要的最少数据行数。"""
    return get_required_rows_simple(args.sample_mode, slow, args.warmup_buffer, args.long_min_rows, args.min_rows)


def process_one_stock(args_tuple):
    """
    处理一只股票：加载 CSV，运行所有参数组合，返回结果。
    供 ProcessPoolExecutor 使用的 worker 函数。
    """
    (item_dict, pairs, start, end, cash, commission, sell_tax, slippage,
     sample_mode, warmup_buffer, long_min_rows, min_rows_arg) = args_tuple

    symbol = item_dict["symbol"]
    market = item_dict["market"]
    security_type = item_dict["security_type"]
    csv_path = Path(item_dict["csv_path"])

    rows = []
    skipped = []
    errors = []

    try:
        df = ma.load_qmt_price_csv(csv_path, start, end)

        for fast, slow in pairs:
            required_rows = get_required_rows_simple(sample_mode, slow, warmup_buffer, long_min_rows, min_rows_arg)

            if len(df) < required_rows:
                skip_item = {
                    "symbol": symbol,
                    "market": market,
                    "security_type": security_type,
                    "fast": fast,
                    "slow": slow,
                    "sample_mode": sample_mode,
                    "rows": len(df),
                    "required_rows": required_rows,
                    "start_date": str(df.index.min().date()) if not df.empty else "",
                    "end_date": str(df.index.max().date()) if not df.empty else "",
                    "csv_path": str(csv_path),
                    "reason": "rows_not_enough",
                }
                skipped.append(skip_item)
                continue

            _, metrics = run_one_backtest(
                df=df,
                fast=fast,
                slow=slow,
                cash=cash,
                commission=commission,
                sell_tax=sell_tax,
                slippage=slippage,
            )

            rows.append(
                build_summary_row(
                    symbol=symbol,
                    security_type=security_type,
                    market=market,
                    csv_path=str(csv_path),
                    df=df,
                    fast=fast,
                    slow=slow,
                    required_rows=required_rows,
                    sample_mode=sample_mode,
                    metrics=metrics,
                )
            )

    except Exception as exc:
        errors.append(
            {
                "symbol": symbol,
                "market": market,
                "security_type": security_type,
                "csv_path": str(csv_path),
                "error": repr(exc),
            }
        )

    return rows, skipped, errors


def run_one_backtest(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    cash: float,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> tuple[pd.DataFrame, dict]:
    result = df.copy()

    result["fast_ma"] = result["close"].rolling(fast).mean()
    result["slow_ma"] = result["close"].rolling(slow).mean()

    # 当日收盘后产生信号
    result["signal"] = np.where(result["fast_ma"] > result["slow_ma"], 1, 0)

    # 次日持仓，避免未来函数
    result["position"] = result["signal"].shift(1).fillna(0)

    result["stock_ret"] = result["close"].pct_change().fillna(0)

    # 简化交易成本模型：
    # 买入：佣金 + 滑点
    # 卖出：佣金 + 印花税 + 滑点
    pos_change = result["position"].diff().fillna(result["position"])
    buy_turnover = pos_change.clip(lower=0)
    sell_turnover = (-pos_change).clip(lower=0)

    result["cost"] = (
        buy_turnover * (commission + slippage)
        + sell_turnover * (commission + sell_tax + slippage)
    )

    result["strategy_ret"] = result["position"] * result["stock_ret"] - result["cost"]
    result["equity"] = cash * (1.0 + result["strategy_ret"]).cumprod()
    result["buy_hold_equity"] = cash * (1.0 + result["stock_ret"]).cumprod()

    metrics = ma.calc_metrics(result)

    buy_hold_total_return = result["buy_hold_equity"].iloc[-1] / result["buy_hold_equity"].iloc[0] - 1.0
    metrics["buy_hold_total_return"] = float(buy_hold_total_return)
    metrics["excess_total_return"] = float(metrics["total_return"] - buy_hold_total_return)

    return result, metrics


def build_summary_row(
    symbol: str,
    security_type: str,
    market: str,
    csv_path: str,
    df: pd.DataFrame,
    fast: int,
    slow: int,
    required_rows: int,
    sample_mode: str,
    metrics: dict,
) -> dict:
    return {
        "symbol": symbol,
        "market": market,
        "security_type": security_type,
        "fast": fast,
        "slow": slow,
        "sample_mode": sample_mode,
        "required_rows": required_rows,
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "rows": len(df),
        "total_return": metrics["total_return"],
        "annual_return": metrics["annual_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe": metrics["sharpe"],
        "trade_count": metrics["trade_count"],
        "final_equity": metrics["final_equity"],
        "buy_hold_total_return": metrics["buy_hold_total_return"],
        "excess_total_return": metrics["excess_total_return"],
        "csv_path": csv_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量均线回测")

    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT), help="QMT CSV 导出根目录")
    parser.add_argument("--market", default="ALL", choices=["ALL", "SH", "SZ"], help="市场筛选")
    parser.add_argument("--security-type", default="stock", choices=["stock", "index", "other", "ALL"], help="证券类型筛选")

    parser.add_argument("--start", default="20150101", help="开始日期，例如 20150101")
    parser.add_argument("--end", default="", help="结束日期，例如 20260514；留空表示最新")

    parser.add_argument("--fast-list", default="20", help="快均线列表，例如 5,10,20")
    parser.add_argument("--slow-list", default="120", help="慢均线列表，例如 60,120,250")

    parser.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0001, help="佣金率，默认万一")
    parser.add_argument("--sell-tax", type=float, default=0.0005, help="卖出印花税，默认万五")
    parser.add_argument("--slippage", type=float, default=0.0, help="滑点率，默认 0")

    parser.add_argument(
        "--sample-mode",
        choices=["short", "long", "custom"],
        default="short",
        help=(
            "样本模式："
            "short=纳入短样本，只要求 rows >= slow + warmup-buffer；"
            "long=长样本筛选，默认 rows >= 1500；"
            "custom=使用 --min-rows 自定义。"
        ),
    )
    parser.add_argument(
        "--warmup-buffer",
        type=int,
        default=10,
        help="慢均线之外额外要求的缓冲行数。默认 10。",
    )
    parser.add_argument(
        "--long-min-rows",
        type=int,
        default=1500,
        help="long 样本模式下要求的最少行数。默认 1500。",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=None,
        help="custom 样本模式下要求的最少行数。例如 --sample-mode custom --min-rows 1000。",
    )

    parser.add_argument("--limit", type=int, default=0, help="只测试前 N 个文件；0 表示不限制")
    parser.add_argument("--print-skips", action="store_true", help="打印被跳过的 symbol；默认只记录到 skipped 文件")
    parser.add_argument("--workers", type=parse_workers, default=1, help="并行进程数。默认 1（单进程）。使用 'auto' 自动检测 CPU 核心数，或指定整数如 --workers 6")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root

    fast_list = parse_int_list(args.fast_list)
    slow_list = parse_int_list(args.slow_list)

    pairs = [(f, s) for f in fast_list for s in slow_list if f < s]
    if not pairs:
        raise ValueError("没有有效均线组合。要求 fast < slow。")

    catalog = ma.scan_qmt_export(export_root)

    if args.market != "ALL":
        catalog = catalog[catalog["market"] == args.market]

    if args.security_type != "ALL":
        catalog = catalog[catalog["security_type"] == args.security_type]

    catalog = catalog.sort_values(["market", "symbol"]).reset_index(drop=True)

    if args.limit and args.limit > 0:
        catalog = catalog.head(args.limit)

    print("\n批量回测配置：")
    print(f"市场：{args.market}")
    print(f"证券类型：{args.security_type}")
    print(f"标的数量：{len(catalog)}")
    print(f"参数组合：{pairs}")
    print(f"日期区间：{args.start} 至 {args.end or 'latest'}")
    print(f"样本模式：{args.sample_mode}")
    print(f"warmup_buffer：{args.warmup_buffer}")

    if args.sample_mode == "long":
        print(f"long_min_rows：{args.long_min_rows}")
    elif args.sample_mode == "custom":
        print(f"min_rows：{args.min_rows}")
    else:
        print("short 模式：不设置固定最少行数，只要求 rows >= slow + warmup_buffer")

    rows = []
    errors = []
    skipped = []

    if args.workers > 1:
        # 并行模式：按股票并行，每个 worker 处理一只股票的所有参数组合
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = []
        for _, item in catalog.iterrows():
            item_dict = item.to_dict()
            tasks.append((
                item_dict, pairs, args.start, args.end, args.cash,
                args.commission, args.sell_tax, args.slippage,
                args.sample_mode, args.warmup_buffer,
                args.long_min_rows, args.min_rows,
            ))

        total_stocks = len(tasks)
        completed_stocks = 0
        print(f"\n并行模式：{args.workers} 进程处理 {total_stocks} 只股票...")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_stock, t): t[0]["symbol"] for t in tasks}

            for future in as_completed(futures):
                completed_stocks += 1
                if completed_stocks % 50 == 0:
                    print(f"进度：{completed_stocks}/{total_stocks} 只股票完成")

                try:
                    stock_rows, stock_skipped, stock_errors = future.result()
                    rows.extend(stock_rows)
                    skipped.extend(stock_skipped)
                    errors.extend(stock_errors)
                except Exception as exc:
                    symbol = futures[future]
                    errors.append({"symbol": symbol, "error": repr(exc)})

    else:
        # 单进程模式：原有逻辑
        total_tasks = len(catalog) * len(pairs)
        task_no = 0

        for _, item in catalog.iterrows():
            symbol = item["symbol"]
            market = item["market"]
            security_type = item["security_type"]
            csv_path = Path(item["csv_path"])

            try:
                df = ma.load_qmt_price_csv(csv_path, args.start, args.end)

                for fast, slow in pairs:
                    task_no += 1

                    if task_no % 100 == 0:
                        print(f"进度：{task_no}/{total_tasks}")

                    required_rows = get_required_rows(args, slow)

                    if len(df) < required_rows:
                        skip_item = {
                            "symbol": symbol,
                            "market": market,
                            "security_type": security_type,
                            "fast": fast,
                            "slow": slow,
                            "sample_mode": args.sample_mode,
                            "rows": len(df),
                            "required_rows": required_rows,
                            "start_date": str(df.index.min().date()) if not df.empty else "",
                            "end_date": str(df.index.max().date()) if not df.empty else "",
                            "csv_path": str(csv_path),
                            "reason": "rows_not_enough",
                        }
                        skipped.append(skip_item)

                        if args.print_skips:
                            print(
                                f"[跳过] {symbol}: rows={len(df)}, "
                                f"required_rows={required_rows}, fast={fast}, slow={slow}"
                            )
                        continue

                    _, metrics = run_one_backtest(
                        df=df,
                        fast=fast,
                        slow=slow,
                        cash=args.cash,
                        commission=args.commission,
                        sell_tax=args.sell_tax,
                        slippage=args.slippage,
                    )

                    rows.append(
                        build_summary_row(
                            symbol=symbol,
                            security_type=security_type,
                            market=market,
                            csv_path=str(csv_path),
                            df=df,
                            fast=fast,
                            slow=slow,
                            required_rows=required_rows,
                            sample_mode=args.sample_mode,
                            metrics=metrics,
                        )
                    )

            except Exception as exc:
                errors.append(
                    {
                        "symbol": symbol,
                        "market": market,
                        "security_type": security_type,
                        "csv_path": str(csv_path),
                        "error": repr(exc),
                    }
                )

    if not rows:
        raise RuntimeError("没有生成任何回测结果。")

    summary = pd.DataFrame(rows)

    # 排名逻辑：不是最终投资建议，只是帮你初筛
    # max_drawdown 是负数，越接近 0 越好。
    summary["score"] = (
        summary["annual_return"]
        + 0.2 * summary["sharpe"].fillna(0)
        + 0.5 * summary["excess_total_return"]
        + summary["max_drawdown"]
    )

    summary = summary.sort_values(
        by=["score", "sharpe", "annual_return"],
        ascending=[False, False, False],
    )

    fast_tag = "f" + "-".join(map(str, fast_list))
    slow_tag = "s" + "-".join(map(str, slow_list))

    if args.sample_mode == "short":
        sample_tag = f"short_warmup{args.warmup_buffer}"
    elif args.sample_mode == "long":
        sample_tag = f"long_minrows{args.long_min_rows}_warmup{args.warmup_buffer}"
    else:
        sample_tag = f"custom_minrows{args.min_rows}_warmup{args.warmup_buffer}"

    tag = (
        f"{args.security_type}_{args.market}_{args.start}_{args.end or 'latest'}_"
        f"{fast_tag}_{slow_tag}_{sample_tag}"
    )

    summary_path = BACKTEST_DIR / f"batch_ma_summary_{tag}.csv"
    top50_path = BACKTEST_DIR / f"batch_ma_top50_{tag}.csv"
    skipped_path = BACKTEST_DIR / f"batch_ma_skipped_{tag}.csv"
    errors_path = BACKTEST_DIR / f"batch_ma_errors_{tag}.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary.head(50).to_csv(top50_path, index=False, encoding="utf-8-sig")

    if skipped:
        pd.DataFrame(skipped).to_csv(skipped_path, index=False, encoding="utf-8-sig")

    if errors:
        pd.DataFrame(errors).to_csv(errors_path, index=False, encoding="utf-8-sig")

    print("\n批量回测完成。")
    print(f"结果数量：{len(summary)}")
    print(f"跳过数量：{len(skipped)}")
    print(f"错误数量：{len(errors)}")
    print(f"汇总文件：{summary_path}")
    print(f"Top50 文件：{top50_path}")

    if skipped:
        print(f"跳过记录文件：{skipped_path}")

    if errors:
        print(f"错误文件：{errors_path}")

    print("\nTop 20：")
    cols = [
        "symbol",
        "security_type",
        "fast",
        "slow",
        "sample_mode",
        "annual_return",
        "max_drawdown",
        "sharpe",
        "total_return",
        "buy_hold_total_return",
        "excess_total_return",
        "trade_count",
        "rows",
        "required_rows",
        "score",
    ]
    print(summary[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception:
        logger.error("程序异常：", exc_info=True)