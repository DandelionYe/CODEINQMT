# -*- coding: utf-8 -*-
"""
ma_market_filter_strategy_csv.py

MA v2：个股均线交叉 + 大盘趋势过滤

策略逻辑：
1. 个股信号：
   stock_fast_ma > stock_slow_ma → 个股趋势向上

2. 大盘过滤：
   benchmark_fast_ma > benchmark_slow_ma → 市场环境允许开仓

3. 最终信号：
   final_signal = stock_signal AND market_filter

4. 执行规则：
   当日收盘生成信号，次日持仓，避免未来函数。

运行示例：
python strategies\\ma_market_filter_strategy_csv.py --stock 000001.SZ --start 20150101

python strategies\\ma_market_filter_strategy_csv.py ^
  --stock 000001.SZ ^
  --fast 20 ^
  --slow 120 ^
  --benchmark 000300.SH ^
  --benchmark-fast 20 ^
  --benchmark-slow 120 ^
  --start 20150101
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR, DEFAULT_BENCHMARK  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.metrics import max_drawdown_from_equity  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backtests" / "ma_market_filter_strategy_csv"


def calc_metrics(result: pd.DataFrame) -> dict:
    strategy_ret = result["strategy_ret"].dropna()
    equity = result["equity"].dropna()

    if strategy_ret.empty or equity.empty:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "trade_count": 0,
            "final_equity": np.nan,
            "buy_hold_total_return": np.nan,
            "stock_only_total_return": np.nan,
            "market_filter_on_ratio": np.nan,
            "strategy_exposure_ratio": np.nan,
        }

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    days = max(len(strategy_ret), 1)
    annual_return = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1.0
    annual_volatility = strategy_ret.std() * SQRT_TRADING_DAYS_PER_YEAR

    if strategy_ret.std() == 0 or np.isnan(strategy_ret.std()):
        sharpe = np.nan
    else:
        sharpe = strategy_ret.mean() / strategy_ret.std() * SQRT_TRADING_DAYS_PER_YEAR

    buy_hold_total_return = (
        result["buy_hold_equity"].dropna().iloc[-1]
        / result["buy_hold_equity"].dropna().iloc[0]
        - 1.0
    )

    stock_only_total_return = (
        result["stock_only_equity"].dropna().iloc[-1]
        / result["stock_only_equity"].dropna().iloc[0]
        - 1.0
    )

    trade_count = int(result["position"].diff().abs().fillna(0).sum())

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "max_drawdown": max_drawdown_from_equity(equity),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "trade_count": trade_count,
        "final_equity": float(equity.iloc[-1]),
        "buy_hold_total_return": float(buy_hold_total_return),
        "stock_only_total_return": float(stock_only_total_return),
        "market_filter_on_ratio": float(result["market_filter"].mean()),
        "strategy_exposure_ratio": float(result["position"].mean()),
    }


def prepare_benchmark_filter(
    benchmark_df: pd.DataFrame,
    benchmark_fast: int,
    benchmark_slow: int,
) -> pd.DataFrame:
    bench = benchmark_df.copy()

    bench["benchmark_fast_ma"] = bench["close"].rolling(benchmark_fast).mean()
    bench["benchmark_slow_ma"] = bench["close"].rolling(benchmark_slow).mean()

    bench["market_filter"] = np.where(
        bench["benchmark_fast_ma"] > bench["benchmark_slow_ma"],
        1,
        0,
    )

    return bench[["date", "close", "benchmark_fast_ma", "benchmark_slow_ma", "market_filter"]].copy()


def run_backtest(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    stock_symbol: str,
    benchmark_symbol: str,
    fast: int,
    slow: int,
    benchmark_fast: int,
    benchmark_slow: int,
    cash: float,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> tuple[pd.DataFrame, dict]:
    result = stock_df.copy()

    result["fast_ma"] = result["close"].rolling(fast).mean()
    result["slow_ma"] = result["close"].rolling(slow).mean()

    result["stock_signal"] = np.where(result["fast_ma"] > result["slow_ma"], 1, 0)

    bench_filter = prepare_benchmark_filter(
        benchmark_df=benchmark_df,
        benchmark_fast=benchmark_fast,
        benchmark_slow=benchmark_slow,
    )

    bench_filter = bench_filter.rename(
        columns={
            "close": "benchmark_close",
        }
    )

    bench_filter = bench_filter.set_index("date", drop=False)
    result = result.set_index("date", drop=False)

    # 使用股票交易日作为主索引；基准过滤信号对齐到股票日期。
    result["benchmark_close"] = bench_filter["benchmark_close"].reindex(result.index).ffill()
    result["benchmark_fast_ma"] = bench_filter["benchmark_fast_ma"].reindex(result.index).ffill()
    result["benchmark_slow_ma"] = bench_filter["benchmark_slow_ma"].reindex(result.index).ffill()
    result["market_filter"] = bench_filter["market_filter"].reindex(result.index).ffill().fillna(0).astype(int)

    # v2 最终信号：个股趋势 + 大盘趋势过滤
    result["final_signal"] = (
        (result["stock_signal"] == 1)
        & (result["market_filter"] == 1)
    ).astype(int)

    # v1 对照信号：只看个股均线，不加大盘过滤
    result["stock_only_signal"] = result["stock_signal"]

    # 次日持仓，避免未来函数
    result["position"] = result["final_signal"].shift(1).fillna(0)
    result["stock_only_position"] = result["stock_only_signal"].shift(1).fillna(0)

    result["stock_ret"] = result["close"].pct_change().fillna(0)

    # v2 交易成本
    pos_change = result["position"].diff().fillna(result["position"])
    buy_turnover = pos_change.clip(lower=0)
    sell_turnover = (-pos_change).clip(lower=0)

    result["cost"] = (
        buy_turnover * (commission + slippage)
        + sell_turnover * (commission + sell_tax + slippage)
    )

    result["strategy_ret"] = result["position"] * result["stock_ret"] - result["cost"]

    # v1 对照成本
    stock_only_pos_change = result["stock_only_position"].diff().fillna(result["stock_only_position"])
    stock_only_buy = stock_only_pos_change.clip(lower=0)
    stock_only_sell = (-stock_only_pos_change).clip(lower=0)

    result["stock_only_cost"] = (
        stock_only_buy * (commission + slippage)
        + stock_only_sell * (commission + sell_tax + slippage)
    )

    result["stock_only_ret"] = result["stock_only_position"] * result["stock_ret"] - result["stock_only_cost"]

    result["equity"] = cash * (1.0 + result["strategy_ret"]).cumprod()
    result["stock_only_equity"] = cash * (1.0 + result["stock_only_ret"]).cumprod()
    result["buy_hold_equity"] = cash * (1.0 + result["stock_ret"]).cumprod()

    metrics = calc_metrics(result)

    print("\n" + "=" * 80)
    print("MA v2：个股均线交叉 + 大盘趋势过滤")
    print("-" * 80)
    print(f"股票标的: {stock_symbol}")
    print(f"过滤基准: {benchmark_symbol}")
    print(f"股票均线: fast={fast}, slow={slow}")
    print(f"基准均线: benchmark_fast={benchmark_fast}, benchmark_slow={benchmark_slow}")
    print(f"数据区间: {result.index.min().date()} 至 {result.index.max().date()}")
    print(f"数据行数: {len(result)}")
    print(f"初始资金: {cash:,.2f}")
    print(f"佣金率: {commission:.5f}")
    print(f"卖出印花税: {sell_tax:.5f}")
    print(f"滑点率: {slippage:.5f}")
    print("-" * 80)
    print(f"v2 策略总收益: {metrics['total_return']:.2%}")
    print(f"v2 策略年化收益: {metrics['annual_return']:.2%}")
    print(f"v2 最大回撤: {metrics['max_drawdown']:.2%}")
    print("v2 夏普比率: NaN" if np.isnan(metrics["sharpe"]) else f"v2 夏普比率: {metrics['sharpe']:.4f}")
    print(f"v2 换仓次数: {metrics['trade_count']}")
    print(f"v2 期末权益: {metrics['final_equity']:,.2f}")
    print("-" * 80)
    print(f"v1 个股均线总收益: {metrics['stock_only_total_return']:.2%}")
    print(f"买入持有总收益: {metrics['buy_hold_total_return']:.2%}")
    print(f"大盘过滤开启比例: {metrics['market_filter_on_ratio']:.2%}")
    print(f"v2 策略持仓暴露比例: {metrics['strategy_exposure_ratio']:.2%}")
    print("=" * 80)

    return result, metrics


def save_outputs(
    result: pd.DataFrame,
    metrics: dict,
    stock_symbol: str,
    benchmark_symbol: str,
    stock_csv_path: Path,
    benchmark_csv_path: Path,
    fast: int,
    slow: int,
    benchmark_fast: int,
    benchmark_slow: int,
    start: str,
    end: str,
    output_dir: Path,
) -> None:
    safe_stock = stock_symbol.replace(".", "_")
    safe_benchmark = benchmark_symbol.replace(".", "_")

    strategy_name = "ma_market_filter_strategy_csv"
    date_tag = f"{start or 'all'}_{end or 'latest'}"

    run_name = (
        f"ma_mf_{safe_stock}_"
        f"stock{fast}-{slow}_"
        f"bench_{safe_benchmark}_{benchmark_fast}-{benchmark_slow}_"
        f"{date_tag}"
    )

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    result_csv_path = run_dir / f"{run_name}.csv"
    metrics_path = run_dir / f"{run_name}_metrics.txt"
    png_path = run_dir / f"{run_name}_equity.png"

    result.to_csv(result_csv_path, encoding="utf-8-sig", index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"strategy_name: {strategy_name}\n")
        f.write(f"stock_symbol: {stock_symbol}\n")
        f.write(f"benchmark_symbol: {benchmark_symbol}\n")
        f.write(f"stock_csv_path: {stock_csv_path}\n")
        f.write(f"benchmark_csv_path: {benchmark_csv_path}\n")
        f.write(f"fast: {fast}\n")
        f.write(f"slow: {slow}\n")
        f.write(f"benchmark_fast: {benchmark_fast}\n")
        f.write(f"benchmark_slow: {benchmark_slow}\n")
        f.write(f"start_date: {result['date'].min()}\n")
        f.write(f"end_date: {result['date'].max()}\n")
        f.write(f"rows: {len(result)}\n\n")

        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.8f}\n")
            else:
                f.write(f"{key}: {value}\n")

    plt.figure(figsize=(12, 6))
    plt.plot(result["date"], result["equity"], label="MA v2: stock MA + market filter")
    plt.plot(result["date"], result["stock_only_equity"], label="MA v1: stock MA only")
    plt.plot(result["date"], result["buy_hold_equity"], label="Buy and Hold")
    plt.title(f"MA Market Filter Backtest - {stock_symbol} / {benchmark_symbol}")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    logger.info("结果已保存到：%s", run_dir)
    logger.info("CSV: %s", result_csv_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("PNG: %s", png_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MA v2：个股均线交叉 + 大盘趋势过滤，单标的 CSV 回测")

    parser.add_argument("--stock", default="000001.SZ", help="股票代码，例如 000001.SZ")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="大盘过滤基准（默认沪深 300）。如需用中证 500 做 regime 过滤，传 --benchmark 000905.SH")

    parser.add_argument("--fast", type=int, default=20, help="个股快均线")
    parser.add_argument("--slow", type=int, default=120, help="个股慢均线")
    parser.add_argument("--benchmark-fast", type=int, default=20, help="基准快均线")
    parser.add_argument("--benchmark-slow", type=int, default=120, help="基准慢均线")

    parser.add_argument("--start", default="20150101", help="开始日期，例如 20150101")
    parser.add_argument("--end", default="", help="结束日期，例如 20260514；留空表示最新")

    parser.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0001, help="佣金率，默认万一")
    parser.add_argument("--sell-tax", type=float, default=0.0005, help="卖出印花税，默认万五")
    parser.add_argument("--slippage", type=float, default=0.0, help="滑点率，默认 0")

    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT), help="QMT CSV 导出根目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="回测输出目录")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fast >= args.slow:
        raise ValueError("个股 fast 必须小于 slow。")

    if args.benchmark_fast >= args.benchmark_slow:
        raise ValueError("基准 benchmark-fast 必须小于 benchmark-slow。")

    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    stock_csv_path, stock_symbol, stock_market, stock_security_type = ma.find_csv_for_stock(
        stock=args.stock,
        export_root=export_root,
    )

    benchmark_csv_path, benchmark_symbol, benchmark_market, benchmark_security_type = ma.find_csv_for_stock(
        stock=args.benchmark,
        export_root=export_root,
    )

    logger.info("当前使用数据文件：")
    logger.info("stock_symbol: %s", stock_symbol)
    logger.info("stock_security_type: %s", stock_security_type)
    logger.info("stock_csv_path: %s", stock_csv_path)
    logger.info("benchmark_symbol: %s", benchmark_symbol)
    logger.info("benchmark_security_type: %s", benchmark_security_type)
    logger.info("benchmark_csv_path: %s", benchmark_csv_path)

    stock_df = ma.load_qmt_price_csv(
        csv_path=stock_csv_path,
        start=args.start,
        end=args.end,
    )

    benchmark_df = ma.load_qmt_price_csv(
        csv_path=benchmark_csv_path,
        start=args.start,
        end=args.end,
    )

    if len(stock_df) < args.slow + 10:
        raise RuntimeError(f"股票数据太少：rows={len(stock_df)}，至少需要 slow + 10 = {args.slow + 10}")

    if len(benchmark_df) < args.benchmark_slow + 10:
        raise RuntimeError(
            f"基准数据太少：rows={len(benchmark_df)}，"
            f"至少需要 benchmark_slow + 10 = {args.benchmark_slow + 10}"
        )

    result, metrics = run_backtest(
        stock_df=stock_df,
        benchmark_df=benchmark_df,
        stock_symbol=stock_symbol,
        benchmark_symbol=benchmark_symbol,
        fast=args.fast,
        slow=args.slow,
        benchmark_fast=args.benchmark_fast,
        benchmark_slow=args.benchmark_slow,
        cash=args.cash,
        commission=args.commission,
        sell_tax=args.sell_tax,
        slippage=args.slippage,
    )

    save_outputs(
        result=result,
        metrics=metrics,
        stock_symbol=stock_symbol,
        benchmark_symbol=benchmark_symbol,
        stock_csv_path=stock_csv_path,
        benchmark_csv_path=benchmark_csv_path,
        fast=args.fast,
        slow=args.slow,
        benchmark_fast=args.benchmark_fast,
        benchmark_slow=args.benchmark_slow,
        start=args.start,
        end=args.end,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception:
        logger.error("程序异常：", exc_info=True)