# -*- coding: utf-8 -*-
"""
ma_demo_strategy.py

用途：
用 xtdata 读取 QMT / miniQMT 的历史行情，做一个最小化均线策略回测。

策略逻辑：
1. 快均线 > 慢均线：持有
2. 快均线 <= 慢均线：空仓
3. 使用前一日信号决定下一日持仓，避免未来函数
4. 不调用任何实盘下单接口

运行示例：
python strategies\\ma_demo_strategy.py
python strategies\\ma_demo_strategy.py --stock 600519.SH --fast 5 --slow 20
python strategies\\ma_demo_strategy.py --stock 000001.SZ --fast 10 --slow 60
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xtquant import xtdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.metrics import max_drawdown_from_equity  # noqa: E402

logger = logging.getLogger(__name__)

BACKTEST_DIR = PROJECT_ROOT / "backtests"
BACKTEST_DIR.mkdir(exist_ok=True)


def download_history(stock: str, period: str, start: str, end: str, incrementally: bool = False) -> None:
    logger.info(
        "下载历史行情: stock=%s, period=%s, start=%s, end=%s, incrementally=%s",
        stock, period, start, end, incrementally,
    )

    try:
        xtdata.download_history_data(
            stock_code=stock,
            period=period,
            start_time=start,
            end_time=end,
            incrementally=incrementally,
        )
        logger.info("历史行情下载完成。")
        return
    except TypeError:
        pass
    except Exception as exc:
        logger.warning("按新版参数下载失败，尝试兼容方式。错误：%r", exc)

    try:
        xtdata.download_history_data(
            stock,
            period,
            start,
            end,
            incrementally,
        )
        logger.info("历史行情下载完成。")
        return
    except Exception as exc:
        logger.warning("按兼容参数下载失败，尝试只按代码和周期下载。错误：%r", exc)

    xtdata.download_history_data(stock, period=period)
    logger.info("历史行情下载完成。")


def load_history(stock: str, period: str, start: str, end: str) -> pd.DataFrame:
    logger.info("读取历史行情: %s", stock)

    data = xtdata.get_market_data_ex(
        field_list=[],
        stock_list=[stock],
        period=period,
        start_time=start,
        end_time=end,
        count=-1,
        dividend_type="front",
        fill_data=True,
    )

    if not isinstance(data, dict) or stock not in data:
        raise RuntimeError(f"没有读取到 {stock} 的行情数据。返回值: {data}")

    df = data[stock]

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"{stock} 返回的数据为空或不是 DataFrame。返回值类型: {type(df)}")

    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    if "close" not in df.columns:
        raise RuntimeError(f"行情数据中没有 close 字段。当前字段: {list(df.columns)}")

    # 尝试把 time 字段转成日期索引，便于画图和保存
    if "time" in df.columns:
        try:
            df["datetime"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
            if df["datetime"].notna().any():
                df = df.set_index("datetime", drop=False)
        except Exception:
            pass

    df = df.dropna(subset=["close"])
    df = df.sort_index()

    logger.info("读取完成，共 %d 行。", len(df))
    logger.debug("数据尾部:\n%s", df.tail())

    return df


def calc_metrics(result: pd.DataFrame) -> dict:
    strategy_ret = result["strategy_ret"].dropna()
    equity = result["equity"].dropna()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0

    days = max(len(strategy_ret), 1)
    annual_return = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1.0

    std = strategy_ret.std()
    if std == 0 or np.isnan(std):
        sharpe = np.nan
    else:
        sharpe = strategy_ret.mean() / std * SQRT_TRADING_DAYS_PER_YEAR

    mdd = max_drawdown_from_equity(equity)

    # position 从 0 到 1 或从 1 到 0 都算一次变化
    trade_count = int(result["position"].diff().abs().fillna(0).sum())

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "trade_count": trade_count,
        "final_equity": float(equity.iloc[-1]),
    }


def run_backtest(
    df: pd.DataFrame,
    stock: str,
    fast: int,
    slow: int,
    initial_cash: float,
) -> tuple[pd.DataFrame, dict]:
    result = df.copy()

    result["fast_ma"] = result["close"].rolling(fast).mean()
    result["slow_ma"] = result["close"].rolling(slow).mean()

    # 当日收盘后产生信号
    result["signal"] = np.where(result["fast_ma"] > result["slow_ma"], 1, 0)

    # 次日持仓，避免未来函数
    result["position"] = result["signal"].shift(1).fillna(0)

    result["stock_ret"] = result["close"].pct_change().fillna(0)
    result["strategy_ret"] = result["position"] * result["stock_ret"]

    result["equity"] = initial_cash * (1.0 + result["strategy_ret"]).cumprod()
    result["buy_hold_equity"] = initial_cash * (1.0 + result["stock_ret"]).cumprod()

    metrics = calc_metrics(result)

    print("\n" + "=" * 80)
    print(f"回测标的: {stock}")
    print(f"快均线: {fast}")
    print(f"慢均线: {slow}")
    print(f"初始资金: {initial_cash:,.2f}")
    print("-" * 80)
    print(f"策略总收益: {metrics['total_return']:.2%}")
    print(f"策略年化收益: {metrics['annual_return']:.2%}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")

    if np.isnan(metrics["sharpe"]):
        print("夏普比率: NaN")
    else:
        print(f"夏普比率: {metrics['sharpe']:.4f}")

    print(f"换仓次数: {metrics['trade_count']}")
    print(f"期末权益: {metrics['final_equity']:,.2f}")
    print("=" * 80)

    return result, metrics


def save_outputs(result: pd.DataFrame, metrics: dict, stock: str, fast: int, slow: int) -> None:
    safe_stock = stock.replace(".", "_")
    strategy_name = "ma_demo_strategy"
    run_name = f"ma_demo_{safe_stock}_fast{fast}_slow{slow}"

    run_dir = BACKTEST_DIR / strategy_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / f"{run_name}.csv"
    metrics_path = run_dir / f"{run_name}_metrics.txt"
    png_path = run_dir / f"{run_name}_equity.png"

    result.to_csv(csv_path, encoding="utf-8-sig")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"strategy_name: {strategy_name}\n")
        f.write(f"run_name: {run_name}\n")
        f.write(f"stock: {stock}\n")
        f.write(f"fast: {fast}\n")
        f.write(f"slow: {slow}\n")
        f.write(f"start_date: {result.index.min()}\n")
        f.write(f"end_date: {result.index.max()}\n")
        f.write(f"rows: {len(result)}\n")
        f.write("\n")

        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.8f}\n")
            else:
                f.write(f"{key}: {value}\n")

    plt.figure(figsize=(12, 6))
    result["equity"].plot(label="MA strategy")
    result["buy_hold_equity"].plot(label="Buy and Hold")
    plt.title(f"MA Demo Backtest - {stock}")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    logger.info("结果已保存到：%s", run_dir)
    logger.info("CSV: %s", csv_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("PNG: %s", png_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最小均线策略回测示例")
    parser.add_argument("--stock", default="000001.SZ", help="证券代码，例如 000001.SZ / 600519.SH")
    parser.add_argument("--period", default="1d", help="周期，例如 1d")
    parser.add_argument("--start", default="20200101", help="开始时间，格式 YYYYMMDD，例如 20200101")
    parser.add_argument("--end", default="", help="结束时间，留空表示取到最新")
    parser.add_argument("--fast", type=int, default=5, help="快均线窗口")
    parser.add_argument("--slow", type=int, default=20, help="慢均线窗口")
    parser.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，直接读取本地数据")
    parser.add_argument("--incremental-download", action="store_true", help="使用增量下载；默认是全量下载")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fast >= args.slow:
        raise ValueError("fast 必须小于 slow，例如 fast=5, slow=20。")

    logger.info("Python executable: %s", sys.executable)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Backtest dir: %s", BACKTEST_DIR)

    if not args.skip_download:
        download_history(
            stock=args.stock,
            period=args.period,
            start=args.start,
            end=args.end,
            incrementally=args.incremental_download,
        )

    df = load_history(args.stock, args.period, args.start, args.end)

    if len(df) < args.slow + 10:
        raise RuntimeError(f"历史数据太少，当前只有 {len(df)} 行，无法做均线回测。")

    result, metrics = run_backtest(
        df=df,
        stock=args.stock,
        fast=args.fast,
        slow=args.slow,
        initial_cash=args.cash,
    )

    save_outputs(result, metrics, args.stock, args.fast, args.slow)


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception:
        logger.error("程序异常：", exc_info=True)