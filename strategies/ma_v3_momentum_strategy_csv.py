# -*- coding: utf-8 -*-
"""
ma_v3_momentum_strategy_csv.py

MA v3：个股趋势强度 / 动量排序 + 趋势确认 + 风险过滤

策略逻辑：
1. 趋势确认（必须同时满足）：
   - close > ma_long
   - ma_mid > ma_long
   - ma_long_slope > 0（过去 20 日 ma_long 的变化率）

2. 动量 / 强度评分：
   - trend_strength = close / ma_long - 1
   - momentum_return = close.pct_change(momentum_window)
   - trend_slope = ma_long.pct_change(20)
   - train_score = normalize(trend_strength) + normalize(momentum_return) + normalize(trend_slope)

3. 风险过滤（可选）：
   - 训练期最大回撤、波动率、收益/波动比

4. 大盘过滤（可选）：
   - benchmark_ma 交叉作为 regime filter，非核心 alpha

5. 执行规则：
   当日收盘生成信号，次日持仓，避免未来函数。

运行示例：
python strategies\\ma_v3_momentum_strategy_csv.py --stock 000001.SZ --start 20150101

python strategies\\ma_v3_momentum_strategy_csv.py ^
  --stock 000001.SZ ^
  --ma-mid 60 ^
  --ma-long 250 ^
  --momentum-window 120 ^
  --benchmark 000300.SH ^
  --benchmark-ma 120 ^
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
from scripts.common.constants import DEFAULT_BENCHMARK  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.feature_expression import normalize_zscore  # noqa: E402
from scripts.common.backtest.engine import single_asset_backtest  # noqa: E402
from scripts.common.benchmark import prepare_benchmark_regime  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backtests" / "ma_v3_momentum_strategy_csv"


def compute_v3_signals(
    stock_df: pd.DataFrame,
    ma_mid: int,
    ma_long: int,
    momentum_window: int,
) -> pd.DataFrame:
    """计算 MA v3 信号列，返回带信号列的 DataFrame。"""
    result = stock_df.copy()

    result["ma_mid"] = result["close"].rolling(ma_mid).mean()
    result["ma_long"] = result["close"].rolling(ma_long).mean()
    result["ma_long_slope"] = result["ma_long"].pct_change(20)
    result["momentum_return"] = result["close"].pct_change(momentum_window)
    result["trend_strength"] = result["close"] / result["ma_long"] - 1.0

    # 趋势确认：三个条件同时满足
    result["trend_confirm"] = (
        (result["close"] > result["ma_long"])
        & (result["ma_mid"] > result["ma_long"])
        & (result["ma_long_slope"] > 0)
    ).astype(int)

    # 综合评分（仅用于排序，不影响单标的信号）
    score_parts = normalize_zscore(result["trend_strength"]) + \
                  normalize_zscore(result["momentum_return"]) + \
                  normalize_zscore(result["ma_long_slope"])
    result["train_score"] = score_parts

    return result



def _adapt_engine_metrics(engine_metrics: dict) -> dict:
    """将引擎返回的 metrics dict 映射为 MA v3 CLI 期望的 key。

    引擎返回 strategy_* 前缀的 key；MA v3 CLI 期望无前缀的 key。
    当传入 comparison_signal_col 时，引擎同时返回 stock_only_* 指标。
    """
    return {
        "total_return": engine_metrics["strategy_total_return"],
        "annual_return": engine_metrics["strategy_annual_return"],
        "annual_volatility": engine_metrics["strategy_annual_volatility"],
        "max_drawdown": engine_metrics["strategy_max_drawdown"],
        "sharpe": engine_metrics["strategy_sharpe"],
        "trade_count": engine_metrics["strategy_trade_count"],
        "final_equity": engine_metrics["strategy_final_equity"],
        "buy_hold_total_return": engine_metrics["buy_hold_total_return"],
        "stock_only_total_return": engine_metrics["stock_only_total_return"],
        "market_filter_on_ratio": engine_metrics["market_filter_on_ratio"],
        "strategy_exposure_ratio": engine_metrics["strategy_exposure_ratio"],
    }


def run_backtest(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    stock_symbol: str,
    benchmark_symbol: str,
    ma_mid: int,
    ma_long: int,
    momentum_window: int,
    benchmark_ma: int,
    cash: float,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> tuple[pd.DataFrame, dict]:
    """使用共享回测引擎执行 MA v3 回测。

    金融正确性检查：
    - 次日持仓：position = final_signal.shift(1)，避免未来函数。
    - 交易成本：买入佣金+滑点，卖出佣金+印花税+滑点。
    - 收益计算：strategy_ret = position * stock_ret - cost。

    以上逻辑已委托给 scripts.common.backtest.engine.single_asset_backtest()，
    保证与 Alpha v6/v7 及其他策略版本的回测口径完全一致。

    stock_only 对照路径（趋势确认 only，不加大盘过滤）通过引擎的
    comparison_signal_col 参数自动计算，保证主路径和对照路径口径一致。
    """
    # 1. 计算个股 v3 信号（MA v3 特有逻辑）
    result = compute_v3_signals(stock_df, ma_mid, ma_long, momentum_window)

    # 2. 大盘 regime filter（MA v3 特有逻辑）
    bench_filter = prepare_benchmark_regime(benchmark_df, benchmark_ma)

    # 3. 委托给共享回测引擎（market filter 对齐 + 信号合成 + 持仓/收益/成本 + 权益曲线 + 指标）
    #    comparison_signal_col="trend_confirm" 让引擎同时计算 stock_only 对照路径
    #    （趋势确认 only，不加大盘过滤），消除手动计算。
    result, engine_metrics = single_asset_backtest(
        result=result,
        benchmark_filter_df=bench_filter,
        signal_col="trend_confirm",
        cash=cash,
        commission=commission,
        sell_tax=sell_tax,
        slippage=slippage,
        compute_benchmark_ret=True,
        comparison_signal_col="trend_confirm",
    )

    # 4. 构建 metrics（引擎已包含 stock_only_total_return）
    metrics = _adapt_engine_metrics(engine_metrics)

    print("\n" + "=" * 80)
    print("MA v3：趋势强度 / 动量 + 趋势确认 + 大盘 regime filter")
    print("-" * 80)
    print(f"股票标的: {stock_symbol}")
    print(f"过滤基准: {benchmark_symbol}")
    print(f"MA 参数: ma_mid={ma_mid}, ma_long={ma_long}, momentum_window={momentum_window}")
    print(f"基准 MA: benchmark_ma={benchmark_ma}")
    print(f"数据区间: {result['date'].min().date()} 至 {result['date'].max().date()}")
    print(f"数据行数: {len(result)}")
    print(f"初始资金: {cash:,.2f}")
    print(f"佣金率: {commission:.5f}")
    print(f"卖出印花税: {sell_tax:.5f}")
    print(f"滑点率: {slippage:.5f}")
    print("-" * 80)
    print(f"v3 策略总收益: {metrics['total_return']:.2%}")
    print(f"v3 策略年化收益: {metrics['annual_return']:.2%}")
    print(f"v3 最大回撤: {metrics['max_drawdown']:.2%}")
    print("v3 夏普比率: NaN" if np.isnan(metrics["sharpe"]) else f"v3 夏普比率: {metrics['sharpe']:.4f}")
    print(f"v3 换仓次数: {metrics['trade_count']}")
    print(f"v3 期末权益: {metrics['final_equity']:,.2f}")
    print("-" * 80)
    print(f"v3 对照（趋势确认 only）总收益: {metrics['stock_only_total_return']:.2%}")
    print(f"买入持有总收益: {metrics['buy_hold_total_return']:.2%}")
    print(f"大盘 regime 开启比例: {metrics['market_filter_on_ratio']:.2%}")
    print(f"v3 策略持仓暴露比例: {metrics['strategy_exposure_ratio']:.2%}")
    print("=" * 80)

    return result, metrics


def save_outputs(
    result: pd.DataFrame,
    metrics: dict,
    stock_symbol: str,
    benchmark_symbol: str,
    stock_csv_path: Path,
    benchmark_csv_path: Path,
    ma_mid: int,
    ma_long: int,
    momentum_window: int,
    benchmark_ma: int,
    start: str,
    end: str,
    output_dir: Path,
) -> None:
    safe_stock = stock_symbol.replace(".", "_")
    safe_benchmark = benchmark_symbol.replace(".", "_")

    strategy_name = "ma_v3_momentum_strategy_csv"
    date_tag = f"{start or 'all'}_{end or 'latest'}"

    run_name = (
        f"ma_v3_{safe_stock}_"
        f"mid{ma_mid}_long{ma_long}_mom{momentum_window}_"
        f"bench_{safe_benchmark}_ma{benchmark_ma}_"
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
        f.write(f"ma_mid: {ma_mid}\n")
        f.write(f"ma_long: {ma_long}\n")
        f.write(f"momentum_window: {momentum_window}\n")
        f.write(f"benchmark_ma: {benchmark_ma}\n")
        f.write(f"start_date: {result['date'].min()}\n")
        f.write(f"end_date: {result['date'].max()}\n")
        f.write(f"rows: {len(result)}\n\n")

        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.8f}\n")
            else:
                f.write(f"{key}: {value}\n")

    plt.figure(figsize=(12, 6))
    plt.plot(result["date"], result["equity"], label="MA v3: momentum + trend confirm + regime")
    plt.plot(result["date"], result["stock_only_equity"], label="MA v3: trend confirm only")
    plt.plot(result["date"], result["buy_hold_equity"], label="Buy and Hold")
    plt.title(f"MA v3 Momentum Backtest - {stock_symbol} / {benchmark_symbol}")
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
    parser = argparse.ArgumentParser(description="MA v3：趋势强度 / 动量 + 趋势确认，单标的 CSV 回测")

    parser.add_argument("--stock", default="000001.SZ", help="股票代码，例如 000001.SZ")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="大盘过滤基准（默认沪深 300）")

    parser.add_argument("--ma-mid", type=int, default=60, help="中期均线窗口")
    parser.add_argument("--ma-long", type=int, default=250, help="长期均线窗口")
    parser.add_argument("--momentum-window", type=int, default=120, help="动量计算窗口")
    parser.add_argument("--benchmark-ma", type=int, default=120, help="基准 regime 过滤均线")

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

    if args.ma_mid >= args.ma_long:
        raise ValueError("ma_mid 必须小于 ma_long。")

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

    min_rows = max(args.ma_long, args.momentum_window, int(args.benchmark_ma * 2.5)) + 10
    if len(stock_df) < min_rows:
        raise RuntimeError(f"股票数据太少：rows={len(stock_df)}，至少需要 {min_rows}")

    result, metrics = run_backtest(
        stock_df=stock_df,
        benchmark_df=benchmark_df,
        stock_symbol=stock_symbol,
        benchmark_symbol=benchmark_symbol,
        ma_mid=args.ma_mid,
        ma_long=args.ma_long,
        momentum_window=args.momentum_window,
        benchmark_ma=args.benchmark_ma,
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
        ma_mid=args.ma_mid,
        ma_long=args.ma_long,
        momentum_window=args.momentum_window,
        benchmark_ma=args.benchmark_ma,
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
