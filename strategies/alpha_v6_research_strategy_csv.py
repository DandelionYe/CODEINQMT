# -*- coding: utf-8 -*-
"""
alpha_v6_research_strategy_csv.py

Alpha v6：真正不同的 alpha 来源

假设：Alpha v4/v5 失败是因为所有 variant 都是动量类信号，高度同质化。
Alpha v6 使用真正不同的 alpha 来源 — 反转、低波动、换手率、量价背离。

支持 4 个 alpha variant：
1. short_term_reversal — 短期反转
2. low_volatility — 低波动异象
3. turnover_reversal — 换手率反转
4. volume_price_divergence — 量价背离

运行示例：
python strategies\\alpha_v6_research_strategy_csv.py --stock 000001.SZ --benchmark 000300.SH --alpha-variant short_term_reversal --reversal-window 10 --vol-window 60 --benchmark-ma 120 --start 20150101
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
from scripts.common.constants import SQRT_TRADING_DAYS_PER_YEAR, DEFAULT_BENCHMARK  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.feature_expression import normalize_zscore  # noqa: E402
from scripts.common.backtest.engine import single_asset_backtest  # noqa: E402
from scripts.common.benchmark import prepare_benchmark_regime  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backtests" / "alpha_v6_research_strategy_csv"

VALID_VARIANTS = [
    "short_term_reversal",
    "low_volatility",
    "turnover_reversal",
    "volume_price_divergence",
]


def compute_alpha_v6_signals(
    stock_df: pd.DataFrame,
    alpha_variant: str,
    reversal_window: int,
    vol_window: int,
    turnover_short: int,
    turnover_long: int,
    divergence_window: int,
) -> pd.DataFrame:
    """计算 alpha v6 信号列。

    4 个 variant 使用真正不同的 alpha 来源：反转、低波动、换手率、量价背离。
    """
    result = stock_df.copy()

    # 公共指标（各 variant 按需使用）
    result["reversal_return"] = result["close"].pct_change(reversal_window)
    result["realized_vol"] = result["close"].pct_change().rolling(vol_window).std() * SQRT_TRADING_DAYS_PER_YEAR
    result["turnover_ratio"] = (
        result["volume"].rolling(turnover_short).mean()
        / result["volume"].rolling(turnover_long).mean().replace(0, np.nan)
    )
    result["price_trend"] = result["close"].pct_change(divergence_window)
    result["volume_trend"] = result["volume"].pct_change(divergence_window)

    if alpha_variant == "short_term_reversal":
        # 短期反转：买跌卖涨
        result["raw_alpha_score"] = -result["reversal_return"]
        result["alpha_signal"] = (result["reversal_return"] < 0).astype(int)

    elif alpha_variant == "low_volatility":
        # 低波动异象：低波动得分高，始终持仓靠排序选股
        result["raw_alpha_score"] = -result["realized_vol"]
        result["alpha_signal"] = 1

    elif alpha_variant == "turnover_reversal":
        # 换手率反转：换手率下降得高分
        result["raw_alpha_score"] = -(result["turnover_ratio"] - 1)
        result["alpha_signal"] = (result["turnover_ratio"] < 1).astype(int)

    elif alpha_variant == "volume_price_divergence":
        # 量价背离：价涨量跌
        result["raw_alpha_score"] = (
            normalize_zscore(result["price_trend"]) - normalize_zscore(result["volume_trend"])
        )
        result["alpha_signal"] = (
            (result["price_trend"] > 0) & (result["volume_trend"] < 0)
        ).astype(int)

    else:
        raise ValueError(f"未知的 alpha_variant: {alpha_variant}")

    # 标准化评分（用于排序）
    result["alpha_score"] = normalize_zscore(result["raw_alpha_score"])

    return result


def _adapt_engine_metrics(engine_metrics: dict) -> dict:
    """将引擎返回的 strategy_*/buy_hold_* 前缀指标适配为 Alpha v6 CLI 期望的 key。

    引擎 key → Alpha v6 key:
      strategy_total_return      → total_return
      strategy_annual_return     → annual_return
      strategy_annual_volatility → annual_volatility
      strategy_max_drawdown      → max_drawdown
      strategy_sharpe            → sharpe
      strategy_trade_count       → trade_count
      strategy_final_equity      → final_equity
      buy_hold_total_return      → buy_hold_total_return
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
        "market_filter_on_ratio": engine_metrics["market_filter_on_ratio"],
        "strategy_exposure_ratio": engine_metrics["strategy_exposure_ratio"],
    }


def run_backtest(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    stock_symbol: str,
    benchmark_symbol: str,
    alpha_variant: str,
    reversal_window: int,
    vol_window: int,
    turnover_short: int,
    turnover_long: int,
    divergence_window: int,
    benchmark_ma: int,
    cash: float,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> tuple[pd.DataFrame, dict]:
    """使用硬编码信号计算并回测（委托给共享回测引擎）。

    金融正确性检查：
    - 次日持仓：position = final_signal.shift(1)，避免未来函数。
    - 交易成本：买入佣金+滑点，卖出佣金+印花税+滑点。
    - 收益计算：strategy_ret = position * stock_ret - cost。

    以上逻辑已委托给 scripts.common.backtest.engine.single_asset_backtest()，
    保证与 Alpha v7 及其他策略版本的回测口径完全一致。
    """
    # 1. 计算 alpha v6 信号（v6 特有逻辑）
    result = compute_alpha_v6_signals(
        stock_df, alpha_variant, reversal_window, vol_window,
        turnover_short, turnover_long, divergence_window,
    )

    # 2. 大盘 regime filter（v6 特有逻辑）
    bench_filter = prepare_benchmark_regime(benchmark_df, benchmark_ma)

    # 3. 委托给共享回测引擎（market filter 对齐 + 信号合成 + 持仓/收益/成本 + 权益曲线 + 指标）
    result, engine_metrics = single_asset_backtest(
        result=result,
        benchmark_filter_df=bench_filter,
        signal_col="alpha_signal",
        cash=cash,
        commission=commission,
        sell_tax=sell_tax,
        slippage=slippage,
        compute_benchmark_ret=True,
    )

    metrics = _adapt_engine_metrics(engine_metrics)

    print("\n" + "=" * 80)
    print(f"Alpha v6 Research: {alpha_variant}")
    print("-" * 80)
    print(f"股票标的: {stock_symbol}")
    print(f"过滤基准: {benchmark_symbol}")
    print(f"Alpha variant: {alpha_variant}")
    print(f"reversal_window={reversal_window}, vol_window={vol_window}")
    print(f"turnover_short={turnover_short}, turnover_long={turnover_long}, divergence_window={divergence_window}")
    print(f"基准 MA: benchmark_ma={benchmark_ma}")
    print(f"数据区间: {result['date'].min().date()} 至 {result['date'].max().date()}")
    print(f"数据行数: {len(result)}")
    print(f"初始资金: {cash:,.2f}")
    print(f"佣金率: {commission:.5f}")
    print(f"卖出印花税: {sell_tax:.5f}")
    print(f"滑点率: {slippage:.5f}")
    print("-" * 80)
    print(f"策略总收益: {metrics['total_return']:.2%}")
    print(f"策略年化收益: {metrics['annual_return']:.2%}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")
    print("夏普比率: NaN" if np.isnan(metrics["sharpe"]) else f"夏普比率: {metrics['sharpe']:.4f}")
    print(f"换仓次数: {metrics['trade_count']}")
    print(f"期末权益: {metrics['final_equity']:,.2f}")
    print("-" * 80)
    print(f"买入持有总收益: {metrics['buy_hold_total_return']:.2%}")
    print(f"大盘 regime 开启比例: {metrics['market_filter_on_ratio']:.2%}")
    print(f"策略持仓暴露比例: {metrics['strategy_exposure_ratio']:.2%}")
    print("=" * 80)

    return result, metrics


def save_outputs(
    result: pd.DataFrame,
    metrics: dict,
    stock_symbol: str,
    benchmark_symbol: str,
    stock_csv_path: Path,
    benchmark_csv_path: Path,
    alpha_variant: str,
    reversal_window: int,
    vol_window: int,
    turnover_short: int,
    turnover_long: int,
    divergence_window: int,
    benchmark_ma: int,
    start: str,
    end: str,
    output_dir: Path,
) -> None:
    safe_stock = stock_symbol.replace(".", "_")
    safe_benchmark = benchmark_symbol.replace(".", "_")

    date_tag = f"{start or 'all'}_{end or 'latest'}"

    run_name = (
        f"alpha_v6_{alpha_variant}_{safe_stock}_"
        f"rev{reversal_window}_vol{vol_window}_ts{turnover_short}_tl{turnover_long}_div{divergence_window}_"
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
        f.write(f"strategy_name: alpha_v6_research_strategy_csv\n")
        f.write(f"stock_symbol: {stock_symbol}\n")
        f.write(f"benchmark_symbol: {benchmark_symbol}\n")
        f.write(f"stock_csv_path: {stock_csv_path}\n")
        f.write(f"benchmark_csv_path: {benchmark_csv_path}\n")
        f.write(f"alpha_variant: {alpha_variant}\n")
        f.write(f"reversal_window: {reversal_window}\n")
        f.write(f"vol_window: {vol_window}\n")
        f.write(f"turnover_short: {turnover_short}\n")
        f.write(f"turnover_long: {turnover_long}\n")
        f.write(f"divergence_window: {divergence_window}\n")
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
    plt.plot(result["date"], result["equity"], label=f"Alpha v6: {alpha_variant}")
    plt.plot(result["date"], result["buy_hold_equity"], label="Buy and Hold")
    plt.plot(result["date"], result["benchmark_equity"], label="Benchmark")
    plt.title(f"Alpha v6 ({alpha_variant}) - {stock_symbol} / {benchmark_symbol}")
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
    parser = argparse.ArgumentParser(description="Alpha v6 Research：真正不同的 alpha 来源，单标的 CSV 回测")

    parser.add_argument("--stock", default="000001.SZ", help="股票代码")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="大盘过滤基准")

    parser.add_argument("--alpha-variant", default="short_term_reversal", choices=VALID_VARIANTS, help="Alpha variant")
    parser.add_argument("--reversal-window", type=int, default=10, help="短期反转窗口")
    parser.add_argument("--vol-window", type=int, default=60, help="波动率计算窗口")
    parser.add_argument("--turnover-short", type=int, default=10, help="换手率短期均值窗口")
    parser.add_argument("--turnover-long", type=int, default=60, help="换手率长期均值窗口")
    parser.add_argument("--divergence-window", type=int, default=20, help="量价背离计算窗口")
    parser.add_argument("--benchmark-ma", type=int, default=120, help="基准 regime 过滤均线")

    parser.add_argument("--start", default="20150101", help="开始日期")
    parser.add_argument("--end", default="", help="结束日期")

    parser.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0001, help="佣金率")
    parser.add_argument("--sell-tax", type=float, default=0.0005, help="卖出印花税")
    parser.add_argument("--slippage", type=float, default=0.0, help="滑点率")

    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT), help="QMT CSV 导出根目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="回测输出目录")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    stock_csv_path, stock_symbol, _, stock_security_type = ma.find_csv_for_stock(
        stock=args.stock,
        export_root=export_root,
    )

    benchmark_csv_path, benchmark_symbol, _, benchmark_security_type = ma.find_csv_for_stock(
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

    min_rows = max(args.reversal_window, args.vol_window, args.turnover_long, args.divergence_window, int(args.benchmark_ma * 2.5)) + 10
    if len(stock_df) < min_rows:
        raise RuntimeError(f"股票数据太少：rows={len(stock_df)}，至少需要 {min_rows}")

    result, metrics = run_backtest(
        stock_df=stock_df,
        benchmark_df=benchmark_df,
        stock_symbol=stock_symbol,
        benchmark_symbol=benchmark_symbol,
        alpha_variant=args.alpha_variant,
        reversal_window=args.reversal_window,
        vol_window=args.vol_window,
        turnover_short=args.turnover_short,
        turnover_long=args.turnover_long,
        divergence_window=args.divergence_window,
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
        alpha_variant=args.alpha_variant,
        reversal_window=args.reversal_window,
        vol_window=args.vol_window,
        turnover_short=args.turnover_short,
        turnover_long=args.turnover_long,
        divergence_window=args.divergence_window,
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
