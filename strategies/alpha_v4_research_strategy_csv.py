# -*- coding: utf-8 -*-
"""
alpha_v4_research_strategy_csv.py

Alpha v4：下一代 alpha 信号研究模块

支持 4 个 alpha variant：
1. pure_momentum — 不使用 MA trend gate，过去 N 日动量收益作为核心信号
2. simplified_trend_momentum — 简化 MA v3 趋势确认（close > ma_long OR ma_mid > ma_long）
3. volatility_adjusted_momentum — momentum_return / realized_volatility 作为排序核心
4. breakout_momentum — close 突破近 N 日高点作为信号

运行示例：
python strategies\\alpha_v4_research_strategy_csv.py --stock 000001.SZ --benchmark 000300.SH --alpha-variant pure_momentum --momentum-window 120 --trend-ma 250 --vol-window 60 --breakout-window 120 --benchmark-ma 120 --start 20150101
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
from scripts.common.feature_expression import normalize_zscore  # noqa: E402
from scripts.common.benchmark import prepare_benchmark_regime  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backtests" / "alpha_v4_research_strategy_csv"

VALID_VARIANTS = [
    "pure_momentum",
    "simplified_trend_momentum",
    "volatility_adjusted_momentum",
    "breakout_momentum",
]


def compute_alpha_v4_signals(
    stock_df: pd.DataFrame,
    alpha_variant: str,
    momentum_window: int,
    trend_ma: int,
    vol_window: int,
    breakout_window: int,
) -> pd.DataFrame:
    """计算 alpha v4 信号列。"""
    result = stock_df.copy()

    # 公共指标
    result["momentum_return"] = result["close"].pct_change(momentum_window)
    result["realized_volatility"] = result["close"].pct_change().rolling(vol_window).std() * SQRT_TRADING_DAYS_PER_YEAR

    # 用于 simplified_trend_momentum
    ma_long = result["close"].rolling(trend_ma).mean()
    ma_mid = result["close"].rolling(max(20, trend_ma // 4)).mean()
    result["ma_long"] = ma_long
    result["ma_mid"] = ma_mid

    # 用于 breakout_momentum
    result["rolling_high"] = result["close"].rolling(breakout_window).max()
    result["breakout_ratio"] = result["close"] / result["rolling_high"] - 1.0

    # 根据 variant 计算 raw_alpha_score 和 signal
    if alpha_variant == "pure_momentum":
        # 不使用 MA trend gate，纯动量
        result["raw_alpha_score"] = result["momentum_return"]
        result["alpha_signal"] = (result["momentum_return"] > 0).astype(int)

    elif alpha_variant == "simplified_trend_momentum":
        # 简化趋势确认：close > ma_long OR ma_mid > ma_long
        trend_confirm = (
            (result["close"] > result["ma_long"])
            | (result["ma_mid"] > result["ma_long"])
        ).astype(int)
        result["raw_alpha_score"] = (
            normalize_zscore(result["momentum_return"])
            + normalize_zscore(result["close"] / result["ma_long"] - 1.0)
        )
        result["alpha_signal"] = trend_confirm

    elif alpha_variant == "volatility_adjusted_momentum":
        # 波动率调整动量
        vol_adj_mom = result["momentum_return"] / result["realized_volatility"].replace(0, np.nan)
        result["raw_alpha_score"] = vol_adj_mom
        result["alpha_signal"] = (vol_adj_mom > 0).astype(int)

    elif alpha_variant == "breakout_momentum":
        # 突破动量：close 接近或突破 N 日高点
        result["raw_alpha_score"] = result["breakout_ratio"]
        # 信号：close 在 rolling high 的 2% 以内
        result["alpha_signal"] = (result["breakout_ratio"] > -0.02).astype(int)

    else:
        raise ValueError(f"未知的 alpha_variant: {alpha_variant}")

    # 标准化评分（用于排序）
    result["alpha_score"] = normalize_zscore(result["raw_alpha_score"])

    return result




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
        "market_filter_on_ratio": float(result["market_filter"].mean()),
        "strategy_exposure_ratio": float(result["position"].mean()),
    }


def run_backtest(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    stock_symbol: str,
    benchmark_symbol: str,
    alpha_variant: str,
    momentum_window: int,
    trend_ma: int,
    vol_window: int,
    breakout_window: int,
    benchmark_ma: int,
    cash: float,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> tuple[pd.DataFrame, dict]:
    # 计算 alpha v4 信号
    result = compute_alpha_v4_signals(
        stock_df, alpha_variant, momentum_window, trend_ma, vol_window, breakout_window,
    )

    # 计算大盘 regime filter
    bench_filter = prepare_benchmark_regime(benchmark_df, benchmark_ma)
    bench_filter = bench_filter.rename(columns={"close": "benchmark_close"})
    bench_filter = bench_filter.set_index("date", drop=False)
    result = result.set_index("date", drop=False)

    # 对齐大盘信号到股票日期
    result["benchmark_close"] = bench_filter["benchmark_close"].reindex(result.index).ffill()
    result["benchmark_ma_short"] = bench_filter["benchmark_ma_short"].reindex(result.index).ffill()
    result["benchmark_ma_long"] = bench_filter["benchmark_ma_long"].reindex(result.index).ffill()
    result["market_filter"] = bench_filter["market_filter"].reindex(result.index).ffill().fillna(0).astype(int)

    # 最终信号：alpha_signal AND market_filter
    result["final_signal"] = (
        (result["alpha_signal"] == 1)
        & (result["market_filter"] == 1)
    ).astype(int)

    # 次日持仓，避免未来函数
    result["position"] = result["final_signal"].shift(1, fill_value=0).astype(float)
    result["stock_ret"] = result["close"].pct_change().fillna(0)

    # 交易成本
    pos_change = result["position"].diff().fillna(0)
    buy_turnover = pos_change.clip(lower=0)
    sell_turnover = (-pos_change).clip(lower=0)

    result["cost"] = (
        buy_turnover * (commission + slippage)
        + sell_turnover * (commission + sell_tax + slippage)
    )

    result["strategy_ret"] = result["position"] * result["stock_ret"] - result["cost"]

    # 权益曲线
    result["equity"] = cash * (1.0 + result["strategy_ret"]).cumprod()
    result["buy_hold_equity"] = cash * (1.0 + result["stock_ret"]).cumprod()

    # 基准收益和回撤
    result["benchmark_ret"] = result["benchmark_close"].pct_change().fillna(0)
    result["benchmark_equity"] = cash * (1.0 + result["benchmark_ret"]).cumprod()
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1.0

    metrics = calc_metrics(result)

    print("\n" + "=" * 80)
    print(f"Alpha v4 Research: {alpha_variant}")
    print("-" * 80)
    print(f"股票标的: {stock_symbol}")
    print(f"过滤基准: {benchmark_symbol}")
    print(f"Alpha variant: {alpha_variant}")
    print(f"momentum_window={momentum_window}, trend_ma={trend_ma}, vol_window={vol_window}, breakout_window={breakout_window}")
    print(f"基准 MA: benchmark_ma={benchmark_ma}")
    print(f"数据区间: {result.index.min().date()} 至 {result.index.max().date()}")
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
    momentum_window: int,
    trend_ma: int,
    vol_window: int,
    breakout_window: int,
    benchmark_ma: int,
    start: str,
    end: str,
    output_dir: Path,
) -> None:
    safe_stock = stock_symbol.replace(".", "_")
    safe_benchmark = benchmark_symbol.replace(".", "_")

    date_tag = f"{start or 'all'}_{end or 'latest'}"

    run_name = (
        f"alpha_v4_{alpha_variant}_{safe_stock}_"
        f"mom{momentum_window}_tma{trend_ma}_vol{vol_window}_brk{breakout_window}_"
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
        f.write(f"strategy_name: alpha_v4_research_strategy_csv\n")
        f.write(f"stock_symbol: {stock_symbol}\n")
        f.write(f"benchmark_symbol: {benchmark_symbol}\n")
        f.write(f"stock_csv_path: {stock_csv_path}\n")
        f.write(f"benchmark_csv_path: {benchmark_csv_path}\n")
        f.write(f"alpha_variant: {alpha_variant}\n")
        f.write(f"momentum_window: {momentum_window}\n")
        f.write(f"trend_ma: {trend_ma}\n")
        f.write(f"vol_window: {vol_window}\n")
        f.write(f"breakout_window: {breakout_window}\n")
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
    plt.plot(result["date"], result["equity"], label=f"Alpha v4: {alpha_variant}")
    plt.plot(result["date"], result["buy_hold_equity"], label="Buy and Hold")
    plt.plot(result["date"], result["benchmark_equity"], label="Benchmark")
    plt.title(f"Alpha v4 ({alpha_variant}) - {stock_symbol} / {benchmark_symbol}")
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
    parser = argparse.ArgumentParser(description="Alpha v4 Research：下一代 alpha 信号研究，单标的 CSV 回测")

    parser.add_argument("--stock", default="000001.SZ", help="股票代码")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="大盘过滤基准")

    parser.add_argument("--alpha-variant", default="pure_momentum", choices=VALID_VARIANTS, help="Alpha variant")
    parser.add_argument("--momentum-window", type=int, default=120, help="动量计算窗口")
    parser.add_argument("--trend-ma", type=int, default=250, help="趋势均线窗口（simplified_trend_momentum 使用）")
    parser.add_argument("--vol-window", type=int, default=60, help="波动率计算窗口（volatility_adjusted_momentum 使用）")
    parser.add_argument("--breakout-window", type=int, default=120, help="突破窗口（breakout_momentum 使用）")
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

    min_rows = max(args.momentum_window, args.trend_ma, args.breakout_window, args.vol_window, int(args.benchmark_ma * 2.5)) + 10
    if len(stock_df) < min_rows:
        raise RuntimeError(f"股票数据太少：rows={len(stock_df)}，至少需要 {min_rows}")

    result, metrics = run_backtest(
        stock_df=stock_df,
        benchmark_df=benchmark_df,
        stock_symbol=stock_symbol,
        benchmark_symbol=benchmark_symbol,
        alpha_variant=args.alpha_variant,
        momentum_window=args.momentum_window,
        trend_ma=args.trend_ma,
        vol_window=args.vol_window,
        breakout_window=args.breakout_window,
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
        momentum_window=args.momentum_window,
        trend_ma=args.trend_ma,
        vol_window=args.vol_window,
        breakout_window=args.breakout_window,
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
