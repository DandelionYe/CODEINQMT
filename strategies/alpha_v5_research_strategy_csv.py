# -*- coding: utf-8 -*-
"""
alpha_v5_research_strategy_csv.py

Alpha v5：信号多样化研究模块

假设：Alpha v4 的失败根因是 4 个 variant 全部是动量类信号，高度同质化。
Alpha v5 测试信号多样化 — 混合不同 alpha 来源，减少单一年份依赖。

支持 4 个 alpha variant：
1. momentum_reversion_blend — 混合动量与均值回归
2. adaptive_momentum — 自适应动量窗口（根据近期波动率调整）
3. multi_timeframe_momentum — 多时间尺度动量组合
4. volatility_regime_momentum — 波动率状态自适应（高波均值回归，低波动量）

运行示例：
python strategies\\alpha_v5_research_strategy_csv.py --stock 000001.SZ --benchmark 000300.SH --alpha-variant momentum_reversion_blend --momentum-window 120 --reversion-window 20 --mom-short 60 --mom-long 250 --vol-window 60 --benchmark-ma 120 --start 20150101
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

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backtests" / "alpha_v5_research_strategy_csv"

VALID_VARIANTS = [
    "momentum_reversion_blend",
    "adaptive_momentum",
    "multi_timeframe_momentum",
    "volatility_regime_momentum",
]


def compute_alpha_v5_signals(
    stock_df: pd.DataFrame,
    alpha_variant: str,
    momentum_window: int,
    vol_window: int,
    reversion_window: int,
    mom_short: int,
    mom_mid: int,
    mom_long: int,
) -> pd.DataFrame:
    """计算 alpha v5 信号列。

    4 个 variant 实现信号多样化：混合动量、均值回归、自适应窗口、波动率状态切换。
    """
    result = stock_df.copy()

    # 公共指标
    result["momentum_return"] = result["close"].pct_change(momentum_window)
    result["realized_vol"] = result["close"].pct_change().rolling(vol_window).std() * SQRT_TRADING_DAYS_PER_YEAR

    # 短期均线（用于均值回归）
    result["sma_short"] = result["close"].rolling(reversion_window).mean()
    result["close_over_sma"] = result["close"] / result["sma_short"] - 1.0

    # 多时间尺度动量
    result["mom_60"] = result["close"].pct_change(mom_short)
    result["mom_mid"] = result["close"].pct_change(mom_mid)
    result["mom_250"] = result["close"].pct_change(mom_long)

    # 波动率状态指标
    vol_rolling_mean = result["realized_vol"].rolling(252).mean()
    vol_ratio = result["realized_vol"] / vol_rolling_mean.replace(0, np.nan)
    vol_median = result["realized_vol"].rolling(252).median()

    if alpha_variant == "momentum_reversion_blend":
        # 混合动量与均值回归
        # 动量强 + 短期未过度偏离均线的股票更稳健
        result["raw_alpha_score"] = (
            normalize_zscore(result["momentum_return"])
            - normalize_zscore(result["close_over_sma"])
        )
        result["alpha_signal"] = 1  # 始终持仓，靠排序区分优劣

    elif alpha_variant == "adaptive_momentum":
        # 自适应动量窗口：根据近期波动率调整
        # 不同波动率环境下最优动量窗口不同
        adaptive_window = (momentum_window / vol_ratio).clip(20, 250)
        # 避免 NaN 参与 int 转换
        adaptive_window = adaptive_window.fillna(momentum_window)
        # 用 Int64 支持 NaN；但 fillna 后不再有 NaN，直接 astype 即可
        adaptive_window = adaptive_window.astype(int)

        # 对每行计算自适应窗口的动量收益
        close_series = result["close"]
        adaptive_mom = pd.Series(np.nan, index=result.index, dtype=float)
        # 按唯一窗口值分组，每组做一次 shift
        for w in sorted(adaptive_window.dropna().astype(int).unique()):
            if w > 0:
                mask = adaptive_window == w
                shifted = close_series.shift(w)
                valid = (shifted != 0) & shifted.notna()
                adaptive_mom.loc[mask & valid] = close_series.loc[mask & valid] / shifted.loc[mask & valid] - 1.0

        result["adaptive_momentum_return"] = adaptive_mom
        result["raw_alpha_score"] = adaptive_mom
        result["alpha_signal"] = (adaptive_mom > 0).astype(int)

    elif alpha_variant == "multi_timeframe_momentum":
        # 多时间尺度动量组合
        # 多时间尺度一致看多的股票更可靠
        result["raw_alpha_score"] = (
            normalize_zscore(result["mom_60"])
            + normalize_zscore(result["mom_mid"])
            + normalize_zscore(result["mom_250"])
        )
        # 短中期都看多才持仓
        result["alpha_signal"] = (
            (result["mom_60"] > 0) & (result["mom_mid"] > 0)
        ).astype(int)

    elif alpha_variant == "volatility_regime_momentum":
        # 波动率状态自适应
        # 高波动市场均值回归更有效，低波动市场动量更有效
        is_high_vol = result["realized_vol"] > vol_median
        # 高波状态：均值回归信号（close/sma_short - 1 < 0 意味着价格低于均线，预期反弹）
        # 低波状态：动量信号（momentum_return > 0 意味着趋势延续）
        result["alpha_signal"] = np.where(
            is_high_vol,
            (result["close_over_sma"] < 0).astype(int),
            (result["momentum_return"] > 0).astype(int),
        )
        # raw_alpha_score：高波时用 -zscore(close/sma_short - 1)，低波时用 zscore(momentum_return)
        # 越低的 close_over_sma（负值越大）对应越高的 -zscore，排序靠前
        result["raw_alpha_score"] = np.where(
            is_high_vol,
            -normalize_zscore(result["close_over_sma"]),
            normalize_zscore(result["momentum_return"]),
        )

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
    vol_window: int,
    benchmark_ma: int,
    reversion_window: int,
    mom_short: int,
    mom_mid: int,
    mom_long: int,
    cash: float,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> tuple[pd.DataFrame, dict]:
    # 计算 alpha v5 信号
    result = compute_alpha_v5_signals(
        stock_df, alpha_variant, momentum_window, vol_window,
        reversion_window, mom_short, mom_mid, mom_long,
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
    print(f"Alpha v5 Research: {alpha_variant}")
    print("-" * 80)
    print(f"股票标的: {stock_symbol}")
    print(f"过滤基准: {benchmark_symbol}")
    print(f"Alpha variant: {alpha_variant}")
    print(f"momentum_window={momentum_window}, vol_window={vol_window}")
    print(f"reversion_window={reversion_window}, mom_short={mom_short}, mom_mid={mom_mid}, mom_long={mom_long}")
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
    vol_window: int,
    benchmark_ma: int,
    reversion_window: int,
    mom_short: int,
    mom_mid: int,
    mom_long: int,
    start: str,
    end: str,
    output_dir: Path,
) -> None:
    safe_stock = stock_symbol.replace(".", "_")
    safe_benchmark = benchmark_symbol.replace(".", "_")

    date_tag = f"{start or 'all'}_{end or 'latest'}"

    run_name = (
        f"alpha_v5_{alpha_variant}_{safe_stock}_"
        f"mom{momentum_window}_rev{reversion_window}_mshort{mom_short}_mmid{mom_mid}_mlong{mom_long}_vol{vol_window}_"
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
        f.write(f"strategy_name: alpha_v5_research_strategy_csv\n")
        f.write(f"stock_symbol: {stock_symbol}\n")
        f.write(f"benchmark_symbol: {benchmark_symbol}\n")
        f.write(f"stock_csv_path: {stock_csv_path}\n")
        f.write(f"benchmark_csv_path: {benchmark_csv_path}\n")
        f.write(f"alpha_variant: {alpha_variant}\n")
        f.write(f"momentum_window: {momentum_window}\n")
        f.write(f"reversion_window: {reversion_window}\n")
        f.write(f"mom_short: {mom_short}\n")
        f.write(f"mom_mid: {mom_mid}\n")
        f.write(f"mom_long: {mom_long}\n")
        f.write(f"vol_window: {vol_window}\n")
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
    plt.plot(result["date"], result["equity"], label=f"Alpha v5: {alpha_variant}")
    plt.plot(result["date"], result["buy_hold_equity"], label="Buy and Hold")
    plt.plot(result["date"], result["benchmark_equity"], label="Benchmark")
    plt.title(f"Alpha v5 ({alpha_variant}) - {stock_symbol} / {benchmark_symbol}")
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
    parser = argparse.ArgumentParser(description="Alpha v5 Research：信号多样化 alpha 信号研究，单标的 CSV 回测")

    parser.add_argument("--stock", default="000001.SZ", help="股票代码")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="大盘过滤基准")

    parser.add_argument("--alpha-variant", default="momentum_reversion_blend", choices=VALID_VARIANTS, help="Alpha variant")
    parser.add_argument("--momentum-window", type=int, default=120, help="动量计算窗口")
    parser.add_argument("--reversion-window", type=int, default=20, help="均值回归短期均线窗口")
    parser.add_argument("--mom-short", type=int, default=60, help="短动量窗口")
    parser.add_argument("--mom-mid", type=int, default=120, help="中动量窗口")
    parser.add_argument("--mom-long", type=int, default=250, help="长动量窗口")
    parser.add_argument("--vol-window", type=int, default=60, help="波动率计算窗口")
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

    min_rows = max(args.momentum_window, args.reversion_window, args.mom_short, args.mom_mid, args.mom_long, args.vol_window, int(args.benchmark_ma * 2.5)) + 10
    if len(stock_df) < min_rows:
        raise RuntimeError(f"股票数据太少：rows={len(stock_df)}，至少需要 {min_rows}")

    result, metrics = run_backtest(
        stock_df=stock_df,
        benchmark_df=benchmark_df,
        stock_symbol=stock_symbol,
        benchmark_symbol=benchmark_symbol,
        alpha_variant=args.alpha_variant,
        momentum_window=args.momentum_window,
        vol_window=args.vol_window,
        benchmark_ma=args.benchmark_ma,
        reversion_window=args.reversion_window,
        mom_short=args.mom_short,
        mom_mid=args.mom_mid,
        mom_long=args.mom_long,
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
        vol_window=args.vol_window,
        benchmark_ma=args.benchmark_ma,
        reversion_window=args.reversion_window,
        mom_short=args.mom_short,
        mom_mid=args.mom_mid,
        mom_long=args.mom_long,
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
