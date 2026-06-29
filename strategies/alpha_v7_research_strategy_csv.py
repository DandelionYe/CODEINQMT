# -*- coding: utf-8 -*-
"""
alpha_v7_research_strategy_csv.py

Alpha v7：使用 UGQlib 新接口的 alpha 策略

与 Alpha v6 的核心区别：
1. 信号计算使用 feature_expression.py 的表达式层，不再硬编码
2. 因子定义从 configs/factor_registry.json 读取
3. 支持 --evaluate-signals 开关，在回测前先做信号截面评估（IC/RankIC）
4. 保留 Alpha v6 的大盘 regime filter 和回测逻辑

假设：Alpha v6 的 4 类非动量信号（反转、低波动、换手率、量价背离）通过
表达式层可组合、可登记、可复用，为后续 Alpha v7 扩展更多因子打下基础。

支持 4 个 alpha variant（与 Alpha v6 一致）：
1. short_term_reversal — 短期反转
2. low_volatility — 低波动异象
3. turnover_reversal — 换手率反转
4. volume_price_divergence — 量价背离

运行示例：
python strategies\\alpha_v7_research_strategy_csv.py --stock 000001.SZ --benchmark 000300.SH --alpha-variant short_term_reversal --reversal-window 10 --start 20150101

# 回测前先做信号评估
python strategies\\alpha_v7_research_strategy_csv.py --stock 000001.SZ --alpha-variant short_term_reversal --evaluate-signals --start 20150101
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from scripts.common.constants import (  # noqa: E402
    DEFAULT_BENCHMARK,
    SQRT_TRADING_DAYS_PER_YEAR,
)
from scripts.common.backtest.engine import single_asset_backtest  # noqa: E402
from scripts.common.feature_expression import (  # noqa: E402
    Expression,
    Field,
    PctChange,
    RollingMean,
    RollingStd,
    Neg,
    ZScore,
    Mul,
    Sub,
    Const,
    normalize_zscore,
)
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.benchmark import prepare_benchmark_regime  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backtests" / "alpha_v7_research_strategy_csv"
REGISTRY_PATH = PROJECT_ROOT / "configs" / "factor_registry.json"

VALID_VARIANTS = [
    "short_term_reversal",
    "low_volatility",
    "turnover_reversal",
    "volume_price_divergence",
]


# ---------------------------------------------------------------------------
# 表达式构建（程序化，不依赖 registry 文件）
# ---------------------------------------------------------------------------

def build_expression(
    alpha_variant: str,
    reversal_window: int = 10,
    vol_window: int = 60,
    turnover_short: int = 10,
    turnover_long: int = 60,
    divergence_window: int = 20,
) -> Tuple[Expression, Expression]:
    """为指定 variant 构建因子表达式。

    Returns
    -------
    Tuple[Expression, Expression]
        (raw_score_expr, signal_expr)
        raw_score_expr 产出 raw_alpha_score
        signal_expr 产出 alpha_signal (0/1)

    Note
    ----
    本函数是因子表达式构建的运行时权威来源。``configs/factor_registry.json`` 中的
    expression 字段是本函数的文本声明形式，由 ``build_feature_matrix_from_registry()``
    使用。两者在数学上一致（测试验证 rtol=1e-10），但本函数是策略脚本的实际调用路径。
    """
    if alpha_variant == "short_term_reversal":
        # raw_alpha_score = -pct_change(close, N)
        # alpha_signal = (pct_change(close, N) < 0)
        reversal_return = PctChange(Field("close"), reversal_window)
        raw_score_expr = Neg(reversal_return)
        signal_expr = (reversal_return < 0)  # Gt/Lt 返回 int 0/1

    elif alpha_variant == "low_volatility":
        # realized_vol = rolling_std(daily_ret, vol_window) * sqrt(252)
        # raw_alpha_score = -realized_vol
        # alpha_signal = 1 (始终持仓)
        daily_ret = PctChange(Field("close"), 1)
        realized_vol = Mul(RollingStd(daily_ret, vol_window), Const(SQRT_TRADING_DAYS_PER_YEAR))
        raw_score_expr = Neg(realized_vol)
        signal_expr = Const(1)  # 始终持仓

    elif alpha_variant == "turnover_reversal":
        # turnover_ratio = rolling_mean(volume, short) / rolling_mean(volume, long)
        # raw_alpha_score = -(turnover_ratio - 1)
        # alpha_signal = (turnover_ratio < 1)
        turnover_ratio = RollingMean(Field("volume"), turnover_short) / RollingMean(Field("volume"), turnover_long)
        raw_score_expr = Neg(Sub(turnover_ratio, Const(1)))
        signal_expr = (turnover_ratio < 1)

    elif alpha_variant == "volume_price_divergence":
        # price_trend = pct_change(close, N)
        # volume_trend = pct_change(volume, N)
        # raw_alpha_score = zscore(price_trend) - zscore(volume_trend)
        # alpha_signal = (price_trend > 0) & (volume_trend < 0)
        price_trend = PctChange(Field("close"), divergence_window)
        volume_trend = PctChange(Field("volume"), divergence_window)
        raw_score_expr = Sub(ZScore(price_trend), ZScore(volume_trend))
        signal_expr = (price_trend > 0) & (volume_trend < 0)

    else:
        raise ValueError(f"未知的 alpha_variant: {alpha_variant}")

    return raw_score_expr, signal_expr


# ---------------------------------------------------------------------------
# 信号计算（使用表达式层）
# ---------------------------------------------------------------------------

def compute_alpha_v7_signals(
    stock_df: pd.DataFrame,
    alpha_variant: str,
    reversal_window: int,
    vol_window: int,
    turnover_short: int,
    turnover_long: int,
    divergence_window: int,
) -> pd.DataFrame:
    """使用表达式层计算 alpha v7 信号列。

    与 Alpha v6 的 compute_alpha_v6_signals() 输出口径一致：
    - raw_alpha_score：原始评分
    - alpha_score：z-score 标准化评分
    - alpha_signal：0/1 信号
    """
    raw_score_expr, signal_expr = build_expression(
        alpha_variant, reversal_window, vol_window,
        turnover_short, turnover_long, divergence_window,
    )

    result = stock_df.copy()
    result["raw_alpha_score"] = raw_score_expr.eval(result)
    result["alpha_signal"] = signal_expr.eval(result).astype(int)
    result["alpha_score"] = normalize_zscore(result["raw_alpha_score"])

    return result


# ---------------------------------------------------------------------------
# 信号评估（可选）
# ---------------------------------------------------------------------------

def run_signal_evaluation(
    stock_df: pd.DataFrame,
    alpha_variant: str,
    reversal_window: int,
    vol_window: int,
    turnover_short: int,
    turnover_long: int,
    divergence_window: int,
    label_horizons: list,
    output_dir: Path,
) -> None:
    """对单只股票的信号做前瞻收益评估。

    金融正确性：
    - 前瞻收益 ret_Nd = close.shift(-N) / close - 1，是标准 label 做法。
    - alpha_score 在 T 日已知，与 ret_Nd（T+N 日收益）做截面相关。
    - 此评估仅用于信号质量参考，不用于回测决策。
    """
    from scripts.common.signal_evaluation import evaluate_signal, save_signal_evaluation

    signals = compute_alpha_v7_signals(
        stock_df, alpha_variant, reversal_window, vol_window,
        turnover_short, turnover_long, divergence_window,
    )

    # 构建 (date, symbol) MultiIndex（单只股票）
    signals = signals.copy()
    signals["symbol"] = stock_df.get("symbol", "single_stock").iloc[0] if "symbol" in stock_df.columns else "single_stock"

    # 计算前瞻收益
    for h in label_horizons:
        signals[f"ret_{h}d"] = signals["close"].shift(-h) / signals["close"] - 1.0

    # 设置 MultiIndex
    signals["date_idx"] = signals["date"]
    signals = signals.set_index(["date_idx", "symbol"])
    signals.index.names = ["date", "symbol"]

    # 评估每个 horizon
    for h in label_horizons:
        label_col = f"ret_{h}d"
        if label_col not in signals.columns:
            continue

        eval_result = evaluate_signal(
            signals, label_col,
            score_col="alpha_score",
            n_quantiles=5,
        )

        run_info = {
            "alpha_variant": alpha_variant,
            "params": {
                "reversal_window": reversal_window,
                "vol_window": vol_window,
                "turnover_short": turnover_short,
                "turnover_long": turnover_long,
                "divergence_window": divergence_window,
            },
            "strategy_version": "alpha_v7",
        }

        eval_dir = output_dir / "signal_evaluation" / label_col
        save_signal_evaluation(eval_result, eval_dir, label_col, run_info)

        # 打印摘要
        summary = eval_result["ic_summary"]
        if not summary.empty:
            row = summary.iloc[0]
            print(f"\n  Signal Evaluation ({label_col}):")
            print(f"    IC mean:     {row.get('ic_mean', np.nan):.6f}")
            print(f"    RankIC mean: {row.get('rank_ic_mean', np.nan):.6f}")
            print(f"    ICIR:        {row.get('icir', np.nan):.4f}")


# ---------------------------------------------------------------------------
# 回测指标（从引擎 build_backtest_metrics 适配）
# ---------------------------------------------------------------------------

def _adapt_engine_metrics(engine_metrics: dict) -> dict:
    """将引擎返回的 strategy_*/buy_hold_* 前缀指标适配为 Alpha v7 CLI 期望的 key。

    引擎 key → Alpha v7 key:
      strategy_total_return      → total_return
      strategy_annual_return     → annual_return
      strategy_annual_volatility → annual_volatility
      strategy_max_drawdown      → max_drawdown
      strategy_sharpe            → sharpe
      strategy_trade_count       → trade_count
      strategy_final_equity      → final_equity
      buy_hold_total_return      → benchmark_total_return (买入持有 ≈ 基准口径)
    """
    return {
        "total_return": engine_metrics["strategy_total_return"],
        "annual_return": engine_metrics["strategy_annual_return"],
        "annual_volatility": engine_metrics["strategy_annual_volatility"],
        "max_drawdown": engine_metrics["strategy_max_drawdown"],
        "sharpe": engine_metrics["strategy_sharpe"],
        "trade_count": engine_metrics["strategy_trade_count"],
        "final_equity": engine_metrics["strategy_final_equity"],
        "benchmark_total_return": engine_metrics["buy_hold_total_return"],
        "market_filter_on_ratio": engine_metrics["market_filter_on_ratio"],
        "strategy_exposure_ratio": engine_metrics["strategy_exposure_ratio"],
    }


# ---------------------------------------------------------------------------
# 回测主逻辑
# ---------------------------------------------------------------------------

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
) -> Tuple[pd.DataFrame, dict]:
    """使用表达式层计算信号并回测（委托给共享回测引擎）。

    金融正确性检查：
    - 次日持仓：position = final_signal.shift(1)，避免未来函数。
    - 交易成本：买入佣金+滑点，卖出佣金+印花税+滑点。
    - 收益计算：strategy_ret = position * stock_ret - cost。

    以上逻辑已委托给 scripts.common.backtest.engine.single_asset_backtest()，
    保证与 Alpha v6 及其他策略版本的回测口径完全一致。
    """
    # 1. 使用表达式层计算信号（v7 特有逻辑）
    result = compute_alpha_v7_signals(
        stock_df, alpha_variant, reversal_window, vol_window,
        turnover_short, turnover_long, divergence_window,
    )

    # 2. 大盘 regime filter（v7 特有逻辑）
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
    print(f"Alpha v7 (expression-based): {alpha_variant}")
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
    print(f"基准总收益: {metrics['benchmark_total_return']:.2%}")
    print(f"大盘 regime 开启比例: {metrics['market_filter_on_ratio']:.2%}")
    print(f"策略持仓暴露比例: {metrics['strategy_exposure_ratio']:.2%}")
    print("=" * 80)

    return result, metrics


# ---------------------------------------------------------------------------
# 保存输出
# ---------------------------------------------------------------------------

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
) -> Path:
    """保存回测结果。"""
    safe_stock = stock_symbol.replace(".", "_")
    safe_benchmark = benchmark_symbol.replace(".", "_")
    date_tag = f"{start or 'all'}_{end or 'latest'}"

    run_name = (
        f"alpha_v7_{alpha_variant}_{safe_stock}_"
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
        f.write(f"strategy_name: alpha_v7_research_strategy_csv\n")
        f.write(f"signal_source: expression_layer\n")
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
    plt.plot(result["date"], result["equity"], label=f"Alpha v7: {alpha_variant}")
    plt.plot(result["date"], result["buy_hold_equity"], label="Buy and Hold")
    plt.plot(result["date"], result["benchmark_equity"], label="Benchmark")
    plt.title(f"Alpha v7 ({alpha_variant}) - {stock_symbol} / {benchmark_symbol}")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    logger.info("结果已保存到：%s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha v7 Research：使用 UGQlib 表达式层的 alpha 策略，单标的 CSV 回测",
    )

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

    parser.add_argument("--evaluate-signals", action="store_true",
                        help="回测前先做信号截面评估（IC/RankIC）")
    parser.add_argument("--label-horizons", default="1,5,20",
                        help="信号评估的前瞻收益期限（逗号分隔天数）")

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
    logger.info("stock_csv_path: %s", stock_csv_path)
    logger.info("benchmark_symbol: %s", benchmark_symbol)
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

    # 可选：信号评估
    if args.evaluate_signals:
        label_horizons = [int(h.strip()) for h in args.label_horizons.split(",")]
        run_signal_evaluation(
            stock_df=stock_df,
            alpha_variant=args.alpha_variant,
            reversal_window=args.reversal_window,
            vol_window=args.vol_window,
            turnover_short=args.turnover_short,
            turnover_long=args.turnover_long,
            divergence_window=args.divergence_window,
            label_horizons=label_horizons,
            output_dir=output_dir,
        )

    # 回测
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
