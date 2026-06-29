# -*- coding: utf-8 -*-
"""
portfolio_backtest_csv.py

真实组合约束回测模块。将 walk-forward 样本外选股结果转换为带真实交易约束的组合层回测。

核心区别 vs walk-forward：
- walk-forward 假设每日等权再平衡、无交易成本、无整手约束
- 本脚本引入真实交易约束：整手、佣金、滑点、信号驱动调仓

信号逻辑复用 walk-forward 的 alpha v4 信号 + market filter，
确保 constrained portfolio 与 walk-forward 使用相同的持仓决策。

输入：walk-forward CSV（selected_by_year, portfolio_daily）
输出：6 个 CSV + 1 个 TXT 报告 + 可选 PNG 图表

运行示例：
python scripts/portfolio_backtest_csv.py --input-tag <tag> --run-id exp004_alpha_v4_full_smoke --no-png
python scripts/portfolio_backtest_csv.py --input-tag ALL --file-prefix wf_alpha_v7_stock --walk-forward-dir backtests/walk_forward_alpha_v7_research_csv --run-id exp_007 --no-png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

logger = logging.getLogger(__name__)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma
from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR  # noqa: E402
from scripts.common.data_io import safe_to_numeric, read_csv_required  # noqa: E402
from scripts.common.metrics import format_pct, format_float, calc_portfolio_metrics  # noqa: E402
from scripts.common.backtest.portfolio import (  # noqa: E402
    run_yearly_rebalance_backtest,
    build_period_summary,
    build_vs_walkforward,
)
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.validation import resolve_path  # noqa: E402

DEFAULT_WF_DIR = PROJECT_ROOT / "backtests" / "walk_forward_alpha_v4_research_csv"
DEFAULT_PRICE_ROOT = PROJECT_ROOT / "data" / "qmt_export"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "backtests" / "portfolio_backtest_csv"


# ---------------------------------------------------------------------------
# Helpers (format_pct, format_float, calc_portfolio_metrics 已迁移到
# scripts.common.metrics，通过顶部 import 导入)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_price_data(csv_path: Path, start: str, end: str) -> pd.DataFrame | None:
    """Load price data using the same logic as the strategy module."""
    try:
        df = ma.load_qmt_price_csv(csv_path, start, end)
        return df if len(df) > 0 else None
    except Exception:
        return None


def load_benchmark_data(benchmark_csv_path: str | Path) -> pd.DataFrame | None:
    """Load benchmark price data."""
    p = Path(benchmark_csv_path)
    if not p.exists():
        return None
    try:
        df = ma.load_qmt_price_csv(p, "20150101", "20261231")
        return df if len(df) > 0 else None
    except Exception:
        return None


def load_input_files(wf_dir: Path, input_tag: str, file_prefix: str = "wf_alpha_v4_stock") -> dict[str, pd.DataFrame]:
    prefix = f"{file_prefix}_{input_tag}"
    required = {
        "selected_by_year": f"{prefix}_selected_by_year.csv",
        "portfolio_daily": f"{prefix}_portfolio_daily.csv",
    }
    optional = {
        "test_detail": f"{prefix}_test_detail.csv",
        "portfolio_period_summary": f"{prefix}_portfolio_period_summary.csv",
    }
    missing = []
    data = {}
    for key, filename in required.items():
        path = wf_dir / filename
        if not path.exists():
            missing.append(str(path))
        else:
            data[key] = read_csv_required(path)
    if missing:
        raise FileNotFoundError("缺少以下必需文件:\n" + "\n".join(f"  - {p}" for p in missing))
    for key, filename in optional.items():
        path = wf_dir / filename
        if path.exists():
            data[key] = read_csv_required(path)
    return data


def infer_file_prefix(wf_dir: Path, input_tag: str) -> str:
    """从 walk-forward 目录自动推导文件前缀。

    扫描目录中匹配 *_{input_tag}_selected_by_year.csv 的文件，
    提取前缀部分。如果有多个匹配，取最新的。
    如果无匹配，回退到 wf_alpha_v4_stock。
    """
    pattern = f"*_{input_tag}_selected_by_year.csv"
    matches = sorted(wf_dir.glob(pattern))
    if matches:
        # filename: wf_alpha_v7_stock_ALL_selected_by_year.csv
        name = matches[-1].name  # 取最新
        suffix = f"_{input_tag}_selected_by_year.csv"
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return "wf_alpha_v4_stock"


# ---------------------------------------------------------------------------
# Price data preparation
# ---------------------------------------------------------------------------

def load_selected_price_data(selected: pd.DataFrame, price_root: Path) -> dict[str, pd.DataFrame]:
    """Load price data for every selected symbol.

    The portfolio layer consumes walk-forward selected_by_year directly. It does
    not recompute alpha signals; selected_by_year is treated as the target
    rebalance list for each test year.
    """
    price_data: dict[str, pd.DataFrame] = {}
    first_row_by_symbol = selected.drop_duplicates("symbol")

    for _, row in first_row_by_symbol.iterrows():
        symbol = row["symbol"]
        csv_path = row.get("csv_path", "")
        if csv_path and Path(csv_path).exists():
            stock_df = load_price_data(Path(csv_path), "20150101", "20261231")
        else:
            parts = symbol.split(".")
            if len(parts) == 2:
                code, market = parts
                stock_df = load_price_data(price_root / market / f"price_{code}.csv", "20150101", "20261231")
            else:
                stock_df = None

        if stock_df is None or stock_df.empty:
            logger.warning("Missing price data, skipped: %s", symbol)
            continue

        stock_df = stock_df.reset_index(drop=True)
        keep_cols = [c for c in ["date", "open", "close"] if c in stock_df.columns]
        stock_df = stock_df[keep_cols].copy()
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        stock_df = safe_to_numeric(stock_df, ["open", "close"])
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        price_data[symbol] = stock_df

    return price_data


# ---------------------------------------------------------------------------
# Metrics & summaries
# ---------------------------------------------------------------------------
# run_yearly_rebalance_backtest, build_period_summary, build_vs_walkforward
# 已迁移到 scripts.common.backtest.portfolio，通过顶部 import 导入。
#
# get_price_on_date, get_price_on_or_after, _legacy_signal_driven_backtest
# 已删除（deprecated 死代码，功能已迁移到 scripts.common.backtest.portfolio）。


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(output_dir: Path, args, daily_df, trades_df, rebalance_df, period_df, vs_wf_df, test_years) -> Path:
    path = output_dir / "portfolio_backtest_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("Portfolio Backtest Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. Run Settings\n")
        f.write("-" * 40 + "\n")
        f.write(f"  input_tag: {args.input_tag}\n")
        f.write(f"  run_id: {args.run_id}\n")
        f.write(f"  test_years: {test_years}\n")
        f.write(f"  trading_days: {len(daily_df)}\n\n")

        f.write("2. Input Files\n")
        f.write("-" * 40 + "\n")
        f.write(f"  walk_forward_dir: {args.walk_forward_dir}\n")
        f.write(f"  file_prefix: {getattr(args, 'file_prefix', 'auto')}\n")
        f.write(f"  price_root: {args.price_root}\n\n")

        f.write("3. Constraint Settings\n")
        f.write("-" * 40 + "\n")
        f.write(f"  initial_cash: {args.initial_cash:,.0f}\n")
        f.write(f"  max_positions: {args.max_positions}\n")
        f.write(f"  max_weight: {args.max_weight:.2%}\n")
        f.write(f"  lot_size: {args.lot_size}\n")
        f.write(f"  commission_rate: {args.commission_rate:.4%}\n")
        f.write(f"  min_commission: {args.min_commission:.0f}\n")
        f.write(f"  slippage_bps: {args.slippage_bps}\n")
        f.write(f"  price_field: {args.price_field}\n")
        f.write(f"  allow_partial_fill: {args.allow_partial_fill}\n")
        f.write("  rebalance_mode: yearly selected_by_year target portfolio\n")
        f.write("  signal_mode: uses walk-forward selected_by_year outputs; does not recompute intrayear alpha signals\n\n")

        f.write("4. Portfolio Performance\n")
        f.write("-" * 40 + "\n")
        overall = period_df[period_df["period"] == "overall"]
        if not overall.empty:
            o = overall.iloc[0]
            f.write(f"  Total return: {format_pct(o['total_return'])}\n")
            f.write(f"  Annual return: {format_pct(o['annual_return'])}\n")
            f.write(f"  Annual volatility: {format_pct(o['annual_volatility'])}\n")
            f.write(f"  Max drawdown: {format_pct(o['max_drawdown'])}\n")
            f.write(f"  Sharpe: {format_float(o['sharpe'])}\n")
            f.write(f"  Calmar: {format_float(o['calmar'])}\n")
            f.write(f"  Turnover: {format_float(o.get('turnover', np.nan))}\n\n")
        f.write("  Yearly breakdown:\n")
        for _, row in period_df[period_df["period"] != "overall"].iterrows():
            f.write(f"    {row['period']}: return={format_pct(row['total_return'])}, "
                    f"sharpe={format_float(row['sharpe'])}, mdd={format_pct(row['max_drawdown'])}\n")
        f.write("\n")

        f.write("5. Trading Cost Summary\n")
        f.write("-" * 40 + "\n")
        total_commission = trades_df["commission"].sum() if not trades_df.empty else 0
        total_slippage = trades_df["slippage_cost"].sum() if not trades_df.empty else 0
        total_trades = len(trades_df) if not trades_df.empty else 0
        f.write(f"  Total trades: {total_trades}\n")
        f.write(f"  Total commission: {total_commission:,.2f}\n")
        f.write(f"  Total slippage: {total_slippage:,.2f}\n")
        f.write(f"  Total cost: {total_commission + total_slippage:,.2f}\n")
        total_notional = trades_df["notional"].abs().sum() if not trades_df.empty else 0
        f.write(f"  Total notional traded: {total_notional:,.2f}\n")
        final_equity = daily_df["equity"].iloc[-1] if not daily_df.empty else 0
        cost_pct = (total_commission + total_slippage) / final_equity if final_equity > 0 else 0
        f.write(f"  Cost as % of final equity: {cost_pct:.4%}\n\n")

        f.write("6. Rebalance Summary\n")
        f.write("-" * 40 + "\n")
        for _, row in rebalance_df.iterrows():
            rd = row['rebalance_date']
            rd_str = rd.strftime('%Y-%m-%d') if hasattr(rd, 'strftime') else str(rd)
            f.write(f"  {int(row['test_year'])}: date={rd_str}, "
                    f"targets={int(row['target_count'])}, bought={int(row['bought_count'])}, "
                    f"sold={int(row['sold_count'])}, skipped={int(row['skipped_count'])}, "
                    f"equity_after={row['equity_after']:,.0f}\n")
        f.write("\n")

        f.write("7. Comparison vs Original Walk-forward\n")
        f.write("-" * 40 + "\n")
        for _, row in vs_wf_df.iterrows():
            f.write(f"  {row['metric']:20s}: constrained={format_float(row['constrained_portfolio'])}, "
                    f"walk_forward={format_float(row['original_walk_forward'])}, "
                    f"diff={format_float(row['difference'])}\n")
        f.write("\n")

        f.write("8. Known Limitations\n")
        f.write("-" * 40 + "\n")
        f.write("  - No dividend/split/rights adjustment (uses raw prices)\n")
        f.write("  - No limit-up/limit-down enforcement\n")
        f.write("  - No real order book simulation\n")
        f.write("  - Uses yearly walk-forward selected_by_year targets; no intrayear alpha re-entry\n")
        f.write("  - Max weight is enforced at yearly rebalance and by close-based risk trims\n")
        f.write("  - No fractional shares (lot-size constraint applied)\n")
        f.write("  - Commission model: max(notional * rate, min_commission)\n")
        f.write("  - Walk-forward uses daily equal-weight rebalance without costs\n\n")

        f.write("9. Final Note\n")
        f.write("-" * 40 + "\n")
        f.write("  This portfolio backtest validates infrastructure only.\n")
        f.write("  Alpha v4 remains revise_alpha_signal.\n")
        f.write("  This result must not be interpreted as strategy promotion.\n")

    return path


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def make_equity_chart(daily_df, wf_daily, output_dir) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    c_eq = (1 + daily_df["daily_return"]).cumprod()
    ax.plot(daily_df["date"], c_eq, label="Constrained Portfolio", linewidth=1.5)
    wf = wf_daily.copy()
    wf["date"] = pd.to_datetime(wf["date"])
    wf_eq = wf["equity"] / wf["equity"].iloc[0]
    ax.plot(wf["date"], wf_eq, label="Walk-forward (Equal-weight)", linewidth=1, alpha=0.7)
    ax.set_title("Portfolio Equity Curve: Constrained vs Walk-forward")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "portfolio_equity_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_drawdown_chart(daily_df, output_dir) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4))
    eq = (1 + daily_df["daily_return"]).cumprod()
    dd = (eq - eq.cummax()) / eq.cummax()
    ax.fill_between(daily_df["date"], dd, 0, color="red", alpha=0.3)
    ax.plot(daily_df["date"], dd, color="red", linewidth=0.8)
    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "portfolio_drawdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_exposure_chart(daily_df, output_dir) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(daily_df["date"], daily_df["gross_exposure"], linewidth=0.8)
    ax1.set_ylabel("Gross Exposure")
    ax1.set_title("Portfolio Exposure & Position Count")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.plot(daily_df["date"], daily_df["position_count"], linewidth=0.8, color="green")
    ax2.set_ylabel("Position Count")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "portfolio_exposure_position_count.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="真实组合约束回测")
    parser.add_argument("--input-tag", required=True, help="精确匹配 walk-forward 文件名中的 tag")
    parser.add_argument("--walk-forward-dir", default=str(DEFAULT_WF_DIR))
    parser.add_argument("--price-root", default=str(DEFAULT_PRICE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="exp004_alpha_v4_full_smoke")
    parser.add_argument("--file-prefix", default="",
                        help="walk-forward 文件前缀，如 wf_alpha_v7_stock。为空则自动推导。")
    parser.add_argument("--initial-cash", type=float, default=1000000)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=5)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--rebalance-frequency", default="yearly",
                        choices=["yearly"], help="调仓频率，当前仅支持 yearly")
    parser.add_argument("--price-field", default="open", choices=["open", "close"])
    parser.add_argument("--allow-partial-fill", action="store_true", default=True)
    parser.add_argument("--no-partial-fill", action="store_false", dest="allow_partial_fill")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    wf_dir = resolve_path(args.walk_forward_dir)
    price_root = resolve_path(args.price_root)
    output_root = resolve_path(args.output_root)
    output_dir = output_root / f"portfolio_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load walk-forward data
    file_prefix = args.file_prefix if args.file_prefix else infer_file_prefix(wf_dir, args.input_tag)
    logger.info("Loading walk-forward data from %s (prefix: %s) ...", wf_dir, file_prefix)
    data = load_input_files(wf_dir, args.input_tag, file_prefix)
    selected = data["selected_by_year"]
    selected = safe_to_numeric(selected, ["test_year", "selected_rank"])
    selected["test_year"] = selected["test_year"].astype(int)
    wf_daily = data["portfolio_daily"]

    test_years = sorted(int(y) for y in selected["test_year"].unique())
    logger.info("  Test years: %s", test_years)
    logger.info("  Unique symbols: %d", selected['symbol'].nunique())
    logger.info("  Total selections: %d", len(selected))

    # Load price data for the walk-forward selected symbols. The portfolio layer
    # must consume the already-selected OOS universe instead of recomputing alpha
    # signals and changing the research decision surface.
    logger.info("Loading selected symbol price data ...")
    price_data = load_selected_price_data(selected, price_root)
    logger.info("  Price data loaded for %d symbols", len(price_data))

    # Run backtest
    logger.info("Running constrained portfolio backtest ...")
    daily_df, trades_df, positions_df, rebalance_df = run_yearly_rebalance_backtest(
        selected, price_data, wf_daily,
        args.initial_cash, args.max_positions, args.max_weight,
        args.lot_size, args.commission_rate, args.min_commission,
        args.slippage_bps, args.price_field, args.allow_partial_fill,
    )

    # Build summaries
    period_df = build_period_summary(daily_df, trades_df, test_years)
    vs_wf_df = build_vs_walkforward(daily_df, wf_daily)

    # Save CSVs
    logger.info("Saving outputs ...")
    csv_outputs = {
        "portfolio_daily": daily_df,
        "portfolio_trades": trades_df,
        "portfolio_positions_daily": positions_df,
        "portfolio_rebalance_log": rebalance_df,
        "portfolio_period_summary": period_df,
        "portfolio_vs_walkforward": vs_wf_df,
    }
    for name, df in csv_outputs.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, encoding="utf-8-sig", index=False)
        logger.info("  Saved: %s", path.name)

    # Report
    report_path = write_report(output_dir, args, daily_df, trades_df, rebalance_df, period_df, vs_wf_df, test_years)
    logger.info("  Report: %s", report_path.name)

    # Charts
    if not args.no_png:
        logger.info("Generating charts ...")
        for name, func in [
            ("equity_curve", lambda: make_equity_chart(daily_df, wf_daily, output_dir)),
            ("drawdown", lambda: make_drawdown_chart(daily_df, output_dir)),
            ("exposure", lambda: make_exposure_chart(daily_df, output_dir)),
        ]:
            try:
                p = func()
                logger.info("  Chart: %s", p.name)
            except Exception as e:
                logger.warning("  Chart %s failed: %s", name, e)

    # Summary
    overall = period_df[period_df["period"] == "overall"]
    if not overall.empty:
        o = overall.iloc[0]
        total_cost = (trades_df["commission"].sum() + trades_df["slippage_cost"].sum()) if not trades_df.empty else 0
        print(f"\n{'=' * 60}")
        print(f"  Total return: {format_pct(o['total_return'])}")
        print(f"  Annual return: {format_pct(o['annual_return'])}")
        print(f"  Sharpe: {format_float(o['sharpe'])}")
        print(f"  Max drawdown: {format_pct(o['max_drawdown'])}")
        print(f"  Total trading cost: {total_cost:,.2f}")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    setup_cli_logging()
    main()
