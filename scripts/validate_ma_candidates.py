# -*- coding: utf-8 -*-
"""
validate_ma_candidates.py

用途：
对均线策略做 walk-forward / 滚动样本外验证。

核心思想：
每一个测试年度，只使用该年度之前的数据进行训练、选股、选参数；
然后在该年度进行样本外测试。

例如：
2015-2020 训练 -> 2021 测试
2015-2021 训练 -> 2022 测试
2015-2022 训练 -> 2023 测试
2015-2023 训练 -> 2024 测试
2015-2024 训练 -> 2025 测试
2015-2025 训练 -> 2026 测试

这比“直接固定使用全样本筛出的 37 只股票”更接近真实可交易策略。

运行示例：

1. 小规模测试，只跑深市前 300 个标的，只测试 2024 年：
python scripts\\validate_ma_candidates.py --market SZ --limit 300 --first-test-year 2024 --last-test-year 2024

2. 深市 walk-forward 验证：
python scripts\\validate_ma_candidates.py --market SZ --security-type stock --first-test-year 2021 --portfolio-size 20

3. 沪市 walk-forward 验证：
python scripts\\validate_ma_candidates.py --market SH --security-type stock --first-test-year 2021 --portfolio-size 20

4. 全市场 walk-forward 验证：
python scripts\\validate_ma_candidates.py --market ALL --security-type stock --first-test-year 2021 --portfolio-size 20

5. 使用更严格的训练筛选条件：
python scripts\\validate_ma_candidates.py --market ALL --min-train-rows 1500 --min-train-trades 8 --max-train-drawdown -0.45 --min-train-sharpe 0.6 --min-train-annual-return 0.08

6. 只在已有候选池中做 walk-forward 验证，不推荐作为第一选择：
python scripts\\validate_ma_candidates.py --candidate-file backtests\\filtered_ma_csv\\ma_filter_ALL_stock_rows1500_trades8_ddm0p45_sharpe0p6_ann0p08_best_per_symbol.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.metrics import max_drawdown_from_equity, calc_metrics_from_returns  # noqa: E402
from scripts.common.validation import resolve_path, parse_int_list, parse_workers, parse_date_yyyymmdd  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "backtests" / "walk_forward_ma_csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)





def run_ma_backtest_frame(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> pd.DataFrame:
    result = df.copy()

    result["fast_ma"] = result["close"].rolling(fast).mean()
    result["slow_ma"] = result["close"].rolling(slow).mean()

    # 当日收盘后产生信号
    result["signal"] = np.where(result["fast_ma"] > result["slow_ma"], 1, 0)

    # 次日持仓，避免未来函数
    result["position"] = result["signal"].shift(1).fillna(0)

    result["stock_ret"] = result["close"].pct_change().fillna(0)

    pos_change = result["position"].diff().fillna(result["position"])
    buy_turnover = pos_change.clip(lower=0)
    sell_turnover = (-pos_change).clip(lower=0)

    result["cost"] = (
        buy_turnover * (commission + slippage)
        + sell_turnover * (commission + sell_tax + slippage)
    )

    result["strategy_ret"] = result["position"] * result["stock_ret"] - result["cost"]

    return result


def calc_train_score(metrics: dict) -> float:
    """
    训练期排序分数。

    max_drawdown 是负数，因此加上 max_drawdown 等于惩罚深回撤。
    """
    sharpe = 0.0 if pd.isna(metrics["sharpe"]) else metrics["sharpe"]
    excess = 0.0 if pd.isna(metrics["excess_total_return"]) else metrics["excess_total_return"]

    return float(
        metrics["annual_return"]
        + 0.30 * sharpe
        + 0.40 * excess
        + 0.80 * metrics["max_drawdown"]
    )


def pass_train_filters(metrics: dict, args: argparse.Namespace) -> bool:
    if pd.isna(metrics["annual_return"]) or pd.isna(metrics["max_drawdown"]) or pd.isna(metrics["sharpe"]):
        return False

    if metrics["days"] < args.min_train_rows:
        return False

    if metrics["trade_count"] < args.min_train_trades:
        return False

    if metrics["max_drawdown"] < args.max_train_drawdown:
        return False

    if metrics["sharpe"] < args.min_train_sharpe:
        return False

    if metrics["annual_return"] < args.min_train_annual_return:
        return False

    if not args.allow_negative_train_excess:
        if pd.isna(metrics["excess_total_return"]) or metrics["excess_total_return"] <= args.min_train_excess_total_return:
            return False

    return True


def pass_train_filters_simple(
    metrics: dict,
    min_train_rows: int,
    min_train_trades: int,
    max_train_drawdown: float,
    min_train_sharpe: float,
    min_train_annual_return: float,
    min_train_excess_total_return: float,
    allow_negative_train_excess: bool,
) -> bool:
    """训练期筛选条件（纯函数版本，供 worker 使用）。"""
    if pd.isna(metrics["annual_return"]) or pd.isna(metrics["max_drawdown"]) or pd.isna(metrics["sharpe"]):
        return False

    if metrics["days"] < min_train_rows:
        return False

    if metrics["trade_count"] < min_train_trades:
        return False

    if metrics["max_drawdown"] < max_train_drawdown:
        return False

    if metrics["sharpe"] < min_train_sharpe:
        return False

    if metrics["annual_return"] < min_train_annual_return:
        return False

    if not allow_negative_train_excess:
        if pd.isna(metrics["excess_total_return"]) or metrics["excess_total_return"] <= min_train_excess_total_return:
            return False

    return True


def train_one_stock_all_years(args_tuple):
    """
    处理一只股票：加载数据一次，对所有测试年份运行训练阶段，返回每年的最佳候选。
    供 ProcessPoolExecutor 使用的 worker 函数。
    """
    (item_dict, test_years, train_start, end, fast_list, slow_list,
     commission, sell_tax, slippage, cash, warmup_buffer,
     min_train_rows, min_train_trades, max_train_drawdown,
     min_train_sharpe, min_train_annual_return,
     min_train_excess_total_return, allow_negative_train_excess) = args_tuple

    symbol = item_dict["symbol"]
    market = item_dict["market"]
    security_type = item_dict["security_type"]
    csv_path = Path(item_dict["csv_path"])

    pairs = [(f, s) for f in fast_list for s in slow_list if f < s]

    try:
        full_df = ma.load_qmt_price_csv(csv_path, train_start, end)
    except Exception:
        return {}

    results_by_year = {}

    for year in test_years:
        train_end_dt = pd.Timestamp(year=year - 1, month=12, day=31)
        train_start_dt = pd.to_datetime(train_start)

        if train_end_dt < train_start_dt:
            continue

        train_df = full_df[(full_df.index >= train_start_dt) & (full_df.index <= train_end_dt)]

        if train_df.empty:
            continue

        best_candidate = None
        best_score = -np.inf

        for fast, slow in pairs:
            if len(train_df) < slow + warmup_buffer:
                continue

            result = run_ma_backtest_frame(
                df=train_df,
                fast=fast,
                slow=slow,
                commission=commission,
                sell_tax=sell_tax,
                slippage=slippage,
            )

            metrics = calc_metrics_from_returns(
                result["strategy_ret"],
                position=result["position"],
                cash=cash,
                stock_ret=result["stock_ret"],
            )

            if not pass_train_filters_simple(
                metrics, min_train_rows, min_train_trades, max_train_drawdown,
                min_train_sharpe, min_train_annual_return,
                min_train_excess_total_return, allow_negative_train_excess,
            ):
                continue

            train_score = calc_train_score(metrics)

            if train_score > best_score:
                best_score = train_score
                best_candidate = {
                    "symbol": symbol,
                    "market": market,
                    "security_type": security_type,
                    "csv_path": str(csv_path),
                    "fast": fast,
                    "slow": slow,
                    "train_start": str(train_start_dt.date()),
                    "train_end": str(train_end_dt.date()),
                    "train_rows": len(train_df),
                    "train_total_return": metrics["total_return"],
                    "train_annual_return": metrics["annual_return"],
                    "train_annual_volatility": metrics["annual_volatility"],
                    "train_max_drawdown": metrics["max_drawdown"],
                    "train_sharpe": metrics["sharpe"],
                    "train_trade_count": metrics["trade_count"],
                    "train_buy_hold_total_return": metrics["buy_hold_total_return"],
                    "train_excess_total_return": metrics["excess_total_return"],
                    "train_score": train_score,
                }

        if best_candidate is not None:
            results_by_year[year] = best_candidate

    return results_by_year


def build_catalog(args: argparse.Namespace) -> pd.DataFrame:
    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root

    catalog = ma.scan_qmt_export(export_root)

    if args.market != "ALL":
        catalog = catalog[catalog["market"] == args.market]

    if args.security_type != "ALL":
        catalog = catalog[catalog["security_type"] == args.security_type]

    if args.candidate_file:
        candidate_path = resolve_path(args.candidate_file)
        if not candidate_path.exists():
            raise FileNotFoundError(f"候选池文件不存在：{candidate_path}")

        candidates = pd.read_csv(candidate_path)
        if "symbol" not in candidates.columns:
            raise RuntimeError(f"候选池文件缺少 symbol 字段：{candidate_path}")

        allowed_symbols = set(candidates["symbol"].astype(str).str.upper())
        catalog = catalog[catalog["symbol"].isin(allowed_symbols)]

    catalog = catalog.sort_values(["market", "symbol"]).reset_index(drop=True)

    if args.limit and args.limit > 0:
        catalog = catalog.head(args.limit)

    if catalog.empty:
        raise RuntimeError("catalog 为空，请检查 market / security-type / candidate-file 参数。")

    return catalog


def get_test_years(args: argparse.Namespace) -> list[int]:
    first_year = args.first_test_year

    if args.last_test_year:
        last_year = args.last_test_year
    elif args.end:
        last_year = parse_date_yyyymmdd(args.end).year
    else:
        last_year = datetime.today().year

    if first_year > last_year:
        raise ValueError(f"first-test-year 不能大于 last-test-year：{first_year} > {last_year}")

    return list(range(first_year, last_year + 1))


def load_symbol_data(csv_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    """
    每只证券只加载一次。

    注意：这里加载 train_start 到 end/latest 的数据。
    后续每个年度只会切片到当期允许使用的时间，不会使用未来数据做训练。
    """
    return ma.load_qmt_price_csv(csv_path, args.train_start, args.end)


def train_select_for_period(
    catalog: pd.DataFrame,
    data_cache: dict[str, pd.DataFrame],
    train_start_dt: pd.Timestamp,
    train_end_dt: pd.Timestamp,
    fast_list: list[int],
    slow_list: list[int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    pairs = [(f, s) for f in fast_list for s in slow_list if f < s]

    for idx, item in catalog.iterrows():
        symbol = item["symbol"]
        market = item["market"]
        security_type = item["security_type"]
        csv_path = Path(item["csv_path"])

        try:
            if symbol not in data_cache:
                data_cache[symbol] = load_symbol_data(csv_path, args)

            full_df = data_cache[symbol]
            train_df = full_df[(full_df.index >= train_start_dt) & (full_df.index <= train_end_dt)]

            if train_df.empty:
                continue

            for fast, slow in pairs:
                # 训练期至少要满足慢均线 + warmup
                if len(train_df) < slow + args.warmup_buffer:
                    continue

                result = run_ma_backtest_frame(
                    df=train_df,
                    fast=fast,
                    slow=slow,
                    commission=args.commission,
                    sell_tax=args.sell_tax,
                    slippage=args.slippage,
                )

                metrics = calc_metrics_from_returns(
                    result["strategy_ret"],
                    position=result["position"],
                    cash=args.cash,
                    stock_ret=result["stock_ret"],
                )

                if not pass_train_filters(metrics, args):
                    continue

                train_score = calc_train_score(metrics)

                rows.append(
                    {
                        "symbol": symbol,
                        "market": market,
                        "security_type": security_type,
                        "csv_path": str(csv_path),
                        "fast": fast,
                        "slow": slow,
                        "train_start": str(train_start_dt.date()),
                        "train_end": str(train_end_dt.date()),
                        "train_rows": len(train_df),
                        "train_total_return": metrics["total_return"],
                        "train_annual_return": metrics["annual_return"],
                        "train_annual_volatility": metrics["annual_volatility"],
                        "train_max_drawdown": metrics["max_drawdown"],
                        "train_sharpe": metrics["sharpe"],
                        "train_trade_count": metrics["trade_count"],
                        "train_buy_hold_total_return": metrics["buy_hold_total_return"],
                        "train_excess_total_return": metrics["excess_total_return"],
                        "train_score": train_score,
                    }
                )

        except Exception as exc:
            logger.warning("训练跳过 %s: %s", symbol, repr(exc))
            continue

        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            logger.info("训练进度：%d/%d", idx + 1, len(catalog))

    if not rows:
        return pd.DataFrame()

    candidates = pd.DataFrame(rows)

    # 每只股票只保留训练期最优参数
    candidates = candidates.sort_values(
        by=["train_score", "train_sharpe", "train_annual_return"],
        ascending=[False, False, False],
    )

    best_per_symbol = candidates.drop_duplicates(subset=["symbol"], keep="first")

    selected = best_per_symbol.head(args.portfolio_size).reset_index(drop=True)
    selected["selected_rank"] = range(1, len(selected) + 1)

    return selected


def test_selected_for_period(
    selected: pd.DataFrame,
    data_cache: dict[str, pd.DataFrame],
    test_start_dt: pd.Timestamp,
    test_end_dt: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    return_series = {}

    for _, item in selected.iterrows():
        symbol = item["symbol"]
        full_df = data_cache.get(symbol)

        if full_df is None or full_df.empty:
            continue

        fast = int(item["fast"])
        slow = int(item["slow"])

        # 这里不能只传测试期数据，否则测试期初的均线没有足够历史。
        # 正确做法：使用 test_end 之前的所有历史计算指标，然后只截取测试期收益。
        calc_df = full_df[full_df.index <= test_end_dt].copy()

        if len(calc_df) < slow + args.warmup_buffer:
            continue

        result = run_ma_backtest_frame(
            df=calc_df,
            fast=fast,
            slow=slow,
            commission=args.commission,
            sell_tax=args.sell_tax,
            slippage=args.slippage,
        )

        test_result = result[(result.index >= test_start_dt) & (result.index <= test_end_dt)].copy()

        if test_result.empty:
            continue

        metrics = calc_metrics_from_returns(
            test_result["strategy_ret"],
            position=test_result["position"],
            cash=args.cash,
            stock_ret=test_result["stock_ret"],
        )

        return_series[symbol] = test_result["strategy_ret"].rename(symbol)

        detail_rows.append(
            {
                "test_year": test_start_dt.year,
                "symbol": symbol,
                "market": item["market"],
                "security_type": item["security_type"],
                "selected_rank": item["selected_rank"],
                "fast": fast,
                "slow": slow,
                "train_start": item["train_start"],
                "train_end": item["train_end"],
                "train_rows": item["train_rows"],
                "train_annual_return": item["train_annual_return"],
                "train_max_drawdown": item["train_max_drawdown"],
                "train_sharpe": item["train_sharpe"],
                "train_trade_count": item["train_trade_count"],
                "train_excess_total_return": item["train_excess_total_return"],
                "train_score": item["train_score"],
                "test_start": str(test_start_dt.date()),
                "test_end": str(test_result.index.max().date()),
                "test_rows": len(test_result),
                "test_total_return": metrics["total_return"],
                "test_annual_return": metrics["annual_return"],
                "test_annual_volatility": metrics["annual_volatility"],
                "test_max_drawdown": metrics["max_drawdown"],
                "test_sharpe": metrics["sharpe"],
                "test_trade_count": metrics["trade_count"],
                "test_buy_hold_total_return": metrics["buy_hold_total_return"],
                "test_excess_total_return": metrics["excess_total_return"],
            }
        )

    detail = pd.DataFrame(detail_rows)

    if return_series:
        returns_df = pd.concat(return_series.values(), axis=1).sort_index()
        # 等权组合：缺失收益按 0 处理，表示该标的当日对组合没有贡献
        returns_df = returns_df.fillna(0.0)
    else:
        returns_df = pd.DataFrame()

    return detail, returns_df


def calc_portfolio_period_metrics(
    returns_df: pd.DataFrame,
    test_year: int,
    cash: float,
) -> dict:
    if returns_df.empty:
        return {
            "test_year": test_year,
            "portfolio_size_actual": 0,
            "period_start": "",
            "period_end": "",
            "days": 0,
            "portfolio_total_return": np.nan,
            "portfolio_annual_return": np.nan,
            "portfolio_annual_volatility": np.nan,
            "portfolio_max_drawdown": np.nan,
            "portfolio_sharpe": np.nan,
            "portfolio_final_equity": cash,
        }

    portfolio_ret = returns_df.mean(axis=1).sort_index()
    metrics = calc_metrics_from_returns(
        portfolio_ret,
        position=None,
        cash=cash,
    )

    return {
        "test_year": test_year,
        "portfolio_size_actual": returns_df.shape[1],
        "period_start": str(portfolio_ret.index.min().date()),
        "period_end": str(portfolio_ret.index.max().date()),
        "days": metrics["days"],
        "portfolio_total_return": metrics["total_return"],
        "portfolio_annual_return": metrics["annual_return"],
        "portfolio_annual_volatility": metrics["annual_volatility"],
        "portfolio_max_drawdown": metrics["max_drawdown"],
        "portfolio_sharpe": metrics["sharpe"],
        "portfolio_final_equity": metrics["final_equity"],
    }


def build_tag(args: argparse.Namespace) -> str:
    fast_tag = "f" + "-".join(parse_int_list(args.fast_list).__str__().replace("[", "").replace("]", "").replace(" ", "").split(","))
    # 上面写法不够直观，下面重新生成一次，覆盖 fast_tag
    fast_tag = "f" + "-".join(map(str, parse_int_list(args.fast_list)))
    slow_tag = "s" + "-".join(map(str, parse_int_list(args.slow_list)))

    return (
        f"{args.output_name}_"
        f"{args.security_type}_{args.market}_"
        f"train{args.train_start}_"
        f"test{args.first_test_year}-{args.last_test_year or 'latest'}_"
        f"{fast_tag}_{slow_tag}_"
        f"top{args.portfolio_size}"
    )


def save_outputs(
    selected_all: pd.DataFrame,
    test_detail_all: pd.DataFrame,
    portfolio_daily_all: pd.DataFrame,
    portfolio_periods: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    tag = build_tag(args)

    selected_path = OUTPUT_DIR / f"{tag}_selected_by_year.csv"
    test_detail_path = OUTPUT_DIR / f"{tag}_test_detail.csv"
    portfolio_daily_path = OUTPUT_DIR / f"{tag}_portfolio_daily.csv"
    portfolio_period_path = OUTPUT_DIR / f"{tag}_portfolio_period_summary.csv"
    report_path = OUTPUT_DIR / f"{tag}_report.txt"

    selected_all.to_csv(selected_path, index=False, encoding="utf-8-sig")
    test_detail_all.to_csv(test_detail_path, index=False, encoding="utf-8-sig")
    portfolio_daily_all.to_csv(portfolio_daily_path, index=False, encoding="utf-8-sig")
    portfolio_periods.to_csv(portfolio_period_path, index=False, encoding="utf-8-sig")

    if not portfolio_daily_all.empty and "portfolio_ret" in portfolio_daily_all.columns:
        ret = portfolio_daily_all.set_index("date")["portfolio_ret"].astype(float)
        overall_metrics = calc_metrics_from_returns(
            ret,
            position=None,
            cash=args.cash,
        )
    else:
        overall_metrics = {}

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("walk-forward 均线策略验证报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("参数设置：\n")
        f.write(f"market: {args.market}\n")
        f.write(f"security_type: {args.security_type}\n")
        f.write(f"train_start: {args.train_start}\n")
        f.write(f"first_test_year: {args.first_test_year}\n")
        f.write(f"last_test_year: {args.last_test_year or 'latest'}\n")
        f.write(f"fast_list: {args.fast_list}\n")
        f.write(f"slow_list: {args.slow_list}\n")
        f.write(f"portfolio_size: {args.portfolio_size}\n")
        f.write(f"min_train_rows: {args.min_train_rows}\n")
        f.write(f"min_train_trades: {args.min_train_trades}\n")
        f.write(f"max_train_drawdown: {args.max_train_drawdown}\n")
        f.write(f"min_train_sharpe: {args.min_train_sharpe}\n")
        f.write(f"min_train_annual_return: {args.min_train_annual_return}\n")
        f.write(f"allow_negative_train_excess: {args.allow_negative_train_excess}\n\n")

        f.write("输出文件：\n")
        f.write(f"selected_by_year: {selected_path}\n")
        f.write(f"test_detail: {test_detail_path}\n")
        f.write(f"portfolio_daily: {portfolio_daily_path}\n")
        f.write(f"portfolio_period_summary: {portfolio_period_path}\n\n")

        f.write("数量统计：\n")
        f.write(f"selected rows: {len(selected_all)}\n")
        f.write(f"test detail rows: {len(test_detail_all)}\n")
        f.write(f"portfolio daily rows: {len(portfolio_daily_all)}\n")
        f.write(f"portfolio period rows: {len(portfolio_periods)}\n\n")

        if overall_metrics:
            f.write("组合整体样本外表现：\n")
            f.write(f"total_return: {overall_metrics['total_return']:.8f}\n")
            f.write(f"annual_return: {overall_metrics['annual_return']:.8f}\n")
            f.write(f"annual_volatility: {overall_metrics['annual_volatility']:.8f}\n")
            f.write(f"max_drawdown: {overall_metrics['max_drawdown']:.8f}\n")
            f.write(f"sharpe: {overall_metrics['sharpe']:.8f}\n")
            f.write(f"days: {overall_metrics['days']}\n")

    logger.info("walk-forward 验证完成。")
    logger.info("年度入选组合：%s", selected_path)
    logger.info("样本外单股明细：%s", test_detail_path)
    logger.info("组合日收益：%s", portfolio_daily_path)
    logger.info("组合年度汇总：%s", portfolio_period_path)
    logger.info("报告：%s", report_path)

    if not portfolio_periods.empty:
        logger.info("组合年度表现：")
        print(portfolio_periods.to_string(index=False))

    if overall_metrics:
        logger.info("组合整体样本外表现：")
        logger.info("总收益: %.2f%%", overall_metrics['total_return'] * 100)
        logger.info("年化收益: %.2f%%", overall_metrics['annual_return'] * 100)
        logger.info("最大回撤: %.2f%%", overall_metrics['max_drawdown'] * 100)
        logger.info("夏普比率: %.4f", overall_metrics['sharpe'])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="walk-forward 均线策略样本外验证")

    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT), help="QMT CSV 导出根目录")
    parser.add_argument("--market", default="ALL", choices=["ALL", "SH", "SZ"], help="市场筛选")
    parser.add_argument("--security-type", default="stock", choices=["stock", "index", "other", "ALL"], help="证券类型筛选")
    parser.add_argument("--candidate-file", default="", help="可选：只在候选池 symbol 中做验证。不传则使用全市场。")

    parser.add_argument("--train-start", default="20150101", help="训练起点，例如 20150101")
    parser.add_argument("--first-test-year", type=int, default=2021, help="第一个样本外测试年份")
    parser.add_argument("--last-test-year", type=int, default=0, help="最后一个测试年份；0 表示自动使用当前年份")
    parser.add_argument("--end", default="", help="数据结束日期，例如 20260514；留空表示使用数据最新日期")

    parser.add_argument("--fast-list", default="5,10,20", help="快均线列表，例如 5,10,20")
    parser.add_argument("--slow-list", default="60,120,250", help="慢均线列表，例如 60,120,250")
    parser.add_argument("--warmup-buffer", type=int, default=10, help="慢均线之外额外要求的缓冲行数")

    parser.add_argument("--portfolio-size", type=int, default=20, help="每期入选股票数量")
    parser.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金，仅用于指标换算")

    parser.add_argument("--commission", type=float, default=0.0001, help="佣金率，默认万一")
    parser.add_argument("--sell-tax", type=float, default=0.0005, help="卖出印花税，默认万五")
    parser.add_argument("--slippage", type=float, default=0.0, help="滑点率，默认 0")

    # 训练期筛选条件
    parser.add_argument("--min-train-rows", type=int, default=1000, help="训练期最少样本行数")
    parser.add_argument("--min-train-trades", type=int, default=6, help="训练期最少换仓次数")
    parser.add_argument("--max-train-drawdown", type=float, default=-0.50, help="训练期最大回撤下限，例如 -0.50 表示不低于 -50%")
    parser.add_argument("--min-train-sharpe", type=float, default=0.4, help="训练期最低夏普")
    parser.add_argument("--min-train-annual-return", type=float, default=0.05, help="训练期最低年化收益")
    parser.add_argument("--min-train-excess-total-return", type=float, default=0.0, help="训练期最低总超额收益")
    parser.add_argument("--allow-negative-train-excess", action="store_true", help="训练期允许跑输买入持有")

    parser.add_argument("--limit", type=int, default=0, help="只测试前 N 个标的；用于调试")
    parser.add_argument("--progress-every", type=int, default=500, help="训练阶段每处理 N 个标的打印一次进度")
    parser.add_argument("--output-name", default="wf_ma", help="输出文件名前缀")
    parser.add_argument("--workers", type=parse_workers, default=1, help="并行进程数。默认 1（单进程）。使用 'auto' 自动检测 CPU 核心数，或指定整数如 --workers 6")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # argparse 中 last-test-year 默认是 0，内部统一转成 None
    if args.last_test_year == 0:
        args.last_test_year = None

    train_start_dt = parse_date_yyyymmdd(args.train_start)
    fast_list = parse_int_list(args.fast_list)
    slow_list = parse_int_list(args.slow_list)

    pairs = [(f, s) for f in fast_list for s in slow_list if f < s]
    if not pairs:
        raise ValueError("没有有效参数组合，要求 fast < slow。")

    catalog = build_catalog(args)
    test_years = get_test_years(args)

    logger.info("walk-forward 验证配置：")
    logger.info("市场：%s", args.market)
    logger.info("证券类型：%s", args.security_type)
    logger.info("标的数量：%d", len(catalog))
    logger.info("测试年份：%s", test_years)
    logger.info("参数组合：%s", pairs)
    logger.info("每期组合数量：%d", args.portfolio_size)
    logger.info("训练起点：%s", args.train_start)
    logger.info("候选池文件：%s", args.candidate_file or '未使用，全市场滚动筛选')

    data_cache: dict[str, pd.DataFrame] = {}

    selected_all_rows = []
    test_detail_all_rows = []
    portfolio_daily_rows = []
    portfolio_period_rows = []

    running_equity = args.cash

    if args.workers > 1:
        # 并行模式：按股票跨所有年份并行训练
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = []
        for _, item in catalog.iterrows():
            item_dict = item.to_dict()
            tasks.append((
                item_dict, test_years, args.train_start, args.end,
                fast_list, slow_list,
                args.commission, args.sell_tax, args.slippage,
                args.cash, args.warmup_buffer,
                args.min_train_rows, args.min_train_trades, args.max_train_drawdown,
                args.min_train_sharpe, args.min_train_annual_return,
                args.min_train_excess_total_return, args.allow_negative_train_excess,
            ))

        total_stocks = len(tasks)
        completed_stocks = 0
        all_candidates_by_year = {year: [] for year in test_years}

        logger.info("并行训练：%d 进程处理 %d 只股票，跨 %d 个年份...", args.workers, total_stocks, len(test_years))

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(train_one_stock_all_years, t): t[0]["symbol"] for t in tasks}

            for future in as_completed(futures):
                completed_stocks += 1
                if completed_stocks % 100 == 0:
                    logger.info("训练进度：%d/%d 只股票完成", completed_stocks, total_stocks)

                try:
                    stock_results = future.result()
                    for year, candidate in stock_results.items():
                        all_candidates_by_year[year].append(candidate)
                except Exception as exc:
                    symbol = futures[future]
                    logger.error("训练错误 %s: %s", symbol, repr(exc))

        # Phase 2 & 3: 串行 - 每年选 top N + 测试
        for year in test_years:
            train_end_dt = pd.Timestamp(year=year - 1, month=12, day=31)
            test_start_dt = pd.Timestamp(year=year, month=1, day=1)
            test_end_dt = pd.Timestamp(year=year, month=12, day=31)

            if args.end:
                end_dt = parse_date_yyyymmdd(args.end)
                if test_end_dt > end_dt:
                    test_end_dt = end_dt

            if train_end_dt < train_start_dt:
                logger.warning("跳过年份 %d: 训练期为空", year)
                continue

            logger.info("=" * 80)
            logger.info("测试年份：%d", year)
            logger.info("训练区间：%s 至 %s", train_start_dt.date(), train_end_dt.date())
            logger.info("测试区间：%s 至 %s", test_start_dt.date(), test_end_dt.date())

            candidates = all_candidates_by_year.get(year, [])
            if not candidates:
                logger.warning("无入选标的 %d", year)
                continue

            candidates_df = pd.DataFrame(candidates)
            candidates_df = candidates_df.sort_values(
                by=["train_score", "train_sharpe", "train_annual_return"],
                ascending=[False, False, False],
            )
            best_per_symbol = candidates_df.drop_duplicates(subset=["symbol"], keep="first")
            selected = best_per_symbol.head(args.portfolio_size).reset_index(drop=True)
            selected["selected_rank"] = range(1, len(selected) + 1)

            if selected.empty:
                logger.warning("无入选标的 %d", year)
                continue

            selected["test_year"] = year
            selected_all_rows.append(selected)

            logger.info("训练筛选后入选数量：%d", len(selected))
            print(selected[["selected_rank", "symbol", "fast", "slow", "train_annual_return", "train_max_drawdown", "train_sharpe", "train_score"]].head(20).to_string(index=False))

            # 测试阶段需要 data_cache，为选中的股票加载数据
            for symbol in selected["symbol"].unique():
                if symbol not in data_cache:
                    csv_path_str = selected.loc[selected["symbol"] == symbol, "csv_path"].iloc[0]
                    data_cache[symbol] = ma.load_qmt_price_csv(Path(csv_path_str), args.train_start, args.end)

            test_detail, returns_df = test_selected_for_period(
                selected=selected,
                data_cache=data_cache,
                test_start_dt=test_start_dt,
                test_end_dt=test_end_dt,
                args=args,
            )

            if not test_detail.empty:
                test_detail_all_rows.append(test_detail)

            period_metrics = calc_portfolio_period_metrics(
                returns_df=returns_df,
                test_year=year,
                cash=args.cash,
            )
            portfolio_period_rows.append(period_metrics)

            if not returns_df.empty:
                portfolio_ret = returns_df.mean(axis=1).sort_index()
                period_equity = running_equity * (1.0 + portfolio_ret).cumprod()
                running_equity = float(period_equity.iloc[-1])

                period_daily = pd.DataFrame(
                    {
                        "date": portfolio_ret.index,
                        "test_year": year,
                        "portfolio_ret": portfolio_ret.values,
                        "portfolio_equity": period_equity.values,
                        "portfolio_size_actual": returns_df.shape[1],
                    }
                )
                portfolio_daily_rows.append(period_daily)

            logger.info("样本外组合表现：")
            print(pd.DataFrame([period_metrics]).to_string(index=False))

    else:
        # 单进程模式：原有逻辑
        for year in test_years:
            train_end_dt = pd.Timestamp(year=year - 1, month=12, day=31)
            test_start_dt = pd.Timestamp(year=year, month=1, day=1)
            test_end_dt = pd.Timestamp(year=year, month=12, day=31)

            if args.end:
                end_dt = parse_date_yyyymmdd(args.end)
                if test_end_dt > end_dt:
                    test_end_dt = end_dt

            if train_end_dt < train_start_dt:
                logger.warning("跳过年份 %d: 训练期为空", year)
                continue

            logger.info("=" * 80)
            logger.info("测试年份：%d", year)
            logger.info("训练区间：%s 至 %s", train_start_dt.date(), train_end_dt.date())
            logger.info("测试区间：%s 至 %s", test_start_dt.date(), test_end_dt.date())

            selected = train_select_for_period(
                catalog=catalog,
                data_cache=data_cache,
                train_start_dt=train_start_dt,
                train_end_dt=train_end_dt,
                fast_list=fast_list,
                slow_list=slow_list,
                args=args,
            )

            if selected.empty:
                logger.warning("无入选标的 %d", year)
                continue

            selected["test_year"] = year
            selected_all_rows.append(selected)

            logger.info("训练筛选后入选数量：%d", len(selected))
            print(selected[["selected_rank", "symbol", "fast", "slow", "train_annual_return", "train_max_drawdown", "train_sharpe", "train_score"]].head(20).to_string(index=False))

            test_detail, returns_df = test_selected_for_period(
                selected=selected,
                data_cache=data_cache,
                test_start_dt=test_start_dt,
                test_end_dt=test_end_dt,
                args=args,
            )

            if not test_detail.empty:
                test_detail_all_rows.append(test_detail)

            period_metrics = calc_portfolio_period_metrics(
                returns_df=returns_df,
                test_year=year,
                cash=args.cash,
            )
            portfolio_period_rows.append(period_metrics)

            if not returns_df.empty:
                portfolio_ret = returns_df.mean(axis=1).sort_index()
                period_equity = running_equity * (1.0 + portfolio_ret).cumprod()
                running_equity = float(period_equity.iloc[-1])

                period_daily = pd.DataFrame(
                    {
                        "date": portfolio_ret.index,
                        "test_year": year,
                        "portfolio_ret": portfolio_ret.values,
                        "portfolio_equity": period_equity.values,
                        "portfolio_size_actual": returns_df.shape[1],
                    }
                )
                portfolio_daily_rows.append(period_daily)

            logger.info("样本外组合表现：")
            print(pd.DataFrame([period_metrics]).to_string(index=False))

    if not selected_all_rows:
        raise RuntimeError("没有任何年度产生入选标的。请放宽训练筛选条件。")

    selected_all = pd.concat(selected_all_rows, ignore_index=True)

    if test_detail_all_rows:
        test_detail_all = pd.concat(test_detail_all_rows, ignore_index=True)
    else:
        test_detail_all = pd.DataFrame()

    if portfolio_daily_rows:
        portfolio_daily_all = pd.concat(portfolio_daily_rows, ignore_index=True)
    else:
        portfolio_daily_all = pd.DataFrame()

    portfolio_periods = pd.DataFrame(portfolio_period_rows)

    save_outputs(
        selected_all=selected_all,
        test_detail_all=test_detail_all,
        portfolio_daily_all=portfolio_daily_all,
        portfolio_periods=portfolio_periods,
        args=args,
    )


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception as exc:
        logger.error("程序异常：%s", repr(exc))
        raise