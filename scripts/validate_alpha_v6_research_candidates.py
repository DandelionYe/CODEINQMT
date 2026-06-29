# -*- coding: utf-8 -*-
"""
validate_alpha_v6_research_candidates.py

Alpha v6 walk-forward / 滚动样本外验证：
支持 4 个 alpha variant 参数网格搜索（短期反转、低波动、换手率反转、量价背离）。

共享逻辑已提取到 scripts/common/wf_batch_shared.py。

核心思想：
每一个测试年度，只使用该年度之前的数据进行训练、选股、选参；
然后在该年度进行样本外测试。

运行示例：

1. 小规模测试：
python scripts\\validate_alpha_v6_research_candidates.py --market SZ --limit 200 --first-test-year 2024 --last-test-year 2024 --portfolio-size 10 --alpha-variant-list short_term_reversal,low_volatility --reversal-window-list 5,10 --vol-window-list 60 --benchmark-list 000300.SH --benchmark-ma-list 120 --workers 4 --progress-every 50

2. 全市场 walk-forward：
python scripts\\validate_alpha_v6_research_candidates.py --market ALL --security-type stock --first-test-year 2021 --last-test-year 2025 --portfolio-size 20 --alpha-variant-list short_term_reversal,low_volatility,turnover_reversal,volume_price_divergence --reversal-window-list 5,10,20 --vol-window-list 20,60,120 --turnover-short-list 10 --turnover-long-list 60 --divergence-window-list 10,20,60 --benchmark-list 000300.SH,000905.SH,000852.SH --benchmark-ma-list 120,250 --workers 10 --progress-every 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies import ma_demo_strategy_csv as ma  # noqa: E402
from strategies.alpha_v6_research_strategy_csv import (
    compute_alpha_v6_signals,
    prepare_benchmark_regime,
    VALID_VARIANTS,
)
from scripts.common.constants import DEFAULT_BENCHMARK_LIST  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.metrics import calc_metrics_from_returns  # noqa: E402
from scripts.common.validation import parse_list, parse_int_list, parse_workers, parse_date_yyyymmdd  # noqa: E402
from scripts.common.wf_batch_shared import (  # noqa: E402
    WFValidateConfig,
    build_variant_param_combos,
    calc_portfolio_period_metrics,
    calc_train_score,
    get_test_years,
    load_benchmark_cache,
    load_symbol_data,
    pass_train_filters_simple,
    run_alpha_frame,
    save_validate_outputs,
    test_selected_for_period,
    train_one_stock_all_years,
)

OUTPUT_DIR = PROJECT_ROOT / "backtests" / "walk_forward_alpha_v6_research_csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CFG = WFValidateConfig(
    alpha_version="alpha_v6",
    output_dir_name=str(OUTPUT_DIR),
    file_prefix="wf_alpha_v6_stock",
    report_title="Alpha v6 Walk-Forward Report",
    compute_signals_fn=compute_alpha_v6_signals,
    prepare_benchmark_regime_fn=prepare_benchmark_regime,
    valid_variants=VALID_VARIANTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha v6 walk-forward 样本外验证")

    parser.add_argument("--export-root", default=str(ma.DEFAULT_EXPORT_ROOT))
    parser.add_argument("--market", default="ALL")
    parser.add_argument("--security-type", default="stock")
    parser.add_argument("--train-start", default="20150101")
    parser.add_argument("--first-test-year", type=int, default=2021)
    parser.add_argument("--last-test-year", type=int, default=0)
    parser.add_argument("--end", default="")

    parser.add_argument("--alpha-variant-list", default=",".join(VALID_VARIANTS))
    parser.add_argument("--reversal-window-list", default="5,10,20")
    parser.add_argument("--vol-window-list", default="20,60,120")
    parser.add_argument("--turnover-short-list", default="10")
    parser.add_argument("--turnover-long-list", default="60")
    parser.add_argument("--divergence-window-list", default="10,20,60")
    parser.add_argument("--benchmark-list", default=DEFAULT_BENCHMARK_LIST)
    parser.add_argument("--benchmark-ma-list", default="120,250")

    parser.add_argument("--warmup-buffer", type=int, default=10)
    parser.add_argument("--portfolio-size", type=int, default=20)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission", type=float, default=0.0001)
    parser.add_argument("--sell-tax", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0)

    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--min-train-trades", type=int, default=4)
    parser.add_argument("--max-train-drawdown", type=float, default=-0.55)
    parser.add_argument("--min-train-sharpe", type=float, default=0.2)
    parser.add_argument("--min-train-annual-return", type=float, default=0.02)
    parser.add_argument("--max-train-volatility", type=float, default=0.0)
    parser.add_argument("--min-train-calmar", type=float, default=0.0)

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--workers", type=parse_workers, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = datetime.now()

    param_combos = build_variant_param_combos(
        parse_list(args.alpha_variant_list, upper=False),
        parse_int_list(args.reversal_window_list),
        parse_int_list(args.vol_window_list),
        parse_int_list(args.turnover_short_list),
        parse_int_list(args.turnover_long_list),
        parse_int_list(args.divergence_window_list),
    )

    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root

    catalog = ma.scan_qmt_export(export_root)
    if args.market != "ALL":
        catalog = catalog[catalog["market"] == args.market]
    if args.security_type != "ALL":
        catalog = catalog[catalog["security_type"] == args.security_type]
    if args.limit > 0:
        catalog = catalog.head(args.limit)

    test_years = get_test_years(args)

    # 预加载基准数据
    benchmark_cache = {}
    benchmark_list = [x.strip().upper() for x in args.benchmark_list.split(",") if x.strip()]
    benchmark_ma_list = parse_int_list(args.benchmark_ma_list)
    for bm_symbol in benchmark_list:
        for bm_ma in benchmark_ma_list:
            bm_csv_path, _, _, _ = ma.find_csv_for_stock(bm_symbol, export_root)
            bm_df = ma.load_qmt_price_csv(bm_csv_path, args.train_start, args.end)
            filter_df = prepare_benchmark_regime(bm_df, bm_ma)
            benchmark_cache[(bm_symbol, bm_ma)] = {
                "benchmark": bm_symbol,
                "benchmark_ma": bm_ma,
                "csv_path": str(bm_csv_path),
                "filter_df": filter_df,
            }

    logger.info("股票数量: %d", len(catalog))
    logger.info("参数组合: %d", len(param_combos))
    logger.info("测试年份: %s", test_years)
    logger.info("基准组合: %d", len(benchmark_cache))

    data_cache = {}
    selected_all_list = []
    test_detail_all_list = []
    portfolio_daily_list = []
    portfolio_periods = []
    running_equity = args.cash

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        benchmark_cache_data = {
            k: {
                "benchmark": v["benchmark"],
                "benchmark_ma": v["benchmark_ma"],
                "csv_path": v["csv_path"],
                "filter_df": v["filter_df"].reset_index(drop=True),
            }
            for k, v in benchmark_cache.items()
        }

        tasks = []
        for _, item in catalog.iterrows():
            tasks.append((
                item.to_dict(),
                test_years,
                args.train_start,
                args.end,
                param_combos,
                benchmark_cache_data,
                args.commission,
                args.sell_tax,
                args.slippage,
                args.cash,
                args.warmup_buffer,
                args.min_train_rows,
                args.min_train_trades,
                args.max_train_drawdown,
                args.min_train_sharpe,
                args.min_train_annual_return,
                args.max_train_volatility,
                args.min_train_calmar,
            ))

        all_candidates_by_year = {year: [] for year in test_years}
        completed_stocks = 0

        logger.info("并行训练：%d 进程处理 %d 只股票，跨 %d 个年份...", args.workers, len(tasks), len(test_years))
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(train_one_stock_all_years, task, compute_alpha_v6_signals): task[0]["symbol"]
                for task in tasks
            }
            for future in as_completed(futures):
                completed_stocks += 1
                if completed_stocks % max(args.progress_every, 1) == 0:
                    logger.info("训练进度：%d/%d 只股票完成", completed_stocks, len(tasks))

                try:
                    stock_results = future.result()
                    for year, candidate in stock_results.items():
                        all_candidates_by_year[year].append(candidate)
                except Exception as exc:
                    logger.error("训练错误 %s: %s", futures[future], repr(exc))

        for test_year in test_years:
            logger.info("=" * 60)
            logger.info("测试年份: %d", test_year)
            test_start_dt = pd.Timestamp(f"{test_year}-01-01")
            test_end_dt = pd.Timestamp(f"{test_year}-12-31")
            if args.end:
                end_dt = parse_date_yyyymmdd(args.end)
                if test_end_dt > end_dt:
                    test_end_dt = end_dt

            candidates = all_candidates_by_year.get(test_year, [])
            if not candidates:
                logger.warning("%d: 无候选，跳过", test_year)
                portfolio_periods.append({"test_year": test_year, "portfolio_size_actual": 0})
                continue

            candidates_df = pd.DataFrame(candidates)
            candidates_df = candidates_df.sort_values(
                ["train_score", "train_sharpe", "train_annual_return"],
                ascending=False,
            )
            selected = candidates_df.drop_duplicates(subset=["symbol"], keep="first")
            selected = selected.head(args.portfolio_size).reset_index(drop=True)
            selected["selected_rank"] = range(1, len(selected) + 1)
            selected["test_year"] = test_year
            selected_all_list.append(selected)
            logger.info("%d: 训练候选 %d 条，选中 %d 只", test_year, len(candidates_df), len(selected))

            for csv_path in selected["csv_path"].unique():
                if csv_path not in data_cache:
                    data_cache[csv_path] = load_symbol_data(Path(csv_path), args.train_start, args.end)

            detail, returns_df = test_selected_for_period(
                selected, data_cache, benchmark_cache,
                test_start_dt, test_end_dt, args,
                compute_alpha_v6_signals,
            )

            if not detail.empty:
                detail["test_year"] = test_year
                test_detail_all_list.append(detail)

            if not returns_df.empty:
                portfolio_ret = returns_df.mean(axis=1)
                equity = running_equity * (1.0 + portfolio_ret).cumprod()
                daily_df = pd.DataFrame({
                    "date": returns_df.index,
                    "portfolio_ret": portfolio_ret.values,
                    "equity": equity.values,
                    "test_year": test_year,
                })
                portfolio_daily_list.append(daily_df)
                period_metrics = calc_portfolio_period_metrics(returns_df, test_year, args.cash, portfolio_ret)
                portfolio_periods.append(period_metrics)
                running_equity = equity.iloc[-1]
                logger.info("%d: return=%.4f, sharpe=%.4f",
                            test_year, period_metrics.get('total_return', np.nan),
                            period_metrics.get('sharpe', np.nan))
    else:
        # 单进程模式
        skipped_load = 0
        for _, item in catalog.iterrows():
            csv_path = item["csv_path"]
            if csv_path not in data_cache:
                df = load_symbol_data(Path(csv_path), args.train_start, args.end)
                if df is not None and not df.empty:
                    data_cache[csv_path] = df
                else:
                    skipped_load += 1

        logger.info("已加载 %d 只股票数据 (跳过 %d 只数据不足)", len(data_cache), skipped_load)

        for test_year in test_years:
            logger.info("=" * 60)
            logger.info("测试年份: %d", test_year)
            train_start_dt = parse_date_yyyymmdd(args.train_start)
            train_end_dt = pd.Timestamp(f"{test_year - 1}-12-31")
            test_start_dt = pd.Timestamp(f"{test_year}-01-01")
            test_end_dt = pd.Timestamp(f"{test_year}-12-31")
            if args.end:
                end_dt = parse_date_yyyymmdd(args.end)
                if test_end_dt > end_dt:
                    test_end_dt = end_dt

            candidates = []
            for _, item in catalog.iterrows():
                symbol = item["symbol"]
                csv_path = item["csv_path"]

                if csv_path not in data_cache:
                    continue
                df = data_cache[csv_path]
                train_df = df[(df["date"] >= train_start_dt) & (df["date"] <= train_end_dt)].copy()
                if len(train_df) < args.min_train_rows:
                    continue

                for alpha_variant, reversal_window, vol_window, turnover_short, turnover_long, divergence_window in param_combos:
                    for (bm_symbol, bm_ma), bm_data in benchmark_cache.items():
                        filter_df = bm_data["filter_df"]
                        filter_train = filter_df[(filter_df["date"] >= train_start_dt) & (filter_df["date"] <= train_end_dt)]

                        try:
                            result = run_alpha_frame(
                                train_df, filter_train,
                                compute_alpha_v6_signals,
                                alpha_variant, reversal_window, vol_window,
                                turnover_short, turnover_long, divergence_window,
                                args.commission, args.sell_tax, args.slippage,
                            )
                            ret = result["strategy_ret"]
                            pos = result["position"]
                            metrics = calc_metrics_from_returns(ret, position=pos, stock_ret=result["stock_ret"])
                            metrics["excess_vs_buy_hold_total_return"] = metrics.pop("excess_total_return", np.nan)

                            if not pass_train_filters_simple(
                                metrics, args.min_train_rows, args.min_train_trades,
                                args.max_train_drawdown, args.min_train_sharpe,
                                args.min_train_annual_return, args.max_train_volatility,
                                args.min_train_calmar,
                            ):
                                continue

                            score = calc_train_score(metrics)
                            candidates.append({
                                "symbol": symbol,
                                "csv_path": csv_path,
                                "alpha_variant": alpha_variant,
                                "reversal_window": reversal_window,
                                "vol_window": vol_window,
                                "turnover_short": turnover_short,
                                "turnover_long": turnover_long,
                                "divergence_window": divergence_window,
                                "benchmark": bm_symbol,
                                "benchmark_ma": bm_ma,
                                "train_score": score,
                                "train_annual_return": metrics["annual_return"],
                                "train_sharpe": metrics["sharpe"],
                                "train_max_drawdown": metrics["max_drawdown"],
                                "train_annual_volatility": metrics["annual_volatility"],
                                "train_excess_vs_buy_hold": metrics["excess_vs_buy_hold_total_return"],
                                "train_days": metrics["days"],
                                "train_trade_count": metrics["trade_count"],
                            })
                        except Exception:
                            continue

            if not candidates:
                logger.warning("%d: 无候选，跳过", test_year)
                portfolio_periods.append({"test_year": test_year, "portfolio_size_actual": 0})
                continue

            cand_df = pd.DataFrame(candidates)
            cand_df = cand_df.sort_values(["train_score", "train_sharpe", "train_annual_return"], ascending=False)
            cand_df = cand_df.drop_duplicates(subset=["symbol"], keep="first")
            selected = cand_df.head(args.portfolio_size).copy()
            selected["selected_rank"] = range(1, len(selected) + 1)
            selected["test_year"] = test_year
            selected_all_list.append(selected)
            logger.info("%d: 选中 %d 只", test_year, len(selected))

            detail, returns_df = test_selected_for_period(
                selected, data_cache, benchmark_cache,
                test_start_dt, test_end_dt, args,
                compute_alpha_v6_signals,
            )

            if not detail.empty:
                detail["test_year"] = test_year
                test_detail_all_list.append(detail)

            if not returns_df.empty:
                portfolio_ret = returns_df.mean(axis=1)
                equity = running_equity * (1.0 + portfolio_ret).cumprod()
                daily_df = pd.DataFrame({
                    "date": returns_df.index,
                    "portfolio_ret": portfolio_ret.values,
                    "equity": equity.values,
                    "test_year": test_year,
                })
                portfolio_daily_list.append(daily_df)
                period_metrics = calc_portfolio_period_metrics(returns_df, test_year, args.cash, portfolio_ret)
                portfolio_periods.append(period_metrics)
                running_equity = equity.iloc[-1]
                logger.info("%d: return=%.4f, sharpe=%.4f",
                            test_year, period_metrics.get('total_return', np.nan),
                            period_metrics.get('sharpe', np.nan))

    # 汇总
    selected_all = pd.concat(selected_all_list, ignore_index=True) if selected_all_list else pd.DataFrame()
    test_detail_all = pd.concat(test_detail_all_list, ignore_index=True) if test_detail_all_list else pd.DataFrame()
    portfolio_daily_all = pd.concat(portfolio_daily_list, ignore_index=True) if portfolio_daily_list else pd.DataFrame()

    save_validate_outputs(CFG, selected_all, test_detail_all, portfolio_daily_all, portfolio_periods, args, OUTPUT_DIR)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("总耗时: %.1fs", elapsed)


if __name__ == "__main__":
    setup_cli_logging()
    main()
