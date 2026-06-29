# backtests/ — 回测输出目录

## 说明

本公开仓库 **不包含完整回测输出**。`backtests/` 是运行研究流水线后的本地输出目录。

## 输出结构

运行流水线后，`backtests/` 目录结构如下：

```
backtests/
  batch_<strategy>_csv/                    # 全市场批量回测结果
    batch_<strategy>_summary_*.csv         # 汇总表
    batch_<strategy>_top50_*.csv           # Top50 候选
    batch_<strategy>_skipped_*.csv         # 跳过的股票

  walk_forward_<strategy>_csv/             # Walk-forward 验证结果
    wf_<strategy>_stock_<symbol>_*.csv     # 单股票 walk-forward 结果
    selected_by_year_*.csv                 # 按年份选股结果
    portfolio_daily_*.csv                  # 组合每日收益

  walk_forward_<strategy>_analysis/        # 分析报告
    <strategy>_wf_analysis_*.txt           # 文本报告
    *.png                                  # 图表

  strategy_diagnosis/                      # 策略诊断
    <strategy>_diagnosis_*/                # 按 run_id 组织
      *_recommendations.txt               # 诊断建议
      *.png                               # 诊断图表

  strategy_robustness/                     # 稳健性验证
    <strategy>_robustness_*/               # 按 run_id 组织
      *_report.txt                        # 稳健性报告
      *.png                               # 稳健性图表

  portfolio_backtest_csv/                  # 组合约束回测
    portfolio_<run_id>/
      portfolio_daily.csv                 # 组合每日收益
      portfolio_trades.csv                # 交易明细
      portfolio_positions_daily.csv       # 每日持仓
      portfolio_rebalance_log.csv         # 调仓日志
      portfolio_period_summary.csv        # 区间汇总
      portfolio_backtest_report.txt       # 报告
      *.png                               # 图表
```

## 注意事项

- 回测输出包含具体股票代码、收益数据和交易明细，不适合公开分发。
- 本仓库的 `.gitignore` 已配置忽略 `backtests/` 下的所有文件。
- 如需分享回测结果，请只分享脱敏后的汇总指标，不要分享完整明细。
