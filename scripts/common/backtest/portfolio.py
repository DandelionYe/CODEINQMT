# -*- coding: utf-8 -*-
"""
scripts/common/backtest/portfolio.py

共享组合回测模拟器。

从 portfolio_backtest_csv.py 提取的核心组合层回测逻辑：
- 年度再平衡（selected_by_year 驱动）
- 整手约束
- 最大持仓数
- 最大单票权重
- 佣金/滑点/最低佣金
- 超权自动减持

设计参考：
- 单资产引擎 engine.py 处理单标的 0/1 信号回测
- 本模块处理多资产组合层再平衡回测
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class PortfolioCostModel:
    """组合回测成本模型。

    commission_rate: 佣金费率（万分之几 / 10000）
    min_commission: 最低佣金（元）
    slippage_bps: 滑点（bps，1bp = 0.01%）
    lot_size: 整手大小（A 股为 100）
    """
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    slippage_bps: float = 5.0
    lot_size: int = 100

    def calc_cost(self, notional: float) -> tuple[float, float]:
        """计算交易成本。返回 (commission, slippage_cost)。"""
        if notional <= 0:
            return 0.0, 0.0
        commission = max(notional * self.commission_rate, self.min_commission)
        slippage_cost = notional * self.slippage_bps / 10000
        return commission, slippage_cost

    @classmethod
    def a_share_default(cls) -> PortfolioCostModel:
        """A 股默认成本模型。"""
        return cls(commission_rate=0.0003, min_commission=5.0, slippage_bps=5.0, lot_size=100)


# ---------------------------------------------------------------------------
# Price lookup utilities
# ---------------------------------------------------------------------------

def get_price_on_date(pdf: pd.DataFrame, date: pd.Timestamp, field: str) -> float | None:
    """获取指定日期的价格。"""
    avail = pdf[pdf["date"] == date]
    if avail.empty:
        return None
    val = avail.iloc[0].get(field)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)


def get_price_on_or_after(pdf: pd.DataFrame, date: pd.Timestamp, field: str) -> tuple[float, pd.Timestamp | None]:
    """获取指定日期或之后的第一个可用价格。"""
    avail = pdf[pdf["date"] >= date]
    if avail.empty:
        return 0.0, None
    val = avail.iloc[0].get(field)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0, None
    return float(val), avail.iloc[0]["date"]


# ---------------------------------------------------------------------------
# Portfolio Simulator
# ---------------------------------------------------------------------------

class PortfolioSimulator:
    """组合再平衡回测模拟器。

    状态驱动的事件循环：
    - 按年度从 selected_by_year 读取目标持仓
    - 每年第一个交易日执行再平衡
    - 每日检查最大权重约束，超权自动减持
    - 记录每日权益、交易、持仓明细、再平衡日志
    """

    def __init__(
        self,
        selected: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],
        wf_daily: pd.DataFrame,
        initial_cash: float,
        max_positions: int,
        max_weight: float,
        cost_model: PortfolioCostModel,
        price_field: str = "open",
        allow_partial: bool = True,
    ) -> None:
        self.selected = selected
        self.price_data = price_data
        self.wf_daily = wf_daily
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.max_weight = max_weight
        self.cost_model = cost_model
        self.price_field = price_field
        self.allow_partial = allow_partial

        # Mutable state
        self.cash: float = initial_cash
        self.holdings: dict[str, int] = {}
        self.last_prices: dict[str, float] = {}
        self.all_daily: list[dict] = []
        self.all_trades: list[dict] = []
        self.all_positions: list[dict] = []
        self.all_rebalance_log: list[dict] = []
        self.previous_equity: float = initial_cash

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """执行组合回测，返回 (daily_df, trades_df, positions_df, rebalance_df)。"""
        test_years, trading_calendar, stocks_by_year, rebalance_dates = self._prepare()

        for date in trading_calendar:
            year = int(date.year)
            if rebalance_dates.get(year) == date:
                self._execute_rebalance(date, year, stocks_by_year)

            market_value, equity, position_details = self._trim_overweights(date)
            daily_return = equity / self.previous_equity - 1 if self.previous_equity > 0 else 0.0
            self.previous_equity = equity
            self.all_daily.append({
                "date": date,
                "equity": equity,
                "cash": self.cash,
                "market_value": market_value,
                "daily_return": daily_return,
                "gross_exposure": market_value / equity if equity > 0 else 0.0,
                "net_exposure": market_value / equity if equity > 0 else 0.0,
                "position_count": len(position_details),
            })
            self.all_positions.extend(position_details)

        return self._build_output()

    def _prepare(self) -> tuple[list[int], list[pd.Timestamp], dict[int, list[str]], dict[int, pd.Timestamp]]:
        """准备交易日历、年度股票池、再平衡日期。"""
        wf_calendar = self.wf_daily.copy()
        wf_calendar["date"] = pd.to_datetime(wf_calendar["date"])
        test_years = sorted(int(y) for y in self.selected["test_year"].unique())
        trading_calendar = sorted(
            pd.Timestamp(d)
            for d in wf_calendar["date"].dropna().unique()
            if int(pd.Timestamp(d).year) in test_years
        )
        if not trading_calendar:
            raise ValueError("No trading calendar from walk-forward portfolio_daily.")

        stocks_by_year: dict[int, list[str]] = {}
        for year in test_years:
            year_sel = self.selected[self.selected["test_year"] == year].copy()
            if "selected_rank" in year_sel.columns:
                year_sel = year_sel.sort_values("selected_rank")
            targets = []
            for sym in year_sel["symbol"].tolist():
                if sym not in targets and sym in self.price_data:
                    targets.append(sym)
                if len(targets) >= self.max_positions:
                    break
            stocks_by_year[year] = targets

        rebalance_dates: dict[int, pd.Timestamp] = {}
        for year in test_years:
            dates = [d for d in trading_calendar if d.year == year]
            if dates:
                rebalance_dates[year] = dates[0]

        return test_years, trading_calendar, stocks_by_year, rebalance_dates

    def _exact_price(self, sym: str, date: pd.Timestamp, field: str) -> float | None:
        """查找指定股票在指定日期的价格。"""
        pdf = self.price_data.get(sym)
        if pdf is None:
            return None
        return get_price_on_date(pdf, date, field)

    def _value_portfolio(self, date: pd.Timestamp) -> tuple[float, float, list[dict]]:
        """计算组合市值和权益。返回 (market_value, equity, position_details)。"""
        market_value = 0.0
        details: list[dict] = []
        for sym, shares in self.holdings.items():
            if shares <= 0:
                continue
            price = self._exact_price(sym, date, "close")
            if price is None or price <= 0:
                price = self.last_prices.get(sym)
            else:
                self.last_prices[sym] = price
            if price is None or price <= 0:
                continue
            mv = shares * price
            market_value += mv
            details.append({
                "date": date,
                "symbol": sym,
                "shares": shares,
                "close": price,
                "market_value": mv,
                "weight": 0.0,
            })
        equity = self.cash + market_value
        for item in details:
            item["weight"] = item["market_value"] / equity if equity > 0 else 0.0
        return market_value, equity, details

    def _record_trade(
        self, date: pd.Timestamp, sym: str, side: str, price: float, shares: int, reason: str,
    ) -> None:
        """记录交易并更新现金。"""
        notional = shares * price
        commission, slippage_cost = self.cost_model.calc_cost(notional)
        if side == "buy":
            self.cash -= notional + commission + slippage_cost
        else:
            self.cash += notional - commission - slippage_cost
        self.last_prices[sym] = price
        self.all_trades.append({
            "trade_date": date,
            "symbol": sym,
            "side": side,
            "price": price,
            "shares": shares,
            "notional": notional,
            "commission": commission,
            "slippage_cost": slippage_cost,
            "cash_after": self.cash,
            "reason": reason,
        })

    def _execute_rebalance(
        self, date: pd.Timestamp, year: int, stocks_by_year: dict[int, list[str]],
    ) -> None:
        """执行年度再平衡。"""
        target_symbols = stocks_by_year.get(year, [])
        cash_before = self.cash
        _, equity_before, _ = self._value_portfolio(date)

        exec_prices: dict[str, float] = {}
        skipped = 0
        for sym in sorted(set(target_symbols) | set(self.holdings.keys())):
            price = self._exact_price(sym, date, self.price_field)
            if price is None or price <= 0:
                price = self._exact_price(sym, date, "close")
            if price is None or price <= 0:
                price = self.last_prices.get(sym)
            if price is None or price <= 0:
                skipped += 1
                continue
            exec_prices[sym] = price

        executable_targets = [s for s in target_symbols if s in exec_prices]
        target_weight = (
            min(1.0 / len(executable_targets), self.max_weight) if executable_targets else 0.0
        )
        target_shares: dict[str, int] = {}
        for sym in executable_targets:
            target_value = equity_before * target_weight
            target_shares[sym] = int(target_value / exec_prices[sym] / self.cost_model.lot_size) * self.cost_model.lot_size

        bought_count = 0
        sold_count = 0

        # Sell first
        for sym in list(self.holdings.keys()):
            current = self.holdings.get(sym, 0)
            target = target_shares.get(sym, 0)
            sell_shares = current - target
            if sell_shares <= 0:
                continue
            price = exec_prices.get(sym)
            if price is None:
                skipped += 1
                continue
            self._record_trade(date, sym, "sell", price, sell_shares, "yearly_rebalance_sell")
            remaining = current - sell_shares
            if remaining > 0:
                self.holdings[sym] = remaining
            else:
                del self.holdings[sym]
            sold_count += 1

        # Then buy
        for sym in executable_targets:
            current = self.holdings.get(sym, 0)
            target = target_shares.get(sym, 0)
            buy_shares = target - current
            if buy_shares <= 0:
                continue
            price = exec_prices[sym]
            while buy_shares > 0:
                notional = buy_shares * price
                commission, slippage_cost = self.cost_model.calc_cost(notional)
                if notional + commission + slippage_cost <= self.cash:
                    break
                if not self.allow_partial:
                    buy_shares = 0
                    break
                buy_shares -= self.cost_model.lot_size
            if buy_shares <= 0:
                skipped += 1
                continue
            self._record_trade(date, sym, "buy", price, buy_shares, "yearly_rebalance_buy")
            self.holdings[sym] = self.holdings.get(sym, 0) + buy_shares
            bought_count += 1

        _, equity_after, _ = self._value_portfolio(date)
        self.all_rebalance_log.append({
            "rebalance_date": date,
            "test_year": year,
            "target_count": len(target_symbols),
            "bought_count": bought_count,
            "sold_count": sold_count,
            "skipped_count": skipped,
            "cash_before": cash_before,
            "cash_after": self.cash,
            "equity_after": equity_after,
            "notes": f"yearly_rebalance_target_weight={target_weight:.4f}",
        })

    def _trim_overweights(self, date: pd.Timestamp) -> tuple[float, float, list[dict]]:
        """检查并减持超权持仓（最多 5 轮）。"""
        market_value, equity, details = self._value_portfolio(date)
        for _ in range(5):
            overweight = [p for p in details if p["weight"] > self.max_weight + 1e-9]
            if not overweight:
                return market_value, equity, details
            changed = False
            for pos in sorted(overweight, key=lambda p: p["weight"], reverse=True):
                sym = pos["symbol"]
                shares = self.holdings.get(sym, 0)
                price = pos["close"]
                if shares <= 0 or price <= 0:
                    continue
                excess_value = pos["market_value"] - equity * self.max_weight
                sell_shares = int(np.ceil(excess_value / price / self.cost_model.lot_size)) * self.cost_model.lot_size
                sell_shares = min(sell_shares, shares)
                if sell_shares <= 0:
                    continue
                self._record_trade(date, sym, "sell", price, sell_shares, "max_weight_trim")
                remaining = shares - sell_shares
                if remaining > 0:
                    self.holdings[sym] = remaining
                else:
                    del self.holdings[sym]
                changed = True
            market_value, equity, details = self._value_portfolio(date)
            if not changed:
                return market_value, equity, details
        return market_value, equity, details

    def _build_output(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """将累积记录组装为 DataFrame 输出。"""
        daily_df = pd.DataFrame(self.all_daily)
        if daily_df.empty:
            raise ValueError("No portfolio daily rows generated.")

        trades_df = (
            pd.DataFrame(self.all_trades) if self.all_trades else pd.DataFrame(
                columns=["trade_date", "symbol", "side", "price", "shares", "notional",
                         "commission", "slippage_cost", "cash_after", "reason"])
        )
        positions_df = (
            pd.DataFrame(self.all_positions) if self.all_positions else pd.DataFrame(
                columns=["date", "symbol", "shares", "close", "market_value", "weight"])
        )
        rebalance_df = (
            pd.DataFrame(self.all_rebalance_log) if self.all_rebalance_log else pd.DataFrame(
                columns=["rebalance_date", "test_year", "target_count", "bought_count",
                         "sold_count", "skipped_count", "cash_before", "cash_after",
                         "equity_after", "notes"])
        )
        return daily_df, trades_df, positions_df, rebalance_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_yearly_rebalance_backtest(
    selected: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    wf_daily: pd.DataFrame,
    initial_cash: float,
    max_positions: int,
    max_weight: float,
    lot_size: int,
    commission_rate: float,
    min_commission: float,
    slippage_bps: float,
    price_field: str,
    allow_partial: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """从 walk-forward selected_by_year 执行年度再平衡组合回测。

    与 portfolio_backtest_csv.py 中原实现行为完全一致。
    """
    cost_model = PortfolioCostModel(
        commission_rate=commission_rate,
        min_commission=min_commission,
        slippage_bps=slippage_bps,
        lot_size=lot_size,
    )
    simulator = PortfolioSimulator(
        selected=selected,
        price_data=price_data,
        wf_daily=wf_daily,
        initial_cash=initial_cash,
        max_positions=max_positions,
        max_weight=max_weight,
        cost_model=cost_model,
        price_field=price_field,
        allow_partial=allow_partial,
    )
    return simulator.run()


# ---------------------------------------------------------------------------
# Summary helpers (also extracted from portfolio_backtest_csv.py)
# ---------------------------------------------------------------------------

def build_period_summary(
    daily_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    test_years: list[int],
) -> pd.DataFrame:
    """按年度构建组合指标摘要。"""
    from scripts.common.metrics import calc_portfolio_metrics

    rows: list[dict] = []
    trade_data = trades_df.copy()
    if not trade_data.empty:
        trade_data["trade_date"] = pd.to_datetime(trade_data["trade_date"])

    for year in test_years:
        year_mask = daily_df["date"].dt.year == year
        year_daily = daily_df.loc[year_mask, "daily_return"]
        if year_daily.empty:
            continue
        if trade_data.empty:
            year_trades = trade_data
        else:
            year_trades = trade_data[trade_data["trade_date"].dt.year == year]
        year_commission = year_trades["commission"].sum() if not year_trades.empty else 0
        year_slippage = year_trades["slippage_cost"].sum() if not year_trades.empty else 0
        year_notional = year_trades["notional"].abs().sum() if not year_trades.empty else 0
        avg_equity = daily_df.loc[year_mask, "equity"].mean()
        year_turnover = year_notional / avg_equity if avg_equity and avg_equity > 0 else np.nan
        m = calc_portfolio_metrics(year_daily, year_commission, year_slippage, year_turnover)
        rows.append({
            "period": str(year), "start_date": daily_df.loc[year_mask, "date"].min(),
            "end_date": daily_df.loc[year_mask, "date"].max(), **m,
        })
    total_commission = trade_data["commission"].sum() if not trade_data.empty else 0
    total_slippage = trade_data["slippage_cost"].sum() if not trade_data.empty else 0
    total_notional = trade_data["notional"].abs().sum() if not trade_data.empty else 0
    avg_equity = daily_df["equity"].mean()
    turnover = total_notional / avg_equity if avg_equity and avg_equity > 0 else np.nan
    all_returns = daily_df["daily_return"]
    m = calc_portfolio_metrics(all_returns, total_commission, total_slippage, turnover)
    rows.append({
        "period": "overall", "start_date": daily_df["date"].min(),
        "end_date": daily_df["date"].max(), **m,
    })
    return pd.DataFrame(rows)


def build_vs_walkforward(
    constrained_daily: pd.DataFrame,
    wf_daily: pd.DataFrame,
) -> pd.DataFrame:
    """构建受限组合 vs 原始 walk-forward 的对比表。"""
    from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR

    wf_daily = wf_daily.copy()
    wf_daily["date"] = pd.to_datetime(wf_daily["date"])
    wf_daily["portfolio_ret"] = pd.to_numeric(wf_daily["portfolio_ret"], errors="coerce")

    def metrics_from_returns(ret: pd.Series) -> dict:
        if ret.empty:
            return {"total_return": np.nan, "annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
        n = len(ret)
        total = (1 + ret).prod() - 1
        annual = (1 + total) ** (TRADING_DAYS_PER_YEAR / n) - 1
        vol = ret.std() * SQRT_TRADING_DAYS_PER_YEAR
        sharpe = annual / vol if vol > 0 else np.nan
        eq = (1 + ret).cumprod()
        dd = (eq - eq.cummax()) / eq.cummax()
        return {"total_return": total, "annual_return": annual, "sharpe": sharpe, "max_drawdown": dd.min()}

    c_all = metrics_from_returns(constrained_daily["daily_return"])
    w_all = metrics_from_returns(wf_daily["portfolio_ret"])
    rows: list[dict] = []
    for metric in ["total_return", "annual_return", "sharpe", "max_drawdown"]:
        rows.append({
            "metric": metric,
            "constrained_portfolio": c_all.get(metric, np.nan),
            "original_walk_forward": w_all.get(metric, np.nan),
            "difference": c_all.get(metric, 0) - w_all.get(metric, 0),
        })
    return pd.DataFrame(rows)
