# -*- coding: utf-8 -*-
"""
ma_demo_strategy_csv.py

用途：
1. 扫描 QMT 导出的 qmt_export 目录，识别 SH / SZ 下的股票和指数 CSV。
2. 根据 stock 代码自动定位 CSV 文件，例如：
   - 000001.SZ -> data/qmt_export/SZ/price_000001.csv
   - 000001.SH -> data/qmt_export/SH/price_000001.csv
3. 使用 QMT 导出的日线 CSV 做最小均线策略回测。
4. 每次回测结果保存到独立文件夹。

运行示例：
python strategies\\ma_demo_strategy_csv.py --scan-only
python strategies\\ma_demo_strategy_csv.py --stock 000001.SZ --fast 20 --slow 120 --start 20150101
python strategies\\ma_demo_strategy_csv.py --stock 000001.SH --fast 20 --slow 120 --start 20150101
python strategies\\ma_demo_strategy_csv.py --stock 000009.SZ --fast 20 --slow 120 --start 20150101
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.constants import TRADING_DAYS_PER_YEAR, SQRT_TRADING_DAYS_PER_YEAR  # noqa: E402
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402
from scripts.common.metrics import max_drawdown_from_equity  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "data" / "qmt_export"
DEFAULT_PARQUET_ROOT = PROJECT_ROOT / "data" / "qmt_parquet"
CATALOG_PATH = PROJECT_ROOT / "data" / "qmt_export_catalog.csv"
BACKTEST_DIR = PROJECT_ROOT / "backtests"


def classify_security(market: str, code: str) -> str:
    """
    根据 A 股常见代码规则粗略分类。
    这里只用于目录识别和结果标注，不作为严格证券类型判定。
    """
    market = market.upper()
    code = code.zfill(6)

    if market == "SH":
        if code.startswith("000"):
            return "index"
        if code.startswith(("600", "601", "603", "605", "688", "689")):
            return "stock"
        return "other"

    if market == "SZ":
        if code.startswith("399"):
            return "index"
        if code.startswith(("000", "001", "002", "003", "300", "301")):
            return "stock"
        return "other"

    return "unknown"


def scan_qmt_export(export_root: Path) -> pd.DataFrame:
    """
    扫描 qmt_export 目录。
    目录结构应类似：
    qmt_export/
      SH/
        price_000001.csv
      SZ/
        price_000001.csv
    """
    export_root = Path(export_root)

    if not export_root.exists():
        raise FileNotFoundError(f"导出目录不存在：{export_root}")

    rows = []

    for market in ["SH", "SZ"]:
        market_dir = export_root / market
        if not market_dir.exists():
            continue

        for csv_path in sorted(market_dir.glob("price_*.csv")):
            code = csv_path.stem.replace("price_", "").zfill(6)
            symbol = f"{code}.{market}"
            sec_type = classify_security(market, code)

            rows.append(
                {
                    "symbol": symbol,
                    "code": code,
                    "market": market,
                    "security_type": sec_type,
                    "csv_path": str(csv_path),
                }
            )

    catalog = pd.DataFrame(rows)

    if catalog.empty:
        raise RuntimeError(f"没有在 {export_root} 下扫描到 price_*.csv 文件。")

    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(CATALOG_PATH, index=False, encoding="utf-8-sig")

    print("\n扫描完成。")
    print(f"导出目录：{export_root}")
    print(f"识别文件数：{len(catalog)}")
    print(f"目录索引已保存：{CATALOG_PATH}")

    print("\n按市场统计：")
    print(catalog.groupby("market").size())

    print("\n按类型统计：")
    print(catalog.groupby(["market", "security_type"]).size())

    return catalog


def normalize_symbol(stock: str) -> tuple[str, str | None]:
    """
    支持：
    000001.SZ
    000001.SH
    SZ_000001
    SH_000001
    sz.000001
    sh.000001

    如果只传 000001，则 market 返回 None，后续如果 SH/SZ 都存在会要求用户补后缀。
    """
    s = stock.strip().upper().replace("-", "_")

    if "_" in s:
        a, b = s.split("_", 1)
        if a in {"SH", "SZ"}:
            return b.zfill(6), a

    if "." in s:
        a, b = s.split(".", 1)

        if b in {"SH", "SZ"}:
            return a.zfill(6), b

        if a in {"SH", "SZ"}:
            return b.zfill(6), a

    if s.isdigit():
        return s.zfill(6), None

    raise ValueError(f"无法识别 stock 格式：{stock}。请使用 000001.SZ 或 000001.SH。")


def find_csv_for_stock(stock: str, export_root: Path) -> tuple[Path, str, str, str]:
    """
    根据 stock 自动找 CSV。
    返回：csv_path, symbol, market, security_type
    """
    code, market = normalize_symbol(stock)
    export_root = Path(export_root)

    if market is not None:
        csv_path = export_root / market / f"price_{code}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到文件：{csv_path}")

        symbol = f"{code}.{market}"
        sec_type = classify_security(market, code)
        return csv_path, symbol, market, sec_type

    # 没有市场后缀时，在 SH / SZ 两边找
    candidates = []
    for m in ["SH", "SZ"]:
        p = export_root / m / f"price_{code}.csv"
        if p.exists():
            candidates.append((p, f"{code}.{m}", m, classify_security(m, code)))

    if not candidates:
        raise FileNotFoundError(
            f"没有找到 {code} 对应的 CSV。请确认文件是否存在于 SH 或 SZ 文件夹。"
        )

    if len(candidates) > 1:
        msg = "\n".join([f"- {symbol}: {path}" for path, symbol, _, _ in candidates])
        raise RuntimeError(
            f"代码 {code} 在多个市场都存在，必须写清楚后缀，例如 {code}.SZ 或 {code}.SH。\n候选项：\n{msg}"
        )

    return candidates[0]


def read_csv_auto_encoding(csv_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]

    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"无法读取 CSV：{csv_path}，最后错误：{last_error!r}")


def load_qmt_price_csv(csv_path: Path, start: str, end: str) -> pd.DataFrame:
    # 优先读取 Parquet 格式（更快）
    csv_path = Path(csv_path)
    parquet_path = (
        csv_path.parent.parent.parent / "qmt_parquet"
        / csv_path.parent.name / csv_path.with_suffix(".parquet").name
    )

    if parquet_path.exists():
        raw = pd.read_parquet(parquet_path, engine="pyarrow")
    else:
        raw = read_csv_auto_encoding(csv_path)

    raw.columns = [str(c).strip().lower() for c in raw.columns]

    required = {"timetag", "open", "high", "low", "close"}
    missing = required - set(raw.columns)

    if missing:
        raise RuntimeError(f"CSV 缺少必要字段：{missing}。当前字段：{list(raw.columns)}")

    df = pd.DataFrame()

    # timetag 可能被 pandas 读成 int，也可能读成 float，统一转成 8 位日期字符串
    timetag = raw["timetag"].dropna()
    raw = raw.loc[timetag.index].copy()

    date_str = (
        raw["timetag"]
        .astype(float)
        .astype(int)
        .astype(str)
        .str.zfill(8)
    )

    df["date"] = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
    df["open"] = pd.to_numeric(raw["open"], errors="coerce")
    df["high"] = pd.to_numeric(raw["high"], errors="coerce")
    df["low"] = pd.to_numeric(raw["low"], errors="coerce")
    df["close"] = pd.to_numeric(raw["close"], errors="coerce")

    if "volumn" in raw.columns:
        df["volume"] = pd.to_numeric(raw["volumn"], errors="coerce")
    elif "volume" in raw.columns:
        df["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    else:
        df["volume"] = np.nan

    if "amount" in raw.columns:
        df["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    else:
        df["amount"] = np.nan

    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[df["close"] > 0]
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.set_index("date", drop=False)

    if start:
        start_dt = pd.to_datetime(start, format="%Y%m%d", errors="coerce")
        if pd.isna(start_dt):
            raise ValueError(f"start 日期格式错误：{start}，应为 YYYYMMDD，例如 20150101。")
        df = df[df.index >= start_dt]

    if end:
        end_dt = pd.to_datetime(end, format="%Y%m%d", errors="coerce")
        if pd.isna(end_dt):
            raise ValueError(f"end 日期格式错误：{end}，应为 YYYYMMDD，例如 20260514。")
        df = df[df.index <= end_dt]

    if df.empty:
        raise RuntimeError("按指定日期过滤后，数据为空。")

    return df


def calc_metrics(result: pd.DataFrame) -> dict:
    strategy_ret = result["strategy_ret"].dropna()
    equity = result["equity"].dropna()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    days = max(len(strategy_ret), 1)
    annual_return = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1.0

    std = strategy_ret.std()
    sharpe = np.nan if std == 0 or np.isnan(std) else strategy_ret.mean() / std * SQRT_TRADING_DAYS_PER_YEAR

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": max_drawdown_from_equity(equity),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "trade_count": int(result["position"].diff().abs().fillna(0).sum()),
        "final_equity": float(equity.iloc[-1]),
    }


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    fast: int,
    slow: int,
    cash: float,
    commission: float,
    sell_tax: float,
    slippage: float,
) -> tuple[pd.DataFrame, dict]:
    result = df.copy()

    result["fast_ma"] = result["close"].rolling(fast).mean()
    result["slow_ma"] = result["close"].rolling(slow).mean()

    # 当日收盘后产生信号
    result["signal"] = np.where(result["fast_ma"] > result["slow_ma"], 1, 0)

    # 次日持仓，避免未来函数
    result["position"] = result["signal"].shift(1).fillna(0)

    result["stock_ret"] = result["close"].pct_change().fillna(0)

    # 换仓成本：这里是简化模型，适合研究阶段。
    # position 从 0 到 1：买入，扣佣金和滑点
    # position 从 1 到 0：卖出，扣佣金、印花税和滑点
    pos_change = result["position"].diff().fillna(result["position"])
    buy_turnover = pos_change.clip(lower=0)
    sell_turnover = (-pos_change).clip(lower=0)

    result["cost"] = (
        buy_turnover * (commission + slippage)
        + sell_turnover * (commission + sell_tax + slippage)
    )

    result["strategy_ret"] = result["position"] * result["stock_ret"] - result["cost"]

    result["equity"] = cash * (1.0 + result["strategy_ret"]).cumprod()
    result["buy_hold_equity"] = cash * (1.0 + result["stock_ret"]).cumprod()

    metrics = calc_metrics(result)

    print("\n" + "=" * 80)
    print(f"回测标的: {symbol}")
    print(f"数据区间: {result.index.min().date()} 至 {result.index.max().date()}")
    print(f"数据行数: {len(result)}")
    print(f"快均线: {fast}")
    print(f"慢均线: {slow}")
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
    print("=" * 80)

    return result, metrics


def save_outputs(
    result: pd.DataFrame,
    metrics: dict,
    symbol: str,
    security_type: str,
    csv_path: Path,
    fast: int,
    slow: int,
    start: str,
    end: str,
) -> None:
    safe_symbol = symbol.replace(".", "_")
    strategy_name = "ma_demo_strategy_csv"
    date_tag = f"{start or 'all'}_{end or 'latest'}"
    run_name = f"ma_demo_csv_{safe_symbol}_fast{fast}_slow{slow}_{date_tag}"

    run_dir = BACKTEST_DIR / strategy_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    result_csv_path = run_dir / f"{run_name}.csv"
    metrics_path = run_dir / f"{run_name}_metrics.txt"
    png_path = run_dir / f"{run_name}_equity.png"

    result.to_csv(result_csv_path, encoding="utf-8-sig")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"strategy_name: {strategy_name}\n")
        f.write(f"symbol: {symbol}\n")
        f.write(f"security_type: {security_type}\n")
        f.write(f"source_csv: {csv_path}\n")
        f.write(f"fast: {fast}\n")
        f.write(f"slow: {slow}\n")
        f.write(f"start_date: {result.index.min()}\n")
        f.write(f"end_date: {result.index.max()}\n")
        f.write(f"rows: {len(result)}\n\n")

        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.8f}\n")
            else:
                f.write(f"{key}: {value}\n")

    plt.figure(figsize=(12, 6))
    result["equity"].plot(label="MA strategy")
    result["buy_hold_equity"].plot(label="Buy and Hold")
    plt.title(f"MA CSV Backtest - {symbol}")
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
    parser = argparse.ArgumentParser(description="读取 QMT 导出 CSV 的均线回测")

    parser.add_argument("--scan-only", action="store_true", help="只扫描 qmt_export 目录并生成 catalog，不运行回测")
    parser.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT), help="QMT 导出的根目录，默认 data\\qmt_export")

    parser.add_argument("--stock", default="000001.SZ", help="证券代码，例如 000001.SZ / 000001.SH / 000009.SZ")
    parser.add_argument("--csv-file", default="", help="手动指定 CSV 文件；一般不需要")

    parser.add_argument("--start", default="20150101", help="开始日期，例如 20150101")
    parser.add_argument("--end", default="", help="结束日期，例如 20260514；留空表示不限制")

    parser.add_argument("--fast", type=int, default=20, help="快均线")
    parser.add_argument("--slow", type=int, default=120, help="慢均线")
    parser.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金")

    parser.add_argument("--commission", type=float, default=0.0001, help="佣金率，默认万一")
    parser.add_argument("--sell-tax", type=float, default=0.0005, help="卖出印花税，默认万五")
    parser.add_argument("--slippage", type=float, default=0.0, help="滑点率，默认 0")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root

    catalog = scan_qmt_export(export_root)

    if args.scan_only:
        return

    if args.fast >= args.slow:
        raise ValueError("fast 必须小于 slow。")

    if args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path

        if not csv_path.exists():
            raise FileNotFoundError(f"找不到手动指定的 CSV：{csv_path}")

        code, market = normalize_symbol(args.stock)
        symbol = f"{code}.{market}" if market else args.stock.upper()
        security_type = classify_security(market, code) if market else "unknown"
    else:
        csv_path, symbol, market, security_type = find_csv_for_stock(args.stock, export_root)

    logger.info("当前使用数据文件：")
    logger.info("symbol: %s", symbol)
    logger.info("security_type: %s", security_type)
    logger.info("csv_path: %s", csv_path)

    df = load_qmt_price_csv(csv_path, args.start, args.end)

    logger.info("数据预览：行数=%d，起始日期=%s，结束日期=%s",
                len(df), df.index.min().date(), df.index.max().date())
    logger.debug("数据尾部:\n%s", df.tail())

    if len(df) < args.slow + 10:
        raise RuntimeError(f"数据太少，当前只有 {len(df)} 行，无法回测。")

    result, metrics = run_backtest(
        df=df,
        symbol=symbol,
        fast=args.fast,
        slow=args.slow,
        cash=args.cash,
        commission=args.commission,
        sell_tax=args.sell_tax,
        slippage=args.slippage,
    )

    save_outputs(
        result=result,
        metrics=metrics,
        symbol=symbol,
        security_type=security_type,
        csv_path=csv_path,
        fast=args.fast,
        slow=args.slow,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception:
        logger.error("程序异常：", exc_info=True)