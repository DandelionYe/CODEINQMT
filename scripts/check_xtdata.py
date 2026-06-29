# -*- coding: utf-8 -*-
"""
check_xtdata.py

用途：
1. 测试 xtdata 是否能连接 QMT / miniQMT 行情服务
2. 下载少量日线历史数据
3. 读取本地历史行情
4. 将样例数据保存到 <PROJECT_ROOT>/data

运行前建议：
1. 启动 QMT 或 miniQMT
2. 登录账号
3. 确认行情连接正常
"""

import logging
from pathlib import Path
import sys

import pandas as pd
from xtquant import xtdata

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.common.logging_setup import setup_cli_logging  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CODE_LIST = [
    "000001.SZ",  # 平安银行
    "600519.SH",  # 贵州茅台
    "000300.SH",  # 沪深300
]

PERIOD = "1d"


def download_one(code: str) -> None:
    print(f"\n开始下载历史行情: {code}, period={PERIOD}")

    try:
        # 新版 xtquant 支持 incrementally=True
        xtdata.download_history_data(code, period=PERIOD, incrementally=True)
        print(f"下载完成: {code}")
    except TypeError:
        # 兼容旧版本 xtquant
        xtdata.download_history_data(code, PERIOD)
        print(f"下载完成: {code}")
    except Exception as exc:
        print(f"下载失败: {code}")
        print(repr(exc))


def read_history() -> dict:
    print("\n开始读取本地历史行情...")

    data = xtdata.get_market_data_ex(
        field_list=[],
        stock_list=CODE_LIST,
        period=PERIOD,
        count=20,
        dividend_type="none",
        fill_data=True,
    )

    print("返回对象类型:", type(data))
    print("返回代码列表:", list(data.keys()) if isinstance(data, dict) else "非 dict 返回")

    return data


def save_sample_data(data: dict) -> None:
    if not isinstance(data, dict) or not data:
        print("没有读取到有效数据。")
        return

    for code, df in data.items():
        print("\n" + "=" * 80)
        print(f"代码: {code}")
        print("数据类型:", type(df))

        if isinstance(df, pd.DataFrame):
            print("字段:", list(df.columns))
            print(df.tail())

            out_path = DATA_DIR / f"xtdata_sample_{code.replace('.', '_')}.csv"
            df.to_csv(out_path, encoding="utf-8-sig")
            print(f"已保存: {out_path}")
        else:
            print(df)


def main() -> None:
    print("Python executable:", sys.executable)
    print("Project root:", PROJECT_ROOT)
    print("Data dir:", DATA_DIR)

    print("\n重要提示：如果这里卡住或报连接错误，请先启动并登录 QMT / miniQMT。")

    for code in CODE_LIST:
        download_one(code)

    data = read_history()
    save_sample_data(data)

    print("\nxtdata 测试完成。")


if __name__ == "__main__":
    setup_cli_logging()
    try:
        main()
    except Exception:
        logger.error("程序异常：", exc_info=True)