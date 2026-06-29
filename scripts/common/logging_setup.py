# -*- coding: utf-8 -*-
"""
scripts/common/logging_setup.py

CLI 脚本的 logging 配置工具。

使用方式：
    from scripts.common.logging_setup import setup_cli_logging
    setup_cli_logging()
"""

from __future__ import annotations

import logging


def setup_cli_logging(level: int = logging.INFO) -> None:
    """为 CLI 脚本配置 logging：控制台输出，带时间戳。

    在每个脚本的 if __name__ == "__main__": 块中调用一次即可。
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
