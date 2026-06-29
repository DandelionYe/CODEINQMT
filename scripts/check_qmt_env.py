# -*- coding: utf-8 -*-

import logging
import sys

logger = logging.getLogger(__name__)

print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    import xtquant
    print("xtquant:", xtquant.__file__)
except Exception as e:
    print("xtquant import failed:", repr(e))

try:
    from xtquant import xtdata
    print("xtdata import ok")
except Exception as e:
    print("xtdata import failed:", repr(e))

try:
    from xtquant.xttrader import XtQuantTrader
    print("XtQuantTrader import ok")
except Exception as e:
    print("XtQuantTrader import failed:", repr(e))

try:
    from xtquant.qmttools import run_strategy_file
    print("qmttools.run_strategy_file import ok")
except Exception as e:
    print("qmttools import failed:", repr(e))