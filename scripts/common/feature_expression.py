# -*- coding: utf-8 -*-
"""
feature_expression.py

轻量因子表达式层，借鉴 Qlib ops.py 的设计。
用于将硬编码的 compute_alpha_v*_signals() 迁移为可登记、可组合、可复用的表达式定义。

支持的表达式对象：
  Field, PctChange, Shift, Neg, AsInt, Abs,
  RollingMean, RollingStd, RollingMax, RollingMin, ZScore,
  Add, Sub, Mul, Div, Gt, Ge, Lt, Le, Eq, And, Or, Not,
  Where, CSRank

使用方式：
  expr = Neg(PctChange(Field("close"), 10))
  result = expr.eval(df)  # df 必须包含 "close" 列

运算符重载：
  a + b -> Add(a, b)
  a > b -> Gt(a, b)
  ~a    -> Not(a)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 共享函数
# ---------------------------------------------------------------------------

def normalize_zscore(series: pd.Series) -> pd.Series:
    """对 Series 做 z-score 标准化。std=0 时返回全 0。"""
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - series.mean()) / std


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class Expression(ABC):
    """表达式基类。所有因子表达式必须实现 eval(df) -> pd.Series。"""

    @abstractmethod
    def eval(self, df: pd.DataFrame) -> pd.Series:
        """在 DataFrame df 上求值，返回与 df 等长的 Series。"""
        ...

    # 运算符重载 ---------------------------------------------------------------
    def __add__(self, other):  # type: ignore[no-untyped-def]
        return Add(self, _ensure_expr(other))

    def __radd__(self, other):  # type: ignore[no-untyped-def]
        return Add(_ensure_expr(other), self)

    def __sub__(self, other):  # type: ignore[no-untyped-def]
        return Sub(self, _ensure_expr(other))

    def __rsub__(self, other):  # type: ignore[no-untyped-def]
        return Sub(_ensure_expr(other), self)

    def __mul__(self, other):  # type: ignore[no-untyped-def]
        return Mul(self, _ensure_expr(other))

    def __rmul__(self, other):  # type: ignore[no-untyped-def]
        return Mul(_ensure_expr(other), self)

    def __truediv__(self, other):  # type: ignore[no-untyped-def]
        return Div(self, _ensure_expr(other))

    def __rtruediv__(self, other):  # type: ignore[no-untyped-def]
        return Div(_ensure_expr(other), self)

    def __neg__(self):  # type: ignore[no-untyped-def]
        return Neg(self)

    def __gt__(self, other):  # type: ignore[no-untyped-def]
        return Gt(self, _ensure_expr(other))

    def __ge__(self, other):  # type: ignore[no-untyped-def]
        return Ge(self, _ensure_expr(other))

    def __lt__(self, other):  # type: ignore[no-untyped-def]
        return Lt(self, _ensure_expr(other))

    def __le__(self, other):  # type: ignore[no-untyped-def]
        return Le(self, _ensure_expr(other))

    def __eq__(self, other):  # type: ignore[no-untyped-def]
        return Eq(self, _ensure_expr(other))

    def __and__(self, other):  # type: ignore[no-untyped-def]
        return And(self, _ensure_expr(other))

    def __or__(self, other):  # type: ignore[no-untyped-def]
        return Or(self, _ensure_expr(other))

    def __invert__(self):  # type: ignore[no-untyped-def]
        return Not(self)


def _ensure_expr(obj: Any) -> Expression:
    """将常量或 Series 包装为 Const 表达式。"""
    if isinstance(obj, Expression):
        return obj
    return Const(obj)


# ---------------------------------------------------------------------------
# 叶节点
# ---------------------------------------------------------------------------

class Field(Expression):
    """引用 DataFrame 的一列。"""

    def __init__(self, name: str):
        self.name = name

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return df[self.name]

    def __repr__(self) -> str:
        return f"Field({self.name!r})"


class Const(Expression):
    """常量或外部 Series。"""

    def __init__(self, value: Any):
        self.value = value

    def eval(self, df: pd.DataFrame) -> pd.Series:
        if isinstance(self.value, pd.Series):
            return self.value.reindex(df.index)
        return pd.Series(self.value, index=df.index)

    def __repr__(self) -> str:
        return f"Const({self.value!r})"


# ---------------------------------------------------------------------------
# 单目操作
# ---------------------------------------------------------------------------

class Neg(Expression):
    def __init__(self, child: Expression):
        self.child = child

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return -self.child.eval(df)

    def __repr__(self) -> str:
        return f"Neg({self.child!r})"


class Abs(Expression):
    def __init__(self, child: Expression):
        self.child = child

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.child.eval(df).abs()

    def __repr__(self) -> str:
        return f"Abs({self.child!r})"


class Not(Expression):
    def __init__(self, child: Expression):
        self.child = child

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return (~self.child.eval(df).astype(bool)).astype(int)

    def __repr__(self) -> str:
        return f"Not({self.child!r})"


class AsInt(Expression):
    """将布尔值转为 int（0/1）。"""

    def __init__(self, child: Expression):
        self.child = child

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.child.eval(df).astype(int)

    def __repr__(self) -> str:
        return f"AsInt({self.child!r})"


# ---------------------------------------------------------------------------
# 时序操作
# ---------------------------------------------------------------------------

class Shift(Expression):
    """滞后 periods 期（正数 = 向后移动，即用过去值）。"""

    def __init__(self, child: Expression, periods: int = 1):
        self.child = child
        self.periods = periods

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.child.eval(df).shift(self.periods)

    def __repr__(self) -> str:
        return f"Shift({self.child!r}, {self.periods})"


class PctChange(Expression):
    """计算 periods 期的百分比变化。"""

    def __init__(self, child: Expression, periods: int = 1):
        self.child = child
        self.periods = periods

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.child.eval(df).pct_change(self.periods)

    def __repr__(self) -> str:
        return f"PctChange({self.child!r}, {self.periods})"


# ---------------------------------------------------------------------------
# 滚动操作
# ---------------------------------------------------------------------------

class RollingMean(Expression):
    def __init__(self, child: Expression, window: int):
        self.child = child
        self.window = window

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.child.eval(df).rolling(self.window).mean()

    def __repr__(self) -> str:
        return f"RollingMean({self.child!r}, {self.window})"


class RollingStd(Expression):
    def __init__(self, child: Expression, window: int):
        self.child = child
        self.window = window

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.child.eval(df).rolling(self.window).std()

    def __repr__(self) -> str:
        return f"RollingStd({self.child!r}, {self.window})"


class RollingMax(Expression):
    def __init__(self, child: Expression, window: int):
        self.child = child
        self.window = window

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.child.eval(df).rolling(self.window).max()

    def __repr__(self) -> str:
        return f"RollingMax({self.child!r}, {self.window})"


class RollingMin(Expression):
    def __init__(self, child: Expression, window: int):
        self.child = child
        self.window = window

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.child.eval(df).rolling(self.window).min()

    def __repr__(self) -> str:
        return f"RollingMin({self.child!r}, {self.window})"


# ---------------------------------------------------------------------------
# 截面标准化
# ---------------------------------------------------------------------------

class ZScore(Expression):
    """对单列做 z-score 标准化（时间序列维度）。"""

    def __init__(self, child: Expression):
        self.child = child

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return normalize_zscore(self.child.eval(df))

    def __repr__(self) -> str:
        return f"ZScore({self.child!r})"


class CSRank(Expression):
    """截面排名（每日排名归一化到 [0, 1]）。适用于 MultiIndex(date, symbol) 数据。"""

    def __init__(self, child: Expression):
        self.child = child

    def eval(self, df: pd.DataFrame) -> pd.Series:
        s = self.child.eval(df)
        if isinstance(df.index, pd.MultiIndex) and len(df.index.names) >= 2:
            return s.groupby(level=0).rank(pct=True)
        return s.rank(pct=True)

    def __repr__(self) -> str:
        return f"CSRank({self.child!r})"


# ---------------------------------------------------------------------------
# 双目算术
# ---------------------------------------------------------------------------

class _BinaryOp(Expression):
    """双目操作基类。"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right


class Add(_BinaryOp):
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.left.eval(df) + self.right.eval(df)
    def __repr__(self) -> str: return f"({self.left!r} + {self.right!r})"


class Sub(_BinaryOp):
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.left.eval(df) - self.right.eval(df)
    def __repr__(self) -> str: return f"({self.left!r} - {self.right!r})"


class Mul(_BinaryOp):
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.left.eval(df) * self.right.eval(df)
    def __repr__(self) -> str: return f"({self.left!r} * {self.right!r})"


class Div(_BinaryOp):
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.left.eval(df) / self.right.eval(df).replace(0, np.nan)
    def __repr__(self) -> str: return f"({self.left!r} / {self.right!r})"


# ---------------------------------------------------------------------------
# 比较操作
# ---------------------------------------------------------------------------

class Gt(_BinaryOp):
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return (self.left.eval(df) > self.right.eval(df)).astype(int)
    def __repr__(self) -> str: return f"({self.left!r} > {self.right!r})"


class Ge(_BinaryOp):
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return (self.left.eval(df) >= self.right.eval(df)).astype(int)
    def __repr__(self) -> str: return f"({self.left!r} >= {self.right!r})"


class Lt(_BinaryOp):
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return (self.left.eval(df) < self.right.eval(df)).astype(int)
    def __repr__(self) -> str: return f"({self.left!r} < {self.right!r})"


class Le(_BinaryOp):
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return (self.left.eval(df) <= self.right.eval(df)).astype(int)
    def __repr__(self) -> str: return f"({self.left!r} <= {self.right!r})"


class Eq(_BinaryOp):
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return (self.left.eval(df) == self.right.eval(df)).astype(int)
    def __repr__(self) -> str: return f"({self.left!r} == {self.right!r})"


# ---------------------------------------------------------------------------
# 逻辑操作
# ---------------------------------------------------------------------------

class And(_BinaryOp):
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return (self.left.eval(df).astype(bool) & self.right.eval(df).astype(bool)).astype(int)
    def __repr__(self) -> str: return f"({self.left!r} & {self.right!r})"


class Or(_BinaryOp):
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return (self.left.eval(df).astype(bool) | self.right.eval(df).astype(bool)).astype(int)
    def __repr__(self) -> str: return f"({self.left!r} | {self.right!r})"


# ---------------------------------------------------------------------------
# 条件表达式
# ---------------------------------------------------------------------------

class Where(Expression):
    """三目条件：condition 为真取 true_expr，否则取 false_expr。"""

    def __init__(self, condition: Expression, true_expr: Expression, false_expr: Expression):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def eval(self, df: pd.DataFrame) -> pd.Series:
        cond = self.condition.eval(df).astype(bool)
        true_val = self.true_expr.eval(df)
        false_val = self.false_expr.eval(df)
        return true_val.where(cond, false_val)

    def __repr__(self) -> str:
        return f"Where({self.condition!r}, {self.true_expr!r}, {self.false_expr!r})"
