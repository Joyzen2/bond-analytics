"""收益率曲线：插值、Nelson-Siegel 拟合、从平价收益率剥离即期利率。

约定：
- 期限按年，利率用小数。
- 中债国债收益率曲线是到期收益率曲线，这里按平价理解：票面利率等于收益率、价格 100。
- 即期利率按年复利；不到 1 年的期限当作零息，即期利率就是收益率。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator

LAMBDA = 0.0609 * 12  # Diebold-Li（2006）的衰减参数 0.0609 是按月算的，换成按年


def par_curve(tenors, yields) -> Callable[[np.ndarray], np.ndarray]:
    """平价收益率曲线：在报价期限上精确，中间用保形三次插值（不会冲过两端），两头外推取端点值。"""
    t, y = np.asarray(tenors, float), np.asarray(yields, float)
    f = PchipInterpolator(t, y)
    return lambda x: f(np.clip(np.asarray(x, float), t[0], t[-1]))


def ns_loadings(tau, lam: float = LAMBDA) -> np.ndarray:
    """Nelson-Siegel 的三个载荷：水平、斜率、曲率。"""
    x = lam * np.asarray(tau, float)
    l1 = (1 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(x), l1, l1 - np.exp(-x)])


def ns_fit(tenors, yields, lam: float = LAMBDA) -> np.ndarray:
    """固定衰减参数，用最小二乘估出 β0、β1、β2。β1 < 0 表示长端高于短端。"""
    beta, *_ = np.linalg.lstsq(ns_loadings(tenors, lam), np.asarray(yields, float), rcond=None)
    return beta


def ns_yield(tau, beta, lam: float = LAMBDA) -> np.ndarray:
    return ns_loadings(tau, lam) @ np.asarray(beta, float)


@dataclass(frozen=True, eq=False)
class ZeroCurve:
    tenors: np.ndarray
    zeros: np.ndarray

    def rate(self, t) -> np.ndarray:
        """即期利率：节点之间线性插值，两头取端点值。"""
        return np.interp(np.asarray(t, float), self.tenors, self.zeros)

    def df(self, t) -> np.ndarray:
        """贴现因子。"""
        t = np.asarray(t, float)
        return (1 + self.rate(t)) ** (-t)

    def bumped(self, bump: Callable[[np.ndarray], np.ndarray]) -> "ZeroCurve":
        """每个节点的即期利率加上 bump(期限)，用来算关键期限久期。"""
        return ZeroCurve(self.tenors, self.zeros + bump(self.tenors))


def bootstrap_par(tenors, yields) -> ZeroCurve:
    """从平价收益率逐年剥离即期利率（每年付息一次）。

    不到 1 年的期限直接当零息；1 年及以上先把平价收益率插值到整年，再从 1 年起逐年解
    100 = c × 100 × (DF1 + … + DFn) + 100 × DFn。
    """
    t, y = np.asarray(tenors, float), np.asarray(yields, float)
    short = t < 1
    years = np.arange(1, int(np.floor(t.max())) + 1, dtype=float)
    par = par_curve(t, y)(years)
    dfs = []
    for c in par:
        dfs.append((1 - c * sum(dfs)) / (1 + c))
    zeros = np.asarray(dfs) ** (-1 / years) - 1
    nodes = np.concatenate([t[short], years])
    rates = np.concatenate([y[short], zeros])
    return ZeroCurve(nodes, rates)
