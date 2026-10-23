"""在即期利率曲线上给债券定价，算有效久期和关键期限久期。"""
from __future__ import annotations

import numpy as np

from .bond import BP, Bond
from .curve import ZeroCurve

KEY_TENORS = (0.25, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)


def price_on_curve(bond: Bond, curve: ZeroCurve) -> float:
    """用即期利率逐笔贴现，得到全价。"""
    t, cf = bond.cashflows()
    return float(cf @ curve.df(t))


def effective_duration(bond: Bond, curve: ZeroCurve, bp: float = BP) -> float:
    """整条即期曲线平行上下移动 1bp，价格变动的百分比。"""
    up = price_on_curve(bond, curve.bumped(lambda x: np.full_like(x, bp)))
    down = price_on_curve(bond, curve.bumped(lambda x: np.full_like(x, -bp)))
    return (down - up) / (2 * price_on_curve(bond, curve) * bp)


def _triangle(keys: tuple[float, ...], i: int):
    """第 i 个关键期限的三角形扰动：在该期限为 1，向相邻关键期限线性降到 0；最短、最长的往外保持 1。"""
    k = np.asarray(keys)

    def bump(x):
        x = np.asarray(x, float)
        w = np.zeros_like(x)
        if i > 0:
            left = (x >= k[i - 1]) & (x <= k[i])
            w[left] = (x[left] - k[i - 1]) / (k[i] - k[i - 1])
        else:
            w[x <= k[0]] = 1.0
        if i < len(k) - 1:
            right = (x >= k[i]) & (x <= k[i + 1])
            w[right] = (k[i + 1] - x[right]) / (k[i + 1] - k[i])
        else:
            w[x >= k[-1]] = 1.0
        return w

    return bump


def key_rate_durations(bond: Bond, curve: ZeroCurve, keys: tuple[float, ...] = KEY_TENORS, bp: float = BP) -> dict[float, float]:
    """关键期限久期：只动某一个期限附近的利率，价格变动的百分比。各期限的扰动加起来正好是平行移动，所以求和约等于有效久期。"""
    p0 = price_on_curve(bond, curve)
    out = {}
    for i, key in enumerate(keys):
        b = _triangle(keys, i)
        up = price_on_curve(bond, curve.bumped(lambda x: bp * b(x)))
        down = price_on_curve(bond, curve.bumped(lambda x: -bp * b(x)))
        out[key] = (down - up) / (2 * p0 * bp)
    return out
