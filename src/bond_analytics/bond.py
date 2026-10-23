"""固定利率债券：现金流、全价和净价、到期收益率、久期、凸性、DV01。

约定：
- 收益率、票面利率都用小数（1.85% 写 0.0185），按每年付息 freq 次复利。
- maturity 是结算日到到期日的年数，可以不是整数；付息日从到期日往前每隔 1/freq 年倒推。
- 价格都按每 100 元面值。全价 = 净价 + 应计利息。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

BP = 1e-4


@dataclass(frozen=True)
class Bond:
    coupon: float
    maturity: float
    freq: int = 1
    face: float = 100.0

    def __post_init__(self) -> None:
        if self.maturity <= 0:
            raise ValueError("剩余期限要大于 0")
        if self.freq not in (1, 2, 4, 12):
            raise ValueError("每年付息次数只支持 1、2、4、12")

    def cashflows(self) -> tuple[np.ndarray, np.ndarray]:
        """返回（每笔现金流距今的年数，金额），按时间先后排。"""
        step = 1 / self.freq
        n = int(np.ceil(self.maturity * self.freq - 1e-9))
        times = self.maturity - step * np.arange(n)[::-1]
        amounts = np.full(n, self.face * self.coupon / self.freq)
        amounts[-1] += self.face
        return times, amounts

    def accrued(self) -> float:
        """应计利息：本付息期已经过去的部分。"""
        first = self.cashflows()[0][0]
        elapsed = 1 - first * self.freq
        return self.face * self.coupon / self.freq * max(elapsed, 0.0)

    def _discount(self, ytm: float, times: np.ndarray) -> np.ndarray:
        return (1 + ytm / self.freq) ** (-self.freq * times)

    def dirty_price(self, ytm: float) -> float:
        t, cf = self.cashflows()
        return float(cf @ self._discount(ytm, t))

    def clean_price(self, ytm: float) -> float:
        return self.dirty_price(ytm) - self.accrued()

    def ytm(self, clean_price: float) -> float:
        """由净价反求到期收益率。"""
        target = clean_price + self.accrued()
        return brentq(lambda y: self.dirty_price(y) - target, -0.5, 1.0, xtol=1e-14)

    def macaulay_duration(self, ytm: float) -> float:
        """麦考利久期（年）：现金流时间按现值加权的平均。"""
        t, cf = self.cashflows()
        pv = cf * self._discount(ytm, t)
        return float(t @ pv / pv.sum())

    def modified_duration(self, ytm: float) -> float:
        """修正久期：收益率变动 1（即 100%）时价格变动的百分比，取正号。"""
        return self.macaulay_duration(ytm) / (1 + ytm / self.freq)

    def convexity(self, ytm: float) -> float:
        """凸性：价格对收益率的二阶导除以价格。"""
        t, cf = self.cashflows()
        pv = cf * self._discount(ytm, t)
        return float(pv @ (t * (t + 1 / self.freq)) / (1 + ytm / self.freq) ** 2 / pv.sum())

    def dv01(self, ytm: float) -> float:
        """收益率下行 1bp，每 100 元面值的全价上涨多少元。"""
        return self.modified_duration(ytm) * self.dirty_price(ytm) * BP

    def price_change_approx(self, ytm: float, dy: float) -> float:
        """久期加凸性的二阶近似：收益率变动 dy 时全价变动多少元。"""
        p = self.dirty_price(ytm)
        return p * (-self.modified_duration(ytm) * dy + 0.5 * self.convexity(ytm) * dy**2)
