"""票息和骑乘收益：买入某个期限的平价债，持有一段时间，假设曲线形状不变，赚多少。

拆成两部分（都按每 100 元面值，换算成基点）：
- 票息减资金成本（carry）：收益率不变时持有期的收入，减去回购融资的成本。
- 骑乘（roll-down）：剩余期限变短，沿着曲线「滑」到更低的收益率，价格上涨。
盈亏平衡：持有期内收益率最多能上行多少 bp，才把这两部分吃掉。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .bond import Bond


@dataclass(frozen=True)
class CarryRoll:
    tenor: float
    ytm: float
    carry_bp: float
    roll_bp: float
    total_bp: float
    duration: float
    breakeven_bp: float


def carry_roll(curve: Callable[[np.ndarray], np.ndarray], tenor: float, horizon: float = 0.25, funding: float = 0.0) -> CarryRoll:
    """curve 是平价收益率曲线（期限 → 收益率），funding 是年化资金成本，horizon 是持有年数。"""
    if not 0 < horizon < tenor:
        raise ValueError("持有期要大于 0 且短于期限")
    y0 = float(curve(tenor))
    y1 = float(curve(tenor - horizon))
    p0 = Bond(y0, tenor).dirty_price(y0)
    later = Bond(y0, tenor - horizon)
    static = later.dirty_price(y0)
    rolled = later.dirty_price(y1)
    carry = static - p0 - funding * horizon * p0
    roll = rolled - static
    dur = later.modified_duration(y1)
    total = carry + roll
    to_bp = 100 / p0 * 100
    return CarryRoll(tenor, y0, carry * to_bp, roll * to_bp, total * to_bp, dur, total / (dur * rolled) * 1e4)
