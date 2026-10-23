"""中债国债收益率曲线：各期限持有 3 个月的票息减资金成本、骑乘收益和盈亏平衡。

运行：uv run python examples/cgb_carry_roll.py --funding 1.40 [--date 2026-10-22] [--csv 输出.csv]
- 曲线取中债国债收益率曲线（akshare 的 bond_china_yield），默认取最近一个交易日。
- 资金成本用年化百分比输入，默认 1.40（7 天逆回购利率）。
- 都按每年付息一次的平价债近似。
- 不算 30 年：这条曲线在 10 年和 30 年之间没有报价点，30 年的骑乘对插值方法非常敏感，算不准。
"""
from __future__ import annotations

import argparse

import akshare as ak
import pandas as pd

from bond_analytics import carry_roll, par_curve

COLS = {"3月": 0.25, "6月": 0.5, "1年": 1, "3年": 3, "5年": 5, "7年": 7, "10年": 10, "30年": 30}
TENORS = (1, 2, 3, 5, 7, 10)


def curve_on(date: str | None) -> tuple[pd.Timestamp, pd.Series]:
    end = pd.Timestamp(date) if date else pd.Timestamp.today()
    raw = ak.bond_china_yield(start_date=(end - pd.Timedelta(days=14)).strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    cgb = raw[raw["曲线名称"] == "中债国债收益率曲线"].sort_values("日期")
    row = cgb.iloc[-1]
    return pd.Timestamp(row["日期"]), pd.Series({t: float(row[c]) / 100 for c, t in COLS.items()})


def table(curve: pd.Series, funding: float, horizon: float = 0.25) -> pd.DataFrame:
    f = par_curve(curve.index.to_numpy(float), curve.to_numpy())
    rows = []
    for t in TENORS:
        r = carry_roll(f, t, horizon=horizon, funding=funding)
        rows.append({"期限（年）": t, "到期收益率（%）": r.ytm * 100, "票息减资金成本（bp）": r.carry_bp, "骑乘（bp）": r.roll_bp,
                     "合计（bp）": r.total_bp, "修正久期": r.duration, "盈亏平衡（bp）": r.breakeven_bp})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funding", type=float, default=1.40, help="年化资金成本，百分比")
    ap.add_argument("--date", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    day, curve = curve_on(a.date)
    out = table(curve, a.funding / 100)
    print(f"中债国债收益率曲线 {day:%Y-%m-%d}，持有 3 个月，资金成本 {a.funding:.2f}%")
    print(out.round(2).to_markdown(index=False))
    if a.csv:
        out.to_csv(a.csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
