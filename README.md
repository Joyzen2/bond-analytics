# bond-analytics

债券计算的小工具包：固定利率债的定价、到期收益率、久期、凸性、DV01，收益率曲线的插值、Nelson-Siegel 拟合和即期利率剥离，关键期限久期，以及票息加骑乘收益。每个公式都有单元测试对着数值导数或已知结果核对。

A small Python toolkit for bond math: pricing, yield, duration, convexity, DV01, curve bootstrapping, key rate durations, and carry & roll-down.

## 能算什么

| 模块 | 内容 |
| --- | --- |
| `Bond` | 现金流、全价和净价、应计利息、到期收益率、麦考利久期、修正久期、凸性、DV01、久期加凸性的价格近似 |
| `par_curve` / `ns_fit` | 平价收益率曲线插值（保形三次）、Nelson-Siegel 拟合 |
| `bootstrap_par` | 从平价收益率逐年剥离即期利率 |
| `key_rate_durations` | 关键期限久期：只动某一个期限附近的利率，价格变多少 |
| `carry_roll` | 持有一段时间、曲线不变时的票息减资金成本、骑乘收益和盈亏平衡 |

## 用法

```python
from bond_analytics import Bond, bootstrap_par, carry_roll, key_rate_durations, par_curve

b = Bond(coupon=0.0185, maturity=9.6)          # 票面 1.85%，剩余 9.6 年，每年付息
b.clean_price(0.0165), b.modified_duration(0.0165), b.convexity(0.0165), b.dv01(0.0165)

tenors = [0.25, 0.5, 1, 3, 5, 7, 10, 30]
yields = [0.0116, 0.0116, 0.0118, 0.0127, 0.0135, 0.0146, 0.0164, 0.0206]
key_rate_durations(b, bootstrap_par(tenors, yields))
carry_roll(par_curve(tenors, yields), tenor=10, horizon=0.25, funding=0.014)
```

## 例子：国债曲线的票息加骑乘

`examples/cgb_carry_roll.py` 取中债国债收益率曲线的最新一天，算各期限国债持有 3 个月的票息减资金成本、骑乘收益和盈亏平衡，资金成本默认用 7 天逆回购利率近似。30 年不算：曲线在 10 年和 30 年之间没有报价点，那一段的骑乘对插值方法太敏感。

```bash
uv sync --extra examples
uv run python examples/cgb_carry_roll.py --funding 1.40
```

## 测试

```bash
uv run pytest -q
```

## 公式

收益率 $y$ 按每年付息 $f$ 次复利，第 $k$ 笔现金流 $CF_k$ 距今 $t_k$ 年：

- 全价 $P = \sum_k CF_k (1 + y/f)^{-f t_k}$，净价 = 全价 − 应计利息
- 麦考利久期 $D = \sum_k t_k \cdot PV_k / P$，修正久期 $D^* = D / (1 + y/f)$
- 凸性 $C = \frac{1}{P} \sum_k PV_k \, t_k (t_k + 1/f) / (1 + y/f)^2$
- 价格变动 $\Delta P \approx P\,(-D^* \Delta y + \tfrac12 C \Delta y^2)$，DV01 $= D^* P \times 0.0001$
- 票息加骑乘：买入 $T$ 年平价债，持有 $h$ 年，曲线不变，剩余期限 $T-h$ 时按曲线上 $T-h$ 年的收益率重新定价。收益率不变部分减去资金成本是票息收益，收益率下滑带来的价格上涨是骑乘收益；两部分之和除以 $D^* P$ 是盈亏平衡的收益率上行幅度。
