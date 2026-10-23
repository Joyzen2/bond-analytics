"""核对公式：平价债价格是 100、零息债久期等于期限、久期凸性和数值导数一致、剥离和关键期限久期、票息和骑乘。"""
import numpy as np
import pytest

from bond_analytics import Bond, bootstrap_par, carry_roll, effective_duration, key_rate_durations, ns_fit, ns_yield, par_curve, price_on_curve


@pytest.mark.parametrize("freq", [1, 2])
def test_par_bond_prices_at_100_on_coupon_date(freq):
    b = Bond(coupon=0.025, maturity=10, freq=freq)
    assert np.isclose(b.dirty_price(0.025), 100) and b.accrued() == 0


def test_zero_coupon_duration_equals_maturity():
    b = Bond(coupon=0.0, maturity=7)
    assert np.isclose(b.macaulay_duration(0.03), 7)
    assert np.isclose(b.convexity(0.03), 7 * 8 / 1.03**2)


def test_ytm_round_trip_between_coupon_dates():
    b = Bond(coupon=0.0185, maturity=9.6)
    assert np.isclose(b.ytm(b.clean_price(0.0172)), 0.0172, atol=1e-12)


def test_accrued_interest_half_way_through_the_period():
    assert np.isclose(Bond(coupon=0.03, maturity=4.5).accrued(), 1.5)


def test_duration_and_convexity_match_numerical_derivatives():
    b, y, h = Bond(coupon=0.02, maturity=10, freq=2), 0.021, 1e-5
    p, up, down = b.dirty_price(y), b.dirty_price(y + h), b.dirty_price(y - h)
    assert np.isclose(b.modified_duration(y), (down - up) / (2 * h * p), rtol=1e-6)
    assert np.isclose(b.convexity(y), (up + down - 2 * p) / (h**2 * p), rtol=1e-4)
    assert np.isclose(b.dv01(y), (down - up) / 2 / (h / 1e-4), rtol=1e-6)


def test_duration_convexity_approximation_beats_duration_alone():
    b, y, dy = Bond(coupon=0.02, maturity=30), 0.02, 0.01
    true = b.dirty_price(y + dy) - b.dirty_price(y)
    linear = -b.modified_duration(y) * b.dirty_price(y) * dy
    assert abs(b.price_change_approx(y, dy) - true) < 0.15 * abs(linear - true)    # 30 年债动 100bp，剩下的是三阶项


def test_bootstrap_flat_curve_gives_flat_zeros_and_reprices_par_bonds():
    tenors, flat = [0.25, 0.5, 1, 3, 5, 7, 10, 30], [0.02] * 8
    z = bootstrap_par(tenors, flat)
    assert np.allclose(z.zeros, 0.02)
    up = [0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0148, 0.0165, 0.0207]
    z2 = bootstrap_par(tenors, up)
    for n in (1, 5, 10, 30):
        assert np.isclose(price_on_curve(Bond(float(par_curve(tenors, up)(n)), n), z2), 100, atol=1e-9)


def test_key_rate_durations_add_up_to_effective_duration():
    z = bootstrap_par([0.25, 0.5, 1, 3, 5, 7, 10, 30], [0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0148, 0.0165, 0.0207])
    b = Bond(coupon=0.0165, maturity=10)
    krd = key_rate_durations(b, z)
    assert np.isclose(sum(krd.values()), effective_duration(b, z), rtol=1e-6)
    zero5 = key_rate_durations(Bond(coupon=0.0, maturity=5), z)
    assert max(zero5, key=zero5.get) == 5.0 and np.isclose(zero5[5.0], sum(zero5.values()), rtol=1e-6)    # 5 年零息债只对 5 年敏感


def test_carry_and_roll_on_flat_and_upward_curves():
    flat = par_curve([1, 30], [0.02, 0.02])
    r = carry_roll(flat, 10, horizon=0.25, funding=0.02)
    assert abs(r.carry_bp) < 0.5 and abs(r.roll_bp) < 1e-6          # 曲线平、资金成本等于收益率：两部分都约等于 0
    up = par_curve([1, 3, 5, 7, 10, 30], [0.013, 0.0135, 0.014, 0.0148, 0.0165, 0.0207])
    r = carry_roll(up, 10, horizon=0.25, funding=0.014)
    y1 = float(up(9.75))
    assert r.roll_bp > 0 and np.isclose(r.roll_bp, r.duration * (0.0165 - y1) * 1e4, rtol=0.02)   # 骑乘约等于久期乘收益率下滑


def test_breakeven_rise_wipes_out_the_return():
    up = par_curve([1, 3, 5, 7, 10, 30], [0.013, 0.0135, 0.014, 0.0148, 0.0165, 0.0207])
    r = carry_roll(up, 10, horizon=0.25, funding=0.014)
    later = Bond(0.0165, 9.75)
    y_end = float(up(9.75)) + r.breakeven_bp * 1e-4
    pnl = later.dirty_price(y_end) - 100 - 0.014 * 0.25 * 100
    assert abs(pnl) < 0.01                                             # 收益率上行到盈亏平衡点，持有期收益约为 0


def test_nelson_siegel_recovers_known_parameters():
    tau = np.array([0.25, 0.5, 1, 3, 5, 7, 10, 30])
    beta = np.array([0.025, -0.012, 0.004])
    assert np.allclose(ns_fit(tau, ns_yield(tau, beta)), beta)
