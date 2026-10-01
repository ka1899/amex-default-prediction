import numpy as np
import pandas as pd
import pytest

from amex_ppi.data import make_synthetic
from amex_ppi.features import customer_features, truncate_history
from amex_ppi.intervention import InterventionAssumptions, economics, sensitivity
from amex_ppi.metrics import amex_metric, evaluate
from amex_ppi.pricing import PricingAssumptions, experience_loss_ratio, gross_premium, pricing_table


@pytest.fixture(scope="module")
def statements():
    return make_synthetic(n_customers=300, seed=1)


# --- features -------------------------------------------------------------

@pytest.mark.parametrize("feature_set", ["snapshot", "aggregate", "trajectory"])
def test_features_are_finite_and_one_row_per_customer(statements, feature_set):
    X, y = customer_features(statements, feature_set)
    assert len(X) == len(y) == statements["customer_ID"].nunique()
    assert not np.isinf(X.to_numpy()).any()


def test_feature_sets_are_nested(statements):
    cols = {fs: set(customer_features(statements, fs)[0].columns)
            for fs in ["snapshot", "aggregate", "trajectory"]}
    assert cols["snapshot"] < cols["aggregate"] < cols["trajectory"]


def test_zero_balances_do_not_produce_inf():
    df = pd.DataFrame({
        "customer_ID": ["a"] * 3, "S_2": pd.date_range("2017-01-01", periods=3, freq="MS"),
        "B_1": [0.0, 5.0, 0.0], "target": [1] * 3,
    })
    X, _ = customer_features(df, "trajectory")
    assert np.isfinite(X["B_1_pct_change_mean"]).all()


def test_slope_recovers_linear_trend():
    df = pd.DataFrame({
        "customer_ID": ["a"] * 5 + ["b"] * 5,
        "S_2": list(pd.date_range("2017-01-01", periods=5, freq="MS")) * 2,
        "P_2": [1, 3, 5, 7, 9] + [4, np.nan, 4, 4, 4], "target": [0] * 10,
    })
    X, _ = customer_features(df, "trajectory")
    assert X.loc["a", "P_2_slope"] == pytest.approx(2.0)
    assert X.loc["b", "P_2_slope"] == pytest.approx(0.0)


def test_truncate_history_drops_last_statements(statements):
    out = truncate_history(statements, 2)
    before = statements.groupby("customer_ID").size()
    after = out.groupby("customer_ID").size()
    assert (after == before.loc[after.index] - 2).all()
    assert set(before[before <= 2].index).isdisjoint(after.index)


# --- metrics --------------------------------------------------------------

def test_amex_metric_bounds():
    rng = np.random.default_rng(0)
    # Few enough positives that all of them fit in the top 4% (by weight), so a
    # perfect ranking scores exactly 1.
    y = np.array([1] * 20 + [0] * 1000)
    assert amex_metric(y, y.astype(float)) == pytest.approx(1.0)
    y = rng.integers(0, 2, 2000)
    assert abs(amex_metric(y, rng.random(2000))) < 0.15


def test_evaluate_keys():
    m = evaluate([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.7])
    assert m["auc"] == 1.0 and m["gini"] == 1.0


# --- pricing --------------------------------------------------------------

def test_gross_premium_loads_on_premium_not_on_cost():
    a = PricingAssumptions(claim_amount=1000, expense_ratio=0.25, profit_margin=0.15)
    assert gross_premium(0.1, a) == pytest.approx(100 / 0.60)
    assert a.target_loss_ratio == pytest.approx(0.60)


def test_experience_loss_ratio_hits_target_when_calibrated():
    a = PricingAssumptions()
    y = np.array([1] * 20 + [0] * 80)
    p = np.full(100, 0.2)  # perfectly calibrated
    assert experience_loss_ratio(gross_premium(p, a), y, a) == pytest.approx(a.target_loss_ratio)


def test_pricing_table_flat_rate_overcharges_low_risk():
    rng = np.random.default_rng(0)
    p = rng.beta(1, 3, 5000)
    y = (rng.random(5000) < p).astype(int)
    t = pricing_table(p, y, flat_rate_p=p.mean())
    assert t["loss_ratio_flat"].iloc[0] < t["loss_ratio_flat"].iloc[-1]
    assert t["customers"].sum() == 5000


# --- intervention ---------------------------------------------------------

def test_intervention_economics_by_hand():
    a = InterventionAssumptions(cost=100, success_rate=0.5, claim_amount=1000)
    e = economics([0.9, 0.8, 0.7, 0.1], [1, 0, 1, 1], threshold=0.5, a=a)
    assert e["flagged"] == 3 and e["true_positives"] == 2
    assert e["program_cost"] == 300 and e["claims_avoided"] == 1000
    assert e["net_savings"] == 700 and e["roi"] == pytest.approx(700 / 300)
    assert e["breakeven_success_rate"] == pytest.approx(300 / 2000)


def test_sensitivity_shape():
    s = sensitivity([0.9, 0.1], [1, 0], 0.5, success_rates=(0.1, 0.2), costs=(100, 200, 300))
    assert s.shape == (2, 3)


def test_int8_targets_with_integer_claim_amount_do_not_overflow():
    # Targets are stored as int8; numpy 2 would overflow int8 * 5000.
    a = PricingAssumptions(claim_amount=5_000)
    y = pd.Series([1, 0, 1, 0], dtype="int8")
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert experience_loss_ratio(gross_premium(p, a), y, a) == pytest.approx(a.target_loss_ratio)
    assert pricing_table(p, y, 0.5, a, n_bands=2)["loss_ratio_flat"].notna().all()
    e = economics(p, y, 0.4, InterventionAssumptions(claim_amount=5_000))
    assert e["claims_avoided"] == pytest.approx(2 * 0.3 * 5_000)
