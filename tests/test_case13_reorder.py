import numpy as np
import pandas as pd
import pytest

from cases.case13_distributor import generate, START_DATE, WEEKS, TRAIN_WEEKS, BUYER_ROP_WEEKS, BUYER_UPTO_WEEKS
from nuri.data_cleaning import clean_sales, weekly_matrix, normalise_code, flag_project_orders, clean_inventory
from nuri.replay import simulate, BuyerRule, metrics
from nuri.reorder_policy import fit_model, ReorderPolicy
from run_case13_frontier import prepare


@pytest.fixture(scope="module")
def ds():
    return generate(13)


def test_generator_is_deterministic():
    a, b = generate(5), generate(5)
    assert np.array_equal(a.truth["demand"], b.truth["demand"])
    assert a.sales_lines.equals(b.sales_lines)


def test_raw_export_is_actually_dirty(ds):
    lines = ds.sales_lines
    assert (lines.sku_raw != lines.sku_raw.map(normalise_code)).any()
    assert lines.duplicated("line_id").any() and (lines.qty < 0).any() and (lines.uom != "EA").any()
    assert (ds.inventory_now.on_hand < 0).any() and ds.sku_master.quoted_lead_weeks.isna().any()


def test_cleaning_recovers_true_weekly_sales_exactly(ds):
    df, rep = clean_sales(ds.sales_lines, ds.sku_master, ds.uom_table, START_DATE)
    assert np.array_equal(weekly_matrix(df, ds.truth["codes"], WEEKS), ds.truth["sales"])
    assert rep["duplicate_lines_removed"] > 0 and rep["uom_inferred_from_price"] > 0


@pytest.mark.parametrize("raw", ["fst00012", "FST 00012 ", "FST-00012", " fst-00012"])
def test_sku_code_variants_normalise_to_one_key(raw):
    assert normalise_code(raw) == "FST-00012"


def test_inventory_cleaning_clips_negatives_and_drops_orphans(ds):
    inv, rep = clean_inventory(ds.inventory_now, ds.sku_master)
    assert (inv.on_hand >= 0).all() and rep["orphan_inventory_rows"] == 25 and len(inv) == 2500


def test_replay_conserves_stock():
    rng = np.random.default_rng(1)
    N, T = 8, 40
    D = rng.poisson(6, (N, T)).astype(float)
    r = simulate(D, np.full(N, 50.0), np.zeros((N, 0)), np.full((N, T), 2), BuyerRule(3, 6, np.ones(N) * 4))
    received = r["arrivals"][:, :T].sum(axis=1)
    assert np.allclose(50 + received - r["sales"].sum(axis=1), r["end_onhand"][:, -1])
    assert (r["end_onhand"] >= 0).all() and (r["sales"] <= D + 1e-9).all()


def test_baseline_replay_reproduces_the_company_s_actual_history(ds):
    tr = ds.truth
    _, sales, soldout, projects, sup_idx, pools, pipe = prepare(13)
    r = simulate(tr["demand"][:, TRAIN_WEEKS:], tr["end_onhand"][:, TRAIN_WEEKS - 1], pipe,
                 tr["lead_weeks"][:, TRAIN_WEEKS:], BuyerRule(BUYER_ROP_WEEKS, BUYER_UPTO_WEEKS, tr["pack"]),
                 sales[:, :TRAIN_WEEKS], soldout[:, :TRAIN_WEEKS])
    assert np.allclose(r["sales"], tr["sales"][:, TRAIN_WEEKS:])


def test_fit_never_sees_the_held_out_weeks(ds):
    _, sales, soldout, projects, sup_idx, pools, pipe = prepare(13)
    tr = ds.truth
    args = (ds.sku_master.category.values, tr["cost"], tr["price"], tr["pack"], sup_idx, pools, TRAIN_WEEKS, 0.95)
    m1, _ = fit_model(sales, soldout, projects, *args)
    tampered = sales.copy()
    tampered[:, TRAIN_WEEKS:] = 10_000
    m2, _ = fit_model(tampered, soldout, projects, *args)
    assert np.array_equal(m1.ratio, m2.ratio) and np.array_equal(m1.bias, m2.bias)


def test_new_rules_beat_the_current_rule_on_revenue_fill_and_inventory(ds):
    _, sales, soldout, projects, sup_idx, pools, pipe = prepare(13)
    tr = ds.truth
    model, Q = fit_model(sales, soldout, projects, ds.sku_master.category.values, tr["cost"], tr["price"],
                         tr["pack"], sup_idx, pools, TRAIN_WEEKS, 0.97)
    D, L = tr["demand"][:, TRAIN_WEEKS:], tr["lead_weeks"][:, TRAIN_WEEKS:]
    on0, prior, pso = tr["end_onhand"][:, TRAIN_WEEKS - 1], sales[:, :TRAIN_WEEKS], soldout[:, :TRAIN_WEEKS]
    seen = prior.sum(axis=1) > 0
    new = simulate(D, on0, pipe, L, ReorderPolicy(model, Q, projects), prior, pso)
    cur = simulate(D, on0, pipe, L, BuyerRule(BUYER_ROP_WEEKS, BUYER_UPTO_WEEKS, tr["pack"]), prior, pso)
    mn, mc = metrics(new, D, tr["cost"], tr["price"], seen), metrics(cur, D, tr["cost"], tr["price"], seen)
    assert mn["revenue_fill"] >= mc["revenue_fill"] and mn["avg_inventory"] < mc["avg_inventory"]
    # orders are always whole packs
    ordered = new["orders"][new["orders"] > 0]
    packs = np.broadcast_to(tr["pack"][:, None], new["orders"].shape)[new["orders"] > 0]
    assert np.allclose(ordered % packs, 0)
