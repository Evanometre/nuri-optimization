"""Forecast demand per SKU and derive a reorder rule from it.

Design choices, each driven by what industrial-distribution demand looks like
(many slow, lumpy SKUs; a few fast ones), not by what is fashionable:

  * Level: Croston-SBA for intermittent SKUs, simple exponential smoothing for
    the rest, on seasonally adjusted weekly demand. Weeks the SKU was sold out
    are treated as censored (sales understate demand) and skipped.
  * Seasonality: one trend+annual-cycle fit per product category -- SKU-level
    seasonality is noise at this volume.
  * Safety stock: not a normal-distribution formula. Lead-time demand is
    bootstrapped from the SKU's own history, with lead times resampled from
    the supplier's *actual* receipts (the ERP's quoted lead times are optimistic).
  * Service level: not a flat 95%. Each SKU's service is chosen by trading its
    holding cost against its expected shortage cost, with one price-of-a-stockout
    multiplier tuned so the network hits the target fill. Top-revenue SKUs
    carry a floor.
  * Order size: EOQ, in whole packs.
"""

from dataclasses import dataclass

import numpy as np

REVIEW_WEEKS = 1
ADI_CUTOFF = 1.32  # Syntetos-Boylan: average demand interval above this -> intermittent


def seasonal_index(matrix, censored, categories, train_weeks):
    """Per-category multiplicative seasonal factor for weeks 0..train_weeks+52, plus a trend
    slope. OLS on [1, t, sin, cos] of uncensored category weekly totals."""
    T = matrix.shape[1]
    t_all = np.arange(T + 60)
    X_all = np.column_stack([np.ones(len(t_all)), t_all, np.sin(2 * np.pi * t_all / 52), np.cos(2 * np.pi * t_all / 52)])
    idx = np.zeros((len(categories), len(t_all)))
    for c in np.unique(categories):
        rows = categories == c
        sub, ok = matrix[rows], ~censored[rows]
        row_mean = (sub * ok).sum(1, keepdims=True) / np.maximum(ok.sum(1, keepdims=True), 1)
        y = np.where(ok, sub, row_mean).mean(axis=0)[:train_weeks]
        X = X_all[:train_weeks]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fit = np.maximum(X_all @ beta, 1e-6)
        seas = fit / (beta[0] + beta[1] * t_all)                   # remove the trend, keep the cycle
        idx[rows] = np.clip(seas, 0.5, 1.5)
    return idx


def smooth_levels(deseason, censored, alphas=(0.05, 0.1, 0.2)):
    """Per-SKU demand level after each week, with the alpha chosen by one-step error on the last
    13 weeks. Returns level [N,T] (level *after* observing week t)."""
    N, T = deseason.shape
    best_err = np.full(N, np.inf)
    best_level = np.zeros((N, T))
    intermittent = (deseason[:, :T] > 0).mean(axis=1) < (1 / ADI_CUTOFF)
    for a in alphas:
        lvl = np.zeros((N, T))
        since = np.zeros(N)
        l = deseason[:, :13].mean(axis=1)
        nz = deseason[:, :13] > 0
        size = np.where(nz.any(1), (deseason[:, :13] * nz).sum(1) / np.maximum(nz.sum(1), 1), 0.0)
        intv = np.where(nz.any(1), 13 / np.maximum(nz.sum(1), 1), 13.0)
        err = np.zeros(N)
        for t in range(T):
            x = deseason[:, t]
            pred = np.where(intermittent, (1 - a / 2) * size / intv, l)
            if t >= T - 13:
                err += np.where(censored[:, t], 0, np.abs(x - pred))
            since += 1
            upd = ~censored[:, t]
            hit = upd & (x > 0)
            size = np.where(hit, a * x + (1 - a) * size, size)
            intv = np.where(hit, a * since + (1 - a) * intv, intv)
            since = np.where(hit, 0, since)
            l = np.where(upd, a * x + (1 - a) * l, l)
            lvl[:, t] = np.where(intermittent, (1 - a / 2) * size / intv, l)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_level[better] = lvl[better]
    return best_level, intermittent


@dataclass
class ReorderModel:
    seas: np.ndarray           # [N, T+60] seasonal index
    ratio: np.ndarray          # [N] stock-to-forecast multiplier on lead-time demand
    horizon: np.ndarray        # [N] protection interval, whole weeks
    eoq_weeks: np.ndarray      # [N] order quantity, in weeks of demand
    pack: np.ndarray
    live: np.ndarray           # [N] False = never sold in the training window -> never reorder
    bias: np.ndarray = None    # [N] multiplicative level correction, calibrated by rolling-origin on training weeks only
    expected_fill: float = 0.0
    stockout_price: float = 0.0


def forecast_level(hist, hist_censored, seas, t, bias=None):
    """Deseasonalised level after observing weeks 0..t-1 of `hist`, rolled forward."""
    ds = hist / seas[:, : hist.shape[1]]
    lvl, _ = smooth_levels(ds, hist_censored)
    return lvl[:, -1] * (bias if bias is not None else 1.0)


def calibrate_bias(ds, X, cens, seas, horizon=4, origins=range(39, 74, 5)):
    """Level correction from rolling-origin forecasts inside the training window:
    sum(observed) / sum(forecast) over uncensored cells, separately for intermittent
    and smooth SKUs. Croston/SES lag a growing series and run low on lumpy ones; this
    puts the level back on target without looking at the held-out weeks."""
    lvl, inter = smooth_levels(ds, cens)
    num = np.zeros(2); den = np.zeros(2)
    for T0 in origins:
        f = lvl[:, T0 - 1][:, None] * seas[:, T0:T0 + horizon]
        ok = ~cens[:, T0:T0 + horizon]
        for g, sel in enumerate((inter, ~inter)):
            num[g] += (X[sel, T0:T0 + horizon] * ok[sel]).sum()
            den[g] += (f[sel] * ok[sel]).sum()
    adj = np.clip(num / np.maximum(den, 1e-9), 0.8, 1.6)
    return np.where(inter, adj[0], adj[1])


def future_demand(level, seas, t, h):
    """Forecast demand over weeks t..t+h-1 (h per SKU, whole weeks)."""
    N = len(level)
    hmax = int(h.max())
    cols = np.minimum(t + np.arange(hmax), seas.shape[1] - 1)
    s = seas[:, cols] * (np.arange(hmax)[None, :] < h[:, None])
    return level * s.sum(axis=1)


def fit_model(matrix, censored, project_flags, categories, cost, price, pack, supplier_idx,
              lead_pool, train_weeks, target_fill, hold_rate=0.20, order_cost=40.0,
              protect_share=0.20, protect_service=0.98, n_boot=800, seed=0):
    """Fit on weeks [0, train_weeks) only."""
    rng = np.random.default_rng(seed)
    N = matrix.shape[0]
    X = matrix[:, :train_weeks]
    cens = censored[:, :train_weeks] | project_flags[:, :train_weeks]
    seas = seasonal_index(X, cens, categories, train_weeks)
    ds = X / seas[:, :train_weeks]

    # lead-time pools from actual receipts, per supplier (weeks)
    L_mean = np.array([np.mean(lead_pool[s]) for s in supplier_idx])
    horizon = np.maximum(1, np.ceil(L_mean + REVIEW_WEEKS)).astype(int)

    # bootstrap protection-interval demand from uncensored, non-project deseasonalised weeks
    valid = ~cens
    Lmax = int(max(1, np.ceil(max(max(p) for p in lead_pool.values()) + REVIEW_WEEKS)))
    S = np.zeros((N, n_boot))
    for sup in np.unique(supplier_idx):
        rows = np.flatnonzero(supplier_idx == sup)
        pool = np.asarray(lead_pool[sup])
        L_b = np.minimum(np.ceil(rng.choice(pool, size=n_boot) + REVIEW_WEEKS).astype(int), Lmax)
        for i in rows:
            ok = np.flatnonzero(valid[i])
            if len(ok) < 8:
                ok = np.arange(train_weeks)
            draws = ds[i, rng.choice(ok, size=(n_boot, Lmax))]
            S[i] = (draws * (np.arange(Lmax)[None, :] < L_b[:, None])).sum(axis=1)
    mu = S.mean(axis=1)

    # candidate stock levels = quantiles of the bootstrap distribution
    probs = np.concatenate([np.linspace(0.30, 0.90, 13), np.linspace(0.92, 0.999, 16)])
    cand = np.quantile(S, probs, axis=1).T                                    # [N, K]
    short = np.maximum(S[:, None, :] - cand[:, :, None], 0).mean(axis=2)      # expected shortage per cycle, [N,K]

    annual_d = np.maximum(ds[:, -52:].sum(axis=1), 1e-9) if train_weeks >= 52 else ds.sum(axis=1) * 52 / train_weeks
    h_cost = hold_rate * cost
    eoq = np.sqrt(2 * order_cost * annual_d / np.maximum(h_cost, 1e-9))
    eoq_weeks = np.clip(eoq / np.maximum(annual_d / 52, 1e-9), 1, 8)
    Q = np.maximum(np.ceil(eoq_weeks * annual_d / 52 / pack) * pack, pack)
    cycles = annual_d / np.maximum(Q, 1)
    margin = np.maximum(price - cost, 0.01)

    rev = annual_d * price
    protect = rev >= np.quantile(rev, 1 - protect_share)
    floor_k = np.searchsorted(probs, protect_service)
    k_min = np.where(protect, floor_k, 0)

    hold_ss = h_cost[:, None] * (cand - mu[:, None])                          # annual holding cost of the safety margin
    exp_short_units_yr = short * cycles[:, None]
    total_units_yr = annual_d

    def choose(lam):
        obj = hold_ss + lam * margin[:, None] * exp_short_units_yr
        obj = np.where(np.arange(len(probs))[None, :] < k_min[:, None], np.inf, obj)
        k = obj.argmin(axis=1)
        fill = 1 - (exp_short_units_yr[np.arange(N), k].sum() / total_units_yr.sum())
        return k, fill

    lo, hi = 1e-7, 1e5
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        _, f = choose(mid)
        lo, hi = (mid, hi) if f < target_fill else (lo, mid)
    k, fill = choose(hi)
    stock = cand[np.arange(N), k]
    ratio = np.where(mu > 0, stock / np.maximum(mu, 1e-9), 0.0)

    live = X.sum(axis=1) > 0
    bias = calibrate_bias(ds, X, cens, seas)
    return ReorderModel(seas=seas, ratio=ratio, horizon=horizon,
                        eoq_weeks=eoq_weeks, pack=pack, live=live, expected_fill=fill, stockout_price=hi, bias=bias), Q


class ReorderPolicy:
    """(s, S) policy driven by a fitted ReorderModel; usable with nuri.replay.simulate."""

    def __init__(self, model, Q, project_flags=None):
        self.m, self.Q = model, Q
        self.project = project_flags

    def __call__(self, t, onhand, position, hist, soldout):
        T0 = hist.shape[1]
        cens = soldout[:, :T0].copy()
        if self.project is not None:
            n = min(T0, self.project.shape[1]); cens[:, :n] |= self.project[:, :n]
        lvl = forecast_level(hist, cens, self.m.seas, T0, self.m.bias)
        need = self.m.ratio * future_demand(lvl, self.m.seas, T0, self.m.horizon)
        s = np.where(self.m.live, need, 0.0)
        q = np.ceil(np.maximum(s + self.Q - position, 0) / self.m.pack) * self.m.pack
        return np.where(position <= s, q, 0.0)
