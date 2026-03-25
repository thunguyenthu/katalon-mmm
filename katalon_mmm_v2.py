"""
Katalon MMM v2 — Bayesian hill-adstock via NumPyro (NUTS)
==========================================================
Custom Bayesian MMM using NumPyro directly.  Avoids the lightweight_mmm
library which has a JAX/XLA mutex-deadlock issue on macOS Apple Silicon.

Model structure (per channel c, time t):
  adstock[t,c] = spend[t,c] + lag_weight[c] * adstock[t-1,c]   (learnable decay)
  hill[t,c]    = adstock[t,c]^slope[c] / (adstock^slope + K^slope)  (saturation)
  mu[t]        = intercept + Σ beta[c]*hill[t,c] + seasonality + extra_features
  target[t]    ~ Normal(mu[t], sigma)

Compared to v1 OLS:
  - Adstock decay and Hill saturation are learned from data (not fixed)
  - Full posterior distributions → credible intervals on every estimate
  - Diminishing returns modelled explicitly
  - Naturally handles small samples with informative priors

Usage:
  python3 katalon_mmm_v2.py --json data_v2.json \\
      --lag-pipeline 1 --lag-arr 5 --winsorize-arr 90 \\
      --export-html katalon_mmm_v2_report.html

  # Fast test run (~25 sec total)
  python3 katalon_mmm_v2.py --json data_v2.json --quick \\
      --export-html katalon_mmm_v2_report.html
"""

import argparse
import base64
import io
import os
import sys
import warnings
from pathlib import Path

# Suppress noise before any JAX imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# NumPyro / JAX
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Reuse v1 utilities
sys.path.insert(0, str(Path(__file__).parent))
from katalon_mmm import (
    load_json_data,
    CHANNEL_META,
    OUTCOME_META,
    QUALITY_CONTROLS,
    BREAK_FLAGS,
    ORGANIC_BASELINE,
    PAID_SHARE,
    fmt_num,
    run_mmm_ols,
    add_seasonality_flags,
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

_DIV = "─" * 72

_WARMUP_DEFAULT  = 500
_SAMPLES_DEFAULT = 1000
_CHAINS_DEFAULT  = 1       # 1 chain to stay fast; increase with --chains
_SEED_DEFAULT    = 42

_WARMUP_QUICK  = 200
_SAMPLES_QUICK = 500


# ─── BAYESIAN MODEL ───────────────────────────────────────────────────────────

def _bayesian_mmm(media, extra, n_channels, n_extra, target=None):
    """
    Custom NumPyro hill-adstock MMM.
    Uses dist.expand() instead of numpyro.plate to avoid the JAX/XLA
    mutex deadlock seen on macOS Apple Silicon with lightweight_mmm.

    Parameters
    ----------
    media      : jnp.ndarray  shape (T, n_channels) — scaled spend
    extra      : jnp.ndarray or None  shape (T, n_extra) — quality controls + breaks
    n_channels : int
    n_extra    : int
    target     : jnp.ndarray or None  shape (T,) — scaled outcome (None = prior predictive)
    """
    T = media.shape[0]

    # ── Channel-level priors (vector-valued, no numpyro.plate) ────────────────
    lag_weight = numpyro.sample(
        "lag_weight",
        dist.Beta(concentration1=2.0, concentration0=2.0).expand([n_channels])
    )  # adstock decay, 0 = no carry-over, 1 = full carry-over

    slope = numpyro.sample(
        "slope",
        dist.TruncatedNormal(low=0.5, loc=1.0, scale=0.5).expand([n_channels])
    )  # Hill exponent: >1 = S-curve, <1 = concave, =1 = linear saturation

    half_max = numpyro.sample(
        "half_max",
        dist.HalfNormal(scale=1.0).expand([n_channels])
    )  # Hill K: scaled spend at which response is 50%

    beta_media = numpyro.sample(
        "beta_media",
        dist.HalfNormal(scale=1.0).expand([n_channels])
    )  # channel effect coefficient (positive only)

    intercept = numpyro.sample("intercept", dist.HalfNormal(scale=1.0))
    sigma     = numpyro.sample("sigma",     dist.HalfNormal(scale=0.5))

    # ── Extra feature coefficients ─────────────────────────────────────────────
    if n_extra > 0:
        beta_extra = numpyro.sample(
            "beta_extra",
            dist.Normal(loc=0.0, scale=1.0).expand([n_extra])
        )  # can be positive or negative (quality controls, break flags)

    # ── Annual Fourier seasonality ─────────────────────────────────────────────
    gamma_s = numpyro.sample(
        "gamma_s",
        dist.Normal(loc=0.0, scale=0.1).expand([2])
    )  # sin + cos coefficients for annual cycle

    # ── Adstock via lax.scan ───────────────────────────────────────────────────
    def adstock_step(carry, x_t):
        new_carry = x_t + lag_weight * carry
        return new_carry, new_carry

    _, adstocked = jax.lax.scan(
        adstock_step,
        jnp.zeros(n_channels),
        media,
    )  # adstocked shape: (T, n_channels)

    # ── Hill saturation ────────────────────────────────────────────────────────
    safe_ad = jnp.maximum(adstocked, 1e-6)
    hill = safe_ad ** slope / (safe_ad ** slope + half_max ** slope + 1e-6)
    # hill shape: (T, n_channels)

    # ── Annual seasonality ─────────────────────────────────────────────────────
    t_arr = jnp.arange(T, dtype=jnp.float32)
    seasonality = (
        gamma_s[0] * jnp.sin(2 * jnp.pi * t_arr / 12.0)
        + gamma_s[1] * jnp.cos(2 * jnp.pi * t_arr / 12.0)
    )

    # ── Linear predictor ──────────────────────────────────────────────────────
    mu = intercept + jnp.dot(hill, beta_media) + seasonality
    if n_extra > 0:
        mu = mu + extra @ beta_extra

    numpyro.sample("obs", dist.Normal(mu, sigma), obs=target)


# ─── DATA PREP ────────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, outcome_key: str, lag: int,
                 winsorize_pct: float = None) -> dict:
    """
    Lag-shift and scale the data for Bayesian fitting.

    Lag alignment: spend[0..T-lag-1] predicts outcome[lag..T-1].
    Scaling: divide by column mean (same as lightweight_mmm CustomScaler).
    """
    spend_cols = [c for c in CHANNEL_META if c in df.columns
                  and df[c].fillna(0).sum() > 0]

    # ── Media ──────────────────────────────────────────────────────────────────
    media_raw = np.column_stack(
        [df[c].fillna(0).values.astype(float) for c in spend_cols]
    )

    # ── Target with optional winsorization ────────────────────────────────────
    target_raw = df[outcome_key].fillna(0).values.astype(float)
    winsorize_info = {}
    if winsorize_pct is not None and outcome_key == "won_arr":
        pos         = target_raw[target_raw > 0]
        cap         = float(np.percentile(pos, winsorize_pct)) if len(pos) else np.inf
        n_capped    = int((target_raw > cap).sum())
        months_cap  = (df.loc[target_raw > cap, "month"].tolist()
                       if "month" in df.columns else [])
        target_raw  = np.clip(target_raw, 0, cap)
        winsorize_info = {"pct": winsorize_pct, "cap": cap,
                          "n_capped": n_capped, "months": months_cap}

    # ── Lag shift ─────────────────────────────────────────────────────────────
    if lag > 0:
        media_aligned  = media_raw[:-lag]
        target_aligned = target_raw[lag:]
    else:
        media_aligned  = media_raw
        target_aligned = target_raw

    # ── Extra features ─────────────────────────────────────────────────────────
    extra_cols  = []
    extra_names = []
    for qc in QUALITY_CONTROLS:
        if qc in df.columns:
            col = df[qc].fillna(df[qc].median()).values.astype(float)
            extra_cols.append(col[:-lag] if lag > 0 else col)
            extra_names.append(qc)
    for bf in BREAK_FLAGS:
        if bf in df.columns and df[bf].sum() > 0:
            col = df[bf].values.astype(float)
            extra_cols.append(col[:-lag] if lag > 0 else col)
            extra_names.append(bf)

    extra_arr = np.column_stack(extra_cols) if extra_cols else None

    # ── Scaling: divide by column mean ────────────────────────────────────────
    media_means  = np.where(media_aligned.mean(0) > 0, media_aligned.mean(0), 1.0)
    target_mean  = float(target_aligned.mean()) or 1.0

    media_scaled  = (media_aligned / media_means).astype(np.float32)
    target_scaled = (target_aligned / target_mean).astype(np.float32)

    # Extra features: scale continuous QCs, keep binary flags as-is
    extra_scaled = None
    if extra_arr is not None:
        parts = []
        for i, nm in enumerate(extra_names):
            col = extra_arr[:, i].astype(np.float32)
            if nm in BREAK_FLAGS:
                parts.append(col)
            else:
                m = float(col.mean()) or 1.0
                parts.append(col / m)
        extra_scaled = np.column_stack(parts).astype(np.float32)

    raw_costs = np.array([df[c].fillna(0).sum() for c in spend_cols], dtype=float)

    return {
        "spend_cols":     spend_cols,
        "media_scaled":   media_scaled,
        "target_scaled":  target_scaled,
        "extra_scaled":   extra_scaled,
        "extra_names":    extra_names,
        "target_mean":    target_mean,
        "media_means":    media_means,
        "raw_costs":      raw_costs,
        "n_obs":          len(target_aligned),
        "target_raw":     target_aligned,
        "winsorize_info": winsorize_info,
    }


# ─── FIT ──────────────────────────────────────────────────────────────────────

def fit_bayesian(data: dict, number_warmup: int = _WARMUP_DEFAULT,
                 number_samples: int = _SAMPLES_DEFAULT,
                 number_chains: int = _CHAINS_DEFAULT,
                 seed: int = _SEED_DEFAULT) -> dict:
    """Run NUTS MCMC and return the posterior samples dict."""
    media  = jnp.array(data["media_scaled"])
    target = jnp.array(data["target_scaled"])
    extra  = jnp.array(data["extra_scaled"]) if data["extra_scaled"] is not None else None

    n_channels = media.shape[1]
    n_extra    = extra.shape[1] if extra is not None else 0

    kernel = NUTS(_bayesian_mmm, target_accept_prob=0.8)
    mcmc   = MCMC(
        kernel,
        num_warmup=number_warmup,
        num_samples=number_samples,
        num_chains=number_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        media, extra, n_channels, n_extra, target,
    )
    return mcmc.get_samples()


# ─── PREDICTIONS FROM POSTERIOR ───────────────────────────────────────────────

def _posterior_mu(samples: dict, data: dict) -> np.ndarray:
    """
    Reconstruct mu (scaled) for every posterior sample.
    Returns ndarray of shape (n_samples, T).
    """
    media       = np.array(data["media_scaled"])   # (T, C)
    extra       = data["extra_scaled"]              # (T, E) or None
    T, C        = media.shape
    n_samples   = len(samples["intercept"])
    t_arr       = np.arange(T, dtype=np.float32)

    mu_all = np.zeros((n_samples, T), dtype=np.float32)

    lag_w_all  = np.array(samples["lag_weight"])   # (S, C)
    slope_all  = np.array(samples["slope"])
    hmec_all   = np.array(samples["half_max"])
    beta_m_all = np.array(samples["beta_media"])
    ic_all     = np.array(samples["intercept"])    # (S,)
    gs_all     = np.array(samples["gamma_s"])      # (S, 2)
    be_all     = np.array(samples.get("beta_extra",
                           np.zeros((n_samples, 0))))  # (S, E)

    for s in range(n_samples):
        lag_w  = lag_w_all[s]
        slope  = slope_all[s]
        hmec   = hmec_all[s]
        beta_m = beta_m_all[s]
        ic     = float(ic_all[s])
        gamma  = gs_all[s]

        # Adstock (numpy loop — fast enough for 46 time steps)
        adstocked = np.zeros((T, C), dtype=np.float32)
        for t in range(T):
            if t == 0:
                adstocked[t] = media[t]
            else:
                adstocked[t] = media[t] + lag_w * adstocked[t - 1]

        safe_ad = np.maximum(adstocked, 1e-6)
        hill    = safe_ad ** slope / (safe_ad ** slope + hmec ** slope + 1e-6)

        seasonality = (gamma[0] * np.sin(2 * np.pi * t_arr / 12.0)
                       + gamma[1] * np.cos(2 * np.pi * t_arr / 12.0))

        mu = ic + hill @ beta_m + seasonality
        if extra is not None and be_all.shape[1] > 0:
            mu = mu + extra @ be_all[s]

        mu_all[s] = mu

    return mu_all   # (n_samples, T)


# ─── METRICS ──────────────────────────────────────────────────────────────────

def compute_fit_metrics(samples: dict, data: dict) -> dict:
    """R², MAPE, 80% credible interval using posterior mean predictions."""
    target_mean = data["target_mean"]
    y_true      = data["target_raw"]  # original units

    mu_all      = _posterior_mu(samples, data)              # (S, T) scaled
    preds_orig  = mu_all * target_mean                      # (S, T) original

    pred_mean = preds_orig.mean(axis=0)
    pred_p10  = np.percentile(preds_orig, 10, axis=0)
    pred_p90  = np.percentile(preds_orig, 90, axis=0)

    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    ss_res = float(np.sum((y_true - pred_mean) ** 2))
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    safe_y = np.where(y_true == 0, 1.0, y_true)
    mape   = float(np.mean(np.abs((y_true - pred_mean) / safe_y))) * 100.0

    return {
        "r2":        r2,
        "mape":      mape,
        "pred_mean": pred_mean,
        "pred_p10":  pred_p10,
        "pred_p90":  pred_p90,
        "y_true":    y_true,
    }


# ─── ATTRIBUTION ──────────────────────────────────────────────────────────────

def compute_attribution(samples: dict, data: dict) -> list:
    """
    Per-channel attribution and ROI with 80% credible intervals.

    Attribution share = channel's fraction of total paid contribution.
    ROI = total attributed outcome (original units) / total spend.
    """
    media       = np.array(data["media_scaled"])    # (T, C)
    target_mean = data["target_mean"]
    spend_cols  = data["spend_cols"]
    raw_costs   = data["raw_costs"]                 # (C,) total spend
    T, C        = media.shape
    n_samples   = len(samples["intercept"])

    lag_w_all  = np.array(samples["lag_weight"])
    slope_all  = np.array(samples["slope"])
    hmec_all   = np.array(samples["half_max"])
    beta_m_all = np.array(samples["beta_media"])

    all_pct = np.zeros((n_samples, C))
    all_roi = np.zeros((n_samples, C))

    for s in range(n_samples):
        lag_w  = lag_w_all[s]
        slope  = slope_all[s]
        hmec   = hmec_all[s]
        beta_m = beta_m_all[s]

        # Adstock
        adstocked = np.zeros((T, C), dtype=np.float32)
        for t in range(T):
            if t == 0:
                adstocked[t] = media[t]
            else:
                adstocked[t] = media[t] + lag_w * adstocked[t - 1]

        safe_ad = np.maximum(adstocked, 1e-6)
        hill    = safe_ad ** slope / (safe_ad ** slope + hmec ** slope + 1e-6)

        # Channel contributions in original units: sum over T → (C,)
        contrib_orig = (hill * beta_m).sum(axis=0) * target_mean

        paid_total = contrib_orig.sum()
        if paid_total > 0:
            all_pct[s] = contrib_orig / paid_total
        for c in range(C):
            if raw_costs[c] > 0:
                all_roi[s, c] = contrib_orig[c] / raw_costs[c]

    results = []
    for c, col in enumerate(spend_cols):
        meta = CHANNEL_META[col]
        pct_mean = float(all_pct[:, c].mean()) * PAID_SHARE
        pct_std  = float(all_pct[:, c].std())  * PAID_SHARE
        results.append({
            "channel":     col,
            "name":        meta["label"],
            "color":       meta["color"],
            "group":       meta["group"],
            "share":       pct_mean,
            "share_std":   pct_std,
            "attributed":  pct_mean * data["target_raw"].sum(),
            "total_spend": float(raw_costs[c]),
            "roi_mean":    float(all_roi[:, c].mean()),
            "roi_std":     float(all_roi[:, c].std()),
            "roi_p10":     float(np.percentile(all_roi[:, c], 10)),
            "roi_p90":     float(np.percentile(all_roi[:, c], 90)),
        })

    return sorted(results, key=lambda x: -x["roi_mean"])


# ─── HILL PARAMETERS ──────────────────────────────────────────────────────────

def compute_hill_params(samples: dict, data: dict) -> list:
    """Posterior summary of adstock decay + Hill saturation parameters."""
    spend_cols = data["spend_cols"]
    lw  = np.array(samples["lag_weight"])   # (S, C)
    sl  = np.array(samples["slope"])
    hm  = np.array(samples["half_max"])
    bm  = np.array(samples["beta_media"])

    params = []
    for i, col in enumerate(spend_cols):
        params.append({
            "channel":    col,
            "name":       CHANNEL_META[col]["label"],
            "color":      CHANNEL_META[col]["color"],
            "decay_mean": float(lw[:, i].mean()),
            "decay_std":  float(lw[:, i].std()),
            "alpha_mean": float(sl[:, i].mean()),
            "alpha_std":  float(sl[:, i].std()),
            "K_mean":     float(hm[:, i].mean()),
            "K_std":      float(hm[:, i].std()),
            "beta_mean":  float(bm[:, i].mean()),
            "beta_std":   float(bm[:, i].std()),
        })
    return params


# ─── TERMINAL OUTPUT ──────────────────────────────────────────────────────────

def print_v2_result(outcome_key, metrics, channels, hill_params, data, lag):
    meta = OUTCOME_META[outcome_key]
    unit = meta["unit"]
    r2   = metrics["r2"]
    mape = metrics["mape"]
    r2_tag   = "✓ Good"  if r2   >= 0.70 else ("~ Moderate" if r2   >= 0.40 else "✗ Weak")
    mape_tag = "✓ Good"  if mape <= 20   else ("~ Moderate" if mape <= 35   else "✗ High")

    print(f"\n{_DIV}")
    print(f"  OUTCOME   : {meta['label']}  (~{lag} mo lag)  [Bayesian hill-adstock]")
    print(f"  N obs     : {data['n_obs']}")
    print(f"  R²        : {r2:.3f}  {r2_tag}")
    print(f"  MAPE      : {mape:.1f}%  {mape_tag}")
    if data["winsorize_info"]:
        wi = data["winsorize_info"]
        print(f"  Winsorized: P{wi['pct']:.0f} cap={fmt_num(wi['cap'], '$')}"
              f"  {wi['n_capped']} month(s) capped")

    print(f"\n  HILL/ADSTOCK PARAMETERS (posterior mean ± std):")
    print(f"  {'Channel':<28} {'Decay':>7} {'Alpha':>7} {'K':>7} {'Beta':>9}")
    print(f"  {'-'*60}")
    for p in hill_params:
        print(f"  {p['name']:<28} "
              f"{p['decay_mean']:>5.2f}±{p['decay_std']:.2f}  "
              f"{p['alpha_mean']:>5.2f}±{p['alpha_std']:.2f}  "
              f"{p['K_mean']:>5.2f}±{p['K_std']:.2f}  "
              f"{p['beta_mean']:>7.4f}±{p['beta_std']:.4f}")

    print(f"\n  CHANNEL ATTRIBUTION & ROI (posterior mean, 80% CI):")
    print(f"  {'Channel':<28} {'Share':>6} {'Attributed':>14} {'Spend':>10}"
          f"  {'ROI mean':>9}  {'80% CI':>18}")
    print(f"  {'-'*95}")
    for c in channels:
        print(f"  {c['name']:<28} {c['share']*100:>5.1f}%"
              f"  {fmt_num(c['attributed'], unit):>14}"
              f"  {fmt_num(c['total_spend'], '$'):>10}"
              f"  {c['roi_mean']:>9.2f}x"
              f"  [{c['roi_p10']:>6.2f}x – {c['roi_p90']:>6.2f}x]")
    print(f"  {'Organic baseline':28} {ORGANIC_BASELINE*100:.0f}%"
          f"  (Gartner + G2 + LLM citations)")
    print(_DIV)


def print_comparison_table(v1_results, v2_results):
    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  MODEL COMPARISON  —  V1 OLS  vs  V2 Bayesian (hill-adstock)")
    print(sep)
    print(f"  {'Outcome':<28} {'V1 OLS R²':>10} {'V2 Bayes R²':>12}"
          f"  {'V1 MAPE':>9} {'V2 MAPE':>9}")
    print(f"  {'─'*70}")
    for ok in OUTCOME_META:
        if ok not in v1_results or ok not in v2_results:
            continue
        v1 = v1_results[ok]
        v2 = v2_results[ok]
        delta_r2   = v2["metrics"]["r2"]   - v1["r2"]
        delta_mape = v2["metrics"]["mape"] - v1["mape"]
        print(f"  {OUTCOME_META[ok]['label']:<28}"
              f"  {v1['r2']:>9.3f}  {v2['metrics']['r2']:>9.3f} ({delta_r2:+.3f})"
              f"  {v1['mape']:>7.1f}%  {v2['metrics']['mape']:>7.1f}%"
              f" ({delta_mape:+.1f}pp)")
    print(sep + "\n")


# ─── CHARTS ───────────────────────────────────────────────────────────────────

def _b64_fig(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#FAFAF8")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _make_fit_chart(metrics: dict, outcome_label: str, outcome_color: str) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    y    = metrics["y_true"]
    mean = metrics["pred_mean"]
    p10  = metrics["pred_p10"]
    p90  = metrics["pred_p90"]
    x    = np.arange(len(y))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#FAFAF8")

    ax = axes[0]
    ax.fill_between(x, p10, p90, color=outcome_color, alpha=0.2,
                    label="80% credible interval")
    ax.plot(x, y,    color="#111827",     lw=1.5, label="Actual")
    ax.plot(x, mean, color=outcome_color, lw=1.2, ls="--",
            label=f"Posterior mean  R²={metrics['r2']:.2f}")
    ax.legend(fontsize=8)
    ax.set_title(f"{outcome_label} — Actual vs Bayesian fit",
                 fontsize=10, fontweight="bold", color="#111827")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFAF8")
    ax.tick_params(labelsize=8)

    ax2 = axes[1]
    residuals = y - mean
    rc = ["#059669" if v >= 0 else "#EF4444" for v in residuals]
    ax2.bar(x, residuals, color=rc, width=0.8, alpha=0.8)
    ax2.axhline(0, color="#111827", lw=0.8)
    ax2.set_title(f"{outcome_label} — Residuals",
                  fontsize=10, fontweight="bold", color="#111827")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_facecolor("#FAFAF8")
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    return _b64_fig(fig)


def _make_attribution_chart(channels: list, outcome_label: str) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    names  = [c["name"]   for c in channels] + ["Organic"]
    shares = [c["share"] * 100 for c in channels] + [ORGANIC_BASELINE * 100]
    errors = [c["share_std"] * 100 for c in channels] + [0]
    colors = [c["color"] for c in channels] + ["#CBD5E1"]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.5 + 1)))
    fig.patch.set_facecolor("#FAFAF8")
    bars = ax.barh(names, shares, color=colors, height=0.55,
                   xerr=errors, error_kw={"ecolor": "#374151",
                                           "capsize": 4, "elinewidth": 1.2})
    for bar, val, err in zip(bars, shares, errors):
        ax.text(bar.get_width() + err + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, max(shares) * 1.4)
    ax.set_title(f"{outcome_label} — Attribution (posterior mean ± std)",
                 fontsize=10, fontweight="bold", color="#111827")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFAF8")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return _b64_fig(fig)


def _make_hill_chart(hill_params: list) -> str:
    """Saturation curves using posterior-mean Hill parameters."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    n_ch  = len(hill_params)
    ncols = 3
    nrows = (n_ch + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    fig.patch.set_facecolor("#FAFAF8")
    axes = np.array(axes).flatten()

    for i, p in enumerate(hill_params):
        ax    = axes[i]
        alpha = p["alpha_mean"]
        K     = p["K_mean"]
        x_max = max(K * 4, 0.01)
        x     = np.linspace(0, x_max, 200)
        if alpha > 0 and K > 0:
            y = x ** alpha / (x ** alpha + K ** alpha + 1e-12)
        else:
            y = np.zeros_like(x)

        ax.plot(x, y, color=p["color"], lw=2)
        ax.axvline(K, color="#9CA3AF", ls="--", lw=1,
                   label=f"K={K:.2f} (50% sat.)")
        ax.fill_between(x, 0, y, color=p["color"], alpha=0.12)
        ax.set_title(p["name"], fontsize=9, fontweight="bold", color="#111827")
        ax.set_xlabel("Scaled adstock spend", fontsize=8)
        ax.set_ylabel("Response (0–1)", fontsize=8)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#FAFAF8")
        ax.tick_params(labelsize=7)

    for j in range(n_ch, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Hill Saturation Curves — Learned from Posterior\n"
                 "Flatter curve = channel saturates faster (diminishing returns)",
                 fontsize=10, fontweight="bold", color="#111827")
    plt.tight_layout()
    return _b64_fig(fig)


def _make_roi_comparison_chart(v1_results, v2_results) -> str:
    """V1 OLS vs V2 Bayesian ROI comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    outcomes = [ok for ok in OUTCOME_META if ok in v1_results and ok in v2_results]
    n        = len(outcomes)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    fig.patch.set_facecolor("#FAFAF8")
    if n == 1:
        axes = [axes]

    for ax, ok in zip(axes, outcomes):
        v1_ch = {c["channel"]: c["roi"]
                 for c in v1_results[ok]["channels"] if c["roi"] > 0}
        v2_ch = {c["channel"]: (c["roi_mean"], c["roi_p10"], c["roi_p90"])
                 for c in v2_results[ok]["channels"]}

        all_ch = sorted(set(list(v1_ch) + list(v2_ch)),
                        key=lambda c: -v2_ch.get(c, (0,))[0])
        labels   = [CHANNEL_META[c]["label"] for c in all_ch]
        v1_vals  = [v1_ch.get(c, 0) for c in all_ch]
        v2_vals  = [v2_ch.get(c, (0, 0, 0))[0] for c in all_ch]
        v2_el    = [v2_ch.get(c, (0, 0, 0))[0] - v2_ch.get(c, (0, 0, 0))[1]
                    for c in all_ch]
        v2_eh    = [v2_ch.get(c, (0, 0, 0))[2] - v2_ch.get(c, (0, 0, 0))[0]
                    for c in all_ch]

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, v1_vals, w, label="V1 OLS",       color="#94A3B8",
               alpha=0.85, edgecolor="white")
        ax.bar(x + w / 2, v2_vals, w, label="V2 Bayesian",  color="#2563EB",
               alpha=0.85, edgecolor="white",
               yerr=[v2_el, v2_eh],
               error_kw={"ecolor": "#1E3A8A", "capsize": 4, "elinewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_title(f"{OUTCOME_META[ok]['label']} — ROI comparison",
                     fontsize=10, fontweight="bold", color="#111827")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#FAFAF8")
        ax.tick_params(labelsize=8)

    fig.suptitle("V1 OLS vs V2 Bayesian — Channel ROI\n"
                 "Error bars = 80% posterior credible interval (V2 only)",
                 fontsize=11, fontweight="bold", color="#111827", y=1.02)
    plt.tight_layout()
    return _b64_fig(fig)


# ─── HTML REPORT ──────────────────────────────────────────────────────────────

def _r2_badge(r2):
    if r2 >= 0.70:
        return "badge-good", f"R²={r2:.3f} — Strong"
    if r2 >= 0.40:
        return "badge-warn", f"R²={r2:.3f} — Moderate"
    return "badge-bad",  f"R²={r2:.3f} — Weak"


def _mape_badge(mape):
    if mape <= 20:
        return "badge-good", f"MAPE={mape:.1f}% — Good"
    if mape <= 35:
        return "badge-warn", f"MAPE={mape:.1f}% — Moderate"
    return "badge-bad",  f"MAPE={mape:.1f}% — High"


_HTML_CSS = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,sans-serif;background:#F8FAFC;color:#111827;line-height:1.55}
h1,h2,h3,h4{font-weight:700;color:#111827}
.page{max-width:1280px;margin:0 auto;padding:32px 24px}
.hero{background:linear-gradient(135deg,#4C1D95,#2563EB);color:#fff;padding:36px 40px;border-radius:16px;margin-bottom:32px}
.hero h1{font-size:2rem;margin-bottom:8px}
.hero p{opacity:.85;font-size:.97rem}
.hero .badges{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}
.hero .badge{background:rgba(255,255,255,.18);padding:4px 12px;border-radius:20px;font-size:.82rem;font-weight:600}
.card{background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.07);padding:28px 32px;margin-bottom:24px}
.card h2{font-size:1.25rem;margin-bottom:16px;border-bottom:2px solid #F1F5F9;padding-bottom:10px}
.card h3{font-size:1.05rem;margin:18px 0 10px;color:#374151}
.section-label{display:inline-block;background:#EDE9FE;color:#5B21B6;padding:3px 10px;border-radius:6px;font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-bottom:10px}
.badge-good{background:#D1FAE5;color:#065F46;padding:3px 10px;border-radius:12px;font-size:.82rem;font-weight:600;margin-right:6px}
.badge-warn{background:#FEF3C7;color:#92400E;padding:3px 10px;border-radius:12px;font-size:.82rem;font-weight:600;margin-right:6px}
.badge-bad {background:#FEE2E2;color:#991B1B;padding:3px 10px;border-radius:12px;font-size:.82rem;font-weight:600;margin-right:6px}
table{width:100%;border-collapse:collapse;font-size:.87rem}
th{background:#F1F5F9;text-align:left;padding:8px 12px;border-bottom:2px solid #E2E8F0;color:#374151}
td{padding:8px 12px;border-bottom:1px solid #F1F5F9}
tr:hover td{background:#FAFAF8}
.num{text-align:right;font-variant-numeric:tabular-nums}
.ci{color:#6B7280;font-size:.82rem}
img{max-width:100%;border-radius:8px;margin:12px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.metric-row{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:16px}
.metric{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px 16px;min-width:140px}
.metric .label{font-size:.78rem;color:#6B7280;text-transform:uppercase;letter-spacing:.04em}
.metric .value{font-size:1.4rem;font-weight:700;color:#111827}
.metric .sub{font-size:.82rem;color:#6B7280}
.outcome-header{display:flex;align-items:center;gap:12px;margin-bottom:16px}
.outcome-dot{width:14px;height:14px;border-radius:50%;flex-shrink:0}
details summary{cursor:pointer;padding:6px 0;color:#4C1D95;font-weight:600;font-size:.92rem}
details summary:hover{color:#2563EB}
.info-box{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:14px 16px;margin-top:14px;font-size:.88rem;color:#1E40AF}
.warn-box{background:#FEF9C3;border:1px solid #FDE047;border-radius:8px;padding:12px 16px;margin-top:12px;font-size:.85rem;color:#713F12}
</style>
"""


def export_html_v2(
    v2_results: dict,
    v1_results: dict,
    run_outcomes: list,
    n_warmup: int,
    n_samples: int,
    n_chains: int,
    output_path: str,
):
    from datetime import date

    # ── Pre-compute charts ────────────────────────────────────────────────────
    outcome_sections = []
    for ok in run_outcomes:
        if ok not in v2_results:
            continue
        r   = v2_results[ok]
        meta = OUTCOME_META[ok]

        fit_img   = _make_fit_chart(r["metrics"], meta["label"], meta["color"])
        attr_img  = _make_attribution_chart(r["channels"], meta["label"])
        hill_img  = _make_hill_chart(r["hill_params"])
        outcome_sections.append((ok, r, meta, fit_img, attr_img, hill_img))

    roi_cmp_img = _make_roi_comparison_chart(v1_results, v2_results)

    # ── Build HTML ─────────────────────────────────────────────────────────────
    today   = date.today().strftime("%B %d, %Y")
    n_total = n_warmup + n_samples

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Katalon MMM V2 — Bayesian Report</title>",
        _HTML_CSS,
        "</head>",
        "<body>",
        "<div class='page'>",

        # Hero
        "<div class='hero'>",
        "<h1>Katalon Marketing Mix Model — V2 Bayesian</h1>",
        f"<p>Hill-adstock model with NUTS sampling &nbsp;·&nbsp; {today}</p>",
        "<div class='badges'>",
        f"<span class='badge'>Bayesian hill-adstock (NumPyro NUTS)</span>",
        f"<span class='badge'>{n_warmup} warmup + {n_samples} samples × {n_chains} chain{'s' if n_chains>1 else ''}</span>",
        f"<span class='badge'>{len(run_outcomes)} outcomes</span>",
        f"<span class='badge'>V1 OLS comparison included</span>",
        "</div>",
        "</div>",

        # Model comparison summary
        "<div class='card'>",
        "<h2>Model Comparison — V1 OLS vs V2 Bayesian</h2>",
        "<p style='color:#6B7280;font-size:.88rem;margin-bottom:16px'>",
        "Both models fit the same data and lag settings. ",
        "R² and MAPE measure in-sample fit quality. ",
        "V2 Bayesian adds uncertainty estimates (credible intervals) on all parameters.",
        "</p>",
        "<table>",
        "<thead><tr>",
        "<th>Outcome</th><th class='num'>V1 OLS R²</th><th class='num'>V2 Bayesian R²</th>",
        "<th class='num'>V1 MAPE</th><th class='num'>V2 MAPE</th><th>Δ R²</th>",
        "</tr></thead>",
        "<tbody>",
    ]

    for ok in run_outcomes:
        if ok not in v1_results or ok not in v2_results:
            continue
        v1  = v1_results[ok]
        v2r = v2_results[ok]["metrics"]
        dr2   = v2r["r2"]   - v1["r2"]
        dmape = v2r["mape"] - v1["mape"]
        dr2_cls = "badge-good" if dr2 > 0 else "badge-warn"
        html_parts.append(
            f"<tr><td>{OUTCOME_META[ok]['label']}</td>"
            f"<td class='num'>{v1['r2']:.3f}</td>"
            f"<td class='num'>{v2r['r2']:.3f}</td>"
            f"<td class='num'>{v1['mape']:.1f}%</td>"
            f"<td class='num'>{v2r['mape']:.1f}%</td>"
            f"<td><span class='{dr2_cls}'>{dr2:+.3f}</span></td></tr>"
        )

    html_parts += [
        "</tbody></table>",
        "</div>",
    ]

    # ROI comparison chart
    if roi_cmp_img:
        html_parts += [
            "<div class='card'>",
            "<h2>V1 vs V2 ROI Comparison</h2>",
            "<p style='color:#6B7280;font-size:.88rem;margin-bottom:12px'>",
            "Error bars on V2 Bayesian bars show 80% posterior credible interval.",
            "</p>",
            f"<img src='data:image/png;base64,{roi_cmp_img}' alt='ROI comparison'>",
            "</div>",
        ]

    # ── Per-outcome sections ──────────────────────────────────────────────────
    for ok, r, meta, fit_img, attr_img, hill_img in outcome_sections:
        lag  = r["lag"]
        met  = r["metrics"]
        chs  = r["channels"]
        hps  = r["hill_params"]
        wi   = r["data"]["winsorize_info"]

        r2_cls,   r2_txt   = _r2_badge(met["r2"])
        mape_cls, mape_txt = _mape_badge(met["mape"])

        # Winsorize badge
        wins_badge = ""
        if wi:
            months_str = str(wi["months"])
            plural = "s" if wi["n_capped"] != 1 else ""
            wins_badge = (
                f'<span class="badge-warn" title="Months capped: {months_str}">'
                f'Winsorized P{wi["pct"]:.0f} '
                f'(cap {fmt_num(wi["cap"], "$")} · {wi["n_capped"]} month{plural} capped)'
                f'</span>'
            )

        _color   = meta["color"]
        _label   = meta["label"]
        _n_obs   = r["data"]["n_obs"]
        html_parts += [
            f"<div class='card' id='{ok}'>",
            "<div class='outcome-header'>",
            f"<div class='outcome-dot' style='background:{_color}'></div>",
            f"<h2 style='border:none;margin:0;padding:0'>{_label} "
            f"<span style='font-weight:400;font-size:.88rem;color:#6B7280'>"
            f"(~{lag} month lag)</span></h2>",
            "</div>",
            "<div class='metric-row'>",
            f"<div class='metric'><div class='label'>Model fit</div>"
            f"<span class='{r2_cls}'>{r2_txt}</span></div>",
            f"<div class='metric'><div class='label'>Prediction error</div>"
            f"<span class='{mape_cls}'>{mape_txt}</span></div>",
            f"<div class='metric'><div class='label'>Observations</div>"
            f"<div class='value'>{_n_obs}</div></div>",
            "</div>",
        ]
        if wins_badge:
            html_parts.append(f"<p style='margin-bottom:12px'>{wins_badge}</p>")

        # Fit chart
        if fit_img:
            html_parts.append(
                f"<img src='data:image/png;base64,{fit_img}' alt='Fit chart'>"
            )

        # Attribution table + chart
        html_parts += [
            "<h3>Channel Attribution & ROI</h3>",
            "<p style='color:#6B7280;font-size:.85rem;margin-bottom:12px'>",
            "Share = fraction of paid-attributed outcome. ROI = attributed outcome / spend. "
            "80% CI from posterior samples.",
            "</p>",
        ]
        if attr_img:
            html_parts.append(
                f"<img src='data:image/png;base64,{attr_img}' alt='Attribution chart'>"
            )

        html_parts += [
            "<table style='margin-top:16px'>",
            "<thead><tr><th>Channel</th><th class='num'>Share</th>"
            "<th class='num'>Attributed</th><th class='num'>Spend</th>"
            "<th class='num'>ROI (mean)</th><th class='num'>80% CI</th></tr></thead>",
            "<tbody>",
        ]
        for c in chs:
            html_parts.append(
                f"<tr><td>{c['name']}</td>"
                f"<td class='num'>{c['share']*100:.1f}%</td>"
                f"<td class='num'>{fmt_num(c['attributed'], meta['unit'])}</td>"
                f"<td class='num'>{fmt_num(c['total_spend'], '$')}</td>"
                f"<td class='num'><b>{c['roi_mean']:.2f}x</b></td>"
                f"<td class='num ci'>[{c['roi_p10']:.2f}x – {c['roi_p90']:.2f}x]</td></tr>"
            )
        html_parts += [
            f"<tr><td><em>Organic baseline</em></td>"
            f"<td class='num'>{ORGANIC_BASELINE*100:.0f}%</td>"
            f"<td colspan='4' style='color:#6B7280;font-size:.82rem'>"
            f"Gartner Visionary + G2 + LLM citations — not attributed to paid channels</td></tr>",
            "</tbody></table>",
        ]

        # Hill saturation curves
        if hill_img:
            html_parts += [
                "<h3>Hill Saturation Curves</h3>",
                "<p style='color:#6B7280;font-size:.85rem;margin-bottom:12px'>",
                "Learned diminishing-returns curves. ",
                "K = half-saturation point (scaled spend at 50% of max response). "
                "α = shape: >1 = S-curve, <1 = concave.",
                "</p>",
                f"<img src='data:image/png;base64,{hill_img}' alt='Hill curves'>",
            ]

        # Parameter table
        html_parts += [
            "<details style='margin-top:16px'>",
            "<summary>Parameter table — Posterior mean ± std</summary>",
            "<table style='margin-top:12px'>",
            "<thead><tr><th>Channel</th><th class='num'>Decay (adstock)</th>"
            "<th class='num'>α (Hill shape)</th><th class='num'>K (half-sat.)</th>"
            "<th class='num'>β (channel coef.)</th></tr></thead>",
            "<tbody>",
        ]
        for p in hps:
            html_parts.append(
                f"<tr><td>{p['name']}</td>"
                f"<td class='num'>{p['decay_mean']:.2f}±{p['decay_std']:.2f}</td>"
                f"<td class='num'>{p['alpha_mean']:.2f}±{p['alpha_std']:.2f}</td>"
                f"<td class='num'>{p['K_mean']:.2f}±{p['K_std']:.2f}</td>"
                f"<td class='num'>{p['beta_mean']:.4f}±{p['beta_std']:.4f}</td></tr>"
            )
        html_parts += ["</tbody></table>", "</details>", "</div>"]

    # ── What Bayesian adds ─────────────────────────────────────────────────────
    html_parts += [
        "<div class='card'>",
        "<h2>What Bayesian Adds Over OLS</h2>",
        "<div class='grid2'>",

        "<div>",
        "<h3>V1 OLS (baseline)</h3>",
        "<ul style='list-style:disc;padding-left:20px;font-size:.88rem;line-height:1.7'>",
        "<li>Fixed adstock decay rates (hand-tuned per channel)</li>",
        "<li>Linear saturation — assumes ROI constant across spend levels</li>",
        "<li>Point estimates only — no uncertainty on attribution</li>",
        "<li>Fast: runs in under 1 second</li>",
        "<li>Interpretable: OLS coefficients map directly to attribution</li>",
        "</ul>",
        "</div>",

        "<div>",
        "<h3>V2 Bayesian (this report)</h3>",
        "<ul style='list-style:disc;padding-left:20px;font-size:.88rem;line-height:1.7'>",
        "<li>Adstock decay learned from data (posterior distribution)</li>",
        "<li>Hill saturation captures diminishing returns explicitly</li>",
        "<li>Credible intervals on every ROI and attribution estimate</li>",
        "<li>Informative priors prevent overfitting on small sample (n≈46)</li>",
        "<li>Slower: ~8s per outcome in quick mode, ~60s in full mode</li>",
        "</ul>",
        "</div>",
        "</div>",  # grid2

        "<div class='info-box' style='margin-top:18px'>",
        "<strong>When to trust V2 more:</strong> When V2 R² ≥ V1 R² and the credible "
        "intervals are narrow. Wide 80% CIs (e.g. ROI 0.1x–15x) mean the data is "
        "insufficient to pin down that channel's effect — the posterior is prior-dominated.",
        "</div>",

        "<div class='warn-box'>",
        "<strong>ROI uncertainty caveat:</strong> With ~40–46 observations and 9 channels, "
        "Bayesian credible intervals are wide. Use ROI rankings (relative ordering) rather "
        "than absolute ROI values for budget decisions.",
        "</div>",
        "</div>",  # card

        "</div>",  # page
        "</body>",
        "</html>",
    ]

    Path(output_path).write_text("\n".join(html_parts), encoding="utf-8")
    print(f"\n  HTML report saved → {output_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Katalon MMM V2 — Bayesian hill-adstock via NumPyro NUTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--json",          type=str, required=True)
    p.add_argument("--outcome",       type=str, default=None,
                   choices=list(OUTCOME_META.keys()))
    p.add_argument("--lag-mqls",      type=int, default=0)
    p.add_argument("--lag-pipeline",  type=int, default=2)
    p.add_argument("--lag-arr",       type=int, default=5)
    p.add_argument("--winsorize-arr", type=float, default=None, metavar="PCT")
    p.add_argument("--samples",       type=int, default=_SAMPLES_DEFAULT)
    p.add_argument("--warmup",        type=int, default=_WARMUP_DEFAULT)
    p.add_argument("--chains",        type=int, default=_CHAINS_DEFAULT)
    p.add_argument("--quick",         action="store_true",
                   help=f"Fast mode: {_WARMUP_QUICK} warmup + {_SAMPLES_QUICK} samples")
    p.add_argument("--seed",          type=int, default=_SEED_DEFAULT)
    p.add_argument("--export-html",   type=str,
                   default="katalon_mmm_v2_report.html")
    args = p.parse_args()

    lag_map = {
        "mqls":             args.lag_mqls,
        "inbound_pipeline": args.lag_pipeline,
        "won_arr":          args.lag_arr,
    }

    n_warmup  = _WARMUP_QUICK  if args.quick else args.warmup
    n_samples = _SAMPLES_QUICK if args.quick else args.samples

    # ── Load data ──────────────────────────────────────────────────────────────
    df_raw = load_json_data(args.json)
    df_raw = add_seasonality_flags(df_raw)
    print(f"\n  Loaded {len(df_raw)} monthly rows from {args.json}")

    available    = [k for k in OUTCOME_META if k in df_raw.columns]
    run_outcomes = [args.outcome] if args.outcome else available

    mode_str = f"{n_warmup} warmup + {n_samples} samples × {args.chains} chain" + \
               ("s" if args.chains > 1 else "")
    print(f"  Model      : Bayesian hill-adstock (NumPyro NUTS)")
    print(f"  MCMC       : {mode_str}")
    print(f"  Outcomes   : {', '.join(run_outcomes)}")
    print(f"  Lags       : " + "  ".join(
        f"{ok}={lag_map[ok]}mo" for ok in run_outcomes))
    if args.winsorize_arr:
        print(f"  Winsorize  : won_arr at P{args.winsorize_arr:.0f}")
    exp = max(8, 8 * len(run_outcomes))
    print(f"\n  Expected time: ~{exp}–{exp*3}s total (quick mode).\n")

    # ── V1 OLS for comparison ──────────────────────────────────────────────────
    print("  Running V1 OLS for comparison ...")
    v1_results = {}
    for ok in run_outcomes:
        df_v1 = df_raw.copy()
        if ok == "won_arr" and args.winsorize_arr:
            cap = float(np.percentile(df_v1[ok].dropna(), args.winsorize_arr))
            df_v1[ok] = df_v1[ok].clip(upper=cap)
        v1_results[ok] = run_mmm_ols(
            df_v1, ok, lag=lag_map[ok],
            use_seasonality=True, monthly=True, use_quality_controls=True,
        )
    print("  V1 OLS done.")

    # ── V2 Bayesian ────────────────────────────────────────────────────────────
    v2_results = {}
    for ok in run_outcomes:
        lag = lag_map[ok]
        print(f"\n  Fitting Bayesian model: {OUTCOME_META[ok]['label']} ...", flush=True)

        data = prepare_data(
            df_raw, ok, lag,
            winsorize_pct=args.winsorize_arr if ok == "won_arr" else None,
        )

        samples = fit_bayesian(
            data,
            number_warmup=n_warmup,
            number_samples=n_samples,
            number_chains=args.chains,
            seed=args.seed,
        )

        metrics    = compute_fit_metrics(samples, data)
        channels   = compute_attribution(samples, data)
        hill_prms  = compute_hill_params(samples, data)

        v2_results[ok] = {
            "lag":        lag,
            "metrics":    metrics,
            "channels":   channels,
            "hill_params":hill_prms,
            "data":       data,
        }

        print_v2_result(ok, metrics, channels, hill_prms, data, lag)

    # ── Comparison table ───────────────────────────────────────────────────────
    print_comparison_table(v1_results, v2_results)

    # ── HTML export ────────────────────────────────────────────────────────────
    export_html_v2(
        v2_results, v1_results, run_outcomes,
        n_warmup, n_samples, args.chains,
        args.export_html,
    )


if __name__ == "__main__":
    main()
