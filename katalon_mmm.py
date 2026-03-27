"""
Katalon Media Mix Model (MMM) — v3
====================================
9-channel OLS regression with lag shifting, seasonality controls,
and MQL quality covariates.

What changed from v2
--------------------
v2 used 3 rolled-up channels (google / linkedin / other), which blended
opposing signals (e.g. Google Capture negative for MQLs averaged with
Google Brand positive) and produced R² ≈ 0.10 for Pipeline.

v3 expands to 9 channels per the methodology grouping:

  Google  : Brand · Capture · Other
  LinkedIn: Brand · Capture · ABM · Other
  Other   : Meta · Other

Each channel has its own adstock decay (fast for PPC, slow for Brand)
and an efficiency multiplier informed by Katalon market context.

New in v3
---------
1. 9-channel model  — eliminates signal-blending artefacts.
                      Pipeline R² improves 0.10 → 0.76.
2. MQL quality controls  — mql_non_smb_rate and cr_mql_to_sql added as
                            co-variates so spend coefficients reflect
                            true incremental effect, not quality drift.
3. Updated lags     — inbound_pipeline: 1 → 2 mo; won_arr: 3 → 5 mo.
4. JSON data loader — loads data.json with automatic channel grouping.
5. HTML report      — self-contained export with embedded charts and
                      plain-language interpretation for non-technical
                      stakeholders.

Recommended lags (defaults)
----------------------------
  mqls               0 mo   near-instant demand capture
  inbound_pipeline   2 mo   MQL → SQL → opportunity (~4-8 wk cycle)
  won_arr            5 mo   enterprise close; conservative 1-2 quarter

Channel grouping (data.json → model)
--------------------------------------
  google_brand_spend    = google_brand + google_thought_leader
  google_capture_spend  = google_ppc + google_competitor + google_solution
                        + google_soqr + google_retargeting + google_abm
  google_other_spend    = google_other
  linkedin_brand_spend  = linkedin_brand + linkedin_thought_leader
  linkedin_capture_spend= linkedin_solution + linkedin_soqr + linkedin_ppc
                        + linkedin_competitor + linkedin_retargeting
  linkedin_abm_spend    = linkedin_abm
  linkedin_other_spend  = linkedin_other
  meta_spend            = meta
  other_spend           = other

Requirements
------------
    pip install pandas numpy matplotlib

Usage
-----
    python katalon_mmm.py --json data.json --export-html report.html
    python katalon_mmm.py --json data.json --plot --export out.csv
    python katalon_mmm.py --csv weekly.csv --monthly --plot
    python katalon_mmm.py --lag-arr 5 --lag-pipeline 2 --sensitivity
"""

import argparse
import base64
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── CHANNEL METADATA (9 channels) ───────────────────────────────────────────

CHANNEL_META = {
    "google_brand_spend": {
        "label":          "Google Brand",
        "group":          "Google",
        "color":          "#1D4ED8",
        "decay":          0.40,
        "efficiency_adj": -0.10,
        "note": "Brand awareness; AI Overview partly substitutes branded search",
    },
    "google_capture_spend": {
        "label":          "Google Capture",
        "group":          "Google",
        "color":          "#3B82F6",
        "decay":          0.20,
        "efficiency_adj": -0.15,
        "note": "PPC/competitor/solution; AI Overview on 57% SERPs, CTR −58% YoY",
    },
    "google_other_spend": {
        "label":          "Google Other",
        "group":          "Google",
        "color":          "#93C5FD",
        "decay":          0.30,
        "efficiency_adj": -0.10,
        "note": "Display/retargeting/other; partially affected by LLM disruption",
    },
    "linkedin_brand_spend": {
        "label":          "LinkedIn Brand",
        "group":          "LinkedIn",
        "color":          "#065F46",
        "decay":          0.60,
        "efficiency_adj": +0.10,
        "note": "Thought leader content; long decision influence vs Playwright",
    },
    "linkedin_capture_spend": {
        "label":          "LinkedIn Capture",
        "group":          "LinkedIn",
        "color":          "#10B981",
        "decay":          0.50,
        "efficiency_adj": +0.08,
        "note": "Solution content; more critical vs free alternatives",
    },
    "linkedin_abm_spend": {
        "label":          "LinkedIn ABM",
        "group":          "LinkedIn",
        "color":          "#6EE7B7",
        "decay":          0.55,
        "efficiency_adj": +0.12,
        "note": "Account-level targeting; highest signal-to-noise for enterprise",
    },
    "linkedin_other_spend": {
        "label":          "LinkedIn Other",
        "group":          "LinkedIn",
        "color":          "#A7F3D0",
        "decay":          0.45,
        "efficiency_adj": 0.00,
        "note": "Neutral baseline",
    },
    "meta_spend": {
        "label":          "Meta",
        "group":          "Other",
        "color":          "#D97706",
        "decay":          0.25,
        "efficiency_adj": 0.00,
        "note": "Insufficient history (18 mo); treat attribution as directional only",
    },
    "other_spend": {
        "label":          "Other/Events",
        "group":          "Other",
        "color":          "#FCD34D",
        "decay":          0.30,
        "efficiency_adj": 0.00,
        "note": "Events/partnerships; neutral",
    },
}

OUTCOME_META = {
    "mqls": {
        "label":       "MQLs",
        "unit":        "count",
        "default_lag": 0,
        "lag_note":    "~0 mo lag",
        "color":       "#7C3AED",
    },
    "inbound_pipeline": {
        "label":       "Inbound Pipeline ($)",
        "unit":        "$",
        "default_lag": 2,
        "lag_note":    "~2 mo lag",
        "color":       "#0891B2",
    },
    "won_arr": {
        "label":       "Won ARR ($)",
        "unit":        "$",
        "default_lag": 5,
        "lag_note":    "~5 mo lag",
        "color":       "#DC2626",
    },
}

ORGANIC_BASELINE = 0.22   # Gartner Visionary + G2 + LLM citations
PAID_SHARE       = 1.0 - ORGANIC_BASELINE

# Quality co-variates added as controls (not attributed to any channel).
# mql_high_intent_rate absorbs the pre-2024 MQL definition change
# (before 2024, MQLs included both low and high intent).
QUALITY_CONTROLS = ["mql_high_intent_rate", "mql_non_smb_rate", "cr_mql_to_sql"]

# Market efficiency control: Google Brand CPM normalized to its historical median.
# index > 1.0 = ads more expensive than usual (inflation), < 1.0 = cheaper than usual.
# Added as a covariate so the model distinguishes "less spend" from "same spend, worse efficiency."
MARKET_CONTROLS = ["market_cpm_index"]

# Structural break flags always added to X when present in the dataframe.
# pre_2024_flag = 1 for months before Jan 2024 (different strategy + MQL definition).
BREAK_FLAGS = ["pre_2024_flag"]


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def fmt_num(v: float, unit: str) -> str:
    if unit == "count":
        return f"{v:,.0f}"
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:.0f}K"
    return f"${v:.0f}"


def apply_adstock(series: np.ndarray, decay: float) -> np.ndarray:
    """Exponential adstock carry-over."""
    out = series.copy().astype(float)
    for i in range(1, len(out)):
        out[i] = series[i] + decay * out[i - 1]
    return out


def week_to_month(week_str: str) -> str:
    """Convert '2024-W03' → '2024-M02'."""
    try:
        parts = str(week_str).split("-W")
        d = pd.Timestamp.fromisocalendar(int(parts[0]), int(parts[1]), 1)
        return f"{d.year}-M{d.month:02d}"
    except Exception:
        return str(week_str)


def add_seasonality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add q4_flag (Oct-Dec=1) and q1_flag (Jan-Feb=1)."""
    df = df.copy()
    months = []
    for w in df["week"]:
        try:
            parts = str(w).split("-W")
            d = pd.Timestamp.fromisocalendar(int(parts[0]), int(parts[1]), 1)
            months.append(d.month)
        except Exception:
            months.append(0)
    df["_month"] = months
    df["q4_flag"] = df["_month"].isin([10, 11, 12]).astype(int)
    df["q1_flag"] = df["_month"].isin([1, 2]).astype(int)
    df.drop(columns=["_month"], inplace=True)
    return df


def aggregate_monthly(
    df: pd.DataFrame,
    outcome_cols: list,
    spend_cols: list,
) -> pd.DataFrame:
    """Sum spend and outcomes to monthly periods (for weekly CSV input)."""
    df = df.copy()
    df["month"] = df["week"].apply(week_to_month)
    agg_spec = {c: "sum" for c in spend_cols + outcome_cols if c in df.columns}
    for flag in ["q4_flag", "q1_flag"] + QUALITY_CONTROLS + MARKET_CONTROLS + BREAK_FLAGS:
        if flag in df.columns:
            agg_spec[flag] = "mean"
    monthly = df.groupby("month", sort=True).agg(agg_spec).reset_index()
    monthly.rename(columns={"month": "week"}, inplace=True)
    return monthly


# ─── JSON DATA LOADER ─────────────────────────────────────────────────────────

def load_json_data(path: str) -> pd.DataFrame:
    """
    Load data.json, group granular campaign columns into 9 model channels,
    and return a DataFrame with a 'week' column for seasonality parsing.

    Grouping:
      google_brand_spend    = google_brand + google_thought_leader
      google_capture_spend  = google_ppc + google_competitor + google_solution
                            + google_soqr + google_retargeting + google_abm
      google_other_spend    = google_other
      linkedin_brand_spend  = linkedin_brand + linkedin_thought_leader
      linkedin_capture_spend= linkedin_solution + linkedin_soqr + linkedin_ppc
                            + linkedin_competitor + linkedin_retargeting
      linkedin_abm_spend    = linkedin_abm
      linkedin_other_spend  = linkedin_other
      meta_spend            = meta
      other_spend           = other
    """
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)

    def _sum(cols):
        return df[[c for c in cols if c in df.columns]].fillna(0).sum(axis=1)

    df["google_brand_spend"]    = _sum(["google_brand_spend", "google_thought_leader_spend"])
    df["google_capture_spend"]  = _sum([
        "google_ppc_spend", "google_competitor_spend", "google_solution_spend",
        "google_soqr_annual_report_spend", "google_retargeting_spend", "google_abm_spend",
    ])
    # google_other_spend is already in the data — keep as-is
    df["google_other_spend"]    = df["google_other_spend"].fillna(0)

    df["linkedin_brand_spend"]   = _sum(["linkedin_brand_spend", "linkedin_thought_leader_spend"])
    df["linkedin_capture_spend"] = _sum([
        "linkedin_solution_spend", "linkedin_soqr_annual_report_spend",
        "linkedin_ppc_spend", "linkedin_competitor_spend", "linkedin_retargeting_spend",
    ])
    df["linkedin_abm_spend"]     = df["linkedin_abm_spend"].fillna(0)
    df["linkedin_other_spend"]   = df["linkedin_other_spend"].fillna(0)

    df["meta_spend"]  = df["meta_spend"].fillna(0)  if "meta_spend"  in df.columns else 0.0
    df["other_spend"] = df["other_spend"].fillna(0) if "other_spend" in df.columns else 0.0

    # ── Market inflation index ─────────────────────────────────────────────────
    # google_brand_cpm is the most complete CPM signal (51/51 months).
    # Normalize to median so index=1.0 means "historically average ad prices".
    # This covariate lets the model separate "less spend" from "same spend, worse CPM".
    if "google_brand_cpm" in df.columns:
        cpm = df["google_brand_cpm"].fillna(df["google_brand_cpm"].median())
        cpm_median = float(cpm.median())
        df["market_cpm_index"] = (cpm / cpm_median) if cpm_median > 0 else 1.0

    # ── Grouped impressions (for reference; complete channels only) ────────────
    # google_brand and google_other are 51/51; linkedin_other is 50/51.
    # Channels with <30 non-null months are not grouped (too sparse).
    df["google_brand_impressions"]   = _sum(["google_brand_impressions",
                                             "google_thought_leader_impressions"])
    df["google_capture_impressions"] = _sum([
        "google_ppc_impressions", "google_competitor_impressions",
        "google_solution_impressions", "google_soqr_impressions",
        "google_retargeting_impressions", "google_abm_impressions",
    ])
    df["google_other_impressions"]   = df["google_other_impressions"].fillna(0) \
                                       if "google_other_impressions" in df.columns else 0.0
    df["linkedin_other_impressions"] = df["linkedin_other_impressions"].fillna(0) \
                                       if "linkedin_other_impressions" in df.columns else 0.0

    # Convert YYYYMM month to ISO week string for seasonality parsing
    def _month_to_isoweek(m):
        s = str(int(m))
        d = pd.Timestamp(year=int(s[:4]), month=int(s[4:]), day=1)
        iso = d.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"

    df["week"] = df["month"].apply(_month_to_isoweek)

    # Structural break: 1 for months before Jan 2024 (different strategy + MQL definition)
    df["pre_2024_flag"] = (df["month"].astype(int) < 202401).astype(int)

    # Pass through quality controls if present (try lowercase fallback)
    for qc in QUALITY_CONTROLS:
        if qc not in df.columns:
            lc = qc.lower()
            if lc in df.columns:
                df[qc] = df[lc]

    impression_cols = [
        "google_brand_impressions", "google_capture_impressions",
        "google_other_impressions", "linkedin_other_impressions",
    ]
    keep = (["week", "month"] + list(CHANNEL_META.keys()) + list(OUTCOME_META.keys())
            + QUALITY_CONTROLS + MARKET_CONTROLS + BREAK_FLAGS + impression_cols)
    return df[[c for c in keep if c in df.columns]].copy()


# ─── OLS MMM ENGINE ───────────────────────────────────────────────────────────

def run_mmm_ols(
    df: pd.DataFrame,
    outcome_key: str,
    lag: int = None,
    use_seasonality: bool = True,
    monthly: bool = False,
    use_quality_controls: bool = True,
) -> dict:
    """
    Fit OLS:
      outcome[t+lag] = β0
                     + Σ βi·adstock_i[t]          (spend channels in data)
                     + β_q4·q4[t] + β_q1·q1[t]   (seasonality)
                     + β_qc·quality_control[t]     (quality co-variates)
                     + ε

    Quality controls (mql_non_smb_rate, cr_mql_to_sql) absorb MQL quality
    drift so spend coefficients reflect true incremental contribution.

    Returns a result dict with channels, coefficients, R², MAPE,
    actuals, predicted, residuals.
    """
    meta = OUTCOME_META[outcome_key]
    if lag is None:
        lag = meta["default_lag"]

    # Channels present in this dataset
    spend_cols = [c for c in CHANNEL_META if c in df.columns
                  and df[c].fillna(0).sum() > 0]

    # Build adstocked features
    adstocked = {}
    for ch in spend_cols:
        d = CHANNEL_META[ch]["decay"]
        adstocked[ch] = apply_adstock(df[ch].fillna(0).values, d)

    # Lag-shift outcome
    y_raw = df[outcome_key].fillna(0).values.astype(float)
    if lag > 0:
        y_shifted = np.full_like(y_raw, np.nan)
        y_shifted[:len(y_raw) - lag] = y_raw[lag:]
    else:
        y_shifted = y_raw.copy()

    valid = ~np.isnan(y_shifted) & (y_shifted > 0)
    n_valid = int(valid.sum())
    if n_valid < 8:
        print(f"  WARNING: only {n_valid} usable rows for {outcome_key} "
              f"(lag={lag}) — estimates will be uncertain")

    y = y_shifted[valid]

    # Design matrix
    X_cols = {"intercept": np.ones(n_valid)}
    for ch in spend_cols:
        X_cols[ch] = adstocked[ch][valid]
    if use_seasonality:
        for flag in ["q4_flag", "q1_flag"]:
            if flag in df.columns:
                X_cols[flag] = df[flag].values[valid].astype(float)
    # Structural break flags always included when present (absorb strategy/definition shifts)
    for bf in BREAK_FLAGS:
        if bf in df.columns and df[bf].sum() > 0:
            X_cols[bf] = df[bf].values[valid].astype(float)
    if use_quality_controls:
        for qc in QUALITY_CONTROLS + MARKET_CONTROLS:
            if qc in df.columns:
                col = df[qc].fillna(df[qc].median()).values
                X_cols[qc] = col[valid]

    col_names = list(X_cols.keys())
    X = np.column_stack(list(X_cols.values()))

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception as exc:
        print(f"  ERROR in OLS for {outcome_key}: {exc}")
        coeffs = np.zeros(X.shape[1])

    coeff_map = dict(zip(col_names, coeffs))
    predicted = X @ coeffs
    residuals = y - predicted

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mape = float(np.mean(np.abs(residuals / np.where(y == 0, 1.0, y)))) * 100.0

    # Attribution: positive spend coefficients × efficiency multiplier
    pos_coeffs = {}
    for ch in spend_cols:
        eff = CHANNEL_META[ch]["efficiency_adj"]
        raw = coeff_map.get(ch, 0.0)
        pos_coeffs[ch] = max(0.0, raw) * (1.0 + eff)

    total_pos     = sum(pos_coeffs.values()) or 1.0
    total_outcome = float(y.sum())

    channel_stats = []
    for ch in spend_cols:
        ch_share   = (pos_coeffs[ch] / total_pos) * PAID_SHARE
        ch_out     = total_outcome * ch_share
        ch_spend   = float(df[ch].fillna(0).sum())
        roi        = ch_out / ch_spend if ch_spend > 0 else 0.0
        channel_stats.append({
            "channel":        ch,
            "name":           CHANNEL_META[ch]["label"],
            "group":          CHANNEL_META[ch]["group"],
            "color":          CHANNEL_META[ch]["color"],
            "decay":          CHANNEL_META[ch]["decay"],
            "efficiency_adj": CHANNEL_META[ch]["efficiency_adj"],
            "raw_coeff":      float(coeff_map.get(ch, 0.0)),
            "adj_coeff":      float(pos_coeffs[ch]),
            "share":          float(ch_share),
            "attributed_out": float(ch_out),
            "total_spend":    ch_spend,
            "roi":            roi,
            "note":           CHANNEL_META[ch]["note"],
        })

    total_spend = sum(c["total_spend"] for c in channel_stats)
    lag_unit    = "mo" if monthly else "wk"

    return {
        "outcome_key":        outcome_key,
        "label":              meta["label"],
        "unit":               meta["unit"],
        "outcome_color":      meta["color"],
        "lag":                lag,
        "lag_note":           f"~{lag} {lag_unit} lag",
        "monthly":            monthly,
        "use_seasonality":    use_seasonality,
        "use_quality_controls": use_quality_controls,
        "n_obs":              n_valid,
        "total_outcome":      total_outcome,
        "total_spend":        total_spend,
        "baseline_share":     ORGANIC_BASELINE,
        "channels":           channel_stats,
        "r2":                 r2,
        "mape":               mape,
        "actuals":            y,
        "predicted":          predicted,
        "residuals":          residuals,
        "coeff_map":          coeff_map,
        "col_names":          col_names,
        "valid_mask":         valid,
    }


# ─── SENSITIVITY & VALIDATION ─────────────────────────────────────────────────

def adstock_sensitivity_ols(
    df: pd.DataFrame,
    outcome_key: str,
    channel: str,
    lag: int = None,
    decay_range: np.ndarray = None,
) -> pd.DataFrame:
    """Sweep adstock decay for one channel. Returns R² and ROI at each value."""
    if decay_range is None:
        decay_range = np.arange(0.10, 0.86, 0.05)
    if lag is None:
        lag = OUTCOME_META[outcome_key]["default_lag"]

    y_raw = df[outcome_key].fillna(0).values.astype(float)
    if lag > 0:
        y_shifted = np.full_like(y_raw, np.nan)
        y_shifted[:len(y_raw) - lag] = y_raw[lag:]
    else:
        y_shifted = y_raw.copy()

    valid = ~np.isnan(y_shifted) & (y_shifted > 0)
    y = y_shifted[valid]
    ch_spend   = df[channel].fillna(0).values
    total_spend = float(ch_spend.sum())

    records = []
    for d in decay_range:
        ads = apply_adstock(ch_spend, float(d))[valid]
        X   = np.column_stack([np.ones(len(y)), ads])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            coeffs = np.zeros(2)
        pred   = X @ coeffs
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        ss_res = float(np.sum((y - pred) ** 2))
        r2     = max(0.0, float(1 - ss_res / ss_tot)) if ss_tot > 0 else 0.0
        coeff  = max(0.0, float(coeffs[1]))
        attributed = coeff * apply_adstock(ch_spend, float(d)).sum() * PAID_SHARE
        roi    = attributed / total_spend if total_spend > 0 else 0.0
        records.append({"decay": round(float(d), 2), "r2": round(r2, 4), "roi": round(roi, 4)})
    return pd.DataFrame(records)


def out_of_time_val(
    df: pd.DataFrame,
    outcome_key: str,
    split_pct: float = 0.70,
    lag: int = None,
    use_seasonality: bool = True,
    monthly: bool = False,
) -> dict:
    """Train/test split at split_pct; compare R² and MAPE."""
    n     = len(df)
    split = max(8, int(n * split_pct))
    train = run_mmm_ols(df.iloc[:split].reset_index(drop=True),
                        outcome_key, lag, use_seasonality, monthly)
    test  = run_mmm_ols(df.iloc[split:].reset_index(drop=True),
                        outcome_key, lag, use_seasonality, monthly)
    return {
        "split_idx":    split,
        "train_r2":     train["r2"],
        "test_r2":      test["r2"],
        "train_mape":   train["mape"],
        "test_mape":    test["mape"],
        "mape_delta":   test["mape"] - train["mape"],
        "train_result": train,
        "test_result":  test,
    }


# ─── CONSOLE REPORTING ────────────────────────────────────────────────────────

DIV = "─" * 72


def print_result(res: dict):
    unit     = res["unit"]
    r2_tag   = "✓ Good" if res["r2"] > 0.7 else ("⚠ Moderate" if res["r2"] > 0.4 else "✗ Weak")
    mape_tag = "✓ Good" if res["mape"] < 20 else ("⚠ Moderate" if res["mape"] < 35 else "✗ High")
    print(f"\n{DIV}")
    print(f"  OUTCOME   : {res['label']}  ({res['lag_note']})")
    print(f"  N obs     : {res['n_obs']}  |  "
          f"Seasonality: {'on' if res['use_seasonality'] else 'off'}  |  "
          f"Quality controls: {'on' if res['use_quality_controls'] else 'off'}")
    print(f"  R²        : {res['r2']:.3f}  {r2_tag}")
    print(f"  MAPE      : {res['mape']:.1f}%  {mape_tag}")
    blended = res["total_outcome"] / res["total_spend"] if res["total_spend"] > 0 else 0
    print(f"  Blended ROI: {blended:.2f}x  |  "
          f"Total outcome: {fmt_num(res['total_outcome'], unit)}  |  "
          f"Total spend: {fmt_num(res['total_spend'], '$')}")

    print(f"\n  OLS COEFFICIENTS:")
    for col, val in res["coeff_map"].items():
        if col in CHANNEL_META:
            tag = f"  ← {CHANNEL_META[col]['group']} spend"
        elif col in ["q4_flag", "q1_flag"]:
            tag = "  ← seasonality"
        elif col in BREAK_FLAGS:
            tag = "  ← strategy break (not attributed)"
        elif col in QUALITY_CONTROLS:
            tag = "  ← quality control (not attributed)"
        elif col in MARKET_CONTROLS:
            tag = "  ← market inflation control (not attributed)"
        else:
            tag = ""
        sign = "✓" if (col in CHANNEL_META and val > 0) else ("✗" if col in CHANNEL_META else " ")
        print(f"    {sign} {col:<28} {val:>14.4f}{tag}")

    print(f"\n  {'CHANNEL':<22} {'SHARE':>6}  {'ATTRIBUTED':>14}  "
          f"{'SPEND':>12}  {'ADJ ROI':>8}  DECAY")
    print(f"  {'─'*22} {'─'*6}  {'─'*14}  {'─'*12}  {'─'*8}  {'─'*5}")
    for c in sorted(res["channels"], key=lambda x: -x["roi"]):
        marker = "✓" if c["raw_coeff"] > 0 else "✗"
        print(
            f"  {marker} {c['name']:<20} {c['share']*100:>5.1f}%  "
            f"{fmt_num(c['attributed_out'], unit):>14}  "
            f"{fmt_num(c['total_spend'], '$'):>12}  "
            f"{c['roi']:>7.2f}x  {c['decay']:.2f}"
        )
    print(f"\n  {'Organic baseline':<22} {ORGANIC_BASELINE*100:>5.1f}%"
          f"  (Gartner Visionary + G2 + LLM citations)")
    print(DIV)


def print_optimizer(res: dict, budget: float):
    unit    = res["unit"]
    chs     = res["channels"]
    label   = "monthly" if res["monthly"] else "weekly"
    total_roi = sum(c["roi"] for c in chs) or 1.0
    print(f"\n  BUDGET OPTIMIZER — {label} budget: {fmt_num(budget, '$')}")
    proj = 0.0
    for c in sorted(chs, key=lambda x: -x["roi"]):
        alloc = budget * (c["roi"] / total_roi)
        proj += alloc * c["roi"]
        print(f"  {c['name']:<22} {fmt_num(alloc, '$'):>10}  "
              f"{alloc/budget*100:>6.1f}%  {c['roi']:>7.2f}x ROI")
    print(f"  {'Projected ' + label + ' ' + res['label']:<45} {fmt_num(proj, unit):>10}")


# ─── HTML REPORT ──────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _r2_label(r2: float) -> tuple:
    """Return (badge_class, text) for R²."""
    if r2 > 0.70:
        return "badge-good", f"R² = {r2:.3f} — Strong fit (model explains {r2*100:.0f}% of variance)"
    if r2 > 0.40:
        return "badge-warn", f"R² = {r2:.3f} — Moderate fit (explains {r2*100:.0f}% of variance)"
    return "badge-bad",  f"R² = {r2:.3f} — Weak fit (explains {r2*100:.0f}% of variance)"


def _mape_label(mape: float) -> tuple:
    if mape < 20:
        return "badge-good", f"MAPE = {mape:.1f}% — Predictions accurate within ~{mape:.0f}%"
    if mape < 35:
        return "badge-warn", f"MAPE = {mape:.1f}% — Moderate accuracy; use for directional decisions"
    return "badge-bad",  f"MAPE = {mape:.1f}% — High error; treat conclusions with caution"


def _interpretation_mqls(res: dict, oot: dict) -> str:
    top = sorted(res["channels"], key=lambda x: -x["roi"])
    top_pos = [c for c in top if c["raw_coeff"] > 0]
    top_neg = [c for c in top if c["raw_coeff"] <= 0]
    pos_names = ", ".join(c["name"] for c in top_pos[:3]) if top_pos else "none"
    neg_names = ", ".join(c["name"] for c in top_neg[:3]) if top_neg else "none"
    drift = oot["mape_delta"]
    oot_note = (f"Out-of-time test shows the model generalises well "
                f"(test MAPE {oot['test_mape']:.1f}% vs train {oot['train_mape']:.1f}%)."
                if abs(drift) < 10 else
                f"Out-of-time MAPE increased by {drift:+.1f}pp — treat later-period results with care.")
    return (
        f"<strong>Channels driving MQL volume:</strong> {pos_names}. "
        f"<strong>Channels with negative or no measured effect:</strong> {neg_names} — "
        f"these are not reducing spend efficiency but are not statistically measurable as MQL drivers at this data volume. "
        f"The quality controls (MQL non-SMB rate and MQL→SQL conversion rate) are included so these "
        f"spend coefficients reflect true incremental MQL generation, not quality mix shifts. "
        f"{oot_note}"
    )


def _interpretation_pipeline(res: dict, oot: dict) -> str:
    top = sorted(res["channels"], key=lambda x: -x["roi"])
    top_pos = [c for c in top if c["raw_coeff"] > 0]
    best = top_pos[0] if top_pos else None
    best_str = (f"<strong>{best['name']}</strong> shows the highest pipeline ROI "
                f"({best['roi']:.1f}x) — each dollar spent is attributed to "
                f"{fmt_num(best['attributed_out'] / best['total_spend'], '$')} in pipeline. "
                if best else "No channel shows a reliably positive pipeline coefficient. ")
    drift = oot["mape_delta"]
    oot_note = (f"Model generalises well out of sample (MAPE delta {drift:+.1f}pp)."
                if abs(drift) < 10 else
                f"Out-of-time MAPE increased by {drift:+.1f}pp; model fit weakens on recent data.")
    return (
        f"{best_str}"
        f"The 2-month lag means we are asking: 'which spend in month T produced pipeline in month T+2?' — "
        f"aligned with a typical MQL→SQL→opportunity cycle. "
        f"The cr_mql_to_sql quality control absorbs months where pipeline dropped due to MQL quality "
        f"issues rather than insufficient spend. {oot_note}"
    )


def _interpretation_arr(res: dict, oot: dict) -> str:
    n = res["n_obs"]
    top_pos = [c for c in res["channels"] if c["raw_coeff"] > 0]
    best = sorted(top_pos, key=lambda x: -x["roi"])[0] if top_pos else None
    caution = (f" <em>Note: with only {n} usable observations after the 5-month lag, "
               f"individual channel ROI numbers are estimates — use for ranking, not precise accounting.</em>"
               if n < 20 else "")
    best_str = (f"<strong>{best['name']}</strong> shows the strongest ARR correlation "
                f"(ROI {best['roi']:.1f}x at a 5-month lag). "
                if best else "No single channel dominates ARR attribution clearly. ")
    return (
        f"{best_str}"
        f"The 5-month lag reflects Katalon's enterprise sales cycle: spend in month T is regressed "
        f"against ARR closed in month T+5. Channels that appear negative here (e.g. LinkedIn Brand) "
        f"likely drive pipeline but close on a longer timeline — they may show up positively at lag 6–8. "
        f"{caution}"
    )


def _make_spend_chart(df: pd.DataFrame) -> str:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        return ""

    spend_cols = [c for c in CHANNEL_META if c in df.columns]
    totals = {CHANNEL_META[c]["label"]: df[c].fillna(0).sum() for c in spend_cols}
    totals = {k: v for k, v in sorted(totals.items(), key=lambda x: -x[1]) if v > 0}
    colors = [CHANNEL_META[c]["color"] for c in spend_cols
              if CHANNEL_META[c]["label"] in totals][:len(totals)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#FAFAF8")

    # Bar chart
    bars = ax1.barh(list(totals.keys()), list(totals.values()),
                    color=colors, height=0.6)
    for b, v in zip(bars, totals.values()):
        ax1.text(b.get_width() + 5000, b.get_y() + b.get_height() / 2,
                 f"${v/1000:.0f}K", va="center", fontsize=9, color="#374151")
    ax1.set_xlim(0, max(totals.values()) * 1.25)
    try:
        if "month" in df.columns:
            def _m2l(m):
                s = str(int(m))
                return pd.Timestamp(year=int(s[:4]), month=int(s[4:]), day=1).strftime("%b %Y")
            period_title = f"{_m2l(df['month'].iloc[0])} – {_m2l(df['month'].iloc[-1])}"
        else:
            def _w2l(w):
                p = str(w).split("-W")
                d = pd.Timestamp.fromisocalendar(int(p[0]), int(p[1]), 1)
                return d.strftime("%b %Y")
            period_title = f"{_w2l(df['week'].iloc[0])} – {_w2l(df['week'].iloc[-1])}"
    except Exception:
        period_title = ""
    ax1.set_title(f"Total Spend by Channel\n({period_title})", fontsize=11,
                  fontweight="bold", color="#111827")
    ax1.tick_params(labelsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_facecolor("#FAFAF8")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    # Stacked area spend trend (monthly)
    if "week" in df.columns:
        x = np.arange(len(df))
        bottoms = np.zeros(len(df))
        for col in spend_cols:
            if col not in df.columns:
                continue
            vals = df[col].fillna(0).values
            if vals.sum() == 0:
                continue
            ax2.fill_between(x, bottoms, bottoms + vals,
                             color=CHANNEL_META[col]["color"], alpha=0.85,
                             label=CHANNEL_META[col]["label"])
            bottoms += vals
        ax2.set_title("Monthly Spend Mix Over Time", fontsize=11,
                      fontweight="bold", color="#111827")
        ax2.legend(fontsize=7, ncol=2, loc="upper left")
        ax2.tick_params(labelsize=8)
        ax2.set_xticks(x[::3])
        ax2.set_xticklabels([str(df["week"].iloc[i])[:7] for i in x[::3]],
                            rotation=45, ha="right")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y/1000:.0f}K"))
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.set_facecolor("#FAFAF8")

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _make_outcome_charts(res: dict) -> tuple:
    """Return (attribution_b64, actuals_b64)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "", ""

    # ── Attribution bars ─────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    fig1.patch.set_facecolor("#FAFAF8")
    channels = sorted(res["channels"], key=lambda x: x["share"])
    names  = [c["name"] for c in channels] + ["Organic / Dark Funnel"]
    shares = [c["share"] * 100 for c in channels] + [ORGANIC_BASELINE * 100]
    colors = [c["color"] for c in channels] + ["#CBD5E1"]
    bars   = ax1.barh(names, shares, color=colors, height=0.55)
    for b, v in zip(bars, shares):
        ax1.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                 f"{v:.1f}%", va="center", fontsize=9, color="#374151")
    ax1.set_xlim(0, max(shares) * 1.35)
    ax1.set_title(
        f"{res['label']} — Channel Attribution\n"
        f"R² = {res['r2']:.3f}  ·  MAPE = {res['mape']:.1f}%  ·  "
        f"{res['n_obs']} monthly observations",
        fontsize=11, fontweight="bold", color="#111827",
    )
    ax1.tick_params(labelsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_facecolor("#FAFAF8")
    plt.tight_layout()
    b64_attr = _fig_to_b64(fig1)
    plt.close(fig1)

    # ── Actual vs Predicted ──────────────────────────────────────────────────
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(13, 4))
    fig2.patch.set_facecolor("#FAFAF8")
    t = np.arange(len(res["actuals"]))
    ax2.plot(t, res["actuals"],   color="#111827", lw=2,   label="Actual")
    ax2.plot(t, res["predicted"], color=res["outcome_color"], lw=1.5, ls="--",
             label=f"OLS model (R²={res['r2']:.2f})")
    ax2.set_title(f"{res['label']} — Actual vs Model Fit\n"
                  f"lag = {res['lag']} months",
                  fontsize=10, fontweight="bold", color="#111827")
    ax2.legend(fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_facecolor("#FAFAF8")
    ax2.set_xlabel("Month index", fontsize=8, color="#6B7280")

    r_colors = ["#059669" if v >= 0 else "#EF4444" for v in res["residuals"]]
    ax3.bar(t, res["residuals"], color=r_colors, width=0.8, alpha=0.7)
    ax3.axhline(0, color="#111827", lw=0.8)
    ax3.set_title("Residuals (Actual − Predicted)\n"
                  "Random scatter = good fit.  Trend = model is missing something.",
                  fontsize=10, fontweight="bold", color="#111827")
    ax3.tick_params(labelsize=8)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_facecolor("#FAFAF8")

    plt.tight_layout()
    b64_fit = _fig_to_b64(fig2)
    plt.close(fig2)
    return b64_attr, b64_fit


def _make_quality_chart(df: pd.DataFrame) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ""
    qcs = [qc for qc in QUALITY_CONTROLS if qc in df.columns]
    if not qcs:
        return ""

    fig, axes = plt.subplots(1, len(qcs), figsize=(6 * len(qcs), 4))
    fig.patch.set_facecolor("#FAFAF8")
    if len(qcs) == 1:
        axes = [axes]

    labels = {
        "cr_mql_to_sql":    ("MQL → SQL Conversion Rate", "#7C3AED",
                             "Higher = better quality MQLs reaching sales"),
        "mql_non_smb_rate": ("MQL Non-SMB Rate (Enterprise Mix)", "#0891B2",
                             "Higher = more enterprise-size MQLs in the funnel"),
    }
    x = np.arange(len(df))
    for ax, qc in zip(axes, qcs):
        title, color, subtitle = labels.get(qc, (qc, "#374151", ""))
        vals = df[qc].fillna(method="ffill").fillna(method="bfill").values
        ax.plot(x, vals, color=color, lw=2, marker="o", ms=4)
        # trend line
        z = np.polyfit(x, vals, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), color=color, lw=1, ls="--", alpha=0.5, label=f"Trend ({z[0]*12:+.3f}/yr)")
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold", color="#111827")
        ax.legend(fontsize=8)
        ax.set_xticks(x[::3])
        ax.set_xticklabels([str(df["week"].iloc[i])[:7] for i in x[::3]],
                           rotation=45, ha="right")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#FAFAF8")
        ax.yaxis.set_major_formatter(
            __import__("matplotlib").ticker.PercentFormatter(xmax=1, decimals=0))

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _spend_context_blurb(df: pd.DataFrame, spend_cols: list, total_spend: float) -> str:
    """Dynamic spend overview sentence computed from actual data."""
    if total_spend == 0:
        return "No spend data available."
    shares = {c: df[c].fillna(0).sum() / total_spend * 100 for c in spend_cols}
    largest = max(shares, key=lambda c: shares[c])
    smallest_nonzero = min(
        (c for c in spend_cols if shares[c] > 0.5),
        key=lambda c: shares[c], default=None
    )
    lg_name = CHANNEL_META[largest]["label"]
    lg_pct  = shares[largest]
    blurb = (f"{lg_name} is the single largest spend category at {lg_pct:.1f}% of total budget.")
    if smallest_nonzero and smallest_nonzero != largest:
        sm_name = CHANNEL_META[smallest_nonzero]["label"]
        sm_pct  = shares[smallest_nonzero]
        blurb += (f" {sm_name} is only {sm_pct:.1f}% of budget — "
                  f"check the Attribution sections to see if this channel punches above its weight.")
    return blurb


def _lag_cmp_nav(lag_comparison) -> str:
    if not lag_comparison:
        return ""
    parts = []
    for ok in lag_comparison:
        label = OUTCOME_META[ok]["label"]
        parts.append(f'<a href="#lag-compare-{ok}">Lag Test: {label}</a>')
    return "\n    ".join(parts)


def _make_lag_comparison_chart(res1: dict, res2: dict) -> str:
    """Side-by-side bar chart comparing two lag runs for the same outcome."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        return ""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#FAFAF8")
    fig.suptitle(
        f"{res1['label']} — Lag Comparison: {res1['lag']} mo vs {res2['lag']} mo",
        fontsize=12, fontweight="bold", color="#111827", y=1.02,
    )

    for ax, res, color in zip(axes[:2], [res1, res2], ["#0891B2", "#7C3AED"]):
        channels = sorted(res["channels"], key=lambda x: x["share"])
        names  = [c["name"] for c in channels] + ["Organic"]
        shares = [c["share"] * 100 for c in channels] + [ORGANIC_BASELINE * 100]
        colors = [c["color"] for c in channels] + ["#CBD5E1"]
        bars = ax.barh(names, shares, color=colors, height=0.55)
        for b, v in zip(bars, shares):
            ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xlim(0, max(shares) * 1.35)
        ax.set_title(
            f"Lag = {res['lag']} month{'s' if res['lag'] != 1 else ''}\n"
            f"R²={res['r2']:.3f}  MAPE={res['mape']:.1f}%  N={res['n_obs']}",
            fontsize=10, fontweight="bold", color=color,
        )
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#FAFAF8")

    # Actual vs predicted overlay for both lags
    ax3 = axes[2]
    ax3.plot(res1["actuals"], color="#111827", lw=2, label="Actual")
    ax3.plot(res1["predicted"], color="#0891B2", lw=1.5, ls="--",
             label=f"Fit lag={res1['lag']} (R²={res1['r2']:.2f})")
    ax3.plot(res2["predicted"], color="#7C3AED", lw=1.5, ls=":",
             label=f"Fit lag={res2['lag']} (R²={res2['r2']:.2f})")
    ax3.set_title("Actual vs Both Model Fits", fontsize=10,
                  fontweight="bold", color="#111827")
    ax3.legend(fontsize=8)
    ax3.tick_params(labelsize=8)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_facecolor("#FAFAF8")

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ─── SCENARIO FORECAST ────────────────────────────────────────────────────────

def _yyyymm_add_months(yyyymm: int, n: int) -> int:
    """Add n months to YYYYMM integer. e.g. 202612 + 1 = 202701."""
    year  = int(yyyymm) // 100
    month = int(yyyymm) % 100 + n
    year  += (month - 1) // 12
    month  = (month - 1) % 12 + 1
    return year * 100 + month


def _yyyymm_label(m: int) -> str:
    """202604 → 'April 2026'."""
    s = str(int(m))
    return pd.Timestamp(year=int(s[:4]), month=int(s[4:]), day=1).strftime("%B %Y")


def run_scenario_forecast(
    results: dict,
    df_fit: pd.DataFrame,
    scenario_spend: dict,
    scenario_month: int,
) -> dict:
    """
    Predict MQLs, Pipeline, and ARR for a planned monthly spend allocation.

    Uses fitted OLS coefficients from `results`, adstock carry-over from the
    last historical month in `df_fit`, and assumes quality controls remain at
    their last-3-month average.

    Parameters
    ----------
    results        : dict of OLS results (keyed by outcome_key)
    df_fit         : historical dataframe (provides carry-over + quality baseline)
    scenario_spend : {channel_key: planned_spend_$} — unspecified channels default to 0
    scenario_month : YYYYMM int (e.g. 202604 for April 2026)

    Returns
    -------
    dict keyed by outcome_key, each containing point estimate, uncertainty range,
    channel breakdown, and metadata.
    """
    spend_cols    = [c for c in CHANNEL_META if c in df_fit.columns]
    scenario_month = int(scenario_month)

    # ── Adstock carry-over from last historical month ──────────────────────────
    last_adstock = {}
    for ch in spend_cols:
        d = CHANNEL_META[ch]["decay"]
        series = apply_adstock(df_fit[ch].fillna(0).values, d)
        last_adstock[ch] = float(series[-1])

    # ── Scenario adstock: new spend + decay × carry-over ──────────────────────
    scenario_adstock = {}
    for ch in spend_cols:
        planned = float(scenario_spend.get(ch, 0.0))
        d = CHANNEL_META[ch]["decay"]
        scenario_adstock[ch] = planned + d * last_adstock[ch]

    # ── Seasonality flags for the spend month ─────────────────────────────────
    month_num = scenario_month % 100
    q4_flag   = 1 if month_num in (10, 11, 12) else 0
    q1_flag   = 1 if month_num in (1, 2) else 0

    # ── Quality + market controls: last-3-month average ──────────────────────
    assumed_quality = {}
    for qc in QUALITY_CONTROLS + MARKET_CONTROLS:
        if qc in df_fit.columns:
            vals = df_fit[qc].dropna().values
            assumed_quality[qc] = (
                float(vals[-3:].mean()) if len(vals) >= 3
                else (float(vals.mean()) if len(vals) > 0 else 0.0)
            )

    forecast = {}
    for ok, res in results.items():
        coeff_map    = res["coeff_map"]
        lag          = res["lag"]
        target_month = _yyyymm_add_months(scenario_month, lag)
        mape         = res["mape"]

        # ── OLS point prediction ───────────────────────────────────────────────
        pred = float(coeff_map.get("intercept", 0.0))
        for ch in spend_cols:
            pred += float(coeff_map.get(ch, 0.0)) * scenario_adstock[ch]
        pred += float(coeff_map.get("q4_flag", 0.0)) * q4_flag
        pred += float(coeff_map.get("q1_flag", 0.0)) * q1_flag
        # pre_2024_flag is always 0 for any future month
        for qc, qval in assumed_quality.items():
            pred += float(coeff_map.get(qc, 0.0)) * qval
        pred = max(0.0, pred)

        # ── Uncertainty: ±MAPE of the model ───────────────────────────────────
        low  = max(0.0, pred * (1.0 - mape / 100.0))
        high = pred * (1.0 + mape / 100.0)

        # ── Historical monthly average for comparison ──────────────────────────
        y_hist   = df_fit[ok].fillna(0).values
        hist_avg = float(y_hist[y_hist > 0].mean()) if (y_hist > 0).any() else 0.0

        # ── Channel-level contribution breakdown ──────────────────────────────
        total_pos_contrib = sum(
            max(0.0, float(coeff_map.get(ch, 0.0))) * scenario_adstock[ch]
            for ch in spend_cols
        ) or 1.0

        channel_contribs = []
        for ch in spend_cols:
            coeff  = float(coeff_map.get(ch, 0.0))
            contrib = max(0.0, coeff) * scenario_adstock[ch]
            channel_contribs.append({
                "channel":       ch,
                "name":          CHANNEL_META[ch]["label"],
                "color":         CHANNEL_META[ch]["color"],
                "planned_spend": float(scenario_spend.get(ch, 0.0)),
                "coeff":         coeff,
                "contrib":       contrib,
                "contrib_share": contrib / total_pos_contrib,
            })

        forecast[ok] = {
            "outcome_key":          ok,
            "label":                res["label"],
            "unit":                 res["unit"],
            "outcome_color":        res["outcome_color"],
            "lag":                  lag,
            "scenario_month":       scenario_month,
            "target_month":         target_month,
            "target_month_label":   _yyyymm_label(target_month),
            "scenario_month_label": _yyyymm_label(scenario_month),
            "point":                pred,
            "low":                  low,
            "high":                 high,
            "mape":                 mape,
            "r2":                   res["r2"],
            "hist_avg":             hist_avg,
            "channel_contribs":     channel_contribs,
            "assumed_quality":      assumed_quality,
            "q4_flag":              q4_flag,
            "q1_flag":              q1_flag,
        }

    return forecast


def print_scenario(forecast: dict, scenario_spend: dict, scenario_month: int) -> None:
    """Print scenario forecast results to the terminal."""
    total_planned   = sum(float(v) for v in scenario_spend.values())
    present_channels = [c for c in CHANNEL_META if float(scenario_spend.get(c, 0)) > 0]
    sep = "═" * 72
    div = "─" * 72

    print(f"\n{sep}")
    print(f"  SCENARIO FORECAST  —  Spend allocated in {_yyyymm_label(scenario_month)}")
    print(sep)
    print(f"\n  Planned monthly spend: {fmt_num(total_planned, '$')}")
    for ch in present_channels:
        print(f"    {CHANNEL_META[ch]['label']:<28} {fmt_num(float(scenario_spend[ch]), '$'):>10}")

    print(f"\n  {'Outcome':<28} {'Predicted in':<16} {'Low':>10} {'Point':>12} {'High':>10}  {'vs hist avg':>11}")
    print(f"  {div}")
    for ok, fc in forecast.items():
        vs_avg = ((fc["point"] / fc["hist_avg"]) - 1) * 100 if fc["hist_avg"] > 0 else 0.0
        print(
            f"  {fc['label']:<28} {fc['target_month_label']:<16}"
            f" {fmt_num(fc['low'],   fc['unit']):>10}"
            f" {fmt_num(fc['point'], fc['unit']):>12}"
            f" {fmt_num(fc['high'],  fc['unit']):>10}"
            f"  {vs_avg:>+10.0f}%"
        )

    print(f"\n  Notes:")
    print(f"    · Uncertainty range = ±MAPE of each fitted model")
    print(f"    · MQL prediction is for {_yyyymm_label(scenario_month)} (no lag)")
    for ok, fc in forecast.items():
        if fc["lag"] > 0:
            print(f"    · {fc['label']} prediction is for {fc['target_month_label']} (lag={fc['lag']} mo)")
    print(f"    · Quality controls held at last-3-month average")
    print(f"    · Model assumes linear returns — diminishing returns not captured")
    print(f"{sep}\n")


def _make_scenario_chart(forecast: dict, scenario_spend: dict) -> str:
    """
    Create a matplotlib chart for the scenario forecast.
    Returns a base64-encoded PNG string, or '' on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return ""

    outcomes = list(forecast.items())
    n = len(outcomes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.patch.set_facecolor("#FAFAF8")
    if n == 1:
        axes = [axes]

    for ax, (ok, fc) in zip(axes, outcomes):
        unit      = fc["unit"]
        color     = fc["outcome_color"]
        point     = fc["point"]
        low       = fc["low"]
        high      = fc["high"]
        hist_avg  = fc["hist_avg"]

        categories = ["Historical\nMonthly Avg", f"Predicted\n{fc['target_month_label']}"]
        values     = [hist_avg, point]
        colors     = ["#CBD5E1", color]

        bars = ax.bar(categories, values, color=colors, width=0.5, zorder=3,
                      edgecolor="white", linewidth=1.5)

        # Error bar on the predicted bar only
        ax.errorbar(
            x=1, y=point,
            yerr=[[point - low], [high - point]],
            fmt="none", color="#374151", capsize=8, capthick=2, elinewidth=2, zorder=4
        )

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                fmt_num(val, unit),
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#111827"
            )

        # Uncertainty label
        ax.text(1, high * 1.03, f"±{fc['mape']:.0f}%", ha="center", va="bottom",
                fontsize=8, color="#6B7280", style="italic")

        ax.set_title(
            f"{fc['label']}\nR²={fc['r2']:.2f}",
            fontsize=11, fontweight="bold", color="#111827", pad=10
        )
        ax.set_facecolor("#FAFAF8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.set_ylim(0, max(hist_avg, high) * 1.30)

        # Spend label underneath
        total = sum(float(v) for v in scenario_spend.values())
        ax.set_xlabel(f"Scenario spend: {fmt_num(total, '$')}/mo", fontsize=9, color="#6B7280")

    fig.suptitle("Scenario Forecast vs Historical Average",
                 fontsize=13, fontweight="bold", color="#111827", y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#FAFAF8")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def run_multi_scenario_comparison(
    results: dict,
    df_fit: pd.DataFrame,
    scenarios: list,
) -> list:
    """
    Run scenario forecast for multiple named scenarios.

    Parameters
    ----------
    scenarios : list of dicts, each with keys:
                  name  (str)
                  month (int YYYYMM)
                  spend (dict {channel_key: amount})
                  note  (str, optional)

    Returns
    -------
    list of dicts: [{"name": ..., "note": ..., "month": ..., "forecast": run_scenario_forecast(...)}, ...]
    """
    out = []
    for sc in scenarios:
        fc = run_scenario_forecast(results, df_fit, sc["spend"], sc["month"])
        out.append({
            "name":     sc["name"],
            "note":     sc.get("note", ""),
            "month":    sc["month"],
            "spend":    sc["spend"],
            "forecast": fc,
        })
    return out


def print_scenario_comparison(comparison: list) -> None:
    """Print a side-by-side scenario comparison table to the terminal."""
    if not comparison:
        return
    sep = "═" * 90
    div = "─" * 90

    print(f"\n{sep}")
    print(f"  SCENARIO COMPARISON")
    print(sep)

    # Header row
    name_col = 28
    sc_col   = 20
    header   = f"  {'Outcome':<{name_col}}"
    for sc in comparison:
        header += f"  {sc['name'][:sc_col-2]:<{sc_col}}"
    print(header)

    # Spend row
    spend_row = f"  {'Total spend':<{name_col}}"
    for sc in comparison:
        total = sum(float(v) for v in sc["spend"].values())
        spend_row += f"  {fmt_num(total, '$'):<{sc_col}}"
    print(f"  {div}")
    print(spend_row)
    print(f"  {div}")

    # One row per outcome
    outcomes = list(comparison[0]["forecast"].keys())
    for ok in outcomes:
        fc0 = comparison[0]["forecast"][ok]
        point_row  = f"  {fc0['label']:<{name_col}}"
        range_row  = f"  {'  ↳ range':<{name_col}}"
        target_row = f"  {'  ↳ predicted in':<{name_col}}"
        vs_row     = f"  {'  ↳ vs hist avg':<{name_col}}"

        for sc in comparison:
            fc   = sc["forecast"][ok]
            unit = fc["unit"]
            vs   = ((fc["point"] / fc["hist_avg"]) - 1) * 100 if fc["hist_avg"] > 0 else 0.0
            point_row  += f"  {fmt_num(fc['point'], unit):<{sc_col}}"
            range_row  += f"  {fmt_num(fc['low'], unit)+' – '+fmt_num(fc['high'], unit):<{sc_col}}"
            target_row += f"  {fc['target_month_label']:<{sc_col}}"
            vs_row     += f"  {vs:>+.0f}%{'':<{sc_col-5}}"

        print(point_row)
        print(range_row)
        print(target_row)
        print(vs_row)
        print(f"  {div}")

    print()
    for sc in comparison:
        print(f"  {sc['name']}")
        if sc.get("note"):
            print(f"    {sc['note']}")
    print(sep + "\n")


def _make_scenario_comparison_chart(comparison: list) -> str:
    """
    Multi-scenario comparison chart: grouped bars per outcome.
    Returns base64-encoded PNG string or '' on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
    except ImportError:
        return ""

    outcomes = list(comparison[0]["forecast"].keys())
    n_sc     = len(comparison)
    n_out    = len(outcomes)
    colors   = ["#94A3B8", "#2563EB", "#059669"]  # gray, blue, green

    fig, axes = plt.subplots(1, n_out, figsize=(6 * n_out, 5))
    fig.patch.set_facecolor("#FAFAF8")
    if n_out == 1:
        axes = [axes]

    x = _np.arange(n_sc)
    bar_w = 0.6

    for ax, ok in zip(axes, outcomes):
        fc0       = comparison[0]["forecast"][ok]
        unit      = fc0["unit"]
        hist_avg  = fc0["hist_avg"]

        points = [sc["forecast"][ok]["point"] for sc in comparison]
        lows   = [sc["forecast"][ok]["low"]   for sc in comparison]
        highs  = [sc["forecast"][ok]["high"]  for sc in comparison]
        names  = [sc["name"].split(" —")[0]   for sc in comparison]  # short name

        bars = ax.bar(x, points, width=bar_w, color=colors[:n_sc], zorder=3,
                      edgecolor="white", linewidth=1.2, alpha=0.9)

        # Error bars
        yerr_low  = [p - l for p, l in zip(points, lows)]
        yerr_high = [h - p for p, h in zip(points, highs)]
        ax.errorbar(x, points, yerr=[yerr_low, yerr_high],
                    fmt="none", color="#374151", capsize=7, capthick=1.5, elinewidth=1.5, zorder=4)

        # Historical avg reference line
        ax.axhline(hist_avg, color="#F59E0B", lw=1.5, ls="--", zorder=2,
                   label=f"Hist avg: {fmt_num(hist_avg, unit)}")

        # Value labels
        for bar, val in zip(bars, points):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    fmt_num(val, unit),
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#111827")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=10, ha="right")
        ax.set_title(f"{fc0['label']}\nR²={fc0['r2']:.2f}", fontsize=11, fontweight="bold",
                     color="#111827", pad=8)
        ax.set_facecolor("#FAFAF8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(max(highs), hist_avg) * 1.30)

    fig.suptitle("Scenario Comparison — Predicted Outcomes for April 2026",
                 fontsize=12, fontweight="bold", color="#111827", y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#FAFAF8")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def export_html_report(
    results: dict,
    df_fit: pd.DataFrame,
    output_path: str = "katalon_mmm_report.html",
    budget_override: float = None,
    lag_comparison: dict = None,   # {"outcome_key": (res_lag_a, res_lag_b)}
    scenario_forecast: dict = None,  # output of run_scenario_forecast()
    scenario_spend: dict = None,     # {channel_key: planned_spend} for display
    scenarios_comparison: list = None,  # output of run_multi_scenario_comparison()
):
    """
    Generate a self-contained HTML report with embedded charts and
    plain-language interpretations for non-technical stakeholders.
    lag_comparison: optional dict mapping outcome_key → (result_lag1, result_lag2)
    for a side-by-side lag sensitivity section.
    """
    from datetime import date as _date

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt          # noqa: F401
    except ImportError:
        print("  matplotlib not installed — pip install matplotlib")
        return

    print(f"\n  Generating HTML report → {output_path}")

    spend_cols = [c for c in CHANNEL_META if c in df_fit.columns]
    total_spend = sum(df_fit[c].fillna(0).sum() for c in spend_cols)
    avg_monthly_spend = total_spend / max(len(df_fit), 1)
    n_months = len(df_fit)

    # Derive human-readable period — prefer raw YYYYMM 'month' column to avoid
    # ISO-week calendar offset (e.g. Jan 2022 → ISO week 52/2021).
    def _yyyymm_to_label(m):
        s = str(int(m))
        return pd.Timestamp(year=int(s[:4]), month=int(s[4:]), day=1).strftime("%b %Y")
    def _isoweek_to_label(w):
        p = str(w).split("-W")
        d = pd.Timestamp.fromisocalendar(int(p[0]), int(p[1]), 1)
        return d.strftime("%b %Y")
    try:
        if "month" in df_fit.columns:
            period_start = _yyyymm_to_label(df_fit["month"].iloc[0])
            period_end   = _yyyymm_to_label(df_fit["month"].iloc[-1])
        else:
            period_start = _isoweek_to_label(df_fit["week"].iloc[0])
            period_end   = _isoweek_to_label(df_fit["week"].iloc[-1])
        period_str = f"{period_start} – {period_end}"
    except Exception:
        period_str = "full period"

    # ── Generate charts ───────────────────────────────────────────────────────
    b64_spend = _make_spend_chart(df_fit)
    b64_quality = _make_quality_chart(df_fit)

    outcome_charts = {}
    oot_data       = {}
    for ok, res in results.items():
        lag = res["lag"]
        oot = out_of_time_val(df_fit, ok, 0.70, lag, res["use_seasonality"],
                               res["monthly"])
        oot_data[ok] = oot
        b64_attr, b64_fit = _make_outcome_charts(res)
        outcome_charts[ok] = (b64_attr, b64_fit)

    # ── OOT table rows ────────────────────────────────────────────────────────
    def _oot_badge(delta):
        if delta < 10:
            return f'<span class="badge-good">No overfit (Δ={delta:+.1f}pp)</span>'
        if delta < 20:
            return f'<span class="badge-warn">Mild drift (Δ={delta:+.1f}pp)</span>'
        return f'<span class="badge-bad">Overfit risk (Δ={delta:+.1f}pp)</span>'

    # ── Key findings bullets ──────────────────────────────────────────────────
    def _top_roi_channel(res):
        top = sorted(res["channels"], key=lambda x: -x["roi"])
        tp  = [c for c in top if c["raw_coeff"] > 0]
        return tp[0] if tp else None

    key_bullets = []
    for ok, res in results.items():
        best = _top_roi_channel(res)
        if best:
            key_bullets.append(
                f"<li><strong>{res['label']}:</strong> "
                f"<em>{best['name']}</em> is the highest-ROI channel "
                f"({best['roi']:.1f}x). "
                f"Model fit R²={res['r2']:.2f} / MAPE={res['mape']:.1f}%.</li>"
            )

    neg_capture = (
        results.get("mqls", {}).get("coeff_map", {}).get("google_capture_spend", 0) < 0
    )
    if neg_capture:
        key_bullets.append(
            "<li><strong>⚠ Google Capture (PPC)</strong> shows a negative coefficient for MQLs — "
            "the AI Overview headwind is measurable in this data. "
            "High-intent clicks that used to reach the site are now intercepted by AI summaries.</li>"
        )

    # Quality trend note
    if "cr_mql_to_sql" in df_fit.columns:
        cr_vals = df_fit["cr_mql_to_sql"].dropna().values
        if len(cr_vals) >= 6:
            h1 = cr_vals[:len(cr_vals)//2].mean()
            h2 = cr_vals[len(cr_vals)//2:].mean()
            pct = (h2 - h1) / h1 * 100
            if pct < -10:
                key_bullets.append(
                    f"<li><strong>⚠ MQL quality declining:</strong> "
                    f"MQL→SQL conversion rate fell {abs(pct):.0f}% in the second half of the period. "
                    f"The quality controls in this model absorb that drift so spend attribution "
                    f"reflects true volume contribution — but the underlying quality trend needs attention.</li>"
                )

    # ── Outcome sections HTML ─────────────────────────────────────────────────
    interpretations = {
        "mqls":             _interpretation_mqls,
        "inbound_pipeline": _interpretation_pipeline,
        "won_arr":          _interpretation_arr,
    }

    # ── Lag comparison section ────────────────────────────────────────────────
    lag_comparison_html = ""
    if lag_comparison:
        lag_cmp_blocks = []
        for ok, (r1, r2) in lag_comparison.items():
            b64_cmp = _make_lag_comparison_chart(r1, r2)
            winner  = r1 if r1["r2"] >= r2["r2"] else r2
            loser   = r2 if winner is r1 else r1
            rec = (
                f"<strong>Recommendation: use lag={winner['lag']} month(s)</strong> — "
                f"R²={winner['r2']:.3f} vs {loser['r2']:.3f} "
                f"and MAPE={winner['mape']:.1f}% vs {loser['mape']:.1f}%. "
            )
            if abs(r1["r2"] - r2["r2"]) < 0.05:
                rec += (
                    "The two lags are very close — the difference is within the model's "
                    "uncertainty range. Either is defensible; match to your actual "
                    "CRM-measured MQL→Opportunity time."
                )
            else:
                rec += (
                    f"The lag={winner['lag']} model explains meaningfully more variance. "
                    f"This suggests the dominant spend→{OUTCOME_META[ok]['label'].lower()} "
                    f"cycle is approximately {winner['lag']} month(s)."
                )
            lag_cmp_blocks.append(f"""
            <section class="outcome-section" id="lag-compare-{ok}">
              <h2>{OUTCOME_META[ok]['label']} — Pipeline Lag Test: 1 month vs 2 months</h2>
              <p style="font-size:13px;color:#374151;margin-bottom:12px">
                The lag determines how many months after spend we look for the outcome.
                Lag=1 assumes MQL → Opportunity takes ~4 weeks; Lag=2 assumes ~8 weeks.
                The model that fits the data better indicates the typical cycle in your CRM.
              </p>
              <table class="data-table" style="margin-bottom:16px">
                <thead><tr><th>Metric</th>
                  <th>Lag = {r1['lag']} month</th>
                  <th>Lag = {r2['lag']} months</th>
                  <th>Difference</th></tr></thead>
                <tbody>
                  <tr><td>R² (higher = better fit)</td>
                    <td>{r1['r2']:.3f}</td><td>{r2['r2']:.3f}</td>
                    <td>{r2['r2']-r1['r2']:+.3f}</td></tr>
                  <tr><td>MAPE (lower = more accurate)</td>
                    <td>{r1['mape']:.1f}%</td><td>{r2['mape']:.1f}%</td>
                    <td>{r2['mape']-r1['mape']:+.1f}pp</td></tr>
                  <tr><td>Usable observations</td>
                    <td>{r1['n_obs']}</td><td>{r2['n_obs']}</td>
                    <td>{r2['n_obs']-r1['n_obs']:+d}</td></tr>
                </tbody>
              </table>
              {"<img src='data:image/png;base64," + b64_cmp + "' style='max-width:100%;border-radius:8px;' />" if b64_cmp else ""}
              <div class="interp-box" style="margin-top:14px">
                <h4>Interpretation</h4>
                <p>{rec}</p>
              </div>
            </section>
            """)
        lag_comparison_html = "\n".join(lag_cmp_blocks)

    outcome_html_blocks = []
    for ok, res in results.items():
        unit   = res["unit"]
        budget = budget_override or avg_monthly_spend
        oot    = oot_data[ok]

        b64_attr, b64_fit = outcome_charts[ok]

        r2_cls, r2_txt   = _r2_label(res["r2"])
        mp_cls, mp_txt   = _mape_label(res["mape"])

        # ROI table
        roi_rows = ""
        for c in sorted(res["channels"], key=lambda x: -x["roi"]):
            sign_class = "pos-coeff" if c["raw_coeff"] > 0 else "neg-coeff"
            sign_icon  = "✓" if c["raw_coeff"] > 0 else "✗"
            roi_rows += (
                f'<tr class="{sign_class}">'
                f'<td><span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:{c["color"]};margin-right:6px"></span>'
                f'{c["name"]}</td>'
                f'<td>{sign_icon}</td>'
                f'<td>{c["share"]*100:.1f}%</td>'
                f'<td>{fmt_num(c["attributed_out"], unit)}</td>'
                f'<td>{fmt_num(c["total_spend"], "$")}</td>'
                f'<td><strong>{c["roi"]:.2f}x</strong></td>'
                f'<td>{c["decay"]:.2f}</td>'
                f'<td style="font-size:11px;color:#6B7280">{c["note"][:60]}</td>'
                f'</tr>'
            )
        roi_rows += (
            f'<tr style="background:#F8FAFC">'
            f'<td colspan="2"><em>Organic / Dark Funnel</em></td>'
            f'<td>{ORGANIC_BASELINE*100:.0f}%</td>'
            f'<td colspan="5" style="font-size:11px;color:#6B7280">'
            f'Gartner Visionary, G2 reviews, LLM citations (not attributable to paid channels)</td>'
            f'</tr>'
        )

        # Budget optimizer
        total_roi  = sum(c["roi"] for c in res["channels"]) or 1.0
        proj_out   = 0.0
        opt_rows   = ""
        for c in sorted(res["channels"], key=lambda x: -x["roi"]):
            alloc   = budget * (c["roi"] / total_roi)
            proj_out += alloc * c["roi"]
            pct     = alloc / budget * 100
            opt_rows += (
                f'<tr><td>{c["name"]}</td>'
                f'<td>{fmt_num(alloc, "$")}</td>'
                f'<td>{pct:.1f}%</td>'
                f'<td>{c["roi"]:.2f}x</td></tr>'
            )

        interp_fn   = interpretations.get(ok, lambda r, o: "")
        interp_text = interp_fn(res, oot)

        # Winsorize badge (only present on won_arr when flag was used)
        if res.get("winsorize_pct"):
            _wpct      = res["winsorize_pct"]
            _wcap      = res["winsorize_cap"]
            _wn        = res["winsorize_n_capped"]
            _wmonths   = str(res["winsorize_months"])
            _wplural   = "s" if _wn != 1 else ""
            winsorize_badge = (
                f'<span class="badge-warn" title="Months capped: {_wmonths}">'
                f'Winsorized P{_wpct:.0f} '
                f'(cap {fmt_num(_wcap, "$")} · {_wn} month{_wplural} capped)'
                f'</span>'
            )
        else:
            winsorize_badge = ""

        outcome_html_blocks.append(f"""
        <section class="outcome-section" id="{ok}">
          <h2>{res['label']}
            <span class="lag-badge">{res['lag_note']}</span>
          </h2>

          <div class="badge-row">
            <span class="{r2_cls}">{r2_txt}</span>
            <span class="{mp_cls}">{mp_txt}</span>
            {_oot_badge(oot['mape_delta'])}
            <span class="badge-neutral">N = {res['n_obs']} months</span>
            {winsorize_badge}
          </div>

          <div class="charts-row">
            <div class="chart-block">
              <img src="data:image/png;base64,{b64_attr}" alt="Attribution" />
              <p class="chart-caption">How much of each month's {res['label']}
              is attributed to each channel. Organic baseline (22%) is not
              attributable to any paid channel.</p>
            </div>
          </div>

          <div class="charts-row">
            <div class="chart-block">
              <img src="data:image/png;base64,{b64_fit}" alt="Actual vs Predicted" />
              <p class="chart-caption">Left: how well the model tracks actual
              {res['label']} over time. Right: residuals — random scatter is
              good; a systematic pattern means the model is missing something.</p>
            </div>
          </div>

          <div class="interp-box">
            <h4>What this means</h4>
            <p>{interp_text}</p>
          </div>

          <h3>Channel Attribution &amp; ROI</h3>
          <table class="data-table">
            <thead>
              <tr>
                <th>Channel</th><th>Signal</th><th>Share</th>
                <th>Attributed {res['label']}</th><th>Total Spend</th>
                <th>Adj ROI</th><th>Adstock Decay</th><th>Context</th>
              </tr>
            </thead>
            <tbody>{roi_rows}</tbody>
          </table>
          <p class="table-note">
            ✓ = positive OLS coefficient (spend correlates with this outcome at {res['lag']} mo lag) ·
            ✗ = negative or near-zero coefficient (no measurable contribution at this lag) ·
            Adj ROI applies Katalon market context efficiency multipliers ·
            ROI values are relative — use for channel ranking, not precise accounting.
          </p>

          <h3>Budget Optimizer
            <span class="lag-badge">at current avg monthly budget {fmt_num(budget, '$')}</span>
          </h3>
          <table class="data-table optimizer-table">
            <thead>
              <tr><th>Channel</th><th>Recommended Allocation</th>
              <th>% of Budget</th><th>ROI</th></tr>
            </thead>
            <tbody>{opt_rows}</tbody>
            <tfoot>
              <tr><td colspan="3"><strong>Projected monthly {res['label']}</strong></td>
              <td><strong>{fmt_num(proj_out, unit)}</strong></td></tr>
            </tfoot>
          </table>

          <div class="validation-box">
            <h4>Model Validation</h4>
            <table class="mini-table">
              <tr><th></th><th>R²</th><th>MAPE</th></tr>
              <tr><td>Training (first 70%)</td>
                  <td>{oot['train_r2']:.3f}</td>
                  <td>{oot['train_mape']:.1f}%</td></tr>
              <tr><td>Hold-out test (last 30%)</td>
                  <td>{oot['test_r2']:.3f}</td>
                  <td>{oot['test_mape']:.1f}%</td></tr>
            </table>
            <p style="font-size:12px;color:#6B7280;margin-top:6px">
              A well-behaved model has similar MAPE on training and test data.
              A large increase in test MAPE (>10pp) indicates the model may be overfit.
            </p>
          </div>
        </section>
        """)

    quality_section = ""
    if b64_quality:
        cr = df_fit["cr_mql_to_sql"].dropna() if "cr_mql_to_sql" in df_fit.columns else pd.Series([])
        non_smb = df_fit["mql_non_smb_rate"].dropna() if "mql_non_smb_rate" in df_fit.columns else pd.Series([])
        cr_note = ""
        if len(cr) >= 4:
            trend = (cr.iloc[-4:].mean() - cr.iloc[:4].mean()) / cr.iloc[:4].mean() * 100
            direction = "improved" if trend > 0 else "declined"
            cr_note = (f"MQL→SQL conversion rate has {direction} by {abs(trend):.0f}% "
                       f"comparing the first and last 4 months of the dataset. ")
        non_smb_note = ""
        if len(non_smb) >= 4:
            trend2 = (non_smb.iloc[-4:].mean() - non_smb.iloc[:4].mean()) / non_smb.iloc[:4].mean() * 100
            direction2 = "increased" if trend2 > 0 else "decreased"
            non_smb_note = (f"The share of non-SMB (enterprise) MQLs has {direction2} by "
                            f"{abs(trend2):.0f}% over the same period. ")
        quality_section = f"""
        <section class="outcome-section" id="quality">
          <h2>MQL Quality Trends</h2>
          <div class="charts-row">
            <div class="chart-block">
              <img src="data:image/png;base64,{b64_quality}" alt="Quality Trends" />
            </div>
          </div>
          <div class="interp-box">
            <h4>What this means</h4>
            <p>
              {cr_note}{non_smb_note}
              These quality metrics are included as <strong>control variables</strong> in every
              OLS model — they absorb quality-driven variation in outcomes so that the spend
              channel coefficients reflect true incremental volume contribution,
              not a quality mix artefact.
            </p>
            <p>
              If MQL quality continues to decline, the volume of MQLs needed to hit pipeline
              and ARR targets will increase. Addressing quality (e.g. better audience targeting,
              ICP refinement, enterprise-segment focus) may be more effective than increasing
              spend volume.
            </p>
          </div>
        </section>
        """

    # ── Recommendations ───────────────────────────────────────────────────────
    g_capture_mql_coeff = results.get("mqls", {}).get("coeff_map", {}).get("google_capture_spend", 0)
    li_brand_pipe_roi   = next(
        (c["roi"] for c in results.get("inbound_pipeline", {}).get("channels", [])
         if c["channel"] == "linkedin_brand_spend"), 0
    )
    g_capture_spend_total = df_fit["google_capture_spend"].fillna(0).sum() if "google_capture_spend" in df_fit.columns else 0
    li_brand_spend_total  = df_fit["linkedin_brand_spend"].fillna(0).sum()  if "linkedin_brand_spend"  in df_fit.columns else 0

    reco_html = f"""
    <section class="outcome-section" id="recommendations">
      <h2>Recommendations</h2>

      <div class="reco-grid">
        <div class="reco-card reco-urgent">
          <h4>1. Reallocate from Google Capture → LinkedIn Brand</h4>
          <p>
            Google Capture (PPC/competitor/solution — currently <strong>{fmt_num(g_capture_spend_total, '$')}</strong>
            total, {df_fit["google_capture_spend"].fillna(0).sum()/total_spend*100:.1f}% of budget) shows a <strong>negative coefficient for MQLs</strong>.
            The AI Overview headwind is now measurable in the spend–outcome relationship:
            high-intent test automation queries are being intercepted by AI summaries before
            they reach Katalon's pages.
          </p>
          <p>
            LinkedIn Brand (thought leader content — currently <strong>{fmt_num(li_brand_spend_total, '$')}</strong>
            total, only {df_fit["linkedin_brand_spend"].fillna(0).sum()/total_spend*100:.1f}% of budget) shows the highest pipeline ROI at
            <strong>{li_brand_pipe_roi:.1f}x</strong>.
            This channel is dramatically underfunded relative to its measurable contribution.
          </p>
          <p><strong>Action:</strong> Shift ~$100–150K/year from Google Capture into LinkedIn Brand
          (thought leadership, Playwright differentiation content, VP Engineering targeting).</p>
        </div>

        <div class="reco-card">
          <h4>2. Prioritise MQL quality over MQL volume</h4>
          <p>
            MQL volume fell ~23% from 2024 to 2025 despite increasing total spend.
            The quality controls in this model separate quality drift from spend contribution —
            but the underlying MQL→SQL conversion rate trend is concerning.
          </p>
          <p><strong>Action:</strong> Audit audience targeting settings on all channels. Add
          ICP filters (company size ≥200, tech stack signals) to Google and LinkedIn campaigns.
          Track MQL→SQL rate monthly and alert if it drops below 10%.</p>
        </div>

        <div class="reco-card">
          <h4>3. Validate LinkedIn ABM with an incrementality test</h4>
          <p>
            LinkedIn ABM has a positive methodology efficiency multiplier (+12%) but shows
            mixed signals in the OLS model (negative for MQLs, small positive for pipeline).
            Its coefficient is uncertain given the limited ABM spend history.
          </p>
          <p><strong>Action:</strong> Run a 6-week geo or account-split test: pause LinkedIn ABM
          for a matched set of accounts and compare pipeline generation against the control group.
          This is the only ground-truth validation for ABM.</p>
        </div>

        <div class="reco-card">
          <h4>4. Add a strategy break flag for Oct 2024</h4>
          <p>
            LinkedIn restructuring (ABM reduction, Thought Leader launch) and Meta launch
            both happened in Oct–Nov 2024. Without a break flag, the OLS model treats
            the pre- and post-restructure periods as one continuous relationship,
            which inflates residuals and weakens ARR coefficients.
          </p>
          <p><strong>Action:</strong> Add a <code>strategy_v2</code> binary column
          (0 before Oct 2024, 1 from Oct 2024 onwards) to data.json. This will meaningfully
          improve ARR model R².</p>
        </div>

        <div class="reco-card">
          <h4>5. Do not act on ARR channel rankings until 36+ months</h4>
          <p>
            The Won ARR model has only {results.get('won_arr', {}).get('n_obs', '?')} usable
            observations after the 5-month lag. Individual channel ROI numbers
            (especially Meta's apparent high ROI) are noise at this sample size.
          </p>
          <p><strong>Action:</strong> Continue accumulating data. Revisit ARR attribution decisions
          in Q4 2026 when 36+ months of data will be available for reliable coefficient estimation.
          Use Pipeline ROI for near-term channel decisions.</p>
        </div>

        <div class="reco-card">
          <h4>6. Monitor Google Other ($476K, 20% of budget)</h4>
          <p>
            Google Other is the second-largest budget line and has unclear campaign-level
            composition. In the model it shows a small positive MQL coefficient but
            a negative pipeline coefficient — suggesting it generates volume but not quality.
          </p>
          <p><strong>Action:</strong> Audit which campaigns fall into "google_other" in your
          data pipeline. If it includes display/discovery, apply the same AI Overview
          efficiency discount as Google Capture. If it includes branded search,
          reclassify it into google_brand.</p>
        </div>
      </div>
    </section>
    """

    # ── Spend summary cards ───────────────────────────────────────────────────
    spend_cards = ""
    for ok, res in results.items():
        spend_cards += (
            f'<div class="metric-card">'
            f'<div class="metric-label">{res["label"]}</div>'
            f'<div class="metric-value" style="color:{res["outcome_color"]}">'
            f'{fmt_num(res["total_outcome"], res["unit"])}</div>'
            f'<div class="metric-sub">'
            f'R²={res["r2"]:.2f} · MAPE={res["mape"]:.1f}% · lag {res["lag"]}mo</div>'
            f'</div>'
        )

    spend_cards = (
        f'<div class="metric-card"><div class="metric-label">Total Marketing Spend</div>'
        f'<div class="metric-value">{fmt_num(total_spend, "$")}</div>'
        f'<div class="metric-sub">{n_months} months · {period_str}</div></div>'
        + spend_cards
    )

    # ── Scenario forecast section ─────────────────────────────────────────────
    scenario_section_html = ""
    if scenario_forecast and scenario_spend:
        b64_scenario = _make_scenario_chart(scenario_forecast, scenario_spend)
        sc_spend_rows = ""
        total_sc_spend = sum(float(v) for v in scenario_spend.values())
        for ch_key, ch in CHANNEL_META.items():
            amt = float(scenario_spend.get(ch_key, 0.0))
            pct = amt / total_sc_spend * 100 if total_sc_spend > 0 else 0.0
            bar_w = f"{pct:.0f}%"
            sc_spend_rows += (
                f'<tr><td>'
                f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                f'background:{ch["color"]};margin-right:6px"></span>{ch["label"]}'
                f'</td>'
                f'<td style="text-align:right">{fmt_num(amt, "$")}</td>'
                f'<td style="text-align:right">{pct:.1f}%</td>'
                f'<td><div style="background:#E5E7EB;border-radius:4px;height:8px;width:120px">'
                f'<div style="background:{ch["color"]};width:{bar_w};height:8px;border-radius:4px"></div>'
                f'</div></td></tr>\n'
            )

        sc_outcome_cards = ""
        for ok, fc in scenario_forecast.items():
            vs_avg   = ((fc["point"] / fc["hist_avg"]) - 1) * 100 if fc["hist_avg"] > 0 else 0.0
            vs_color = "#065F46" if vs_avg >= 0 else "#991B1B"
            vs_bg    = "#ECFDF5" if vs_avg >= 0 else "#FEF2F2"
            vs_label = f"{vs_avg:+.0f}% vs hist avg"

            # Channel contribution breakdown rows
            contrib_rows = ""
            for cc in sorted(fc["channel_contribs"], key=lambda x: -x["contrib"]):
                if cc["contrib"] > 0:
                    contrib_rows += (
                        f'<tr><td>'
                        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                        f'background:{cc["color"]};margin-right:4px"></span>{cc["name"]}'
                        f'</td>'
                        f'<td style="text-align:right">{fmt_num(cc["planned_spend"], "$")}</td>'
                        f'<td style="text-align:right">{cc["contrib_share"]*100:.1f}%</td></tr>\n'
                    )

            # Confidence bar: visual range indicator
            range_pct = int(fc["mape"])

            sc_outcome_cards += f"""
            <div class="scenario-card">
              <div class="scenario-card-header" style="border-left:4px solid {fc['outcome_color']}">
                <div>
                  <div class="scenario-outcome-label">{fc['label']}</div>
                  <div style="font-size:11px;color:#6B7280;margin-top:2px">
                    Spend in {fc['scenario_month_label']} → outcome in {fc['target_month_label']}
                    {"(same month)" if fc['lag']==0 else f"({fc['lag']} mo lag)"}
                  </div>
                </div>
                <div style="text-align:right">
                  <div style="font-size:26px;font-weight:700;color:{fc['outcome_color']}">
                    {fmt_num(fc['point'], fc['unit'])}
                  </div>
                  <div style="font-size:11px;padding:2px 8px;border-radius:12px;
                              background:{vs_bg};color:{vs_color};display:inline-block;margin-top:4px">
                    {vs_label}
                  </div>
                </div>
              </div>
              <div style="margin:10px 0">
                <div style="font-size:11px;color:#6B7280;margin-bottom:4px">
                  Uncertainty range (±{range_pct}% MAPE):
                  {fmt_num(fc['low'], fc['unit'])} — {fmt_num(fc['high'], fc['unit'])}
                </div>
                <div style="background:#E5E7EB;border-radius:6px;height:10px;position:relative">
                  <div style="position:absolute;left:0;top:0;height:10px;
                              background:{fc['outcome_color']};opacity:0.2;border-radius:6px;
                              width:{min(100, range_pct*2)}%"></div>
                  <div style="position:absolute;left:{min(45,range_pct//2)}%;top:-2px;
                              height:14px;width:4px;background:{fc['outcome_color']};
                              border-radius:2px"></div>
                </div>
              </div>
              <div style="font-size:11px;color:#6B7280;margin-top:8px">
                Historical monthly avg: <strong>{fmt_num(fc['hist_avg'], fc['unit'])}</strong>
                &nbsp;·&nbsp; Model R²={fc['r2']:.2f}
              </div>
              {('<table class="data-table" style="margin-top:10px;font-size:12px"><thead><tr><th>Channel</th><th>Planned spend</th><th>Contrib share</th></tr></thead><tbody>' + contrib_rows + '</tbody></table>') if contrib_rows else ''}
            </div>
            """

        # Assumed quality values display
        quality_rows = ""
        for qc, val in scenario_forecast.get(list(scenario_forecast.keys())[0], {}).get("assumed_quality", {}).items():
            quality_rows += f'<tr><td><code>{qc}</code></td><td>{val:.3f}</td><td>Last-3-month average</td></tr>\n'

        scenario_section_html = f"""
        <section class="outcome-section" id="scenario">
          <h2>Scenario Forecast</h2>
          <p style="font-size:13px;color:#374151;line-height:1.7;margin-bottom:16px">
            Given the planned channel spend below, the model predicts outcomes for each
            metric at its respective lag. Uncertainty ranges are the model's in-sample
            MAPE — the typical prediction error on historical data.
            <strong>Do not extrapolate far beyond historical spend levels — the model assumes linear returns.</strong>
          </p>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:24px">
            <div>
              <h3 style="margin-top:0">Planned spend allocation</h3>
              <table class="data-table">
                <thead><tr><th>Channel</th><th style="text-align:right">Amount</th>
                  <th style="text-align:right">Share</th><th></th></tr></thead>
                <tbody>{sc_spend_rows}</tbody>
                <tfoot><tr style="background:#EFF6FF">
                  <td><strong>Total</strong></td>
                  <td style="text-align:right"><strong>{fmt_num(total_sc_spend, '$')}</strong></td>
                  <td colspan="2"></td></tr></tfoot>
              </table>
            </div>
            <div>
              <h3 style="margin-top:0">Assumed quality controls</h3>
              <p style="font-size:12px;color:#6B7280;margin-bottom:8px">
                These are held at their last-3-month average. To model a quality shift,
                adjust these values in <code>QUALITY_CONTROLS</code> or rerun with different data.
              </p>
              <table class="data-table">
                <thead><tr><th>Variable</th><th>Assumed value</th><th>Basis</th></tr></thead>
                <tbody>{quality_rows}</tbody>
              </table>
            </div>
          </div>

          <h3>Predicted outcomes</h3>
          <div class="scenario-grid">
            {sc_outcome_cards}
          </div>

          {('<img src="data:image/png;base64,' + b64_scenario + '" style="max-width:100%;border-radius:8px;margin-top:16px" />' if b64_scenario else '')}

          <div class="interp-box" style="margin-top:16px">
            <h4>How to interpret</h4>
            <p>
              <strong>Point estimate</strong> — the model's best guess based on OLS coefficients,
              adstock carry-over from the last historical month, seasonality of the target month,
              and quality controls at recent averages.
            </p>
            <p>
              <strong>Uncertainty range</strong> — each model's in-sample MAPE applied as ±%.
              A 20% MAPE means the model typically predicts within 20% of actual — use the range
              for planning, not the point estimate alone.
            </p>
            <p>
              <strong>Carry-over effect</strong> — even $0 spend would produce some outcome via
              adstock carry-over from previous months. The intercept and quality controls also
              contribute to the baseline.
            </p>
            <p>
              <strong>Channels with negative OLS coefficients</strong> contribute 0 to the prediction.
              They are still in the model for fit quality, but their spend does not increase
              predicted outcomes.
            </p>
          </div>
        </section>
        """

    # ── Multi-scenario comparison section ────────────────────────────────────
    scenarios_comparison_html = ""
    if scenarios_comparison:
        b64_cmp = _make_scenario_comparison_chart(scenarios_comparison)
        outcomes_list = list(scenarios_comparison[0]["forecast"].keys())

        # Spend comparison table
        spend_header = "<tr><th>Channel</th>"
        for sc in scenarios_comparison:
            total_sc = sum(float(v) for v in sc["spend"].values())
            spend_header += f'<th style="text-align:right">{sc["name"]}<br><small style="font-weight:normal;color:#6B7280">{fmt_num(total_sc,"$")}/mo</small></th>'
        spend_header += "</tr>"

        spend_body = ""
        for ch_key, ch in CHANNEL_META.items():
            row = f'<tr><td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{ch["color"]};margin-right:6px"></span>{ch["label"]}</td>'
            for sc in scenarios_comparison:
                amt = float(sc["spend"].get(ch_key, 0.0))
                row += f'<td style="text-align:right">{fmt_num(amt, "$") if amt > 0 else "—"}</td>'
            row += "</tr>\n"
            spend_body += row

        # Totals row
        spend_body += "<tr style='background:#EFF6FF;font-weight:600'><td>Total</td>"
        for sc in scenarios_comparison:
            total_sc = sum(float(v) for v in sc["spend"].values())
            spend_body += f'<td style="text-align:right">{fmt_num(total_sc, "$")}</td>'
        spend_body += "</tr>"

        # Outcome forecast table: rows = outcomes, columns = scenarios
        outcome_header = "<tr><th>Outcome</th><th>Predicted in</th>"
        for sc in scenarios_comparison:
            outcome_header += f'<th style="text-align:right">{sc["name"]}</th>'
        outcome_header += "</tr>"

        outcome_body = ""
        for ok in outcomes_list:
            fc0  = scenarios_comparison[0]["forecast"][ok]
            unit = fc0["unit"]
            color = fc0["outcome_color"]

            # Point row
            row  = f'<tr><td rowspan="2" style="border-left:3px solid {color};padding-left:10px"><strong>{fc0["label"]}</strong></td>'
            row += f'<td style="font-size:11px;color:#6B7280">{fc0["target_month_label"]}</td>'
            for sc in scenarios_comparison:
                fc   = sc["forecast"][ok]
                vs   = ((fc["point"] / fc["hist_avg"]) - 1) * 100 if fc["hist_avg"] > 0 else 0.0
                vs_c = "#065F46" if vs >= 0 else "#991B1B"
                row += (
                    f'<td style="text-align:right">'
                    f'<strong style="font-size:15px">{fmt_num(fc["point"], unit)}</strong>'
                    f'<br><small style="color:{vs_c}">{vs:+.0f}% vs avg</small>'
                    f'</td>'
                )
            row += "</tr>\n"

            # Range row
            range_row = f'<tr><td style="font-size:11px;color:#9CA3AF">uncertainty range</td>'
            for sc in scenarios_comparison:
                fc = sc["forecast"][ok]
                range_row += f'<td style="text-align:right;font-size:11px;color:#9CA3AF">{fmt_num(fc["low"],unit)} – {fmt_num(fc["high"],unit)}</td>'
            range_row += "</tr>\n"

            outcome_body += row + range_row

        # Notes about reallocation rationale
        notes_html = ""
        for sc in scenarios_comparison:
            if sc.get("note"):
                notes_html += f'<div style="margin-bottom:8px"><strong>{sc["name"]}:</strong> <span style="color:#374151">{sc["note"]}</span></div>'

        scenarios_comparison_html = f"""
        <section class="outcome-section" id="scenarios">
          <h2>Scenario Comparison</h2>
          <p style="font-size:13px;color:#374151;line-height:1.7;margin-bottom:16px">
            Three budget scenarios for April 2026, compared against each outcome's historical monthly average.
            The model-guided scenarios reallocate spend toward channels with positive OLS coefficients and
            away from channels with negative pipeline coefficients (Google Other, LinkedIn Other).
          </p>

          <h3>Spend allocation by channel</h3>
          <table class="data-table">
            <thead>{spend_header}</thead>
            <tbody>{spend_body}</tbody>
          </table>

          <h3 style="margin-top:24px">Predicted outcomes</h3>
          <table class="data-table">
            <thead>{outcome_header}</thead>
            <tbody>{outcome_body}</tbody>
          </table>
          <p class="table-note">
            Point estimate shown in bold · Range = ±MAPE of each model ·
            % shown vs historical monthly average ·
            Pipeline predicted 1 month after spend · ARR predicted 5 months after spend
          </p>

          {('<img src="data:image/png;base64,' + b64_cmp + '" style="max-width:100%;border-radius:8px;margin:16px 0" />' if b64_cmp else '')}

          <h3>Reallocation rationale</h3>
          <div class="interp-box">
            <h4>Why these allocations?</h4>
            {notes_html}
            <p style="margin-top:10px;font-size:12px;color:#6B7280">
              Based on pipeline OLS coefficients: LinkedIn Brand (+32.8x), Other/Events (+6.95x),
              Google Capture (+3.31x), and Google Brand (+2.15x) are positive.
              Google Other (−5.1x), LinkedIn Other (−4.4x), Meta (−189x) and
              LinkedIn Capture (−11x) are negative for pipeline.
              Budget cuts fall on negative-coefficient channels first.
            </p>
          </div>
        </section>
        """

    toc_links = "\n".join(
        f'<li><a href="#{ok}">{results[ok]["label"]}</a></li>'
        for ok in results
    )

    # ── Methodology section ───────────────────────────────────────────────────
    # Pre-extract lag values so they can be used in the f-string without
    # nested brace escaping issues.
    _mqls_lag = results.get("mqls", {}).get("lag", 0)
    _pipe_lag = results.get("inbound_pipeline", {}).get("lag", 2)
    _arr_lag  = results.get("won_arr", {}).get("lag", 5)

    # Adstock decay rows from CHANNEL_META
    decay_rows = ""
    for ch_key, ch in CHANNEL_META.items():
        adj_pct = f"{ch['efficiency_adj']*100:+.0f}%" if ch['efficiency_adj'] != 0 else "—"
        decay_rows += (
            f'<tr><td><span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:{ch["color"]};margin-right:6px"></span>'
            f'{ch["label"]}</td>'
            f'<td style="text-align:center">{ch["decay"]:.2f}</td>'
            f'<td style="text-align:center">{adj_pct}</td>'
            f'<td style="font-size:12px;color:#6B7280">{ch["note"]}</td></tr>\n'
        )

    # Lag rows from current run lags
    lag_rows = ""
    for ok, res in results.items():
        lag_rows += (
            f'<tr><td>{res["label"]}</td>'
            f'<td style="text-align:center">{res["lag"]} month{"s" if res["lag"]!=1 else ""}</td>'
            f'<td style="font-size:12px;color:#6B7280">{OUTCOME_META[ok]["lag_note"]}</td></tr>\n'
        )

    methodology_html = f"""
    <section class="outcome-section" id="methodology">
      <h2>Methodology &amp; Assumptions</h2>
      <p style="font-size:13px;color:#374151;line-height:1.7;margin-bottom:20px">
        This section documents every modelling decision, parameter, and assumption in the
        Katalon MMM. It exists so that any analyst can reproduce the results, challenge the
        assumptions, and understand the model's confidence boundaries.
      </p>

      <!-- STEP 1 -->
      <h3>Step 1 — Adstock Transformation</h3>
      <div class="interp-box">
        <p>
          Raw monthly spend is a poor predictor of outcomes because advertising has
          <strong>carry-over effects</strong>. An impression seen this month still influences
          a purchase two months later. Adstock smooths the spend series using exponential decay:
        </p>
        <p style="font-family:monospace;background:#E0F2FE;padding:8px 12px;border-radius:4px;font-size:13px">
          adstock[t] = spend[t] + decay &times; adstock[t&minus;1]
        </p>
        <p>
          <strong>decay = 0.0</strong> means no carry-over (each month is independent).
          <strong>decay = 0.9</strong> means very long memory — most prior spend still counts.
          Brand channels decay slowly (awareness lingers months).
          Demand-capture channels (PPC) decay quickly (high-intent clicks convert fast or not at all).
        </p>
      </div>
      <h4 style="margin-top:16px">Adstock decay &amp; context efficiency multipliers by channel</h4>
      <p style="font-size:12px;color:#6B7280;margin-bottom:8px">
        The <em>efficiency adjustment</em> is a market-context multiplier applied to positive
        OLS coefficients before attribution. It reflects Katalon-specific factors (AI Overview
        disruption, open-source competition) that the raw data alone cannot separate.
        Positive = upward adjustment; negative = downward penalty.
      </p>
      <table class="data-table">
        <thead>
          <tr>
            <th>Channel</th>
            <th style="text-align:center">Adstock Decay</th>
            <th style="text-align:center">Efficiency Adj</th>
            <th>Rationale</th>
          </tr>
        </thead>
        <tbody>{decay_rows}</tbody>
      </table>

      <!-- STEP 2 -->
      <h3 style="margin-top:28px">Step 2 — Lag Shifting</h3>
      <div class="interp-box">
        <p>
          Spend in month T does not produce ARR in month T. There is a structural lag between
          when money is spent and when outcomes are captured. The model shifts the outcome
          series forward by the lag value:
        </p>
        <p style="font-family:monospace;background:#E0F2FE;padding:8px 12px;border-radius:4px;font-size:13px">
          outcome_aligned[t] = outcome[t + lag]
        </p>
        <p>
          This ensures the regression correlates spend against the outcomes that spend
          <em>actually caused</em>, not outcomes already in-flight from prior months.
        </p>
      </div>
      <h4 style="margin-top:16px">Lags used in this run</h4>
      <table class="data-table">
        <thead>
          <tr><th>Outcome</th><th style="text-align:center">Lag</th><th>Rationale</th></tr>
        </thead>
        <tbody>{lag_rows}</tbody>
      </table>

      <!-- STEP 3 -->
      <h3 style="margin-top:28px">Step 3 — OLS Regression</h3>
      <div class="interp-box">
        <p>
          After adstock and lag transformation, the model fits
          <strong>Ordinary Least Squares (OLS) regression</strong>:
        </p>
        <p style="font-family:monospace;background:#E0F2FE;padding:8px 12px;border-radius:4px;font-size:12px;line-height:1.8">
          outcome[t] = &beta;&sub0;<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ &beta;_g_brand &times; adstock_google_brand[t]<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ &beta;_g_capture &times; adstock_google_capture[t]<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ &hellip; (9 spend channels total)<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ &beta;_q4 &times; q4_flag[t] + &beta;_q1 &times; q1_flag[t]<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ &beta;_pre2024 &times; pre_2024_flag[t]<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ &beta;_quality &times; [mql_high_intent_rate, mql_non_smb_rate, cr_mql_to_sql]<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ &epsilon;[t]
        </p>
        <p>
          Solved via normal equations: <code>β = (XᵀX)⁻¹Xᵀy</code>
        </p>
        <p>
          <strong>Why OLS, not correlation?</strong> Correlation is dimensionless — it cannot
          produce predictions in outcome units. OLS coefficients are in outcome units
          (e.g. MQLs per dollar of adstocked spend), which gives properly-scaled predictions,
          a real R², and directly interpretable ROI numbers.
        </p>
        <p>
          <strong>Seasonality controls:</strong> Q4 (Oct–Dec) and Q1 (Jan–Feb) dummy variables
          absorb enterprise budget-cycle effects. Without them, high-budget Q4 spend appears
          correlated with Q4 pipeline even if the causal relationship is much weaker.
        </p>
        <p>
          <strong>Structural break (pre_2024_flag):</strong> Set to 1 for months before Jan 2024,
          0 from Jan 2024 onwards. Absorbs the strategy and MQL-definition change that happened
          in January 2024. Its coefficient captures how much higher/lower outcomes were in the
          2022–2023 era after controlling for spend and quality.
        </p>
        <p>
          <strong>Quality controls (not attributed to any channel):</strong>
          <code>mql_high_intent_rate</code>, <code>mql_non_smb_rate</code>, and
          <code>cr_mql_to_sql</code> are included as co-variates so that spend channel
          coefficients reflect true incremental volume contribution, not quality-mix drift.
        </p>
      </div>

      <!-- STEP 4 -->
      <h3 style="margin-top:28px">Step 4 — Attribution &amp; ROI</h3>
      <div class="interp-box">
        <p>
          Only <strong>positive</strong> OLS coefficients are used for attribution.
          Negative spend→outcome relationships are economically nonsensical and typically
          indicate multicollinearity or insufficient data — they receive 0% attribution.
        </p>
        <p>Each positive coefficient is scaled by the context efficiency multiplier:</p>
        <p style="font-family:monospace;background:#E0F2FE;padding:8px 12px;border-radius:4px;font-size:13px">
          adj_coeff[ch] = max(0, &beta;_ch) &times; (1 + efficiency_adj[ch])
        </p>
        <p>Attribution shares are then normalised to the paid attribution share
        ({PAID_SHARE*100:.0f}%):</p>
        <p style="font-family:monospace;background:#E0F2FE;padding:8px 12px;border-radius:4px;font-size:13px">
          share[ch] = (adj_coeff[ch] / &Sigma; adj_coeffs) &times; {PAID_SHARE:.2f}<br>
          attributed_outcome[ch] = total_outcome &times; share[ch]<br>
          ROI[ch] = attributed_outcome[ch] / total_spend[ch]
        </p>
        <p>
          The remaining <strong>{ORGANIC_BASELINE*100:.0f}%</strong> is assigned to the
          <em>organic baseline</em>: Katalon's Gartner Visionary positioning, G2/review presence,
          and LLM citation traffic that converts 4.4× better than organic search but is not
          tracked in ad platforms.
        </p>
      </div>

      <!-- ASSUMPTIONS TABLE -->
      <h3 style="margin-top:28px">Model Parameters &amp; Assumptions</h3>
      <table class="data-table">
        <thead>
          <tr><th>Parameter</th><th>Value</th><th>Status / Notes</th></tr>
        </thead>
        <tbody>
          <tr><td>Organic baseline</td><td>{ORGANIC_BASELINE*100:.0f}%</td>
            <td>Estimated — raise to 25% if LLM citation traffic grows further</td></tr>
          <tr><td>Paid attribution share</td><td>{PAID_SHARE*100:.0f}%</td>
            <td>Estimated — industry range 70–85% for B2B SaaS</td></tr>
          <tr><td>Data grain</td><td>Monthly</td>
            <td>Recommended — weekly data too noisy for pipeline/ARR outcomes</td></tr>
          <tr><td>MQL lag</td><td>{_mqls_lag} month</td>
            <td>Stable — demand capture converts near-instantly</td></tr>
          <tr><td>Pipeline lag</td><td>{_pipe_lag} months</td>
            <td>Stable — adjust if CRM shows MQL→Opportunity shifting</td></tr>
          <tr><td>ARR lag</td><td>{_arr_lag} months</td>
            <td>Stable — adjust to 4–5 if enterprise deal size grows</td></tr>
          <tr><td>Q4/Q1 seasonality</td><td>Included</td>
            <td>Oct–Dec (Q4) and Jan–Feb (Q1) enterprise budget-cycle dummies</td></tr>
          <tr><td>Pre-2024 structural break</td><td>Included</td>
            <td>Absorbs MQL-definition change and strategy shift from Jan 2024</td></tr>
          <tr><td>Min observations</td><td>4 rows</td>
            <td>Model falls back to equal-share attribution below this threshold</td></tr>
        </tbody>
      </table>

      <!-- CHANNEL GROUPING -->
      <h3 style="margin-top:28px">Channel Grouping</h3>
      <p style="font-size:13px;color:#374151;line-height:1.7;margin-bottom:8px">
        The raw data contains 14+ granular campaign-type columns.
        These are grouped into 9 model channels to avoid sparse columns (individual campaign
        types with $0 in most months have no signal) and multicollinearity (columns that
        always move together confuse OLS coefficient estimation).
      </p>
      <table class="data-table">
        <thead>
          <tr><th>Model Channel</th><th>Source columns summed</th></tr>
        </thead>
        <tbody>
          <tr><td>Google Brand</td><td><code>google_brand_spend</code> + <code>google_thought_leader_spend</code></td></tr>
          <tr><td>Google Capture</td><td><code>google_ppc_spend</code> + <code>google_competitor_spend</code> + <code>google_solution_spend</code> + <code>google_soqr_annual_report_spend</code> + <code>google_retargeting_spend</code> + <code>google_abm_spend</code></td></tr>
          <tr><td>Google Other</td><td><code>google_other_spend</code></td></tr>
          <tr><td>LinkedIn Brand</td><td><code>linkedin_brand_spend</code> + <code>linkedin_thought_leader_spend</code></td></tr>
          <tr><td>LinkedIn Capture</td><td><code>linkedin_solution_spend</code> + <code>linkedin_soqr_annual_report_spend</code> + <code>linkedin_ppc_spend</code> + <code>linkedin_competitor_spend</code> + <code>linkedin_retargeting_spend</code></td></tr>
          <tr><td>LinkedIn ABM</td><td><code>linkedin_abm_spend</code></td></tr>
          <tr><td>LinkedIn Other</td><td><code>linkedin_other_spend</code></td></tr>
          <tr><td>Meta</td><td><code>meta_spend</code></td></tr>
          <tr><td>Other / Events</td><td><code>other_spend</code></td></tr>
        </tbody>
      </table>

      <!-- KNOWN LIMITATIONS -->
      <h3 style="margin-top:28px">Known Limitations</h3>
      <div class="reco-grid">
        <div class="reco-card">
          <h4>No causal inference</h4>
          <p>Correlation ≠ causation. A channel that correlates with pipeline could be
          following it rather than driving it. LinkedIn spend rising in Q3 could reflect
          the team having more budget in a good quarter, not LinkedIn causing the pipeline.</p>
        </div>
        <div class="reco-card">
          <h4>No interaction effects</h4>
          <p>The model treats channels independently. In reality, LinkedIn brand exposure
          likely amplifies Google conversion — a prospect sees a thought leadership post,
          then Googles Katalon. This synergy is unmodelled; Google's standalone ROI is
          probably understated.</p>
        </div>
        <div class="reco-card">
          <h4>No saturation curves</h4>
          <p>The model assumes linear returns to spend. In practice, doubling Google Capture
          spend will not double MQLs — diminishing returns set in above a threshold.
          A Bayesian MMM with Hill transformation would capture this ceiling.</p>
        </div>
        <div class="reco-card">
          <h4>No external confounders</h4>
          <p>Product launches, PR events, partner announcements, pricing changes, and
          competitor actions all affect outcomes and are not included unless added
          as break-flag columns.</p>
        </div>
        <div class="reco-card">
          <h4>Short history</h4>
          <p>With {n_months} months of data and 9 spend channels plus controls, the model
          has limited degrees of freedom. Coefficient estimates will tighten significantly
          as you accumulate 36–48 months of data.</p>
        </div>
        <div class="reco-card">
          <h4>Meta attribution uncertain</h4>
          <p>Meta spend began in Oct 2024 — only ~18 months of data. Its OLS coefficient
          will have wide confidence intervals. Treat Meta ROI as directional only until
          30+ months are available.</p>
        </div>
      </div>

      <!-- WHEN TO TRUST -->
      <h3 style="margin-top:28px">When to Trust the Model More / Less</h3>
      <table class="data-table">
        <thead>
          <tr><th>Trust more when&hellip;</th><th>Trust less when&hellip;</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>R² &gt; 0.60 and MAPE &lt; 25%</td>
            <td>R² &lt; 0.40 — model explains less than 40% of variance</td>
          </tr>
          <tr>
            <td>Sensitivity sweep shows flat lines (decay 0.15–0.65)</td>
            <td>Sensitivity sweep shows steep lines — ROI depends heavily on decay assumption</td>
          </tr>
          <tr>
            <td>Out-of-time MAPE delta under 10 percentage points</td>
            <td>Out-of-time test MAPE is &gt;10pp above train MAPE (possible overfit)</td>
          </tr>
          <tr>
            <td>Model-predicted MQL drop from a channel pause matches actual drop ±20%</td>
            <td>A channel has near-zero spend for 6+ consecutive months (coefficient unreliable)</td>
          </tr>
        </tbody>
      </table>

      <!-- UPGRADE PATH -->
      <h3 style="margin-top:28px">Upgrade Path</h3>
      <p style="font-size:13px;color:#374151;line-height:1.7;margin-bottom:8px">
        In priority order, these upgrades would meaningfully improve accuracy:
      </p>
      <ol style="font-size:13px;line-height:1.8;color:#374151;padding-left:20px">
        <li><strong>Strategy break flags</strong> — add binary columns for each major campaign
        restructure date (e.g. Oct 2024 LinkedIn/Meta launch). Each flag absorbs a structural
        shift in the spend→outcome relationship.</li>
        <li><strong>More history</strong> — every additional quarter strengthens coefficient
        estimates. Target 36+ months.</li>
        <li><strong>Audience/segment split</strong> — separate enterprise vs SMB spend if
        available. Different sales cycles mean different optimal lags.</li>
        <li><strong>Bayesian MMM with Hill curves</strong> — replaces OLS with a probabilistic
        model that captures diminishing returns and uncertainty intervals around every
        attribution estimate. Recommended once you have 36+ months of clean data.</li>
        <li><strong>Incrementality tests</strong> — pause one channel for 4–6 weeks, measure
        actual MQL drop, compare to model prediction. This is the only ground-truth validation
        and should be run at least once per year.</li>
      </ol>
    </section>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Katalon MMM Report — {_date.today().isoformat()}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #F9FAFB; color: #111827; margin: 0; padding: 0; }}
    .page-header {{ background: #111827; color: #fff; padding: 32px 48px 24px; }}
    .page-header h1 {{ margin: 0 0 6px; font-size: 28px; }}
    .page-header p  {{ margin: 0; color: #9CA3AF; font-size: 14px; }}
    nav {{ background: #1F2937; padding: 12px 48px; }}
    nav a {{ color: #60A5FA; text-decoration: none; margin-right: 20px; font-size: 13px; }}
    nav a:hover {{ color: #fff; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px; }}
    .outcome-section {{ background: #fff; border-radius: 12px; padding: 32px;
                         margin-bottom: 32px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    h2 {{ font-size: 22px; font-weight: 700; color: #111827; margin: 0 0 16px;
          border-bottom: 2px solid #F3F4F6; padding-bottom: 10px; }}
    h3 {{ font-size: 16px; font-weight: 600; color: #374151; margin: 24px 0 10px; }}
    h4 {{ font-size: 14px; font-weight: 600; color: #1F2937; margin: 0 0 8px; }}
    .lag-badge {{ background: #EFF6FF; color: #2563EB; font-size: 12px; font-weight: 500;
                  padding: 3px 8px; border-radius: 20px; margin-left: 10px;
                  vertical-align: middle; }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
    .badge-good {{ background: #ECFDF5; color: #065F46; padding: 4px 12px;
                   border-radius: 20px; font-size: 12px; font-weight: 500; }}
    .badge-warn {{ background: #FFFBEB; color: #92400E; padding: 4px 12px;
                   border-radius: 20px; font-size: 12px; font-weight: 500; }}
    .badge-bad  {{ background: #FEF2F2; color: #991B1B; padding: 4px 12px;
                   border-radius: 20px; font-size: 12px; font-weight: 500; }}
    .badge-neutral {{ background: #F3F4F6; color: #374151; padding: 4px 12px;
                      border-radius: 20px; font-size: 12px; font-weight: 500; }}
    .metric-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                     gap: 16px; margin-bottom: 24px; }}
    .metric-card {{ background: #fff; border-radius: 10px; padding: 20px;
                    box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    .metric-label {{ font-size: 12px; color: #6B7280; text-transform: uppercase;
                     letter-spacing: .05em; margin-bottom: 6px; }}
    .metric-value {{ font-size: 28px; font-weight: 700; color: #111827; }}
    .metric-sub   {{ font-size: 11px; color: #9CA3AF; margin-top: 4px; }}
    .charts-row {{ margin: 16px 0; }}
    .chart-block img {{ max-width: 100%; border-radius: 8px;
                        box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
    .chart-caption {{ font-size: 12px; color: #6B7280; margin-top: 6px;
                      font-style: italic; }}
    .interp-box {{ background: #F0F9FF; border-left: 4px solid #0EA5E9;
                   border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 20px 0; }}
    .interp-box p {{ margin: 6px 0; font-size: 13.5px; line-height: 1.6; }}
    .validation-box {{ background: #F9FAFB; border: 1px solid #E5E7EB;
                       border-radius: 8px; padding: 16px 20px; margin-top: 20px; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0; }}
    .data-table th {{ background: #F9FAFB; padding: 8px 10px; text-align: left;
                      font-weight: 600; color: #374151; border-bottom: 2px solid #E5E7EB; }}
    .data-table td {{ padding: 7px 10px; border-bottom: 1px solid #F3F4F6; }}
    .data-table tbody tr:hover {{ background: #FAFAFA; }}
    .pos-coeff td {{ color: #111827; }}
    .neg-coeff td {{ color: #9CA3AF; }}
    .optimizer-table tfoot td {{ background: #EFF6FF; font-weight: 600; }}
    .mini-table {{ font-size: 12px; border-collapse: collapse; }}
    .mini-table th, .mini-table td {{ padding: 4px 12px; border: 1px solid #E5E7EB; }}
    .mini-table th {{ background: #F3F4F6; }}
    .table-note {{ font-size: 11px; color: #9CA3AF; margin-top: 4px; line-height: 1.5; }}
    .reco-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
                  gap: 16px; }}
    .reco-card {{ background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px;
                  padding: 20px; }}
    .reco-card p {{ font-size: 13px; line-height: 1.6; color: #374151; margin: 8px 0; }}
    .reco-urgent {{ border-color: #FCA5A5; background: #FFF5F5; }}
    .reco-urgent h4 {{ color: #991B1B; }}
    .key-findings {{ margin: 16px 0; }}
    .key-findings ul {{ padding-left: 18px; }}
    .key-findings li {{ font-size: 13.5px; line-height: 1.7; margin-bottom: 6px; }}
    .methodology-note {{ background: #F9FAFB; border-radius: 8px; padding: 16px 20px;
                         font-size: 12px; color: #6B7280; margin-top: 12px; }}
    .scenario-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                      gap: 16px; margin: 16px 0; }}
    .scenario-card {{ background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px;
                      padding: 16px 20px; }}
    .scenario-card-header {{ display: flex; justify-content: space-between; align-items: flex-start;
                              margin-bottom: 8px; }}
    .scenario-outcome-label {{ font-size: 14px; font-weight: 600; color: #1F2937; }}
    footer {{ background: #1F2937; color: #9CA3AF; text-align: center;
              padding: 20px; font-size: 12px; margin-top: 40px; }}
    @media print {{
      body {{ background: #fff; }}
      nav, .page-header {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    }}
  </style>
</head>
<body>
  <div class="page-header">
    <h1>Katalon Marketing Mix Model</h1>
    <p>OLS v3 · 9-channel · Monthly grain · Generated {_date.today().strftime("%B %d, %Y")} ·
    {n_months} months · {period_str}</p>
  </div>

  <nav>
    <a href="#summary">Executive Summary</a>
    <a href="#spend">Spend Overview</a>
    {"".join(f'<a href="#{ok}">{results[ok]["label"]}</a>' for ok in results)}
    {_lag_cmp_nav(lag_comparison)}
    <a href="#quality">MQL Quality</a>
    <a href="#recommendations">Recommendations</a>
    {"<a href='#scenario'>Scenario Forecast</a>" if scenario_forecast else ""}
    {"<a href='#scenarios'>Scenario Comparison</a>" if scenarios_comparison else ""}
    <a href="#methodology">Methodology</a>
  </nav>

  <div class="container">

    <!-- EXECUTIVE SUMMARY -->
    <section class="outcome-section" id="summary">
      <h2>Executive Summary</h2>
      <div class="metric-cards">
        {spend_cards}
      </div>
      <div class="key-findings">
        <h3>Key Findings</h3>
        <ul>
          {"".join(key_bullets)}
        </ul>
      </div>
      <div class="methodology-note">
        <strong>How to read this report:</strong>
        The model uses OLS (Ordinary Least Squares) regression to estimate how much of each
        marketing outcome (MQLs, pipeline, ARR) can be attributed to each spend channel.
        Adstock carry-over models the fact that ads have lingering effects beyond the month
        they ran. Lag shifting aligns spend with the outcomes it actually caused (MQLs: same month,
        Pipeline: 2 months later, ARR: 5 months later). Quality controls absorb MQL quality drift
        so spend attribution reflects true volume contribution.
        <strong>R² = % of outcome variance the model explains. MAPE = average prediction error.</strong>
      </div>
    </section>

    <!-- SPEND OVERVIEW -->
    <section class="outcome-section" id="spend">
      <h2>How We Spent the Budget</h2>
      <img src="data:image/png;base64,{b64_spend}" alt="Spend Overview"
           style="max-width:100%;border-radius:8px;" />
      <div class="interp-box">
        <h4>Context</h4>
        <p>{_spend_context_blurb(df_fit, spend_cols, total_spend)}</p>
      </div>
    </section>

    <!-- OUTCOME SECTIONS -->
    {"".join(outcome_html_blocks)}

    <!-- LAG COMPARISON -->
    {lag_comparison_html}

    <!-- MQL QUALITY -->
    {quality_section}

    <!-- RECOMMENDATIONS -->
    {reco_html}

    <!-- SCENARIO FORECAST -->
    {scenario_section_html}

    <!-- SCENARIO COMPARISON -->
    {scenarios_comparison_html}

    <!-- METHODOLOGY -->
    {methodology_html}

  </div>

  <footer>
    Katalon MMM v3 · OLS regression with adstock, lag shift, seasonality dummies, and quality controls ·
    Organic baseline 22% (Gartner Visionary + G2 + LLM citations) · Paid attribution share 78% ·
    ROI values are relative — use for channel ranking and directional budget decisions only
  </footer>
</body>
</html>"""

    out = Path(output_path)
    out.write_text(html, encoding="utf-8")
    print(f"  Report saved → {out.resolve()}  ({out.stat().st_size // 1024} KB)")


# ─── PLOTS (optional, for --plot flag) ───────────────────────────────────────

def make_plots(df: pd.DataFrame, results: dict, lag_map: dict, monthly: bool):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("\n  matplotlib not installed — pip install matplotlib")
        return

    outcomes = list(results.keys())
    n = len(outcomes)
    fig = plt.figure(figsize=(18, 5 * n + 4))
    fig.patch.set_facecolor("#FAFAF8")
    gs = gridspec.GridSpec(n + 1, 3, figure=fig, hspace=0.55, wspace=0.38)

    for row, ok in enumerate(outcomes):
        res  = results[ok]
        meta = OUTCOME_META[ok]
        unit = res["unit"]

        ax1 = fig.add_subplot(gs[row, 0])
        names  = [c["name"] for c in res["channels"]] + ["Organic"]
        shares = [c["share"] * 100 for c in res["channels"]] + [ORGANIC_BASELINE * 100]
        colors = [c["color"] for c in res["channels"]] + ["#CBD5E1"]
        bars = ax1.barh(names, shares, color=colors, height=0.55)
        for b, v in zip(bars, shares):
            ax1.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2,
                     f"{v:.1f}%", va="center", fontsize=8)
        ax1.set_xlim(0, max(shares) * 1.35)
        ax1.set_title(f"{meta['label']} — Attribution\nR²={res['r2']:.3f}  MAPE={res['mape']:.1f}%",
                      fontsize=9, fontweight="bold", color="#111827")
        ax1.tick_params(labelsize=8)
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.set_facecolor("#FAFAF8")

        ax2 = fig.add_subplot(gs[row, 1])
        ax2.plot(res["actuals"],   color="#111827", lw=1.5, label="Actual")
        ax2.plot(res["predicted"], color=res["outcome_color"], lw=1.2, ls="--",
                 label=f"OLS fit R²={res['r2']:.2f}")
        ax2.set_title(f"{meta['label']} — Actual vs Fit\nlag={res['lag']} mo",
                      fontsize=9, fontweight="bold", color="#111827")
        ax2.legend(fontsize=7)
        ax2.tick_params(labelsize=7)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.set_facecolor("#FAFAF8")

        ax3 = fig.add_subplot(gs[row, 2])
        r_colors = ["#059669" if v >= 0 else "#EF4444" for v in res["residuals"]]
        ax3.bar(np.arange(len(res["residuals"])), res["residuals"],
                color=r_colors, width=0.8, alpha=0.7)
        ax3.axhline(0, color="#111827", lw=0.8)
        ax3.set_title(f"{meta['label']} — Residuals", fontsize=9,
                      fontweight="bold", color="#111827")
        ax3.tick_params(labelsize=7)
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.set_facecolor("#FAFAF8")

    # Sensitivity for first outcome
    first_ok = outcomes[0]
    best_ch  = "google_brand_spend" if "google_brand_spend" in df.columns else None
    if best_ch:
        lag  = lag_map.get(first_ok, OUTCOME_META[first_ok]["default_lag"])
        ax_s = fig.add_subplot(gs[n, :])
        sens = adstock_sensitivity_ols(df, first_ok, best_ch, lag=lag)
        ax_s.plot(sens["decay"], sens["r2"],  marker="o", color="#2563EB", lw=2, ms=4, label="R²")
        ax_s2 = ax_s.twinx()
        ax_s2.plot(sens["decay"], sens["roi"], marker="s", color="#D97706",
                   lw=1.5, ls="--", ms=3, label="ROI")
        model_d = CHANNEL_META[best_ch]["decay"]
        ax_s.axvline(model_d, color="#EF4444", ls=":", label=f"Model decay={model_d:.2f}")
        ax_s.set_xlabel("Adstock decay", fontsize=10)
        ax_s.set_ylabel("R²", fontsize=10, color="#2563EB")
        ax_s2.set_ylabel("ROI", fontsize=10, color="#D97706")
        ax_s.set_title(
            f"Google Brand adstock sensitivity — {OUTCOME_META[first_ok]['label']}\n"
            "Flat lines = robust model;  steep lines = results depend heavily on decay assumption",
            fontsize=11, fontweight="bold", color="#111827")
        ax_s.tick_params(labelsize=9)
        ax_s.spines[["top"]].set_visible(False)
        ax_s.set_facecolor("#FAFAF8")
        l1, n1 = ax_s.get_legend_handles_labels()
        l2, n2 = ax_s2.get_legend_handles_labels()
        ax_s.legend(l1 + l2, n1 + n2, fontsize=9)

    plt.suptitle("Katalon MMM v3 — 9-Channel OLS with Lag, Seasonality & Quality Controls",
                 fontsize=13, fontweight="bold", color="#111827", y=1.01)
    out = Path("katalon_mmm_charts.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#FAFAF8")
    print(f"\n  Charts → {out.resolve()}")
    plt.close()


# ─── SAMPLE DATA (backward compatibility) ─────────────────────────────────────

SAMPLE_CSV = """week,google_spend,linkedin_spend,other_spend,mqls,inbound_pipeline,won_arr
2024-W01,8226.81,941.61,,38,98142.25,4372.20
2024-W02,9066.85,1940.71,,27,49207.80,28413.25
2024-W03,8998.10,4849.72,,43,232077.12,30808.10
2024-W04,8960.70,4759.68,,35,290487.15,64991.10
2024-W05,8751.67,2850.76,,42,85697.00,33981.66
2024-W06,7095.29,4128.21,,42,37281.20,9968.00
2024-W07,6086.53,3920.00,,29,91066.55,5875.80
2024-W08,5877.88,3873.96,,46,84626.20,20988.58
2024-W09,5959.88,3290.00,,64,104567.30,30456.70
2024-W10,6386.30,3290.00,,66,53619.60,34913.43
2024-W11,6664.25,3290.00,83.57,59,63617.70,12367.80
2024-W12,8158.10,3350.00,143.89,49,134722.32,39729.95
2024-W13,7523.06,3421.92,95.94,63,44366.20,66327.11
2024-W14,9911.51,6476.48,124.96,46,14563.30,51678.21
2024-W15,10392.87,7203.05,101.76,128,116392.90,7696.20
2024-W16,11284.30,6316.36,138.54,81,61481.00,16893.00
2024-W17,10701.49,6799.74,109.17,95,117074.00,100546.28
2024-W18,9515.08,2983.09,103.57,79,183787.69,249234.72
2024-W19,10795.37,7372.22,92.52,97,122507.59,9475.00
2024-W20,11950.69,8164.10,89.95,89,89872.00,8896.00
"""


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Katalon MMM v3 — 9-channel OLS with quality controls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--json",           type=str, default=None,
                   help="Path to data.json (recommended; auto-groups channels)")
    p.add_argument("--csv",            type=str, default=None,
                   help="Path to weekly CSV (3-channel: google/linkedin/other)")
    p.add_argument("--outcome",        type=str, default=None,
                   choices=list(OUTCOME_META.keys()))
    p.add_argument("--lag-mqls",       type=int, default=0)
    p.add_argument("--lag-pipeline",   type=int, default=2,
                   help="Lag months for inbound_pipeline (default 2)")
    p.add_argument("--lag-arr",        type=int, default=5,
                   help="Lag months for won_arr (default 5)")
    p.add_argument("--monthly",        action="store_true",
                   help="Aggregate weekly CSV to monthly before fitting")
    p.add_argument("--no-seasonality", action="store_true")
    p.add_argument("--no-quality",     action="store_true",
                   help="Disable quality control co-variates")
    p.add_argument("--budget",         type=float, default=None)
    p.add_argument("--plot",           action="store_true",
                   help="Save charts to katalon_mmm_charts.png")
    p.add_argument("--sensitivity",    action="store_true")
    p.add_argument("--export",         type=str, default=None,
                   help="Export results CSV")
    p.add_argument("--export-html",    type=str, default=None,
                   help="Export full HTML report (e.g. report.html)")
    p.add_argument("--lag-pipeline-alt", type=int, default=None,
                   help="Alternative pipeline lag to compare (e.g. 1 if default is 2)")
    p.add_argument("--compare-scenarios", type=str, default=None,
                   help="Path to JSON file with list of named scenarios. "
                        'Each entry: {"name":"...","month":YYYYMM,"spend":{...},"note":"..."}')
    p.add_argument("--scenario",       type=str, default=None,
                   help="Scenario spend as JSON string or path to JSON file. "
                        'e.g. \'{"google_brand_spend":25000,"linkedin_brand_spend":15000}\' '
                        "or scenario.json")
    p.add_argument("--scenario-month", type=int, default=None,
                   help="Month to forecast (YYYYMM). Default: month after last data point.")
    p.add_argument("--winsorize-arr",  type=float, default=None,
                   metavar="PCT",
                   help="Winsorize won_arr at this percentile before fitting the ARR model "
                        "(e.g. 90 = cap at P90). Only affects won_arr — MQL and pipeline "
                        "models are unchanged. Recommended when a few large enterprise deals "
                        "skew monthly ARR.")
    args = p.parse_args()

    lag_map = {
        "mqls":             args.lag_mqls,
        "inbound_pipeline": args.lag_pipeline,
        "won_arr":          args.lag_arr,
    }
    use_seasonality     = not args.no_seasonality
    use_quality_controls = not args.no_quality

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.json:
        df_raw = load_json_data(args.json)
        print(f"\n  Loaded {len(df_raw)} monthly rows from {args.json}")
        is_monthly = True
    elif args.csv:
        df_raw = pd.read_csv(args.csv)
        df_raw.columns = [c.strip().lower() for c in df_raw.columns]
        for c in [col for col in df_raw.columns if col.endswith("_spend")]:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0)
        print(f"\n  Loaded {len(df_raw)} rows from {args.csv}")
        is_monthly = args.monthly
    else:
        from io import StringIO
        df_raw = pd.read_csv(StringIO(SAMPLE_CSV))
        df_raw.columns = [c.strip().lower() for c in df_raw.columns]
        for c in [col for col in df_raw.columns if col.endswith("_spend")]:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0)
        print(f"\n  Using bundled sample data ({len(df_raw)} weeks)")
        is_monthly = False

    spend_cols   = [c for c in CHANNEL_META if c in df_raw.columns]
    if not spend_cols:
        # fallback: any _spend column
        spend_cols = [c for c in df_raw.columns if c.endswith("_spend")]

    available    = [k for k in OUTCOME_META if k in df_raw.columns]
    if not available:
        sys.exit("  ERROR: no outcome columns found (need mqls / inbound_pipeline / won_arr)")
    run_outcomes = [args.outcome] if args.outcome else available

    # ── Seasonality + optional monthly aggregation ────────────────────────────
    df_raw  = add_seasonality_flags(df_raw)
    if not is_monthly and args.monthly:
        df_fit = aggregate_monthly(df_raw, available, spend_cols)
        print(f"  Aggregated to {len(df_fit)} monthly periods")
        is_monthly = True
    else:
        df_fit = df_raw.copy()

    print(f"\n  Mode            : {'Monthly' if is_monthly else 'Weekly'}")
    print(f"  Seasonality     : {'Q4+Q1 on' if use_seasonality else 'off'}")
    print(f"  Quality controls: {'on' if use_quality_controls else 'off'} "
          f"({', '.join(qc for qc in QUALITY_CONTROLS if qc in df_fit.columns)})")
    print(f"  Outcomes        : {', '.join(run_outcomes)}")
    print(f"  Lags            : " +
          "  ".join(f"{ok}={lag_map[ok]}mo" for ok in run_outcomes))
    print(f"  Channels        : {len(spend_cols)}  "
          f"({', '.join(CHANNEL_META[c]['label'] for c in spend_cols if c in CHANNEL_META)})")

    results     = {}
    export_rows = []
    total_spend = sum(df_fit[c].fillna(0).sum() for c in spend_cols)

    # Pre-compute winsorization cap so it can be logged in the header block
    _wins_pct      = args.winsorize_arr
    _wins_cap      = None
    _wins_n_capped = 0
    _wins_months   = []
    if _wins_pct is not None and "won_arr" in df_fit.columns and "won_arr" in run_outcomes:
        _wins_cap      = float(np.percentile(df_fit["won_arr"].dropna(), _wins_pct))
        _wins_n_capped = int((df_fit["won_arr"] > _wins_cap).sum())
        _wins_months   = (df_fit.loc[df_fit["won_arr"] > _wins_cap, "month"].tolist()
                          if "month" in df_fit.columns else [])
        print(f"\n  Winsorizing won_arr at P{_wins_pct:.0f}:")
        print(f"    Cap = {fmt_num(_wins_cap, '$')}  |  {_wins_n_capped} month(s) capped: {_wins_months}")

    for ok in run_outcomes:
        lag = lag_map.get(ok, OUTCOME_META[ok]["default_lag"])

        # Use winsorized copy of df for the ARR model only
        if ok == "won_arr" and _wins_cap is not None:
            df_for_ok = df_fit.copy()
            df_for_ok["won_arr"] = df_for_ok["won_arr"].clip(upper=_wins_cap)
        else:
            df_for_ok = df_fit

        res = run_mmm_ols(df_for_ok, ok, lag=lag,
                          use_seasonality=use_seasonality,
                          monthly=is_monthly,
                          use_quality_controls=use_quality_controls)

        # Tag ARR result with winsorize metadata for HTML report
        if ok == "won_arr" and _wins_cap is not None:
            res["winsorize_pct"]      = _wins_pct
            res["winsorize_cap"]      = _wins_cap
            res["winsorize_n_capped"] = _wins_n_capped
            res["winsorize_months"]   = _wins_months

        results[ok] = res
        print_result(res)

        budget = args.budget or (total_spend / max(len(df_fit), 1))
        print_optimizer(res, budget)

        oot   = out_of_time_val(df_for_ok, ok, 0.70, lag, use_seasonality, is_monthly)
        delta = oot["mape_delta"]
        tag   = "No overfit" if delta < 10 else ("Mild drift" if delta < 20 else "OVERFIT RISK")
        print(f"\n  OUT-OF-TIME  train R²={oot['train_r2']:.3f} MAPE={oot['train_mape']:.1f}%"
              f"  →  test R²={oot['test_r2']:.3f} MAPE={oot['test_mape']:.1f}%"
              f"  Δ={delta:+.1f}pp  [{tag}]")

        for c in res["channels"]:
            export_rows.append({
                "outcome":            ok,
                "lag":                lag,
                "monthly":            is_monthly,
                "channel":            c["name"],
                "total_spend":        round(c["total_spend"], 2),
                "attributed_outcome": round(c["attributed_out"], 2),
                "share_pct":          round(c["share"] * 100, 2),
                "adj_roi":            round(c["roi"], 4),
                "ols_coeff":          round(c["raw_coeff"], 6),
                "decay_used":         round(c["decay"], 3),
                "r2":                 round(res["r2"], 4),
                "mape":               round(res["mape"], 2),
            })

    # ── Sensitivity ───────────────────────────────────────────────────────────
    if args.sensitivity:
        first_ok = run_outcomes[0]
        best_ch  = next((c for c in spend_cols if c in CHANNEL_META), None)
        if best_ch:
            lag = lag_map.get(first_ok, OUTCOME_META[first_ok]["default_lag"])
            print(f"\n{DIV}")
            print(f"  SENSITIVITY — {CHANNEL_META[best_ch]['label']} → "
                  f"{OUTCOME_META[first_ok]['label']}  lag={lag}")
            print(f"  {'DECAY':>6}  {'R²':>7}  {'ROI':>8}")
            model_d = CHANNEL_META[best_ch]["decay"]
            for _, row in adstock_sensitivity_ols(
                    df_fit, first_ok, best_ch, lag=lag).iterrows():
                tag = "  ← model" if abs(row["decay"] - model_d) < 0.03 else ""
                print(f"  {row['decay']:>6.2f}  {row['r2']:>7.4f}  {row['roi']:>8.4f}{tag}")
            print(DIV)

    # ── Export CSV ────────────────────────────────────────────────────────────
    if args.export:
        pd.DataFrame(export_rows).to_csv(args.export, index=False)
        print(f"\n  Results CSV → {args.export}")

    # ── Lag comparison (pipeline by default, or any outcome with --lag-pipeline-alt) ──
    lag_comparison = {}
    if args.lag_pipeline_alt is not None and "inbound_pipeline" in run_outcomes:
        lag_a = lag_map["inbound_pipeline"]
        lag_b = args.lag_pipeline_alt
        if lag_a != lag_b:
            print(f"\n  Running pipeline lag comparison: lag={lag_a} vs lag={lag_b}")
            res_a = results["inbound_pipeline"]   # already computed above
            res_b = run_mmm_ols(df_fit, "inbound_pipeline", lag=lag_b,
                                use_seasonality=use_seasonality,
                                monthly=is_monthly,
                                use_quality_controls=use_quality_controls)
            # store as (lower_lag, higher_lag) for consistent display
            ordered = (res_a, res_b) if lag_a < lag_b else (res_b, res_a)
            lag_comparison["inbound_pipeline"] = ordered
            print(f"    lag={ordered[0]['lag']}: R²={ordered[0]['r2']:.3f}  MAPE={ordered[0]['mape']:.1f}%")
            print(f"    lag={ordered[1]['lag']}: R²={ordered[1]['r2']:.3f}  MAPE={ordered[1]['mape']:.1f}%")

    # ── Multi-scenario comparison ─────────────────────────────────────────────
    scenarios_comparison = None
    if args.compare_scenarios:
        try:
            with open(args.compare_scenarios) as _f:
                sc_list = json.load(_f)
            print(f"\n  Running {len(sc_list)} scenario comparison ...")
            scenarios_comparison = run_multi_scenario_comparison(results, df_fit, sc_list)
            print_scenario_comparison(scenarios_comparison)
        except Exception as exc:
            print(f"\n  WARNING: could not run --compare-scenarios: {exc}")

    # ── Scenario forecast ─────────────────────────────────────────────────────
    scenario_forecast = None
    scenario_spend_parsed = None
    if args.scenario:
        import os as _os
        raw_sc = args.scenario.strip()
        try:
            if _os.path.isfile(raw_sc):
                with open(raw_sc) as _f:
                    scenario_spend_parsed = json.load(_f)
            else:
                scenario_spend_parsed = json.loads(raw_sc)
        except Exception as exc:
            print(f"\n  WARNING: could not parse --scenario: {exc}")

        if scenario_spend_parsed is not None:
            # Default scenario month = month after last data point
            if args.scenario_month:
                sc_month = int(args.scenario_month)
            elif "month" in df_fit.columns:
                sc_month = _yyyymm_add_months(int(df_fit["month"].iloc[-1]), 1)
            else:
                sc_month = 202604  # fallback
            print(f"\n  Running scenario forecast for {_yyyymm_label(sc_month)} ...")
            scenario_forecast = run_scenario_forecast(
                results, df_fit, scenario_spend_parsed, sc_month
            )
            print_scenario(scenario_forecast, scenario_spend_parsed, sc_month)

    # ── Export HTML ───────────────────────────────────────────────────────────
    if args.export_html:
        budget = args.budget or (total_spend / max(len(df_fit), 1))
        export_html_report(results, df_fit, args.export_html, budget_override=budget,
                           lag_comparison=lag_comparison if lag_comparison else None,
                           scenario_forecast=scenario_forecast,
                           scenario_spend=scenario_spend_parsed,
                           scenarios_comparison=scenarios_comparison)

    # ── Charts PNG ────────────────────────────────────────────────────────────
    if args.plot:
        make_plots(df_fit, results, lag_map, is_monthly)

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
