# Katalon Media Mix Model — Methodology

> Version 2 · OLS regression · Monthly grain · Last updated March 2026

---

## Overview

This document describes the full methodology of Katalon's Marketing Mix Model (MMM): what the model does, how it works mathematically, what assumptions it makes, what context adjustments are applied, and where it is likely to be wrong.

The model answers one question: **given our historical monthly spend across channels, how much of each outcome (MQLs, Pipeline, Won ARR) can we attribute to each channel, and what is the ROI?**

---

## The four-step process

### Step 1 — Adstock transformation

Raw monthly spend is a poor predictor of outcomes because advertising has **carry-over effects**. An impression seen this month still influences a purchase two months later. Adstock smooths the spend series using exponential decay:

```
adstock[t] = spend[t] + decay × adstock[t−1]
```

- `decay = 0.0` → no carry-over (each month is independent)
- `decay = 0.9` → very long memory (most prior spend still counts)

The decay parameter encodes how quickly a channel's effect fades. Brand advertising decays slowly (prospects remember it for months). Demand capture advertising (PPC, competitor targeting) decays quickly (high-intent clicks convert fast or not at all).

**Katalon decay parameters by channel:**

| Channel | Base decay | Adjustment | Final | Rationale |
|---|---|---|---|---|
| Google Brand | 0.30 | +0.10 | 0.40 | Brand awareness lingers; thought leadership has long memory |
| Google Capture | 0.30 | −0.10 | 0.20 | PPC/competitor clicks convert fast; AI Overview disruption shortens funnel |
| Google Other | 0.30 | 0 | 0.30 | Neutral baseline |
| LinkedIn Brand | 0.45 | +0.15 | 0.60 | Thought leader content has very long decision influence |
| LinkedIn Capture | 0.45 | +0.05 | 0.50 | Solution content needs time to move decision-makers |
| LinkedIn ABM | 0.45 | +0.10 | 0.55 | Account-level targeting requires sustained multi-touch |
| LinkedIn Other | 0.45 | 0 | 0.45 | Neutral baseline |
| Meta | 0.30 | −0.05 | 0.25 | Short attention span; mostly retargeting use |
| Other | 0.30 | 0 | 0.30 | Neutral; insufficient data to adjust |

The decay assumption is the single most important model parameter. The sensitivity sweep in the Validation page tests whether ROI conclusions are stable across decay values 0.10→0.85. **Flat lines = robust model. Steep lines = treat conclusions with lower confidence.**

---

### Step 2 — Lag shifting

Spend in month T does not produce ARR in month T. There is a structural lag:

```
outcome_aligned[t] = outcome[t + lag]
```

By shifting the outcome forward by `lag` months, the model regresses spend against the outcomes that spend actually caused, not outcomes that were already in flight.

**Default lags:**

| Outcome | Lag | Rationale |
|---|---|---|
| MQLs | 0 months | Demand capture (PPC) converts near-instantly |
| Inbound Pipeline | 1 month | MQL → SQL → opportunity takes ~4 weeks |
| Won ARR | 3 months | Enterprise sales cycle; conservative mid-range of 1–2 quarters |

Lags are adjustable in the Validation page. For companies with longer cycles (e.g. Katalon's largest enterprise accounts), try ARR lag = 4–5 months.

---

### Step 3 — OLS regression

After adstock transformation and lag shifting, the model fits **Ordinary Least Squares regression**:

```
outcome[t] = β₀
           + β_g_brand   × adstock_google_brand[t]
           + β_g_capture × adstock_google_capture[t]
           + β_g_other   × adstock_google_other[t]
           + β_li_brand  × adstock_linkedin_brand[t]
           + β_li_cap    × adstock_linkedin_capture[t]
           + β_li_abm    × adstock_linkedin_abm[t]
           + β_li_other  × adstock_linkedin_other[t]
           + β_meta      × adstock_meta[t]
           + β_other     × adstock_other[t]
           + β_q4        × q4_flag[t]
           + β_q1        × q1_flag[t]
           + β_sv2       × strategy_v2_flag[t]   ← if provided
           + ε[t]
```

Solved via normal equations: `β = (XᵀX)⁻¹Xᵀy`

**Why OLS, not correlation?** Correlation is dimensionless — it cannot produce predictions in outcome units. OLS coefficients are in outcome units (MQLs per dollar of adstocked spend), which gives properly-scaled predictions, a real R², and directly interpretable ROI numbers.

**Seasonality controls:** Q4 (Oct–Dec) and Q1 (Jan–Feb) dummy variables absorb enterprise budget cycle effects. Without them, high-budget Q4 spend appears correlated with Q4 pipeline even if the causal relationship is much weaker.

**Structural break flags:** If you changed campaign strategy on a known date, add a binary column (`strategy_v2` = 0 before, 1 from that month onwards). The OLS model estimates a coefficient for each flag, absorbing the level shift in outcomes caused by the strategy change.

---

### Step 4 — Attribution and ROI

Only positive OLS coefficients are used for attribution (negative spend→outcome relationships are economically nonsensical and typically indicate multicollinearity or insufficient data).

Each positive coefficient is multiplied by a **context efficiency multiplier**:

```
adj_coeff[ch] = max(0, β_ch) × (1 + efficiency_adj[ch])
```

Attribution shares are then normalised to the paid attribution share (78%):

```
share[ch] = (adj_coeff[ch] / Σ adj_coeffs) × 0.78
attributed_outcome[ch] = total_outcome × share[ch]
ROI[ch] = attributed_outcome[ch] / total_spend[ch]
```

The remaining 22% is assigned to organic baseline (Gartner Visionary positioning, G2/review presence, LLM citation traffic).

**ROI values are relative, not absolute.** Use them for channel ranking and budget allocation decisions. Do not use them for precise revenue accounting.

---

## Context efficiency adjustments

These multipliers adjust raw OLS coefficients to reflect Katalon's specific market conditions in 2024–2026. They are research-backed but estimated, not measured.

| Channel | Adjustment | Basis |
|---|---|---|
| Google Brand | −10% | AI Overviews reduce branded search CTR; LLM citations partially substitute |
| Google Capture | −15% | AI Overviews on 57% of SERPs; organic CTR −58% YoY; top-of-funnel test automation queries intercepted by LLMs |
| Google Other | −10% | Display/other Google channels partially affected by same LLM disruption |
| LinkedIn Brand | +10% | Thought leader content reaches VPs Eng and QA Directors; Playwright 45% adoption means buyers need differentiation signals |
| LinkedIn Capture | +8% | Solution content is more critical vs free alternatives (Playwright, Cypress) |
| LinkedIn ABM | +12% | Account-based targeting at enterprise accounts has higher signal-to-noise; multi-stakeholder buying committee requires sustained presence |
| Meta | 0% | Insufficient history; treat as neutral until more data |
| Other | 0% | Neutral |

**Organic baseline raised to 22%** (from typical B2B SaaS baseline of 18%) because:
- Katalon named Gartner Visionary 2025 (Magic Quadrant for AI-Augmented Software Testing Tools)
- G2 review presence drives high-intent dark-funnel pipeline
- LLM citations convert 4.4× better than organic search and are not tracked in ad platforms

---

## Assumptions and parameters

| Parameter | Value | Status |
|---|---|---|
| Organic baseline | 22% | Estimated — raise to 25% if LLM citation traffic grows further |
| Paid attribution share | 78% | Estimated — industry range 70–85% for B2B SaaS |
| Monthly aggregation | Default on | Recommended — weekly data too noisy for pipeline/ARR |
| MQL lag | 0 months | Stable |
| Pipeline lag | 1 month | Stable |
| ARR lag | 3 months | Stable — adjust to 4–5 for enterprise-heavy quarters |
| Q4/Q1 seasonality | Included | Stable |
| Min observations | 4 rows | Model falls back to equal-share attribution below this |

---

## Channel grouping rationale

The raw data contains 14+ granular campaign-type columns. These were grouped into 9 model channels to avoid:

1. **Sparse columns** — individual campaign types with $0 in most months have no signal
2. **Multicollinearity** — columns that always move together (e.g. all Google campaigns ramping up simultaneously) confuse OLS coefficient estimation

**Grouping logic:**

```
google_brand_spend    = google_brand + google_thought_leader
google_capture_spend  = google_ppc + google_competitor + google_solution
                      + google_soqr + google_retargeting + google_abm
google_other_spend    = google_other

linkedin_brand_spend    = linkedin_brand + linkedin_thought_leader
linkedin_capture_spend  = linkedin_solution + linkedin_soqr + linkedin_ppc
                        + linkedin_competitor + linkedin_retargeting
linkedin_abm_spend      = linkedin_abm
linkedin_other_spend    = linkedin_other

meta_spend   = meta
other_spend  = other
```

**Do not include both grouped totals and sub-columns in the same model.** This creates perfect multicollinearity and makes OLS unsolvable.

---

## Known limitations

### What this model cannot do

**No causal inference.** Correlation ≠ causation. A channel that correlates with pipeline could be following it rather than driving it. LinkedIn spend rising in Q3 could reflect the team having more budget in a good quarter, not LinkedIn causing the pipeline.

**No interaction effects.** The model treats channels independently. In reality, LinkedIn brand exposure likely amplifies Google conversion rates — a prospect sees a thought leadership post, then Googles Katalon. This synergy is unmodelled and means Google's standalone ROI is probably understated.

**No saturation curves.** The model assumes linear returns to spend. In practice, doubling Google Capture spend will not double MQLs — diminishing returns set in above a certain threshold. A full Bayesian MMM with Hill transformation would capture this ceiling.

**No external confounders.** Product launches, PR events, partner announcements, pricing changes, and competitor actions all affect outcomes and are not included unless you add them as flag columns.

**Short history.** With 27 months of data and 9 spend channels plus seasonality, the model has limited degrees of freedom. Coefficient estimates will tighten significantly as you accumulate 36–48 months.

### When to trust the model less

- R² below 0.40 — the model is explaining less than 40% of outcome variance; conclusions are noisy
- Sensitivity sweep shows steep lines — ROI conclusions are sensitive to the decay assumption
- Out-of-time test MAPE is >10pp above train MAPE — the model may be overfitting
- A channel has near-zero spend for more than 6 consecutive months — its coefficient is unreliable

### When to trust the model more

- R² above 0.60 and MAPE below 25%
- Sensitivity sweep shows flat lines across decay 0.15–0.65
- Out-of-time MAPE delta under 10pp
- The model's predicted MQL drop from a LinkedIn pause matches the actual drop within ±20%

---

## Upgrade path

The current model is a solid foundation. In priority order, these upgrades would meaningfully improve accuracy:

1. **Strategy break flags** — add binary columns for each major campaign restructure date. Each flag absorbs a structural shift in the spend→outcome relationship.
2. **More history** — every additional quarter strengthens coefficient estimates. Target 36+ months.
3. **Audience/segment split** — separate enterprise vs SMB spend if available. Different sales cycles mean different lags.
4. **Bayesian MMM with Hill curves** — replaces OLS with a probabilistic model that captures diminishing returns and uncertainty intervals around every attribution estimate. Recommended once you have 36+ months of clean data.
5. **Incrementality tests** — pause one channel for 4–6 weeks, measure actual MQL drop, compare to model prediction. This is the only ground-truth validation and should be run at least once per year.
