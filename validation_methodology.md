# Katalon MMM — Validation Guide

> How to test whether the model's numbers can be trusted

---

## Overview

A model that cannot be validated is a model that cannot be improved. This document covers four validation layers — from quick statistical diagnostics to ground-truth channel pause experiments — ordered from easiest to most rigorous.

---

## Layer 1 — In-sample fit diagnostics

These run automatically on every model run. Open the **Validation** page in the dashboard.

### R² (coefficient of determination)

Measures what fraction of outcome variance the model explains.

```
R² = 1 − (SS_residuals / SS_total)
   = 1 − Σ(actual − predicted)² / Σ(actual − mean)²
```

| R² | Interpretation |
|---|---|
| > 0.70 | Good — model explains most variance |
| 0.40–0.70 | Moderate — usable for directional decisions |
| < 0.40 | Weak — treat attribution shares as rough estimates only |

**Important:** R² on monthly B2B SaaS data above 0.80 should trigger suspicion of overfitting, not celebration.

### MAPE (Mean Absolute Percentage Error)

Average prediction error as a percentage of the actual value.

```
MAPE = mean(|actual[t] − predicted[t]| / actual[t]) × 100
```

| MAPE | Interpretation |
|---|---|
| < 20% | Good |
| 20–35% | Moderate |
| > 35% | High — check for missing confounders or structural breaks |

### Residual bias

Whether the model systematically over- or under-predicts.

```
bias = mean(residuals) / mean(actuals) × 100
```

Target: below 10% of the mean. A large positive bias means the model consistently under-predicts (likely missing a positive confounder). A large negative bias means it over-predicts (likely missing a negative shock or spend efficiency decline).

### How to read the residuals chart

- **Random scatter around zero** → model is capturing the main signal; remaining error is noise
- **Trend in residuals** (upward or downward drift) → a systematic factor is missing, likely a strategy change or external event; add a structural break flag
- **Clusters of same-sign residuals** (several red or green bars in a row) → same diagnosis; add a flag column for that period
- **Single large spike** → one-off event (major deal, product launch, PR moment); consider excluding that month or adding a one-month dummy

---

## Layer 2 — Out-of-time (OOT) validation

Trains the model on the first 70% of months, then tests on the remaining 30%. The split point is adjustable in the Validation page.

### How it works

```
train_set = months[0 : split_idx]
test_set  = months[split_idx : end]

train_model = OLS(train_set)
test_model  = OLS(test_set)

compare: train_MAPE vs test_MAPE
```

### Interpreting the result

| Test MAPE − Train MAPE | Verdict |
|---|---|
| < 10pp | No overfit — model generalises |
| 10–20pp | Mild drift — model may be partially overfitting to 2024 patterns that shifted in 2025 |
| > 20pp | Overfit risk — reduce model complexity (fewer channels, longer lags) or add structural break flags |

### Common cause of OOT failure

Katalon's channel mix changed substantially between 2024 and 2025 (Google Other dominated early; Google Capture ramped up mid-2024; LinkedIn ABM became significant in H2 2024; Meta launched in Q4 2024). If the training window predates these changes, the test window will show high MAPE. Solution: add a strategy break flag at the date of the major shift.

---

## Layer 3 — Adstock sensitivity sweep

Tests whether the model's ROI conclusions depend heavily on the decay assumption.

### How it works

For each channel, the model sweeps adstock decay from 0.10 to 0.85 in steps of 0.05, refits OLS at each value, and records R² and ROI. The Validation page plots R² vs decay for all channels.

### Interpreting the chart

- **Flat lines** → the channel's ROI estimate is stable regardless of the decay assumption. This is the ideal result — conclusions are robust.
- **Steep lines** → the channel's ROI changes significantly with decay. Be more cautious about acting on those numbers. Tighten the decay assumption by examining time-to-conversion data in your CRM.
- **The marked point** → the decay value currently used in the model. It should sit near the flat part of the curve if the model is well-specified.

### Stability threshold

A range (max ROI − min ROI across the sweep) below 0.15 is considered stable for R². For ROI itself, a coefficient of variation below 30% across the sweep is acceptable.

---

## Layer 4 — Incremental holdout test (ground truth)

The only way to truly validate the model is to deliberately pause a channel and measure the outcome drop. Statistical validation can tell you the model fits historical data well; only a holdout test can tell you whether the attribution is causally correct.

### Protocol

**Step 1: Choose the channel to pause**

Start with LinkedIn (lower business disruption risk than Google). Do not pause Google Capture — the MQL impact would be too large and too fast to safely absorb.

**Step 2: Set the measurement window**

Pause for 4–6 consecutive months. Shorter pauses are too noisy to measure; longer pauses risk pipeline damage.

**Step 3: Record the model's prediction**

Before pausing, note the model's current MQL attribution share for LinkedIn. This is the predicted drop. For example, if LinkedIn drives 18% of MQLs and your monthly average is 280 MQLs, the predicted drop is ~50 MQLs/month.

**Step 4: Measure the actual drop**

Compare MQL volume during the pause period to the 3-month pre-pause average (seasonality-adjusted). The actual drop should be compared to the predicted drop.

**Step 5: Calculate accuracy ratio**

```
accuracy_ratio = actual_mql_drop / predicted_mql_drop
```

| Accuracy ratio | Verdict |
|---|---|
| 0.8–1.2 | Model validated — attribution is approximately correct |
| 0.5–0.8 | Model over-attributes to LinkedIn — reduce LinkedIn efficiency adjustment |
| 1.2–1.5 | Model under-attributes to LinkedIn — increase LinkedIn efficiency adjustment |
| < 0.5 or > 1.5 | Model is significantly miscalibrated — review decay parameters and channel groupings |

**Step 6: Confirm recovery**

Re-enable LinkedIn spend and track whether MQLs recover within the model's predicted lag window (~2–4 months for LinkedIn Brand, ~1–2 months for LinkedIn Capture). Recovery confirmation doubles the confidence in the attribution direction.

### What to do with the result

Update the `efficiency_adj` parameter for the tested channel based on the accuracy ratio. If the model predicted 50 MQL drop and actual was 30, LinkedIn's efficiency is overstated by ~40%. Reduce `linkedin_brand_spend.effAdj` and `linkedin_capture_spend.effAdj` proportionally.

---

## Validation cadence recommendation

| Validation type | Frequency | Trigger |
|---|---|---|
| In-sample R² / MAPE | Every model run | Automatic |
| Out-of-time split | Monthly | After adding new months of data |
| Sensitivity sweep | Quarterly | After any channel mix change |
| Incrementality test | Annually | Schedule into marketing calendar |
| Full model recalibration | Annually | After 12+ new months of data or major strategy change |

---

## Validation thresholds summary

| Metric | Target | Acceptable | Investigate |
|---|---|---|---|
| R² | > 0.70 | 0.40–0.70 | < 0.40 |
| MAPE | < 20% | 20–35% | > 35% |
| Residual bias | < 10% of mean | 10–25% | > 25% |
| OOT MAPE delta | < 10pp | 10–20pp | > 20pp |
| Sensitivity R² range | < 0.15 | 0.15–0.30 | > 0.30 |
| Holdout accuracy ratio | 0.8–1.2 | 0.6–1.4 | < 0.6 or > 1.4 |

---

## Appendix — OLS normal equations

The model solves:

```
β = (XᵀX)⁻¹ Xᵀy
```

Where X is the design matrix (intercept + adstocked spend columns + seasonality dummies + break flags) and y is the lag-shifted outcome vector. The implementation uses Gaussian elimination on the augmented matrix `[XᵀX | Xᵀy]` rather than explicit matrix inversion, which is numerically more stable for near-singular cases.

The model clamps all negative spend coefficients to zero before computing attribution shares. A negative coefficient means the OLS fit found a negative spend→outcome relationship, which is economically implausible for a spend that the business chose to run. The most common causes are multicollinearity (two channels always moving together) or omitted variable bias (outcome driven by something unrelated to spend). These channels receive 0% attribution share and should be investigated before the next model run.
