# Katalon Media Mix Model — README

> A complete know-how guide for building, running, and validating a B2B SaaS MMM.
> Written for future agents and analysts who need to reproduce or extend this model.

---

## What this is

A **Media Mix Model (MMM)** that answers: *given our historical monthly spend across channels, how much of each outcome (MQLs, Pipeline, Won ARR) can we attribute to each channel, and what ROI does each channel produce?*

The model uses **OLS regression with adstock-transformed spend, lag-shifted outcomes, and seasonality controls**. It runs in two forms:
- `mmm_katalon_v3.html` — interactive browser dashboard (no server required)
- `katalon_mmm.py` — Python script with full diagnostics, charts, and CSV export

---

## Files in this package

| File | Purpose |
|---|---|
| `mmm_katalon_v3.html` | Interactive dashboard — open in any browser |
| `katalon_mmm.py` | Python implementation with CLI, diagnostics, sensitivity analysis |
| `methodology.md` | Full model methodology, assumptions, and parameters |
| `validation_methodology.md` | How to validate the model statistically and experimentally |
| `ai_insights.md` | AI-generated insights framework and current findings |
| `README.md` | This file — step-by-step build and validation guide |

---

## Quick start

### Browser dashboard
```
1. Open mmm_katalon_v3.html in any modern browser
2. The model loads with Katalon's real data automatically
3. Navigate pages: Overview → Attribution → Trends → Optimizer → Validation → AI Insights
4. To load your own data: go to Data Input, paste CSV or JSON, click Run Analysis
```

### Python script
```bash
pip install pandas numpy matplotlib scipy

# Run with bundled data
python katalon_mmm.py

# Run with your own data and generate charts
python katalon_mmm.py --csv your_data.csv --plot --sensitivity

# Tune lags and export results
python katalon_mmm.py --monthly --lag-arr 3 --lag-pipeline 1 --export results.csv

# Full options
python katalon_mmm.py --help
```

---

## Step-by-step: how to build this model from scratch

### Step 1 — Define the business question

Before touching data, answer:
- What outcomes do we care about? (MQLs, pipeline, closed revenue)
- What channels do we spend on? (Google, LinkedIn, Meta, etc.)
- What time horizon makes sense? (need 18+ months; 36+ months is better)
- What grain is right? (monthly is almost always better than weekly for B2B SaaS)

**Rule:** Monthly aggregation smooths the noise from lumpy deals and irregular campaign pacing. Only use weekly if MQL volume is very high (500+/week) and you need that resolution.

---

### Step 2 — Collect and structure the data

**Minimum required columns:**
```
month          — YYYYMM, YYYY-MM, YYYY-Mxx, or YYYY-Www (auto-detected)
*_spend        — at least one column ending in _spend
mqls           — and/or inbound_pipeline and/or won_arr
```

**Recommended structure:**
```
month, google_brand_spend, google_capture_spend, google_other_spend,
       linkedin_brand_spend, linkedin_capture_spend, linkedin_abm_spend, linkedin_other_spend,
       meta_spend, other_spend,
       mqls, inbound_pipeline, won_arr
```

**Critical rules:**
1. **Never include both a grouped total and its sub-columns.** If you have `google_spend` = sum of all Google, and also `google_brand_spend`, `google_capture_spend`, etc., include only one level. Including both causes perfect multicollinearity and breaks OLS.
2. **Group sparse columns.** Any column that is zero or null in more than 30% of rows will produce unreliable coefficients. Merge it into a broader bucket.
3. **Zero vs null.** Zero means "we spent nothing and chose not to." Null means "we don't have this data." The model treats both as zero — make sure your nulls really are zeros.

**Grouping strategy for granular ad platform data:**

```
Google brand     = google_brand + google_thought_leader
Google capture   = google_ppc + google_competitor + google_solution
                 + google_soqr + google_retargeting + google_abm
Google other     = google_other (display, YouTube, etc.)

LinkedIn brand   = linkedin_brand + linkedin_thought_leader
LinkedIn capture = linkedin_solution + linkedin_soqr + linkedin_competitor
                 + linkedin_ppc + linkedin_retargeting
LinkedIn ABM     = linkedin_abm
LinkedIn other   = linkedin_other
```

---

### Step 3 — Add structural context columns

**Seasonality flags** (auto-added by the model):
```
q4_flag = 1 if month is Oct/Nov/Dec, else 0
q1_flag = 1 if month is Jan/Feb, else 0
```
These absorb enterprise budget cycles. Without them, Q4 spend appears more effective than it is (because Q4 pipeline was already high before the spend).

**Strategy break flags** (add manually):
```
strategy_v2 = 0 for all months before the change, 1 from the change month onwards
```
Add one flag per major structural change: new campaign objective, major targeting overhaul, significant budget reallocation between channels, new attribution model in your ad platforms.

For Katalon: a break flag around Oct–Nov 2024 (LinkedIn restructure + Meta launch) would significantly improve model fit.

```json
{"month": "202410", ..., "strategy_v2": 0}
{"month": "202411", ..., "strategy_v2": 1}
{"month": "202412", ..., "strategy_v2": 1}
```

---

### Step 4 — Set adstock decay parameters

Adstock decay encodes how quickly a channel's effect fades. Higher decay = longer memory.

**Framework for setting decay:**

| Channel type | Decay range | Rationale |
|---|---|---|
| Brand / awareness | 0.50–0.70 | Impressions remembered for months |
| Thought leadership | 0.55–0.75 | Long-form content has extended influence |
| Demand capture (PPC) | 0.15–0.30 | High-intent clicks convert fast or not at all |
| ABM / account targeting | 0.45–0.65 | Multiple touches over weeks/months |
| Retargeting | 0.20–0.35 | Warm audience; conversion window is medium |
| Social / Meta | 0.20–0.40 | Short attention span; relatively fast conversion |

**How to validate your decay choice:** Run the sensitivity sweep (`--sensitivity` flag in Python, or the Validation page in the dashboard). If R² peaks at the decay value you've chosen, it's well-calibrated. If R² peaks much lower or higher, update the decay parameter.

---

### Step 5 — Set outcome lag parameters

Lag encodes the delay between spend and observable outcome.

**Framework:**

```
MQL lag      = time from campaign impression to form fill
             = typically 0–1 months for demand capture
             = 1–3 months for brand campaigns

Pipeline lag = MQL lag + MQL-to-opportunity progression
             = typically 1–2 months for B2B SaaS

ARR lag      = Pipeline lag + sales cycle
             = typically 2–5 months for mid-market
             = 4–8 months for enterprise
```

**How to determine your actual lags:**
1. Pull a sample of 50–100 closed-won deals from your CRM
2. For each deal, find the first attributed marketing touch (earliest UTM-tagged session or self-reported channel)
3. Calculate days from first touch to: MQL date, opportunity create date, close date
4. Use the median values converted to months

**For Katalon (1–2 quarter sales cycle, mid-market to enterprise):**
- MQL lag: 0 months (demand capture is near-instant)
- Pipeline lag: 1 month
- ARR lag: 3 months (conservative; try 4 for enterprise-heavy periods)

---

### Step 6 — Set context efficiency adjustments

These are multipliers applied to the raw OLS coefficients before computing attribution shares. They encode research-backed beliefs about channel efficiency that the model cannot derive from data alone (because the data doesn't contain counterfactuals).

**How to set them:**
1. Start all channels at 0 (no adjustment)
2. Apply positive adjustments for channels where you have evidence of above-average effectiveness not captured in spend data (e.g. ABM reaching key decision-makers, thought leadership with high organic amplification)
3. Apply negative adjustments for channels affected by external headwinds (e.g. Google facing AI Overview CTR collapse)
4. Keep adjustments within ±20% unless you have very strong external evidence

**For Katalon 2024–2026:**
```
google_capture_spend:   -15%  (AI Overviews on 57% of SERPs, CTR -58% YoY)
google_brand_spend:     -10%  (LLM citations partially substitute branded search)
linkedin_abm_spend:     +12%  (targeted account reach; high signal-to-noise)
linkedin_brand_spend:   +10%  (critical for Playwright counter-positioning)
linkedin_capture_spend: +8%   (solution content drives differentiation vs OSS)
```

---

### Step 7 — Run the model and check fit

**Target metrics:**

| Metric | Target | Minimum acceptable |
|---|---|---|
| R² | > 0.60 | > 0.35 |
| MAPE | < 25% | < 40% |
| OOT MAPE delta | < 10pp | < 20pp |

**If R² is low:**
1. Check for structural breaks — add strategy flags
2. Check if a major outcome driver is missing (e.g. product launches, pricing changes)
3. Try monthly aggregation if running weekly
4. Check that lags are correctly set — wrong lag is the most common cause of low R²

**If MAPE is high:**
1. Look for outlier months in the residuals chart — consider adding one-month dummy flags
2. Check for lumpy ARR (single large deals) — aggregate further or use pipeline as proxy
3. Reduce the number of spend channels if you have fewer than 24 months of data (rule of thumb: at least 3 observations per model parameter)

---

### Step 8 — Interpret attribution and ROI

**What attribution shares mean:**
```
share[channel] = fraction of total outcome attributed to this channel
               = (adj_coefficient / sum_adj_coefficients) × paid_share (0.78)
```

The 22% organic baseline is assigned to untracked channels: Gartner/analyst relations, G2/review sites, word-of-mouth, LLM citations, direct traffic.

**What ROI means:**
```
ROI[channel] = attributed_outcome / total_spend
```
For MQLs: MQLs per dollar. For Pipeline/ARR: dollars of attributed outcome per dollar spent.

**ROI values are relative, not absolute.** A Google Capture ROI of 3.2x and LinkedIn Brand ROI of 1.8x means Google Capture is more efficient per dollar for MQLs — it does NOT mean LinkedIn Brand is unprofitable. Brand channels have long-term compounding effects not captured in a 27-month window.

**Budget optimizer logic:**
```
alloc[channel] = total_budget × (ROI[channel] / sum(all ROIs))
```
Channels with higher ROI receive a larger budget share. The optimizer is a starting point for discussion, not a hard prescription.

---

### Step 9 — Validate with an incrementality test

Statistical validation confirms the model fits historical data. Only an incrementality test confirms the attribution is causally correct.

**Minimum viable test:**
1. Pause LinkedIn (brand + capture) for 3 months
2. Predict: model says LinkedIn drives X% of MQLs → expect X% drop
3. Measure: actual MQL drop during pause (seasonality-adjusted)
4. Compute: accuracy_ratio = actual_drop / predicted_drop
5. Recalibrate efficiency_adj if ratio is outside 0.8–1.2

---

### Step 10 — Maintain and improve the model

**Monthly:** Add new data rows, rerun model, check if R² and MAPE are stable.

**Quarterly:** Review sensitivity sweep. If a channel's R² curve has shifted (peaked at a different decay), update the decay parameter. Review if new channels need to be added or existing ones split.

**Annually:** Run a full incrementality test. Recalibrate context efficiency adjustments based on test results and updated market research. Consider upgrading to Bayesian MMM with Hill curves once you have 36+ months of clean data.

---

## Data format reference

### CSV format
```csv
month,google_brand_spend,google_capture_spend,google_other_spend,linkedin_brand_spend,linkedin_capture_spend,linkedin_abm_spend,linkedin_other_spend,meta_spend,other_spend,mqls,inbound_pipeline,won_arr
202401,4282.68,11481.27,22998.86,0,2138.96,7191.96,4510.55,0,0,174,731821.72,162566.31
202402,526.13,5134.15,22291.80,0,2693.18,12039.96,579.59,0,0,175,329334.55,63091.88
```

### JSON format
```json
[
  {
    "month": "202401",
    "google_brand_spend": 4282.68,
    "google_capture_spend": 11481.27,
    "google_other_spend": 22998.86,
    "linkedin_brand_spend": 0,
    "linkedin_capture_spend": 2138.96,
    "linkedin_abm_spend": 7191.96,
    "linkedin_other_spend": 4510.55,
    "meta_spend": 0,
    "other_spend": 0,
    "mqls": 174,
    "inbound_pipeline": 731821.72,
    "won_arr": 162566.31
  }
]
```

**Accepted month formats:** `202401`, `2024-01`, `2024-M01`, `2024-W03`

**Adding strategy break flags:**
```json
{"month": "202411", ..., "strategy_v2": 0}
{"month": "202412", ..., "strategy_v2": 1}
```

---

## Python CLI reference

```bash
# Basic run
python katalon_mmm.py

# With your data
python katalon_mmm.py --csv data.csv

# Adjust lags (months)
python katalon_mmm.py --lag-mqls 0 --lag-pipeline 1 --lag-arr 3

# Monthly aggregation (recommended for pipeline/ARR)
python katalon_mmm.py --monthly

# Disable seasonality dummies
python katalon_mmm.py --no-seasonality

# Adstock sensitivity sweep for Google
python katalon_mmm.py --sensitivity

# Generate charts
python katalon_mmm.py --plot

# Export results
python katalon_mmm.py --export results.csv

# Set monthly budget for optimizer
python katalon_mmm.py --budget 80000

# Full analysis
python katalon_mmm.py --csv data.csv --monthly --lag-arr 3 --plot --sensitivity --export out.csv
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| R² = 0 or very close to 0 | Wrong lag — spend and outcome in same period | Set --lag-arr 3 and --lag-pipeline 1 |
| All channels show equal attribution | OLS returned null; fell back to equal-share | Increase data (need 4+ valid rows per outcome after lag) |
| One channel dominates with implausibly high ROI | Multicollinearity with another channel | Check if two channels always move together; merge them |
| Negative OLS coefficient for a major channel | Omitted variable or wrong lag | Add strategy break flag; try different lag |
| MAPE > 50% | Outlier months (single large deal) | Use monthly aggregation; check for one-off pipeline spikes |
| Sensitivity sweep shows steep lines | Decay parameter matters a lot | Tighten decay using CRM time-to-convert data |
| Test MAPE >> Train MAPE | Overfitting or structural break between train/test | Add strategy break flags; reduce number of channels |

---

## Architecture notes for agents

The JavaScript implementation in `mmm_katalon_v3.html` implements the full model client-side with no server or dependencies beyond Chart.js. Key functions:

- `toMonthKey(w)` — normalises any date format to `YYYY-Mxx`
- `aggregateToMonthly(rows, channels, outcomes)` — sums weekly to monthly, adds seasonality flags
- `applyAdstock(series, decay)` — exponential carry-over transformation
- `olsSolve(X, y)` — Gaussian elimination on normal equations `(XᵀX)⁻¹Xᵀy`
- `runOLS(rows, outcomeKey, lag, channels)` — full pipeline: adstock → lag shift → design matrix → OLS → attribution
- `makeEmptyResult(...)` — fallback when OLS fails (returns equal-share attribution)
- `computeAll(monthly, channels, outcomes)` — runs all three outcomes with progressive lag fallback
- `olsSensitivity(rows, outcomeKey, channel, lag)` — sweeps decay 0.10→0.85, returns R² and ROI per step

The Python implementation mirrors this exactly with additional features: matplotlib charts, out-of-time validation split, CLI argument parsing, CSV export.

---

## Extending the model

### Adding a new channel
1. Add the spend column to your data (e.g. `youtube_spend`)
2. Add a colour to `CH_COL` in the HTML
3. Add context adjustments to `CTX`
4. No other changes needed — the model auto-detects `*_spend` columns

### Adding a new outcome
1. Add the outcome column to `OUTCOME_META` with label, unit, baseDecay, defaultLag, fmt
2. Add the column to your data
3. The model will auto-detect and include it in all pages

### Upgrading to Bayesian MMM
When you have 36+ months of data, the logical upgrade is a full Bayesian MMM using PyMC or LightweightMMM. The current OLS model's parameter estimates (adstock decay, lags, efficiency adjustments) become strong informative priors for the Bayesian model — so this work is not wasted.
