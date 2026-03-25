# Katalon MMM — Run Manual

> **Last updated:** March 2026 · Model version: v3 · 9-channel OLS

This document explains how to re-run the Marketing Mix Model, tune its settings, add new data, and interpret the outputs. No Python knowledge is required to follow the Quick Start section.

---

## Quick Start

```bash
# Standard run: load latest data, generate HTML report + CSV export
python3 katalon_mmm.py \
  --json data_v2.json \
  --lag-pipeline 1 \
  --lag-pipeline-alt 2 \
  --lag-arr 5 \
  --export-html katalon_mmm_report.html \
  --export results.csv \
  --plot
```

Open `katalon_mmm_report.html` in any browser. It is self-contained — no internet required.

---

## All Command-Line Options

| Flag | Default | Description |
|---|---|---|
| `--json <file>` | — | **Recommended.** Load granular channel data from a JSON file. Channels are auto-grouped (see [Channel Grouping](#channel-grouping)). |
| `--csv <file>` | — | Load a weekly CSV file. Must have columns: `week`, `google_spend`, `linkedin_spend`, `other_spend`. |
| `--lag-mqls <n>` | `0` | Months of lag between MQL-driving spend and MQL capture. Default 0 (near-instant). |
| `--lag-pipeline <n>` | `2` | Months of lag between spend and Inbound Pipeline. Default 2 (~8-week MQL→Opp cycle). |
| `--lag-arr <n>` | `5` | Months of lag between spend and Won ARR. Default 5 (enterprise sales cycle). |
| `--lag-pipeline-alt <n>` | — | Run a **second** pipeline model at this lag and add a side-by-side comparison to the HTML report. Example: `--lag-pipeline-alt 1`. |
| `--outcome <name>` | all | Run only one outcome. Choices: `mqls`, `inbound_pipeline`, `won_arr`. |
| `--monthly` | off | (CSV only) Aggregate weekly CSV data to monthly before fitting. Not needed for JSON input — it is already monthly. |
| `--no-seasonality` | off | Disable Q4 / Q1 seasonality dummies. |
| `--no-quality` | off | Disable quality co-variates (`mql_high_intent_rate`, `mql_non_smb_rate`, `cr_mql_to_sql`). |
| `--budget <amount>` | avg monthly spend | Override the monthly budget used in the budget optimizer. Example: `--budget 100000`. |
| `--sensitivity` | off | Print adstock decay sensitivity table for the first outcome and first channel. |
| `--export <file.csv>` | — | Save per-channel attribution results to a CSV file. |
| `--export-html <file.html>` | — | Save the full stakeholder HTML report. |
| `--plot` | off | Save a multi-panel chart PNG to `katalon_mmm_charts.png`. |

---

## Adding New Monthly Data

1. Open `data_v2.json` (or your data file).
2. Append a new JSON object at the end of the array for each new month. The `month` field must be in `YYYYMM` format (e.g. `202604` for April 2026).
3. Fill in spend columns with the actual amounts. Use `null` for channels that had no activity.
4. Fill in `mqls`, `inbound_pipeline`, `won_arr`, `mql_high_intent_rate`, `mql_non_smb_rate`, `cr_mql_to_sql`.

**Minimum required columns per row:**

```json
{
  "month": 202604,
  "google_brand_spend": 12000,
  "google_competitor_spend": null,
  "google_solution_spend": 8000,
  "google_ppc_spend": 15000,
  "google_other_spend": 20000,
  "linkedin_abm_spend": 5000,
  "linkedin_thought_leader_spend": 4000,
  "linkedin_other_spend": 8000,
  "meta_spend": 3000,
  "other_spend": 2000,
  "mqls": 280,
  "mql_high_intent_rate": 0.95,
  "mql_non_smb_rate": 0.42,
  "cr_mql_to_sql": 0.16,
  "inbound_pipeline": 520000,
  "won_arr": 140000
}
```

Then re-run the Quick Start command above.

---

## Channel Grouping

The JSON file contains granular campaign-level columns. The model automatically groups them into 9 channels:

| Model Channel | JSON source columns |
|---|---|
| `google_brand_spend` | `google_brand_spend` + `google_thought_leader_spend` |
| `google_capture_spend` | `google_ppc_spend` + `google_competitor_spend` + `google_solution_spend` + `google_soqr_annual_report_spend` + `google_retargeting_spend` + `google_abm_spend` |
| `google_other_spend` | `google_other_spend` |
| `linkedin_brand_spend` | `linkedin_brand_spend` + `linkedin_thought_leader_spend` |
| `linkedin_capture_spend` | `linkedin_solution_spend` + `linkedin_soqr_annual_report_spend` + `linkedin_ppc_spend` + `linkedin_competitor_spend` + `linkedin_retargeting_spend` |
| `linkedin_abm_spend` | `linkedin_abm_spend` |
| `linkedin_other_spend` | `linkedin_other_spend` |
| `meta_spend` | `meta_spend` |
| `other_spend` | `other_spend` |

If you rename a source column, update the grouping lists inside `load_json_data()` in `katalon_mmm.py`.

---

## Tuning the Lag

The lag is the most important tuning parameter. It controls *how many months after spend we look for the outcome*.

| Outcome | Default | When to increase | When to decrease |
|---|---|---|---|
| MQLs | 0 | If form fills lag behind the campaign (e.g. event-driven MQLs) | — |
| Inbound Pipeline | 2 | If your CRM shows MQL→Opportunity taking 8–12 weeks | If deals are moving faster (e.g. PLG motion) |
| Won ARR | 5 | For enterprise-heavy quarters with 6–9 month cycles | If a significant portion of ARR closes within 2–3 months |

**How to choose:** run `--lag-pipeline-alt 1` (or other values) and compare R² and MAPE in the HTML report's "Lag Test" section. Pick the lag with better R² — it means spend and outcomes are better aligned at that time distance.

```bash
# Test lag=1 vs lag=2 vs lag=3 for pipeline (run twice, compare HTMLs)
python3 katalon_mmm.py --json data_v2.json --lag-pipeline 1 --lag-pipeline-alt 2 \
  --export-html report_lag1v2.html

python3 katalon_mmm.py --json data_v2.json --lag-pipeline 2 --lag-pipeline-alt 3 \
  --export-html report_lag2v3.html
```

---

## Understanding the Outputs

### R² (coefficient of determination)

How well the model fits historical data.

| R² | Meaning |
|---|---|
| > 0.70 | Strong — model explains most of the variance. Attribution is reliable. |
| 0.40 – 0.70 | Moderate — directional conclusions are valid; specific ROI numbers have wider uncertainty. |
| < 0.40 | Weak — do not make budget decisions based on these numbers alone. |

### MAPE (mean absolute percentage error)

Average prediction error. A MAPE of 15% means predictions are off by ~15% on average.

| MAPE | Meaning |
|---|---|
| < 20% | Good — reliable for planning |
| 20–35% | Moderate — use for directional planning |
| > 35% | High — model is missing something important |

### ROI values

ROI in this model = attributed outcome / total spend. **These are relative numbers, not absolute revenue accounting.**

- Use them for channel *ranking*: a 5x ROI channel is more efficient than a 2x channel.
- Do not use them to predict absolute revenue from a spend increase.
- Channels with negative OLS coefficients receive 0% attribution — this means no statistically detectable relationship at the configured lag, not that the channel has zero value.

### Structural break (pre_2024_flag)

A coefficient that absorbs the strategy and MQL-definition change that happened in January 2024. Its value represents how much higher/lower outcomes were in the 2022–2023 era *after controlling for spend and quality*. It is not attributed to any channel.

### Quality controls

Three variables added to absorb quality drift. Their coefficients are not attributed to spend channels.

| Variable | Meaning |
|---|---|
| `mql_high_intent_rate` | Share of MQLs that are high-intent. Before 2024, MQLs included low-intent — this variable corrects for that definition change. |
| `mql_non_smb_rate` | Share of MQLs from non-SMB companies. Higher = more enterprise. |
| `cr_mql_to_sql` | MQL-to-SQL conversion rate. Higher = better quality MQLs reaching sales. |

---

## Adding a New Structural Break Flag

If Katalon makes a major strategy change (new campaign structure, pricing change, product launch), add a break flag:

1. In `data_v2.json`, add a new column (e.g. `strategy_v3`) set to `0` before the change date and `1` from the change date onwards.
2. In `katalon_mmm.py`, add the column name to the `BREAK_FLAGS` list:

```python
BREAK_FLAGS = ["pre_2024_flag", "strategy_v3"]
```

The model will automatically include it in the OLS design matrix.

---

## Adjusting Channel Parameters

In `katalon_mmm.py`, the `CHANNEL_META` dictionary controls:

| Key | What it does |
|---|---|
| `decay` | Adstock carry-over rate (0 = no memory, 0.9 = very long memory). Brand channels: 0.40–0.60. Capture/PPC channels: 0.20–0.30. |
| `efficiency_adj` | Multiplier applied to raw OLS coefficient before attribution. Reflects Katalon market context (AI Overview headwind, LinkedIn enterprise premium). |

To adjust Google Capture's decay (e.g. if AI Overview disruption worsens):

```python
"google_capture_spend": {
    "decay": 0.15,          # was 0.20, reduce if effect fades faster
    "efficiency_adj": -0.20, # was -0.15, increase penalty
    ...
},
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: no outcome columns found` | JSON/CSV missing `mqls`, `inbound_pipeline`, or `won_arr` | Check column names match exactly (lowercase) |
| `WARNING: only N usable rows` | Lag too large for dataset size | Reduce `--lag-arr` or add more data |
| R² = 1.0 in out-of-time test | Test split too small (< 6 rows after lag) | Ignore OOT result — need more historical data |
| Meta ROI appears very high (>50x) | Only 18 months of Meta data | Treat as noise; Meta attribution is unreliable until 30+ months |
| Pipeline R² drops when increasing lag | Wrong lag for your sales cycle | Run lag comparison with `--lag-pipeline-alt` |
| All channels show 0% attribution | OLS returned all-negative coefficients | Check for multicollinearity (channels always moving together); try `--no-quality` to isolate |

---

## File Reference

| File | Purpose |
|---|---|
| `katalon_mmm.py` | Main model code. Edit here to change channel config, lags, or add features. |
| `data_v2.json` | Current monthly data. Append new rows monthly. |
| `katalon_mmm_report.html` | Latest HTML report. Regenerated on each run. |
| `katalon_mmm_charts.png` | Latest chart PNG. Regenerated with `--plot`. |
| `results.csv` | Latest per-channel results table. Regenerated with `--export`. |
| `MANUAL.md` | This file. |
| `methodology.md` | Full statistical methodology documentation. |
| `ai_insights.md` | Notes on Claude API integration for AI Insights section. |
| `validation_methodology.md` | Out-of-time validation methodology. |

---

## Recommended Monthly Cadence

1. **Data update** — append the new month's row to `data_v2.json`.
2. **Run the model** — use the Quick Start command above.
3. **Review the report** — open `katalon_mmm_report.html`, check R² / MAPE, review attribution changes.
4. **Flag anomalies** — if a channel's coefficient sign flips, check for data errors or structural changes.
5. **Budget planning** — use the Budget Optimizer table for next-month allocation guidance. Apply judgment to Meta recommendations (insufficient history).

Every 6 months:
- Re-test the pipeline lag with `--lag-pipeline-alt` to check if sales cycle has shifted.
- Consider increasing `--lag-arr` if enterprise deal size is growing.
- Re-evaluate `CHANNEL_META` efficiency adjustments based on platform performance reports.
