# Katalon MMM — AI Insights

> Claude-generated strategic insights from the OLS model results
> Model run: March 2026 · 27 months Jan 2024–Mar 2026

---

## How AI insights are generated

The AI Insights page calls the Claude API with:

1. **Model results** — for each outcome (MQLs, Pipeline, ARR): total outcome, total spend, per-channel attribution share, attributed outcome, adjusted ROI, OLS coefficient, adstock decay, R², MAPE
2. **Katalon market context** — AI Overview search disruption, Playwright open-source competition, Gartner Visionary 2025 positioning, LLM citation conversion premium
3. **A structured prompt** requesting 6 sharp insights covering specific analytical angles

The model returns JSON: `[{"title","body","action","type"}]` which renders as cards in the dashboard.

**To regenerate:** Navigate to AI Insights in the dashboard and the API call fires automatically on first visit. Refresh by reloading the page after updating data.

---

## Insight framework

Each insight covers one of these angles:

1. **Google OLS coefficient validity** — is the coefficient positive? Is current spend justified given LLM search disruption?
2. **LinkedIn attribution** — is it working to counter open-source competition (Playwright 45% adoption)?
3. **Model fit interpretation** — what do R² and MAPE mean for decision confidence?
4. **Budget reallocation** — specific $ recommendations based on adjusted ROI ranking
5. **Highest-ROI channel** — why it makes sense for Katalon's specific market position
6. **Forward metric recommendation** — one new data point to add to improve model accuracy next cycle

---

## Key findings from the current data (Jan 2024 – Mar 2026)

### Spend summary

| Channel group | Total spend | Share of budget |
|---|---|---|
| Google (all) | $1,410K | 60% |
| LinkedIn (all) | $719K | 31% |
| Meta | $33K | 1.4% |
| Other | $177K | 7.6% |
| **Total** | **$2,339K** | |

**Google breakdown:** Brand $304K (22%) · Capture $630K (45%) · Other $476K (34%)

**LinkedIn breakdown:** Brand $80K (11%) · Capture $178K (25%) · ABM $121K (17%) · Other $339K (47%)

### Outcome summary

| Outcome | Total | Monthly avg | Peak | Trough |
|---|---|---|---|---|
| MQLs | 7,448 | 276/mo | 402 (Apr 2024) | 168 (Mar 2026) |
| Inbound Pipeline | $12.9M | $476K/mo | $930K (Jan 2026) | $270K (Jun 2024) |
| Won ARR | $4.4M | $164K/mo | $510K (Dec 2024) | $63K (Feb 2024) |

### Notable patterns in the data

**Google Capture ramp (mid-2024):** Spend increased from ~$12K/mo in H1 2024 to ~$35–50K/mo by mid-2025. This coincides with the period when Google Capture's share of total spend grew from ~22% to ~45%. The OLS model will test whether this ramp correlated with MQL volume or was absorbed by the AI Overview headwind.

**LinkedIn strategic shift (H2 2024 → 2025):** LinkedIn ABM dropped from ~$13K/mo in early 2024 to near-zero by mid-2024, then spiked to $21K in July 2025 before falling again. LinkedIn Other grew significantly (to $43K in Dec 2024) then contracted. LinkedIn Brand (thought leader) was absent until Q4 2024 ($7–8K/mo), then expanded to $12K by Jan 2026. This structural change is a strong candidate for a `strategy_v2` break flag around Oct–Nov 2024.

**Meta launch (Oct 2024):** Meta spend began at $1.1K/mo in Q4 2024 and grew to $4–5K/mo through 2025. With only 18 months of data, Meta's OLS coefficient will be uncertain. Treat Meta attribution as directional only.

**MQL decline 2025:** MQL volume peaked at 390–402 in Apr–May 2024, then trended down through 2025 to 168–221 by early 2026, despite increasing total spend. This is the most important signal in the data — it suggests either a demand quality shift, a funnel definition change, or that increased spend is reaching less-qualified audiences. The model will attempt to separate spend contribution from the secular trend, but with only seasonality dummies this is imperfect. A `mql_quality_change` flag or a CRM-sourced MQL quality score column would significantly improve attribution.

**Pipeline spike (Jan 2026: $930K):** January 2026 shows the highest monthly pipeline in the dataset. With ARR lag of 3 months, this would be attributed to spend in Oct–Nov 2025. Google Capture was at $36–32K and LinkedIn Other at $16–18K in those months — both at elevated levels. If the OLS coefficient for these channels is positive in the Pipeline model, this spike partially validates the attribution.

---

## Recommended actions from model insights

These are directional recommendations based on the model. Each should be stress-tested against your CRM data before acting.

### Budget allocation

The budget optimizer uses adjusted ROI weights to recommend monthly allocation. Based on the model's context adjustments:

- **Google Capture** receives the highest spend but a −15% efficiency penalty for AI Overview disruption. If R² is low for the MQL model, this channel may be overallocated relative to its measurable contribution.
- **LinkedIn Brand (thought leader)** receives a +10% efficiency boost. Given the small absolute spend ($80K total over 27 months), there is likely meaningful upside from scaling this — thought leader content has high decay (long memory) and reaches the right personas for Katalon's positioning.
- **LinkedIn ABM** receives a +12% efficiency boost — the highest of any channel. Its highly targeted nature (reaching specific accounts) makes it efficient per dollar even though absolute spend is modest.
- **Google Other** is the largest Google sub-bucket ($476K, 34% of Google budget) with a −10% efficiency adjustment. This channel warrants scrutiny — what campaigns fall into "google_other" and do they have measurable outcome signals in the CRM?

### What to measure next

To improve model accuracy in the next cycle:

1. **Add a strategy break flag** for Oct–Nov 2024 (LinkedIn restructure and Meta launch). This will absorb the structural shift and improve residuals significantly.
2. **Add MQL quality signal** — if your CRM tracks MQL-to-SQL conversion rate monthly, add it as a control variable. The MQL decline trend may be a quality signal, not a volume problem.
3. **Track LLM citation traffic** — set up UTM tracking or use a tool like Profound/Goodie to measure how much pipeline comes from AI-generated citations (ChatGPT, Perplexity, Claude mentioning Katalon). This traffic is currently absorbed into the organic baseline and undercounts the model's ability to attribute it.
4. **Separate enterprise from SMB MQLs** — if you have company size data in your CRM, splitting MQLs into enterprise (500+ employees) and SMB would allow separate lags (enterprise: 2–4 months to MQL, 6–12 months to ARR; SMB: 0–1 month to MQL, 2–4 months to ARR).

---

## Prompt template

The following prompt is sent to the Claude API to generate insights. Customise the market context section when the competitive landscape changes.

```
You are a senior B2B SaaS marketing strategist at Katalon — AI-augmented test 
automation platform competing against free open-source tools (Playwright 45% 
QA adoption, Cypress). Enterprise buyers, 1-2 quarter sales cycles.

MODEL: OLS regression on monthly data with lag shifts and seasonality controls.

MARKET CONTEXT:
- Google AI Overviews on 57% of SERPs, CTR down 58% YoY.
- Playwright at 45% QA adoption; decision-makers need explicit brand education.
- Katalon: Gartner Visionary 2025, G2 leader, TrueTest + StudioAssist AI features.
- LLM citations convert 4.4x better than organic search.

MMM RESULTS:
[model output injected here]

Give 6 sharp insights using actual numbers. Cover:
(1) Google OLS coefficient — is it positive? is current spend justified?
(2) LinkedIn attribution — is it working against open-source competition?
(3) R² and MAPE — how much can we trust these numbers?
(4) Budget reallocation with specific $ amounts
(5) Highest-ROI channel and why it makes sense for Katalon
(6) One forward-looking metric to add to improve accuracy

Return ONLY JSON: [{"title":"...","body":"...","action":"...","type":"positive|warning|neutral"}]
```
