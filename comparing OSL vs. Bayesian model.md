V1 vs V2: Conclusions & Recommendations
The headline: V1 OLS wins on fit quality, V2 wins on honesty
Outcome	V1 R²	V2 R²	Verdict
MQLs	0.855	0.782	V1 better, both usable
Inbound Pipeline	0.703	0.383	V1 significantly better
Won ARR	0.348	−0.162	V1 less bad; both weak
Why V1 fits better (and why that's partly a cheat)
V1 OLS uses manually tuned, fixed adstock decay rates in CHANNEL_META — e.g. Google Capture decay=0.25, LinkedIn Brand decay=0.50. Those were calibrated to explain historical variance. That's domain knowledge baked in.

V2 Bayesian starts from scratch and tries to learn decay from 46 observations across 9 channels. With that data-to-parameter ratio, the posterior stays close to the prior (Beta(2,2) → mean 0.5 decay for every channel). It can't recover the optimal decays V1 already knows.

So V1's better R² isn't evidence it's a more truthful model — it reflects pre-loaded knowledge V2 doesn't have.

What V2 uniquely tells you (that V1 can't)
ROI uncertainty on Pipeline (the most actionable model):

Channel	V2 ROI mean	80% CI
Meta	11.6x	[1.2x – 26x] → very uncertain
LinkedIn Capture	10.6x	[1.4x – 22.6x] → uncertain
Google Brand	7.7x	[0.9x – 17.5x] → uncertain
Google Other	3.1x	[0.2x – 7.7x] → widest uncertainty
Every channel has a wide CI. This is the honest answer: with ~50 monthly observations, you cannot pin down individual channel ROI with precision. V1's point estimates create false confidence.

Recommendations
1. Use V1 OLS for monthly reporting and budget decisions.
Better R², faster, more stable. The hand-tuned parameters are appropriate given the data constraints.

2. Use V2 Bayesian for one thing: communicating uncertainty to stakeholders.
When asked "how confident are you that LinkedIn Capture is better than Google Brand?", V2's answer ("ROI could be anywhere from 1x to 23x for both — we can't distinguish them statistically") is the correct one.

3. Ignore V2 ARR results entirely.
R² = −0.162 means the model predicts ARR worse than just guessing the average. Neither model should be used for ARR budget decisions until you have more data (target: 60+ months).

4. The path to improving V2:
The Bayesian model would get meaningfully better if you either:

Added 2–3 more years of data (patience), or
Supplied tighter priors on adstock decay from platform-specific benchmarks (e.g. "Google Search decay is typically 0.2–0.3 based on industry literature")
Bottom line: V1 for decisions, V2 for humility. Run both quarterly and watch whether the V2 credible intervals narrow as you add more monthly data — that's your signal that the Bayesian model is becoming trustworthy enough to act on.