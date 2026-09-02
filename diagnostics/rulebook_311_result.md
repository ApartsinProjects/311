# Per-organization convention mining: 311 result (2026-09-02)

**Claim demonstrated:** an LLM can mine a city's own *filing conventions* (the part of the label the
text does not entail) from a small labeled sample of that city, and injecting those validated rules
improves the city's labeling, beating a placebo, where conventions exist, and doing no harm where they
do not.

Method: base-classify (gpt-4o-mini zero-shot, 14 harmonized classes) -> per city, iterative LLM rule
mining over confusion pairs (predicted X but filed Y) -> keep a candidate only if it raises accuracy on
a held-out validation slice -> apply validated keyword-gated remaps to that city's frozen test set.
Controls: placebo = each rule fires but points to a random WRONG label (avg of 20 draws); bootstrap
95% CI on the rules-minus-no-rules test delta. Code: `per_city_rulebook.py` (mining in `rulebook_v3.py`).
Mining sample = 1200 rows/city; test = frozen 500/city.

## Result (test macro-accuracy)
| city | rules | no-rules | rules | placebo | delta | 95% CI |
|---|---|---|---|---|---|---|
| Richmond | 3 | 0.802 | 0.916 | 0.788 | +0.114 | (0.082, 0.148) sig |
| Auburn_WA | 2 | 0.730 | 0.754 | 0.730 | +0.024 | (0.012, 0.038) sig |
| Honolulu | 2 | 0.750 | 0.762 | 0.751 | +0.012 | (0.004, 0.022) sig |
| Bloomington | 1 | 0.876 | 0.888 | 0.874 | +0.012 | (0.002, 0.022) sig |
| BatonRouge | 2 | 0.844 | 0.864 | 0.816 | +0.020 | (-0.004, 0.046) |
| Gainesville | 1 | 0.754 | 0.750 | 0.748 | -0.004 | (-0.012, 0.004) |
| SanFrancisco | 0 | 0.942 | 0.942 | 0.942 | 0 | -- |

Mean: no-rules 0.814 -> rules 0.839 (+2.6 pts); placebo 0.807 (real beats placebo by +3.3). Placebo drops
BELOW no-rules, so the gain is rule CONTENT, not extra prompt text. 5/7 cities improve; 4 significant.

## Example mined rules (interpretable, institution-specific)
- Richmond / BatonRouge: IF classifier says Trees_Vegetation AND text mentions tree debris / limbs /
  yard waste / missed pickup -> file as Waste_Sanitation. (yard-waste-is-sanitation convention)
- Auburn_WA: IF classifier says Homelessness AND text mentions camp / garbage / encampment ->
  file as Waste_Sanitation. (encampment-cleanup-is-sanitation convention)

## Why this is the honest framing (see [[311-rulebook-framing]])
- The SAME convention (tree debris -> Waste) is mined in Richmond and BatonRouge but is ABSENT in
  Gainesville (which files trees as Trees). Same content, different filing -> conventions are
  organization-specific. This is why a shared/transferred rulebook fails (confirmed separately) but
  per-organization mining works: rules that help in-org hurt cross-org because they are conventions,
  not semantic facts.
- Effect size varies a lot by city (Richmond carries the headline); the method needs each city's own
  labels; 2/7 cities have no mineable convention. State all three plainly.

## Prior art to cite (not our novelty)
Mechanism (LLM mines rules): ERGO (arXiv 2607.20497, closest; no per-rule validation), APE (2211.01910),
MIPRO (2406.11695, instructions>few-shot for conditional rules), Min et al. 2022 (2202.12837).
Our differentiators: per-rule statistical validation + coverage; institution-CONVENTION target;
transfer-failure-proves-convention finding on a real many-class administrative benchmark.
