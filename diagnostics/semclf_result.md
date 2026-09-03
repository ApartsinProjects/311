# All-LLM classification matches RAG without a store (Bloomington, 2026-09-03)

**Setting**: classify municipal 311 requests into the city's 18 native categories using ONLY vendor LLM
API calls. No fine-tuning, no local model, no retrieval store, no embeddings at inference.

**Result** (fixed 1500-row test, stratified 2000-example budget, classify=gpt-4o-mini, mine=gpt-4.1):

| method | acc | 95% CI | training | store | vs zero-shot (paired) |
|---|---|---|---|---|---|
| RAG (lexical, k=12) | 0.8713 | (.855,.889) | no | YES | +0.049 |
| mined rulebook + triggers | 0.8620 | (.845,.879) | no | no | +0.040, p=2.2e-07 |
| fine-tuned TF-IDF+LR | 0.8613 | | YES | no | +0.039 |
| mined rulebook alone | 0.8467 | | no | no | +0.025, p=8.0e-04 |
| zero-shot | 0.8220 | (.803,.841) | no | no | - |
| k-shot/class | 0.7780 | | no | no | -0.044 |
| majority baseline | 0.5040 | | | | |

RAG vs ours: delta=+0.0093, McNemar p=0.184 -> NOT significantly different.
Ours vs fine-tuned: +0.0007. Ours vs zero-shot: +0.040 (p=2.2e-07).

## Method (two mined artifacts, both plain text)
1. RULEBOOK: per-class positive/negative descriptions, mined by an error-driven diagnose->edit loop
   (gpt-4.1), each edit accepted only if it improves a ROTATING held-out gate slice by a margin.
2. TRIGGERS: (yes/no question, from_class, to_class). Applied only to items currently predicted as
   from_class, remapping only on YES. Kept only if net-positive (or zero-damage) on held-out data.
Inference: ~1.03 LLM calls/item (1 classify + a trigger check only where a from_class matches).

## Why triggers were necessary (the decisive diagnostic)
The measured convention "missed pickup + recycling -> Trash" (89% supported in data) placed as prompt
TEXT changed 130 predictions and netted ZERO (59 fixed / 59 broken); 68% of the changed items contained
no missed-pickup wording at all, and it perturbed unrelated classes. The same convention as a
TRIGGER-GATED rule: fix 8 / broke 0. A rule in a prompt is a suggestion; it must be a conditional remap.

## Bugs found and fixed along the way (each changed the numbers materially)
* verbatim answer-key leakage inflated RAG (37% of test appeared verbatim in the budget)
* unequal labeling budgets (RAG 2k vs others 38k)
* batch gating: 8 edits accepted/rejected together -> one inverted rule rode in (b=1000: 0.719)
* gate overfitting: fixed 200-item slice reused for ~40 decisions -> rotating slice + margin (b=2000: 0.821 -> 0.847)
* non-stratified budgets: 5 of 18 classes absent at b=200
* independent CIs instead of paired tests (paired resolves RAG vs zero-shot at p=1e-07)

## Negative results (all measured, all reproducible)
* classical multiclass decompositions do NOT transfer: OvA 0.770, OvO 0.775, hierarchical 0.747,
  ECOC 0.708, all at or below flat 0.772. Confusion-driven grouping improved router recall to 0.957
  but still lost (0.829 vs flat 0.847): a routing stage adds an unrecoverable error mode without a
  compensating gain, because the flat LLM call is already strong.
* a zero-shot-failure ROUTER is at chance (precision 0.178 vs 0.176 base rate).
* b=200 mining is unreliable: only ~60 held-out items, so the gate cannot select rules (0.667-0.843
  across runs). Rule mining needs a minimum budget.

## Error ceiling
Perfect memorization of repeated texts = 0.994, but 49.1% of test texts appear exactly once in the whole
dataset. Of our 210 errors: 73.8% are unique phrasings, 11% contentless, 5.2% ambiguous repeats.
Irreducible noise is real: "Exceeds weight limit of 40 lbs" appears 150 times, filed Yard Waste 99x and
Trash 51x. Estimated achievable ceiling ~0.90-0.91; all three methods sit within ~5 points of it.
