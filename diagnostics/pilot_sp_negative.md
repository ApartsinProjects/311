# Semantic-Policy pivot pilot: NEGATIVE (registry entry, 2026-09-01)

Falsification test of `semantic_policy_project.md` (semantic bottleneck + swappable policy mapper,
headline = few-shot sample efficiency on unseen authority). 3 cities (BatonRouge, Bloomington,
Gainesville), 5-attribute compositional schema extracted by gpt-4o-mini (batch), frozen embedding =
OpenAI text-embedding-3-small. Code: `pilot_sp.py`. Results: `results/pilot_sp.json`.

## Exp A - policy residual I(Y;A|S)
- Predictive: cv_acc S=0.740 -> S+A=0.804 (+0.064); cv_macroF1 0.551 -> 0.594.
- MI: I_obs=0.142 bits vs permutation null 0.062+/-0.007 (z=10.9). BUT restricted to genuine-semantics
  rows (object != 'other'): excess MI halves to 0.070 vs 0.036 null (z=5.6). ~half the raw signal is an
  artifact of extraction failure (state='other' 67% of rows; the largest clean groups - waste, road -
  are UNANIMOUS across cities). Real but small residual; not the clean "same tree, different department"
  splits the proposal predicted.

## Exp C - few-shot sample efficiency on unseen authority (Gainesville, majority floor 0.210)
| k/class | semantic macroF1 | embedding macroF1 | delta |
|---:|---:|---:|---:|
| 2  | 0.324+/-0.040 | 0.421+/-0.027 | -0.097 |
| 5  | 0.430+/-0.021 | 0.546+/-0.018 | -0.116 |
| 10 | 0.473+/-0.021 | 0.605+/-0.023 | -0.132 |
| 25 | 0.492+/-0.013 | 0.651+/-0.021 | -0.159 |

Frozen embedding dominates at EVERY budget; gap WIDENS with data. No low-data regime where the
interpretable bottleneck wins. The semantic arm is well above floor (not broken - it carries real
signal), just strictly weaker than a 1536-dim frozen embedding + LR. **This is Kill 3 (§18):** direct
frozen text embeddings adapt at least as well with the same small labeled data.

## Verdict
Do NOT pivot to the sample-efficiency framing. A 35-dim interpretable bottleneck discards text signal
a frozen embedding keeps; that tradeoff is exactly what Kill 3 predicted, and schema tweaks will not
close a 10-16 point gap. The one salvageable positive - a real (if small) policy residual after
controlling for semantics - SUPPORTS the existing paper's thesis ("administrative gold is not semantic
ground truth"), so it strengthens the current paper rather than motivating a new one.

Invariants all passed (I_obs>>null; in-sample S+A>=S; both arms > floor at max budget), so the negative
is trustworthy, not a bug.

## Follow-up: model-free test of "same text, different city, different label?" (`pilot_neighbor.py`)
Motivated by the objection that Exp A's hand-schema was too lossy (state='other' 67%). Dropped the
schema; used embedding similarity (text-embedding-3-small) as the semantic control. 1200 train rows/city.

Cross- vs within-city harmonized-label agreement, binned by cosine similarity (thousands of pairs):
| cosine bin | within-city agree | cross-city agree | gap |
|---|---|---|---|
| 0.5-0.6 | 0.64 | 0.69 | -0.05 |
| 0.6-0.7 | 0.81 | 0.83 | -0.02 |
| 0.7-0.8 | 0.91 | 0.90 | +0.01 |
| 0.8-0.9 | 0.96 | 1.00 | -0.04 |
| 0.9-1.0 | 0.98 | 1.00 | -0.02 |

Cross-city agreement TRACKS within-city agreement at every similarity level; the disagreement that
exists is at LOW similarity (different situations), not high. Only 2 genuine cross-city near-paraphrases
(cos>0.86) exist in the sample (the 3 cities word requests differently = language shift), and BOTH route
to the same department even in NATIVE taxonomies ("Excessive Growth"=="Grass (Overgrown)"->Trees;
"Missed Garbage"=="Garbage Pick Up"->Waste). Native-category join: 0 harmonized-label disagreements.

## Reconciliation with the earlier "Baton Rouge" observation
The earlier signal was NOT "same text, different city, different label." It was (a) a WITHIN-city gap
between the LLM judge's semantic read and the city's administrative label (CASE noise ~7-8%), and
(b) transfer loss, which is better explained by how Baton Rouge WORDS requests (language shift) and its
base rates (population shift) than by cross-city policy divergence. Strict policy shift P(Y;A|S) is small
in this data.

## Implication for the CURRENT paper (flag)
The framing "transfer loss conflates semantic shift with POLICY shift" is only weakly supported at the
cross-jurisdiction level. The strongly-supported claims are WITHIN-example: administrative labels are
multi-membership and carry content-label noise (CASE floors, cross-vendor agreement, set-valued training
gains). Recommend the paper lead with the within-example construct claim and NOT overclaim cross-city
policy divergence. This refines, not refutes, "administrative gold is not semantic ground truth."
