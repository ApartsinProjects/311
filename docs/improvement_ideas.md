# Method Improvement Ideas (Fable brainstorm, grounded in train.py)

Goal: make the rule-mining method robust, cheaper (fewer tokens / smaller prompts), faster to converge,
and strong enough to SURPASS RAG and fine-tuning on the sweet-spot datasets (banking77, hupd, mimic).

**Key structural insight driving the top ideas:** the loss is pair-structured (`min_err=2`), so the ~75%
of errors that are unique-phrasing singletons are invisible to the training signal; and the highest-value
artifact (the diagnosed axis in `diagnose_batch`) is used once to steer a rewrite, then discarded.

Dimensions: (1) robustness/variance, (2) fewer tokens/smaller prompts, (3) faster convergence,
(4) superb performance / surpass RAG, (5) SGD tricks, (6) LLM prompting strategies.

---

## ⭐ FIRST THREE TO IMPLEMENT (+ one free win)

### ①  Disambiguation section from existing diagnostics  [dim 4, 6 | HIGH | low cost]
`diagnose_batch` already produces `dimension` / `gt_signal` / `pr_signal` per confusion pair, then
discards it. Instead keep a persistent set of pairwise tie-breaker lines rendered at the end of the
rulebook: "X vs Y: decide by <axis>; X if <gt_signal>, Y if <pr_signal>". Val-gate each pair line like a
class rewrite (pairs are independent → reuse per-class acceptance). Gives banking77's neighbor confusions
the explicit decision BOUNDARY that RAG structurally cannot provide (retrieval returns instances, not the
boundary). Cap at top-N pairs by error mass; let the compacter see the section. Risk: prompt growth.

### ②  Per-class recall loss for singleton errors  [dim 4, 3 | HIGH | one extra batched call/epoch]
The pair loss only sees errors sharing a (gold,pred) cell ≥2 times. Add a second diagnostic per class:
collect ALL errors with gold=c (incl. singletons), ask "what phrasing families of 'c' do the current pos
rules fail to cover? Output 2-3 generalized coverage conditions." Route into `refine_batch` as extra
evidence. The ONLY mechanism attacking the dominant 75% error bucket. Phrase additions as "also counts as
'c' when <condition>" and rely on the per-class val gate to stop over-widening.

### ③  Logprob-margin gating (with numeric single-token labels)  [dim 1, 3, 2 | HIGH | low-med]
Gate on the gold-vs-best-competitor logprob margin instead of 0/1 accuracy. A continuous signal needs
~3-5x fewer val items for the same discrimination power → cheaper OR less-noisy acceptance. Make labels
single-token ("reply with the class NUMBER") so margins are clean; this also shrinks banking77's prompt
and output parsing, and the top-k candidates fall out of the same logprobs (unlocks ④'s free shortlist).
Use margin for SEL (variant selection); keep accuracy for the final ACC accept.

### ⑩  Prompt caching by construction  [dim 2 | FREE, do immediately]
Put the ENTIRE rendered rulebook in the SYSTEM message and only the item in the USER message (currently
interleaved). OpenAI auto-caches prefixes >1024 tokens → 50-90% input-cost cut on every forward, gate,
and eval batch, zero accuracy change. No risk. Budget saved converts to more training/capacity.

---

## The rest (ranked)

### ④  Confusion-cluster two-stage shortlist, NO retrieval store  [dim 2, 4 | HIGH for banking | medium]
Stage 1: cheap call with class names + one-line gists → top-3 candidates (or read from top_logprobs with
numeric labels → nearly free). Stage 2: render FULL rulebooks of only those 3 classes + their remap
targets + relevant disambiguation lines (①) → decide. Prompt size O(3) not O(K); a 77-way decision
becomes a 3-way contrastive one where narrow rules are strongest; and per-class rule budget is no longer
capped by the global prompt so `RULEBOOK_MAX` can grow several-fold (capacity is the proven lever).
Measure recall@3 of stage 1 first (expect 95%+; use top-5 if not). Best structural shot at beating 0.86
on banking. Check router.py / br_native_hier.py for what was already tried; the new elements are the
logprob shortlist (free stage 1) and cluster-local capacity above the old global budget.

### ⑤  Bagged mining with rule-level merge  [dim 1, 4 | HIGH variance / MED accuracy | M× offline mining]
M=3 independent mining runs on different shards/seeds; a merge LLM keeps pos rules appearing (semantically)
in ≥2 books (rule-level majority vote) + the union of remaps, each remap individually val-gated. Ensemble
variance reduction at 1× inference (unlike self-consistency's 3×); different seeds discover different
conventions → higher-capacity merged book. Gate remaps one at a time via existing per-class machinery.

### ⑥  Robust loss: excise label-noise items from diagnostics  [dim 1, 3 | MED-HIGH | trivial]
5-9% of errors are irreducible (identical text, conflicting gold). They poison `diagnose_batch` (force an
invented axis). Detect near-duplicate texts with conflicting gold in the mine set; add `"separable":
true/false` to the diagnosis JSON and drop non-separable/noise items from error pools (keep val untouched).
Gradient clipping for the loss.

### ⑦  LR schedule as edit-scope decay (trust region)  [dim 1, 3 | MED | prompt change]
Refiner currently emits a full replacement every epoch (max step forever). Schedule: epochs 1-2 full
rewrite; later epochs diff-mode ("keep rules verbatim; change ≤1 pos, add ≤1 remap"). Small steps near
convergence cut late-epoch churn variance. Keep one NVAR variant as the full rewrite so the gate can still
choose a big step if needed.

### ⑧  Per-rule credit assignment via fired-rule attribution  [dim 2, 1 | MED | low]
Classifier outputs `label|rule#` (stable rule IDs). Over the val forward pass, a rule firing mostly on
errors is negative-utility → prune; a remap that never fires is dead weight → prune (weight decay for
prompts). Gives the compacter evidence, keeps the book small without losing discriminative rules. Use to
nominate pruning candidates; still gate on val (self-reported attribution is imperfect).

### ⑨  Sequential testing (SPRT) to cut gate cost  [dim 3 | MED (indirect) | low]
Acceptance gate (candidates × full val) is the token bottleneck. Evaluate in ~100-item chunks; kill
candidates clearly below base early, spend full val only on near-ties → 3-5× cheaper gates → more
epochs / more NVAR at fixed budget. Final accept still on the untouched ACC half.

### ⑪  Structured two-step answer at inference (CoT-lite)  [dim 4, 6 | MED | low]
Ask gpt-4o-mini to output `candidates: A, B` → `overrides checked: …` → `answer: X` (~30-40 extra tokens).
Forces it to consult remaps instead of pattern-matching (where unique-phrasing errors die). Val-gate it as
a formulation choice (output format is an axis population search never explored). Risk: output-token cost,
parse failures → strict last-line format.

### ⑫  Prediction-conditioned boosting stage  [dim 4 | MED | one extra phase]
After plateau, freeze the book, collect residual errors, mine a new remap family conditioned on the
model's own tendency: "if it looks like X but <narrow condition>, answer Y". Compiles into class X's
OVERRIDE line → inference stays one call. Boosting on residuals with a frozen strong learner.

### ⑬  Contentless-item base-rate fallback rule  [dim 4 | LOW-MED | free]
~10% of errors are contentless items. Compute empirical gold argmax of contentless train items; append one
global rule: "if the item lacks actionable content, answer <argmax>". Converts a near-chance bucket to
argmax rate. Gate it.

### ⑭  Synthetic paraphrase probes as auxiliary SELECTION signal  [dim 1, 4 | MED | offline]
Have gpt-4.1 generate 10-20 unusual phrasings per class (labels known) + near-miss neighbor items. Use as
a free extension of the SEL half when choosing among NVAR variants (NEVER for final acceptance, which
stays on real val). More selection data at zero labeling cost; probes the unique-phrasing failure mode.

---

## Recommended sequencing
1. Do ⑩ (caching) immediately — free budget multiplier.
2. Implement ① + ② (reuse existing machinery; attack banking neighbors + the 75% singleton bucket).
3. Add ③ (logprob-margin gating + numeric labels) — cuts variance, shrinks banking, unlocks ④.
4. If banking still < 0.86, do ④ (confusion-cluster shortlist with cluster-local capacity) — the
   structural change with the most headroom, since capacity is the proven lever.
