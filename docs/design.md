# Semantic Multiclass by Rule Mining — Best Design (as of 2026-09-04)

A text classifier built **entirely from LLM calls over mined text**: no fine-tuning, no local model,
no retrieval store, no embeddings at inference. Everything the classifier "knows" is a compact,
human-readable rulebook, produced offline by an error-driven training loop.

## Objective and where each baseline sits (the claim)

The objective is a **prompt-only classifier**: one FIXED prompt, no store, no per-query retrieval. The
claim is that **instruction mining is the best way to tune that fixed prompt**.

| method | prompt-only? | how the prompt is tuned | needs store? | role |
|---|---|---|---|---|
| zero-shot | yes | nothing | no | lower bound |
| fixed k-shot | yes | demonstrations chosen offline | no | **head-to-head baseline** |
| **instruction mining (ours)** | yes | mined rules (pos + remap) | no | **our method** |
| RAG | NO (prompt changes per query) | retrieval | YES | store-based **upper reference** |

RAG is a baseline but NOT prompt-only -- it rewrites the prompt per query and needs a live store, so it
is the ceiling we chase with a fixed prompt, not a peer. The decisive comparison is **ours vs a
WELL-TUNED fixed k-shot**: beating a naive k-shot proves nothing. Two fair-baseline requirements:
- **Demo selection must be real, not frequency.** "Most frequent template" is meaningful only when texts
  repeat (Bloomington: top template recurs 565x). On unique-text domains it degenerates to arbitrary
  first-k (MIMIC: every text unique). Fair k-shot uses DIVERSE selection (farthest-point on TF-IDF,
  `select="diverse"`) or validation-optimized selection (greedy, val-maximizing -- the demo analog of
  our rule mining). Frequency selection is kept only for the templated domain.
- **"RAG distilled to a fixed instruction"** is the conceptual bridge and a good extra baseline:
  summarize what RAG retrieves into one fixed prompt. That is literally naive instruction mining;
  our error-driven diagnose->refine is the principled version. Framing: mining = compressing retrieval
  into an inspectable fixed prompt.

Prompt-only methods are limited by construction where our method wins most: a demo is one INSTANCE, it
cannot state a general convention, and fixed k-shot cannot scale to many classes (the prompt explodes).
HUPD (20-160 non-semantic classes) is where tuned k-shot breaks and mining wins clearest.

## The method (single method, single artifact, single inference pass)

The classifier is **one rulebook**. Each class has:
- `pos` rules — general conditions that make an item this class (no cross-class pointers).
- `remap` rules — **imperative overrides** of the form *"if <narrow condition>, classify as <CLASS>
  instead"*. These encode the counterintuitive, convention-driven routing where the surface wording
  fits one class but the organization files it under another.

**Inference = one LLM call per item.** The prompt lists every class as `NAME: <pos>  OVERRIDE: <remap>`
and instructs the model to apply overrides literally even when the wording fits the current category.
No second pass, no separate trigger stage, no store.

Why remaps instead of a separate "trigger" model: a convention placed in a flat prompt as a *description*
gets over-applied (the measured "missed recycling -> Trash" convention, as descriptive prose, changed
130 predictions for a NET OF ZERO — 59 fixed / 59 broken). Written as a **narrow imperative override**
it fires only on the real exception. The remap is the trigger, folded into the class it belongs to.

## Training (offline, framed as batched mini-batch optimization)

| training concept | here |
|---|---|
| weights | the per-class rulebook (pos + remap) |
| initialization | `_seed`: contrastive class descriptions mined together (mutually exclusive) |
| forward pass | classify all train items with the current rulebook |
| loss | a **reasoning LLM** (gpt-5) diagnoses every confusion pair: finds the UNDERLYING AXIS that separates the two classes (intent, action-vs-object, stage, severity, responsible party), not the shared surface words |
| backprop | reshuffle diagnostics by GOLD class so each class collects all error signal about it |
| optimizer | one refiner call per class (gpt-5) rewrites its pos + remap rules from that signal |
| step acceptance | **per-class, on the FULL validation set** — each class rewrite is kept only if it does not hurt val; a bad rewrite for one class cannot sink a good one for another |
| epoch | forward -> diagnose -> refine -> per-class accept |
| early stopping | stop after `patience` epochs with no val gain; keep the best rulebook |
| test | fixed hold-out, scored once |

Models: **MINE_MODEL = gpt-5** (diagnose + refine — the intelligence-critical, low-volume steps),
**CLF_MODEL = gpt-4o-mini** (classification — the high-volume forward passes). Mining is a few dozen
calls/epoch; classification is thousands. Smart where it is rare, cheap where it is bulk.

## Hard-won lessons (each was a real failure, diagnosed and fixed)

1. **Per-EDIT / per-CLASS acceptance is essential; whole-epoch acceptance fails.** Rewriting many
   classes at once and accepting/rejecting the whole batch collapses validation (0.82 -> 0.60) because
   good rewrites are discarded with bad ones, and the failure worsens with more classes (8 barely
   works, 14-16 crash). Judge each class's rewrite alone and keep only the ones that help.
2. **Acceptance must use enough validation data.** A 150-item slice was too noisy against a strong
   seed: rewrites passed the slice but hurt full val, so nothing beat the seed and training stalled.
   Judge each candidate on the FULL val set (all candidates in ONE batched classification call).
3. **Rules must not embed cross-class arrows.** Asking the refiner for `"condition -> {c}"` made it
   write `-> OtherClass` verdicts INTO each class's description; rendered into the prompt, a class's own
   definition then pointed at other classes and cross-contaminated the classifier (0.82 -> 0.60). Pos
   rules state only the condition; remaps are the only place a class names another class.
4. **A rule in a prompt is a suggestion, not a surgical remap.** Descriptive conventions over-apply.
   Narrow imperative overrides ("if <narrow condition>, classify as X instead") do not.
5. **Reasoning models need `max_completion_tokens` with large headroom** (output budget + ~6000 for
   internal reasoning) and NO temperature, or the visible JSON truncates to empty. Give each API call an
   explicit timeout so one hung request cannot stall a whole concurrent batch.
6. **Cap rule length generously** (>= 260 chars): reasoning-model rules are long, and a tight cap
   truncated remaps before their target class ("...classify as Water Utility Pr").
7. **Prompts are domain-independent.** Diagnose/refine key only on `(item_noun, label_noun)` and speak
   of "underlying axis / meaning over name / generalize don't memorize" — no municipal vocabulary — so
   the method ports to any taxonomy.

## Infrastructure

- **Persistent response cache** (SQLite, keyed on model+messages+params, temperature=0): reruns and
  cross-script duplicates are free; failures are never cached; cache-lock writes are exception-guarded.
- **Async Batch API everywhere** for bulk calls (standing project rule): `chat_many` submits cache
  MISSES as one Batch job (50% cheaper) and blocking-polls; `LLM_SYNC=1` forces the threaded path for
  fast local iteration. No raw per-request calls: single `_call` routes through the same path; the
  trigger-writing loop was converted to one batched call. See [[311-batch-api-rule]].
- **Prompt-cache-friendly layout**: the large static block (system + rulebook) comes first, the item
  text last, so the vendor's automatic prefix caching (another 50%) applies across a wave.
- Batch latency is per-job (minutes), so the mining loop's dependent waves run sequentially; this trades
  wall-clock (~1-3 h vs ~20 min) for ~50-75% lower dollar cost.

## Status of results (must be re-verified after the OpenAI billing hard limit is lifted)

Verified earlier (gpt-4.1 mining, two-stage rulebook + separate triggers, fixed 1500 test, budget 2000):
- zero-shot 0.822; RAG 0.871 (needs a store); fine-tuned 0.861 (needs training);
- rulebook 0.847; rulebook + triggers **0.862**, not significantly different from RAG (McNemar p=0.18).

**CONFIRMED (2026-09-04)** — the unified single-pass design (gpt-5 mining, one rulebook with remaps,
full-val per-class acceptance) on the fixed 1500 test, budget 2000:
- **rulebook + remaps = 0.8607** (95% CI 0.843-0.877, UNPARSED 0.003), single pass, single artifact;
- vs zero-shot 0.822: **+0.039, McNemar p=9e-06** (significant);
- vs RAG 0.871: gap +0.011, **p=0.181 (NOT significantly different)**;
- val re-scored 0.875 (training reported 0.877), so the training curve was genuine.

This **strictly dominates** the old two-stage method: same accuracy (0.861 vs 0.862) in ONE pass with
ONE artifact instead of a rulebook plus a separate trigger stage, and it stays statistically
indistinguishable from RAG while needing no store, no training, no embeddings. The full-val per-class
acceptance was the fix that let training climb past a strong 0.82 seed.

(The earlier run's 0.32 test number was a billing-limit artifact — the OpenAI hard limit was hit during
final evaluation, so most test calls returned empty. Verified after the limit was lifted.)

## Key files
- `train.py` — the training loop (forward / diagnose / refine / per-class accept) + single-pass eval.
- `semclf.py` — Task, rulebook rendering (`_dline`, `desc_sys`, `_desc_classify`), seed, stratified
  budget, paired significance test, cache-backed helpers.
- `oaillm.py` — cache-aware `chat_many` (Batch API default; `LLM_SYNC=1` threaded), reasoning-model
  param handling.
- `llmcache.py` — persistent SQLite response cache.
- `openai_batch.py` — Batch API submit/poll/collect.
- Diagnostics: `diagnostics/semclf_result.md` (the verified two-stage numbers + all negative results).
