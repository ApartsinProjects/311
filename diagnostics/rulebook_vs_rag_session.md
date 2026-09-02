# Rulebook vs RAG: session wrap (2026-09-02)

Explored the reframed question: **when doing RAG-based classification of free text into a large,
organization-specific label set, can a compiled interpretable rulebook replace per-query retrieval?**
Rules would be cheaper (fixed prompt), cacheable (constant prefix, unlike RAG's per-query demos),
storeless (no vector index at inference), and inspectable/editable.

## Datasets tried and what each taught
- **HUPD patents -> IPC subclass (539-way)**: WRONG testbed. Zero-shot LLM = 0.488, beating fine-tuned
  on all 24768 (0.463/0.465 TF-IDF/DistilBERT) and RAG (0.479). IPC is a public taxonomy the LLM has
  memorized -> no room for a learned codebook. Lesson: criterion "labels non-semantic to the LLM" rules
  out ALL well-known public taxonomies (IPC/ICD/NAICS). See [[hupd-negative]] (diagnostics/hupd_negative.md).
- **Baton Rouge NATIVE categories (80 org-specific labels)**: right kind of task. Zero-shot LLM FAILS
  (0.487/0.523) -- it cannot distinguish the city's fine service subtypes (woody-waste vs handpile vs
  missed-service; damaged vs missing cart). Fine-tuned TF-IDF+LR: 0.507(200)/0.684(2k)/0.796(full).

## Decisive result (Baton Rouge native, budget=2000)
| method | acc | inference |
|---|---|---|
| zero-shot | 0.523 | fixed prompt |
| rulebook (8 mined rules) | 0.660 | fixed, CACHEABLE, no store |
| fine-tuned TF-IDF+LR | 0.684 | model |
| RAG-few-shot (k=12) | 0.757 | per-query retrieval + demos, needs store |

The rulebook closes ~60% of the zero-shot->RAG gap with 8 interpretable, correct rules (cart
damaged/missing; garbage-service vs woody-waste vs handpile; mowing->tall-grass; illegal-dumping->junk),
but is ~10 pts BELOW RAG on accuracy and mining PLATEAUED at 8 rules (further confusion pairs failed
validation). So a fixed rulebook has an accuracy ceiling below RAG here.

## Honest conclusion
- Clean "replace RAG on accuracy" claim FAILS at budget 2000: RAG 0.757 > rulebook 0.660.
- Rules WIN on cost / prompt-caching / no-store / interpretability / editability, at a ~10pt accuracy cost.
- UNTESTED (the remaining shot for an accuracy win): low-data regime (B=200/500), where RAG's retrieval
  store is thin and neighbors are poor; a compact rulebook may generalize better. Not run (session stopped).

## Method note
RAG few-shot is the real competitor and it does NOT break as labels grow (retrieves relevant demos
per query regardless of label count), which undercuts the original "few-shot can't scale to many labels"
motivation -- that only applies to few-shot-PER-LABEL, not RAG. Any future claim must beat RAG, not
few-shot-per-label. Prior-art to cite for the miner: ERGO, APE, MIPRO (see [[311-rulebook-framing]]).

Code: hupd_compare.py, modal_hupd_bert_train.py, mimic_build.py, mimic_rulebook.py,
br_native_compare.py, br_native_instruct.py. Large embeddings/splits and credentialed MIMIC data are
gitignored (regenerable / DUA).
