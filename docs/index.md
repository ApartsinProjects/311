# Multi-City 311 Free-Text Benchmark

A public benchmark and **cross-jurisdiction transfer study** for free-text municipal 311
service-request classification. Citizen complaint narratives from **7 US cities** are mapped into a
shared **14-class** civic taxonomy, so classifiers can be compared *across* cities — measuring how
well a model trained on some cities generalizes to an unseen one.

- **📄 [Read the paper](paper.html)** · [Download DOCX](paper.docx)
- Code: [github.com/ApartsinProjects/311](https://github.com/ApartsinProjects/311)
- Background research: [scouting report](scouting.md)

## The gap this fills

Most large 311 open-data portals expose only a structured category dropdown, not citizen free text,
and **no canonical multi-city free-text 311 benchmark or leaderboard exists**. Prior work trains on a
single city's private split with incomparable metrics. This project assembles a harmonized, multi-city,
free-text corpus and quantifies the cross-jurisdiction generalization gap.

## Cities

| City | Benchmark rows | Portal scale | Register / source |
|---|---|---|---|
| Baton Rouge | 44,226 | ~1M | call-center transcription |
| Bloomington IN | 39,954 | 129k | web/app, terse |
| Richmond VA | 36,974 | 43k | SeeClickFix |
| Auburn WA | 20,122 | 24k | SeeClickFix |
| Gainesville FL | 11,544 | 16k | myGNV app |
| Honolulu | 2,141 | 2.5k | letter-style, rich |
| San Francisco | 1,455 | millions | Open311 API |

"Benchmark rows" are the harmonized content rows after filtering (~156k total); "Portal scale" is the raw
open-data volume. The register varies sharply across cities, a built-in domain-shift signal.

## Task & protocols

- **Input:** citizen free-text complaint → **Output:** one of 14 harmonized civic categories.
- **Frozen test set:** up to 500 stratified requests per city (3,502 total); every arm predicts on these same rows.
- **In-city:** train on the held-out city's remaining data (upper bound, no domain shift).
- **Leave-one-city-out (LOCO):** train on all other cities, test on the held-out city.
- **Metric:** pooled macro-F1 over classes present in the test set, with 95% bootstrap CIs, identical across arms.

## Results (pooled macro-F1, aligned test set, 95% CI)

| Arm | In-city | Cross-city |
|---|---|---|
| TF-IDF + LogReg | 0.785 [.76,.81] | 0.523 [.50,.54] |
| Fine-tuned DistilBERT | 0.827 [.80,.85] | 0.558 [.54,.58] |
| LLM zero-shot (gpt-4o-mini, taxonomy-in-prompt) | — | 0.654 [.63,.68] |

**Headline:** crossing city boundaries costs roughly a third of macro-F1 for both trained models. A zero-shot
LLM given the taxonomy transfers best across cities (0.65 > 0.56 > 0.52, all significant), yet all sit below
in-domain (~0.8). Every arm is scored on one frozen test set. A blind label judge finds **14.3% of city
labels aren't text-supported**, and under defensibility-adjusted scoring the LLM's cross-city accuracy rises
to 0.98 — most "errors" are defensible alternatives. See the [paper](paper.html) for the full analysis.

## Error analysis: much of the "gap" is label artifact, not model error

Reading the actual misclassifications shows a large share of cross-city "errors" are **benchmark/label
artifacts**, not model mistakes:

- **Service-category vs. content mismatch** (dominant): city labels `"MISSED WOODY WASTE SERVICE"`
  (→ Waste) while the text says *"tree limbs not picked up"* (→ Trees). Both defensible.
- **Source mislabeling:** e.g. text *"graffiti"* filed under `"Sidewalk Repair"`.
- **Uninformative shorthand:** `"B/U"`, `"o.w."`, `"no sticker"` — unclassifiable from text.
- **Genuinely multi-topic** complaints forced into one label.

A blind label judge (independent of the models) finds 14.3% of city labels are not text-supported, and under
defensibility-adjusted scoring most cross-city "errors" are judge-acceptable alternatives. The measured
transfer gap conflates genuine domain shift, taxonomy non-comparability, and label noise; separating them is
the core methodological contribution.

_Numbers regenerated from the released predictions by `score_aligned.py`._
