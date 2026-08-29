# Multi-City 311 Free-Text Benchmark

A public benchmark and **cross-jurisdiction transfer study** for free-text municipal 311
service-request classification. Citizen complaint narratives from **7 US cities** are mapped into a
shared **14-class** civic taxonomy, so classifiers can be compared *across* cities — measuring how
well a model trained on some cities generalizes to an unseen one.

- **📄 [Read the paper (draft)](paper.html)**
- Code: [github.com/ApartsinProjects/311](https://github.com/ApartsinProjects/311)
- Background research: [scouting report](scouting.md)

## The gap this fills

Most large 311 open-data portals expose only a structured category dropdown, not citizen free text,
and **no canonical multi-city free-text 311 benchmark or leaderboard exists**. Prior work trains on a
single city's private split with incomparable metrics. This project assembles a harmonized, multi-city,
free-text corpus and quantifies the cross-jurisdiction generalization gap.

## Cities

| City | Rows | Register / source |
|---|---|---|
| Baton Rouge | ~1M | call-center transcription |
| Bloomington IN | 129k | web/app, terse |
| Richmond VA | 43k | SeeClickFix |
| Auburn WA | 24k | SeeClickFix |
| Gainesville FL | 16k | myGNV app |
| Honolulu | 2.5k | letter-style, rich |
| San Francisco | millions | Open311 API |

The register varies sharply across cities — a built-in domain-shift signal.

## Task & protocols

- **Input:** citizen free-text complaint → **Output:** one of 14 harmonized civic categories.
- **In-city:** stratified 80/20 within a city (upper bound, no domain shift).
- **Leave-one-city-out (LOCO):** train on all other cities, test on the held-out city.
- **Metric:** macro-F1 over classes present in the test set (identical across all arms).

## Results (pooled macro-F1, aligned test set, 95% CI)

| Arm | In-city | Cross-city |
|---|---|---|
| TF-IDF + LogReg | 0.785 [.76,.81] | 0.523 [.50,.54] |
| Fine-tuned DistilBERT | 0.827 [.80,.85] | 0.558 [.54,.58] |
| LLM zero-shot (gpt-4o-mini, taxonomy-in-prompt) | — | 0.654 [.63,.68] |

**Headline:** crossing city boundaries roughly halves performance for both trained models. A zero-shot LLM
given the taxonomy transfers best across cities (0.65 > 0.56 > 0.52, all significant), yet all sit below
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

~48% of errors have the predicted topic literally present in the text. The measured transfer gap
conflates genuine domain shift, taxonomy non-comparability, and label noise — separating them is the
core methodological contribution.

_Work in progress. Numbers update as final runs complete._
