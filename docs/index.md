# Multi-City 311 Free-Text Benchmark

A public benchmark and **cross-jurisdiction transfer study** for free-text municipal 311
service-request classification. Citizen complaint narratives from **7 US cities** are mapped into a
shared **14-class** civic taxonomy, so classifiers can be compared *across* cities — measuring how
well a model trained on some cities generalizes to an unseen one.

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

## Results so far (macro-F1)

| Arm | In-city | Cross-city |
|---|---|---|
| TF-IDF + LogReg | ~0.80 | ~0.38 (LOCO) |
| LLM zero-shot (gpt-4o-mini, taxonomy-in-prompt) | — | ~0.49 |
| Fine-tuned DistilBERT | *finalizing* | *finalizing* |

**Headline:** crossing city boundaries roughly halves performance. An off-the-shelf LLM (given the
taxonomy in-context) transfers *somewhat better* than a classical model whose label space is frozen to
the training cities — but still far below in-domain.

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
