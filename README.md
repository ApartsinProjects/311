# Multi-City 311 Free-Text Benchmark

A public benchmark and cross-jurisdiction transfer study for **free-text municipal 311 service-request
classification**. Citizen complaint narratives from **7 US cities** are mapped into a shared 14-class
civic taxonomy so that classification models can be compared *across* cities — measuring how well a
model trained on some cities generalizes to an unseen one.

**Project page:** https://apartsinprojects.github.io/311/

## Why

Most large 311 open-data portals publish only a structured category dropdown, not citizen free text,
and no canonical multi-city free-text 311 benchmark or leaderboard exists. Prior work trains on a
single city's private split with incomparable metrics. This repo assembles a harmonized, multi-city,
free-text corpus and measures the cross-jurisdiction generalization gap.

## Cities (free text confirmed, citizen narrative + category label)

| City | Rows | Text field | Source |
|---|---|---|---|
| Baton Rouge | ~1M | `comments` | data.brla.gov (call-center) |
| Bloomington IN | 129k | `description` | data.bloomington.in.gov |
| Richmond VA | 43k | `description` | data.richmondgov.com (SeeClickFix) |
| Auburn WA | 24k | `description` | data.auburnwa.gov (SeeClickFix) |
| Gainesville FL | 16k | `description` | data.cityofgainesville.org (myGNV) |
| Honolulu | 2.5k | `description` | data.honolulu.gov (letter-style) |
| San Francisco | millions | `description` | Open311 API |

## Pipeline

```bash
python collect_311.py --per-city 50000     # assemble unified corpus -> data/raw/*.csv
python harmonize.py                         # 14-class shared taxonomy -> data/harmonization_map.json
python baseline_tfidf.py --cap-per-city 8000   # TF-IDF + LogReg: in-city vs leave-one-city-out
python llm_arm.py --n-per-city 100          # LLM zero-shot (taxonomy-in-prompt) via OpenRouter
python rp_distilbert.py                      # fine-tuned DistilBERT (RunPod)
```

Every run is logged to `results/summary.csv` via `results_log.py`.

## Task & protocols

- **Input:** citizen free-text complaint. **Output:** one of 14 harmonized civic categories.
- **In-city:** stratified 80/20 within a city (no domain shift; upper bound).
- **Leave-one-city-out (LOCO):** train on all other cities, test on the held-out city (the transfer number).
- **Metric:** macro-F1 over classes present in the test set (identical across all arms).

## Status

Work in progress. Results and the full write-up are on the project page.

## License

Code: MIT. Data: from municipal open-data portals under their respective public/open licenses
(e.g. SF Open311 PDDL); see each city's portal.
