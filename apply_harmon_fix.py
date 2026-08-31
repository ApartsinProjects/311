"""Surgically correct ONLY the 6 mis-mapped native categories in the frozen eval_split.
Preserves every other stored label (which predates later rule drift). Same rows, corrected gold."""
import csv, importlib
csv.field_size_limit(10**7)
import harmonize as H; importlib.reload(H)

CHANGED = {  # native category -> corrected class (verified bugs)
 "INSIDE RESIDENTIAL BACKUP": "Water_Sewer_Drainage",
 "OUTSIDE OVERFLOW": "Water_Sewer_Drainage",
 "INSIDE COMMERCIAL BACKUP": "Water_Sewer_Drainage",
 "GARBAGE TRUCK LEAKS/SPILLS/ISSUES": "Waste_Sanitation",
 "REQUEST FOR STREET GRADING": "Streets_Sidewalks",
 "Civil Engineering (grading or trenching within the City right-of-way)": "Streets_Sidewalks",
}

nat = {}
with open("data/raw/all_cities.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        nat[(r["city"], r["text"].strip())] = r["native_category"]

rows = list(csv.DictReader(open("data/eval_split.csv", encoding="utf-8")))
from collections import Counter
flips = Counter(); test_flips = 0
for r in rows:
    ncat = nat.get((r["city"], r["text"].strip()))
    if ncat in CHANGED:
        new = CHANGED[ncat]
        if new != r["label"]:
            flips[(r["label"], new)] += 1
            if r["role"] == "test":
                test_flips += 1
            r["label"] = new
w = csv.DictWriter(open("data/eval_split.csv", "w", newline="", encoding="utf-8"),
                   fieldnames=["city", "role", "label", "text"])
w.writeheader(); w.writerows(rows)
print(f"total flips={sum(flips.values())}  test-set flips={test_flips}")
for k, v in flips.most_common():
    print(f"   {v:5d}  {k[0]} -> {k[1]}")
