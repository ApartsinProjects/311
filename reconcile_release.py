"""Reconcile released data to 'frozen scheme + the 6 documented bug-fixes' (not full rule drift).
Start from the pre-correction backups and apply ONLY the 6 corrected native-category mappings, so
harmonization_map.json, harmonized_filtered.csv and eval_split.csv all share one labeling scheme."""
import csv, json
csv.field_size_limit(10**7)

CHANGED = {
 "INSIDE RESIDENTIAL BACKUP": "Water_Sewer_Drainage",
 "OUTSIDE OVERFLOW": "Water_Sewer_Drainage",
 "INSIDE COMMERCIAL BACKUP": "Water_Sewer_Drainage",
 "GARBAGE TRUCK LEAKS/SPILLS/ISSUES": "Waste_Sanitation",
 "REQUEST FOR STREET GRADING": "Streets_Sidewalks",
 "Civil Engineering (grading or trenching within the City right-of-way)": "Streets_Sidewalks",
}

# 1. map: backup + 6 fixes
m = json.load(open("_precorr_backup/harmonization_map.json", encoding="utf-8"))
fixed = 0
for city, d in m.items():
    for nat in list(d):
        if nat in CHANGED and d[nat] != CHANGED[nat]:
            d[nat] = CHANGED[nat]; fixed += 1
json.dump(m, open("data/harmonization_map.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"map: corrected {fixed} entries")

# 2. harmonized_filtered.csv: backup + surgical relabel via (city,text)->native
nat = {}
with open("data/raw/all_cities.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        nat[(r["city"], r["text"].strip())] = r["native_category"]
rows = list(csv.DictReader(open("_precorr_backup/harmonized_filtered.csv", encoding="utf-8")))
flips = 0
for r in rows:
    nc = nat.get((r["city"], r["text"].strip()))
    if nc in CHANGED and r["label"] != CHANGED[nc]:
        r["label"] = CHANGED[nc]; flips += 1
w = csv.DictWriter(open("data/harmonized_filtered.csv", "w", newline="", encoding="utf-8"),
                   fieldnames=rows[0].keys())
w.writeheader(); w.writerows(rows)
print(f"harmonized_filtered: relabeled {flips} rows, total {len(rows)}")
