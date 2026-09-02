"""Build a single-label MIMIC-III diagnosis-coding dataset for the convention-mining test.
Text = presentation portion of the discharge summary (first ~1800 chars, excludes the trailing
'Discharge Diagnosis' section to avoid label leakage). Label = ICD-9 CHAPTER of the PRINCIPAL
diagnosis (SEQ_NUM=1). ~18 broad classes, single-label, mirroring the 311 setup.
Writes data/mimic_dx.json = [{text, chapter}].
"""
import gzip, csv, json, re
from collections import Counter
csv.field_size_limit(10**7)
SRC = r"D:\afeka\data\mimic III"
N_MAX = 4000

CH = [  # (lo, hi, name) over the integer part of the ICD-9 code
 (1, 139, "Infectious_Parasitic"), (140, 239, "Neoplasms"), (240, 279, "Endocrine_Metabolic"),
 (280, 289, "Blood"), (290, 319, "Mental"), (320, 389, "Nervous_Sense"),
 (390, 459, "Circulatory"), (460, 519, "Respiratory"), (520, 579, "Digestive"),
 (580, 629, "Genitourinary"), (630, 679, "Pregnancy_Childbirth"), (680, 709, "Skin"),
 (710, 739, "Musculoskeletal"), (740, 759, "Congenital"), (760, 779, "Perinatal"),
 (780, 799, "Symptoms_Illdefined"), (800, 999, "Injury_Poisoning"),
]


def chapter(code):
    code = (code or "").strip()
    if not code:
        return None
    if code[0] == "V":
        return "Supplementary_V"
    if code[0] == "E":
        return "External_Cause_E"
    try:
        n = int(code[:3])
    except ValueError:
        return None
    for lo, hi, name in CH:
        if lo <= n <= hi:
            return name
    return None


def principal():
    d = {}
    with gzip.open(f"{SRC}/DIAGNOSES_ICD.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["SEQ_NUM"] == "1" and r["HADM_ID"]:
                d[r["HADM_ID"]] = r["ICD9_CODE"]
    return d


def clean(t):
    t = re.sub(r"\[\*\*.*?\*\*\]", " ", t)      # de-id placeholders
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def main():
    prin = principal()
    print(f"admissions with principal dx: {len(prin)}")
    seen = set(); rows = []
    with gzip.open(f"{SRC}/NOTEEVENTS.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["CATEGORY"] != "Discharge summary":
                continue
            h = r["HADM_ID"]
            if h in seen or h not in prin:
                continue
            ch = chapter(prin[h])
            if not ch:
                continue
            seen.add(h)
            rows.append({"text": clean(r["TEXT"])[:1800], "chapter": ch})
            if len(rows) >= N_MAX:
                break
    json.dump(rows, open("data/mimic_dx.json", "w"), ensure_ascii=False)
    print(f"wrote data/mimic_dx.json  n={len(rows)}")
    print("chapter distribution:", Counter(r["chapter"] for r in rows).most_common())


if __name__ == "__main__":
    main()
