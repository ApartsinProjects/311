"""Shared helpers for the ALIGNED evaluation: one frozen split, one prediction format."""
import csv, os, json
from collections import defaultdict

csv.field_size_limit(10**7)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "preds")

LABELS = ["Waste_Sanitation", "Streets_Sidewalks", "Street_Lighting", "Traffic_Signals_Signs",
          "Trees_Vegetation", "Graffiti_Postings", "Parking_Vehicles", "Property_Housing_Code",
          "Water_Sewer_Drainage", "Homelessness", "Animals_Pests", "Noise", "Transit", "Parks_Recreation"]


def load_split(path=None):
    """Return {'train': {city:[(text,label)]}, 'test': {city:[(text,label)]}} in file order."""
    path = path or os.path.join(DATA, "eval_split.csv")
    out = {"train": defaultdict(list), "test": defaultdict(list)}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["role"]][r["city"]].append((r["text"], r["label"]))
    return {k: dict(v) for k, v in out.items()}


def save_preds(arm, preds):
    """preds: {protocol: {city: [pred_label,... aligned to test rows in split order]}}."""
    os.makedirs(PRED_DIR, exist_ok=True)
    p = os.path.join(PRED_DIR, f"{arm}.json")
    json.dump(preds, open(p, "w"), ensure_ascii=False)
    print(f"[eval] wrote {p}")
    return p
