"""
taxonomy_sensitivity.py -- Exp E: does the transfer gap survive a coarser ontology?
Merge the confusable neighboring classes into an 8-class taxonomy, remap BOTH the frozen gold and
every arm's existing predictions, and recompute in-city/LOCO pooled macro-F1. No model re-run:
this isolates how much of the difficulty is taxonomy-boundary-induced (cross-boundary errors within
a merged group become correct under the coarse labels).
"""
import os, json, glob
from sklearn.metrics import f1_score
from eval_common import load_split, PRED_DIR as PRED

# 14 -> 8 coarse map, merging the dominant confusion neighbors (Waste<->Trees, Streets<->Water, etc.)
COARSE = {
 "Waste_Sanitation": "Sanitation_Vegetation", "Trees_Vegetation": "Sanitation_Vegetation",
 "Streets_Sidewalks": "Roads_Infrastructure", "Street_Lighting": "Roads_Infrastructure",
 "Traffic_Signals_Signs": "Roads_Infrastructure", "Water_Sewer_Drainage": "Roads_Infrastructure",
 "Parking_Vehicles": "Parking_Vehicles",
 "Property_Housing_Code": "Property_Housing",
 "Homelessness": "Homelessness", "Graffiti_Postings": "Graffiti_Postings",
 "Animals_Pests": "Animals_Pests",
 "Noise": "Community", "Transit": "Community", "Parks_Recreation": "Community",
}


def gold_by_city():
    from collections import defaultdict
    g = defaultdict(list)
    for c, rows in load_split()["test"].items():
        g[c] = [y for _, y in rows]
    return g


def macro(gold, pred):
    return f1_score(gold, pred, average="macro", zero_division=0)


def pooled(gold, preds_bycity):
    G = []; P = []
    for c in gold:
        if c in preds_bycity:
            G += [COARSE[y] for y in gold[c]]
            P += [COARSE.get(p, p) for p in preds_bycity[c]]
    return macro(G, P)


def main():
    gold = gold_by_city()
    print(f"{'arm/protocol':30s}{'14-class':>10s}{'8-class':>10s}{'delta':>8s}")
    for pf in sorted(glob.glob(os.path.join(PRED, "*.json"))):
        name = os.path.splitext(os.path.basename(pf))[0]
        if name.startswith("acceptable_sets"):
            continue
        data = json.load(open(pf, encoding="utf-8"))
        for proto, preds in data.items():
            # 14-class pooled
            G14 = []; P14 = []
            for c in gold:
                if c in preds:
                    G14 += gold[c]; P14 += preds[c]
            f14 = macro(G14, P14); f8 = pooled(gold, preds)
            print(f"{name+'/'+proto:30s}{f14:10.3f}{f8:10.3f}{f8-f14:+8.3f}")


if __name__ == "__main__":
    main()
