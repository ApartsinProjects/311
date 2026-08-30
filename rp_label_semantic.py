"""
rp_label_semantic.py -- label-semantic zero-shot baselines (W8), on RunPod.
Both use the 14 class DESCRIPTIONS (not source-domain training), so they isolate whether access to
target-label semantics is what helps cross-jurisdiction transfer, independent of model scale:
  (1) sentence-transformer: embed text vs class-description, argmax cosine  -> preds/sbert.json
  (2) NLI zero-shot: entailment of "This request is about <class>"          -> preds/nli.json
Predicts on the frozen test set (eval_split.csv). Follows gpu2runpod conventions.
"""
import csv, os, sys, json
from collections import defaultdict
import torch
csv.field_size_limit(10**7)
assert torch.cuda.is_available(), "needs GPU"
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

LABELS = ["Waste_Sanitation", "Streets_Sidewalks", "Street_Lighting", "Traffic_Signals_Signs",
          "Trees_Vegetation", "Graffiti_Postings", "Parking_Vehicles", "Property_Housing_Code",
          "Water_Sewer_Drainage", "Homelessness", "Animals_Pests", "Noise", "Transit", "Parks_Recreation"]
DESC = {
 "Waste_Sanitation":"garbage or recycling collection, missed pickup, litter, illegal dumping, debris",
 "Streets_Sidewalks":"pothole or road, sidewalk, curb or driveway repair, blocked street",
 "Street_Lighting":"street light or lamp out or damaged",
 "Traffic_Signals_Signs":"traffic signal, traffic or street sign, road markings, speeding",
 "Trees_Vegetation":"trees, tree limbs, overgrown grass or weeds, vegetation, mowing",
 "Graffiti_Postings":"graffiti or illegal postings and signs",
 "Parking_Vehicles":"abandoned or junk vehicle, illegal parking, parking meter",
 "Property_Housing_Code":"building or zoning code violation, vacant or unsafe building, rental housing",
 "Water_Sewer_Drainage":"sewer, drainage, flooding, catch basin, manhole, water quality, pipe",
 "Homelessness":"homeless encampment or person living in a vehicle or on the sidewalk",
 "Animals_Pests":"animal control, stray pet, wildlife, rodents or pests",
 "Noise":"noise complaint", "Transit":"public transit or bus, taxi, scooter",
 "Parks_Recreation":"park, playground, beach, trail or recreation facility",
}


def load_test():
    by = defaultdict(list)
    with open("eval_split.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["role"] == "test":
                by[r["city"]].append((r["text"], r["label"]))
    return by


def save(tag, preds):
    os.makedirs("results/preds", exist_ok=True)
    json.dump({"zeroshot": preds}, open(f"results/preds/{tag}.json", "w"), ensure_ascii=False)
    print(f"[train] wrote results/preds/{tag}.json"); sys.stdout.flush()


def run_sbert(by):
    from sentence_transformers import SentenceTransformer, util
    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    desc_emb = m.encode([f"{k}: {DESC[k]}" for k in LABELS], convert_to_tensor=True, normalize_embeddings=True)
    preds = {}
    for c, rows in by.items():
        txts = [t for t, _ in rows]
        te = m.encode(txts, convert_to_tensor=True, normalize_embeddings=True, batch_size=128)
        sims = util.cos_sim(te, desc_emb)
        preds[c] = [LABELS[int(i)] for i in sims.argmax(1).cpu()]
        print(f"[train] sbert {c} done"); sys.stdout.flush()
    save("sbert", preds)


def run_nli(by):
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    cand = [DESC[k] for k in LABELS]; d2l = {DESC[k]: k for k in LABELS}
    preds = {}
    for c, rows in by.items():
        out = []
        txts = [t[:400] for t, _ in rows]
        for i in range(0, len(txts), 32):
            res = clf(txts[i:i+32], cand, multi_label=False)
            res = res if isinstance(res, list) else [res]
            out += [d2l[r["labels"][0]] for r in res]
        preds[c] = out
        print(f"[train] nli {c} done"); sys.stdout.flush()
    save("nli", preds)


def main():
    by = load_test()
    run_sbert(by)
    run_nli(by)
    print("[train] === DONE ==="); sys.stdout.flush()


if __name__ == "__main__":
    main()
