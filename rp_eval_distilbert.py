"""
rp_eval_distilbert.py -- ALIGNED DistilBERT eval on RunPod.
Reads eval_split.csv (city, role, label, text); trains in-city and LOCO; predicts on the
FROZEN per-city TEST rows; saves results/preds/distilbert.json = {"incity":{city:[preds]},
"loco":{city:[preds]}} in split row order. Follows gpu2runpod conventions.
"""
import csv, os, sys, json
from collections import defaultdict
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

csv.field_size_limit(10**7)
assert torch.cuda.is_available(), "CUDA not available. This script requires a GPU."
DEV = torch.device("cuda")
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

MODEL = "distilbert-base-uncased"
LABELS = ["Waste_Sanitation", "Streets_Sidewalks", "Street_Lighting", "Traffic_Signals_Signs",
          "Trees_Vegetation", "Graffiti_Postings", "Parking_Vehicles", "Property_Housing_Code",
          "Water_Sewer_Drainage", "Homelessness", "Animals_Pests", "Noise", "Transit", "Parks_Recreation"]
L2I = {l: i for i, l in enumerate(LABELS)}
CAP = 12000


def load(path):
    d = {"train": defaultdict(list), "test": defaultdict(list)}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"] in L2I:
                d[r["role"]][r["city"]].append((r["text"], r["label"]))
    return d


def cap(rows, c=CAP, seed=0):
    return rows if len(rows) <= c else [rows[i] for i in np.random.RandomState(seed).permutation(len(rows))[:c]]


class DS(Dataset):
    def __init__(self, rows, tok, ml=64):
        self.rows = rows; self.tok = tok; self.ml = ml
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        t, y = self.rows[i]
        e = self.tok(t, truncation=True, max_length=self.ml, padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, L2I[y]


def train_predict(train, test, tok, epochs, bs, lr, tag):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    tl = DataLoader(DS(train, tok), batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    total = epochs * len(tl); step = 0; model.train()
    for ep in range(epochs):
        for batch, y in tl:
            batch = {k: v.to(DEV, non_blocking=True) for k, v in batch.items()}; y = y.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.cross_entropy(model(**batch).logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); step += 1
            if step % 200 == 0:
                print(f"  {tag} {step}/{total} loss={loss.item():.4f}"); sys.stdout.flush()
    model.eval(); preds = []
    el = DataLoader(DS(test, tok), batch_size=128, num_workers=2)
    with torch.no_grad():
        for batch, _ in el:
            batch = {k: v.to(DEV) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                logits = model(**batch).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    del model; torch.cuda.empty_cache()
    return [LABELS[p] for p in preds]


def main():
    epochs, bs, lr = 3, 64, 3e-5
    sp = load("eval_split.csv")
    tok = AutoTokenizer.from_pretrained(MODEL)
    cities = list(sp["test"])
    os.makedirs("results/preds", exist_ok=True)
    out = {"incity": {}, "loco": {}}
    for c in cities:
        test = sp["test"][c]
        out["incity"][c] = train_predict(cap(sp["train"][c]), test, tok, epochs, bs, lr, f"incity:{c}")
        print(f"[train] in-city {c} done"); sys.stdout.flush()
        pool = [d for oc in cities if oc != c for d in cap(sp["train"][oc])]
        out["loco"][c] = train_predict(pool, test, tok, epochs, bs, lr, f"loco:{c}")
        print(f"[train] loco {c} done"); sys.stdout.flush()
    json.dump(out, open("results/preds/distilbert.json", "w"), ensure_ascii=False)
    print("[train] === DONE ==="); sys.stdout.flush()


if __name__ == "__main__":
    main()
