"""
rp_distilbert.py -- DistilBERT fine-tune arm for the multi-city 311 benchmark, RunPod edition.
Reads harmonized.csv (city,label,text); runs IN-CITY (80/20) and LEAVE-ONE-CITY-OUT.
Writes results/results_distilbert.json. Follows the gpu2runpod output convention.
"""
import csv, json, os, sys, argparse
from collections import Counter, defaultdict
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

csv.field_size_limit(10**7)
assert torch.cuda.is_available(), "CUDA not available. This script requires a GPU."
DEV = torch.device("cuda")
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

MODEL = "distilbert-base-uncased"
LABELS = ["Waste_Sanitation","Streets_Sidewalks","Street_Lighting","Traffic_Signals_Signs",
          "Trees_Vegetation","Graffiti_Postings","Parking_Vehicles","Property_Housing_Code",
          "Water_Sewer_Drainage","Homelessness","Animals_Pests","Noise","Transit","Parks_Recreation"]
L2I = {l: i for i, l in enumerate(LABELS)}


def load(path, cap, seed=0):
    by = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"] in L2I and len((r["text"] or "").strip()) >= 3:
                by[r["city"]].append((r["text"].strip(), r["label"]))
    if cap:
        rng = np.random.RandomState(seed)
        for c in by:
            if len(by[c]) > cap:
                idx = rng.permutation(len(by[c]))[:cap]
                by[c] = [by[c][i] for i in idx]
    return by


class DS(Dataset):
    def __init__(self, rows, tok, maxlen):
        self.rows = rows; self.tok = tok; self.maxlen = maxlen
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        t, y = self.rows[i]
        e = self.tok(t, truncation=True, max_length=self.maxlen, padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, L2I[y]


def train_eval(train_rows, test_rows, tok, epochs, bs, lr, maxlen, tag):
    tr_labels = set(y for _, y in train_rows)
    test_rows = [(t, y) for t, y in test_rows if y in tr_labels]
    if len(test_rows) < 20 or len(set(y for _, y in test_rows)) < 2:
        return None
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    tl = DataLoader(DS(train_rows, tok, maxlen), batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    total = epochs * len(tl); step = 0
    model.train()
    for ep in range(epochs):
        for batch, y in tl:
            batch = {k: v.to(DEV, non_blocking=True) for k, v in batch.items()}
            y = y.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.cross_entropy(model(**batch).logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            step += 1
            if step % 100 == 0:
                print(f"  {tag} {step}/{total} loss={loss.item():.4f}"); sys.stdout.flush()
    model.eval(); preds = []
    el = DataLoader(DS(test_rows, tok, maxlen), batch_size=128, num_workers=2)
    with torch.no_grad():
        for batch, _ in el:
            batch = {k: v.to(DEV) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                logits = model(**batch).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    y_true = [y for _, y in test_rows]; y_pred = [LABELS[p] for p in preds]
    del model; torch.cuda.empty_cache()
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))  # over present classes (matches all arms)
    acc = float(accuracy_score(y_true, y_pred))
    per_item = [{"text": t, "gold": yt, "pred": yp} for (t, yt), yp in zip(test_rows, y_pred)]
    return macro, acc, per_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="harmonized.csv")
    ap.add_argument("--cap-per-city", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--maxlen", type=int, default=64)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    by = load(args.data, args.cap_per_city)
    cities = sorted(by, key=lambda c: -len(by[c]))
    print("[train] dataset:"); sys.stdout.flush()
    for c in cities:
        print(f"[train]   {c:14s} {len(by[c]):>7d} rows  {len(set(y for _,y in by[c])):>2d} classes")
    sys.stdout.flush()

    incity, loco, preds_out = {}, {}, {"incity": {}, "loco": {}}
    print("[train] === IN-CITY ==="); sys.stdout.flush()
    for c in cities:
        data = by[c]
        keep = {k for k, v in Counter(y for _, y in data).items() if v >= 5}
        data = [(t, y) for t, y in data if y in keep]
        if len(set(y for _, y in data)) < 2:
            continue
        tr, te = train_test_split(data, test_size=0.2, random_state=0, stratify=[y for _, y in data])
        r = train_eval(tr, te, tok, args.epochs, args.bs, args.lr, args.maxlen, f"incity:{c}")
        if r:
            incity[c] = {"macroF1": round(r[0], 4), "acc": round(r[1], 4)}
            preds_out["incity"][c] = r[2]
            print(f"[train] IN-CITY {c:14s} macroF1={r[0]:.3f} acc={r[1]:.3f}"); sys.stdout.flush()

    print("[train] === LOCO ==="); sys.stdout.flush()
    for c in cities:
        train = [d for oc in cities if oc != c for d in by[oc]]
        r = train_eval(train, by[c], tok, args.epochs, args.bs, args.lr, args.maxlen, f"loco:{c}")
        if r:
            loco[c] = {"macroF1": round(r[0], 4), "acc": round(r[1], 4)}
            preds_out["loco"][c] = r[2]
            gap = incity.get(c, {}).get("macroF1", float("nan")) - r[0]
            print(f"[train] LOCO {c:14s} macroF1={r[0]:.3f} acc={r[1]:.3f} gap={gap:+.3f}"); sys.stdout.flush()
    json.dump(preds_out, open("results/predictions_distilbert.json", "w"), ensure_ascii=False)

    summary = {}
    common = [c for c in loco if c in incity]
    if common:
        mi = np.mean([incity[c]["macroF1"] for c in common])
        ml = np.mean([loco[c]["macroF1"] for c in common])
        summary = {"mean_incity": round(float(mi), 4), "mean_loco": round(float(ml), 4),
                   "transfer_gap": round(float(mi - ml), 4)}
        print(f"[train] SUMMARY mean_incity={mi:.3f} mean_loco={ml:.3f} gap={mi-ml:+.3f}"); sys.stdout.flush()

    json.dump({"config": vars(args), "incity": incity, "loco": loco, "summary": summary},
              open("results/results_distilbert.json", "w"), indent=2)
    print("[train] === DONE ==="); sys.stdout.flush()


if __name__ == "__main__":
    main()
