"""E1 on RunPod: for each leave-one-city-out fold and each label condition (raw / judge_relabel /
random_relabel), fine-tune DistilBERT on the fold's training CSV and predict the held-out city's
FROZEN test rows. Saves results/preds/e1_distilbert.json = {city: {cond: [preds]}}. gpu2runpod conventions.
"""
import csv, os, sys, json, glob
from collections import defaultdict
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

csv.field_size_limit(10**7)
assert torch.cuda.is_available(), "CUDA not available. This script requires a GPU."
DEV = torch.device("cuda")
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

MODEL = "distilbert-base-uncased"; MAXLEN = 64; SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
LABELS = ["Waste_Sanitation", "Streets_Sidewalks", "Street_Lighting", "Traffic_Signals_Signs",
          "Trees_Vegetation", "Graffiti_Postings", "Parking_Vehicles", "Property_Housing_Code",
          "Water_Sewer_Drainage", "Homelessness", "Animals_Pests", "Noise", "Transit", "Parks_Recreation"]
L2I = {l: i for i, l in enumerate(LABELS)}
CONDS = os.environ.get("RP_CONDS", "raw,judge_relabel,random_relabel").split(",")
OUTFILE = os.environ.get("RP_OUT", "e1_distilbert.json")


def load_test(path="eval_split.csv"):
    d = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["role"] == "test" and r["label"] in L2I:
                d[r["city"]].append((r["text"], r["label"]))
    return d


def load_train_data(path="e1_train_data.json"):
    return json.load(open(path, encoding="utf-8"))  # {city: {texts:[...], labels:{cond:[...]}}}


class DS(Dataset):
    """rows: list of (text, target). target is a label string (single) or a multi-hot list (soft)."""
    def __init__(self, rows, tok, ml=MAXLEN): self.rows = rows; self.tok = tok; self.ml = ml
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        t, y = self.rows[i]
        e = self.tok(t, truncation=True, max_length=self.ml, padding="max_length", return_tensors="pt")
        tgt = torch.tensor(y, dtype=torch.float32) if isinstance(y, list) else L2I[y]
        return {k: v.squeeze(0) for k, v in e.items()}, tgt


def train_predict_soft(train, test, tok, tag, epochs=3, bs=64, lr=3e-5):
    """train: list of (text, multi_hot_list[14]); BCE multi-label; predict argmax (top-1) for strict scoring."""
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr); scaler = torch.amp.GradScaler("cuda")
    tl = DataLoader(DS(train, tok), batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    total = epochs * len(tl); step = 0; model.train()
    for ep in range(epochs):
        for batch, y in tl:
            batch = {k: v.to(DEV, non_blocking=True) for k, v in batch.items()}; y = y.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(**batch).logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); step += 1
            if step % 200 == 0:
                print(f"  {tag} {step}/{total} loss={loss.item():.4f}"); sys.stdout.flush()
    model.eval(); preds = []
    with torch.no_grad():
        for batch, _ in DataLoader(DS(test, tok), batch_size=128, num_workers=2):
            batch = {k: v.to(DEV) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                preds.extend(model(**batch).logits.argmax(-1).cpu().tolist())
    del model; torch.cuda.empty_cache()
    return [LABELS[p] for p in preds]


def train_predict(train, test, tok, tag, epochs=3, bs=64, lr=3e-5):
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr); scaler = torch.amp.GradScaler("cuda")
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
    with torch.no_grad():
        for batch, _ in DataLoader(DS(test, tok), batch_size=128, num_workers=2):
            batch = {k: v.to(DEV) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                preds.extend(model(**batch).logits.argmax(-1).cpu().tolist())
    del model; torch.cuda.empty_cache()
    return [LABELS[p] for p in preds]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    test = load_test()
    data = load_train_data()
    folds = sorted(data)
    print(f"[train] folds={folds}"); sys.stdout.flush()
    out = {}
    for city in folds:
        out[city] = {}
        texts = data[city]["texts"]
        raw = data[city]["labels"]["raw"]
        for cond in CONDS:
            if cond == "soft":
                acc = data[city]["accept_sets"]
                tgts = []
                for i in range(len(texts)):
                    lbls = set(acc[i]) | {raw[i]}                 # keep raw label, add judge-acceptable ones
                    mh = [1.0 if l in lbls else 0.0 for l in LABELS]
                    tgts.append(mh)
                tr = list(zip(texts, tgts))
                out[city][cond] = train_predict_soft(tr, test[city], tok, f"{city}:soft")
            else:
                tr = list(zip(texts, data[city]["labels"][cond]))
                out[city][cond] = train_predict(tr, test[city], tok, f"{city}:{cond}")
            print(f"[train] {city}/{cond} done (train={len(tr)})"); sys.stdout.flush()
            os.makedirs("results/preds", exist_ok=True)
            json.dump(out, open(f"results/preds/{OUTFILE}", "w"), ensure_ascii=False)
    print("[train] wrote results/preds/e1_distilbert.json"); print("[train] === DONE ==="); sys.stdout.flush()


if __name__ == "__main__":
    main()
