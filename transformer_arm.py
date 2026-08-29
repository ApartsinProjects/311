"""
transformer_arm.py -- Fine-tuned DistilBERT arm for the multi-city 311 benchmark.

Same two protocols as the TF-IDF baseline, so numbers are directly comparable:
  (A) IN-CITY   : fine-tune + test within each city (80/20).
  (B) LOCO      : fine-tune on all other cities, test on the held-out city.

Model: distilbert-base-uncased + linear head. Local RTX 2060 (fp16, max_len 64).
Sanity invariant printed: IN-CITY macro-F1 must exceed LOCO macro-F1 per city.

Usage:
  python transformer_arm.py --cap-per-city 8000 --epochs 3
  python transformer_arm.py --protocol loco --cap-per-city 8000     # LOCO only (headline)
"""
import argparse, json, os, numpy as np, torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from bench_common import load_harmonized, LABELS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "distilbert-base-uncased"
L2I = {l: i for i, l in enumerate(LABELS)}


class DS(Dataset):
    def __init__(self, rows, tok, maxlen=64):
        self.texts = [t for t, _ in rows]
        self.y = [L2I[y] for _, y in rows]
        self.tok = tok; self.maxlen = maxlen
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        e = self.tok(self.texts[i], truncation=True, max_length=self.maxlen,
                     padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, self.y[i]


def train_eval(train_rows, test_rows, tok, epochs, bs, lr):
    # restrict to labels present in training
    tr_labels = set(y for _, y in train_rows)
    test_rows = [(t, y) for t, y in test_rows if y in tr_labels]
    if len(test_rows) < 20 or len(set(y for _, y in test_rows)) < 2:
        return None
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(DEV) if DEV == "cuda" else None
    tl = DataLoader(DS(train_rows, tok), batch_size=bs, shuffle=True, num_workers=0)
    model.train()
    for ep in range(epochs):
        for batch, y in tl:
            batch = {k: v.to(DEV) for k, v in batch.items()}
            y = torch.tensor(y).to(DEV)
            opt.zero_grad()
            if scaler:
                with torch.amp.autocast(DEV):
                    out = model(**batch).logits
                    loss = torch.nn.functional.cross_entropy(out, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                out = model(**batch).logits
                loss = torch.nn.functional.cross_entropy(out, y)
                loss.backward(); opt.step()
    # eval
    model.eval(); preds = []
    el = DataLoader(DS(test_rows, tok), batch_size=64)
    with torch.no_grad():
        for batch, _ in el:
            batch = {k: v.to(DEV) for k, v in batch.items()}
            with torch.amp.autocast(DEV) if DEV == "cuda" else torch.no_grad():
                logits = model(**batch).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    y_true = [y for _, y in test_rows]
    y_pred = [LABELS[p] for p in preds]
    del model; torch.cuda.empty_cache() if DEV == "cuda" else None
    return (f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
            accuracy_score(y_true, y_pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-per-city", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--protocol", choices=["both", "incity", "loco"], default="both")
    ap.add_argument("--out", default="results_distilbert.json")
    args = ap.parse_args()

    print(f"device={DEV}  model={MODEL}  cap={args.cap_per_city}  epochs={args.epochs}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    by_city = load_harmonized(cap=args.cap_per_city, drop_other=True)
    cities = sorted(by_city, key=lambda c: -len(by_city[c]))
    for c in cities:
        print(f"  {c:14s} {len(by_city[c]):>7d} rows  {len(set(y for _,y in by_city[c])):>2d} classes")

    incity, loco = {}, {}
    if args.protocol in ("both", "incity"):
        print("\n=== (A) IN-CITY (DistilBERT) ===")
        for c in cities:
            data = by_city[c]
            keep = {k for k, v in Counter(y for _, y in data).items() if v >= 5}
            data = [(t, y) for t, y in data if y in keep]
            if len(set(y for _, y in data)) < 2:
                print(f"  {c:14s} n/a"); continue
            tr, te = train_test_split(data, test_size=0.2, random_state=0,
                                      stratify=[y for _, y in data])
            r = train_eval(tr, te, tok, args.epochs, args.bs, args.lr)
            if r: incity[c] = r[0]; print(f"  {c:14s} macroF1={r[0]:.3f}  acc={r[1]:.3f}")

    if args.protocol in ("both", "loco"):
        print("\n=== (B) LEAVE-ONE-CITY-OUT (DistilBERT) ===")
        for c in cities:
            train = [d for oc in cities if oc != c for d in by_city[oc]]
            r = train_eval(train, by_city[c], tok, args.epochs, args.bs, args.lr)
            if r:
                loco[c] = r[0]
                gap = incity.get(c, float('nan')) - r[0]
                flag = "  <-- INVARIANT VIOLATION" if (c in incity and r[0] >= incity[c]) else ""
                print(f"  {c:14s} macroF1={r[0]:.3f}  acc={r[1]:.3f}  gap={gap:+.3f}{flag}")

    out = {"incity": incity, "loco": loco}
    if incity and loco:
        mi = np.mean([incity[c] for c in loco if c in incity])
        ml = np.mean([loco[c] for c in loco])
        out["summary"] = {"mean_incity": round(float(mi), 3), "mean_loco": round(float(ml), 3),
                          "transfer_gap": round(float(mi - ml), 3)}
        print(f"\nSUMMARY  mean IN-CITY={mi:.3f}  mean LOCO={ml:.3f}  gap={mi-ml:+.3f}")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
