"""Fine-tune DistilBERT on HUPD titles -> IPC subclass (539-way) at several data budgets.
Same frozen split as the LLM comparison (hupd_split.json). Prints an accuracy/macroF1 curve.
Runs as a Modal job (data mounted at /app/). Plain script; wrapped via gpu2modal autowrap.
"""
import json, sys, os
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

assert torch.cuda.is_available(), "needs GPU"
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()
MODEL = "distilbert-base-uncased"; MAXLEN = 40; SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
BUDGETS = [500, 2000, 8000, 24768]


def load():
    p = "/app/hupd_split.json" if os.path.exists("/app/hupd_split.json") else "results/hupd_split.json"
    d = json.load(open(p, encoding="utf-8"))
    return d["pool"], d["test"]


class DS(Dataset):
    def __init__(self, rows, tok, l2i):
        self.rows = rows; self.tok = tok; self.l2i = l2i
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        e = self.tok(r["text"], truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, self.l2i[r["label"]]


def run_budget(pool, test, n, tok, labels, l2i, dev):
    from sklearn.metrics import accuracy_score, f1_score
    tr = pool[:n]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5); scaler = torch.amp.GradScaler("cuda")
    dl = DataLoader(DS(tr, tok, l2i), batch_size=64, shuffle=True, num_workers=2)
    epochs = 4 if n <= 2000 else 3
    model.train()
    for ep in range(epochs):
        for batch, y in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}; y = torch.tensor(y).to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.cross_entropy(model(**batch).logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval(); preds = []
    with torch.no_grad():
        for batch, _ in DataLoader(DS(test, tok, l2i), batch_size=128):
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                preds.extend(model(**batch).logits.argmax(-1).cpu().tolist())
    del model; torch.cuda.empty_cache()
    gold = [l2i[r["label"]] for r in test]
    acc = accuracy_score(gold, preds); f1 = f1_score(gold, preds, average="macro", zero_division=0)
    print(f"[train] budget={n:6d} epochs={epochs} distilbert acc={acc:.4f} macroF1={f1:.4f}"); sys.stdout.flush()
    return {"budget": n, "acc": round(float(acc), 4), "macroF1": round(float(f1), 4)}


def main():
    pool, test = load()
    labels = sorted(set(r["label"] for r in pool + test))
    l2i = {l: i for i, l in enumerate(labels)}
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"[train] pool={len(pool)} test={len(test)} labels={len(labels)}"); sys.stdout.flush()
    out = [run_budget(pool, test, min(n, len(pool)), tok, labels, l2i, dev) for n in BUDGETS]
    os.makedirs("/results", exist_ok=True) if os.path.exists("/results") else None
    dst = "/results/hupd_bert.json" if os.path.exists("/results") else "results/hupd_bert.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(f"[train] wrote {dst}"); print("[train] === DONE ==="); sys.stdout.flush()


if __name__ == "__main__":
    main()
