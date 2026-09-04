"""Fine-tune DistilBERT on the 2000-example budget as the real 'fine-tuned model' baseline (vs the
TF-IDF+LR linear reference). Same budget/test as every other method. Reports test accuracy + lift.

  python finetune_bert.py [task] [epochs]
"""
import sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import semclf
from semclf import TASKS, score, stratified_budget

def run(task, epochs=4, max_len=256, bs=16, lr=2e-5, seed=0):
    torch.manual_seed(seed)
    T = TASKS[task]
    bud = stratified_budget(T.pool, 2000, seed=seed)
    LBL = sorted(set(r["label"] for r in bud) | set(T.LBL)); L2I = {l: i for i, l in enumerate(LBL)}
    test = T.test + T.test_dup
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(LBL)).to(dev)

    def enc(texts):
        e = tok([t[:2000] for t in texts], truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    tr_x, tr_m = enc([r["text"] for r in bud]); tr_y = torch.tensor([L2I[r["label"]] for r in bud])
    te_x, te_m = enc([r["text"] for r in test]); te_gold = [r["label"] for r in test]
    dl = DataLoader(TensorDataset(tr_x, tr_m, tr_y), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"FINETUNE-BERT {task}: train={len(bud)} test={len(test)} classes={len(LBL)} dev={dev}")
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, mb, yb in dl:
            opt.zero_grad()
            out = model(input_ids=xb.to(dev), attention_mask=mb.to(dev), labels=yb.to(dev))
            out.loss.backward(); opt.step(); tot += out.loss.item()
        print(f"  epoch {ep+1}: loss={tot/len(dl):.4f}")

    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(test), 64):
            logits = model(input_ids=te_x[i:i+64].to(dev), attention_mask=te_m[i:i+64].to(dev)).logits
            preds += [LBL[j] for j in logits.argmax(1).cpu().numpy()]
    a, ci, _ = score(T, preds, te_gold)
    # reference zero-shot from cached baselines if present
    zs = json.load(open(f"results/baselines_{task}.json")).get("zero_shot") if __import__("os").path.exists(f"results/baselines_{task}.json") else None
    lift = f"  lift_vs_zs={a-zs:+.4f}" if zs else ""
    print(f"  DistilBERT test acc={a:.4f} CI=({ci[0]:.3f},{ci[1]:.3f}){lift}")
    json.dump({"task": task, "distilbert": a, "ci": ci, "zero_shot": zs, "epochs": epochs},
              open(f"results/distilbert_{task}.json", "w"), indent=2)
    print(f"  wrote results/distilbert_{task}.json")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "banking",
        int(sys.argv[2]) if len(sys.argv) > 2 else 4)
