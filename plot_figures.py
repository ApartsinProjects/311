"""Generate paper figures from results/final_scores.json + results/preds/. White background."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from eval_common import load_split, LABELS

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.size": 11})
OUT = os.path.join("docs")
os.makedirs(OUT, exist_ok=True)
S = json.load(open("results/final_scores.json", encoding="utf-8"))["arms"]

# ---- F1: transfer gap, in-city vs cross-city pooled macro-F1 with 95% CI ----
def bar(ax, key, x, color, label=None):
    v = S[key]["pooled_macroF1"]; lo, hi = S[key]["pooled_CI"]
    ax.bar(x, v, 0.8, color=color, label=label)
    ax.errorbar(x, v, yerr=[[v-lo],[hi-v]], fmt="none", ecolor="#333", capsize=4, lw=1.2)
    ax.text(x, hi+0.015, f"{v:.2f}", ha="center", fontsize=9)

fig, ax = plt.subplots(figsize=(6.2, 3.8))
groups = [("TF-IDF", "tfidf/incity", "tfidf/loco", None),
          ("DistilBERT", "distilbert/incity", "distilbert/loco", None),
          ("LLM zero-shot", None, "llm_gpt4omini/zeroshot", None)]
x = 0; ticks = []; labs = []
for name, inc, cross, _ in groups:
    if inc:
        bar(ax, inc, x, "#4c78a8"); ticks.append(x); labs.append(f"{name}\nin-city"); x += 1
    bar(ax, cross, x, "#e45756"); ticks.append(x); labs.append(f"{name}\ncross-city"); x += 1
    x += 0.4
ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("Macro-F1 (pooled, 95% CI)"); ax.set_ylim(0, 1.0)
ax.set_title("In-city vs cross-jurisdiction transfer")
ax.axhline(0, color="#ccc", lw=.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4c78a8", label="in-city"), Patch(color="#e45756", label="cross-city")],
          fontsize=9, loc="upper right")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_transfer.png", dpi=150); plt.close()
print("wrote docs/fig_transfer.png")

# ---- F2: defensibility, strict vs lenient accuracy ----
fig, ax = plt.subplots(figsize=(6.2, 3.6))
arms = ["tfidf/loco", "distilbert/loco", "llm_gpt4omini/zeroshot"]
names = ["TF-IDF\ncross-city", "DistilBERT\ncross-city", "LLM\nzero-shot"]
xs = np.arange(len(arms)); w = 0.38
strict = [S[a]["pooled_strict_acc"] for a in arms]
lenient = [S[a]["pooled_lenient_acc"] for a in arms]
ax.bar(xs-w/2, strict, w, color="#e45756", label="strict")
ax.bar(xs+w/2, lenient, w, color="#59a14f", label="lenient (defensibility-adjusted)")
for i,(s,l) in enumerate(zip(strict,lenient)):
    ax.text(i-w/2, s+.01, f"{s:.2f}", ha="center", fontsize=8)
    ax.text(i+w/2, l+.01, f"{l:.2f}", ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Accuracy"); ax.set_ylim(0,1.05); ax.legend(fontsize=9)
ax.set_title("Most cross-city 'errors' are judge-defensible")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_defensibility.png", dpi=150); plt.close()
print("wrote docs/fig_defensibility.png")

# ---- F3: pooled LOCO confusion matrix (DistilBERT) ----
sp = load_split(); preds = json.load(open("results/preds/distilbert.json", encoding="utf-8"))["loco"]
gold, pred = [], []
for c, rows in sp["test"].items():
    gs = [y for _, y in rows]; ps = preds[c]
    gold += gs; pred += ps
present = [l for l in LABELS if l in set(gold)]
cm = confusion_matrix(gold, pred, labels=present, normalize="true")
fig, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
short = [l.replace("_", " ")[:16] for l in present]
ax.set_xticks(range(len(present))); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(present))); ax.set_yticklabels(short, fontsize=8)
for i in range(len(present)):
    for j in range(len(present)):
        if cm[i,j] >= 0.08:
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm[i,j]>0.5 else "#333", fontsize=7)
ax.set_xlabel("Predicted"); ax.set_ylabel("True (city label)")
ax.set_title("DistilBERT cross-city confusion (row-normalized)")
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_confusion.png", dpi=150); plt.close()
print("wrote docs/fig_confusion.png")

# ---- F4: per-city cross-city macro-F1 by arm ----
sp_cities = list(load_split()["test"])
fig, ax = plt.subplots(figsize=(7.0, 3.8))
xs = np.arange(len(sp_cities)); w = 0.26
for k,(arm,col,lab) in enumerate([("tfidf/loco","#4c78a8","TF-IDF"),
                                   ("distilbert/loco","#f58518","DistilBERT"),
                                   ("llm_gpt4omini/zeroshot","#59a14f","LLM zero-shot")]):
    vals = [S[arm]["per_city"].get(c,{}).get("macroF1",0) for c in sp_cities]
    ax.bar(xs + (k-1)*w, vals, w, color=col, label=lab)
ax.set_xticks(xs); ax.set_xticklabels([c.replace("_"," ") for c in sp_cities], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Cross-city macro-F1"); ax.set_ylim(0,1.0); ax.legend(fontsize=8)
ax.set_title("Per-held-out-city transfer (note SF degeneracy)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_percity.png", dpi=150); plt.close()
print("wrote docs/fig_percity.png")
