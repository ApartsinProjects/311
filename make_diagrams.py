"""Generate two schematic figures: corpus-construction flow and taxonomy-boundary graph.
The taxonomy graph edges are the real top confusion pairs from DistilBERT cross-city predictions."""
import os, json
from collections import Counter, defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
csv_split = os.path.join("data", "eval_split.csv")
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 10})

# ---------- Figure A: corpus construction flow (roomy blocks, text fully inside, gap-only arrows) ----------
STAGES = [
    ("Municipal portals", "~12 probed", "#eaf0f6"),
    ("Free-text cities", "7 confirmed", "#dde8f2"),
    ("Harmonize", "545 → 14 classes", "#d0e0ef"),
    ("Drop admin", "categories", "#c3d9ec"),
    ("Informativeness", "filter (−8%)", "#b6d2e9"),
    ("PII scrub", "phones, emails", "#a9cbe6"),
    ("MC311 benchmark", "156k · 3,502 test", "#93c1e1"),
]
BW, BH, GAP, MARGIN = 2.15, 1.15, 0.7, 0.35
n = len(STAGES)
total_w = MARGIN * 2 + n * BW + (n - 1) * GAP
fig, ax = plt.subplots(figsize=(total_w * 0.9, 2.1)); ax.axis("off")
ymid = 0.6
for i, (title, sub, col) in enumerate(STAGES):
    x = MARGIN + i * (BW + GAP)
    ax.add_patch(FancyBboxPatch((x, ymid - BH / 2), BW, BH,
                                boxstyle="round,pad=0.015,rounding_size=0.10",
                                fc=col, ec="#2f5677", lw=1.2))
    cx = x + BW / 2
    ax.text(cx, ymid + 0.17, title, ha="center", va="center", fontsize=9.5, weight="bold", color="#1c3a56")
    ax.text(cx, ymid - 0.20, sub, ha="center", va="center", fontsize=8.8, color="#2f5677")
    if i < n - 1:                                   # arrow strictly inside the gap
        ax.add_patch(FancyArrowPatch((x + BW + 0.07, ymid), (x + BW + GAP - 0.07, ymid),
                                     arrowstyle="-|>", mutation_scale=13, color="#2f5677", lw=1.4))
ax.set_xlim(0, total_w); ax.set_ylim(0, 1.2)
plt.savefig("docs/fig_corpus_flow.png", dpi=150, bbox_inches="tight", pad_inches=0.12); plt.close()
print("wrote docs/fig_corpus_flow.png")

# ---------- Figure B: taxonomy-boundary graph ----------
LABELS = ["Waste_Sanitation", "Streets_Sidewalks", "Street_Lighting", "Traffic_Signals_Signs",
          "Trees_Vegetation", "Graffiti_Postings", "Parking_Vehicles", "Property_Housing_Code",
          "Water_Sewer_Drainage", "Homelessness", "Animals_Pests", "Noise", "Transit", "Parks_Recreation"]
# real confusion pairs from DistilBERT LOCO
gold = defaultdict(list)
import csv
csv.field_size_limit(10**7)
with open(csv_split, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["role"] == "test":
            gold[r["city"]].append(r["label"])
pred = json.load(open("results/preds/distilbert.json"))["loco"]
pairs = Counter()
for c in gold:
    for g, p in zip(gold[c], pred[c]):
        if g != p:
            pairs[tuple(sorted((g, p)))] += 1
top = pairs.most_common(12)
maxc = top[0][1] if top else 1

ang = np.linspace(0, 2 * np.pi, len(LABELS), endpoint=False)
pos = {l: (np.cos(a), np.sin(a)) for l, a in zip(LABELS, ang)}
fig, ax = plt.subplots(figsize=(7.4, 7.0)); ax.axis("off")
for (a, b), cnt in top:
    (x1, y1), (x2, y2) = pos[a], pos[b]
    ax.plot([x1, x2], [y1, y2], "-", color="#e45756", lw=0.6 + 3.4 * cnt / maxc, alpha=0.55, zorder=1)
for l, (x, y) in pos.items():
    ax.scatter([x], [y], s=260, c="#4c78a8", zorder=2, edgecolors="white", linewidths=1)
    ha = "left" if x > 0.05 else ("right" if x < -0.05 else "center")
    ax.text(x * 1.14, y * 1.14, l.replace("_", " "), ha=ha, va="center", fontsize=8.2, zorder=3)
ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.4, 1.4)
ax.set_title("Taxonomy boundaries: top cross-city confusion pairs (DistilBERT)", fontsize=10)
plt.tight_layout(); plt.savefig("docs/fig_taxonomy_boundary.png", dpi=150, bbox_inches="tight"); plt.close()
print("wrote docs/fig_taxonomy_boundary.png")
print("top confusion pairs:", [(f"{a[:10]}~{b[:10]}", n) for (a, b), n in top[:6]])
