"""
transfer_matrix.py -- Exp C: full 7x7 city-to-city TF-IDF transfer. Train on city A, test on
city B's frozen test set, all 49 cells. Reveals platform/register vs jurisdiction structure and
gives the city-to-city heatmap (Figure B). Also writes docs/fig_transfer_matrix.png.
"""
import os, json
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from eval_common import load_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CAP = 5000
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def vec():
    w = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_features=15000, strip_accents="unicode")
    c = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=15000)
    return w, c


def cap(rows, n=CAP, seed=0):
    return rows if len(rows) <= n else [rows[i] for i in np.random.RandomState(seed).permutation(len(rows))[:n]]


def train_predict(train, test):
    tw = [t for t, _ in train]; ty = [y for _, y in train]; ew = [t for t, _ in test]; ey = [y for _, y in test]
    w, c = vec(); Xtr = hstack([w.fit_transform(tw), c.fit_transform(tw)]).tocsr(); Xte = hstack([w.transform(ew), c.transform(ew)]).tocsr()
    clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced", solver="saga", tol=1e-3); clf.fit(Xtr, ty)
    return f1_score(ey, clf.predict(Xte), average="macro", zero_division=0)


def main():
    sp = load_split()
    cities = list(sp["test"])
    M = np.zeros((len(cities), len(cities)))
    for i, a in enumerate(cities):
        tr = cap(sp["train"][a])
        for j, b in enumerate(cities):
            M[i, j] = train_predict(tr, sp["test"][b])
        print(f"[matrix] trained {a}: " + " ".join(f"{cities[j][:4]}={M[i,j]:.2f}" for j in range(len(cities))))
    json.dump({"cities": cities, "matrix": M.tolist()}, open("results/transfer_matrix.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    lab = [c.replace("_", " ")[:10] for c in cities]
    ax.set_xticks(range(len(cities))); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cities))); ax.set_yticklabels(lab, fontsize=8)
    for i in range(len(cities)):
        for j in range(len(cities)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i,j] < 0.55 else "black", fontsize=7)
    ax.set_xlabel("Test city"); ax.set_ylabel("Train city")
    ax.set_title("City-to-city TF-IDF transfer (macro-F1)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig("docs/fig_transfer_matrix.png", dpi=150); plt.close()
    print("wrote results/transfer_matrix.json + docs/fig_transfer_matrix.png")
    # quick diagonal (in-city) vs off-diagonal (cross) summary
    diag = np.mean([M[i, i] for i in range(len(cities))])
    off = np.mean([M[i, j] for i in range(len(cities)) for j in range(len(cities)) if i != j])
    print(f"mean diagonal (single-city in-city) = {diag:.3f}   mean off-diagonal (single-source cross) = {off:.3f}")


if __name__ == "__main__":
    main()
