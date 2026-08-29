"""Aligned TF-IDF eval: in-city and LOCO on the FROZEN test rows. Saves per-example preds."""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from eval_common import load_split, save_preds

CAP = 8000  # per-city train cap for the LOCO pool


def vec():
    w = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_features=30000,
                        strip_accents="unicode", lowercase=True)
    c = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        max_features=30000, lowercase=True)
    return w, c


def cap_rows(rows, cap, seed=0):
    if len(rows) <= cap:
        return rows
    idx = np.random.RandomState(seed).permutation(len(rows))[:cap]
    return [rows[i] for i in idx]


def train_predict(train, test):
    tr_txt = [t for t, _ in train]; tr_y = [y for _, y in train]
    te_txt = [t for t, _ in test]
    w, c = vec()
    Xtr = hstack([w.fit_transform(tr_txt), c.fit_transform(tr_txt)]).tocsr()
    Xte = hstack([w.transform(te_txt), c.transform(te_txt)]).tocsr()
    clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced", solver="saga", tol=1e-3)
    clf.fit(Xtr, tr_y)
    return list(clf.predict(Xte))


def main():
    sp = load_split()
    cities = list(sp["test"])
    preds = {"incity": {}, "loco": {}}
    for c in cities:
        test = sp["test"][c]
        # in-city
        preds["incity"][c] = train_predict(cap_rows(sp["train"][c], CAP), test)
        print(f"[tfidf] in-city {c} done ({len(test)} test)")
        # loco
        pool = []
        for oc in cities:
            if oc != c:
                pool += cap_rows(sp["train"][oc], CAP)
        preds["loco"][c] = train_predict(pool, test)
        print(f"[tfidf] loco    {c} done")
    save_preds("tfidf", preds)


if __name__ == "__main__":
    main()
