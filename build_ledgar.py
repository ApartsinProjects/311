"""Build the LEDGAR split (legal contract-provision classification) with the project's standard recipe:
top-K provision types, single source, proportional fixed test (1500), stratified budget (2000),
novel/dup split vs the budget. Regenerates results/ledgar_split.json from the public LexGLUE dataset.

  python build_ledgar.py [K]
"""
import sys, json
import numpy as np
from collections import Counter
from scipy import stats
from datasets import load_dataset
import semclf

K = int(sys.argv[1]) if len(sys.argv) > 1 else 25
ds = load_dataset("coastalcph/lex_glue", "ledgar")
names = ds["train"].features["label"].names
all_rows = [{"text": r["text"], "label": names[r["label"]]} for r in ds["train"]]
c = Counter(r["label"] for r in all_rows)
keep = {l for l, _ in c.most_common(K)}
R = [r for r in all_rows if r["label"] in keep]
rng = np.random.RandomState(0); idx = rng.permutation(len(R))
test = [R[i] for i in idx[:1500]]; pool = [R[i] for i in idx[1500:]]
ref = set(semclf.norm(r["text"]) for r in semclf.stratified_budget(pool, 2000))
novel = [r for r in test if semclf.norm(r["text"]) not in ref]
dup = [r for r in test if semclf.norm(r["text"]) in ref]
json.dump({"pool": pool, "test": novel, "test_dup": dup},
          open("results/ledgar_split.json", "w"), ensure_ascii=False)
cp = Counter(r["label"] for r in pool); ct = Counter(r["label"] for r in test)
labs = sorted(cp, key=lambda x: -cp[x]); n = len(test); N = len(pool)
obs = np.array([ct[l] for l in labs], float); exp = np.array([cp[l] / N * n for l in labs], float)
chi2 = ((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum(); p = 1 - stats.chi2.cdf(chi2, len(labs) - 1)
print(f"LEDGAR top-{K}: pool={len(pool)} test={n} (novel={len(novel)} dup={len(dup)}) "
      f"classes={K} maj={max(ct.values())/n:.3f} test_prop_p={p:.3f}")
