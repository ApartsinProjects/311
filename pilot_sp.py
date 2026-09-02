"""Semantic-Policy decomposition PILOT (falsification test for the pivot).

Tests the two load-bearing claims of `semantic_policy_project.md` on 3 cities, cheaply:
  Exp A (policy residual):  is I(Y ; A | S) > 0 ?  i.e. after controlling for a compositional
                            semantic state S, does authority identity still change the admin label?
  Exp C (sample efficiency): on an UNSEEN authority, does S -> Y adapt with fewer labels than
                             a frozen sentence-embedding -> Y ? (leftward learning-curve shift)

S = 5 compositional attributes (object/state/location/effect/action) extracted by an LLM (gpt-4o-mini,
batch). Embedding baseline = OpenAI text-embedding-3-small (frozen, hosted). Policy mapper = LR.

  python pilot_sp.py submit     # annotate the pool (attribute batch) + cache embeddings
  python pilot_sp.py analyze    # collect batch, run Exp A + Exp C, print + save results/pilot_sp.json

Pre-registered invariants (a violation = bug, not a finding):
  * I_perm (city labels shuffled within each semantic group) must be ~0; I_obs is meaningful only
    relative to this permutation null (plug-in MI is upward-biased on small groups).
  * in-sample: LR(Y ~ S+A) accuracy >= LR(Y ~ S) (nested feature set).
  * few-shot: at the largest budget both arms are well above the majority-class floor.
"""
import sys, os, json, re, time
import numpy as np
from collections import defaultdict, Counter
from eval_common import load_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

CITIES = ["BatonRouge", "Bloomington", "Gainesville"]
HELD = "Gainesville"                 # unseen authority for Exp C
N_TRAIN_A = 400                      # rows/city for the Exp-A policy-residual pool
N_ADAPT = 800                        # held-out train rows to sample few-shot from (Exp C)
TAG = "pilot_sp_attr"
POOL_F = "results/pilot_sp_pool.json"
EMB_F = "results/pilot_sp_emb.json"
ATTR_MODEL = "gpt-4o-mini"
EMB_MODEL = "text-embedding-3-small"

# ---- compositional semantic schema (interpretable, ~fixed small vocab; unknown -> 'other') ----
ATTRS = {
 "object":   ["tree", "waste", "vehicle", "road_sidewalk", "streetlight", "traffic_sign",
              "water_sewer", "building", "animal", "graffiti", "park", "person", "other"],
 "state":    ["fallen", "broken", "abandoned", "blocked", "overflowing", "out", "leaking",
              "overgrown", "dirty", "other"],
 "location": ["road", "sidewalk", "private_property", "public_property", "park", "waterway", "other"],
 "effect":   ["obstruction", "safety_hazard", "cleanliness", "service_failure", "nuisance",
              "property_damage", "other"],
 "action":   ["remove", "repair", "inspect", "collect", "maintain", "other"],
}
FIELDS = list(ATTRS)
SYS = ("You extract a compositional semantic description of a municipal 311 service request. "
       "Judge ONLY from the text; do not guess the city's department. Output STRICT JSON with exactly "
       "these keys: object, state, location, effect, action. Each value MUST be one token from its "
       "allowed list (use 'other' if none fits).")


def prompt(t):
    opts = "\n".join(f"  {k}: {', '.join(ATTRS[k])}" for k in FIELDS)
    return (f"Allowed values per key:\n{opts}\n\nRequest:\n\"\"\"{t[:1200]}\"\"\"\n\n"
            f"Reply with ONLY the JSON object.")


def parse(o):
    """Coerce model output to a valid attribute dict; unknown/missing -> 'other'."""
    d = {}
    m = re.search(r"\{.*\}", o or "", re.S)
    raw = {}
    if m:
        try:
            raw = json.loads(m.group(0))
        except Exception:
            raw = {}
    for k in FIELDS:
        v = str(raw.get(k, "other")).strip().lower().replace(" ", "_")
        d[k] = v if v in ATTRS[k] else "other"
    return d


# ------------------------------- submit -------------------------------
def build_pool():
    sp = load_split()
    pool = []  # (text, label, city, split_role)
    rng = np.random.RandomState(0)
    for c in CITIES:
        rows = sp["train"][c]
        idx = rng.permutation(len(rows))[:N_TRAIN_A]
        for i in idx:
            pool.append((rows[i][0], rows[i][1], c, "train_A"))
    # held-out adaptation pool + full held-out test (for Exp C)
    hrows = sp["train"][HELD]
    idx = rng.permutation(len(hrows))[:N_ADAPT]
    seen = set(i for i in idx)  # keep Exp-C adapt rows distinct from the Exp-A Gainesville rows below
    for i in idx:
        pool.append((hrows[i][0], hrows[i][1], HELD, "adapt"))
    for (t, y) in sp["test"][HELD]:
        pool.append((t, y, HELD, "test"))
    return pool


def _embed(texts):
    from openai_batch import client
    out = []
    for i in range(0, len(texts), 256):
        chunk = [t[:1200] for t in texts[i:i+256]]
        for a in range(4):
            try:
                r = client.embeddings.create(model=EMB_MODEL, input=chunk)
                out.extend([d.embedding for d in r.data]); break
            except Exception as e:
                if a == 3:
                    raise
                time.sleep(2*(a+1))
        print(f"  embedded {min(i+256, len(texts))}/{len(texts)}"); sys.stdout.flush()
    return out


def submit():
    os.makedirs("results", exist_ok=True)
    pool = build_pool()
    json.dump(pool, open(POOL_F, "w"), ensure_ascii=False)
    print(f"pool={len(pool)} rows  by_role={Counter(r[3] for r in pool)}")
    # attribute extraction batch
    from openai_batch import submit_chat_batch
    items = [(f"r{i}", {"messages": [{"role": "system", "content": SYS},
                                     {"role": "user", "content": prompt(t)}],
                        "temperature": 0, "max_tokens": 120})
             for i, (t, y, c, r) in enumerate(pool)]
    submit_chat_batch(ATTR_MODEL, items, tag=TAG)
    # embeddings only needed for the held-out city (Exp C): adapt + test rows
    emb_idx = [i for i, (t, y, c, r) in enumerate(pool) if r in ("adapt", "test")]
    print(f"embedding {len(emb_idx)} held-out rows via {EMB_MODEL} ...")
    embs = _embed([pool[i][0] for i in emb_idx])
    json.dump({"idx": emb_idx, "emb": embs}, open(EMB_F, "w"))
    print(f"wrote {EMB_F}. Now: python pilot_sp.py analyze")


# ------------------------------- analyze -------------------------------
def _S_matrix(attr_rows, enc=None):
    """One-hot the 5 categorical attributes."""
    M = [[a[k] for k in FIELDS] for a in attr_rows]
    if enc is None:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return enc.fit_transform(M), enc
    return enc.transform(M), enc


def _mi_within_group(profiles, cities, labels, min_group=8, n_perm=200, seed=0):
    """Plug-in estimate of I(Y;A|S) using semantic-profile groups, with a permutation null.
    Returns (I_obs, I_perm_mean, I_perm_std, n_informative_groups, examples)."""
    groups = defaultdict(list)
    for i in range(len(profiles)):
        groups[profiles[i]].append(i)
    tot = len(profiles)

    def cond_mi(city_arr):
        I = 0.0
        for g, idx in groups.items():
            if len(idx) < min_group:
                continue
            cA = city_arr[idx]; yy = labels[idx]
            if len(set(cA.tolist())) < 2:      # group must span >=2 cities to carry A-info
                continue
            pg = len(idx) / tot
            # H(Y|g)
            hy = _entropy(yy)
            # H(Y|g,A)
            hya = 0.0
            for a in set(cA.tolist()):
                sel = cA == a
                hya += (sel.sum()/len(idx)) * _entropy(yy[sel])
            I += pg * max(hy - hya, 0.0)
        return I

    cities = np.asarray(cities); labels = np.asarray(labels)
    I_obs = cond_mi(cities)
    rng = np.random.RandomState(seed)
    perms = []
    for _ in range(n_perm):
        cc = cities.copy()
        for g, idx in groups.items():                 # shuffle city labels WITHIN each group -> null
            if len(idx) >= min_group:
                cc[idx] = rng.permutation(cc[idx])
        perms.append(cond_mi(cc))
    perms = np.array(perms)
    # example informative groups (biggest H(Y|g)-drop contributors)
    ex = []
    for g, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(idx) < min_group:
            continue
        cA = np.asarray(cities)[idx]
        if len(set(cA.tolist())) < 2:
            continue
        dist = {}
        for a in sorted(set(cA.tolist())):
            sel = cA == a
            c = Counter(np.asarray(labels)[idx][sel])
            top = c.most_common(1)[0]
            dist[a] = f"{top[0]} {top[1]}/{sel.sum()}"
        ex.append({"profile": g, "n": len(idx), "by_city": dist})
        if len(ex) >= 8:
            break
    n_info = sum(1 for g, idx in groups.items()
                 if len(idx) >= min_group and len(set(np.asarray(cities)[idx].tolist())) >= 2)
    return I_obs, float(perms.mean()), float(perms.std()), n_info, ex


def _entropy(arr):
    n = len(arr)
    if n == 0:
        return 0.0
    c = Counter(arr.tolist() if hasattr(arr, "tolist") else arr)
    return -sum((v/n) * np.log2(v/n) for v in c.values())


def exp_A(pool, attrs):
    """Policy residual: predictive (Y~S vs Y~S+A) + information-theoretic (I(Y;A|S) vs permutation)."""
    idx = [i for i, (t, y, c, r) in enumerate(pool) if r == "train_A"]
    A = [pool[i][2] for i in idx]; Y = [pool[i][1] for i in idx]
    attr_rows = [attrs[i] for i in idx]
    S, enc = _S_matrix(attr_rows)
    cityenc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Acol = cityenc.fit_transform(np.array(A).reshape(-1, 1))
    SA = np.hstack([S, Acol])
    y = np.array(Y)
    # StratifiedKFold(5) needs every class >=5 rows; drop rarer classes from the predictive comparison
    cnt = Counter(y.tolist())
    keep = np.array([cnt[v] >= 5 for v in y])
    Sk, SAk, yk = S[keep], SA[keep], y[keep]
    n_dropped = int((~keep).sum())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def cv(X):
        clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
        f1 = cross_val_score(clf, X, yk, cv=skf, scoring="f1_macro")
        acc = cross_val_score(clf, X, yk, cv=skf, scoring="accuracy")
        return f1.mean(), acc.mean()

    f1_S, acc_S = cv(Sk)
    f1_SA, acc_SA = cv(SAk)
    # in-sample nested invariant
    clfS = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced").fit(S, y)
    clfSA = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced").fit(SA, y)
    ins_S = accuracy_score(y, clfS.predict(S)); ins_SA = accuracy_score(y, clfSA.predict(SA))
    # information-theoretic: coarse semantic key (object+state+location) -> denser, interpretable groups
    GKEY = ["object", "state", "location"]
    profiles = ["|".join(a[k] for k in GKEY) for a in attr_rows]
    I_obs, I_perm, I_perm_sd, n_info, ex = _mi_within_group(profiles, A, Y)
    return {
        "n": len(idx), "n_dropped_rareclass": n_dropped, "group_key": GKEY, "cities": CITIES,
        "cv_macroF1_S": round(f1_S, 4), "cv_macroF1_S+A": round(f1_SA, 4),
        "cv_acc_S": round(acc_S, 4), "cv_acc_S+A": round(acc_SA, 4),
        "insample_acc_S": round(ins_S, 4), "insample_acc_S+A": round(ins_SA, 4),
        "I(Y;A|S)_obs_bits": round(I_obs, 4), "I_perm_null_bits": round(I_perm, 4),
        "I_perm_null_sd": round(I_perm_sd, 4),
        "I_zscore": round((I_obs - I_perm) / (I_perm_sd + 1e-9), 2),
        "n_informative_groups": n_info, "examples": ex,
    }


def exp_C(pool, attrs, seeds=8):
    """Few-shot on unseen authority: S->Y vs embedding->Y, k examples/class, mean+/-std macroF1."""
    emb = json.load(open(EMB_F, encoding="utf-8"))
    emb_of = {emb["idx"][j]: np.array(emb["emb"][j], dtype=np.float32) for j in range(len(emb["idx"]))}
    adapt = [i for i, r in enumerate(pool) if r[3] == "adapt"]
    test = [i for i, r in enumerate(pool) if r[3] == "test"]
    # frozen features
    _, enc = _S_matrix([attrs[i] for i in adapt])
    S_adapt, _ = _S_matrix([attrs[i] for i in adapt], enc)
    S_test, _ = _S_matrix([attrs[i] for i in test], enc)
    E_adapt = np.vstack([emb_of[i] for i in adapt]); E_test = np.vstack([emb_of[i] for i in test])
    y_adapt = np.array([pool[i][1] for i in adapt]); y_test = np.array([pool[i][1] for i in test])
    by_cls = defaultdict(list)
    for j, yy in enumerate(y_adapt):
        by_cls[yy].append(j)
    maj = Counter(y_test).most_common(1)[0]
    floor = maj[1] / len(y_test)

    def fewshot(Xa, Xt, k, seed):
        rng = np.random.RandomState(seed)
        sel = []
        for cls, js in by_cls.items():
            take = min(k, len(js))
            sel += rng.choice(js, take, replace=False).tolist()
        clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
        clf.fit(Xa[sel], y_adapt[sel])
        return f1_score(y_test, clf.predict(Xt), average="macro", zero_division=0)

    curve = {}
    for k in [2, 5, 10, 25]:
        sm = [fewshot(S_adapt, S_test, k, s) for s in range(seeds)]
        em = [fewshot(E_adapt, E_test, k, s) for s in range(seeds)]
        curve[k] = {"semantic": [round(np.mean(sm), 4), round(np.std(sm), 4)],
                    "embedding": [round(np.mean(em), 4), round(np.std(em), 4)],
                    "delta": round(np.mean(sm) - np.mean(em), 4)}
    return {"held_out": HELD, "n_test": len(test), "majority_floor": round(floor, 4),
            "n_classes_adapt": len(by_cls), "curve": curve}


def analyze():
    from openai_batch import collect_chat_batch
    res = collect_chat_batch(tag=TAG)
    if res is None:
        print("attribute batch not ready; rerun: python pilot_sp.py analyze"); return
    pool = [tuple(x) for x in json.load(open(POOL_F, encoding="utf-8"))]
    attrs = [parse(res.get(f"r{i}", "")) for i in range(len(pool))]
    # sanity: how many parsed to all-'other' (extraction failure)?
    allother = sum(1 for a in attrs if all(a[k] == "other" for k in FIELDS))
    print(f"parsed {len(attrs)} attribute rows; all-'other' (likely parse fail)={allother} "
          f"({allother/len(attrs):.1%})")
    for k in FIELDS:
        print(f"  {k:9s}: {dict(Counter(a[k] for a in attrs).most_common(5))}")
    A = exp_A(pool, attrs)
    C = exp_C(pool, attrs)
    out = {"exp_A_policy_residual": A, "exp_C_sample_efficiency": C}
    json.dump(out, open("results/pilot_sp.json", "w"), indent=2)
    print("\n===== EXP A: policy residual I(Y;A|S) =====")
    for k in ["n", "cv_macroF1_S", "cv_macroF1_S+A", "cv_acc_S", "cv_acc_S+A",
              "insample_acc_S", "insample_acc_S+A", "I(Y;A|S)_obs_bits", "I_perm_null_bits",
              "I_perm_null_sd", "I_zscore", "n_informative_groups"]:
        print(f"  {k:22s} {A[k]}")
    print("  example policy-split semantic groups (same S, different city -> different label):")
    for e in A["examples"][:6]:
        print(f"    [{e['n']:3d}] {e['profile']}  ->  {e['by_city']}")
    print("\n===== EXP C: few-shot on unseen authority =====")
    print(f"  held-out={C['held_out']}  n_test={C['n_test']}  majority_floor={C['majority_floor']}  "
          f"classes={C['n_classes_adapt']}")
    print(f"  {'k/class':>8s} {'semantic':>16s} {'embedding':>16s} {'delta':>8s}")
    for k, v in C["curve"].items():
        s = f"{v['semantic'][0]:.3f}±{v['semantic'][1]:.3f}"
        e = f"{v['embedding'][0]:.3f}±{v['embedding'][1]:.3f}"
        print(f"  {k:>8d} {s:>16s} {e:>16s} {v['delta']:>8.3f}")
    print("\nwrote results/pilot_sp.json")
    # invariant checks
    print("\n[invariants]")
    print(f"  I_obs >> I_perm_null?  {A['I(Y;A|S)_obs_bits']} vs {A['I_perm_null_bits']} "
          f"(z={A['I_zscore']})  {'PASS' if A['I_zscore'] > 3 else 'WEAK/FAIL'}")
    print(f"  in-sample acc S+A >= S? {A['insample_acc_S+A']} >= {A['insample_acc_S']} "
          f"{'PASS' if A['insample_acc_S+A'] >= A['insample_acc_S'] - 1e-9 else 'BUG'}")
    big = C['curve'][25]
    print(f"  largest-budget both > floor? sem {big['semantic'][0]}, emb {big['embedding'][0]} "
          f"> {C['majority_floor']}  {'PASS' if min(big['semantic'][0], big['embedding'][0]) > C['majority_floor'] else 'CHECK'}")


if __name__ == "__main__":
    {"submit": submit, "analyze": analyze}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()
