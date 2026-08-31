"""E1 pilot: does REPAIRING training labels with the acceptable-set judge improve cross-city transfer?
Cheap decisive test: one held-out city, single judge, local TF-IDF arm, with a random-drop control.
  python pilot_e1.py submit   # judge the held-out fold's training pool (batch), save pool + state
  python pilot_e1.py eval      # collect judge, build repaired/control pools, retrain TF-IDF, report
"""
import sys, os, json, numpy as np
from eval_common import load_split
from eval_tfidf import vec, cap_rows, train_predict
from defensibility_judge import SYS, prompt, parse_set
from sklearn.metrics import f1_score

HELD = "Gainesville"
CAP = 5000
TAG = "pilot_e1_judge"
POOL_F = "results/pilot_e1_pool.json"
JUDGE_MODEL = "gpt-4o-mini"


def build_pool():
    sp = load_split()
    pool = []
    for c in sp["train"]:
        if c == HELD:
            continue
        pool += [(t, y) for t, y in cap_rows(sp["train"][c], CAP)]
    return pool, sp


def submit():
    pool, _ = build_pool()
    json.dump(pool, open(POOL_F, "w"), ensure_ascii=False)
    import openai_batch as B
    items = [(f"r{i}", {"messages": [{"role": "system", "content": SYS},
                                     {"role": "user", "content": prompt(t)}],
                        "temperature": 0, "max_tokens": 30}) for i, (t, y) in enumerate(pool)]
    B.submit_chat_batch(JUDGE_MODEL, items, tag=TAG)
    print(f"submitted {len(items)} judge requests for held-out={HELD}; collect with: python pilot_e1.py eval")


def evaluate():
    import openai_batch as B
    res = B.collect_chat_batch(tag=TAG)
    if res is None:
        print("judge batch not ready yet; rerun: python pilot_e1.py eval"); return
    pool = [tuple(x) for x in json.load(open(POOL_F, encoding="utf-8"))]
    jsets = [set(parse_set(res.get(f"r{i}", ""))) - {"UNPARSED"} for i in range(len(pool))]
    sp = load_split()
    test = sp["test"][HELD]
    # judge-clean subset of the held-out TEST (gold in its own acceptable set)
    accH = json.load(open("results/preds/acceptable_sets.json"))[HELD]
    clean_idx = [i for i, (t, y) in enumerate(test) if y in set(accH[i])]

    def macro(train, subset=None):
        preds = train_predict(train, test)
        g = [y for _, y in test]
        if subset is not None:
            g = [g[i] for i in subset]; preds = [preds[i] for i in subset]
        return f1_score(g, preds, average="macro", zero_division=0)

    raw = pool
    rej = [i for i in range(len(pool)) if jsets[i] and pool[i][1] not in jsets[i]]  # judge rejects raw label
    d_drop = [pool[i] for i in range(len(pool)) if i not in set(rej)]
    d_relabel = [(pool[i][0], (sorted(jsets[i])[0] if i in set(rej) else pool[i][1])) for i in range(len(pool))]
    rng = np.random.RandomState(0)
    rand_drop_idx = set(rng.choice(len(pool), len(rej), replace=False).tolist())
    rand_drop = [pool[i] for i in range(len(pool)) if i not in rand_drop_idx]

    print(f"held-out={HELD}  pool={len(pool)}  judge-rejected raw labels={len(rej)} ({len(rej)/len(pool):.1%})")
    print(f"held-out test rows={len(test)}  judge-clean={len(clean_idx)}")
    print(f"{'condition':14s}{'macroF1_all':>12s}{'macroF1_clean':>14s}")
    for name, tr in [("raw", raw), ("repair_drop", d_drop), ("repair_relabel", d_relabel), ("random_drop", rand_drop)]:
        print(f"{name:14s}{macro(tr):12.4f}{macro(tr, clean_idx):14.4f}")
    print("\n[invariant] repair_drop should beat random_drop; repair_relabel vs raw is the training-payoff signal.")


if __name__ == "__main__":
    {"submit": submit, "eval": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()
