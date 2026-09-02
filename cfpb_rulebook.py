"""CFPB cross-COMPANY replication of the rulebook-transfer pilot (second domain).

Same idea as pilot_rulebook.py, but the 'jurisdiction' is the COMPANY and the label is the CFPB
product. Learn a filing-convention rulebook from where a text-only classifier disagrees with the
recorded product on SOURCE companies, transfer it to a held-out company, measure hard-case recovery.
Data: data/cfpb_company.json  (fields c=company, p=product, t=narrative), 4 big banks x 200.

  python cfpb_rulebook.py submit    # round-1 base classification (batch)
  python cfpb_rulebook.py induce     # mine disagreements -> rulebook -> round-2 batch
  python cfpb_rulebook.py report     # hard-case recovery per arm
"""
import sys, os, json
import numpy as np
from collections import Counter
from openai_batch import client, submit_chat_batch, collect_chat_batch
from pilot_rulebook import _induce_rules, _corrupt   # reuse rule induction + placebo control

HELD = "CAPITAL"
N_TEST = 100          # held-out test rows
N_UPDATE = 50         # held-out update rows (disjoint)
POOL_F = "results/cfpb_rb_pool.json"
RULES_F = "results/cfpb_rb_rules.json"
BASE_TAG = "cfpb_rb_base"
RULES_TAG = "cfpb_rb_rules"
MODEL = "gpt-4o-mini"

GLOSS = {
 "Checking or savings account": "deposit account: checking/savings, fees, overdraft, holds, closures",
 "Credit card": "credit or charge card: billing, rewards, disputes, interest",
 "Credit reporting or other personal consumer reports": "credit report errors/disputes, bureaus, inquiries",
 "Mortgage": "home loan: servicing, escrow, foreclosure, modification",
 "Money transfer, virtual currency, or money service": "transfers, wallets, crypto, money services",
 "Debt collection": "a collector's practices, collection calls, validation",
 "Vehicle loan or lease": "auto loan or lease",
 "Prepaid card": "prepaid or gift card",
 "Payday loan, title loan, personal loan, or advance loan": "payday/title/personal/advance loan",
 "Student loan": "student loan",
 "Debt or credit management": "debt settlement or credit-repair services",
}


def load():
    return json.load(open("data/cfpb_company.json", encoding="utf-8"))


def labels(rows):
    # keep products with enough overall support to be a real class
    cnt = Counter(r["p"] for r in rows)
    return [p for p in GLOSS if cnt.get(p, 0) >= 8]


def clsprompt(text, LBL, rules=None):
    cats = "\n".join(f"- {k}: {GLOSS[k]}" for k in LBL)
    p = f"Product categories:\n{cats}\n\n"
    if rules:
        p += ("Filing rules (how this company actually files complaints; apply when they match):\n"
              f"{rules}\n\n")
    p += f"Complaint:\n\"\"\"{text[:1000]}\"\"\"\n\nProduct category:"
    return p


def parse_label(o, LBL):
    o = (o or "").strip()
    for l in LBL:
        if o == l:
            return l
    low = o.lower()
    for l in LBL:
        if l.lower()[:20] in low:
            return l
    return "UNPARSED"


SYS = ("You categorize consumer-finance complaints by product from the narrative text. Choose the "
       "SINGLE best product. Reply with ONLY the exact product category, nothing else.")


def submit():
    os.makedirs("results", exist_ok=True)
    rows = load(); LBL = labels(rows)
    rng = np.random.RandomState(0)
    src = [r for r in rows if r["c"] != HELD]
    held = [r for r in rows if r["c"] == HELD]
    idx = rng.permutation(len(held))
    test = [held[i] for i in idx[:N_TEST]]
    update = [held[i] for i in idx[N_TEST:N_TEST+N_UPDATE]]
    pool = {"src": src, "test": test, "update": update, "labels": LBL}
    json.dump(pool, open(POOL_F, "w"), ensure_ascii=False)
    items = []
    for grp in ("src", "test", "update"):
        for i, r in enumerate(pool[grp]):
            items.append((f"{grp}:{i}", {"messages": [{"role": "system", "content": SYS},
                          {"role": "user", "content": clsprompt(r["t"], LBL)}],
                          "temperature": 0, "max_tokens": 20}))
    submit_chat_batch(MODEL, items, tag=BASE_TAG)
    print(f"held-out company={HELD}  labels={len(LBL)}  src={len(src)} test={len(test)} update={len(update)}")
    print(f"submitted {len(items)} base-classify requests")


def induce():
    res = collect_chat_batch(tag=BASE_TAG)
    if res is None:
        print("base batch not ready; rerun induce"); return
    pool = json.load(open(POOL_F, encoding="utf-8")); LBL = pool["labels"]
    base = {g: [parse_label(res.get(f"{g}:{i}", ""), LBL) for i in range(len(pool[g]))]
            for g in ("src", "test", "update")}
    acc = np.mean([base["src"][i] == pool["src"][i]["p"] for i in range(len(pool["src"]))])
    src_cases = [(pool["src"][i]["t"], base["src"][i], pool["src"][i]["p"])
                 for i in range(len(pool["src"]))
                 if base["src"][i] != "UNPARSED" and base["src"][i] != pool["src"][i]["p"]]
    print(f"source rows={len(pool['src'])} base-acc={acc:.3f} disagreements={len(src_cases)}")
    source_rules = _induce_rules([(t, g, y) for t, g, y in src_cases])
    # relabel 'file as <Category>' wording already matches _induce_rules format
    corrupt_rules = _corrupt(source_rules)
    upd_cases = [(pool["update"][i]["t"], base["update"][i], pool["update"][i]["p"])
                 for i in range(len(pool["update"]))
                 if base["update"][i] != "UNPARSED" and base["update"][i] != pool["update"][i]["p"]]
    update_extra = _induce_rules([(t, g, y) for t, g, y in upd_cases], k_max=8) if upd_cases else ""
    updated_rules = (source_rules + "\n" + update_extra).strip()
    json.dump({"source": source_rules, "corrupt": corrupt_rules, "updated": updated_rules,
               "base_test": base["test"], "n_src_disagree": len(src_cases),
               "n_upd_disagree": len(upd_cases)}, open(RULES_F, "w"), indent=2, ensure_ascii=False)
    print("\n=== SOURCE RULEBOOK (CFPB) ===\n" + source_rules)
    arms = {"source_rules": source_rules, "source_corrupt": corrupt_rules, "updated_rules": updated_rules}
    items = []
    for arm, rules in arms.items():
        for i, r in enumerate(pool["test"]):
            items.append((f"{arm}:{i}", {"messages": [{"role": "system", "content": SYS},
                          {"role": "user", "content": clsprompt(r["t"], LBL, rules)}],
                          "temperature": 0, "max_tokens": 20}))
    submit_chat_batch(MODEL, items, tag=RULES_TAG)
    print(f"\nsubmitted {len(items)} rule-classify requests (3 arms x {len(pool['test'])} test)")


def report():
    res = collect_chat_batch(tag=RULES_TAG)
    if res is None:
        print("rules batch not ready; rerun report"); return
    pool = json.load(open(POOL_F, encoding="utf-8")); LBL = pool["labels"]
    rules = json.load(open(RULES_F, encoding="utf-8"))
    gold = [r["p"] for r in pool["test"]]; base = rules["base_test"]; n = len(gold)
    arms = {"no_rules": base}
    for arm in ("source_rules", "source_corrupt", "updated_rules"):
        arms[arm] = [parse_label(res.get(f"{arm}:{i}", ""), LBL) for i in range(n)]
    hard = [i for i in range(n) if base[i] != gold[i]]
    easy = [i for i in range(n) if base[i] == gold[i]]
    print(f"held-out company={HELD}  test={n}  baseline hard(wrong)={len(hard)} easy={len(easy)}")
    print(f"{'arm':16s}{'acc_all':>9s}{'acc_hard':>9s}{'recovered':>11s}{'broke':>7s}")
    out = {}
    for arm, pred in arms.items():
        acc = np.mean([pred[i] == gold[i] for i in range(n)])
        ah = np.mean([pred[i] == gold[i] for i in hard]) if hard else float("nan")
        rec = sum(1 for i in hard if pred[i] == gold[i]); broke = sum(1 for i in easy if pred[i] != gold[i])
        print(f"{arm:16s}{acc:9.3f}{ah:9.3f}{rec:11d}{broke:7d}")
        out[arm] = {"acc_all": round(float(acc), 4), "acc_hard": round(float(ah), 4),
                    "recovered": int(rec), "broke": int(broke)}
    json.dump({"held_out": HELD, "n_test": n, "n_hard": len(hard), "arms": out,
               "n_src_disagree": rules["n_src_disagree"], "n_upd_disagree": rules["n_upd_disagree"]},
              open("results/cfpb_rulebook.json", "w"), indent=2)
    sr, nr, cc = out["source_rules"], out["no_rules"], out["source_corrupt"]
    print("\n[invariants]")
    print(f"  source_rules recovers hard? recovered={sr['recovered']} (>0), net {sr['acc_all']} vs {nr['acc_all']} "
          f"{'PASS' if sr['acc_all']>=nr['acc_all'] else 'CHECK'}")
    print(f"  placebo weaker? corrupt {cc['acc_all']} < source {sr['acc_all']} "
          f"{'PASS' if cc['acc_all']<sr['acc_all'] else 'FAIL(just extra text)'}")
    print("wrote results/cfpb_rulebook.json")


if __name__ == "__main__":
    {"submit": submit, "induce": induce, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()
