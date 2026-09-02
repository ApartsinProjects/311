"""Rulebook-transfer pilot: can an LLM-induced 'filing convention' rulebook, learned from where a
semantic classifier disagrees with recorded labels on SOURCE cities, transfer to a held-out city
and fix the labels the text alone gets wrong?

Arms on the held-out city's frozen test set:
  no_rules        : LLM classifies from text semantics only (baseline)
  source_rules    : + rulebook induced from SOURCE-city disagreements (ZERO new-city data)
  source_corrupt  : + the same rulebook with THEN-labels permuted (placebo control)
  updated_rules   : source rulebook + rules induced from a small held-out sample (few-shot update)

Decisive metric = recovery on HARD cases (rows the no_rules arm gets wrong). Rules must fix those
without breaking easy cases. Control arm must NOT help (else lift is just 'more text in prompt').

  python pilot_rulebook.py submit     # round-1 base classification (batch)
  python pilot_rulebook.py induce      # collect r1, mine disagreements, build rulebooks, submit round-2
  python pilot_rulebook.py report      # collect r2, compute hard-case recovery per arm
"""
import sys, os, json, re
import numpy as np
from collections import defaultdict, Counter
from eval_common import load_split, LABELS
from openai_batch import client, submit_chat_batch, collect_chat_batch

HELD = "Gainesville"
N_SRC = 500          # source rows/city for disagreement mining
N_UPDATE = 150       # held-out rows for the few-shot rulebook update (disjoint from test)
POOL_F = "results/rb_pool.json"
RULES_F = "results/rb_rules.json"
BASE_TAG = "rb_base"
RULES_TAG = "rb_rules"
MODEL = "gpt-4o-mini"

DESC = {
 "Waste_Sanitation": "garbage or recycling collection, missed pickup, litter, illegal dumping, debris",
 "Streets_Sidewalks": "pothole or road, sidewalk, curb or driveway repair, blocked street",
 "Street_Lighting": "street light or lamp out or damaged",
 "Traffic_Signals_Signs": "traffic signal, traffic or street sign, road markings, speeding",
 "Trees_Vegetation": "trees, tree limbs, overgrown grass or weeds, vegetation, mowing",
 "Graffiti_Postings": "graffiti or illegal postings and signs",
 "Parking_Vehicles": "abandoned or junk vehicle, illegal parking, parking meter",
 "Property_Housing_Code": "building or zoning code violation, vacant or unsafe building, rental housing",
 "Water_Sewer_Drainage": "sewer, drainage, flooding, catch basin, manhole, water quality, pipe",
 "Homelessness": "homeless encampment or person living in a vehicle or on the sidewalk",
 "Animals_Pests": "animal control, stray pet, wildlife, rodents or pests",
 "Noise": "noise complaint", "Transit": "public transit or bus, taxi, scooter",
 "Parks_Recreation": "park, playground, beach, trail or recreation facility",
}
CATLIST = "\n".join(f"- {k}: {DESC[k]}" for k in LABELS)
SYS = ("You categorize municipal 311 service requests. Choose the SINGLE best category for what the "
       "request is about. Reply with ONLY the exact category name, nothing else.")


def clsprompt(text, rules=None):
    p = f"Categories:\n{CATLIST}\n\n"
    if rules:
        p += ("Filing rules (how this jurisdiction actually files requests; apply when they match):\n"
              f"{rules}\n\n")
    p += f"Request:\n\"\"\"{text[:1000]}\"\"\"\n\nCategory:"
    return p


def parse_label(o):
    o = (o or "").strip()
    for l in LABELS:                       # exact, then substring
        if o == l:
            return l
    low = o.lower()
    for l in LABELS:
        if l.lower() in low or l.lower().replace("_", " ") in low:
            return l
    return "UNPARSED"


# ------------------------------ round 1: base classify ------------------------------
def submit():
    os.makedirs("results", exist_ok=True)
    sp = load_split()
    rng = np.random.RandomState(0)
    pool = {"src": [], "test": [], "update": []}
    for c in sp["train"]:
        if c == HELD:
            continue
        rows = sp["train"][c]; idx = rng.permutation(len(rows))[:N_SRC]
        pool["src"] += [(rows[i][0], rows[i][1], c) for i in idx]
    pool["test"] = [(t, y, HELD) for t, y in sp["test"][HELD]]
    hrows = sp["train"][HELD]; uidx = rng.permutation(len(hrows))[:N_UPDATE]
    pool["update"] = [(hrows[i][0], hrows[i][1], HELD) for i in uidx]
    json.dump(pool, open(POOL_F, "w"), ensure_ascii=False)
    items = []
    for grp in ("src", "test", "update"):
        for i, (t, y, c) in enumerate(pool[grp]):
            items.append((f"{grp}:{i}", {"messages": [{"role": "system", "content": SYS},
                          {"role": "user", "content": clsprompt(t)}], "temperature": 0, "max_tokens": 15}))
    submit_chat_batch(MODEL, items, tag=BASE_TAG)
    print(f"submitted {len(items)} base-classify requests "
          f"(src={len(pool['src'])} test={len(pool['test'])} update={len(pool['update'])})")


def _induce_rules(cases, k_max=18):
    """cases: list of (text, semantic_guess, actual_label). Ask the LLM for filing rules."""
    lines = []
    for t, g, y in cases[:220]:
        lines.append(f"- text: {t[:180]!r} | text-implies: {g} | actually-filed: {y}")
    body = ("Below are municipal 311 requests where a plain reading of the TEXT implies one category, "
            "but the city actually FILED it under a different category. These gaps reflect the city's "
            "filing conventions, not text meaning.\n\n" + "\n".join(lines) +
            f"\n\nWrite at most {k_max} concise filing rules that capture the RECURRING conventions "
            "(ignore one-off cases). Each rule format exactly:\n"
            "IF <textual condition> THEN file as <Category>\n"
            "Only output the rules, one per line.")
    r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=700,
        messages=[{"role": "system", "content": "You extract concise, general filing rules."},
                  {"role": "user", "content": body}])
    return r.choices[0].message.content.strip()


def _corrupt(rules):
    """Permute the THEN-categories among the rules -> placebo control rulebook."""
    lines = [l for l in rules.splitlines() if l.strip()]
    cats = []
    for l in lines:
        m = re.search(r"THEN file as\s+(.+)$", l.strip())
        cats.append(m.group(1).strip() if m else None)
    present = [c for c in cats if c]
    rng = np.random.RandomState(1); perm = rng.permutation(len(present)).tolist()
    shuffled = [present[i] for i in perm]; it = iter(shuffled); out = []
    for l, c in zip(lines, cats):
        out.append(re.sub(r"(THEN file as\s+).+$", lambda m: m.group(1) + next(it), l.strip()) if c else l)
    return "\n".join(out)


def induce():
    res = collect_chat_batch(tag=BASE_TAG)
    if res is None:
        print("base batch not ready; rerun induce"); return
    pool = json.load(open(POOL_F, encoding="utf-8"))
    base = {grp: [parse_label(res.get(f"{grp}:{i}", "")) for i in range(len(pool[grp]))]
            for grp in ("src", "test", "update")}
    # source disagreements -> rulebook (semantic guess != actual filed label)
    src_cases = [(pool["src"][i][0], base["src"][i], pool["src"][i][1])
                 for i in range(len(pool["src"]))
                 if base["src"][i] not in ("UNPARSED",) and base["src"][i] != pool["src"][i][1]]
    print(f"source rows={len(pool['src'])}  base-acc={np.mean([base['src'][i]==pool['src'][i][1] for i in range(len(pool['src']))]):.3f}"
          f"  disagreements={len(src_cases)}")
    source_rules = _induce_rules(src_cases)
    corrupt_rules = _corrupt(source_rules)
    # held-out update disagreements -> extra rules; combine
    upd_cases = [(pool["update"][i][0], base["update"][i], pool["update"][i][1])
                 for i in range(len(pool["update"]))
                 if base["update"][i] not in ("UNPARSED",) and base["update"][i] != pool["update"][i][1]]
    update_extra = _induce_rules(upd_cases, k_max=10) if upd_cases else ""
    updated_rules = (source_rules + "\n" + update_extra).strip()
    json.dump({"source": source_rules, "corrupt": corrupt_rules, "updated": updated_rules,
               "base_test": base["test"], "n_src_disagree": len(src_cases),
               "n_upd_disagree": len(upd_cases)}, open(RULES_F, "w"), indent=2, ensure_ascii=False)
    print("\n=== SOURCE RULEBOOK ===\n" + source_rules)
    # round-2: reclassify held-out TEST under each rulebook
    arms = {"source_rules": source_rules, "source_corrupt": corrupt_rules, "updated_rules": updated_rules}
    items = []
    for arm, rules in arms.items():
        for i, (t, y, c) in enumerate(pool["test"]):
            items.append((f"{arm}:{i}", {"messages": [{"role": "system", "content": SYS},
                          {"role": "user", "content": clsprompt(t, rules)}], "temperature": 0, "max_tokens": 15}))
    submit_chat_batch(MODEL, items, tag=RULES_TAG)
    print(f"\nsubmitted {len(items)} rule-classify requests (3 arms x {len(pool['test'])} test rows)")


def report():
    res = collect_chat_batch(tag=RULES_TAG)
    if res is None:
        print("rules batch not ready; rerun report"); return
    pool = json.load(open(POOL_F, encoding="utf-8"))
    rules = json.load(open(RULES_F, encoding="utf-8"))
    gold = [y for t, y, c in pool["test"]]
    base = rules["base_test"]
    arms = {"no_rules": base}
    for arm in ("source_rules", "source_corrupt", "updated_rules"):
        arms[arm] = [parse_label(res.get(f"{arm}:{i}", "")) for i in range(len(pool["test"]))]
    n = len(gold)
    hard = [i for i in range(n) if base[i] != gold[i]]          # rows the baseline gets wrong
    easy = [i for i in range(n) if base[i] == gold[i]]
    print(f"held-out={HELD}  test={n}  baseline hard(wrong)={len(hard)} easy(right)={len(easy)}")
    print(f"{'arm':16s}{'acc_all':>9s}{'acc_hard':>9s}{'acc_easy':>9s}{'recovered':>11s}{'broke':>7s}")
    out = {}
    for arm, pred in arms.items():
        acc = np.mean([pred[i] == gold[i] for i in range(n)])
        ah = np.mean([pred[i] == gold[i] for i in hard]) if hard else float("nan")
        ae = np.mean([pred[i] == gold[i] for i in easy]) if easy else float("nan")
        rec = sum(1 for i in hard if pred[i] == gold[i])         # wrong->right
        broke = sum(1 for i in easy if pred[i] != gold[i])       # right->wrong
        print(f"{arm:16s}{acc:9.3f}{ah:9.3f}{ae:9.3f}{rec:11d}{broke:7d}")
        out[arm] = {"acc_all": round(float(acc), 4), "acc_hard": round(float(ah), 4),
                    "acc_easy": round(float(ae), 4), "recovered": int(rec), "broke": int(broke)}
    json.dump({"held_out": HELD, "n_test": n, "n_hard": len(hard), "arms": out,
               "n_src_disagree": rules["n_src_disagree"], "n_upd_disagree": rules["n_upd_disagree"]},
              open("results/pilot_rulebook.json", "w"), indent=2)
    print("\n[invariants]")
    sr, nr, cc = out["source_rules"], out["no_rules"], out["source_corrupt"]
    print(f"  source_rules recovers hard cases? recovered={sr['recovered']} (need >0), "
          f"net acc {sr['acc_all']} vs {nr['acc_all']}  {'PASS' if sr['acc_all']>=nr['acc_all'] else 'CHECK'}")
    print(f"  placebo control weaker than real rules? corrupt acc_all={cc['acc_all']} < source {sr['acc_all']}  "
          f"{'PASS' if cc['acc_all'] < sr['acc_all'] else 'FAIL(lift is just extra text)'}")
    print("wrote results/pilot_rulebook.json")


if __name__ == "__main__":
    {"submit": submit, "induce": induce, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()
