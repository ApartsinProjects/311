"""Per-city (within-organization) convention mining -- the version that should actually work.

For EACH city independently: mine that city's own filing conventions from a small labeled sample
(iterative LLM rule mining + per-rule held-out validation), then apply to that city's frozen test set.
Compares no_rules vs mined rules vs placebo (shuffled rule targets). This is the 'adapt to an
organization from a few of its own labels, no retraining' value proposition.

  python per_city_rulebook.py submit    # base-classify each city's mine+test sample (one batch)
  python per_city_rulebook.py report     # per-city mine -> validate -> apply; table + averages
"""
import sys, os, json
import numpy as np
from collections import Counter
from eval_common import load_split, LABELS
from pilot_rulebook import parse_label
from openai_batch import submit_chat_batch, collect_chat_batch
from rulebook_v3 import iterative_mine
from rulebook_v2 import apply_rules, score

N_MINE = 1200
N_TEST = 500
TAG = "pc_base"
POOL_F = "results/pc_pool.json"
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


def clsprompt(t):
    return f"Categories:\n{CATLIST}\n\nRequest:\n\"\"\"{t[:1000]}\"\"\"\n\nCategory:"


def submit():
    os.makedirs("results", exist_ok=True)
    sp = load_split(); rng = np.random.RandomState(0)
    pool = {}
    for c in sp["train"]:
        tr = sp["train"][c]; idx = rng.permutation(len(tr))[:N_MINE]
        mine = [(tr[i][0], tr[i][1]) for i in idx]
        test = [(t, y) for t, y in sp["test"][c]][:N_TEST]
        pool[c] = {"mine": mine, "test": test}
    json.dump(pool, open(POOL_F, "w"), ensure_ascii=False)
    items = []
    for c in pool:
        for grp in ("mine", "test"):
            for i, (t, y) in enumerate(pool[c][grp]):
                items.append((f"{c}|{grp}|{i}", {"messages": [{"role": "system", "content": SYS},
                              {"role": "user", "content": clsprompt(t)}], "temperature": 0, "max_tokens": 15}))
    submit_chat_batch(MODEL, items, tag=TAG)
    print(f"cities={list(pool)}  submitted {len(items)} base-classify requests")


def report():
    res = collect_chat_batch(tag=TAG)
    if res is None:
        print("base batch not ready; rerun report"); return
    pool = json.load(open(POOL_F, encoding="utf-8"))
    rng = np.random.RandomState(0)
    rows_out = []
    print(f"{'city':14s}{'n_rules':>8s}{'no_rules':>10s}{'rules':>8s}{'placebo':>9s}{'d_rules':>9s}{'rec':>5s}{'brk':>5s}")
    agg = {"no_rules": [], "rules": [], "placebo": []}
    for c in pool:
        mine = [(pool[c]["mine"][i][0], pool[c]["mine"][i][1]) for i in range(len(pool[c]["mine"]))]
        mb = [parse_label(res.get(f"{c}|mine|{i}", "")) for i in range(len(mine))]
        test = [(pool[c]["test"][i][0], pool[c]["test"][i][1]) for i in range(len(pool[c]["test"]))]
        tb = [parse_label(res.get(f"{c}|test|{i}", "")) for i in range(len(test))]
        # split the city's mine sample into mine/val for rule validation
        pm = rng.permutation(len(mine)); cut = int(0.7 * len(mine))
        mi, vi = pm[:cut], pm[cut:]
        m_rows = [mine[i] for i in mi]; m_base = [mb[i] for i in mi]
        v_rows = [mine[i] for i in vi]; v_base = [mb[i] for i in vi]
        kept = iterative_mine(m_rows, m_base, v_rows, v_base, min_pair=4, max_rules=30)
        s_no = score(tb, test, tb)
        s_ru = score(apply_rules(kept, test, tb), test, tb)
        # proper placebo: each rule fires but points to a RANDOM WRONG label (avg over 20 draws)
        pl_accs = []
        for _ in range(20):
            pl = []
            for r in kept:
                alt = r["to"]
                while alt == r["to"] or alt == r["from"]:
                    alt = LABELS[rng.randint(len(LABELS))]
                pl.append(dict(r, to=alt))
            pl_accs.append(score(apply_rules(pl, test, tb), test, tb)["acc_all"])
        s_pl = {"acc_all": float(np.mean(pl_accs)) if kept else s_no["acc_all"]}
        # bootstrap 95% CI on the rules-minus-norules test delta
        gold = [y for t, y in test]; rp = apply_rules(kept, test, tb)
        n = len(gold); bi = np.random.RandomState(1).randint(0, n, (1000, n))
        d_bo = [np.mean([rp[j] == gold[j] for j in b]) - np.mean([tb[j] == gold[j] for j in b]) for b in bi]
        ci = (round(float(np.percentile(d_bo, 2.5)), 4), round(float(np.percentile(d_bo, 97.5)), 4))
        s_ru["delta_ci95"] = ci
        agg["no_rules"].append(s_no["acc_all"]); agg["rules"].append(s_ru["acc_all"]); agg["placebo"].append(s_pl["acc_all"])
        print(f"{c:14s}{len(kept):>8d}{s_no['acc_all']:>10.4f}{s_ru['acc_all']:>8.4f}{s_pl['acc_all']:>9.4f}"
              f"{s_ru['acc_all']-s_no['acc_all']:>+9.4f}  CI{ci}  rec={s_ru['recovered']} brk={s_ru['broke']}")
        rows_out.append({"city": c, "n_rules": len(kept), "no_rules": s_no, "rules": s_ru, "placebo": s_pl,
                         "kept_rules": kept})
    dr = np.mean(agg["rules"]) - np.mean(agg["no_rules"])
    dp = np.mean(agg["rules"]) - np.mean(agg["placebo"])
    print(f"\nMEAN no_rules={np.mean(agg['no_rules']):.4f}  rules={np.mean(agg['rules']):.4f}  "
          f"placebo={np.mean(agg['placebo']):.4f}   d(rules-no)={dr:+.4f}  d(rules-placebo)={dp:+.4f}")
    json.dump({"per_city": rows_out, "mean": {k: float(np.mean(v)) for k, v in agg.items()},
               "delta_rules_vs_norules": float(dr), "delta_rules_vs_placebo": float(dp)},
              open("results/per_city_rulebook.json", "w"), indent=2)
    print("wrote results/per_city_rulebook.json")


if __name__ == "__main__":
    {"submit": submit, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()
