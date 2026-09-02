"""MIMIC-III within-organization convention mining: does the coding-convention gap (principal-diagnosis
sequencing / billing rules not entailed by the note text) yield mineable, validated rules that improve
LLM coding? Single hospital -> one organization, same harness as per_city_rulebook.

  python mimic_rulebook.py submit    # base-classify discharge text -> ICD-9 chapter (batch)
  python mimic_rulebook.py report     # iterative LLM mining -> validate -> apply; CI + placebo
"""
import sys, os, json
import numpy as np
from collections import Counter
from openai_batch import submit_chat_batch, collect_chat_batch
from rulebook_v3 import iterative_mine
from rulebook_v2 import apply_rules, score

N_MINE = 2500
N_TEST = 1000
TAG = "mimic_base"
POOL_F = "results/mimic_pool.json"
MODEL = "gpt-4o-mini"

GLOSS = {
 "Circulatory": "heart failure, myocardial infarction, arrhythmia, vascular, stroke, hypertension",
 "Respiratory": "pneumonia, COPD, respiratory failure, asthma, pulmonary",
 "Infectious_Parasitic": "sepsis, bacteremia, systemic infection",
 "Digestive": "GI bleed, liver disease, pancreatitis, bowel, biliary",
 "Injury_Poisoning": "trauma, fracture, overdose, poisoning, complication of care",
 "Neoplasms": "cancer, malignancy, tumor",
 "Endocrine_Metabolic": "diabetes, electrolyte or metabolic disorder, thyroid",
 "Genitourinary": "kidney/renal failure, urinary tract, genitourinary",
 "Musculoskeletal": "joint, bone, spine, connective tissue",
 "Nervous_Sense": "seizure, neurologic, brain (non-vascular), eye/ear",
 "Symptoms_Illdefined": "ill-defined symptoms without a specific diagnosis",
 "Mental": "psychiatric disorder, substance use",
 "Skin": "cellulitis, skin/subcutaneous",
 "Blood": "anemia, coagulation or blood disorder",
 "Congenital": "congenital anomaly",
 "Perinatal": "perinatal condition", "Pregnancy_Childbirth": "pregnancy, childbirth",
 "Supplementary_V": "aftercare, status, screening, follow-up",
 "External_Cause_E": "external cause of injury",
}
SYS = ("You assign the PRINCIPAL diagnosis category (ICD-9 chapter) for a hospital admission from the "
       "discharge note. Choose the SINGLE chapter for the condition chiefly responsible for admission. "
       "Reply with ONLY the exact category name.")


def load():
    return json.load(open("data/mimic_dx.json", encoding="utf-8"))


def present_labels(rows):
    cnt = Counter(r["chapter"] for r in rows)
    return [c for c in GLOSS if cnt.get(c, 0) >= 8]


def clsprompt(text, LBL):
    cats = "\n".join(f"- {k}: {GLOSS[k]}" for k in LBL)
    return f"Categories:\n{cats}\n\nDischarge note:\n\"\"\"{text[:1600]}\"\"\"\n\nPrincipal diagnosis category:"


def parse_label(o, LBL):
    o = (o or "").strip()
    for l in LBL:
        if o == l:
            return l
    low = o.lower().replace(" ", "_")
    for l in LBL:
        if l.lower() in low:
            return l
    return "UNPARSED"


def submit():
    os.makedirs("results", exist_ok=True)
    rows = load(); LBL = present_labels(rows)
    rng = np.random.RandomState(0); idx = rng.permutation(len(rows))
    mine = [rows[i] for i in idx[:N_MINE]]; test = [rows[i] for i in idx[N_MINE:N_MINE+N_TEST]]
    json.dump({"mine": mine, "test": test, "labels": LBL}, open(POOL_F, "w"), ensure_ascii=False)
    items = []
    for grp, data in (("mine", mine), ("test", test)):
        for i, r in enumerate(data):
            items.append((f"{grp}:{i}", {"messages": [{"role": "system", "content": SYS},
                          {"role": "user", "content": clsprompt(r["text"], LBL)}],
                          "temperature": 0, "max_tokens": 18}))
    submit_chat_batch(MODEL, items, tag=TAG)
    print(f"labels={len(LBL)} mine={len(mine)} test={len(test)}; submitted {len(items)} base-classify")


def report():
    res = collect_chat_batch(tag=TAG)
    if res is None:
        print("base batch not ready; rerun report"); return
    pool = json.load(open(POOL_F, encoding="utf-8")); LBL = pool["labels"]
    mine = [(pool["mine"][i]["text"], pool["mine"][i]["chapter"]) for i in range(len(pool["mine"]))]
    mb = [parse_label(res.get(f"mine:{i}", ""), LBL) for i in range(len(mine))]
    test = [(pool["test"][i]["text"], pool["test"][i]["chapter"]) for i in range(len(pool["test"]))]
    tb = [parse_label(res.get(f"test:{i}", ""), LBL) for i in range(len(test))]
    rng = np.random.RandomState(0); pm = rng.permutation(len(mine)); cut = int(0.7 * len(mine))
    m_rows = [mine[i] for i in pm[:cut]]; m_base = [mb[i] for i in pm[:cut]]
    v_rows = [mine[i] for i in pm[cut:]]; v_base = [mb[i] for i in pm[cut:]]
    kept = iterative_mine(m_rows, m_base, v_rows, v_base, min_pair=6, max_rules=40)
    s_no = score(tb, test, tb); s_ru = score(apply_rules(kept, test, tb), test, tb)
    pl_accs = []
    for _ in range(20):
        pl = []
        for r in kept:
            alt = r["to"]
            while alt == r["to"] or alt == r["from"]:
                alt = LBL[rng.randint(len(LBL))]
            pl.append(dict(r, to=alt))
        pl_accs.append(score(apply_rules(pl, test, tb), test, tb)["acc_all"])
    plac = float(np.mean(pl_accs)) if kept else s_no["acc_all"]
    gold = [y for t, y in test]; rp = apply_rules(kept, test, tb); n = len(gold)
    bi = np.random.RandomState(1).randint(0, n, (1000, n))
    d_bo = [np.mean([rp[j] == gold[j] for j in b]) - np.mean([tb[j] == gold[j] for j in b]) for b in bi]
    ci = (round(float(np.percentile(d_bo, 2.5)), 4), round(float(np.percentile(d_bo, 97.5)), 4))
    print(f"\nMIMIC-III principal-dx coding (labels={len(LBL)}, test={n})")
    print(f"  base disagreements mined: {len(kept)} validated rules")
    for r in kept[:16]:
        print(f"   {r['from']:>20s} -> {r['to']:<20s} kw={r['kw'][:5]}")
    print(f"\n  no_rules = {s_no['acc_all']:.4f}")
    print(f"  rules    = {s_ru['acc_all']:.4f}   delta={s_ru['acc_all']-s_no['acc_all']:+.4f}  CI95={ci}")
    print(f"  placebo  = {plac:.4f}")
    print(f"  recovered(hard) = {s_ru['recovered']}  broke(easy) = {s_ru['broke']}")
    json.dump({"labels": len(LBL), "n_test": n, "n_rules": len(kept),
               "no_rules": s_no["acc_all"], "rules": s_ru["acc_all"], "placebo": plac,
               "delta_ci95": ci, "recovered": s_ru["recovered"], "broke": s_ru["broke"],
               "kept_rules": kept}, open("results/mimic_rulebook.json", "w"), indent=2)
    print("\n[invariant] rules beat placebo AND no_rules?",
          "PASS" if s_ru["acc_all"] > plac and s_ru["acc_all"] > s_no["acc_all"] else "CHECK")
    print("wrote results/mimic_rulebook.json")


if __name__ == "__main__":
    {"submit": submit, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()
