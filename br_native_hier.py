"""Rebuilt instruction method with the review fixes:
 (1) represent EVERY category (cues + canonical example snippets; never empty; long-tail fallback);
 (2) HIERARCHICAL narrowing: stage A classifies into the coarse harmonized group (14-way), stage B
     picks the native category within that group from a focused NUMBERED candidate list (fix parse);
 both stages are FIXED, cacheable prompts (no per-query retrieval).
Reports predicted-group pipeline AND oracle-group upper bound, vs RAG (0.756). Same frozen split.

  python br_native_hier.py
"""
import csv, json, re
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client
from br_native_compare import labelset

csv.field_size_limit(10**7)
MODEL = "gpt-4o-mini"
GROUP_GLOSS = {
 "Waste_Sanitation": "garbage/recycling collection, missed pickup, carts, woody waste, debris, dumping",
 "Streets_Sidewalks": "potholes, road/sidewalk/curb/driveway repair, road striping, barricades",
 "Street_Lighting": "street light out or damaged", "Traffic_Signals_Signs": "traffic signals, signs, markings",
 "Trees_Vegetation": "trees, limbs, mowing, tall grass, overgrowth", "Graffiti_Postings": "graffiti, illegal postings",
 "Parking_Vehicles": "abandoned/junk vehicles, illegal parking", "Property_Housing_Code": "building/zoning code, unsafe/vacant buildings",
 "Water_Sewer_Drainage": "sewer, drainage, catch basins, flooding, pipes, sinkholes", "Homelessness": "homeless encampments",
 "Animals_Pests": "animal control, strays, wildlife, pests", "Noise": "noise complaints",
 "Transit": "transit, bus", "Parks_Recreation": "parks, playgrounds, trails",
}


def native_to_group():
    m = defaultdict(Counter)
    for r in csv.DictReader(open("data/mc311_harmonized.csv", encoding="utf-8")):
        if r["city"] == "BatonRouge" and r["native_category"].strip():
            m[r["native_category"].strip()][r["harmonized_label"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in m.items()}


def grams(t):
    toks = re.findall(r"[a-z]{2,}", t.lower()); g = set(toks)
    for i in range(len(toks)-1): g.add(toks[i]+" "+toks[i+1])
    return g


def build_entries(budget, LBL):
    """Per native category: distinctive cues + up to 3 example snippets. NEVER empty."""
    docf = Counter(); catf = defaultdict(Counter); catN = Counter(); exs = defaultdict(list)
    for r in budget:
        gs = grams(r["text"]); catN[r["label"]] += 1; exs[r["label"]].append(r["text"])
        for g in gs: docf[g] += 1; catf[r["label"]][g] += 1
    N = len(budget); ent = {}
    rng = np.random.RandomState(0)
    for l in LBL:
        n = catN[l] or 1; scored = []
        for g, c in catf[l].items():
            if c < 2: continue
            insh = c/n; ov = docf[g]/N
            if insh >= 0.10 and insh/(ov+1e-9) >= 1.5: scored.append((insh/(ov+1e-9)*insh, g))
        scored.sort(reverse=True); cues = [g for _, g in scored[:8]]
        e = exs.get(l, [])
        idx = rng.permutation(len(e))[:3] if e else []
        snip = "; ".join('"'+e[i][:80].replace("\n", " ")+'"' for i in idx)
        parts = []
        if cues: parts.append("cues: " + ", ".join(cues))
        if snip: parts.append("e.g.: " + snip)
        ent[l] = " | ".join(parts) if parts else l.lower()   # never empty
    return ent


def classify(prompts, LBL_or_none):
    """prompts: list of (system, user). Returns raw outputs."""
    out = [None]*len(prompts)
    def one(i):
        s, u = prompts[i]
        for a in range(3):
            try:
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=8,
                    messages=[{"role": "system", "content": s}, {"role": "user", "content": u}])
                return r.choices[0].message.content.strip()
            except Exception:
                import time; time.sleep(1.5*(a+1))
        return ""
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(one, i): i for i in range(len(prompts))}
        for f in as_completed(futs): out[futs[f]] = f.result()
    return out


def pick_num(o, n):
    m = re.search(r"\d+", o or "")
    if not m: return -1
    v = int(m.group(0))
    return v-1 if 1 <= v <= n else -1


def main():
    d = json.load(open("results/br_split.json", encoding="utf-8")); pool = d["pool"]; test = d["test"]
    LBL = labelset(pool, test); budget = pool[:2000]
    n2g = native_to_group()
    groups = sorted(set(n2g.get(l, "Waste_Sanitation") for l in LBL))
    g_cats = defaultdict(list)
    for l in LBL: g_cats[n2g.get(l, "Waste_Sanitation")].append(l)
    ent = build_entries(budget, LBL)
    gold = [r["label"] for r in test]; gold_g = [n2g.get(g, "Waste_Sanitation") for g in gold]
    # ---- Stage A: text -> group (numbered) ----
    glist = "\n".join(f"{i+1}. {g}: {GROUP_GLOSS.get(g,'')}" for i, g in enumerate(groups))
    sysA = ("List the TWO most likely coarse service groups for the 311 request, best first. "
            "Reply with ONLY two group numbers separated by a comma.")
    pA = [(sysA, f"Groups:\n{glist}\n\nRequest: {test[i]['text'][:500]}\nTwo group numbers:") for i in range(len(test))]
    outA = classify(pA, None)
    def top2(o):
        nums = [int(x)-1 for x in re.findall(r"\d+", o or "")][:2]
        return [groups[k] for k in nums if 0 <= k < len(groups)] or ["Waste_Sanitation"]
    predg2 = [top2(outA[i]) for i in range(len(test))]
    predg = [g[0] for g in predg2]
    print(f"Stage A top-1 group acc = {np.mean([predg[i]==gold_g[i] for i in range(len(test))]):.4f}")
    print(f"Stage A top-2 group recall = {np.mean([gold_g[i] in predg2[i] for i in range(len(test))]):.4f}")

    def stageB_cats(cand_of):
        prompts = []
        for i in range(len(test)):
            cats = cand_of[i] or LBL
            body = "\n".join(f"{j+1}. {c} [{ent[c][:160]}]" for j, c in enumerate(cats))
            s = "Pick the EXACT city service category for the request from the numbered candidates. Reply with ONLY the number."
            prompts.append((s, f"Candidates:\n{body}\n\nRequest: {test[i]['text'][:500]}\nNumber:"))
        outB = classify(prompts, None); preds = []
        for i in range(len(test)):
            cats = cand_of[i] or LBL; k = pick_num(outB[i], len(cats))
            preds.append(cats[k] if k >= 0 else "UNPARSED")
        return preds

    # real pipeline: candidates = union of the top-2 predicted groups
    real_cands = []
    for i in range(len(test)):
        cs = []
        for g in predg2[i]:
            cs += g_cats[g]
        real_cands.append(sorted(set(cs)))
    print(f"avg Stage-B candidates (top-2 union) = {np.mean([len(c) for c in real_cands]):.1f}")
    pred_real = stageB_cats(real_cands)
    pred_oracle = stageB_cats([g_cats[gold_g[i]] for i in range(len(test))])
    E = np.array(json.load(open("results/br_emb.json", encoding="utf-8"))["emb"], dtype=np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True)+1e-9); nnsim = (E[2000:]@E[:2000].T).max(1)
    def strat(p):
        r = {}
        for nm, f in [("nov", lambda s: s < 0.7), ("mid", lambda s: 0.7 <= s < 0.9), ("dup", lambda s: s >= 0.9)]:
            idx = [i for i in range(len(test)) if f(nnsim[i])]
            r[nm] = round(float(np.mean([p[i] == gold[i] for i in idx])), 3)
        return r
    ar = np.mean([pred_real[i] == gold[i] for i in range(len(test))])
    ao = np.mean([pred_oracle[i] == gold[i] for i in range(len(test))])
    print(f"\nHIERARCHICAL (predicted group): acc={ar:.4f}  strat={strat(pred_real)}")
    print(f"HIERARCHICAL (ORACLE group)   : acc={ao:.4f}  strat={strat(pred_oracle)}")
    print(f"  reference: RAG=0.756, flat cue-codebook=0.657, fine-tuned=0.684, 1-NN=0.680")
    json.dump({"stageA_group_acc": float(np.mean([predg[i]==gold_g[i] for i in range(len(test))])),
               "hier_pred": float(ar), "hier_oracle": float(ao), "strat_pred": strat(pred_real)},
              open("results/br_native_hier.json", "w"), indent=2)
    print("wrote results/br_native_hier.json")


if __name__ == "__main__":
    main()
