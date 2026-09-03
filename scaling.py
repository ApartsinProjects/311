"""Is the plateau caused by DATA or by ARTIFACT CAPACITY?
  A) uncapped discovery at b=8000 (mine_cap=6000)  -> tests data scaling
  B) larger artifact at b=8000 (more rounds/edits/rules/triggers) -> tests capacity
Same fixed test throughout; everything cached.
"""
import json, os, sys
import numpy as np
from collections import defaultdict
import semclf, triggers
from semclf import TASKS, score, stratified_budget, paired_test

T = TASKS["bloom"]
test = T.test + T.test_dup
txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
semclf.set_trace("results/scaling_trace.jsonl")
zs = semclf.zero_shot(T, txt); azs,_,_ = score(T, zs, gold)
rag = semclf.lexical_rag(T, txt)  # uses whatever budget is set; set below per condition
print(f"SCALING  test={len(test)} zero-shot={azs:.4f}")

def run(tag, b, mine_cap, rounds, batches, maxpos, maxneg, n_trig):
    semclf.MAXPOS, semclf.MAXNEG = maxpos, maxneg
    bud = stratified_budget(T.pool, b, seed=0)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    nm = int(0.7*b)
    art = f"results/scal_{tag}.json"
    if os.path.exists(art):
        D = json.load(open(art, encoding="utf-8"))["D"]
    else:
        D,g = semclf.mine_rulebook(T, bud[:nm], bud[nm:], rounds=rounds, batches=batches,
                                   mine_cap=mine_cap)
        json.dump({"D": D, "gate": g}, open(art,"w"), indent=2, ensure_ascii=False)
    base = semclf._desc_classify(T, txt, D); a0,_,_ = score(T, base, gold)
    tf = f"results/scal_{tag}_trig.json"
    if os.path.exists(tf):
        rules = json.load(open(tf, encoding="utf-8"))["rules"]
    else:
        cap = min(mine_cap, nm)
        mtxt=[r["text"] for r in bud[:nm]][:cap]; mgold=[r["label"] for r in bud[:nm]][:cap]
        vtxt=[r["text"] for r in bud[nm:]][:600]; vgold=[r["label"] for r in bud[nm:]][:600]
        mb=semclf._desc_classify(T,mtxt,D); vb=semclf._desc_classify(T,vtxt,D)
        rules,_=triggers.mine_triggers(T,mtxt,mgold,mb,vtxt,vgold,vb,max_rules=n_trig)
        json.dump({"rules": rules}, open(tf,"w"), indent=2, ensure_ascii=False)
    fin = triggers.apply_triggers(T, txt, base, rules)
    a1,ci,_ = score(T, fin, gold)
    r = semclf.lexical_rag(T, txt); ar,_,_ = score(T, r, gold)
    ptz = paired_test(fin, zs, gold); ptr = paired_test(r, fin, gold)
    npos=sum(len(v['pos']) for v in D.values()); nneg=sum(len(v['neg']) for v in D.values())
    print(f"  [{tag}] b={b} cap={mine_cap} rounds={rounds} rules={npos}p/{nneg}n +{len(rules)}trig")
    print(f"      rulebook={a0:.4f}  final={a1:.4f} CI=({ci[0]:.3f},{ci[1]:.3f})  "
          f"vs_zs={ptz['delta']:+.4f} p={ptz['p_mcnemar']:.1e} | RAG={ar:.4f} gap={ar-a1:+.4f} p={ptr['p_mcnemar']:.3f}")
    sys.stdout.flush()
    return {"tag":tag,"b":b,"rulebook":a0,"final":a1,"rag":ar,"n_pos":npos,"n_neg":nneg,"n_trig":len(rules)}

out=[]
out.append(run("A_data8k",   8000, mine_cap=6000, rounds=5, batches=8,  maxpos=5,  maxneg=4, n_trig=12))
out.append(run("B_cap8k",    8000, mine_cap=6000, rounds=8, batches=12, maxpos=8,  maxneg=6, n_trig=24))
json.dump(out, open("results/scaling.json","w"), indent=2)
