"""Probe zero-shot on a candidate dataset (small sample) BEFORE committing to a full sweep.
Only pursue domains where zero-shot is low (the sweet spot). Cheap: ~80 classify calls."""
import sys, numpy as np
from collections import Counter
import semclf
from semclf import CLF_MODEL, TRUNC, chat_many

def probe(name, texts, labels, LBL, n=80):
    idx = np.random.RandomState(0).permutation(len(texts))[:n]
    tx = [texts[i][:TRUNC] for i in idx]; gd = [labels[i] for i in idx]
    catlist = "\n".join(f"- {l}" for l in LBL)
    sysm = f"Classify the item into its single best category. Reply with ONLY one category name copied verbatim."
    msgs = [[{"role":"system","content":sysm},
             {"role":"user","content":f"Category options:\n{catlist}\n\nItem: {t}\nCategory:"}] for t in tx]
    def parse(o):
        o=(o or "").strip().strip('".').lower()
        for l in LBL:
            if o==l.lower(): return l
        c=[l for l in LBL if l.lower() in o]
        return max(c,key=len) if c else "UNPARSED"
    preds=[parse(o) for o in chat_many(msgs, model=CLF_MODEL, max_tokens=24)]
    acc=np.mean([preds[i]==gd[i] for i in range(len(gd))])
    maj=Counter(labels).most_common(1)[0][1]/len(labels)
    print(f"  {name}: classes={len(LBL)} zero-shot(n={n})={acc:.3f} majority={maj:.3f}  {'<-- SWEET (low)' if acc<0.70 else 'too good' if acc>0.80 else 'borderline'}")
    return acc
