"""Build Banking77 split (fine-grained banking-intent classification, 77 classes) with the standard
recipe: single source, proportional fixed test (1500), stratified budget (2000), novel/dup split.
  python build_banking.py"""
import json, numpy as np, semclf
from collections import Counter
from scipy import stats
from datasets import load_dataset
ds=load_dataset('mteb/banking77')
R=[{'text':ds[s]['text'][i],'label':ds[s]['label_text'][i]} for s in ds for i in range(len(ds[s]))]
rng=np.random.RandomState(0); idx=rng.permutation(len(R))
test=[R[i] for i in idx[:1500]]; pool=[R[i] for i in idx[1500:]]
ref=set(semclf.norm(r['text']) for r in semclf.stratified_budget(pool,2000))
novel=[r for r in test if semclf.norm(r['text']) not in ref]; dup=[r for r in test if semclf.norm(r['text']) in ref]
json.dump({'pool':pool,'test':novel,'test_dup':dup}, open('results/banking_split.json','w'), ensure_ascii=False)
cp=Counter(r['label'] for r in pool); ct=Counter(r['label'] for r in test)
labs=sorted(cp,key=lambda x:-cp[x]); n=len(test); N=len(pool)
obs=np.array([ct[l] for l in labs],float); exp=np.array([cp[l]/N*n for l in labs],float)
p=1-stats.chi2.cdf(((obs-exp)**2/np.maximum(exp,1e-9)).sum(),len(labs)-1)
print(f'Banking77: pool={len(pool)} test={n} classes={len(cp)} maj={max(ct.values())/n:.3f} prop_p={p:.3f}')
