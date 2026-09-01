"""Full E1: judge every city's capped training pool once (reused across folds), then for each
leave-one-city-out fold export raw / judge-relabel / random-relabel training sets for a DistilBERT
retrain (RunPod). Judging is batch (50% off, reap-safe).
  python e1_full.py submit         # judge the full training pool (batch)
  python e1_full.py export         # collect judge, write per-fold training CSVs for RunPod
"""
import sys, os, json, csv, numpy as np
from eval_common import load_split, LABELS
from eval_tfidf import cap_rows
from defensibility_judge import SYS, prompt, parse_set

CAP = 8000
JUDGE = "gpt-4o-mini"
TAG = "e1_train"
POOL_F = "results/e1_pool.json"
OUT_DIR = "results/e1_folds"


def build_pool():
    sp = load_split()
    pool = []  # (city, text, raw_label)
    for c in sp["train"]:
        for t, y in cap_rows(sp["train"][c], CAP):
            pool.append((c, t, y))
    return pool, sp


def submit():
    pool, _ = build_pool()
    json.dump(pool, open(POOL_F, "w"), ensure_ascii=False)
    import openai_batch as B
    items = [(f"r{i}", {"messages": [{"role": "system", "content": SYS},
                                     {"role": "user", "content": prompt(t)}],
                        "temperature": 0, "max_tokens": 30}) for i, (c, t, y) in enumerate(pool)]
    B.submit_chat_batch(JUDGE, items, tag=TAG)
    print(f"submitted {len(items)} judge requests ({len(set(p[0] for p in pool))} cities); "
          f"collect+export with: python e1_full.py export")


def _download_any(bid):
    """Download whatever output exists, even for a cancelled/expired batch (partial results)."""
    import openai_batch as B
    b = B.client.batches.retrieve(bid)
    if not getattr(b, "output_file_id", None):
        return None, b.status
    text = B.client.files.content(b.output_file_id).text
    res = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line); cid = rec.get("custom_id")
        try:
            res[cid] = rec["response"]["body"]["choices"][0]["message"]["content"]
        except Exception:
            res[cid] = ""
    return res, b.status


def export(res=None):
    if res is None:
        import openai_batch as B
        res = B.collect_chat_batch(tag=TAG)
        if res is None:
            print("judge batch not ready yet; rerun: python e1_full.py export"); return
    pool = [tuple(x) for x in json.load(open(POOL_F, encoding="utf-8"))]
    jsets = [set(parse_set(res.get(f"r{i}", ""))) - {"UNPARSED"} for i in range(len(pool))]
    rng = np.random.RandomState(0)
    os.makedirs(OUT_DIR, exist_ok=True)
    cities = sorted(set(p[0] for p in pool))
    data = {}; summary = {}
    for held in cities:
        idxs = [i for i in range(len(pool)) if pool[i][0] != held]  # train on the other cities
        rej = [i for i in idxs if jsets[i] and pool[i][2] not in jsets[i]]
        rejset = set(rej)
        texts = [pool[i][1] for i in idxs]
        raw = [pool[i][2] for i in idxs]
        judge = [sorted(jsets[i])[0] if i in rejset else pool[i][2] for i in idxs]
        rand = [LABELS[rng.randint(len(LABELS))] if i in rejset else pool[i][2] for i in idxs]
        data[held] = {"texts": texts, "labels": {"raw": raw, "judge_relabel": judge, "random_relabel": rand}}
        summary[held] = {"train_rows": len(idxs), "judge_rejected": len(rej),
                         "reject_frac": round(len(rej) / len(idxs), 4)}
        print(f"{held:12s} train={len(idxs)} rejected={len(rej)} ({len(rej)/len(idxs):.1%})")
    json.dump(data, open(os.path.join(OUT_DIR, "e1_train_data.json"), "w"), ensure_ascii=False)
    json.dump(summary, open(os.path.join(OUT_DIR, "_summary.json"), "w"), indent=2)
    print(f"wrote {OUT_DIR}/e1_train_data.json (compact: shared texts + per-condition labels)")


if __name__ == "__main__":
    {"submit": submit, "export": export}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()
