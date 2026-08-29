"""
analyze_errors.py -- Root-cause error analysis for the cross-jurisdiction gap.
Re-runs the LLM zero-shot arm on a sample BUT logs per-example (city, native_category,
gold_super, pred_super, text). Then: confusion matrix, per-class recall + top confusion,
and concrete failing examples grouped by (gold -> pred). The native_category is kept so we
can tell MODEL errors from LABEL errors (city's own label disagreeing with the text).
"""
import csv, json, os, sys
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from llm_arm import make_client, classify_one
from bench_common import LABELS

csv.field_size_limit(10**7)
DATA = r"E:\Projects\Submitted\311\data"


def load_rows(cap_per_city, seed=0):
    mp = json.load(open(os.path.join(DATA, "harmonization_map.json"), encoding="utf-8"))
    by = defaultdict(list)
    with open(os.path.join(DATA, "raw", "all_cities.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            city, text, nat = r["city"], (r["text"] or "").strip(), r["native_category"]
            if len(text) < 3:
                continue
            sup = mp.get(city, {}).get(nat, "General_Admin_Other")
            if sup == "General_Admin_Other":
                continue
            by[city].append((text, nat, sup))
    rng = np.random.RandomState(seed)
    out = []
    for c, rows in by.items():
        idx = rng.permutation(len(rows))[:cap_per_city]
        out += [(c, *rows[i]) for i in idx]
    return out


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    model = sys.argv[2] if len(sys.argv) > 2 else "openai/gpt-4o-mini"
    client, prov = make_client(r"E:\Projects\.env.all")
    rows = load_rows(cap)
    print(f"provider={prov} model={model}  n={len(rows)}\n"); sys.stdout.flush()

    preds = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(classify_one, client, rows[i][1], model): i for i in range(len(rows))}
        for f in as_completed(futs):
            preds[futs[f]] = f.result()
    recs = []
    for (city, text, nat, gold), pred in zip(rows, preds):
        p = pred if pred in LABELS else "UNPARSED"
        recs.append(dict(city=city, native=nat, gold=gold, pred=p, correct=(p == gold), text=text))

    gold = [r["gold"] for r in recs]; pred = [r["pred"] for r in recs]
    print(f"overall acc={accuracy_score(gold,pred):.3f}  macroF1={f1_score(gold,pred,average='macro',zero_division=0):.3f}\n")

    # per-class recall + top wrong prediction
    print("=== per gold-class: recall and most-common wrong prediction ===")
    byg = defaultdict(list)
    for r in recs: byg[r["gold"]].append(r)
    for cls in sorted(byg, key=lambda c: -len(byg[c])):
        rs = byg[cls]; n = len(rs); ncorr = sum(r["correct"] for r in rs)
        wrong = Counter(r["pred"] for r in rs if not r["correct"]).most_common(1)
        wp = f"{wrong[0][0]}({wrong[0][1]})" if wrong else "-"
        print(f"  {cls:22s} n={n:3d} recall={ncorr/n:.2f}  top_wrong={wp}")

    # top confusion pairs with concrete examples
    print("\n=== top confusion pairs (gold -> pred) with examples ===")
    conf = Counter((r["gold"], r["pred"]) for r in recs if not r["correct"])
    for (g, p), cnt in conf.most_common(8):
        print(f"\n[{g}  ->  {p}]  x{cnt}")
        exs = [r for r in recs if r["gold"] == g and r["pred"] == p][:3]
        for r in exs:
            print(f"    native={r['native']!r}")
            print(f"    text  ={r['text'][:120]!r}")
    json.dump(recs, open("results_llm_error_records.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote results_llm_error_records.json ({len(recs)} records)")
    try:
        from results_log import save_result
        acc = accuracy_score(gold, pred)
        macro = f1_score(gold, pred, average="macro", zero_division=0)
        save_result("llm_error_analysis",
                    {"overall_acc": round(float(acc), 4), "overall_macroF1": round(float(macro), 4),
                     "n": len(recs), "n_errors": sum(1 for r in recs if not r["correct"])},
                    config={"model": model, "cap_per_city": cap},
                    per_item=recs, note="per-example error records with native label")
    except Exception as e:
        print(f"[results_log] skipped: {e}")


if __name__ == "__main__":
    main()
