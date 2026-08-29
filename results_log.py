"""
results_log.py -- one persistent home for every experiment result.

save_result(arm, metrics, config=None, per_item=None, note="") writes:
  results/<arm>_<timestamp>.json   full record (incl. per_item if given)
  results/runs.jsonl               append-only master log (one line per run, no per_item)
  results/summary.csv              flat table (union of metric keys) rebuilt from runs.jsonl

Every arm script calls this at the end so nothing is ephemeral. Import and call:
  from results_log import save_result
  save_result("tfidf_baseline", {"mean_loco": 0.38, ...}, config={"cap": 8000})
"""
import json, os, csv, datetime

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
        # lists / nested arrays are skipped in the flat CSV (kept in the JSON)
    return out


def _rebuild_summary():
    path = os.path.join(RES_DIR, "runs.jsonl")
    if not os.path.exists(path):
        return
    rows, keys = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            flat = {"run_id": rec["run_id"], "timestamp": rec["timestamp"], "arm": rec["arm"],
                    "note": rec.get("note", "")}
            flat.update(_flatten(rec.get("metrics"), "m."))
            flat.update(_flatten(rec.get("config"), "c."))
            rows.append(flat)
            for k in flat:
                if k not in keys:
                    keys.append(k)
    with open(os.path.join(RES_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_result(arm, metrics, config=None, per_item=None, note=""):
    os.makedirs(RES_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{arm}_{ts}"
    rec = {"run_id": run_id, "arm": arm, "timestamp": ts,
           "config": config or {}, "metrics": metrics, "note": note}
    with open(os.path.join(RES_DIR, run_id + ".json"), "w", encoding="utf-8") as f:
        json.dump({**rec, "per_item": per_item}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RES_DIR, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _rebuild_summary()
    print(f"[results_log] saved run {run_id} -> results/{run_id}.json (+ runs.jsonl, summary.csv)")
    return run_id


def ingest_external(arm, json_path, config=None, note=""):
    """Fold a RunPod-produced results JSON (e.g. results_distilbert.json) into the log."""
    data = json.load(open(json_path, encoding="utf-8"))
    metrics = data.get("summary", {}) or {}
    metrics.update({"incity": data.get("incity"), "loco": data.get("loco")})
    return save_result(arm, {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))} | {"detail": data},
                       config=config or data.get("config"), note=note)


if __name__ == "__main__":
    _rebuild_summary()
    print("rebuilt results/summary.csv from runs.jsonl")
