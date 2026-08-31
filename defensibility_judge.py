"""
defensibility_judge.py -- rigorous (LLM-adjudicated) quantification of the label-artifact claim.

For every FROZEN test row, an LLM judge returns the SET of acceptable categories for the text,
BLIND to any model prediction and to the city's label. From these sets we later compute:
  - strict accuracy (pred == city gold)              [as before]
  - lenient accuracy (pred in acceptable set)          [credits defensible alternatives]
  - city-label-noise rate (city gold NOT in acceptable set)  [how noisy the gold is]
This measures the artifact share on the SAME population the gap is computed on, replacing the
earlier single-annotator anecdote. Human validation of a subset remains future work.

Saves results/preds/acceptable_sets.json = {city: [[label,...], ...]} aligned to split test rows.
Usage: python defensibility_judge.py [model] [workers]
"""
import sys, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from eval_common import load_split, LABELS, PRED_DIR
from llm_arm import make_client
import os

CONTENT_GLOSS = {
 "Waste_Sanitation": "garbage/recycling collection, missed pickup, carts, litter, illegal dumping, debris, dead animal pickup",
 "Streets_Sidewalks": "potholes, road/sidewalk/curb/driveway repair, blocked street, snow removal",
 "Street_Lighting": "street lights/lamps out or damaged",
 "Traffic_Signals_Signs": "traffic signals, traffic/street signs, markings, speeding, crosswalks",
 "Trees_Vegetation": "trees, limbs, overgrown grass/lots, mowing, weeds, vegetation",
 "Graffiti_Postings": "graffiti, illegal postings/signs",
 "Parking_Vehicles": "abandoned/junk vehicles, illegal parking, meters, shopping carts",
 "Property_Housing_Code": "building/zoning code, vacant/unsafe buildings, rental housing, permits, property damage",
 "Water_Sewer_Drainage": "sewer, drainage, flooding, catch basins, manholes, water quality, pipes, hydrants",
 "Homelessness": "homeless encampments, people living in vehicles/sidewalks",
 "Animals_Pests": "animal control, pets, wildlife, rodents, pests",
 "Noise": "noise complaints",
 "Transit": "bus/transit, taxis, scooters",
 "Parks_Recreation": "parks, playgrounds, beaches, trails, recreation",
}
SYS = ("You are validating a municipal 311 taxonomy. For a complaint text, list EVERY category that "
       "is a reasonable classification of what the text describes (usually 1, sometimes 2-3 for "
       "multi-topic or genuinely ambiguous text). Judge only from the text.")


def prompt(text):
    cats = "\n".join(f"- {k}: {v}" for k, v in CONTENT_GLOSS.items())
    return (f"Categories:\n{cats}\n\nComplaint text:\n\"\"\"{text[:700]}\"\"\"\n\n"
            f"Reply with ONLY the acceptable category names, comma-separated, from the list above.")


def parse_set(out):
    found = []
    low = (out or "").lower()
    for l in LABELS:
        if l.lower() in low:
            found.append(l)
    return found or ["UNPARSED"]


def judge_one(client, text, model):
    for a in range(4):
        try:
            r = client.chat.completions.create(model=model, temperature=0, max_tokens=40,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": prompt(text)}])
            return parse_set(r.choices[0].message.content)
        except Exception:
            import time; time.sleep(2 * (a + 1))
    return ["ERR"]


def _body(text):
    return {"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt(text)}],
            "temperature": 0, "max_tokens": 40}


def main():
    """Batch-first acceptable-set judge (50% off; survives reaping via submit/collect state).
      python defensibility_judge.py <model> <tag> [submit|collect|run]
    Writes results/preds/acceptable_sets_<tag>.json (or acceptable_sets.json when tag=='default')."""
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
    tag = sys.argv[2] if len(sys.argv) > 2 else "default"
    mode = sys.argv[3] if len(sys.argv) > 3 else "run"
    sp = load_split()
    items = [(f"{c}|{i}", t) for c, rows in sp["test"].items() for i, (t, _) in enumerate(rows)]
    out_name = "acceptable_sets.json" if tag == "default" else f"acceptable_sets_{tag}.json"

    provider = "openrouter" if "/" in model else "openai"
    if provider == "openai":
        import openai_batch as B
        submit = lambda: B.submit_chat_batch(model, [(cid, _body(t)) for cid, t in items], tag=f"judge_{tag}")
    else:
        import or_batch as B
        submit = lambda: B.submit_chat_batch(model, items, lambda t: _body(t), tag=f"judge_{tag}")

    def finish(res):
        out = {}
        for c, rows in sp["test"].items():
            out[c] = [parse_set(res.get(f"{c}|{i}", "")) for i in range(len(rows))]
            gold = [y for _, y in rows]
            noise = sum(1 for g, s in zip(gold, out[c]) if g not in s) / len(gold)
            print(f"[judge] {c}: city-label NOT in acceptable set = {noise:.1%}")
        os.makedirs(PRED_DIR, exist_ok=True)
        json.dump(out, open(os.path.join(PRED_DIR, out_name), "w"), ensure_ascii=False)
        print(f"[judge] wrote {PRED_DIR}/{out_name}  (model={model})")

    if mode == "submit":
        submit(); print(f"[judge] submitted; collect with: python defensibility_judge.py {model} {tag} collect")
    elif mode == "collect":
        res = B.collect_chat_batch(tag=f"judge_{tag}")
        print("[judge] batch not ready yet; rerun collect later") if res is None else finish(res)
    else:  # run: submit + bounded poll (state saved, so a reap is recoverable via collect)
        submit()
        import time as _t
        res = None; t0 = _t.time()
        while _t.time() - t0 < 1800:
            res = B.collect_chat_batch(tag=f"judge_{tag}")
            if res is not None:
                break
            _t.sleep(20)
        print(f"[judge] still running; resume: python defensibility_judge.py {model} {tag} collect") if res is None else finish(res)


if __name__ == "__main__":
    main()
