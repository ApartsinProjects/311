"""
eval_fewshot_batch.py -- few-shot LLM arm (Exp B / W8): does in-context target-label semantics,
not just a taxonomy, close the cross-jurisdiction gap? Prepends k exemplars (one per class, drawn
from TRAINING cities so it stays cross-city) to the classification prompt. Runs on the frozen test
set via the batch API (50% off). Self-contained.

Usage: python eval_fewshot_batch.py gpt-4o-mini gpt4omini_fewshot
Saves results/preds/llm_<tag>.json = {"zeroshot": {city:[label,...]}}
"""
import csv, os, sys, json
from collections import defaultdict, OrderedDict
csv.field_size_limit(10**7)
DATA = "data"; PRED = os.path.join("results", "preds")
LABELS = ["Waste_Sanitation", "Streets_Sidewalks", "Street_Lighting", "Traffic_Signals_Signs",
          "Trees_Vegetation", "Graffiti_Postings", "Parking_Vehicles", "Property_Housing_Code",
          "Water_Sewer_Drainage", "Homelessness", "Animals_Pests", "Noise", "Transit", "Parks_Recreation"]
GLOSS = {
 "Waste_Sanitation":"garbage/recycling, missed pickup, carts, litter, illegal dumping, debris, dead animal pickup",
 "Streets_Sidewalks":"potholes, road/sidewalk/curb/driveway repair, blocked street, snow removal",
 "Street_Lighting":"street lights/lamps out, damaged, or requested",
 "Traffic_Signals_Signs":"traffic signals, traffic/street signs, markings, speeding, crosswalks",
 "Trees_Vegetation":"trees, limbs, overgrown grass/lots, mowing, weeds, vegetation",
 "Graffiti_Postings":"graffiti, illegal postings/signs",
 "Parking_Vehicles":"abandoned/junk vehicles, illegal parking, meters, shopping carts",
 "Property_Housing_Code":"building/zoning code, vacant/unsafe buildings, rental housing, permits, property damage",
 "Water_Sewer_Drainage":"sewer, drainage, flooding, catch basins, manholes, water quality, pipes, hydrants",
 "Homelessness":"homeless encampments, people living in vehicles/sidewalks",
 "Animals_Pests":"animal control, pets, wildlife, rodents, pests",
 "Noise":"noise complaints", "Transit":"public transit/bus, taxis, scooters",
 "Parks_Recreation":"parks, playgrounds, beaches, trails, recreation",
}
SYS = "You are an expert municipal 311 dispatcher. Reply with ONLY one category name."


def load_split():
    tr, te = defaultdict(list), defaultdict(list)
    with open(os.path.join(DATA, "eval_split.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            (tr if r["role"] == "train" else te)[r["city"]].append((r["text"], r["label"]))
    return tr, te


import re as _re
_JUNK = _re.compile(r"test only|should go to|reply-?referred|referred to|wanted public info|"
                    r"reporting cell phone|see picture|see attach|per (lt|sgt|capt)", _re.I)


def pick_exemplars(train):
    """One clean, representative exemplar per class: filter test stubs / referrals / mislabeled
    junk, then take the median-length remaining request (deterministic, prototypical, not the
    first all-caps call-center row)."""
    ex = OrderedDict()
    for lab in LABELS:
        cands = [t for rows in train.values() for (t, y) in rows
                 if y == lab and 25 <= len(t) <= 110 and not _JUNK.search(t)]
        if not cands:
            cands = [t for rows in train.values() for (t, y) in rows if y == lab]
        cands.sort(key=len)
        ex[lab] = cands[len(cands) // 2].strip().replace("\n", " ")
    return ex


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
    tag = sys.argv[2] if len(sys.argv) > 2 else "fewshot"
    tr, te = load_split()
    ex = pick_exemplars(tr)
    cats = "\n".join(f"- {k}: {GLOSS[k]}" for k in LABELS)
    shots = "\n".join(f'Request: "{t}" -> {lab}' for lab, t in ex.items())

    def build_body(text):
        user = (f"Categories:\n{cats}\n\nExamples:\n{shots}\n\n"
                f"Now classify this request. Reply with ONLY the category name.\nRequest: \"{text[:600]}\" ->")
        return {"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                "temperature": 0, "max_tokens": 12}

    items = []
    for city, rows in te.items():
        for i, (t, _) in enumerate(rows):
            items.append((f"{city}|{i}", t))
    print(f"[fewshot] model={model} tag={tag} items={len(items)} exemplars={len(ex)}")

    provider = "openrouter" if "/" in model else "openai"
    if provider == "openai":
        from openai_batch import run_chat_batch as run
        bodies = [(cid, build_body(t)) for cid, t in items]
        res = run(model, bodies, interval=20)
    else:
        from or_batch import run_chat_batch as run
        res = run(model, items, build_body, interval=20)

    def parse(o):
        low = (o or "").lower()
        for l in LABELS:
            if l.lower() in low: return l
        return "UNPARSED"
    out = {"zeroshot": {}}
    for city, rows in te.items():
        out["zeroshot"][city] = [parse(res.get(f"{city}|{i}", "")) for i in range(len(rows))]
    os.makedirs(PRED, exist_ok=True)
    json.dump(out, open(os.path.join(PRED, f"llm_{tag}.json"), "w"), ensure_ascii=False)
    print(f"[fewshot] wrote results/preds/llm_{tag}.json")


if __name__ == "__main__":
    main()
