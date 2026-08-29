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


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-4o-mini"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    client, prov = make_client(r"E:\Projects\.env.all")
    sp = load_split()
    out = {}
    for c, rows in sp["test"].items():
        texts = [t for t, _ in rows]; sets = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(judge_one, client, texts[i], model): i for i in range(len(texts))}
            for f in as_completed(futs):
                sets[futs[f]] = f.result()
        out[c] = sets
        # quick city-label-noise readout
        gold = [y for _, y in rows]
        noise = sum(1 for g, s in zip(gold, sets) if g not in s) / len(gold)
        print(f"[judge] {c}: city-label NOT in acceptable set = {noise:.1%}")
    os.makedirs(PRED_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(PRED_DIR, "acceptable_sets.json"), "w"), ensure_ascii=False)
    print(f"[judge] wrote {PRED_DIR}/acceptable_sets.json  (model={model})")


if __name__ == "__main__":
    main()
