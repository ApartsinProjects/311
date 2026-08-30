"""
eval_llm_batch.py -- run the LLM classification arm OR the acceptable-set judge on the FROZEN
test set via the OpenRouter Batch API (50% discount). Self-contained (no sklearn), so it runs
under memory pressure. Only models with a :batch endpoint work (e.g. google/gemini-2.5-flash,
z-ai/glm-5.3-flash, anthropic/claude-sonnet-5, openai/o3); gpt-4o-mini has NO batch endpoint.

Usage:
  python eval_llm_batch.py classify google/gemini-2.5-flash gemini25flash
  python eval_llm_batch.py judge    google/gemini-2.5-flash
Outputs (scorer-compatible):
  classify -> results/preds/llm_<tag>.json  = {"zeroshot": {city:[label,...]}}
  judge    -> results/preds/acceptable_sets_<tag>.json = {city:[[label,...],...]}
"""
import csv, os, sys, json
from collections import defaultdict


def _run_batch(model, items, build_body, interval=20):
    """Route to OpenAI native batch (bare model id, e.g. 'gpt-4o-mini') or OpenRouter batch
    (namespaced id, e.g. 'google/gemini-2.5-flash'). Both give the 50% batch discount."""
    provider = os.environ.get("PROVIDER") or ("openrouter" if "/" in model else "openai")
    if provider == "openai":
        from openai_batch import run_chat_batch as run_oai
        bodies = [(cid, build_body(text)) for cid, text in items]
        return run_oai(model, bodies, interval=interval)
    from or_batch import run_chat_batch as run_or
    return run_or(model, items, build_body, interval=interval)

csv.field_size_limit(10**7)
DATA = "data"; PRED = os.path.join("results", "preds")
LABELS = ["Waste_Sanitation", "Streets_Sidewalks", "Street_Lighting", "Traffic_Signals_Signs",
          "Trees_Vegetation", "Graffiti_Postings", "Parking_Vehicles", "Property_Housing_Code",
          "Water_Sewer_Drainage", "Homelessness", "Animals_Pests", "Noise", "Transit", "Parks_Recreation"]
GLOSS = {
 "Waste_Sanitation":"garbage/recycling collection, missed pickup, carts, litter, illegal dumping, debris, dead animal pickup",
 "Streets_Sidewalks":"potholes, road/sidewalk/curb/driveway repair, blocked street, snow removal",
 "Street_Lighting":"street lights or lamps out, damaged, or requested",
 "Traffic_Signals_Signs":"traffic signals, traffic/street signs, markings, speeding, crosswalks",
 "Trees_Vegetation":"trees, limbs, overgrown grass/lots, mowing, weeds, vegetation",
 "Graffiti_Postings":"graffiti, illegal postings/signs",
 "Parking_Vehicles":"abandoned/junk vehicles, illegal parking, meters, shopping carts",
 "Property_Housing_Code":"building/zoning code, vacant/unsafe buildings, rental housing, permits, property damage",
 "Water_Sewer_Drainage":"sewer, drainage, flooding, catch basins, manholes, water quality, pipes, hydrants",
 "Homelessness":"homeless encampments, people living in vehicles/sidewalks",
 "Animals_Pests":"animal control, pets, wildlife, rodents, pests",
 "Noise":"noise complaints",
 "Transit":"public transit/bus, taxis, scooters",
 "Parks_Recreation":"parks, playgrounds, beaches, trails, recreation",
}
SYS_CLS = "You are an expert municipal 311 dispatcher. Reply with ONLY one category name."
SYS_JUDGE = ("You are validating a municipal 311 taxonomy. For a complaint text, list EVERY category that is a "
             "reasonable classification of what the text describes (usually 1, sometimes 2-3). Judge only from the text.")


def load_test():
    by = defaultdict(list)
    with open(os.path.join(DATA, "eval_split.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["role"] == "test":
                by[r["city"]].append(r["text"])
    return by


def cls_prompt(text):
    cats = "\n".join(f"- {k}: {v}" for k, v in GLOSS.items())
    return (f"Categories (choose exactly one; reply with the category name verbatim):\n{cats}\n\n"
            f"Service request:\n\"\"\"{text[:700]}\"\"\"\n\nCategory:")


def judge_prompt(text):
    cats = "\n".join(f"- {k}: {v}" for k, v in GLOSS.items())
    return (f"Categories:\n{cats}\n\nComplaint text:\n\"\"\"{text[:700]}\"\"\"\n\n"
            f"Reply with ONLY the acceptable category names, comma-separated, from the list above.")


def parse_one(out):
    low = (out or "").lower()
    for l in LABELS:
        if l.lower() in low:
            return l
    return "UNPARSED"


def parse_set(out):
    low = (out or "").lower()
    found = [l for l in LABELS if l.lower() in low]
    return found or ["UNPARSED"]


def main():
    mode = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "google/gemini-2.5-flash"
    tag = sys.argv[3] if len(sys.argv) > 3 else model.split("/")[-1].replace(".", "").replace("-", "")
    by = load_test()
    # one flat item list with custom_id = "city|idx" so we can regroup
    items, order = [], []
    sysmsg = SYS_CLS if mode == "classify" else SYS_JUDGE
    pfn = cls_prompt if mode == "classify" else judge_prompt
    cap = int(os.environ.get("SMOKE_N", "0"))  # >0 = cap items per city for a cheap smoke test
    for city, texts in by.items():
        use = texts[:cap] if cap else texts
        by[city] = use
        for i, t in enumerate(use):
            cid = f"{city}|{i}"
            items.append((cid, t)); order.append((city, i))
    def build_body(text):
        return {"messages": [{"role": "system", "content": sysmsg},
                             {"role": "user", "content": pfn(text)}],
                "temperature": 0, "max_tokens": 15 if mode == "classify" else 40}
    print(f"[batch-{mode}] model={model} tag={tag} items={len(items)}")
    res = _run_batch(model, items, build_body, interval=20)

    os.makedirs(PRED, exist_ok=True)
    if mode == "classify":
        out = {"zeroshot": defaultdict(list)}
        for city, texts in by.items():
            for i in range(len(texts)):
                raw = res.get(f"{city}|{i}", "")
                out["zeroshot"][city].append(parse_one(raw))
        out["zeroshot"] = dict(out["zeroshot"])
        p = os.path.join(PRED, f"llm_{tag}.json")
        json.dump(out, open(p, "w"), ensure_ascii=False)
    else:
        out = {}
        for city, texts in by.items():
            out[city] = [parse_set(res.get(f"{city}|{i}", "")) for i in range(len(texts))]
        p = os.path.join(PRED, f"acceptable_sets_{tag}.json")
        json.dump(out, open(p, "w"), ensure_ascii=False)
    nbad = sum(1 for v in res.values() if str(v).startswith("ERR"))
    print(f"[batch-{mode}] wrote {p}  ({len(res)} results, {nbad} errors)")


if __name__ == "__main__":
    main()
