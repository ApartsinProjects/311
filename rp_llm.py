"""
rp_llm.py -- Open-LLM zero-shot arm for the multi-city 311 benchmark, RunPod edition.

Loads an open instruct model (default Qwen2.5-7B-Instruct), gives it the shared 14-class
taxonomy IN THE PROMPT, and classifies a sample of each city's complaints. No training,
no API credits. This is the cross-jurisdiction LLM condition to compare against fine-tuned
LEAVE-ONE-CITY-OUT. Writes results/results_llm_open.json. Follows gpu2runpod conventions.
"""
import csv, json, os, sys, argparse, re
from collections import defaultdict
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, accuracy_score

csv.field_size_limit(10**7)
assert torch.cuda.is_available(), "CUDA not available. This script requires a GPU."
DEV = torch.device("cuda")
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

CONTENT = {
 "Waste_Sanitation":"garbage/trash/recycling, missed pickup, carts, litter, illegal dumping, debris, street sweeping, dead animal pickup",
 "Streets_Sidewalks":"potholes, road/pavement defects, sidewalk/curb/driveway repair, blocked street/sidewalk, snow removal",
 "Street_Lighting":"street lights or lamps out, damaged, flickering, or requested",
 "Traffic_Signals_Signs":"traffic signals/lights, traffic or street-name signs, pavement markings/striping, speeding, crosswalks",
 "Trees_Vegetation":"trees, limbs, brush, overgrown grass/lots, mowing, weeds, vegetation, stream/canal clearing",
 "Graffiti_Postings":"graffiti, illegal postings, illegal signs on public property",
 "Parking_Vehicles":"abandoned/junk vehicles, illegal or nuisance parking, parking enforcement, meters, shopping carts",
 "Property_Housing_Code":"building/zoning code violations, vacant/unsafe buildings, rental housing, permits, fences, pools, construction sites, property damage",
 "Water_Sewer_Drainage":"sewer, drainage, flooding, catch basins, manholes, storm drains, water quality/utility, pipes, hydrants, leaks",
 "Homelessness":"homeless encampments, people living in vehicles or on sidewalks, related cleanups",
 "Animals_Pests":"animal control, stray/neglected pets, wildlife, feral chickens, rodents, mosquitoes, pests",
 "Noise":"noise complaints",
 "Transit":"public transit/bus feedback, taxis, scooters, bike-share",
 "Parks_Recreation":"parks, playgrounds, beaches, trails, recreation facilities/programs",
}
LABELS = list(CONTENT)
NORM = {l.lower(): l for l in LABELS}


def load(path, n_per_city, seed=0):
    by = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"] in NORM.values() and len((r["text"] or "").strip()) >= 3:
                by[r["city"]].append((r["text"].strip(), r["label"]))
    rng = np.random.RandomState(seed)
    out = {}
    for c, rows in by.items():
        if len(rows) > n_per_city:
            idx = rng.permutation(len(rows))[:n_per_city]
            rows = [rows[i] for i in idx]
        out[c] = rows
    return out


def parse(out):
    s = (out or "").strip()
    if s in LABELS: return s
    low = s.lower()
    if low in NORM: return NORM[low]
    for l in LABELS:
        if l.lower() in low: return l
    return "UNPARSED"


def build_msgs(text):
    cats = "\n".join(f"- {k}: {v}" for k, v in CONTENT.items())
    user = (f"Categories (choose exactly one; reply with the category name verbatim):\n{cats}\n\n"
            f"Service request:\n\"\"\"{text[:700]}\"\"\"\n\nCategory:")
    return [{"role": "system", "content": "You are an expert municipal 311 dispatcher. Reply with ONLY one category name."},
            {"role": "user", "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="harmonized.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-per-city", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    print(f"[train] loading {args.model}"); sys.stdout.flush()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    print("[train] model loaded"); sys.stdout.flush()

    data = load(args.data, args.n_per_city)
    cities = sorted(data, key=lambda c: -len(data[c]))
    results = {}
    for c in cities:
        rows = data[c]; texts = [t for t, _ in rows]; gold = [y for _, y in rows]
        preds = []
        for i in range(0, len(texts), args.bs):
            chunk = texts[i:i + args.bs]
            prompts = [tok.apply_chat_template(build_msgs(t), tokenize=False, add_generation_prompt=True) for t in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEV)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                new = gen[j][enc["input_ids"].shape[1]:]
                preds.append(parse(tok.decode(new, skip_special_tokens=True)))
            print(f"  llm:{c} {min(i+args.bs,len(texts))}/{len(texts)}"); sys.stdout.flush()
        y_pred = [p if p in LABELS else "UNPARSED" for p in preds]
        macro = float(f1_score(gold, y_pred, average="macro", labels=LABELS, zero_division=0))
        acc = float(accuracy_score(gold, y_pred))
        nbad = sum(1 for p in y_pred if p == "UNPARSED")
        results[c] = {"n": len(gold), "macroF1": round(macro, 4), "acc": round(acc, 4),
                      "unparsed": nbad, "gold_classes": len(set(gold))}
        print(f"[train] LLM {c:14s} macroF1={macro:.3f} acc={acc:.3f} unparsed={nbad}"); sys.stdout.flush()

    mm = float(np.mean([r["macroF1"] for r in results.values()]))
    ma = float(np.mean([r["acc"] for r in results.values()]))
    print(f"[train] SUMMARY mean_macroF1={mm:.3f} mean_acc={ma:.3f}"); sys.stdout.flush()
    json.dump({"model": args.model, "n_per_city": args.n_per_city, "per_city": results,
               "mean_macroF1": round(mm, 4), "mean_acc": round(ma, 4)},
              open("results/results_llm_open.json", "w"), indent=2)
    print("[train] === DONE ==="); sys.stdout.flush()


if __name__ == "__main__":
    main()
