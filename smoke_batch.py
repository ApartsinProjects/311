"""Smoke test the OpenRouter Batch API path end-to-end on a few 311 classification requests.
Self-contained (no sklearn import) so it runs even under memory pressure."""
import sys, re
from or_batch import run_chat_batch

LABELS = ["Waste_Sanitation", "Streets_Sidewalks", "Street_Lighting", "Traffic_Signals_Signs",
          "Trees_Vegetation", "Graffiti_Postings", "Parking_Vehicles", "Property_Housing_Code",
          "Water_Sewer_Drainage", "Homelessness", "Animals_Pests", "Noise", "Transit", "Parks_Recreation"]
SYS = "You are an expert municipal 311 dispatcher. Classify each citizen service request into exactly one category."

def build_prompt(text):
    cats = "\n".join(f"- {l}" for l in LABELS)
    return (f"Categories (choose exactly one, respond with the category name verbatim):\n{cats}\n\n"
            f"Service request text:\n\"\"\"{text[:800]}\"\"\"\n\nAnswer with ONLY the single best category name.")

def parse_label(out):
    s = (out or "").strip()
    if s in LABELS: return s
    low = s.lower()
    for l in LABELS:
        if l.lower() in low: return l
    return "UNPARSED"

TEXTS = [
    "Large pothole on NW 2nd Ave, popped my tire.",
    "Graffiti sprayed on the wall of the library.",
    "Missed garbage pickup on my street this week.",
    "Street light out on the corner of Main and 5th.",
    "Homeless encampment growing under the overpass.",
    "Tree limb hanging low over the sidewalk after the storm.",
]

def build_body(text):
    return {"messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": build_prompt(text)}],
            "temperature": 0, "max_tokens": 15}

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-4o-mini"
    items = [(i, t) for i, t in enumerate(TEXTS)]
    print(f"Batch smoke test: model={model}, {len(items)} requests")
    res = run_chat_batch(model, items, build_body, interval=10)
    print("\n=== results ===")
    ok = 0
    for i, t in items:
        raw = res.get(str(i), "MISSING")
        lab = parse_label(raw) if not str(raw).startswith("ERR") else raw
        ok += lab not in ("UNPARSED", "MISSING") and not str(lab).startswith("ERR")
        print(f"  [{i}] {t[:45]:47s} -> {lab}")
    print(f"\n{ok}/{len(items)} parsed to a valid label. Batch path {'WORKS' if ok==len(items) else 'has issues'}.")

if __name__ == "__main__":
    main()
