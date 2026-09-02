"""Rulebook v4: iterative mining of natural-language CLASSIFICATION INSTRUCTIONS (not keywords),
each kept only if re-prompting the classifier with it IMPROVES held-out accuracy.

Loop (within one organization): base-classify -> find top confusion pair (predicted X, filed Y) ->
ask the LLM to write ONE concise instruction that would correctly route those cases -> re-classify a
validation slice with the accumulated instructions + candidate -> KEEP the instruction only if
validation accuracy rises -> repeat. Apply the final instruction block to the test set.

Fixes v1 (which injected unvalidated instructions and corrupted good predictions): every instruction
must pay its way on held-out data. Compare to the keyword version (per_city_rulebook.py).

  python rulebook_v4.py Gainesville
"""
import sys, os, json
import numpy as np
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from eval_common import load_split, LABELS
from pilot_rulebook import parse_label
from openai_batch import client

MODEL = "gpt-4o-mini"
N_MINE = 400
N_TEST = 500
DESC = {
 "Waste_Sanitation": "garbage or recycling collection, missed pickup, litter, illegal dumping, debris",
 "Streets_Sidewalks": "pothole or road, sidewalk, curb or driveway repair, blocked street",
 "Street_Lighting": "street light or lamp out or damaged",
 "Traffic_Signals_Signs": "traffic signal, traffic or street sign, road markings, speeding",
 "Trees_Vegetation": "trees, tree limbs, overgrown grass or weeds, vegetation, mowing",
 "Graffiti_Postings": "graffiti or illegal postings and signs",
 "Parking_Vehicles": "abandoned or junk vehicle, illegal parking, parking meter",
 "Property_Housing_Code": "building or zoning code violation, vacant or unsafe building, rental housing",
 "Water_Sewer_Drainage": "sewer, drainage, flooding, catch basin, manhole, water quality, pipe",
 "Homelessness": "homeless encampment or person living in a vehicle or on the sidewalk",
 "Animals_Pests": "animal control, stray pet, wildlife, rodents or pests",
 "Noise": "noise complaint", "Transit": "public transit or bus, taxi, scooter",
 "Parks_Recreation": "park, playground, beach, trail or recreation facility",
}
CATLIST = "\n".join(f"- {k}: {DESC[k]}" for k in LABELS)
SYS = ("You categorize municipal 311 service requests. Choose the SINGLE best category. "
       "Reply with ONLY the exact category name.")


def clsprompt(t, instructions):
    p = f"Categories:\n{CATLIST}\n\n"
    if instructions:
        p += "This jurisdiction's filing rules (follow them):\n" + "\n".join(f"- {x}" for x in instructions) + "\n\n"
    return p + f"Request:\n\"\"\"{t[:1000]}\"\"\"\n\nCategory:"


def classify_all(texts, instructions):
    out = [None] * len(texts)
    def one(i):
        for a in range(3):
            try:
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=15,
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": clsprompt(texts[i], instructions)}])
                return parse_label(r.choices[0].message.content)
            except Exception:
                import time; time.sleep(1.5 * (a + 1))
        return "UNPARSED"
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(one, i): i for i in range(len(texts))}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    return out


def write_instruction(X, Y, texts):
    sample = "\n".join(f"- {t[:200]}" for t in texts[:20])
    msg = (f"A classifier labeled these requests '{X}', but this jurisdiction files them as '{Y}'. "
           f"Write ONE concise, general instruction (max 25 words) that tells a classifier when to "
           f"choose '{Y}' over '{X}' for cases like these. State the textual cue and the target category.\n\n"
           f"Examples:\n{sample}\n\nInstruction:")
    try:
        r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=60,
            messages=[{"role": "system", "content": "You write concise, general classification rules."},
                      {"role": "user", "content": msg}])
        return r.choices[0].message.content.strip().strip('"')
    except Exception:
        return ""


def acc(pred, gold):
    return np.mean([pred[i] == gold[i] for i in range(len(gold))])


def run(city):
    sp = load_split(); rng = np.random.RandomState(0)
    tr = sp["train"][city]; idx = rng.permutation(len(tr))[:N_MINE]
    mine = [(tr[i][0], tr[i][1]) for i in idx]
    test = [(t, y) for t, y in sp["test"][city]][:N_TEST]
    pm = rng.permutation(len(mine)); cut = int(0.6 * len(mine))
    m = [mine[i] for i in pm[:cut]]; v = [mine[i] for i in pm[pm.size and cut:]]
    m_txt = [t for t, y in m]; m_gold = [y for t, y in m]
    v_txt = [t for t, y in v]; v_gold = [y for t, y in v]
    print(f"[{city}] mine={len(m)} val={len(v)} test={len(test)}  base-classifying...")
    m_base = classify_all(m_txt, [])
    v_base = classify_all(v_txt, [])
    instructions = []
    v_cur = list(v_base); v_acc = acc(v_base, v_gold)
    print(f"  base val acc={v_acc:.4f}")
    tried = set()
    for it in range(20):
        # top uncovered confusion on the mine set under CURRENT instructions
        m_cur = classify_all(m_txt, instructions) if instructions else m_base
        conf = Counter((m_cur[i], m_gold[i]) for i in range(len(m)) if m_cur[i] != "UNPARSED" and m_cur[i] != m_gold[i])
        pair = next((p for p, _ in conf.most_common() if p not in tried and conf[p] >= 5), None)
        if pair is None:
            break
        tried.add(pair); X, Y = pair
        exs = [m_txt[i] for i in range(len(m)) if m_cur[i] == X and m_gold[i] == Y]
        instr = write_instruction(X, Y, exs)
        if not instr:
            continue
        v_try = classify_all(v_txt, instructions + [instr])
        a_try = acc(v_try, v_gold)
        keep = a_try > v_acc + 0.002
        print(f"  it{it} {X}->{Y} (n={conf[pair]}) val {v_acc:.4f}->{a_try:.4f} {'KEEP' if keep else 'drop'}: {instr[:80]}")
        if keep:
            instructions.append(instr); v_acc = a_try
    # evaluate on test
    t_txt = [t for t, y in test]; t_gold = [y for t, y in test]
    t_base = classify_all(t_txt, [])
    t_rule = classify_all(t_txt, instructions)
    a_base, a_rule = acc(t_base, t_gold), acc(t_rule, t_gold)
    # placebo: same number of GENERIC (mismatched) instructions -> control for 'more text'
    print(f"\n[{city}] TEST: no_rules={a_base:.4f}  instructions={a_rule:.4f}  delta={a_rule-a_base:+.4f}  "
          f"(#instructions={len(instructions)})")
    for x in instructions:
        print("   *", x)
    json.dump({"city": city, "n_instructions": len(instructions), "instructions": instructions,
               "test_no_rules": float(a_base), "test_rules": float(a_rule), "delta": float(a_rule - a_base),
               "val_final": float(v_acc)}, open(f"results/rulebook_v4_{city}.json", "w"), indent=2)
    print(f"wrote results/rulebook_v4_{city}.json")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "Gainesville")
