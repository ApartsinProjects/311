"""
harmonize.py  --  v1 harmonization ontology for the multi-city 311 benchmark.

Maps each city's native service categories into a shared 15-class civic taxonomy so
cross-city labels are comparable. Rule-based, ordered, FIRST-MATCH-WINS (specific rules
before general ones). Emits:
  data/harmonization_map.json   {city: {native_category: super_class}}
  data/harmonization_pivot.csv  super_class x city  (record counts, from native_categories.json)
and prints coverage + the largest categories that fell through to General_Admin_Other.

The mapping is applied by name (native category string), NOT by free text, so it is
transparent and auditable. Re-run after editing RULES.
"""
import json, os, re, csv
from collections import defaultdict, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# 15-class shared taxonomy. Order = match priority (first match wins).
RULES = [
 ("Homelessness",         r"homeless|encampment|\btent\b"),
 ("Graffiti_Postings",    r"graffiti|illegal post|illegal sign|illegal placard|posting"),
 ("Noise",                r"noise"),
 ("Transit",              r"\bmuni\b|\bbus\b|handi-?van|transit|\btaxi\b|scooter|bike-?share|\bmoped\b|skyline|\bveo\b|\bbird\b|\blime\b"),
 ("Animals_Pests",        r"animal|\bpet\b|wildlife|chicken|rodent|mosquito|\bpest|feral|crow"),
 ("Trees_Vegetation",     r"\btree|limb|brush|vegetation|overgrow|overgrowth|excessive growth|tall grass|\bmow|\bgrass|\bweed|leaf|stream|canal clean|botanical|garden|median sprinkler|land clearing"),
 ("Street_Lighting",      r"street ?light|streetlight|\blamp\b|nuisance light"),
 ("Traffic_Signals_Signs",r"traffic signal|traffic light|signal issue|signal study|pavement marking|restrip|road striping|traffic sign|street sign|street name sign|name sign|sign repair|dumping sign|speeding|sight distance|blind street|line of sight|school zone|crosswalk|traffic calming|traffic safety|traffic issue|traffic related|traffic suggestion|traffic enforcement|guardrail|safety study"),
 ("Water_Sewer_Drainage", r"sewer|\bdrain|flood|catch basin|manhole|storm ?water|storm drain|stormdrain|water quality|water utility|water pipe|pipe repair|standing water|water ponding|pump station|liberty pump|culvert|ditch|erosion|cave-in|sink ?hole|hydrant|waste ?water|\bwater\b|low pressure|waste of water|chemical spill|\bleak"),
 ("Parking_Vehicles",     r"abandoned.*(vehicle|car)|junk.*vehicle|inoperable|nuisance vehicle|\bparking\b|parking meter|parking enforcement|color curb|\brpp\b|shopping cart|autonomous vehicle|vehicle \(abandoned|vehicle gas tank"),
 ("Property_Housing_Code",r"building code|zoning|code violation|code issue|condemn|unsafe building|vacant|abandoned building|dangerous building|rental|housing|sfha|residential building|damage.?d? .*property|damaged property|damage property|personal property|missing windows|windows or doors|substandard|construction site|new construction|building permit|electrical code|mechanical code|blocked exit|smoke detector|\bfence|swimming pool|\bblight|overcrowd|short.?term rental|business license|sidewalk cafe|sign without a permit|signage w/?o|temporary sign|illegal sign in the public|fence line|servitude|land ?fill|central business district waste|inspection"),
 ("Parks_Recreation",     r"\bpark(s|\b)|playground|beach|\btrail|recreation|rec and park|rec buildings|rpd|community garden|volunteer program"),
 ("Waste_Sanitation",     r"garbage|trash|recycl|litter|debris|dumping|refuse|bulky|bulk pick|yard waste|woody waste|handpile|solid waste|\bcart\b|street sweep|street clean|sidewalk clean|receptacle|backup|overflow|dead animal|white goods|\btire|no dumping|special debris|street grading"),
 ("Streets_Sidewalks",    r"pothole|road repair|road buckle|road failure|street defect|street/road|street repair|\bcurb|driveway|sidewalk|\bshoulder|blocked street|blocked side|barricade|bike path|\broad\b|concrete road|gravel drive|street curb|sidewalk cafe|accessibility|ada access|defaced sidewalk|snow removal|\bsnow\b"),
]
RULES = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in RULES]
FALLBACK = "General_Admin_Other"
TAXONOMY = [n for n, _ in RULES] + [FALLBACK]


def classify(native):
    s = (native or "").strip()
    for name, rx in RULES:
        if rx.search(s):
            return name
    return FALLBACK


def main():
    nat = json.load(open(os.path.join(DATA, "native_categories.json"), encoding="utf-8"))
    mapping = OrderedDict()
    pivot = defaultdict(lambda: defaultdict(int))   # super -> city -> count
    other_examples = defaultdict(int)               # native -> count (only Other)
    per_city_total = defaultdict(int)
    per_city_other = defaultdict(int)

    for city, cats in nat.items():
        mapping[city] = {}
        for native, n in cats:
            sup = classify(native)
            mapping[city][native] = sup
            pivot[sup][city] += n
            per_city_total[city] += n
            if sup == FALLBACK:
                per_city_other[city] += n
                other_examples[f"{city} :: {native}"] += n

    json.dump(mapping, open(os.path.join(DATA, "harmonization_map.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    cities = list(nat.keys())
    with open(os.path.join(DATA, "harmonization_pivot.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["super_class"] + cities + ["TOTAL"])
        for sup in TAXONOMY:
            row = [pivot[sup][c] for c in cities]
            w.writerow([sup] + row + [sum(row)])

    # ---- report ----
    print(f"{'super_class':24s}" + "".join(f"{c[:9]:>10s}" for c in cities) + f"{'TOTAL':>11s}")
    for sup in TAXONOMY:
        row = [pivot[sup][c] for c in cities]
        print(f"{sup:24s}" + "".join(f"{v:>10d}" for v in row) + f"{sum(row):>11d}")

    print("\n--- coverage (share NOT in General_Admin_Other) ---")
    for c in cities:
        tot = per_city_total[c]; oth = per_city_other[c]
        print(f"  {c:14s} {100*(tot-oth)/tot:5.1f}% covered   ({oth:,} of {tot:,} -> Other)")

    print("\n--- largest native categories that fell to General_Admin_Other (review these) ---")
    for k, n in sorted(other_examples.items(), key=lambda x: -x[1])[:25]:
        print(f"  {n:>9,}  {k}")


if __name__ == "__main__":
    main()
