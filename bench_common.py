"""Shared loader + label taxonomy for the multi-city 311 benchmark arms."""
import csv, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
csv.field_size_limit(10**7)

# 14 content classes (General_Admin_Other excluded from the classification task) + glosses.
CONTENT_CLASSES = {
 "Waste_Sanitation":      "garbage/trash/recycling collection, missed pickup, carts, litter, illegal dumping, debris, street sweeping, dead animal pickup",
 "Streets_Sidewalks":     "potholes, road/pavement defects, sidewalk/curb/driveway repair, blocked street or sidewalk, snow removal, barricades",
 "Street_Lighting":       "street lights or lamps out, damaged, flickering, or requested",
 "Traffic_Signals_Signs": "traffic signals/lights, traffic or street-name signs, pavement markings/striping, speeding, sight distance, crosswalks",
 "Trees_Vegetation":      "trees, limbs, brush, overgrown grass/lots, mowing, weeds, vegetation, stream/canal clearing",
 "Graffiti_Postings":     "graffiti, illegal postings, illegal signs/placards on public property",
 "Parking_Vehicles":      "abandoned/junk vehicles, illegal or nuisance parking, parking enforcement, meters, shopping carts",
 "Property_Housing_Code": "building/zoning code violations, vacant/condemned/unsafe buildings, rental housing, permits, fences, pools, construction sites, property damage",
 "Water_Sewer_Drainage":  "sewer, drainage, flooding, catch basins, manholes, storm drains, water quality/utility, pipes, hydrants, leaks",
 "Homelessness":          "homeless encampments, individuals living in vehicles or on sidewalks, related cleanups",
 "Animals_Pests":         "animal control, stray/neglected pets, wildlife, feral chickens, rodents, mosquitoes, pests",
 "Noise":                 "noise complaints",
 "Transit":               "public transit/bus feedback, taxis, scooters, bike-share",
 "Parks_Recreation":      "parks, playgrounds, beaches, trails, recreation facilities/programs",
}
LABELS = list(CONTENT_CLASSES)
OTHER = "General_Admin_Other"


def load_harmonized(cap=0, drop_other=True, seed=0):
    """Return {city: [(text, super_class), ...]} using data/raw/all_cities.csv + harmonization_map.json."""
    mp = json.load(open(os.path.join(DATA, "harmonization_map.json"), encoding="utf-8"))
    by_city = defaultdict(list)
    with open(os.path.join(DATA, "raw", "all_cities.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            city, text, nat = r["city"], (r["text"] or "").strip(), r["native_category"]
            if len(text) < 3:
                continue
            sup = mp.get(city, {}).get(nat, OTHER)
            if drop_other and sup == OTHER:
                continue
            by_city[city].append((text, sup))
    if cap:
        rng = np.random.RandomState(seed)
        for c in by_city:
            if len(by_city[c]) > cap:
                idx = rng.permutation(len(by_city[c]))[:cap]
                by_city[c] = [by_city[c][i] for i in idx]
    return by_city


def sample_test(by_city, n_per_city, seed=0):
    """Stratified-ish sample of n rows per city for LLM evaluation (label-balanced where possible)."""
    rng = np.random.RandomState(seed)
    out = {}
    for c, data in by_city.items():
        if len(data) <= n_per_city:
            out[c] = list(data)
        else:
            idx = rng.permutation(len(data))[:n_per_city]
            out[c] = [data[i] for i in idx]
    return out
