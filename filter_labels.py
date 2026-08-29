"""
filter_labels.py -- text-informativeness filter for the benchmark.
Removes rows whose text carries no learnable signal (call-center shorthand, stubs, test
entries) which the error analysis identified as unclassifiable-from-text. Produces
data/harmonized_filtered.csv and reports drop rates + samples of what is removed.
"""
import csv, os, re, sys
from collections import Counter, defaultdict
csv.field_size_limit(10**7)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SHORTHAND = re.compile(
    r"^(b/?u|o\.?w\.?|n/?a|na|see (below|above|note)|same|same as above|back again|"
    r"no sticker|test\d*|test|tbd|none|\.+|-+|\?+|ok|done|dup(licate)?|per .{0,12}|lowes|"
    r"call(er)?|follow ?up|fyi)\W*$", re.IGNORECASE)

def alpha_tokens(t):
    return [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", t) if len(w) >= 2]

def is_informative(text, min_chars=12, min_tokens=3):
    t = (text or "").strip()
    if len(t) < min_chars:
        return False
    if SHORTHAND.match(t):
        return False
    if len(alpha_tokens(t)) < min_tokens:
        return False
    return True

def main():
    src = os.path.join(DATA, "harmonized.csv")
    out = os.path.join(DATA, "harmonized_filtered.csv")
    kept = defaultdict(int); dropped = defaultdict(int); drop_examples = []
    with open(src, encoding="utf-8") as f, open(out, "w", newline="", encoding="utf-8") as g:
        r = csv.DictReader(f); w = csv.writer(g); w.writerow(["city", "label", "text"])
        for row in r:
            c, lab, txt = row["city"], row["label"], row["text"]
            if is_informative(txt):
                w.writerow([c, lab, txt]); kept[c] += 1
            else:
                dropped[c] += 1
                if len(drop_examples) < 20:
                    drop_examples.append((c, txt[:50]))
    print(f"{'city':14s}{'kept':>8s}{'dropped':>9s}{'drop%':>7s}")
    tk = td = 0
    for c in sorted(kept, key=lambda x: -(kept[x] + dropped[x])):
        k, d = kept[c], dropped[c]; tk += k; td += d
        print(f"{c:14s}{k:8d}{d:9d}{100*d/(k+d):6.1f}%")
    print(f"{'TOTAL':14s}{tk:8d}{td:9d}{100*td/(tk+td):6.1f}%")
    print("\nsamples of DROPPED text (should be junk/shorthand):")
    for c, t in drop_examples:
        print(f"  [{c}] {t!r}")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
