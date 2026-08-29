"""
scrub_pii.py -- remove phone numbers and email addresses from the RELEASED text files.
Street addresses are retained: they are intrinsic to a 311 request and are public record.
Replaces phones with [PHONE] and emails with [EMAIL] in place, in the files that ship in the repo.
"""
import csv, re, os, sys
csv.field_size_limit(10**7)
HERE = os.path.dirname(os.path.abspath(__file__))

PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

FILES = ["data/harmonized.csv", "data/harmonized_filtered.csv", "data/eval_split.csv"]


def scrub(t):
    t = EMAIL.sub("[EMAIL]", t)
    t = PHONE.sub("[PHONE]", t)
    return t


def main():
    total_p = total_e = 0
    for rel in FILES:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            print(f"skip {rel} (missing)"); continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        cols = rows[0].keys() if rows else []
        np = ne = 0
        for r in rows:
            t = r["text"]
            ne += len(EMAIL.findall(t)); np += len(PHONE.findall(t))
            r["text"] = scrub(t)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cols)); w.writeheader(); w.writerows(rows)
        total_p += np; total_e += ne
        print(f"{rel}: {np} phones, {ne} emails scrubbed ({len(rows)} rows)")
    print(f"TOTAL: {total_p} phones, {total_e} emails removed")


if __name__ == "__main__":
    main()
