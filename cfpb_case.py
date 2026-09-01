"""CFPB second-domain validation of the CASE protocol: does the same pipeline (blind acceptable-set
judges + discriminant floors + noise rate) work on consumer-complaint Product labels?"""
import json, ssl, urllib.request, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client as OAI
from or_batch import _key

ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
ORKEY = _key(r"E:\Projects\.env.all")
# short code -> canonical CFPB product
PROD = {
 "credit_reporting": "Credit reporting or other personal consumer reports",
 "debt_collection": "Debt collection",
 "bank_account": "Checking or savings account",
 "vehicle_loan": "Vehicle loan or lease",
 "credit_card": "Credit card",
 "mortgage": "Mortgage",
 "personal_loan": "Payday loan, title loan, personal loan, or advance loan",
 "money_transfer": "Money transfer, virtual currency, or money service",
 "student_loan": "Student loan",
}
CODES = list(PROD); CANON2CODE = {v: k for k, v in PROD.items()}
GLOSS = {"credit_reporting": "errors/disputes on a credit report or consumer report, credit bureaus",
 "debt_collection": "a debt collector's practices, collection calls, validation", "bank_account": "checking/savings accounts, deposits, fees, overdraft",
 "vehicle_loan": "auto loan or lease", "credit_card": "credit or prepaid card", "mortgage": "home mortgage or loan",
 "personal_loan": "payday/title/personal/advance loan", "money_transfer": "money transfer, virtual currency, wallet",
 "student_loan": "student loan"}
SYS = ("You are validating how consumer-financial complaints are categorized by product. For a complaint, list "
       "EVERY product category that is a reasonable classification of what the complaint describes (usually one, "
       "sometimes two for complaints that span products). Judge only from the text.")


def prompt(t):
    cats = "\n".join(f"- {k}: {GLOSS[k]}" for k in CODES)
    return f"Product categories:\n{cats}\n\nComplaint:\n\"\"\"{t[:1400]}\"\"\"\n\nReply with ONLY the category codes, comma-separated."


def parse(o):
    low = (o or "").lower(); return set(k for k in CODES if k in low)


def judge_oai(t):
    for a in range(4):
        try:
            r = OAI.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=30,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": prompt(t)}])
            return parse(r.choices[0].message.content)
        except Exception:
            import time; time.sleep(2*(a+1))
    return set()


def judge_or(t, model="google/gemini-2.5-flash"):
    body = {"model": model, "temperature": 0, "max_tokens": 30, "reasoning": {"enabled": False},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt(t)}]}
    for a in range(4):
        try:
            r = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                method="POST", headers={"Authorization": f"Bearer {ORKEY}", "Content-Type": "application/json"})
            o = json.loads(urllib.request.urlopen(r, timeout=90, context=ctx).read())
            return parse(o["choices"][0]["message"]["content"])
        except Exception:
            import time; time.sleep(2*(a+1))
    return set()


def run(fn, rows):
    out = [None]*len(rows)
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(fn, rows[i]["text"]): i for i in range(len(rows))}
        for f in as_completed(futs): out[futs[f]] = f.result()
    return out


def report(name, rows, sets):
    gold = [CANON2CODE.get(r["product"]) for r in rows]
    ok = [i for i in range(len(rows)) if gold[i]]
    noise = np.mean([gold[i] not in sets[i] for i in ok])
    sizes = [len(sets[i]) for i in ok]; empt = sum(1 for i in ok if len(sets[i]) == 0)
    rng = np.random.RandomState(0)
    rand = np.mean([CODES[rng.randint(len(CODES))] in sets[i] for i in ok])
    print(f"{name:16s} noise={noise:.3f} mean_set={np.mean(sizes):.2f} rand_acc={rand:.3f} margin={(1-noise)-rand:.3f} empty={empt}")
    return sets, gold, ok


def main():
    rows = json.load(open("data/cfpb_sample.json", encoding="utf-8"))
    print(f"CFPB n={len(rows)}  products={len(set(r['product'] for r in rows))}")
    so = run(judge_oai, rows); sg = run(judge_or, rows)
    print("=== CASE floors on CFPB ==="); report("gpt-4o-mini", rows, so); report("gemini-2.5", rows, sg)
    # cross-vendor agreement + multi-label rate
    gold = [CANON2CODE.get(r["product"]) for r in rows]; ok = [i for i in range(len(rows)) if gold[i]]
    ex = np.mean([so[i] == sg[i] for i in ok]); jac = np.mean([len(so[i]&sg[i])/len(so[i]|sg[i]) if (so[i]|sg[i]) else 1 for i in ok])
    multi = np.mean([len(so[i]) > 1 for i in ok])
    cons_noise = np.mean([gold[i] not in so[i] and gold[i] not in sg[i] for i in ok])
    print(f"cross-vendor agreement exact={ex:.2f} Jaccard={jac:.2f}; multi-label(gpt)={multi:.2f}; two-judge-consensus noise={cons_noise:.3f}")
    json.dump({"n": len(ok), "consensus_noise": float(cons_noise), "agreement_exact": float(ex),
               "multi_label_frac": float(multi)}, open("results/cfpb_case.json", "w"), indent=2)
    print("wrote results/cfpb_case.json")


if __name__ == "__main__":
    main()
