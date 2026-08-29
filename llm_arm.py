"""
llm_arm.py -- LLM zero-shot arm for the multi-city 311 benchmark.

An LLM is given the shared 14-class taxonomy (with glosses) IN THE PROMPT and asked to
classify each citizen complaint into exactly one class. The LLM sees NO city-specific
training data, so this is inherently a cross-jurisdiction setting -- the honest comparison
is:  LLM zero-shot  vs  fine-tuned LEAVE-ONE-CITY-OUT (both transfer to an unseen city).

Thesis under test: giving the taxonomy in-context lets an LLM transfer across cities far
better than a model whose label space is frozen to the training cities.

Usage:
  python llm_arm.py --n-per-city 60 --model gpt-4o-mini      # quick signal
  python llm_arm.py --n-per-city 200 --model gpt-4o-mini     # fuller
"""
import argparse, json, os, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score, accuracy_score
from bench_common import load_harmonized, sample_test, CONTENT_CLASSES, LABELS
from openai import OpenAI


def load_key_from_env_file(path, name):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        return None
    return None


def make_client(env_file):
    # Prefer OpenRouter (OpenAI-compatible gateway); fall back to OpenAI.
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and env_file:
        key = load_key_from_env_file(env_file, "OPENROUTER_API_KEY")
    if key:
        return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1"), "openrouter"
    return OpenAI(), "openai"

SYS = "You are an expert municipal 311 dispatcher. Classify each citizen service request into exactly one category."

def build_prompt(text):
    cat_lines = "\n".join(f"- {k}: {v}" for k, v in CONTENT_CLASSES.items())
    return (f"Categories (choose exactly one, respond with the category name verbatim):\n{cat_lines}\n\n"
            f"Service request text:\n\"\"\"{text[:800]}\"\"\"\n\n"
            f"Answer with ONLY the single best category name from the list above.")

NORM = {l.lower(): l for l in LABELS}

def parse_label(out):
    s = (out or "").strip()
    if s in LABELS:
        return s
    low = s.lower()
    if low in NORM:
        return NORM[low]
    for l in LABELS:                       # substring / fuzzy fallback
        if l.lower() in low or low in l.lower():
            return l
    toks = re.findall(r"[A-Za-z_]+", s)
    for l in LABELS:
        if any(t.lower() in l.lower() for t in toks if len(t) > 4):
            return l
    return "UNPARSED"

def classify_one(client, text, model):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=15,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": build_prompt(text)}])
            return parse_label(r.choices[0].message.content)
        except Exception as e:
            if attempt == 3:
                return f"ERR:{type(e).__name__}"
            time.sleep(2 * (attempt + 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-city", type=int, default=60)
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--env-file", default=r"E:\Projects\.env.all")
    ap.add_argument("--filter", action="store_true", help="drop uninformative/shorthand text")
    ap.add_argument("--out", default="results_llm.json")
    args = ap.parse_args()

    client, provider = make_client(args.env_file)
    by_city = load_harmonized(cap=0, drop_other=True, informative_only=args.filter)
    test = sample_test(by_city, args.n_per_city, seed=0)
    print(f"LLM arm: provider={provider}  model={args.model}  n/city={args.n_per_city}\n")

    results = {}
    for city, data in test.items():
        texts = [t for t, _ in data]; gold = [y for _, y in data]
        preds = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(classify_one, client, texts[i], args.model): i for i in range(len(texts))}
            for f in as_completed(futs):
                preds[futs[f]] = f.result()
        # score only rows that parsed to a valid label; count unparsed/err as wrong
        valid = [(g, p) for g, p in zip(gold, preds)]
        y_true = [g for g, p in valid]
        y_pred = [p if p in LABELS else "UNPARSED" for g, p in valid]
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)  # over classes present (matches all arms)
        acc = accuracy_score(y_true, y_pred)
        nbad = sum(1 for p in y_pred if p == "UNPARSED")
        results[city] = dict(n=len(y_true), macroF1=round(macro, 3), acc=round(acc, 3),
                             unparsed=nbad, n_gold_classes=len(set(y_true)))
        print(f"  {city:14s} n={len(y_true):4d}  macroF1={macro:.3f}  acc={acc:.3f}  "
              f"unparsed={nbad}  goldClasses={len(set(y_true))}")

    ms = sum(r["macroF1"] for r in results.values()) / len(results)
    ma = sum(r["acc"] for r in results.values()) / len(results)
    print(f"\nMEAN  macroF1={ms:.3f}  acc={ma:.3f}  (model={args.model})")
    json.dump({"model": args.model, "n_per_city": args.n_per_city,
               "per_city": results, "mean_macroF1": round(ms, 3), "mean_acc": round(ma, 3)},
              open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    try:
        from results_log import save_result
        save_result("llm_zeroshot",
                    {"mean_macroF1": round(ms, 4), "mean_acc": round(ma, 4),
                     "per_city": {c: r["macroF1"] for c, r in results.items()}},
                    config={"provider": provider, "model": args.model, "n_per_city": args.n_per_city,
                            "filter": args.filter},
                    note="zero-shot, taxonomy-in-prompt" + (" [filtered]" if args.filter else ""))
    except Exception as e:
        print(f"[results_log] skipped: {e}")

if __name__ == "__main__":
    main()
