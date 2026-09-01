"""External validation: does the CASE acceptable-set judge's set SIZE track HUMAN disagreement?
Runs the acceptable-set judge on ChaosNLI-MNLI (100 human labels/example) and correlates
judge |set| with human label entropy. gpt-4o-mini via OpenAI (reliable, cheap)."""
import json, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import spearmanr
from openai_batch import client
from sklearn.metrics import f1_score

REL = ["entailment", "neutral", "contradiction"]
SYS = ("You are validating natural-language-inference annotations. Given a premise and a hypothesis, "
       "list EVERY relation a reasonable annotator could assign: entailment, neutral, or contradiction. "
       "Usually one, sometimes two or three for genuinely ambiguous pairs. Judge only from the text.")


def prompt(p, h):
    return (f"Premise: \"{p}\"\nHypothesis: \"{h}\"\n\n"
            f"Which of these relations could a reasonable annotator assign: entailment, neutral, contradiction?\n"
            f"Reply with ONLY the acceptable relation names, comma-separated.")


def parse(o):
    low = (o or "").lower()
    return [r for r in REL if r in low] or ["UNPARSED"]


def judge(row):
    for a in range(4):
        try:
            r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=25,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": prompt(row["premise"], row["hypothesis"])}])
            return set(parse(r.choices[0].message.content)) - {"UNPARSED"}
        except Exception:
            import time; time.sleep(2 * (a + 1))
    return set()


def main():
    rows = json.load(open("data/chaosnli_mnli_sample.json", encoding="utf-8"))
    sets = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(judge, rows[i]): i for i in range(len(rows))}
        for f in as_completed(futs):
            sets[futs[f]] = f.result()
    size = np.array([len(s) for s in sets])
    ent = np.array([r["entropy"] for r in rows])
    human_multi = (ent > 0.5).astype(int)                 # humans clearly split
    judge_multi = (size > 1).astype(int)
    rho, p = spearmanr(size, ent)
    print(f"judge |set| vs human entropy: Spearman rho={rho:.3f} (p={p:.2e})  n={len(rows)}")
    print(f"mean judge |set|: {size.mean():.2f}; distribution {dict(zip(*np.unique(size, return_counts=True)))}")
    # binary agreement: does judge flag ambiguity where humans do
    tp = np.sum((judge_multi == 1) & (human_multi == 1));
    print(f"human-split rows (entropy>0.5): {human_multi.sum()}/{len(rows)}; judge-multi rows: {judge_multi.sum()}")
    print(f"agreement on ambiguous-or-not: {np.mean(judge_multi==human_multi):.2f}")
    # judge majority vs human majority (map e/n/c)
    m = {"e": "entailment", "n": "neutral", "c": "contradiction"}
    jmaj = []
    for i, s in enumerate(sets):
        lc = rows[i]["label_count"]; hm = m[["e","n","c"][int(np.argmax(lc))]]
        jmaj.append(1 if (s and hm in s) else 0)
    print(f"human-majority label in judge's acceptable set: {np.mean(jmaj):.2f}")
    # discriminant floor: a random relation accepted?
    rng = np.random.RandomState(0)
    rand_acc = np.mean([REL[rng.randint(3)] in sets[i] for i in range(len(rows))])
    print(f"discriminant floor: random relation accepted {rand_acc:.2f} (mean|set|/3 = {size.mean()/3:.2f})")
    json.dump({"rho_setsize_entropy": float(rho), "p": float(p), "mean_set": float(size.mean()),
               "ambig_agreement": float(np.mean(judge_multi==human_multi)),
               "human_maj_in_set": float(np.mean(jmaj)), "n": len(rows)},
              open("results/chaosnli_validation.json", "w"), indent=2)
    print("wrote results/chaosnli_validation.json")


if __name__ == "__main__":
    main()
