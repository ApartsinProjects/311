"""E2 judge-selection study: per-vendor noise, mean set size, discriminant margin, and pairwise
cross-vendor agreement, over whichever acceptable_sets_*.json are present. Re-run as judges land."""
import json, os, itertools, numpy as np
from eval_common import load_split

sp = load_split(); cities = list(sp["test"])
gold = [y for c in cities for _, y in sp["test"][c]]
LABELS = sorted(set(gold)); n = len(gold)
PRED = "results/preds"
JUDGES = [("gpt-4o-mini (OpenAI)", "acceptable_sets.json"),
          ("gemini-2.5-flash (Google)", "acceptable_sets_gemini25flash.json"),
          ("glm-5.3-flash (Zhipu)", "acceptable_sets_glm.json"),
          ("mistral-small (Mistral)", "acceptable_sets_mistral.json"),
          ("deepseek-chat (DeepSeek)", "acceptable_sets_deepseek.json"),
          ("command-r (Cohere)", "acceptable_sets_cohere.json")]


def load_sets(fn):
    d = json.load(open(os.path.join(PRED, fn), encoding="utf-8"))
    return [set(s) - {"UNPARSED", "ERR"} for c in cities for s in d[c]]


present = [(name, load_sets(fn)) for name, fn in JUDGES if os.path.exists(os.path.join(PRED, fn))]
print(f"judges present: {len(present)}/{len(JUDGES)}\n")
print(f"{'judge':28s}{'noise%':>8s}{'meanset':>9s}{'rand_acc':>9s}{'margin':>8s}{'err%':>7s}")
stats = {}
for name, S in present:
    sizes = [len(s) for s in S]
    empt = sum(1 for s in S if len(s) == 0)
    nonempt = [x for x in sizes if x > 0] or [0]
    noise = np.mean([gold[i] not in S[i] for i in range(n)])
    rand = np.mean(nonempt) / len(LABELS)          # P(random label accepted)
    margin = (1 - noise) - rand                      # discriminant margin
    stats[name] = S
    print(f"{name:28s}{noise*100:8.1f}{np.mean(nonempt):9.3f}{rand:9.3f}{margin:8.3f}{empt/n*100:7.1f}")

print("\npairwise cross-vendor agreement (exact-set match | mean Jaccard):")
def jac(a, b): return len(a & b) / len(a | b) if (a or b) else 1.0
names = [nm for nm, _ in present]
for a, b in itertools.combinations(names, 2):
    Sa, Sb = stats[a], stats[b]
    ex = np.mean([Sa[i] == Sb[i] for i in range(n)])
    jm = np.mean([jac(Sa[i], Sb[i]) for i in range(n)])
    print(f"  {a.split('(')[0].strip():16s} vs {b.split('(')[0].strip():16s}  exact={ex:.2f}  Jaccard={jm:.2f}")

if len(present) >= 3:
    # majority consensus: label accepted by >half the judges
    m = len(present)
    cons_noise = np.mean([sum(gold[i] in S[i] for _, S in present) <= m/2 for i in range(n)])
    print(f"\nmajority-of-{m} consensus: gold rejected by majority on {cons_noise*100:.1f}% of rows")
