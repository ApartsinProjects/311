"""FULL BENCHMARK: 9 methods x 3 labeling budgets (10%/50%/100% of 2000 = 200/1000/2000),
on identical items, via the OpenAI BATCH API (50% cheaper) with CACHE-FRIENDLY prompt layout.

Design for efficiency
---------------------
* BATCH API for every classification pass (submit all, collect once) -- 50% off.
* PROMPT CACHING: every prompt is built as [STATIC PREFIX][VARIABLE SUFFIX]. The static prefix
  (system + category book / mined rules / demo block) is byte-identical across all items of a run,
  so it is cached after the first call. The per-item request text is always LAST.
  NOTE: RAG is the one method whose prefix CANNOT be cached (its demos change per query) -- that is
  an inherent cost property of retrieval and is reported as a result, not a flaw in the harness.
* zero-shot is budget-independent -> run ONCE and reuse across all three budgets.
* mined artifacts are mined once per budget and cached to disk.
* fine-tuned is local (sklearn) -> free, no API.

Usage
-----
  python bench.py prepare    # build budgets, mine artifacts per budget (few gpt-4.1 calls), cache
  python bench.py submit     # build + submit ALL classification batches
  python bench.py collect    # collect, score, write results/bench_table.{json,md}
  python bench.py cost       # print request/token/cost estimate without calling anything
"""
import sys, json, re, os
import numpy as np
from collections import defaultdict, Counter
import semclf
from semclf import TASKS, CLF_MODEL, MINE_MODEL, TRUNC, _dline
import flows

BUDGETS = [200, 1000, 2000]          # 10% / 50% / 100% of the 2k reference budget
N_TEST = None                        # None = use the FULL fixed 1500-row test set (all budgets/methods)
ART = "results/bench_art_{b}.json"   # mined artifact per budget
PLAN = "results/bench_plan.json"
TAG = "bench_{b}_{m}"


# ---------------- shared helpers ----------------
def fixed_test(T, b_for_dupflag=2000):
    """THE test set: full 1500 held-out rows, IDENTICAL for every budget and every method.
    Primary metric = OVERALL accuracy on this natural distribution. `is_dup` is a diagnostic flag
    (computed against the reference 2k budget so the breakdown is comparable across budgets)."""
    ref = set(norm(r["text"]) for r in T.pool[:b_for_dupflag])
    rows = [{**r, "is_dup": norm(r["text"]) in ref} for r in (T.test + T.test_dup)]
    assert len(rows) == 1500, f"expected fixed 1500-row test, got {len(rows)}"
    return rows


def subsample(T, n=N_TEST, seed=0):
    rows = fixed_test(T)
    if n is None: return rows                      # default: the whole fixed test set
    idx = np.random.RandomState(seed).permutation(len(rows))[:n]
    return [rows[i] for i in idx]


def budget_of(T, b):
    return semclf.stratified_budget(T.pool, b)   # proportional, all classes covered


def norm(t): return semclf.norm(t)


CONV = flows.CONV


def static_then_item(system, static_block, T, text):
    """Cache-friendly: static_block is identical across items; the item text goes LAST."""
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{static_block}\n\n{T.item.capitalize()}: {text[:TRUNC]}\n{T.label.capitalize()}:"}]


# ---------------- per-method request builders (return list of (custom_id, body)) ----------------
def build_zero_shot(T, items):
    sysm = (f"Classify the {T.item} into its single best {T.label}. Reply with ONLY one {T.label} name copied verbatim.")
    static = f"{T.label.capitalize()} options:\n" + "\n".join(f"- {l}" for l in T.LBL)
    return [(f"zs:{i}", {"messages": static_then_item(sysm, static, T, r["text"]),
                         "temperature": 0, "max_tokens": 24}) for i, r in enumerate(items)]


def build_rag(T, items, budget, k=12):
    """NOT cacheable: demos differ per query (reported as a cost property of RAG)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=40000)
    Xb = vec.fit_transform([r["text"] for r in budget]); Xt = vec.transform([r["text"] for r in items])
    sims = (Xt @ Xb.T).toarray(); bn = [norm(r["text"]) for r in budget]
    sysm = f"Classify the {T.item} into its single best {T.label} using the labeled examples. Reply with ONLY one {T.label} name."
    out = []
    for i, r in enumerate(items):
        qn = norm(r["text"]); order = np.argsort(-sims[i])
        nn = [j for j in order if bn[j] != qn][:k]
        demos = "Examples:\n" + "\n".join(f"- \"{budget[j]['text'][:120]}\" -> {budget[j]['label']}" for j in nn)
        out.append((f"rag:{i}", {"messages": static_then_item(sysm, demos, T, r["text"]),
                                 "temperature": 0, "max_tokens": 24}))
    return out


def build_kshot(T, items, budget, k=2):
    by = defaultdict(list)
    for r in budget: by[r["label"]].append(r["text"])
    def templates(ts, kk):
        c = Counter(norm(x) for x in ts); reps = {}
        for x in ts:
            n = norm(x)
            if n not in reps: reps[n] = x
        return [reps[n] for n, _ in c.most_common(kk)]
    block = "Examples:\n" + "\n".join(f"- \"{ex[:100]}\" -> {l}" for l in T.LBL for ex in templates(by[l], k))
    sysm = f"Classify the {T.item} into its single best {T.label} using the labeled examples. Reply with ONLY one {T.label} name."
    return [(f"ks:{i}", {"messages": static_then_item(sysm, block, T, r["text"]),
                         "temperature": 0, "max_tokens": 24}) for i, r in enumerate(items)]


def build_flat(T, items, D):
    sysm = (f"Classify the {T.item} into its single best {T.label} using the descriptions "
            f"(what belongs, and what does NOT). {CONV} Reply with ONLY one {T.label} name.")
    static = f"{T.label.capitalize()} descriptions:\n" + "\n".join(_dline(T, c, D) for c in T.LBL)
    return [(f"fl:{i}", {"messages": static_then_item(sysm, static, T, r["text"]),
                         "temperature": 0, "max_tokens": 24}) for i, r in enumerate(items)]


def build_ova(T, items, D):
    """N binary calls/item. Static prefix per CLASS is cached across all items of that class."""
    out = []
    for ci, c in enumerate(T.LBL):
        sysm = f"Answer YES or NO only. {CONV}"
        static = f"{T.label.capitalize()} '{c}': {_dline(T, c, D)}"
        for i, r in enumerate(items):
            out.append((f"ova:{i}:{ci}", {"messages": [
                {"role": "system", "content": sysm},
                {"role": "user", "content": f"{static}\n\n{T.item.capitalize()}: {r['text'][:TRUNC]}\n\n"
                 f"Is this {T.item} filed under '{c}'? YES or NO:"}], "temperature": 0, "max_tokens": 4}))
    return out


def build_ovo_shortlist(T, items, D, K=3):
    sysm = f"List the {K} most likely {T.label} NUMBERS, best first, comma-separated. {CONV}"
    static = f"{T.label.capitalize()}s:\n" + "\n".join(f"{i+1}. {_dline(T, c, D)}" for i, c in enumerate(T.LBL))
    return [(f"ovosl:{i}", {"messages": static_then_item(sysm, static, T, r["text"]),
                            "temperature": 0, "max_tokens": 14}) for i, r in enumerate(items)]


def build_hier_stage1(T, items, groups):
    gn = list(groups)
    sysm = f"Choose the TWO most likely groups. Reply with two numbers, comma-separated. {CONV}"
    static = "Groups:\n" + "\n".join(f"{i+1}. {g}: " + "; ".join(groups[g][:6]) for i, g in enumerate(gn))
    return [(f"h1:{i}", {"messages": static_then_item(sysm, static, T, r["text"]),
                         "temperature": 0, "max_tokens": 12}) for i, r in enumerate(items)]


def build_ecoc(T, items, D, C):
    """M binary calls/item; static prefix per BIT is cached across all items."""
    out = []
    for j in range(C.shape[1]):
        s0 = [T.LBL[i] for i in range(len(T.LBL)) if C[i, j] == 0]
        s1 = [T.LBL[i] for i in range(len(T.LBL)) if C[i, j] == 1]
        d0 = "; ".join(_dline(T, c, D)[:90] for c in s0[:6]); d1 = "; ".join(_dline(T, c, D)[:90] for c in s1[:6])
        sysm = f"Decide which GROUP the {T.item} belongs to. Reply ONLY 1 or 2. {CONV}"
        static = f"GROUP 1 ({', '.join(s0)}): {d0}\n\nGROUP 2 ({', '.join(s1)}): {d1}"
        for i, r in enumerate(items):
            out.append((f"ec:{i}:{j}", {"messages": [
                {"role": "system", "content": sysm},
                {"role": "user", "content": f"{static}\n\n{T.item.capitalize()}: {r['text'][:TRUNC]}\nGroup:"}],
                "temperature": 0, "max_tokens": 4}))
    return out


# ---------------- prepare: mine one artifact per budget ----------------
def prepare():
    T = TASKS["bloom"]; items = subsample(T)
    json.dump({"n_test": len(items)}, open(PLAN, "w"))
    for b in BUDGETS:
        p = ART.format(b=b)
        if os.path.exists(p): print(f"[prepare] budget={b}: artifact cached"); continue
        bud = budget_of(T, b)
        Tb = TASKS["bloom"]; Tb.budget = bud                       # mine from THIS budget only
        Tb.by = defaultdict(list)
        for r in bud: Tb.by[r["label"]].append(r["text"])
        n_mine = max(int(0.7 * b), 20); n_val = max(b - n_mine, 10)
        mine, val = bud[:n_mine], bud[n_mine:n_mine + n_val]
        print(f"[prepare] budget={b}: mining (mine={len(mine)} val={len(val)}) ...")
        flows.DCACHE = p                                            # cache to per-budget file
        D = flows.mine_artifact(Tb, mine, val)
        groups = flows.mine_groups(Tb, D)
        C = flows.make_code(Tb, M=12)
        json.dump({"D": D, "groups": groups, "code": C.tolist()}, open(p, "w"), indent=2, ensure_ascii=False)
        print(f"[prepare] budget={b}: artifact written -> {p}")


# ---------------- cost estimate ----------------
def cost():
    """Per-method call counts + realistic per-method prompt sizes. Batch gpt-4o-mini:
    $0.075/1M in, $0.30/1M out. Prompt caching needs a >=1024-token static prefix, so it applies
    only to the long-prefix methods -- short binary prompts (OvA/ECOC/duels) are below threshold."""
    T = TASKS["bloom"]; n = len(fixed_test(T)); nL = len(T.LBL); M = 12; K = 3
    # (calls per budget, approx input tokens/call, cacheable static prefix?)
    spec = {
        "rag":   (n,                      560,  False),   # demos differ per query -> NEVER cacheable
        "kshot": (n,                     1200,  True),
        "flat":  (n,                      950,  True),
        "ova":   (n * nL + n,             190,  False),   # short prompt, under cache threshold
        "ovo":   (n + n * (K*(K-1)//2),   350,  False),   # shortlist cacheable, duels not
        "hier":  (n + n,                  360,  True),
        "ecoc":  (n * M,                  460,  False),
    }
    tot_calls = sum(v[0] for v in spec.values())
    tot_in = sum(v[0] * v[1] for v in spec.values())
    zs_calls, zs_in = n, 250
    total_calls = zs_calls + tot_calls * len(BUDGETS)
    total_in = zs_in * n / n * zs_calls + tot_in * len(BUDGETS)
    total_out = total_calls * 6
    cin = total_in / 1e6 * 0.075; cout = total_out / 1e6 * 0.30
    print(f"FIXED test n={n} (same for every budget & method)  labels={nL}  budgets={BUDGETS}")
    print(f"{'method':8s}{'calls/budget':>14s}{'tok/call':>10s}{'cacheable':>11s}")
    for k, (c, t, ca) in spec.items():
        print(f"  {k:6s}{c:>14d}{t:>10d}{('yes' if ca else 'no'):>11s}")
    print(f"\nper-budget calls={tot_calls}  x{len(BUDGETS)} budgets + zero-shot({zs_calls}, budget-independent)")
    print(f"TOTAL calls={total_calls}   input~{total_in/1e6:.1f}M tok   output~{total_out/1e6:.2f}M tok")
    print(f"EST BATCH COST: ${cin+cout:.2f}   (standard API ~${(cin+cout)*2:.2f})")
    print(f"note: OvA ({spec['ova'][0]} calls) + ECOC ({spec['ecoc'][0]}) = "
          f"{(spec['ova'][0]+spec['ecoc'][0])/tot_calls*100:.0f}% of all calls -- the expensive flows")
    print(f"mining (one-time, gpt-4.1): ~{len(BUDGETS)*50} calls total across the 3 budgets")
    print(f"fine-tuned: local sklearn, $0")


CORE = ["rag", "kshot", "flat"]          # swept over all 3 budgets
FLOWS_2K = ["ova", "ovo", "hier", "ecoc"]  # flow ablation at the 2k budget only (cost control)


def _art(b):
    a = json.load(open(ART.format(b=b), encoding="utf-8"))
    return a["D"], a["groups"], np.array(a["code"])


def _tag(b, m): return TAG.format(b=b, m=m)


def submit():
    """Round 1: all single-stage passes. (Round 2 = OvO duels + hier stage-2, built in collect.)"""
    from openai_batch import submit_chat_batch
    T = TASKS["bloom"]; items = subsample(T)
    json.dump({"n_test": len(items), "budgets": BUDGETS, "core": CORE, "flows2k": FLOWS_2K},
              open(PLAN, "w"), indent=2)
    subs = []
    # zero-shot: budget-independent -> once
    subs.append(("zs", 0, build_zero_shot(T, items)))
    for b in BUDGETS:
        bud = budget_of(T, b); D, groups, C = _art(b)
        Tb = TASKS["bloom"]; Tb.budget = bud
        Tb.by = defaultdict(list)
        for r in bud: Tb.by[r["label"]].append(r["text"])
        subs.append(("rag", b, build_rag(Tb, items, bud)))
        subs.append(("kshot", b, build_kshot(Tb, items, bud)))
        subs.append(("flat", b, build_flat(Tb, items, D)))
        if b == 2000:
            subs.append(("ova", b, build_ova(Tb, items, D)))
            subs.append(("ecoc", b, build_ecoc(Tb, items, D, C)))
            subs.append(("ovosl", b, build_ovo_shortlist(Tb, items, D)))
            subs.append(("h1", b, build_hier_stage1(Tb, items, groups)))
    tot = 0
    for m, b, reqs in subs:
        # OpenAI batch cap is 50k requests -> chunk
        for ci in range(0, len(reqs), 45000):
            chunk = reqs[ci:ci+45000]
            tag = _tag(b, m) + (f"_c{ci//45000}" if len(reqs) > 45000 else "")
            submit_chat_batch(CLF_MODEL, chunk, tag=tag, verbose=False)
            print(f"  submitted {tag}: {len(chunk)} requests"); tot += len(chunk)
    print(f"round-1 total submitted: {tot} requests")


def _get(tag, n_expect=None):
    from openai_batch import collect_chat_batch
    parts = {}
    for suf in ["", "_c0", "_c1", "_c2"]:
        try:
            r = collect_chat_batch(tag=tag + suf, verbose=False)
        except Exception:
            r = None
        if r: parts.update(r)
        if suf == "" and parts: break
    return parts or None


def collect():
    """Collect round-1, run round-2 (OvO duels + hier stage-2) live, score everything, write table."""
    from openai_batch import submit_chat_batch
    T = TASKS["bloom"]; items = subsample(T)
    gold = [r["label"] for r in items]; is_dup = [r["is_dup"] for r in items]
    rows = {}

    def add(name, preds, b):
        acc, ci, unp = score(T, preds, gold)
        nd = [i for i in range(len(items)) if not is_dup[i]]; dd = [i for i in range(len(items)) if is_dup[i]]
        nov = float(np.mean([preds[i] == gold[i] for i in nd])) if nd else float("nan")
        dup = float(np.mean([preds[i] == gold[i] for i in dd])) if dd else float("nan")
        rows[f"{name}@{b}"] = {"method": name, "budget": b, "overall": acc, "ci": ci,
                               "novel": nov, "dup": dup, "unparsed": unp}
        print(f"  {name:12s} b={b:5d} overall={acc:.4f} CI=({ci[0]:.3f},{ci[1]:.3f}) "
              f"novel={nov:.3f} dup={dup:.3f} unp={unp:.3f}")

    zs = _get(_tag(0, "zs"))
    if zs is None: print("zero-shot batch not ready"); return
    for b in BUDGETS:
        add("zero_shot", [T.parse(zs.get(f"zs:{i}", "")) for i in range(len(items))], b)
    for b in BUDGETS:
        for m, pre in [("rag", "rag"), ("kshot", "ks"), ("flat", "fl")]:
            r = _get(_tag(b, m))
            if r is None: print(f"  {m}@{b} not ready"); continue
            add(m, [T.parse(r.get(f"{pre}:{i}", "")) for i in range(len(items))], b)
    # --- flow ablation at 2k ---
    b = 2000; D, groups, C = _art(b)
    ova = _get(_tag(b, "ova"))
    if ova:
        yes = defaultdict(list)
        for i in range(len(items)):
            for ci_, c in enumerate(T.LBL):
                if (ova.get(f"ova:{i}:{ci_}", "") or "").strip().upper().startswith("Y"): yes[i].append(c)
        preds = [(yes[i][0] if len(yes.get(i, [])) == 1 else (yes[i][0] if yes.get(i) else T.LBL[0]))
                 for i in range(len(items))]
        add("ova", preds, b)
    ec = _get(_tag(b, "ecoc"))
    if ec:
        M = C.shape[1]; B = np.zeros((len(items), M), dtype=int)
        for i in range(len(items)):
            for j in range(M):
                o = ec.get(f"ec:{i}:{j}", ""); mm = re.search(r"[12]", o or "")
                B[i, j] = 0 if (mm and mm.group(0) == "1") else 1
        add("ecoc", [T.LBL[int(np.argmin((C != B[i]).sum(1)))] for i in range(len(items))], b)
    # OvO + hierarchical need a round-2 (dependent on round-1); run them live (small)
    sl = _get(_tag(b, "ovosl")); h1 = _get(_tag(b, "h1"))
    if sl:
        import flows
        Tb = TASKS["bloom"]; by = defaultdict(list)
        for r in budget_of(T, b): by[r["label"]].append(r["text"])
        cands = []
        for i in range(len(items)):
            nums = [int(x) - 1 for x in re.findall(r"\d+", sl.get(f"ovosl:{i}", "") or "")][:3]
            c = [T.LBL[k] for k in nums if 0 <= k < len(T.LBL)]
            cands.append(list(dict.fromkeys(c)) or [T.LBL[0]])
        add("ovo", flows.flow_ovo_from_cands(Tb, [r["text"] for r in items], cands, by), b)
    if h1:
        import flows
        gn = list(groups); cands = []
        for i in range(len(items)):
            nums = [int(x) - 1 for x in re.findall(r"\d+", h1.get(f"h1:{i}", "") or "")][:2]
            gs = [gn[k] for k in nums if 0 <= k < len(gn)] or [gn[0]]
            cs = []
            for g in gs: cs += groups[g]
            cands.append(list(dict.fromkeys(cs)) or list(T.LBL))
        add("hier", flows.flow_stage2_from_cands(TASKS["bloom"], [r["text"] for r in items], cands, D), b)
    # fine-tuned (local, free) at every budget
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    Xte = [r["text"] for r in items]
    for b2 in BUDGETS:
        tr = budget_of(T, b2)
        v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=40000)
        X = v.fit_transform([r["text"] for r in tr])
        clf = LogisticRegression(max_iter=600, C=6.0).fit(X, [r["label"] for r in tr])
        add("fine_tuned", list(clf.predict(v.transform(Xte))), b2)
    json.dump(rows, open("results/bench_table.json", "w"), indent=2)
    print("\nwrote results/bench_table.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cost"
    {"prepare": prepare, "cost": cost, "submit": submit, "collect": collect}.get(cmd, cost)()
