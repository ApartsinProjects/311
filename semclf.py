"""Semantic Multiclass: generic, dataset-agnostic no-train LLM classification framework.
FAIR + bug-fixed version (after 3 code reviews):
  * mining model = gpt-4.1 (one-time), classification model = gpt-4o-mini (per-sample, cheap);
  * equal 2k labeling budget for EVERY method (by built from budget, not full pool);
  * lexical-RAG excludes any demo whose normalized text == the query (no verbatim answer-key leakage);
  * primary eval on the NOVEL test slice (no verbatim duplicate in budget); dup slice reported separately;
  * robust parser (strip, exact, longest-substring, unique-reverse else UNPARSED); UNPARSED reported;
  * desc refinement replaces the weakest clause instead of appending past a silent cap.
Prompts parameterized by (item_noun,label_noun) -- portable to any dataset.

  python semclf.py <task> <m1,m2,...>
"""
import sys, json, re, copy, os, time
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from oaillm import chat_many, _call

CLF_MODEL = "gpt-4o-mini"     # per-sample classification (cheap)
MINE_MODEL = "gpt-4.1"        # one-time rule mining (cheaper for the multi-domain sweep)
MAXPOS, MAXNEG = 5, 4
TRUNC = 500

# ---- full provenance trace of the mining process (every request + response) ----
TRACE_PATH = os.environ.get("SEMCLF_TRACE", "results/mining_trace.jsonl")


def set_trace(path):
    global TRACE_PATH
    TRACE_PATH = path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _trace(phase, **rec):
    """Append one JSON record (prompts, raw response, parsed result, state) for later re-analysis."""
    try:
        os.makedirs(os.path.dirname(TRACE_PATH) or ".", exist_ok=True)
        rec = {"t": time.time(), "phase": phase, **rec}
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _traced_call(phase, messages, model, max_tokens, **meta):
    """LLM call that records the full request and raw response to the trace."""
    out = _call(messages, model=model, max_tokens=max_tokens)
    _trace(phase, model=model, max_tokens=max_tokens,
           system=messages[0]["content"] if messages else "",
           user=messages[-1]["content"] if messages else "",
           response=out, **meta)
    return out


def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()


def stratified_budget(pool, b, seed=0):
    """Sample a labeling budget of size b that PRESERVES class proportions, with a floor of 1 per class
    (so no class is invisible at small budgets). Deterministic. Returns rows in shuffled order."""
    by = defaultdict(list)
    for r in pool: by[r["label"]].append(r)
    classes = sorted(by); n = len(pool)
    rng = np.random.RandomState(seed)
    # proportional allocation, floor 1, then fix the total by largest-remainder on the proportional part
    quota = {}
    for c in classes:
        quota[c] = max(1, int(round(b * len(by[c]) / n)))
    # adjust to exactly b
    while sum(quota.values()) > b:
        c = max(classes, key=lambda c: (quota[c] - 1, len(by[c])))   # trim the largest, never below 1
        if quota[c] <= 1: break
        quota[c] -= 1
    while sum(quota.values()) < b:
        c = max(classes, key=lambda c: len(by[c]) / max(quota[c], 1))
        if quota[c] >= len(by[c]):
            c = max(classes, key=lambda c: len(by[c]) - quota[c])
            if quota[c] >= len(by[c]): break
        quota[c] += 1
    out = []
    for c in classes:
        idx = rng.permutation(len(by[c]))[:min(quota[c], len(by[c]))]
        out += [by[c][i] for i in idx]
    rng.shuffle(out)
    return out


class Task:
    def __init__(self, name, split_file, item_noun="item", label_noun="category"):
        self.name = name; self.item = item_noun; self.label = label_noun
        d = json.load(open(split_file, encoding="utf-8"))
        self.pool = d["pool"]; self.test = d["test"]; self.test_dup = d.get("test_dup", [])
        self.LBL = sorted(set(r["label"] for r in self.pool + self.test + self.test_dup))
        self.budget = stratified_budget(self.pool, 2000)
        self.by = defaultdict(list)
        for r in self.budget: self.by[r["label"]].append(r["text"])   # equal budget for all methods

    def parse(self, o):
        o = (o or "").strip().strip('".').lower()
        if not o: return "UNPARSED"
        for l in self.LBL:
            if o == l.lower(): return l
        cont = [l for l in self.LBL if l.lower() in o]        # label appears in output
        if cont: return max(cont, key=len)                    # longest wins (handles nested labels)
        rev = [l for l in self.LBL if len(o) > 4 and o in l.lower()]  # output is a partial label
        return rev[0] if len(rev) == 1 else "UNPARSED"        # only if UNIQUE


def score(T, preds, gold):
    corr = [preds[i] == gold[i] for i in range(len(gold))]
    unp = np.mean([p == "UNPARSED" for p in preds])
    c = np.array(corr); rng = np.random.RandomState(0)
    bs = c[rng.randint(0, len(c), (2000, len(c)))].mean(1)
    return float(c.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), float(unp)


def _catlist(T): return "\n".join(f"- {l}" for l in T.LBL)


def zero_shot(T, texts):
    sys = f"Classify the {T.item} into its single best {T.label}. Reply with ONLY one {T.label} name copied verbatim."
    msgs = [[{"role": "system", "content": sys},
             {"role": "user", "content": f"{T.label.capitalize()} options:\n{_catlist(T)}\n\n{T.item.capitalize()}: {t[:TRUNC]}\n{T.label.capitalize()}:"}] for t in texts]
    return [T.parse(o) for o in chat_many(msgs, model=CLF_MODEL, max_tokens=24)]


def lexical_rag(T, texts, k=12):
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=40000)
    Xb = vec.fit_transform([r["text"] for r in T.budget]); Xt = vec.transform(texts)
    sims = (Xt @ Xb.T).toarray()
    bnorm = [norm(r["text"]) for r in T.budget]
    sys = f"Classify the {T.item} into its single best {T.label} using the labeled examples. Reply with ONLY one {T.label} name."
    msgs = []
    for i in range(len(texts)):
        qn = norm(texts[i]); order = np.argsort(-sims[i])
        nn = [j for j in order if bnorm[j] != qn][:k]         # exclude verbatim-duplicate demos
        demos = "\n".join(f"- \"{T.budget[j]['text'][:120]}\" -> {T.budget[j]['label']}" for j in nn)
        msgs.append([{"role": "system", "content": sys},
                     {"role": "user", "content": f"Examples:\n{demos}\n\n{T.item.capitalize()}: {texts[i][:TRUNC]}\n{T.label.capitalize()}:"}])
    return [T.parse(o) for o in chat_many(msgs, model=CLF_MODEL, max_tokens=24)]


def _diverse_demos(ts, k):
    """Pick k demos that COVER a class's variation (farthest-point sampling on TF-IDF). Works when texts
    are all unique (frequency-selection degenerates there); falls back to frequent templates if repeated."""
    c = Counter(norm(x) for x in ts); reps = {}
    for x in ts:
        n = norm(x)
        if n not in reps: reps[n] = x
    uniq = list(reps.values())
    if len(uniq) <= k:
        return uniq
    # if strongly templated, most-frequent is meaningful; else diversify
    if c.most_common(1)[0][1] >= 5:
        return [reps[n] for n, _ in c.most_common(k)]
    try:
        V = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_features=8000)
        X = V.fit_transform([u[:200] for u in uniq])
        chosen = [0]                                    # farthest-point sampling
        sims = (X @ X.T).toarray()
        while len(chosen) < k:
            mind = sims[:, chosen].max(1)               # similarity to nearest chosen (higher=closer)
            nxt = int(np.argmin([mind[i] if i not in chosen else 2 for i in range(len(uniq))]))
            chosen.append(nxt)
        return [uniq[i] for i in chosen]
    except Exception:
        return uniq[:k]


def kshot_per_class(T, texts, k=2, select="diverse"):
    """Fixed (prompt-only) k-shot. select='diverse' covers each class's variation (fair on unique text);
    'frequent' is the old most-common-template selection (only meaningful when texts repeat)."""
    def demos(ts, kk):
        if select == "frequent":
            c = Counter(norm(x) for x in ts); reps = {}
            for x in ts:
                n = norm(x)
                if n not in reps: reps[n] = x
            return [reps[n] for n, _ in c.most_common(kk)]
        return _diverse_demos(ts, kk)
    block = "\n".join(f"- \"{ex[:100]}\" -> {l}" for l in T.LBL for ex in demos(T.by[l], k))
    sys = f"Classify the {T.item} into its single best {T.label} using the labeled examples. Reply with ONLY one {T.label} name."
    msgs = [[{"role": "system", "content": sys},
             {"role": "user", "content": f"Examples:\n{block}\n\n{T.item.capitalize()}: {t[:TRUNC]}\n{T.label.capitalize()}:"}] for t in texts]
    return [T.parse(o) for o in chat_many(msgs, model=CLF_MODEL, max_tokens=24)]


def _rtext(r): return r["t"] if isinstance(r, dict) else r


def _revid(r, k=2):
    return (r.get("ev", [])[:k], r.get("n")) if isinstance(r, dict) else ([], None)


def _dline(T, c, D):
    """Rendered for the CLASSIFIER: the class definition plus its REMAP overrides (imperative
    conditions that send a look-alike to another class). The 'neg' slot holds remap phrases."""
    pos = "; ".join(_rtext(x) for x in D[c]["pos"][:MAXPOS]) or c.lower()
    rem = "; ".join(_rtext(x) for x in D[c]["neg"][:MAXNEG])
    return f"{c}: {pos}" + (f"  OVERRIDE: {rem}" if rem else "")


def desc_sys(T):
    """The classifier's system prompt (shared so per-class acceptance renders identically)."""
    return (f"Classify the {T.item} into its single best {T.label} using the definitions below. These "
            f"{T.label}s are the organization's FILING CONVENTIONS, which sometimes contradict common sense.\n"
            f"Each definition may list OVERRIDE conditions: if an OVERRIDE condition matches the {T.item}, "
            f"FOLLOW IT and assign the {T.label} it names, even if the wording otherwise fits this category. "
            f"Apply overrides literally; they encode counterintuitive routing the organization actually uses. "
            f"Reply with ONLY one {T.label} name.")


def _desc_classify(T, texts, D):
    book = "\n".join(_dline(T, c, D) for c in T.LBL)
    sys = desc_sys(T)
    msgs = [[{"role": "system", "content": sys},
             {"role": "user", "content": f"{T.label.capitalize()} definitions:\n{book}\n\n{T.item.capitalize()}: {t[:TRUNC]}\n{T.label.capitalize()}:"}] for t in texts]
    return [T.parse(o) for o in chat_many(msgs, model=CLF_MODEL, max_tokens=24)]


def _seed(T):
    """CONTRASTIVE seeding: mine all class descriptions together so they are mutually exclusive and
    shared/generic phrasings are assigned to their dominant (base-rate) class."""
    D = {c: {"pos": [], "neg": []} for c in T.LBL}
    freq = {c: len(T.by[c]) for c in T.LBL}
    blocks = []
    for c in T.LBL:
        ex = "; ".join(f'"{t[:70]}"' for t in T.by[c][:6])
        blocks.append(f"[{c}] (freq={freq[c]}): {ex}")
    allblocks = "\n".join(blocks)
    sys = (f"You document how ONE ORGANIZATION actually files {T.item}s into its {T.label}s.\n"
           f"CRITICAL: these {T.label}s are ADMINISTRATIVE CONVENTIONS, not common-sense semantic classes. "
           f"An organization often files an {T.item} under a {T.label} that contradicts its literal wording "
           f"(e.g. a request that mentions one topic is routed by the ACTION or WORKFLOW involved, not the "
           f"topic named). Your job is to capture the organization's ACTUAL filing behavior as shown in the "
           f"examples, even where it defies intuition. Never correct the organization; describe it.\n"
           f"Descriptions must be MUTUALLY EXCLUSIVE. If a phrase/feature appears under several {T.label}s, "
           f"assign it to the MOST FREQUENT one and tell the others to exclude it.")
    usr = (f"{T.label.capitalize()}s with real example {T.item}s and their frequencies:\n{allblocks}\n\n"
           f"Study the examples for CONVENTIONS that a common-sense classifier would get wrong: wording that "
           f"suggests one {T.label} but is actually filed under another. State those explicitly.\n\n"
           f"For EACH {T.label} output one line: \"<name>: <what this organization files here, including any "
           f"counterintuitive cases> | NOT: <what looks like it belongs but is filed elsewhere>\". "
           f"Use the exact names. Output all {len(T.LBL)} lines, nothing else.")
    o = _traced_call("seed", [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                     MINE_MODEL, 1400, n_labels=len(T.LBL))
    # parse "<name>: <pos> | NOT: <neg>" lines back to classes
    lname = {l.lower(): l for l in T.LBL}
    for line in o.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        m = re.match(r"(.+?):\s*(.*)", line)
        if not m: continue
        nm = m.group(1).strip().strip("[]").lower(); body = m.group(2).strip()
        if nm not in lname: continue
        c = lname[nm]
        if "NOT:" in body:
            pos, neg = body.split("NOT:", 1)
            D[c]["pos"].append(pos.strip(" |")[:160]); D[c]["neg"].append(neg.strip()[:120])
        else:
            D[c]["pos"].append(body[:160])
    for c in T.LBL:                                  # fallback if a class was missed
        if not D[c]["pos"]: D[c]["pos"].append(c.lower())
    return D


def _rules_json(D, c, with_evidence=True):
    """For the EDITOR: each rule plus the evidence that produced it, so a well-supported rule is not
    silently inverted. (The classifier prompt never sees this.)"""
    def fmt(lst):
        out = []
        for r in lst:
            if isinstance(r, dict):
                ev, n = _revid(r)
                item = {"rule": r["t"]}
                if with_evidence and ev: item["evidence"] = [e[:90] for e in ev]
                if with_evidence and n: item["supported_by_n_examples"] = n
                out.append(item)
            else:
                out.append({"rule": r})
        return out
    return json.dumps({"pos": fmt(D[c]["pos"]), "neg": fmt(D[c]["neg"])}, ensure_ascii=False)


def _freq_examples(exs, k=8):
    """Most FREQUENT error phrasings (templates) with counts -- the highest-value triggers."""
    c = Counter(norm(t) for t in exs); reps = {}
    for t in exs:
        n = norm(t)
        if n not in reps: reps[n] = t
    out = []
    for n, cnt in c.most_common(k):
        out.append(f'  ({cnt}x) "{reps[n][:130]}"')
    return "\n".join(out)


def _diagnose(T, gt, pred, exs, contrast, D, n_err=None):
    """STEP A -- DIAGNOSTIC. Explain the failure before touching the rulebook: which phrasings recur,
    which EXISTING rule caused the misfiling (attribution), what distinction the rulebook is missing,
    and what must not break. Returns a short structured JSON diagnosis (also inspectable by us)."""
    errs = _freq_examples(exs)
    cb = _freq_examples(contrast, 6) if contrast else "(none)"
    sys = (f"You are a diagnostician for a {T.label} rulebook. Do NOT propose new rules yet. Explain WHY the "
           f"classifier failed. Note that these {T.label}s are administrative conventions, not common-sense "
           f"classes: the organization may route by action/workflow rather than the topic the text names.")
    msg = (f"{n_err if n_err is not None else len(exs)} {T.item}s that truly belong to '{gt}' were filed as '{pred}'.\n\n"
           f"Rulebook for '{gt}': {_rules_json(D, gt)}\nRulebook for '{pred}': {_rules_json(D, pred)}\n\n"
           f"MISFILED (really '{gt}'), most frequent phrasings:\n{errs}\n\n"
           f"CORRECTLY filed as '{pred}' (must keep working):\n{cb}\n\n"
           f"Output STRICT JSON only:\n"
           '{"recurring_patterns": ["phrasing/feature that repeats in the misfiled items", ...],\n'
           ' "culprit_rules": ["quote the EXISTING rule (and whose class) that caused the wrong choice, or '
           '\'none\' if the rulebook is merely silent", ...],\n'
           ' "missing_distinction": "the distinction the rulebook fails to express, in one sentence",\n'
           ' "must_not_break": "what separates the correctly-filed contrast items, in one sentence"}')
    o = _traced_call("diagnose", [{"role": "system", "content": sys}, {"role": "user", "content": msg}],
                     MINE_MODEL, 500, gt=gt, pred=pred, n_err=n_err)
    m = re.search(r"\{.*\}", o, re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return {}


def _refine(T, gt, pred, exs, D, contrast=None, base_rates=None, n_err=None, diag=None):
    """STEP B -- RULEBOOK EDITOR: acts on the diagnosis. May ADD, EDIT, MERGE, SPLIT, CLARIFY, REORDER
    or DELETE rules for BOTH classes. Sees the diagnosis AND the raw evidence (so no detail is lost)."""
    errs = _freq_examples(exs)
    cblock = ""
    if contrast:
        cblock = (f"\nMUST NOT BREAK -- {T.item}s correctly filed as '{pred}':\n" + _freq_examples(contrast, 6) + "\n")
    rates = ""
    if base_rates:
        tot = sum(base_rates.values()) or 1
        rates = (f"\nBase rates in the labeled data: '{gt}'={base_rates.get(gt,0)} "
                 f"({base_rates.get(gt,0)/tot*100:.0f}%), '{pred}'={base_rates.get(pred,0)} "
                 f"({base_rates.get(pred,0)/tot*100:.0f}%). Use these for 'if wording is generic, choose X' clauses.\n")
    sys = (f"You MAINTAIN a rulebook that documents how ONE ORGANIZATION files {T.item}s into {T.label}s.\n"
           f"These {T.label}s are ADMINISTRATIVE CONVENTIONS, not common-sense semantic classes: the "
           f"organization often routes by the ACTION/WORKFLOW/RESPONSIBLE UNIT rather than the topic the text "
           f"names, so a correct rule may contradict the literal wording. Document behavior; never correct it.\n\n"
           f"You have FULL EDITORIAL CONTROL over both classes' rules. You may:\n"
           f"  ADD a new rule; EDIT or CLARIFY an existing one; MERGE redundant rules; SPLIT an overloaded "
           f"rule; REORDER by importance; DELETE a rule that is wrong, vague, or contradicts another.\n"
           f"Rules may take any useful form: a literal trigger phrase (\"if the text contains 'X'\"), a "
           f"conditional (\"if A and not B\"), an exception, a priority/default (\"if only generic wording, "
           f"choose X\"), or a short definition. Be creative but PRECISE.\n"
           f"IMPORTANT: existing rules carry the EVIDENCE that produced them. Do NOT invert or delete a rule "
           f"whose evidence still holds; if a rule conflicts with the new errors, NARROW it with a condition "
           f"instead of reversing it, and keep both cases correct. "
           f"Quality bar: each rule must DISCRIMINATE (state something that separates these two classes), be "
           f"NON-REDUNDANT, and NOT CONTRADICT another rule. Prefer few sharp rules over many vague ones "
           f"(max {MAXPOS} pos and {MAXNEG} neg per class). Keep each rule under 30 words.")
    dblock = ""
    if diag:
        dblock = ("\nDIAGNOSIS (from the analyst):\n"
                  f"  recurring patterns: {diag.get('recurring_patterns')}\n"
                  f"  CULPRIT RULES (fix or delete these): {diag.get('culprit_rules')}\n"
                  f"  missing distinction: {diag.get('missing_distinction')}\n"
                  f"  must not break: {diag.get('must_not_break')}\n")
    msg = (f"CONFUSION: {n_err if n_err is not None else len(exs)} {T.item}s that truly belong to '{gt}' were "
           f"filed by the classifier as '{pred}'.\n\n"
           f"Current rules for '{gt}': {_rules_json(D, gt)}\n"
           f"Current rules for '{pred}': {_rules_json(D, pred)}\n{rates}{dblock}\n"
           f"MOST FREQUENT misfiled phrasings (these are really '{gt}'):\n{errs}\n{cblock}\n"
           f"Act on the diagnosis: FIX or DELETE any culprit rule, then rewrite both rulebooks so a classifier "
           f"gets these right WITHOUT breaking the contrast examples.\n\n"
           f"Output STRICT JSON only:\n"
           f'{{"{gt}": {{"pos": ["..."], "neg": ["..."]}}, "{pred}": {{"pos": ["..."], "neg": ["..."]}}}}')
    o = _traced_call("edit", [{"role": "system", "content": sys}, {"role": "user", "content": msg}],
                     MINE_MODEL, 900, gt=gt, pred=pred, n_err=n_err,
                     rules_before={gt: D[gt], pred: D[pred]}, diagnosis=diag)
    # full-rewrite parse: returns updated rule sets for both classes
    m = re.search(r"\{.*\}", o, re.S)
    if m:
        try:
            j = json.loads(m.group(0)); upd = {}
            for c in (gt, pred):
                if c in j and isinstance(j[c], dict):
                    p = [str(x).strip()[:180] for x in j[c].get("pos", []) if str(x).strip()][:MAXPOS]
                    n = [str(x).strip()[:180] for x in j[c].get("neg", []) if str(x).strip()][:MAXNEG]
                    if p: upd[c] = {"pos": p, "neg": n}
            if upd:
                ev = [t[:120] for t in exs[:3]]
                for c in upd:
                    upd[c]["pos"] = [{"t": x, "ev": ev, "n": n_err} for x in upd[c]["pos"]]
                    upd[c]["neg"] = [{"t": x, "ev": ev, "n": n_err} for x in upd[c]["neg"]]
                return upd
        except Exception:
            pass
    return {}


def apply_update(D, upd):
    """Apply the editor's rewritten rule sets (full replacement per class)."""
    for c, v in (upd or {}).items():
        if c in D and v.get("pos"):
            D[c]["pos"] = v["pos"][:MAXPOS]; D[c]["neg"] = v.get("neg", [])[:MAXNEG]
    return D


def desc(T, texts, mine, val, rounds=6, batches=8):
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    v_txt = [r["text"] for r in val]; v_gold = [r["label"] for r in val]
    D = _seed(T); tried = set()
    def vacc(DD):
        vp = _desc_classify(T, v_txt, DD); return float(np.mean([vp[i] == v_gold[i] for i in range(len(val))]))
    best_D, best_v = copy.deepcopy(D), vacc(D)
    for _ in range(rounds):
        m_pred = _desc_classify(T, m_txt, D)
        conf = Counter((m_gold[i], m_pred[i]) for i in range(len(mine)) if m_pred[i] != m_gold[i] and m_pred[i] != "UNPARSED")
        bt = [(g, p) for (g, p), _ in conf.most_common() if (g, p) not in tried and conf[(g, p)] >= 2][:batches]
        if not bt: break
        base = {c: len(T.by[c]) for c in T.LBL}
        for gt, pr in bt:
            exs = [m_txt[i] for i in range(len(mine)) if m_gold[i] == gt and m_pred[i] == pr]
            # contrast: items the classifier called `pr` and WAS right -> keeps the rule conditional
            contrast = [m_txt[i] for i in range(len(mine)) if m_gold[i] == pr and m_pred[i] == pr]
            diag = _diagnose(T, gt, pr, exs, contrast, D, n_err=conf[(gt, pr)])
            upd = _refine(T, gt, pr, exs, D, contrast=contrast, base_rates=base,
                          n_err=conf[(gt, pr)], diag=diag)
            apply_update(D, upd); tried.add((gt, pr))
        v = vacc(D)
        if v >= best_v: best_v, best_D = v, copy.deepcopy(D)
        else: D = copy.deepcopy(best_D); tried -= set(bt)   # allow reverted pairs to be retried
    return _desc_classify(T, texts, best_D)


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "bloom"
    methods = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["zero_shot", "lexical_rag", "kshot", "desc"])
    T = TASKS[task]
    test = T.test; texts = [r["text"] for r in test]; gold = [r["label"] for r in test]
    dup = T.test_dup; dtx = [r["text"] for r in dup]; dgold = [r["label"] for r in dup]
    mine = T.budget[:800]; val = T.budget[800:1100]
    print(f"Task={task} item='{T.item}' label='{T.label}' labels={len(T.LBL)} NOVEL-test={len(test)} dup-test={len(dup)}")
    print(f"models: mine={MINE_MODEL} classify={CLF_MODEL}")
    for m in methods:
        fn = {"zero_shot": zero_shot, "lexical_rag": lexical_rag, "kshot": kshot_per_class}.get(m)
        preds = fn(T, texts) if fn else (desc(T, texts, mine, val) if m == "desc" else None)
        if preds is None: print(f"  unknown {m}"); continue
        acc, ci, unp = score(T, preds, gold)
        line = f"  {m:12s} NOVEL acc={acc:.4f} 95%CI=({ci[0]:.3f},{ci[1]:.3f}) unparsed={unp:.3f}"
        if dup:   # dup slice (where verbatim answer is retrievable)
            dp = fn(T, dtx) if fn else desc(T, dtx, mine, val)
            dacc, _, _ = score(T, dp, dgold); line += f"  | dup acc={dacc:.4f}"
        print(line)


def _try(name, *a):
    try: return Task(name, *a)
    except Exception: return None


TASKS = {k: v for k, v in {
    "bloom": _try("bloom", "results/bloom_fair.json", "municipal service request", "service category"),
    "br":    _try("br", "results/br_split.json", "municipal service request", "service category"),
    "hupd":  _try("hupd", "results/hupd_split.json", "patent", "technical classification code"),
    "mimic": _try("mimic", "results/mimic_split.json", "clinical note", "diagnosis category"),
}.items() if v is not None}
if __name__ == "__main__":
    main()


def paired_test(preds_a, preds_b, gold, n_boot=5000, seed=0):
    """Paired comparison of two methods on the SAME items (far more sensitive than independent CIs).
    Returns delta (a-b), a bootstrap CI on the delta, McNemar counts, and a two-sided p-value."""
    a = np.array([preds_a[i] == gold[i] for i in range(len(gold))])
    b = np.array([preds_b[i] == gold[i] for i in range(len(gold))])
    d = a.astype(int) - b.astype(int)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(1)
    n01 = int(((~a) & b).sum())      # b right, a wrong
    n10 = int((a & (~b)).sum())      # a right, b wrong
    # exact binomial (McNemar) on the discordant pairs
    from scipy import stats as _st
    p = _st.binomtest(n10, n10 + n01, 0.5).pvalue if (n10 + n01) > 0 else 1.0
    return {"delta": float(d.mean()),
            "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
            "a_only_right": n10, "b_only_right": n01, "p_mcnemar": float(p),
            "significant": bool(p < 0.05)}


MARGIN = 0.01


def mine_rulebook(T, mine, val, rounds=5, batches=8, gate_n=300, mine_cap=1500, verbose=True):
    """CANONICAL miner with PER-EDIT validation.
    Each confusion pair's edit is applied only if it does not hurt a held-out gate slice, so a single
    inverted rule can no longer ride in on a batch (the failure that made mining degrade with budget)."""
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    V_txt = [r["text"] for r in val]; V_gold = [r["label"] for r in val]
    _rng = np.random.RandomState(0)
    def gate_slice():
        """ROTATING gate sample: a fresh random subset per edit, so ~40 accept/reject decisions
        cannot overfit one fixed slice (the failure that made b=2000 regress)."""
        if len(V_txt) <= gate_n: return list(range(len(V_txt)))
        return _rng.permutation(len(V_txt))[:gate_n].tolist()
    D = _seed(T); tried = set()
    def vacc(DD, sl):
        vp = _desc_classify(T, [V_txt[i] for i in sl], DD)
        return float(np.mean([vp[j] == V_gold[sl[j]] for j in range(len(sl))]))
    full = list(range(len(V_txt)))
    cur = vacc(D, full)
    if verbose: print(f"  [mine] seed gate={cur:.3f} (val n={len(V_txt)}, rotating slice {gate_n}, margin {MARGIN})")
    # error DISCOVERY runs on a capped sample (confusion ranking is stable well before the full budget);
    # rule WRITING still draws its examples from the full mine set below.
    if len(m_txt) > mine_cap:
        _s = np.random.RandomState(1).permutation(len(m_txt))[:mine_cap]
        d_txt = [m_txt[i] for i in _s]; d_gold = [m_gold[i] for i in _s]
    else:
        d_txt, d_gold = m_txt, m_gold
    for rnd in range(rounds):
        sl_round = gate_slice()                    # rotate ONCE per round -> v_old reused across edits
        v_old_round = vacc(D, sl_round)
        m_pred = _desc_classify(T, d_txt, D)
        conf = Counter((d_gold[i], m_pred[i]) for i in range(len(d_txt))
                       if m_pred[i] != d_gold[i] and m_pred[i] != "UNPARSED")
        bt = [(g, p) for (g, p), _ in conf.most_common() if (g, p) not in tried and conf[(g, p)] >= 2][:batches]
        if not bt:
            if verbose: print("  [mine] no recurring confusions left"); 
            break
        base = {c: len(T.by[c]) for c in T.LBL}
        kept = 0
        for gt, pr in bt:
            exs = [d_txt[i] for i in range(len(d_txt)) if d_gold[i] == gt and m_pred[i] == pr]
            contrast = [d_txt[i] for i in range(len(d_txt)) if d_gold[i] == pr and m_pred[i] == pr]
            diag = _diagnose(T, gt, pr, exs, contrast, D, n_err=conf[(gt, pr)])
            upd = _refine(T, gt, pr, exs, D, contrast=contrast, base_rates=base,
                          n_err=conf[(gt, pr)], diag=diag)
            tried.add((gt, pr))
            if not upd: continue
            trial = copy.deepcopy(D); apply_update(trial, upd)
            sl = sl_round                                  # one slice per round (cheaper, still rotates)
            v_old, v_new = v_old_round, vacc(trial, sl)
            ok = v_new >= v_old + MARGIN                   # require a real margin, not noise
            if ok:
                D, kept = trial, kept + 1
                v_old_round = v_new
                cur = vacc(D, full)                        # re-anchor on the FULL val
            _trace("edit_gate", gt=gt, pred=pr, slice_n=len(sl),
                   gate_before=v_old, gate_trial=v_new, margin=MARGIN, accepted=bool(ok), full_gate=cur)
        if verbose: print(f"  [mine] round {rnd+1}: {kept}/{len(bt)} edits accepted, gate={cur:.3f}")
        if kept == 0: break
    return D, cur
