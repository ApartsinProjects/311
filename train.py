"""TRAIN: rule mining framed as batched mini-batch training.

  weights   = per-class rulebook (pos/neg rules)
  forward   = classify all train items with the current rulebook        (1 batch)
  loss      = diagnostic LLM over ALL confusion pairs at once            (1 batch)
  backprop  = reshuffle diagnostics by GOLD class (route error signal)
  optimizer = one refiner call per class, rewriting its rules            (1 batch)
  epoch     = forward -> diagnose -> refine
  val gate  = keep the epoch only if held-out accuracy improved (else revert)  [SGD step accept]
  test      = fixed hold-out, scored once at the end

3 batched LLM calls per epoch (vs ~16 sequential diagnose->refine pairs before).

  python train.py [budget] [epochs]
"""
import sys, os, json, re, copy
import numpy as np
from collections import defaultdict, Counter
import semclf, triggers
from semclf import (TASKS, CLF_MODEL, MINE_MODEL, score, paired_test, stratified_budget,
                    _rules_json, _freq_examples, _desc_classify, apply_update, chat_many, _trace)


# ---------- LOSS: diagnose every confusion pair in ONE batch ----------
def diagnose_batch(T, pairs, exs_by, contrast_by, D):
    msgs, keys = [], []
    for (gt, pr) in pairs:
        errs = _freq_examples(exs_by[(gt, pr)]); cb = _freq_examples(contrast_by.get((gt, pr), []), 6) or "(none)"
        sysm = (
            f"You analyze why a classifier confuses two {T.label}s. Find the UNDERLYING DIMENSION that truly "
            f"separates them, not the surface words. Real distinctions are usually about intent, action vs "
            f"object, stage of a process, severity, who is responsible, or a defining attribute -- while both "
            f"classes may share the same surface vocabulary. A {T.label}'s true membership rule can be narrower, "
            f"broader, or different from what its name suggests; infer it from the examples, not the name. "
            f"Diagnose only; do not write rules yet.")
        usr = (f"{len(exs_by[(gt, pr)])} {T.item}s whose true {T.label} is '{gt}' were misclassified as '{pr}'.\n\n"
               f"Rules for '{gt}': {_rules_json(D, gt)}\nRules for '{pr}': {_rules_json(D, pr)}\n\n"
               f"MISCLASSIFIED (truly '{gt}'):\n{errs}\n\nCORRECTLY '{pr}' (must stay correct):\n{cb}\n\n"
               'Output STRICT JSON: {'
               '"dimension":"the underlying axis that separates these two classes (not surface words)",'
               '"gt_signal":"what marks an item as \'' + gt + '\' on that axis",'
               '"pr_signal":"what marks an item as \'' + pr + '\' on that axis",'
               '"culprit":"which current rule/word caused the error, or none",'
               '"fix":"the general distinction the rulebook must add, one sentence"}')
        msgs.append([{"role": "system", "content": sysm}, {"role": "user", "content": usr}]); keys.append((gt, pr))
    outs = chat_many(msgs, model=MINE_MODEL, max_tokens=350)
    diags = {}
    for k, o in zip(keys, outs):
        m = re.search(r"\{.*\}", o or "", re.S)
        try: diags[k] = json.loads(m.group(0)) if m else {}
        except Exception: diags[k] = {}
        _trace("diagnose_batch", gt=k[0], pred=k[1], response=o)
    return diags


NVAR = int(__import__("os").environ.get("TRAIN_NVAR", "1"))     # candidate rewrites proposed per class
MARGIN = float(__import__("os").environ.get("TRAIN_MARGIN", "0"))        # acceptance margin on the ACC val half
MINE_TEMP = float(__import__("os").environ.get("TRAIN_MINE_TEMP", "0"))  # refiner temperature (search noise)


# ---------- OPTIMIZER: NVAR refiner variants per GOLD class, all in ONE batch ----------
def refine_batch(T, class_diags, exs_by, contrast_by, D, base, recall=None):
    """Returns {class: [candidate_update, ...]} -- NVAR proposals per class (population search).
    class_diags[c] = list of (other_class, role, diag) where role='gt' (c true) or 'pr' (c wrong pick)."""
    msgs, keys = [], []
    for c, items in class_diags.items():
        blocks = []
        for other, role, dg in items:
            dim = dg.get("dimension", ""); fix = dg.get("fix", "")
            if role == "gt":
                ex = _freq_examples(exs_by[(c, other)], 5)
                blocks.append(f"* vs '{other}' -- these are truly '{c}' but were lost to '{other}'.\n"
                              f"    axis: {dim}\n    '{c}' side: {dg.get('gt_signal','')}\n    fix: {fix}\n    examples:\n{ex}")
            else:
                ex = _freq_examples(exs_by[(other, c)], 5)
                blocks.append(f"* vs '{other}' -- these are truly '{other}' but leaked INTO '{c}'.\n"
                              f"    axis: {dim}\n    '{other}' side: {dg.get('gt_signal','')}\n    fix: {fix}\n    examples of the leak:\n{ex}")
        if recall and recall.get(c):
            blocks.append(f"* RECALL GAPS (missed '{c}' items incl. rare phrasings) -- add general pos "
                          f"coverage for these WITHOUT over-widening:\n    " + "\n    ".join(recall[c]))
        rate = base.get(c, 0) / (sum(base.values()) or 1) * 100
        confusers = sorted({other for other, _, _ in items})
        pos_ex = _freq_examples(T.by.get(c, [])[:40], 6)
        sysm = (
            f"You write the COMPLETE, self-contained definition of ONE {T.label}: '{c}'.\n"
            f"A compact set of general rules can capture everything needed to recognize a class. Write that set "
            f"so it classifies UNSEEN {T.item}s correctly, not just the examples shown.\n\n"
            f"PRINCIPLES (domain-independent):\n"
            f"1. MEANING OVER NAME. The name is just a label; infer the ACTUAL membership criterion from the "
            f"examples. It may be narrower, broader, or qualitatively different from what the name connotes. If "
            f"the name is misleading, state the real criterion explicitly.\n"
            f"2. GENERALIZE, DON'T MEMORIZE. Turn the specific examples into the underlying principle. Each rule "
            f"should decide cases you have not seen; never encode one instance.\n"
            f"3. DISCRIMINATE ON THE UNDERLYING AXIS. For every class '{c}' is confused with, give the ONE "
            f"decisive difference on the axis that truly separates them (intent, action vs object, stage, "
            f"severity, responsible party, defining attribute). Never rely on a surface word both classes use.\n"
            f"4. CAPTURE COUNTERINTUITIVE CASES. Where surface features point elsewhere but the true label is "
            f"'{c}' (or the reverse), make that an explicit conditional -- these are the rules that carry weight.\n"
            f"5. MUTUAL EXCLUSIVITY + DEFAULT. If a shared feature belongs to '{c}' only under a condition, state "
            f"the condition precisely. If wording is generic, decide by base rate.\n\n"
            f"Output two kinds of rules (max {semclf.MAXPOS} pos, {semclf.MAXNEG} remap), each a sharp "
            f"general condition under 25 words, no redundancy, no contradiction:\n"
            f"- pos: what BELONGS to '{c}'. State only the condition; do NOT append '-> {c}' or any class "
            f"name (it is understood).\n"
            f"- remap: IMPERATIVE overrides for look-alikes -- {T.item}s whose wording fits '{c}' but which "
            f"the organization actually files elsewhere. Write each as 'if <narrow condition>, classify as "
            f"<CLASS> instead'. Keep the condition NARROW so it fires only on the real exception, never on "
            f"ordinary '{c}' items. These are how '{c}' gives up cases it would otherwise wrongly capture.")
        usr = (
            f"{T.label.upper()} TO DEFINE: '{c}'  (base rate {rate:.0f}%; confused with: {', '.join(confusers)})\n\n"
            f"Current rules: {_rules_json(D, c)}\n\n"
            f"Representative '{c}' {T.item}s:\n{pos_ex}\n\n"
            f"ERROR EVIDENCE (each must be handled):\n" + "\n".join(blocks) + "\n\n"
            f"Infer what '{c}' TRULY is, then write pos rules and narrow remap overrides.\n"
            f'STRICT JSON: {{"pos":["condition that makes it {c}"],"remap":["if <narrow condition>, classify as <CLASS> instead"]}}.')
        # emit NVAR variant proposals for this class (distinct prompts -> distinct, cacheable rewrites;
        # combined with MINE_TEMP this samples the rule space)
        for k in range(NVAR):
            vhint = usr if k == 0 else (usr + f"\n\nProposal variant #{k+1}: offer a DIFFERENT valid "
                                        f"formulation than an obvious first attempt -- rephrase the axis, "
                                        f"try alternative narrow conditions -- while staying correct.")
            msgs.append([{"role": "system", "content": sysm}, {"role": "user", "content": vhint}])
            keys.append((c, k))
    outs = chat_many(msgs, model=MINE_MODEL, max_tokens=700, temperature=MINE_TEMP)
    cands = defaultdict(list)
    for (c, k), o in zip(keys, outs):
        m = re.search(r"\{.*\}", o or "", re.S)
        if not m: continue
        def _clean(x, c=c):
            x = str(x).strip()
            x = re.sub(r"\s*-+>\s*" + re.escape(c) + r"\b.*$", "", x).strip()
            return x[:260]
        try:
            j = json.loads(m.group(0))
            pos = [_clean(x) for x in j.get("pos", []) if str(x).strip()]
            pos = [x for x in pos if x][:semclf.MAXPOS]
            rem = j.get("remap", j.get("neg", []))
            rem = [str(x).strip()[:260] for x in rem if str(x).strip()][:semclf.MAXNEG]
            if pos: cands[c].append({"pos": pos, "neg": rem})
        except Exception:
            pass
    _trace("refine_variants", n_classes=len(cands), n_proposals=sum(len(v) for v in cands.values()))
    return dict(cands)


RULEBOOK_MAX = int(__import__("os").environ.get("TRAIN_RULEBOOK_MAX", "6000"))  # target chars for the whole book


def _book_chars(T, D):
    return sum(len(semclf._dline(T, c, D)) for c in T.LBL)


def compact_batch(T, D, over_frac=1.3):
    """COMPACTER: when the rendered rulebook exceeds RULEBOOK_MAX, ask the LLM to rewrite the LONGEST
    classes' rules more tightly -- merge redundant rules, drop low-value ones, shorten wording -- WITHOUT
    losing the discriminative pos rules or the remap overrides. Returns {class: compacted_update}; each is
    gated on val by the caller exactly like a refine candidate, so a lossy compaction is rejected."""
    sizes = {c: len(semclf._dline(T, c, D)) for c in T.LBL}
    total = sum(sizes.values())
    if total <= RULEBOOK_MAX:
        return {}
    fair = RULEBOOK_MAX / max(len(T.LBL), 1)
    targets = [c for c in T.LBL if sizes[c] > fair * over_frac and (D[c]["pos"] or D[c]["neg"])]
    targets = sorted(targets, key=lambda c: -sizes[c])[:12]
    if not targets:
        return {}
    msgs = []
    for c in targets:
        budget_words = max(int(fair / 6), 25)          # ~6 chars/word
        sysm = (f"You COMPACT the rulebook for ONE {T.label}: '{c}', to fit a tight length budget WITHOUT "
                f"losing accuracy. Merge redundant rules, delete low-value ones, and shorten wording, but KEEP "
                f"every discriminating condition and every REMAP override (the 'if ... classify as X instead' "
                f"rules that route look-alikes elsewhere -- these carry the most weight). Prefer fewer, sharper "
                f"rules. Do not add new content. Target about {budget_words} words total across pos+remap.")
        usr = (f"Current rules for '{c}' (too long): {_rules_json(D, c)}\n\n"
               f"Rewrite them shorter.\n"
               f'STRICT JSON: {{"pos":["..."],"remap":["if <condition>, classify as <CLASS> instead"]}}.')
        msgs.append([{"role": "system", "content": sysm}, {"role": "user", "content": usr}])
    outs = chat_many(msgs, model=MINE_MODEL, max_tokens=600)
    upd = {}
    for c, o in zip(targets, outs):
        m = re.search(r"\{.*\}", o or "", re.S)
        if not m: continue
        try:
            j = json.loads(m.group(0))
            pos = [str(x).strip()[:260] for x in j.get("pos", []) if str(x).strip()][:semclf.MAXPOS]
            rem = [str(x).strip()[:260] for x in j.get("remap", j.get("neg", [])) if str(x).strip()][:semclf.MAXNEG]
            if pos: upd[c] = {"pos": pos, "neg": rem}
        except Exception:
            pass
    _trace("compact", n_targets=len(targets), book_before=total, target=RULEBOOK_MAX)
    return upd


RECALL_OFF = __import__("os").environ.get("RECALL_OFF") == "1"


def recall_batch(T, m_txt, m_gold, pred, D, min_missed=3):
    """IDEA 2: per-class RECALL loss. The pair loss (min_err>=2) is blind to the ~75% of errors that are
    unique-phrasing singletons. For each gold class, gather ALL its missed items (regardless of predicted
    class, incl. singletons) and ask the miner what phrasing families the pos rules fail to cover. Returns
    {class: coverage_text} to inject into refine_batch as extra evidence."""
    if RECALL_OFF: return {}
    missed = defaultdict(list)
    for i in range(len(m_txt)):
        if pred[i] != m_gold[i] and pred[i] != "UNPARSED":
            missed[m_gold[i]].append(m_txt[i])
    targets = [c for c in T.LBL if len(missed[c]) >= min_missed]
    if not targets: return {}
    msgs = []
    for c in targets:
        sysm = (f"You improve the RECALL of ONE {T.label}: '{c}'. Below are {T.item}s that TRULY are '{c}' "
                f"but the current rules missed (they include one-off phrasings, not just repeated patterns). "
                f"Identify the phrasing FAMILIES the current pos rules fail to cover, and output 2-3 GENERAL "
                f"coverage conditions that would catch them WITHOUT over-widening to other classes. Each must "
                f"be a general principle (not one instance), under 25 words.")
        usr = (f"Current pos rules for '{c}': {_rules_json(D, c)}\n\n"
               f"MISSED '{c}' {T.item}s (incl. rare phrasings):\n{_freq_examples(missed[c], 12)}\n\n"
               f'STRICT JSON: {{"coverage": ["also counts as {c} when <general condition>", ...]}}')
        msgs.append([{"role": "system", "content": sysm}, {"role": "user", "content": usr}])
    outs = chat_many(msgs, model=MINE_MODEL, max_tokens=300)
    rec = {}
    for c, o in zip(targets, outs):
        m = re.search(r"\{.*\}", o or "", re.S)
        if not m: continue
        try:
            cov = [str(x).strip() for x in json.loads(m.group(0)).get("coverage", []) if str(x).strip()]
            if cov: rec[c] = cov
        except Exception:
            pass
    return rec


def train(T, mine, val, epochs=6, min_err=2, patience=2, gate_n=10**9):  # full val for acceptance
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    v_txt = [r["text"] for r in val]; v_gold = [r["label"] for r in val]
    base = {c: len(T.by[c]) for c in T.LBL}
    D = semclf._seed(T)
    def vacc(DD):
        vp = _desc_classify(T, v_txt, DD)
        return float(np.mean([vp[i] == v_gold[i] for i in range(len(v_gold))]))
    best_D, best_v = copy.deepcopy(D), vacc(D)
    print(f"  [epoch 0] seed val={best_v:.4f}")
    bad = 0
    for ep in range(1, epochs + 1):
        # FORWARD (1 batch)
        pred = _desc_classify(T, m_txt, D)
        tr_acc = np.mean([pred[i] == m_gold[i] for i in range(len(mine))])
        conf = Counter((m_gold[i], pred[i]) for i in range(len(mine)) if pred[i] != m_gold[i] and pred[i] != "UNPARSED")
        pairs = [p for p, n in conf.most_common() if n >= min_err]
        if not pairs:
            print(f"  [epoch {ep}] train={tr_acc:.3f} no recurring errors -> stop"); break
        exs_by = {(gt, pr): [m_txt[i] for i in range(len(mine)) if m_gold[i] == gt and pred[i] == pr] for (gt, pr) in pairs}
        contrast_by = {(gt, pr): [m_txt[i] for i in range(len(mine)) if m_gold[i] == pr and pred[i] == pr] for (gt, pr) in pairs}
        # LOSS (1 batch)
        diags = diagnose_batch(T, pairs, exs_by, contrast_by, D)
        # BACKPROP: route diagnostics to each class by GOLD (and as neg-signal to the wrong pick)
        class_diags = defaultdict(list)
        for (gt, pr) in pairs:
            class_diags[gt].append((pr, "gt", diags[(gt, pr)]))
            class_diags[pr].append((gt, "pr", diags[(gt, pr)]))
        # IDEA 2: per-class RECALL signal from ALL missed items (incl. singletons); ensure those classes
        # get refined even if they have no ≥2 confusion pair.
        recall = recall_batch(T, m_txt, m_gold, pred, D)
        for c in recall:
            class_diags.setdefault(c, [])
        # OPTIMIZER: all class rewrites computed in ONE batch, then accepted PER CLASS. Each candidate is
        # judged alone against a fixed val slice so one bad rewrite cannot sink the good ones; all the
        # per-class judgements run as ONE batched classification (slice x candidates), keeping it cheap.
        # POPULATION SEARCH: NVAR variant rewrites per class; each variant judged ALONE on the val slice
        # in one big batch; keep the BEST variant per class if it beats the current book (fitness gate).
        cands = refine_batch(T, class_diags, exs_by, contrast_by, D, base, recall=recall)
        n = min(gate_n, len(v_txt))
        # SPLIT val: SEL half picks the best variant, ACC half gates acceptance -> the argmax is not
        # computed on the same data that judges it (removes the maximization/overfitting bias).
        sel = list(range(0, n, 2)); acc_i = list(range(1, n, 2))
        sl_txt = [v_txt[i] for i in range(n)]; sl_gold = [v_gold[i] for i in range(n)]
        base_pred = _desc_classify(T, sl_txt, D)
        base_acc = np.mean([base_pred[i] == sl_gold[i] for i in acc_i])
        big_msgs, spans = [], []
        for c, variants in cands.items():
            for vi, cand in enumerate(variants):
                trial = copy.deepcopy(D); apply_update(trial, {c: cand})
                book = "\n".join(semclf._dline(T, cc, trial) for cc in T.LBL)
                start = len(big_msgs)
                for t in sl_txt:
                    big_msgs.append([{"role": "system", "content": semclf.desc_sys(T)},
                                     {"role": "user", "content": f"{T.label.capitalize()} definitions:\n{book}\n\n{T.item.capitalize()}: {t[:semclf.TRUNC]}\n{T.label.capitalize()}:"}])
                spans.append((c, vi, start, len(big_msgs)))
        outs = chat_many(big_msgs, model=CLF_MODEL, max_tokens=24)      # ONE batch, all classes x variants
        best_var = {}                                                  # c -> (sel_acc, acc_acc, cand)
        for c, vi, s0, s1 in spans:
            preds = [T.parse(o) for o in outs[s0:s1]]
            sel_acc = np.mean([preds[i] == sl_gold[i] for i in sel])
            acc_acc = np.mean([preds[i] == sl_gold[i] for i in acc_i])
            if c not in best_var or sel_acc > best_var[c][0]:          # pick winner on SEL
                best_var[c] = (sel_acc, acc_acc, cands[c][vi])
        accepted = 0
        for c, (sel_acc, acc_acc, cand) in best_var.items():
            if acc_acc >= base_acc + MARGIN:                          # gate the SEL-winner on ACC half
                apply_update(D, {c: cand}); accepted += 1
        v = vacc(D)
        nprop = sum(len(vs) for vs in cands.values())
        # COMPACTER: if the rulebook is over budget, compress the longest classes; keep each compaction
        # only if it holds accuracy on the ACC val half (a lossy compaction is rejected, same gate).
        comp_upd = compact_batch(T, D); comp_acc = 0
        if comp_upd:
            cbase = [T.parse(o) for o in _desc_classify(T, sl_txt, D)]
            cbase_acc = np.mean([cbase[i] == sl_gold[i] for i in acc_i])
            cmsgs, cspans = [], []
            for c, cand in comp_upd.items():
                trial = copy.deepcopy(D); apply_update(trial, {c: cand})
                book = "\n".join(semclf._dline(T, cc, trial) for cc in T.LBL); s0 = len(cmsgs)
                for t in sl_txt:
                    cmsgs.append([{"role": "system", "content": semclf.desc_sys(T)},
                                  {"role": "user", "content": f"{T.label.capitalize()} definitions:\n{book}\n\n{T.item.capitalize()}: {t[:semclf.TRUNC]}\n{T.label.capitalize()}:"}])
                cspans.append((c, s0, len(cmsgs)))
            couts = chat_many(cmsgs, model=CLF_MODEL, max_tokens=24)
            for c, s0, s1 in cspans:
                preds = [T.parse(o) for o in couts[s0:s1]]
                if np.mean([preds[i] == sl_gold[i] for i in acc_i]) >= cbase_acc - 0.005:  # small size/acc tolerance
                    apply_update(D, {c: comp_upd[c]}); comp_acc += 1
            v = vacc(D)
        nchars = _book_chars(T, D)
        print(f"  [epoch {ep}] train={tr_acc:.3f} pairs={len(pairs)} classes={len(cands)} "
              f"proposals={nprop} accepted={accepted} compacted={comp_acc} book={nchars}c val={v:.4f} (best {max(v,best_v):.4f})")
        if v > best_v:
            best_v, best_D = v, copy.deepcopy(D); bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  [early stop] no val gain for {patience} epochs"); break
    # FINAL compaction on the returned artifact: shrink best_D under budget, keep only compactions that
    # hold val within tolerance. Loops until at/under target or no more lossless compaction is possible.
    for _ in range(4):
        if _book_chars(T, best_D) <= RULEBOOK_MAX: break
        cu = compact_batch(T, best_D)
        if not cu: break
        vt = [r["text"] for r in val]; vg = [r["label"] for r in val]
        b0 = np.mean([p == vg[i] for i, p in enumerate(_desc_classify(T, vt, best_D))])
        kept = 0
        for c, cand in cu.items():
            trial = copy.deepcopy(best_D); apply_update(trial, {c: cand})
            a = np.mean([p == vg[i] for i, p in enumerate(_desc_classify(T, vt, trial))])
            if a >= b0 - 0.005: best_D = trial; kept += 1
        print(f"  [final compact] book={_book_chars(T, best_D)}c kept={kept} val={vacc(best_D):.4f}")
        if kept == 0: break
    return best_D, vacc(best_D)


def add_disambiguation(T, D, mine, val, min_err=2, gate_n=400):
    """IDEA 1: turn the (already-computed) diagnostic axes into persistent pairwise TIE-BREAKER lines
    rendered at the end of the prompt, val-gated. Gives confusable-neighbor pairs the decision BOUNDARY
    that RAG can't provide. Reuses diagnose_batch; each line greedily kept only if it doesn't hurt val."""
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    v_txt = [r["text"] for r in val][:gate_n]; v_gold = [r["label"] for r in val][:gate_n]
    pred = _desc_classify(T, m_txt, D)
    conf = Counter((m_gold[i], pred[i]) for i in range(len(mine)) if pred[i] != m_gold[i] and pred[i] != "UNPARSED")
    pairs = [p for p, n in conf.most_common(semclf.MAXDISAMB * 2) if n >= min_err]
    if not pairs: return D
    exs_by = {(g, p): [m_txt[i] for i in range(len(mine)) if m_gold[i] == g and pred[i] == p] for (g, p) in pairs}
    contrast_by = {(g, p): [m_txt[i] for i in range(len(mine)) if m_gold[i] == p and pred[i] == p] for (g, p) in pairs}
    diags = diagnose_batch(T, pairs, exs_by, contrast_by, D)
    def vacc(DD):
        vp = _desc_classify(T, v_txt, DD)
        return float(np.mean([vp[i] == v_gold[i] for i in range(len(v_gold))]))
    D = copy.deepcopy(D); D.setdefault("__disamb__", [])
    cur = vacc(D); kept = 0
    for (g, p) in pairs:
        dg = diags.get((g, p), {})
        line = f"{g} vs {p}: decide by {dg.get('dimension','the key difference')} -- {g} if {dg.get('gt_signal','')}; {p} if {dg.get('pr_signal','')}"
        if len(line) > 240: line = line[:240]
        trial = copy.deepcopy(D); trial["__disamb__"] = D["__disamb__"] + [line]
        v = vacc(trial)
        if v >= cur:
            D["__disamb__"].append(line); cur = v; kept += 1
    print(f"  [disambig] kept {kept}/{len(pairs)} tie-breakers, val={cur:.4f}")
    return D


def main():
    task = sys.argv[3] if len(sys.argv) > 3 else "bloom"
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    T = TASKS[task]
    semclf.set_trace(f"results/train_trace_{task}_{b}.jsonl")
    bud = stratified_budget(T.pool, b, seed=0)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    nm = int(0.7 * b); mine, val = bud[:nm], bud[nm:]
    test = T.test + T.test_dup
    txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    print(f"TRAIN b={b} epochs={epochs}: train={len(mine)} val={len(val)} test={len(test)}")

    D, vfin = train(T, mine, val, epochs=epochs)
    # IDEA 1: post-training pairwise tie-breakers (unless DISAMB_OFF=1)
    a_before = None
    if os.environ.get("DISAMB_OFF") != "1":
        preds0 = _desc_classify(T, txt, D); a_before, _, _ = score(T, preds0, gold)
        D = add_disambiguation(T, D, mine, val)
    json.dump({"D": D, "val": vfin}, open(f"results/train_art_{task}_{b}.json", "w"), indent=2, ensure_ascii=False)
    # SINGLE-PASS: the richer rulebook (pos + REMAP overrides + tie-breakers) IS the whole method.
    preds = _desc_classify(T, txt, D); a0, ci0, _ = score(T, preds, gold)
    zs = semclf.zero_shot(T, txt); azs, _, _ = score(T, zs, gold)
    ptz = paired_test(preds, zs, gold)
    rag = semclf.lexical_rag(T, txt); ar, _, _ = score(T, rag, gold); ptr = paired_test(rag, preds, gold)
    print(f"\n=== RESULTS (test n={len(test)}, single-pass rulebook w/ remaps+tie-breakers) ===")
    print(f"  zero-shot           {azs:.4f}")
    if a_before is not None: print(f"  rulebook (no disambig) {a_before:.4f}")
    print(f"  rulebook FINAL      {a0:.4f} CI=({ci0[0]:.3f},{ci0[1]:.3f})  vs zero-shot {ptz['delta']:+.4f} p={ptz['p_mcnemar']:.1e}")
    print(f"  RAG {ar:.4f}  gap={ar-a0:+.4f} p={ptr['p_mcnemar']:.3f} RAG_better={ptr['significant']}")
    json.dump({"budget": b, "epochs": epochs, "zero_shot": azs, "rulebook_no_disambig": a_before,
               "rulebook": a0, "rag": ar, "vs_zs": ptz, "rag_vs_ours": ptr},
              open(f"results/train_{task}_{b}.json", "w"), indent=2)
    print(f"wrote results/train_{task}_{b}.json")


if __name__ == "__main__":
    main()
