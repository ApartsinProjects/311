"""DSPy MIPROv2 baseline -- the state-of-the-art prompt optimizer (jointly optimizes INSTRUCTIONS +
DEMONSTRATIONS via Bayesian search). This is the strongest prompt-tuning baseline and the key related
work to distinguish from: our method mines per-class RULES only, no demos, single pass.

Fair model split matching our setup:
  task/inference model = gpt-4o-mini (cheap; same as every other method's inference)
  prompt/proposer model = gpt-4.1     (the optimizer; analog of our MINE_MODEL)

  python miprov2.py [task] [auto]      # auto in {light, medium, heavy}; 'smoke' = tiny fast check
"""
import sys, os, json
import numpy as np
from collections import defaultdict
import dspy
import semclf
from semclf import TASKS, score, paired_test, stratified_budget
from openai_batch import _load_key

os.environ["OPENAI_API_KEY"] = _load_key()   # FORCE the .env key (a stale env key was shadowing it)


def run(task="bloom", auto="light", smoke=False):
    T = TASKS[task]
    bud = stratified_budget(T.pool, 2000, seed=0)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    LBL = T.LBL
    task_lm = dspy.LM("openai/gpt-4o-mini", temperature=0, max_tokens=24, cache=True, num_retries=8)
    prompt_lm = dspy.LM("openai/gpt-4.1", temperature=0.7, max_tokens=2000, cache=True, num_retries=8)
    dspy.configure(lm=task_lm)

    class Classify(dspy.Signature):
        """Classify the item into exactly one of the allowed categories."""
        text: str = dspy.InputField()
        category: str = dspy.OutputField(desc="one of: " + ", ".join(LBL))

    program = dspy.Predict(Classify)

    def metric(example, pred, trace=None):
        return 1.0 if T.parse(getattr(pred, "category", "")) == example.label else 0.0

    def ex(r): return dspy.Example(text=r["text"][:500], label=r["label"]).with_inputs("text")
    n_tr = 60 if smoke else 1400
    n_va = 40 if smoke else 600
    trainset = [ex(r) for r in bud[:n_tr]]
    valset = [ex(r) for r in bud[n_tr:n_tr + n_va]]
    print(f"MIPROv2 {task}: train={len(trainset)} val={len(valset)} classes={len(LBL)} auto={auto}")

    tp = dspy.MIPROv2(metric=metric, prompt_model=prompt_lm, task_model=task_lm,
                      auto=("light" if smoke else auto), num_threads=3)
    kwargs = dict(requires_permission_to_run=False, max_bootstrapped_demos=8, max_labeled_demos=8)
    if smoke:
        kwargs.update(max_bootstrapped_demos=2, max_labeled_demos=2)
    compiled = tp.compile(program, trainset=trainset, valset=valset, **kwargs)

    # evaluate on the fixed test
    test = (T.test + T.test_dup)[:100] if smoke else (T.test + T.test_dup)
    t_txt = [r["text"] for r in test]; t_gold = [r["label"] for r in test]
    preds = []
    for t in t_txt:
        try: preds.append(T.parse(getattr(compiled(text=t[:500]), "category", "")))
        except Exception: preds.append("UNPARSED")
    a, ci, unp = score(T, preds, t_gold)
    zs = semclf.zero_shot(T, t_txt); azs, _, _ = score(T, zs, t_gold)
    ptz = paired_test(preds, zs, t_gold)
    print(f"\n=== MIPROv2 RESULTS ({task}, {'SMOKE' if smoke else auto}) ===")
    print(f"  zero-shot   {azs:.4f}")
    print(f"  MIPROv2     {a:.4f} CI=({ci[0]:.3f},{ci[1]:.3f}) UNPARSED={unp:.3f}  vs zero-shot {ptz['delta']:+.4f} p={ptz['p_mcnemar']:.1e}")
    if not smoke:
        json.dump({"task": task, "auto": auto, "zero_shot": azs, "miprov2": a, "vs_zs": ptz},
                  open(f"results/miprov2_{task}.json", "w"), indent=2)
        print(f"wrote results/miprov2_{task}.json")
    return a


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "bloom"
    mode = sys.argv[2] if len(sys.argv) > 2 else "light"
    run(task, auto=("light" if mode == "smoke" else mode), smoke=(mode == "smoke"))
