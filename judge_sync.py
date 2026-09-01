"""Synchronous acceptable-set judge via OpenRouter chat/completions (bypasses the flaky batch
writer). Thread-pooled, reasoning disabled, per-row retry. For a reliable different-vendor 3rd judge.
  python judge_sync.py z-ai/glm-5.3-flash glm
Writes results/preds/acceptable_sets_<tag>.json aligned to the frozen test rows.
"""
import sys, os, json, ssl, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from eval_common import load_split, PRED_DIR
from defensibility_judge import SYS, prompt, parse_set
from or_batch import _key

URL = "https://openrouter.ai/api/v1/chat/completions"
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
MODEL = sys.argv[1] if len(sys.argv) > 1 else "z-ai/glm-5.3-flash"
TAG = sys.argv[2] if len(sys.argv) > 2 else "glm"
KEY = _key(r"E:\Projects\.env.all")


def _post(body):
    r = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                               headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=90, context=CTX) as resp:
        o = json.loads(resp.read().decode("utf-8", "replace"))
    c = o["choices"][0]["message"]["content"]
    if not c:
        raise ValueError("empty content")
    return c


def judge_one(text):
    base = {"model": MODEL, "temperature": 0, "max_tokens": 160,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt(text)}]}
    for a in range(4):
        # try with reasoning disabled first (helps reasoning models); fall back to plain body on any error
        for body in ({**base, "reasoning": {"enabled": False}}, base):
            try:
                return parse_set(_post(body))
            except Exception:
                continue
        time.sleep(2 * (a + 1))
    return ["ERR"]


def main():
    sp = load_split()
    out = {}
    for c, rows in sp["test"].items():
        texts = [t for t, _ in rows]; sets = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=24) as ex:
            futs = {ex.submit(judge_one, texts[i]): i for i in range(len(texts))}
            for f in as_completed(futs):
                sets[futs[f]] = f.result()
        out[c] = sets
        gold = [y for _, y in rows]
        noise = sum(1 for g, s in zip(gold, sets) if g not in s) / len(gold)
        err = sum(1 for s in sets if s == ["ERR"])
        print(f"[sync-judge] {c}: noise={noise:.1%} err={err}"); sys.stdout.flush()
    os.makedirs(PRED_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(PRED_DIR, f"acceptable_sets_{TAG}.json"), "w"), ensure_ascii=False)
    print(f"[sync-judge] wrote {PRED_DIR}/acceptable_sets_{TAG}.json (model={MODEL})")


if __name__ == "__main__":
    main()
