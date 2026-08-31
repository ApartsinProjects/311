"""Robust few-shot v2 batch: submit-then-collect (no long-lived poller, survives process reaping).
  python fewshot_v2_run.py submit    # builds prompts, creates batch, saves state, exits
  python fewshot_v2_run.py collect   # checks batch; if done, downloads + writes preds; else prints status
"""
import sys, os, json, io
import eval_fewshot_batch as E
from openai_batch import _load_key
from openai import OpenAI

MODEL = "gpt-4o-mini"; TAG = "gpt4omini_fewshot_v2"
STATE = "fewshot_v2_state.json"
PRED = os.path.join("results", "preds")


def build():
    tr, te = E.load_split()
    ex = E.pick_exemplars(tr)
    cats = "\n".join(f"- {k}: {E.GLOSS[k]}" for k in E.LABELS)
    shots = "\n".join(f'Request: "{t}" -> {lab}' for lab, t in ex.items())
    order = []
    bodies = []
    for city, rows in te.items():
        for i, (t, _) in enumerate(rows):
            cid = f"{city}|{i}"; order.append(cid)
            user = (f"Categories:\n{cats}\n\nExamples:\n{shots}\n\n"
                    f"Now classify this request. Reply with ONLY the category name.\nRequest: \"{t[:600]}\" ->")
            bodies.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                           "body": {"model": MODEL, "messages": [{"role": "system", "content": E.SYS},
                                    {"role": "user", "content": user}], "temperature": 0, "max_tokens": 12}})
    return order, bodies, te


def submit():
    order, bodies, _ = build()
    client = OpenAI(api_key=_load_key())
    buf = io.BytesIO("\n".join(json.dumps(b) for b in bodies).encode())
    f = client.files.create(file=("batch.jsonl", buf), purpose="batch")
    batch = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                                  completion_window="24h")
    json.dump({"batch_id": batch.id, "order": order}, open(STATE, "w"))
    print(f"submitted batch {batch.id} status={batch.status} items={len(order)}")


def collect():
    st = json.load(open(STATE))
    client = OpenAI(api_key=_load_key())
    batch = client.batches.retrieve(st["batch_id"])
    print(f"batch {batch.id} status={batch.status} counts={batch.request_counts}")
    if batch.status != "completed":
        return
    text = client.files.content(batch.output_file_id).text
    res = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        cid = o["custom_id"]
        try:
            res[cid] = o["response"]["body"]["choices"][0]["message"]["content"]
        except Exception:
            res[cid] = ""

    def parse(s):
        low = (s or "").lower()
        for l in E.LABELS:
            if l.lower() in low:
                return l
        return "UNPARSED"
    _, _, te = build()
    out = {"zeroshot": {}}
    for city, rows in te.items():
        out["zeroshot"][city] = [parse(res.get(f"{city}|{i}", "")) for i in range(len(rows))]
    os.makedirs(PRED, exist_ok=True)
    json.dump(out, open(os.path.join(PRED, f"llm_{TAG}.json"), "w"))
    print(f"wrote results/preds/llm_{TAG}.json")


if __name__ == "__main__":
    {"submit": submit, "collect": collect}[sys.argv[1]]()
