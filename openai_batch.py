"""
openai_batch.py -- OpenAI native Batch API client (50% off; supports gpt-4o-mini, unlike OpenRouter).
Flow: build JSONL -> upload to /v1/files (purpose=batch) -> create /v1/batches -> poll -> download output.
Requires a FUNDED OpenAI key (OPENAI_API_KEY). Uses the openai SDK.

  from openai_batch import run_chat_batch
  preds = run_chat_batch("gpt-4o-mini", [(cid, body), ...])   # body = chat completion body dict
"""
import io, json, time
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY


def run_chat_batch(model, items, interval=20, verbose=True):
    """items: list of (custom_id, body). body is a /v1/chat/completions body (may omit 'model').
    Returns {custom_id: assistant_text}."""
    lines = []
    for cid, body in items:
        b = dict(body)
        b.setdefault("model", model)
        lines.append(json.dumps({"custom_id": str(cid), "method": "POST",
                                 "url": "/v1/chat/completions", "body": b}))
    buf = io.BytesIO(("\n".join(lines)).encode("utf-8"))
    f = client.files.create(file=("batch.jsonl", buf), purpose="batch")
    if verbose: print(f"  uploaded file {f.id} ({len(lines)} requests)")
    batch = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                                  completion_window="24h")
    if verbose: print(f"  created batch {batch.id} status={batch.status}")
    t0 = time.time()
    while True:
        batch = client.batches.retrieve(batch.id)
        if verbose:
            print(f"  [batch {batch.id[:16]}] status={batch.status} counts={batch.request_counts} ({int(time.time()-t0)}s)")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(interval)
    if batch.status != "completed":
        raise RuntimeError(f"batch {batch.status}: {getattr(batch,'errors',None)}")
    content = client.files.content(batch.output_file_id).text
    out = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec.get("custom_id")
        try:
            out[cid] = rec["response"]["body"]["choices"][0]["message"]["content"]
        except Exception:
            out[cid] = f"ERR:{str(rec.get('error'))[:60]}"
    return out
