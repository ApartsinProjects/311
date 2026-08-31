"""
openai_batch.py -- OpenAI native Batch API client (50% off; supports gpt-4o-mini, unlike OpenRouter).
Flow: build JSONL -> upload to /v1/files (purpose=batch) -> create /v1/batches -> poll -> download.
Requires a FUNDED OpenAI key (OPENAI_API_KEY). Uses the openai SDK.

Robust submit/collect pattern (survives the host process-guard reaping a long poll): the batch id
is persisted to a state file at SUBMIT time, so a killed run is resumed with a short collect step.

  from openai_batch import submit_chat_batch, collect_chat_batch, run_chat_batch
  submit_chat_batch("gpt-4o-mini", [(cid, body), ...], tag="myrun")   # creates batch, saves state, returns fast
  res = collect_chat_batch(tag="myrun")   # None if still running; {cid: text} when done
  res = run_chat_batch("gpt-4o-mini", items, tag="myrun")             # submit + bounded poll (None on timeout)
"""
import io, json, time, os, sys
from openai import OpenAI


def _load_key():
    """Prefer the project .env key (funded) over the stale environment OPENAI_API_KEY."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, ".env"), ".env"):
        try:
            for line in open(p, encoding="utf-8"):
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return os.environ.get("OPENAI_API_KEY")


client = OpenAI(api_key=_load_key())


def _state_path(tag):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f".batch_state_{tag}.json")


def submit_chat_batch(model, items, tag="default", verbose=True):
    """items: list of (custom_id, body). Uploads + creates the batch, writes the state file, returns id."""
    lines = []
    for cid, body in items:
        b = dict(body); b.setdefault("model", model)
        lines.append(json.dumps({"custom_id": str(cid), "method": "POST",
                                 "url": "/v1/chat/completions", "body": b}))
    buf = io.BytesIO(("\n".join(lines)).encode("utf-8"))
    f = client.files.create(file=("batch.jsonl", buf), purpose="batch")
    batch = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                                  completion_window="24h")
    json.dump({"provider": "openai", "model": model, "batch_id": batch.id, "n": len(lines)},
              open(_state_path(tag), "w"))
    if verbose:
        print(f"  submitted batch {batch.id} status={batch.status} ({len(lines)} requests) -> {_state_path(tag)}")
    return batch.id


def _download(batch):
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


def collect_chat_batch(tag="default", verbose=True):
    """One retrieve. Returns {cid: text} when completed, None while still running, raises if failed."""
    st = json.load(open(_state_path(tag)))
    batch = client.batches.retrieve(st["batch_id"])
    if verbose:
        print(f"  [batch {batch.id[:16]}] status={batch.status} counts={batch.request_counts}")
    if batch.status == "completed":
        return _download(batch)
    if batch.status in ("failed", "expired", "cancelled"):
        raise RuntimeError(f"batch {batch.status}: {getattr(batch, 'errors', None)}")
    return None


def run_chat_batch(model, items, interval=20, verbose=True, tag="default", max_wait=1800):
    """Submit then bounded-poll. Returns {cid: text} on completion, or None on timeout (state is
    saved, so resume with collect_chat_batch(tag=...) rather than resubmitting)."""
    submit_chat_batch(model, items, tag=tag, verbose=verbose)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        res = collect_chat_batch(tag=tag, verbose=verbose)
        if res is not None:
            return res
        time.sleep(interval)
    print(f"  batch still running after {max_wait}s; state saved. Resume: "
          f"python openai_batch.py collect {tag}")
    return None


if __name__ == "__main__":  # CLI: python openai_batch.py collect <tag>
    if len(sys.argv) >= 3 and sys.argv[1] == "collect":
        r = collect_chat_batch(tag=sys.argv[2])
        print("not ready" if r is None else f"collected {len(r)} results")
