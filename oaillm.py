"""gpt-4o-mini / gpt-4.1 threaded chat helper with a persistent response cache.

Efficiency measures:
  * CACHE: every (model, messages, max_tokens, temperature) response is stored in SQLite and reused,
    so reruns and cross-script duplicates (e.g. zero-shot on the fixed test set) cost nothing.
  * chat_many only issues API calls for cache MISSES, preserving input order.
  * Prompt layout is the caller's job: put the static block first and the item last so the vendor's
    automatic prompt caching applies to the shared prefix.
  * Reports cache hit/miss per batch so we can see what an experiment actually cost.
"""
import os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client, run_chat_batch
import llmcache

MODEL = "gpt-4o-mini"
_stats = {"hit": 0, "miss": 0}
SYNC = os.environ.get("LLM_SYNC", "") == "1"   # LLM_SYNC=1 forces the threaded path (debug/smoke)
_batch_seq = 0


def _body(messages, model, max_tokens, temperature):
    """Per-request body for the Batch API, with the same reasoning-model handling as _api."""
    b = {"messages": messages}
    if _is_reasoning(model):
        b["max_completion_tokens"] = max_tokens + 6000
    else:
        b["max_tokens"] = max_tokens; b["temperature"] = temperature
    return b


def _is_reasoning(model):
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _api(messages, model, max_tokens, temperature):
    reasoning = _is_reasoning(model)
    for a in range(4):
        try:
            kw = dict(model=model, messages=messages, timeout=180)
            if reasoning:
                # reasoning models reject temperature and use max_completion_tokens; the budget must
                # cover internal reasoning tokens FIRST, so give large headroom over the visible cap or
                # the JSON gets truncated to empty. Scale headroom with the requested output size.
                kw["max_completion_tokens"] = max_tokens + 6000
            else:
                kw["temperature"] = temperature
                kw["max_tokens"] = max_tokens
            r = client.chat.completions.create(**kw)
            return (r.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(1.5 * (a + 1))
    return ""


def _call(messages, model=MODEL, max_tokens=20, temperature=0):
    """Single call routed through the SAME cache+batch path as chat_many, so the pipeline never issues
    a raw per-request API call. A cache hit returns instantly; a miss becomes a 1-item batch (or a
    threaded call under LLM_SYNC=1). Prefer batching many prompts via chat_many -- a 1-item batch pays
    full batch latency for one result."""
    return chat_many([messages], model=model, max_tokens=max_tokens, temperature=temperature)[0]


def chat_many(msg_list, model=MODEL, max_tokens=20, workers=16, temperature=0, verbose=False):
    """Cache-aware. Default path submits cache-MISSES as ONE OpenAI async Batch job (50% cheaper) and
    blocking-polls to completion; LLM_SYNC=1 uses the threaded path instead. Order preserved."""
    global _batch_seq
    out = [None] * len(msg_list)
    keys = [llmcache.key(model, m, max_tokens, temperature) for m in msg_list]
    todo = []
    for i, k in enumerate(keys):
        c = llmcache.get(k)
        if c is not None: out[i] = c
        else: todo.append(i)
    _stats["hit"] += len(msg_list) - len(todo); _stats["miss"] += len(todo)
    if verbose:
        print(f"    [cache] {len(msg_list)-len(todo)} hit / {len(todo)} miss")
    if not todo:
        return out

    if SYNC:
        def one(i): return _api(msg_list[i], model, max_tokens, temperature)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, i): i for i in todo}
            for f in as_completed(futs):
                i = futs[f]; out[i] = f.result()
                if out[i]: llmcache.put(keys[i], model, out[i])
        return out

    # ---- OpenAI async Batch API path (the standing rule) ----
    _batch_seq += 1
    tag = f"cm_{_batch_seq}"
    items = [(str(i), _body(msg_list[i], model, max_tokens, temperature)) for i in todo]
    res = run_chat_batch(model, items, tag=tag, verbose=verbose, interval=20, max_wait=7200)
    if res is None:
        # batch did not finish within the wait window; state saved. Fall back to threaded for the
        # misses so the run still completes (results still get cached).
        if verbose: print(f"    [batch {tag}] not finished in window -> threaded fallback")
        def one(i): return _api(msg_list[i], model, max_tokens, temperature)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, i): i for i in todo}
            for f in as_completed(futs):
                i = futs[f]; out[i] = f.result()
                if out[i]: llmcache.put(keys[i], model, out[i])
        return out
    for i in todo:
        o = (res.get(str(i)) or "").strip()
        if o.startswith("ERR:"): o = ""
        out[i] = o
        if o: llmcache.put(keys[i], model, o)
    return out


def cache_stats():
    return dict(_stats)
