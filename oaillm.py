"""gpt-4o-mini / gpt-4.1 threaded chat helper with a persistent response cache.

Efficiency measures:
  * CACHE: every (model, messages, max_tokens, temperature) response is stored in SQLite and reused,
    so reruns and cross-script duplicates (e.g. zero-shot on the fixed test set) cost nothing.
  * chat_many only issues API calls for cache MISSES, preserving input order.
  * Prompt layout is the caller's job: put the static block first and the item last so the vendor's
    automatic prompt caching applies to the shared prefix.
  * Reports cache hit/miss per batch so we can see what an experiment actually cost.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client
import llmcache

MODEL = "gpt-4o-mini"
_stats = {"hit": 0, "miss": 0}


def _api(messages, model, max_tokens, temperature):
    for a in range(4):
        try:
            r = client.chat.completions.create(model=model, temperature=temperature,
                                               max_tokens=max_tokens, messages=messages)
            return (r.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(1.5 * (a + 1))
    return ""


def _call(messages, model=MODEL, max_tokens=20, temperature=0):
    k = llmcache.key(model, messages, max_tokens, temperature)
    hit = llmcache.get(k)
    if hit is not None:
        _stats["hit"] += 1
        return hit
    out = _api(messages, model, max_tokens, temperature)
    _stats["miss"] += 1
    if out: llmcache.put(k, model, out)
    return out


def chat_many(msg_list, model=MODEL, max_tokens=20, workers=16, temperature=0, verbose=False):
    """Cache-aware batch: only misses hit the API; order preserved."""
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
    if todo:
        def one(i): return _api(msg_list[i], model, max_tokens, temperature)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, i): i for i in todo}
            for f in as_completed(futs):
                i = futs[f]; out[i] = f.result()
                if out[i]: llmcache.put(keys[i], model, out[i])
    return out


def cache_stats():
    return dict(_stats)
