"""
or_batch.py -- OpenRouter Batch API client (50% off standard per-token pricing).
Submits an inline requests array to POST /api/beta/batches and polls GET /api/beta/batches/:id
until terminal. Results map back to inputs by custom_id. Used by the LLM eval/judge arms so
all bulk classification runs at the batch discount.

Optional prompt caching: pass cache_prefix messages with provider cache_control for models that
support it (Anthropic/Gemini); ignored by models that don't. For gpt-4o-mini the ~400-token
prompt is below the auto-cache threshold, so batch (50%) is the effective discount there.
"""
import json, time, ssl, os, urllib.request, urllib.error

BASE = "https://openrouter.ai/api/beta/batches"
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE


def load_key_from_env_file(path, name="OPENROUTER_API_KEY"):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        return None
    return None


def _key(env_file):
    return os.environ.get("OPENROUTER_API_KEY") or load_key_from_env_file(env_file)


def _req(url, key, method="GET", payload=None, tries=5):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Authorization": f"Bearer {key}",
                                        "Content-Type": "application/json"})
    for a in range(tries):
        try:
            with urllib.request.urlopen(r, timeout=120, context=CTX) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            body = ""
            try: body = e.read().decode("utf-8", "replace")
            except Exception: pass
            raise RuntimeError(f"HTTP {e.code} on {method} {url}: {body[:500]}") from None
        except Exception as e:  # transient network blip (timeout, conn reset) -> retry
            if a == tries - 1:
                raise RuntimeError(f"network error on {method} {url}: {type(e).__name__}: {e}") from None
            time.sleep(5 * (a + 1))


def submit(model, requests, env_file=r"E:\Projects\.env.all", endpoint="/v1/chat/completions"):
    """requests: list of {custom_id, body}. Returns batch id."""
    key = _key(env_file)
    out = _req(BASE, key, "POST", {"endpoint": endpoint, "model": model, "requests": requests})
    bid = out.get("id") or out.get("batch", {}).get("id")
    if not bid:
        raise RuntimeError(f"no batch id in response: {str(out)[:300]}")
    return bid, key


def poll(bid, key, interval=15, max_wait=86400, verbose=True):
    """Poll until terminal. Returns the list of result items. Tolerates initial 404 (registration lag)."""
    t0 = time.time()
    time.sleep(5)  # let the batch register before first poll
    while True:
        try:
            out = _req(f"{BASE}/{bid}", key)
        except RuntimeError as e:
            transient = ("network error" in str(e)) or ("404" in str(e) and time.time() - t0 < 120)
            if transient and time.time() - t0 < max_wait:
                if verbose: print(f"  [batch {bid[:16]}] poll retry ({str(e)[:50]})")
                time.sleep(interval); continue
            raise
        status = out.get("status") or out.get("batch", {}).get("status")
        counts = out.get("request_counts") or out.get("counts") or {}
        if verbose:
            print(f"  [batch {bid[:12]}] status={status} counts={counts} ({int(time.time()-t0)}s)")
        if status in ("completed", "failed", "expired", "cancelled"):
            results = out.get("results") or out.get("output") or []
            if status != "completed":
                raise RuntimeError(f"batch {status}: {str(out)[:300]}")
            return results
        if time.time() - t0 > max_wait:
            raise TimeoutError(f"batch {bid} not done after {max_wait}s")
        time.sleep(interval)


def _state_path(tag):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f".batch_state_{tag}.json")


def _parse_results(results):
    out = {}
    for r in results:
        cid = r.get("custom_id"); resp = r.get("response")
        if resp is None:
            out[cid] = f"ERR:{str(r.get('error'))[:60]}"; continue
        body = resp.get("body", resp)
        try:
            out[cid] = body["choices"][0]["message"]["content"]
        except Exception:
            out[cid] = f"ERR:parse:{str(body)[:60]}"
    return out


def submit_chat_batch(model, items, build_body, tag="default", env_file=r"E:\Projects\.env.all",
                      endpoint="/v1/chat/completions", verbose=True):
    """Create the batch, persist its id, return fast (survives a reaped poller)."""
    reqs = [{"custom_id": str(cid), "body": build_body(p)} for cid, p in items]
    bid, _ = submit(model, reqs, env_file, endpoint)
    json.dump({"provider": "openrouter", "model": model, "batch_id": bid, "env_file": env_file, "n": len(reqs)},
              open(_state_path(tag), "w"))
    if verbose:
        print(f"  submitted batch {bid} ({len(reqs)} requests, model={model}) -> {_state_path(tag)}")
    return bid


def collect_chat_batch(tag="default", verbose=True):
    """One retrieve. {cid: text} when completed, None while running, raises if failed."""
    st = json.load(open(_state_path(tag)))
    key = _key(st.get("env_file", r"E:\Projects\.env.all"))
    out = _req(f"{BASE}/{st['batch_id']}", key)
    status = out.get("status") or out.get("batch", {}).get("status")
    if verbose:
        print(f"  [batch {st['batch_id'][:16]}] status={status} counts={out.get('request_counts') or out.get('counts')}")
    if status in ("failed", "expired", "cancelled"):
        raise RuntimeError(f"batch {status}: {str(out)[:300]}")
    if status == "completed":
        return _parse_results(out.get("results") or out.get("output") or [])
    return None


def run_chat_batch(model, items, build_body, env_file=r"E:\Projects\.env.all", interval=15,
                   tag="default", max_wait=1800):
    """Submit then bounded-poll. Returns {cid: text} on completion, or None on timeout (state saved,
    resume with collect_chat_batch(tag=...))."""
    submit_chat_batch(model, items, build_body, tag=tag, env_file=env_file, verbose=True)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        res = collect_chat_batch(tag=tag)
        if res is not None:
            return res
        time.sleep(interval)
    print(f"  batch still running after {max_wait}s; state saved. Resume: python or_batch.py collect {tag}")
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "collect":
        r = collect_chat_batch(tag=sys.argv[2])
        print("not ready" if r is None else f"collected {len(r)} results")
