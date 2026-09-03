"""Persistent, process-safe LLM response cache (SQLite).

Why: the same calls recur constantly across scripts and reruns (zero-shot on the fixed test set, the
same mined prompt re-evaluated, a method rerun after an unrelated bug fix). Caching them makes reruns
free and keeps every experiment reproducible from stored responses.

Key = sha256(model, messages, max_tokens, temperature). Deterministic decoding (temperature=0) makes
reuse sound. Set LLM_CACHE_OFF=1 to bypass.
"""
import os, json, sqlite3, hashlib, threading

DB = os.environ.get("LLM_CACHE_DB", "results/llm_cache.sqlite")
OFF = os.environ.get("LLM_CACHE_OFF", "") == "1"
_lock = threading.Lock()
_conn = None


def _db():
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
        _conn = sqlite3.connect(DB, check_same_thread=False, timeout=30)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, model TEXT, resp TEXT)")
        _conn.commit()
    return _conn


def key(model, messages, max_tokens, temperature):
    blob = json.dumps({"m": model, "msgs": messages, "mt": max_tokens, "t": temperature},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(k):
    if OFF: return None
    with _lock:
        r = _db().execute("SELECT resp FROM c WHERE k=?", (k,)).fetchone()
    return r[0] if r else None


def put(k, model, resp):
    if OFF or resp is None: return
    with _lock:
        _db().execute("INSERT OR REPLACE INTO c VALUES (?,?,?)", (k, model, resp))
        _db().commit()


def stats():
    with _lock:
        n = _db().execute("SELECT COUNT(*) FROM c").fetchone()[0]
        by = _db().execute("SELECT model, COUNT(*) FROM c GROUP BY model").fetchall()
    return n, dict(by)


if __name__ == "__main__":
    n, by = stats()
    print(f"cached responses: {n}")
    for m, c in by.items(): print(f"  {m}: {c}")
