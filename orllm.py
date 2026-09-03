"""Shared OpenRouter LLM helper (single model for the no-train classification study, since OpenAI
credits are exhausted). Threaded chat, reasoning disabled, plain-body fallback."""
import json, ssl, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from or_batch import _key

MODEL = "google/gemini-2.5-flash"
_ctx = ssl.create_default_context(); _ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE
_KEY = _key(r"E:\Projects\.env.all")
_URL = "https://openrouter.ai/api/v1/chat/completions"


def _call(messages, model=MODEL, max_tokens=20, temperature=0):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "reasoning": {"enabled": False}, "messages": messages}
    for a in range(4):
        try:
            req = urllib.request.Request(_URL, data=json.dumps(body).encode(), method="POST",
                headers={"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json"})
            o = json.loads(urllib.request.urlopen(req, timeout=90, context=_ctx).read())
            return (o["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            time.sleep(1.5 * (a + 1))
    return ""


def chat_many(msg_list, model=MODEL, max_tokens=20, workers=12):
    """msg_list: list of `messages` arrays. Returns list of response strings (order preserved)."""
    out = [None] * len(msg_list)
    def one(i):
        return _call(msg_list[i], model=model, max_tokens=max_tokens)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, i): i for i in range(len(msg_list))}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    return out


if __name__ == "__main__":
    print(_call([{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=5))
