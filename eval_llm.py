"""Aligned LLM eval: zero-shot on the FROZEN test rows (same rows as every arm). Saves preds."""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from eval_common import load_split, save_preds, LABELS
from llm_arm import make_client, classify_one


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-4o-mini"
    tag = sys.argv[2] if len(sys.argv) > 2 else model.split("/")[-1].replace(".", "")
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    client, prov = make_client(r"E:\Projects\.env.all")
    sp = load_split()
    print(f"[llm] provider={prov} model={model}")
    out = {"zeroshot": {}}
    for c, rows in sp["test"].items():
        texts = [t for t, _ in rows]
        preds = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(classify_one, client, texts[i], model): i for i in range(len(texts))}
            for f in as_completed(futs):
                preds[futs[f]] = f.result()
        out["zeroshot"][c] = [p if p in LABELS else "UNPARSED" for p in preds]
        print(f"[llm] {c} done ({len(texts)} rows)")
    save_preds(f"llm_{tag}", out)


if __name__ == "__main__":
    main()
