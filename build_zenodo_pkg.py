"""Build a curated Zenodo package for MC311 into ../zenodo_build/ (data/code/results zips + manifest)."""
import os, glob, hashlib, zipfile
REPO = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(os.path.dirname(REPO), "zenodo_build")
os.makedirs(BUILD, exist_ok=True)

DATA = ["data/mc311_harmonized.csv", "data/harmonized.csv", "data/harmonized_filtered.csv",
        "data/eval_split.csv", "data/eval_split_dedup.csv", "data/harmonization_map.json",
        "data/native_categories.json", "data/harmonization_pivot.csv"]
CODE = sorted(glob.glob(os.path.join(REPO, "*.py"))) + \
       [os.path.join(REPO, f) for f in ["requirements.txt", "requirements_enc.txt", "requirements_llm.txt",
                                        "README.md", "EXTERNAL_DATA.md"]]
RESULTS = sorted(glob.glob(os.path.join(REPO, "results", "*.json"))) + \
          sorted(glob.glob(os.path.join(REPO, "results", "*.csv"))) + \
          sorted(glob.glob(os.path.join(REPO, "results", "runs.jsonl"))) + \
          sorted(glob.glob(os.path.join(REPO, "results", "preds", "*.json")))


def zipit(name, files, arc):
    p = os.path.join(BUILD, name)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            f = f if os.path.isabs(f) else os.path.join(REPO, f)
            if os.path.exists(f):
                z.write(f, os.path.join(arc, os.path.relpath(f, REPO)))
    return p


def main():
    paper = [os.path.join(REPO, "docs", f) for f in
             ["paper.html", "paper.docx", "datasheet.html", "fig_transfer.png", "fig_percity.png",
              "fig_defensibility.png", "fig_confusion.png", "fig_transfer_matrix.png",
              "fig_corpus_flow.png", "fig_taxonomy_boundary.png"]]
    made = [zipit("mc311_data.zip", DATA, "mc311"),
            zipit("mc311_code.zip", CODE, "mc311"),
            zipit("mc311_results.zip", RESULTS, "mc311"),
            zipit("mc311_paper.zip", paper, "mc311/paper")]
    lines = []; sha = []
    for p in made:
        sz = os.path.getsize(p)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        lines.append(f"{os.path.basename(p):24s} {sz:>12,} bytes")
        sha.append(f"{h}  {os.path.basename(p)}")
    open(os.path.join(BUILD, "MANIFEST.txt"), "w").write(
        "MC311 Zenodo package\n\n" + "\n".join(lines) + "\n")
    open(os.path.join(BUILD, "SHA256SUMS.txt"), "w").write("\n".join(sha) + "\n")
    print("built into", BUILD)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
