# HUPD (patent IPC) is the wrong testbed: zero-shot LLM already wins (2026-09-02)

Method-comparison pilot on HUPD titles -> IPC subclass (539-way), low-data regime. Goal was a
many-label + non-semantic task where instruction mining beats few-shot/fine-tune. Result FALSIFIED
the premise for HUPD.

| method | data used | acc |
|---|---|---|
| zero-shot LLM (gpt-4o-mini) | 0 | 0.488 |
| RAG-few-shot (k=12) | 2000 | 0.479 |
| fine-tuned TF-IDF+LR | 24768 | 0.463 |
| fine-tuned DistilBERT (Modal) | 24768 | 0.465 |
| fine-tuned TF-IDF+LR | 2000 | 0.270 |

Zero-shot LLM beats fine-tuning-on-everything AND RAG, with no labels. IPC is a widely-documented
public taxonomy the LLM has memorized -> the "codebook" is already in its weights -> instruction mining
has nothing to add. RAG slightly HURTS vs zero-shot.

## Lesson (sharpens dataset criterion 2)
Any well-known PUBLIC taxonomy (IPC, ICD, NAICS) fails "labels are non-semantic to the LLM" because the
LLM already knows it. Criterion 2 requires a codebook the LLM has NOT memorized:
- proprietary/anonymized codes (Rakuten 3008 anon IDs) -> LLM ~0, but arbitrary -> favors RAG/kNN;
- org-specific rule-governed conventions the LLM cannot know (our 311 per-city result) -> our mechanism,
  but 311 harmonized has only 14 labels. 311 NATIVE categories (27-112/city) may combine many-label +
  org-specific, but category names are semantic strings (LLM can guess given the list).

DistilBERT did NOT beat TF-IDF+LR on short titles (transformers need more signal than a title gives);
with abstracts BERT would likely separate, but titles favor simple models.

Data: data/hupd_title.json (26768 titles, 539 IPC subclasses, from HUPD metadata feather).
Code: hupd_compare.py, modal_hupd_bert_train.py. Results: results/hupd_compare.json, results/hupd_bert.json.
