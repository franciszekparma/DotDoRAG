# SimpleRAG

<!-- TODO: hero image goes here -->
<p align="center">
  <!-- Drop a banner / screenshot here, e.g.: -->
  <!-- <img src="imgs/banner.png" alt="SimpleRAG" width="800"> -->
</p>

A small semantic retrieval system for clinical literature. An encoder is fine-tuned with LoRA so that **queries and the documents that answer them land near each other in vector space**, and search at query time is a single matrix multiply against a pre-encoded corpus.

Dataset: [BEIR / NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) — ~3.6k medical/nutritional documents with human-annotated relevance labels.
>>>>>>> af2015b (Added sample photos and README.md)

---

## Why a bi-encoder

Four options for "given a query and a corpus, return relevant documents":

| Approach | Speed | Quality | Notes |
|---|---|---|---|
| BM25 / TF-IDF | Fast | OK | Lexical only, dies on paraphrase |
| Cross-encoder | Slow | Best | Re-runs the model per (query, doc) pair |
| **Bi-encoder** | Fast | Good | Encode once offline, matmul at query time |
| Generative RAG | — | — | Bi-encoder + LLM on top; out of scope here |

Bi-encoder is the right trade-off at this scale: real transformer quality, but the heavy compute moves to a one-time encoding pass.

## The backbone: Ettin-150M (ModernBERT)

[`jhu-clsp/ettin-encoder-150m`](https://huggingface.co/jhu-clsp/ettin-encoder-150m). Reasons:

- **Small enough to fine-tune on a laptop** (~150M params, and LoRA only trains a fraction of them).
- **ModernBERT internals.** Rotary position embeddings (the dot product between tokens depends on their *relative* offset, not absolute position), GeGLU activations, and alternating local/global attention so long inputs stay cheap.
- **Pretrained for retrieval.** Ettin's encoder half was explicitly evaluated as a retrieval backbone, so the representations start near where we want them.

The model is wrapped as `RAGEtin` in `src/model.py` — a thin module that runs the encoder on query, positive, and negative batches and returns the first-position hidden state of each.

## Role tokens

Bi-encoders see two very different kinds of text: terse queries and long documents. Instead of making the model infer that from surface form, we tell it directly via six new special tokens (`src/utils.py`):

```
<QRY> ... </QRY>     for queries
<TLE> ... </TLE>     for titles
<TXT> ... </TXT>     for body text
```

The tokens get added to the tokenizer; the embedding matrix is resized to make room for them (`resize_token_embeddings`); the six new rows are random-init and learn during fine-tuning what "this is a query" should mean geometrically.

## Pooling: CLS + L2 norm

A variable-length sequence needs to become one vector. Options are CLS pooling (position 0), mean pooling, max pooling, or attention pooling. This project uses **CLS** — `last_hidden_state[..., 0, :]` — because the contrastive loss trains that one slot to act as a summary register.

The vectors are then L2-normalized. Two reasons:

1. **Dot product = cosine similarity** when both sides are unit norm:
   `cos(u, v) = (u · v) / (‖u‖ · ‖v‖) = u · v`. So `que_vec @ doc_vecs.T` is literally the cosine-similarity matrix.
2. **Better geometry.** The loss can only move directions, not magnitudes — the model can't cheat by inflating "important" vectors.

## The training objective: InfoNCE

The loss (`MultiNCELoss` in `src/train.py`). For each query, sample one positive `p` from the qrels and `N = 8` random negatives `nᵢ`. With temperature `τ = 0.2`:

```
              exp((q · p) / τ)
L = -log ─────────────────────────────────
         exp((q · p) / τ) + Σᵢ exp((q · nᵢ) / τ)
```

Minimizing this does two things simultaneously: pushes `cos(q, p) → 1` (numerator up) and `cos(q, nᵢ) → −1` (denominator down). Standard InfoNCE.

**Why τ = 0.2?** Smaller `τ` makes the loss sharper — it amplifies similarity gaps before the softmax. Aggressive temperatures pair with hard-mined negatives; with random negatives a gentler 0.2 keeps the model from over-fitting to easy contrasts.

The implementation supports multiple positives per query — the numerator generalizes to `Σⱼ exp((q · pⱼ) / τ)`. We sample one in practice.

## LoRA fine-tuning

Full fine-tuning of 150M params on 3.6k documents would overfit and steamroll the pretrained weights. **LoRA** freezes the backbone and adds trainable low-rank updates to each linear layer:

```
y = W x  +  (α / r) · (B A) x
```

`A ∈ ℝ^{r × d_in}`, `B ∈ ℝ^{d_out × r}`, with `r ≪ d`. `A` is Gaussian-init, `B` is zero-init, so the adapter starts as a no-op. The empirical justification is that fine-tuning updates `ΔW` tend to be intrinsically low-rank, so a rank-16 approximation is enough.

Config in `src/train.py`:

- `r = 16`, `lora_alpha = 32` → effective scale `α/r = 2`.
- `target_modules = "all-linear"` → PEFT resolves this to `["Wi", "Wo", "Wqkv"]` (ModernBERT's FFN-in, FFN-out, and fused QKV projection).
- `lora_dropout = 0.065`, biases frozen.

Result: a ~13 MB `adapter_model.safetensors` instead of a full checkpoint. The high `lr = 1e-3` is normal for LoRA — adapters start near zero, only they get updated, and the frozen base can't be destabilized.

## Training loop

`src/train.py`: AdamW @ 1e-3, 200 epochs, linear warmup (10% of total steps) then linear decay, `max_length = 256`. Validation runs on NFCorpus' `dev` split each epoch; best-dev-loss adapter is saved.

Per step: encode (query, positive, negatives) → compute InfoNCE → backprop only through LoRA adapters and the six new token embeddings.

## Encoding the corpus once

`src/encode_corpus.py` runs every document through the trained model once, L2-normalizes, and saves `{doc_vecs, pdf_paths}` to `corpus_encoded.pt`. This is the whole point of a bi-encoder: trade one `O(N)` encoding pass for an `O(N·d)` matmul at every query.

Only documents with a PDF on disk are included so the UI can always serve a result.

## Search

`src/search.py`:

```python
que_vec = F.normalize(encoder(query), p=2, dim=1)     # (1, d)
sims = (que_vec @ doc_vecs.T).squeeze(0)              # (N,)
hits = [(s, path) for s, path in sorted(...) if s >= threshold]
```

One matmul, one sort, one threshold filter. At ~3.6k docs this is sub-millisecond. For millions of docs you would swap the matmul for FAISS/HNSW — the geometry stays the same, only the lookup changes.

## Repo layout

```
src/
  model.py           Ettin + tokenizer + six role tokens
  utils.py           AddTokens wrapper
  train.py           NFCorpusDataset, MultiNCELoss, training loop
  encode_corpus.py   one-shot pass → corpus_encoded.pt
  search.py          CLI search with clickable PDF links
  indexer.py         encode-and-append helper for the web UI
  config.py          paths, defaults, host/port
app/
  app.py             Flask wrapper
  templates/         the MedSearch UI

adapter_model.safetensors      trained LoRA weights (~13 MB)
corpus_encoded.pt              {doc_vecs, pdf_paths}
data/nfcorpus/                 BEIR NFCorpus
```

## Running it

```bash
python src/train.py                                       # optional retrain
python src/encode_corpus.py path/to/checkpoint out.pt     # optional re-encode
python src/search.py . corpus_encoded.pt 0.5              # CLI
python app/app.py                                         # UI at :5000
```

The pre-trained adapter and pre-encoded corpus ship with the repo.

---

## Addition: the MedSearch web UI

A Flask wrapper around the same encoder + same `corpus_encoded.pt`. No new ML — just a UI, plus a way to grow the index after training.

### Main page

<p align="center">
  <img src="imgs/main_page.png" alt="Main page" width="520">
</p>

Search bar with two knobs: **Top-K** (max results) and **Threshold** (minimum cosine similarity to count as a hit).

### Searching

<p align="center">
  <img src="imgs/search_sample.png" alt="Search results" width="420">
</p>

Query goes to `/search` as JSON, gets wrapped in `<QRY>...</QRY>`, encoded and L2-normalized exactly like training, then `que_vec @ doc_vecs.T` ranks every document. The score next to each result is the raw cosine similarity. The query *"How to eat healthy"* pulls back *"Essentials of Healthy Eating: A Guide"* with only one lexical word in common — that is the encoder doing its job.

### Adding a document

<p align="center">
  <img src="imgs/doc_add.png" alt="Add document" width="520">
</p>

Three fields: title, an abstract/text (this is what gets indexed — the PDF itself is not parsed, intentionally), and the PDF file. On submit, `/add`:

1. Saves the PDF to `data/nfcorpus/pdf_docs/`.
2. Runs the same encoding path used at build time (`src/indexer.py`).
3. Appends the new vector to the in-memory matrix and persists `corpus_encoded.pt`, under a lock to avoid races.

From the next search onward the document is in the candidate pool. **No retraining** — the LoRA adapter stays put; we just extend the matrix the query gets multiplied against.

---

## What's intentionally missing

- **Hard-negative mining.** Random negatives are weak; BM25- or model-mined negatives are the obvious upgrade.
- **Cross-encoder reranker** over the top-K.
- **ANN index** (FAISS / HNSW). Needed past ~10⁵ docs.
- **PDF text extraction.** User-supplied abstracts beat blind page-1 slurps.
- **Proper IR eval** (NDCG@10, Recall@k on the test qrels).

None of these change the skeleton — encoder → contrastive loss → cached vectors → matmul. That part is the same recipe most production retrieval stacks use.