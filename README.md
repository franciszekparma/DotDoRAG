# SimpleRAG

A compact, end-to-end **dense retrieval** system for scientific PDFs. SimpleRAG fine-tunes a transformer encoder with **LoRA** on the [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) biomedical retrieval benchmark, encodes the corpus into a single dense index, and serves a Flask web app that lets you search the index, open the matching PDFs, and add new papers on the fly.

The project is intentionally small. The point is not to ship another retrieval framework — it is to show, in a few hundred lines of code, how every moving part of a modern dense retriever fits together: the encoder, the contrastive objective, the parameter-efficient fine-tuning, the index, and the serving layer.

---

## Table of contents

- [How it works](#how-it-works)
  - [1. The encoder: Ettin-150M](#1-the-encoder-ettin-150m)
  - [2. Role-marker tokens](#2-role-marker-tokens)
  - [3. Fine-tuning with LoRA](#3-fine-tuning-with-lora)
  - [4. The training objective: multi-positive InfoNCE](#4-the-training-objective-multi-positive-infonce)
  - [5. Building the index](#5-building-the-index)
  - [6. Retrieval at query time](#6-retrieval-at-query-time)
- [The application](#the-application)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)

---

## How it works

The system follows the standard **bi-encoder** retrieval recipe: queries and documents are mapped independently into the same unit-norm vector space, and relevance is measured by cosine similarity. Below is what each stage does and why it was chosen.

### 1. The encoder: Ettin-150M

The backbone is [`jhu-clsp/ettin-encoder-150m`](https://huggingface.co/jhu-clsp/ettin-encoder-150m), a 150M-parameter encoder from Johns Hopkins' CLSP group built on the **ModernBERT** architecture. It was picked for three reasons:

- **Right size.** Large enough to learn useful semantics, small enough to fine-tune and serve on a single consumer GPU (or Apple Silicon via MPS).
- **Long context out of the box.** ModernBERT-style models handle long sequences efficiently thanks to local/global alternating attention and RoPE, which matters for full-abstract document encoding.
- **Strong retrieval prior.** Ettin was pre-trained with retrieval in mind, so the off-the-shelf representations are already reasonable; fine-tuning mainly has to specialize them to the biomedical domain.

The model wrapper (`model.py`) exposes a single encoder `self.enc` and a `forward` that returns the **CLS pooled** representation (`last_hidden_state[..., 0, :]`) for queries, positives, and negatives in one call. CLS pooling is used (rather than mean pooling) because it is the simplest sentence-level summary the model was pre-trained to produce, and it lines up cleanly with how the contrastive loss is computed downstream.

Device selection is automatic: CUDA → MPS → CPU.

### 2. Role-marker tokens

A subtle but important detail. Both queries and documents are wrapped in dedicated special tokens before tokenization (`utils.py`):

```
<QRY> ... </QRY>     # queries
<TLE> ... </TLE>     # document titles
<TXT> ... </TXT>     # document body text
```

The tokens are added to the tokenizer's vocabulary and the embedding matrix is resized accordingly (`model.py`). This gives the model an explicit, learnable signal of **which side of the asymmetric query-document relationship a piece of text belongs to**, and of the **structural role** (title vs. body) inside a document. The same trick is used by E5, BGE, and other strong retrievers — the only difference is that here the markers are learned rather than fixed natural-language prefixes ("query:" / "passage:").

The token-wrapping logic is centralized in `AddTokens` so that training, corpus encoding, and serving are guaranteed to use identical formatting. Any drift here silently destroys retrieval quality.

### 3. Fine-tuning with LoRA

The base encoder is frozen and adapted with **LoRA** (Low-Rank Adaptation) via the `peft` library (`train.py`). LoRA was chosen because:

- It reduces the number of trainable parameters by ~100x, which makes the optimizer state fit comfortably in memory and the training run cheap.
- The base model stays intact on disk; only a small adapter (a few MB) needs to be versioned, shipped, or swapped.
- Empirically, low-rank adapters perform on par with full fine-tuning for retrieval on small-to-mid corpora like NFCorpus.

Configuration:

```python
LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.065,
    target_modules="all-linear",   # adapts every linear layer in the encoder
    bias="none", task_type=None,
)
```

`target_modules="all-linear"` adapts every linear projection in the model — attention `Wqkv`/`Wo` plus the MLP `Wi`/`Wo` (see `adapter_config.json`). `r=16` with `alpha=32` gives a scaling factor of 2, a standard middle-ground for retrieval tuning.

Optimization details:

- **AdamW**, learning rate `1e-3` — high but appropriate since only adapter weights are trained.
- **Linear warmup + linear decay** for 10% of total steps (`get_linear_schedule_with_warmup`), which stabilizes the first few hundred steps when the adapter weights are initialized from zero.
- Best checkpoint by lowest dev loss is saved with `model.enc.save_pretrained(...)`.

### 4. The training objective: multi-positive InfoNCE

The loss (`MultiNCELoss` in `train.py`) is a small generalization of standard InfoNCE that handles the fact that NFCorpus queries often have multiple relevant documents.

For each query vector `q` with positive set `P` and negative set `N`, all unit-normalized:

```
L = -log( Σ exp(q·p / τ)   /   ( Σ exp(q·p / τ) + Σ exp(q·n / τ) ) )
        p∈P                       p∈P                n∈N
```

Why this loss, and why the details matter:

- **Multi-positive sum** in the numerator means the model gets credit for pulling *any* relevant document closer, not just one — important on NFCorpus where queries can have 10+ relevant entries.
- **L2 normalization** before the dot product turns inner products into cosine similarity, which keeps gradients well-scaled and matches the search-time metric exactly.
- **Temperature `τ = 0.2`** sharpens the softmax. Lower temperatures push the model to make harder discriminations between positives and negatives; 0.2 is a common sweet spot for retrieval contrastive losses.
- **`1e-8` epsilon** inside the `log` is purely numerical hygiene to avoid `log(0)` when scores collapse early in training.

Negatives are sampled randomly from the corpus per query (`num_negatives=8`). Random negatives are weaker than hard negatives but require no extra mining infrastructure, which keeps the project simple; harder negatives would be the obvious next improvement.

### 5. Building the index

After training, `encode_corpus.py` loads the base model, layers the trained LoRA adapter on top with `PeftModel.from_pretrained`, and encodes every document in the corpus that has a matching PDF on disk:

1. For each `corpus.jsonl` entry, build `<TLE>title</TLE> <TXT>text</TXT>`.
2. Tokenize (max length 256) and run through the encoder.
3. Take the CLS vector and L2-normalize it.
4. Stack everything into a `(N, D)` float tensor and save alongside the list of PDF paths to `corpus_encoded.pt`:

   ```python
   torch.save({'doc_vecs': doc_vecs, 'pdf_paths': pdf_paths}, out_path)
   ```

This is deliberately a flat dense tensor rather than FAISS / HNSW / a vector DB. At NFCorpus scale (~3.6k docs) a single matmul is already faster than any ANN library's overhead, and it keeps the dependency list short. Swapping in FAISS later is a few lines of code.

### 6. Retrieval at query time

Both the CLI (`search.py`) and the web app (`app.py`) follow the same three steps:

1. Wrap the query in `<QRY>…</QRY>`, tokenize, encode with the LoRA-adapted model, L2-normalize → `q ∈ ℝ^D`.
2. Compute `sims = q @ doc_vecs.T` — one dense matrix-vector product, no approximate search.
3. Sort descending; keep results whose cosine similarity is above a configurable threshold (default `0.75`) and trim to `top_k` (default `5`).

Because both sides are unit-normalized, the dot product *is* the cosine similarity, and the threshold has a stable, interpretable meaning across queries.

---

## The application

`app.py` is a small Flask app that wraps the retriever in a browser UI. It does three things:

### `GET /` — search UI

Renders `templates/index.html`, a single-page interface with a query box, threshold/top-k controls, and a results list. Clicking a result opens the corresponding PDF in a new tab.

### `POST /search` — JSON search API

Accepts:

```json
{ "query": "...", "top_k": 5, "threshold": 0.75 }
```

Encodes the query, scores it against the in-memory `doc_vecs`, and returns:

```json
{
  "query": "...",
  "results": [
    { "title": "...", "snippet": "...", "score": 0.812,
      "filename": "...pdf", "pdf_url": "/pdf/...pdf" }
  ]
}
```

Snippets and titles come from a `corpus.jsonl` metadata index built once at startup (`load_metadata`), keyed by both title and filename stem so lookups survive minor filename differences.

### `POST /add` — index a new PDF live

Multipart form with a `pdf` file plus `title` and/or `text`. The handler:

1. Saves the PDF under `data/nfcorpus/pdf_docs/`, deduplicating the filename if necessary (`secure_filename` + numeric suffix).
2. Encodes `<TLE>title</TLE> <TXT>text</TXT>` with the same `encode_doc` helper used at indexing time (`indexer.py`).
3. Appends the new vector to `doc_vecs` and the new path to `pdf_paths`, then atomically persists the updated `corpus_encoded.pt` under a lock.
4. Updates the in-memory metadata so the document is searchable immediately, without restarting the server.

A `threading.Lock` in `indexer.append_to_index` serializes concurrent writes so that two simultaneous uploads cannot corrupt the index file.

### `GET /pdf/<filename>` — safe PDF serving

PDFs are served from disk with an explicit path-traversal guard: the resolved real path must live under the configured `PDF_DIR`, otherwise the request is 404'd. This is the only piece of the app that touches user-supplied filenames, so it gets explicit hardening.

### `GET /stats`

Returns `{"total_docs": N}`. Useful for health checks and for the UI to show how many documents are indexed.

---

## Project layout

```
simplerag/
├── model.py            # Ettin encoder + tokenizer + special tokens + device selection
├── utils.py            # AddTokens: the <QRY>/<TLE>/<TXT> wrapping logic
├── train.py            # NFCorpus dataset, multi-positive InfoNCE, LoRA training loop
├── encode_corpus.py    # One-shot corpus → corpus_encoded.pt encoder
├── indexer.py          # Shared encode_doc + thread-safe append_to_index
├── search.py           # Terminal search REPL with OSC-8 clickable PDF links
├── app.py              # Flask web app (search + live add + PDF serving)
├── config.py           # Paths, defaults, host/port
├── templates/
│   └── index.html      # Single-page search UI
├── adapter_config.json # LoRA adapter config (peft)
├── adapter_model.safetensors  # Trained LoRA weights
├── corpus_encoded.pt   # Dense index: {doc_vecs: (N,D), pdf_paths: [str]}
└── data/nfcorpus/      # Corpus, queries, qrels, PDFs
```

---

## Setup

Requirements: Python 3.10+, PyTorch with CUDA / MPS / CPU support.

```bash
pip install torch transformers peft flask pandas tqdm
```

The dataset and PDFs are expected under `data/nfcorpus/` with the standard NFCorpus layout (`corpus.jsonl`, `queries.jsonl`, `qrels/{train,dev,test}.tsv`, `pdf_docs/*.pdf`).

---

## Usage

**Train the LoRA adapter** (writes checkpoints under `checkpoints/`):

```bash
python train.py
```

**Encode the corpus** with a trained checkpoint:

```bash
python encode_corpus.py <checkpoint_dir> corpus_encoded.pt
```

**Search from the terminal** (Cmd/Ctrl-click results to open the PDF):

```bash
python search.py <checkpoint_dir> corpus_encoded.pt 0.75
```

**Run the web app:**

```bash
python app.py
# → http://127.0.0.1:5000
```

---

## Configuration

All tunables live in `config.py`:

| Setting              | Default                       | Meaning                                         |
| -------------------- | ----------------------------- | ----------------------------------------------- |
| `PDF_DIR`            | `data/nfcorpus/pdf_docs`      | Where PDFs are stored and served from           |
| `CORPUS_JSONL`       | `data/nfcorpus/corpus.jsonl`  | Source of titles/snippets for the metadata index |
| `CHECKPOINT_PATH`    | project root                  | Directory containing the LoRA adapter           |
| `ENCODED_PATH`       | `corpus_encoded.pt`           | Dense index file                                |
| `DEFAULT_TOP_K`      | `5`                           | Max results returned per query                  |
| `DEFAULT_THRESHOLD`  | `0.75`                        | Minimum cosine similarity to count as a hit     |
| `MAX_QUERY_LENGTH`   | `256`                         | Query truncation length                         |
| `MAX_DOC_LENGTH`     | `256`                         | Document truncation length                      |
| `HOST` / `PORT`      | `127.0.0.1` / `5000`          | Flask bind address                              |
