<div align="center">

# SimpleRAG

**A from-scratch PyTorch dense retriever — LoRA-fine-tuned on NFCorpus, served behind a Flask app you can search, browse, and grow in real time.**

<p>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" />
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img alt="Flask" src="https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green" />
</p>

</div>

---

## At a glance

| | |
|---|---|
| **Task** | Domain-adapted semantic search over scientific PDFs |
| **Domain** | NFCorpus — biomedical IR benchmark (~3.6k papers) |
| **Backbone** | `jhu-clsp/ettin-encoder-150m` (ModernBERT, 150 M params) |
| **Adaptation** | LoRA — ~1.5 M trainable params, **~6 MB** adapter |
| **Index** | Flat tensor `(N, 768)` — brute-force matmul, sub-ms search |
| **Loss** | Multi-positive InfoNCE, `τ = 0.2` |
| **Interface** | Flask web app + terminal REPL |
| **Live updates** | Drop a PDF → searchable on the next query, no restart |

---

## Table of contents

1. [Why a fine-tuned bi-encoder?](#why-a-fine-tuned-bi-encoder)
2. [The training objective](#the-training-objective)
3. [Architecture](#architecture)
4. [Live indexing](#live-indexing)
5. [Hyperparameters](#hyperparameters)
6. [Project structure](#project-structure)
7. [Quickstart](#quickstart)
8. [Dataset](#dataset)
9. [References](#references)

---

## Why a fine-tuned bi-encoder?

Lexical retrieval like BM25 matches on surface words. It fails the moment the query says *"fasting and blood sugar"* and the relevant paper is titled *"intermittent caloric restriction and glycemic response."*

A **bi-encoder** maps queries and documents independently into the same unit-norm vector space and ranks by cosine similarity. Documents are encoded once, offline; a query becomes one forward pass plus a single matmul against the cached index. Sub-millisecond search, no re-ranking.

Off-the-shelf encoders are trained on generic text, so the geometry isn't tuned to the corpus. Fine-tuning fixes that — and **LoRA** does it by moving only ~1% of the parameters, leaving the base model untouched and shipping a 6 MB diff.

> **Bottom line.** BM25 fails on paraphrase. A generic encoder gets you most of the way. A LoRA-adapted encoder closes the rest of the gap at the cost of one weekend of training.

---

## The training objective

NFCorpus queries can have ten or more relevant documents. Vanilla InfoNCE assumes exactly one positive, so it's the wrong loss. SimpleRAG uses a **multi-positive InfoNCE** — all known positives go into the numerator:

```
            Σ exp(q · p / τ)
            p∈P
L = -log ─────────────────────────────────────
         Σ exp(q · p / τ)  +  Σ exp(q · n / τ)
         p∈P                   n∈N
```

| Choice | Why |
|---|---|
| **L2-normalize before the dot** | Inner products become cosine similarities, gradients stay scale-stable, and the train metric matches the serve metric exactly. |
| **Sum-of-exp in the numerator** (log-of-sum) | Rewards ranking *any* positive above all negatives. Sum-of-log penalizes the model when positives have unequal scores even if the ranking is correct — wrong incentive for set-valued relevance. |
| **`τ = 0.2`** | Lower temperature sharpens the softmax. Too high → all docs treated as equally important and learning stalls. Too low → loss becomes hinge-like and only the top item matters. `0.2` is the centre of mass in the dense-retrieval literature. |
| **8 random negatives per query** | Weak but zero-infrastructure. Hard-negative mining is the obvious next upgrade. |

```python
q     = F.normalize(encoder(q_tokens), p=2, dim=-1)
P, N  = F.normalize(encoder(pos_tokens), ...), F.normalize(encoder(neg_tokens), ...)
loss  = -torch.log(torch.exp(q @ P.T / tau).sum() /
                  (torch.exp(q @ P.T / tau).sum() + torch.exp(q @ N.T / tau).sum()))
```

---

## Architecture

### Encoder · `text → ℝ⁷⁶⁸`

```
"<QRY> fasting and blood sugar </QRY>"
   ↓ tokenize  (with three pairs of learned special tokens added to the vocab)
   ↓ ModernBERT-150M (frozen base + LoRA on every linear layer)
   ↓ CLS pooling — h[0]
   ↓ L2 normalize
vector ∈ 𝕊⁷⁶⁷
```

### Index · `(N, 768)` flat float32 tensor

```
for each PDF with a corpus.jsonl entry:
    text   = "<TLE> title </TLE> <TXT> body </TXT>"
    v      = L2norm( encoder(text) )
    doc_vecs.append(v);  pdf_paths.append(path)

torch.save({'doc_vecs': stack(doc_vecs),     # (N, 768)
            'pdf_paths': pdf_paths},          # list[str], aligned by row
           'corpus_encoded.pt')
```

### Search · one matmul

```
q     = L2norm( encoder("<QRY> " + query + " </QRY>") )    # (1, 768)
sims  = q @ doc_vecs.T                                      # (1, N), in [-1, 1]
hits  = [(sims[i], pdf_paths[i]) for i in argsort(sims, descending=True)
                                  if sims[i] >= threshold][:top_k]
```

### Why these design choices?

| Choice | Why |
|---|---|
| **`<QRY>` / `<TLE>` / `<TXT>` learned tokens** | Queries and documents are asymmetric; title and body play different structural roles. Learned tokens give the model an explicit, unambiguous signal of which is which, in two embedding rows each. E5 and BGE do the same thing with natural-language prefixes (`"query: ..."`); learned tokens are the cleaner version. |
| **CLS pooling**, not mean | Mean pooling has a hidden `1/L` length bias that underweights long documents. CLS is the simplest aggregation and was the head ModernBERT was pre-trained against. |
| **LoRA on `all-linear`**, not attention only | Adapting only attention is a common shortcut, but the MLP projections carry a large fraction of the useful domain shift on retrieval. The parameter saving from skipping them is small. |
| **`r = 16, α = 32`** | Effective scaling `α/r = 2`. High enough to influence training early, low enough not to destabilize it. |
| **Flat tensor, not FAISS** | NFCorpus is ~3.6k docs. An `(N, 768)` float32 matmul is bandwidth-bound and finishes in ~200 µs on CPU — any ANN library's own overhead would be larger. Same code scales cleanly to ~100k documents; swap to FAISS past that. |
| **One `AddTokens` helper** for every entry point (train, encode, serve) | If train and serve format the same string differently, the geometry breaks silently — same model, same weights, wrong answers. Centralizing the format is the cheapest insurance against this failure mode. |

---

## Live indexing

The Flask app's `/add` endpoint accepts a PDF + title + abstract, encodes the text with the same helper used at corpus-build time, appends one row to the in-memory `doc_vecs`, and atomically rewrites `corpus_encoded.pt` under a `threading.Lock`. The new document is searchable on the very next `/search` — no restart, no re-encode.

```python
new_vec = encode_doc(model, tokenizer, device, title, text)   # (1, 768), L2-normed
with _lock:
    doc_vecs = torch.cat([doc_vecs, new_vec], dim=0)
    pdf_paths.append(save_path)
    torch.save({'doc_vecs': doc_vecs.cpu(), 'pdf_paths': pdf_paths}, encoded_path)
```

> The lock is the only thing standing between two concurrent uploads and a corrupted index file. Don't remove it.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Single-page search UI |
| `/search` | POST | Encode query → score against `doc_vecs` → top-k above threshold as JSON |
| `/add` | POST | Multipart upload — saves the PDF, encodes title+text, appends to the live index |
| `/pdf/<filename>` | GET | Serves a PDF with a real-path guard pinned inside `PDF_DIR` |
| `/stats` | GET | `{"total_docs": N}` |

---

## Hyperparameters

All in [`src/config.py`](src/config.py) and [`src/train.py`](src/train.py).

<table>
<tr><th align="left">Training</th><th></th><th align="left">Model & LoRA</th><th></th></tr>
<tr><td>Optimizer</td><td><code>AdamW</code></td><td>Backbone</td><td><code>ettin-encoder-150m</code></td></tr>
<tr><td>Learning rate</td><td><code>1e-3</code></td><td>Hidden dim <code>D</code></td><td><code>768</code></td></tr>
<tr><td>Schedule</td><td>Linear warmup (10%) → decay</td><td>LoRA rank <code>r</code> / <code>α</code></td><td><code>16</code> / <code>32</code></td></tr>
<tr><td>Batch size (queries)</td><td><code>2</code></td><td>Adapter dropout</td><td><code>0.065</code></td></tr>
<tr><td>Epochs</td><td><code>200</code></td><td>Target modules</td><td><code>all-linear</code></td></tr>
<tr><td>Loss</td><td>Multi-positive InfoNCE</td><td>Trainable / total</td><td>~1.5 M / ~150 M</td></tr>
<tr><td>Temperature <code>τ</code></td><td><code>0.2</code></td><td>Adapter size on disk</td><td>~6 MB</td></tr>
<tr><td>Negatives / query</td><td><code>8</code> random</td><td>Max seq len</td><td><code>256</code> (q & doc)</td></tr>
</table>

> **Why `lr = 1e-3`?** Unusually high for fine-tuning — but only the adapter moves. Base weights would not survive `1e-3`; freshly-initialized rank-16 matrices do, and they need it to escape LoRA's zero initialization quickly.

> **Why `α/r = 2`?** Standard middle-ground scaling for retrieval. High enough that the adapter has meaningful influence early in training, low enough that it doesn't destabilize the optimization.

---

## Project structure

```
.
├── src/                          # all Python sources except the web app
│   ├── config.py                 # paths, defaults, host/port
│   ├── model.py                  # encoder, tokenizer, special tokens, device pick
│   ├── utils.py                  # AddTokens — single source of truth for string format
│   ├── indexer.py                # encode_doc, thread-safe append_to_index
│   ├── train.py                  # NFCorpus dataset, multi-positive InfoNCE, LoRA loop
│   ├── encode_corpus.py          # corpus → corpus_encoded.pt
│   └── search.py                 # terminal REPL with OSC-8 clickable PDF links
├── app/                          # Flask web app
│   ├── app.py                    # /search, /add, /pdf, /stats
│   └── templates/index.html      # single-page UI
├── adapter_config.json           # LoRA config (peft)
├── adapter_model.safetensors     # trained LoRA weights (~6 MB)
├── corpus_encoded.pt             # {doc_vecs: (N, 768) float32, pdf_paths: [str]}
└── data/nfcorpus/                # corpus, queries, qrels, PDFs
```

| File | Description |
|---|---|
| `src/config.py` | All paths, retrieval defaults (`top_k = 5`, `threshold = 0.75`), Flask host/port |
| `src/model.py` | `RAGEtin` wrapper, tokenizer with `<QRY>`/`<TLE>`/`<TXT>` added, device autoselect (CUDA → MPS → CPU) |
| `src/utils.py` | `AddTokens` — every train/serve string goes through here |
| `src/indexer.py` | `encode_doc(...)` and the locked `append_to_index(...)` |
| `src/train.py` | `NFCorpusDataset`, `MultiNCELoss`, LoRA loop with best-on-dev checkpointing |
| `src/encode_corpus.py` | One-shot: base model + trained adapter → `corpus_encoded.pt` |
| `src/search.py` | Terminal REPL; clickable file URIs via OSC-8 |
| `app/app.py` | Flask server + the five endpoints described above |

---

## Quickstart

```bash
git clone https://github.com/franciszekparma/SimpleRAG.git
cd SimpleRAG
pip install torch transformers peft flask pandas tqdm
```

A pre-trained adapter and pre-encoded index are committed to the repo, so you can skip straight to the app.

### 1 · Run the web app

```bash
python app/app.py
# → http://127.0.0.1:5000
```

### 2 · Search from the terminal

```bash
python src/search.py <checkpoint_dir> corpus_encoded.pt 0.75
```

Cmd/Ctrl-click any result to open the PDF.

### 3 · (Re)train the adapter

```bash
python src/train.py
# → checkpoints/epoch_K_train_X.XXXX_val_Y.YYYY/
```

### 4 · (Re)build the index

```bash
python src/encode_corpus.py <checkpoint_dir> corpus_encoded.pt
```

> All commands run from the repo root.

---

## Dataset

**NFCorpus** (Boteva et al., 2016) — a full-text biomedical IR benchmark of ~3.6k documents and ~3.2k queries with multi-relevance qrels.

Expected layout under `data/nfcorpus/`:

```
corpus.jsonl                # {"_id", "title", "text", ...}
queries.jsonl               # {"_id", "text"}
qrels/{train,dev,test}.tsv  # query-id ↔ corpus-id ↔ relevance
pdf_docs/                   # one .pdf per corpus entry, filename = title
```

Multi-positive relevance is the reason the loss is what it is — see [The training objective](#the-training-objective).

---

## References

- Karpukhin et al. (2020). [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906).
- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- Wang et al. (2022). [Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)](https://arxiv.org/abs/2212.03533).
- Warner et al. (2024). [ModernBERT: Smarter, Better, Faster, Longer](https://arxiv.org/abs/2412.13663).
- van den Oord et al. (2018). [Representation Learning with Contrastive Predictive Coding (InfoNCE)](https://arxiv.org/abs/1807.03748).
- Boteva et al. (2016). [NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/).

---

<div align="center"><sub>MIT &copy; franciszekparma</sub></div>
