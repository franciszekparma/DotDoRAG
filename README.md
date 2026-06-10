<div align="center">

# SimpleRAG

**A from-scratch PyTorch dense retriever — a 150 M-parameter ModernBERT encoder adapted to scientific literature with LoRA, served against a flat-tensor index over a corpus of PDFs.**

<sub>Built by <b>Franciszek Parma</b> &amp; <b>Jan Juszczyk</b></sub>

</div>

---

## At a glance

| | |
|---|---|
| **Task** | Domain-adapted semantic search over scientific PDFs |
| **Domain** | NFCorpus — biomedical IR benchmark (~3.6 k papers) |
| **Backbone** | `jhu-clsp/ettin-encoder-150m` (ModernBERT, 150 M params) |
| **Adaptation** | LoRA — ~1.5 M trainable params, **~6 MB** adapter |
| **Loss** | Multi-positive InfoNCE, `τ = 0.2` |
| **Index** | Flat tensor `(N, 768)` float32 — brute-force matmul, sub-ms search |
| **Live updates** | Drop a PDF → searchable on the next query, no restart |

---

## Table of contents

1. [Why a fine-tuned bi-encoder?](#why-a-fine-tuned-bi-encoder)
2. [The training objective](#the-training-objective)
3. [Architecture](#architecture)
4. [Retrieval & live indexing](#retrieval--live-indexing)
5. [Hyperparameters](#hyperparameters)
6. [Project structure](#project-structure)
7. [How to run it](#how-to-run-it)
8. [Dataset](#dataset)
9. [References](#references)

---

## Why a fine-tuned bi-encoder?

Lexical retrieval like BM25 matches on surface words. It fails the moment the query says *"fasting and blood sugar"* and the relevant paper is titled *"intermittent caloric restriction and glycemic response."*

A **bi-encoder** sidesteps this by mapping queries and documents independently into the same unit-norm vector space and ranking by cosine similarity. Documents are encoded once, offline; a query becomes a single forward pass plus one matmul against the cached index — sub-millisecond search, no re-ranker.

Off-the-shelf encoders are trained on generic text, so the vector geometry isn't tuned to *this* corpus's notion of relevance. Fine-tuning fixes that, and **LoRA** does it by moving only ~1 % of the parameters: the base model stays intact and the adapter is a 6 MB diff.

> **Bottom line.** BM25 fails on paraphrase. A generic encoder gets most of the way. A LoRA-adapted encoder closes the rest of the gap at the cost of one training run.

---

## The training objective

NFCorpus queries can have ten or more relevant documents, so vanilla InfoNCE (one positive per query) is the wrong loss. SimpleRAG uses a **multi-positive InfoNCE** — all known positives go into the numerator:

```
            Σ exp(q · p / τ)
            p∈P
L = -log ─────────────────────────────────────
         Σ exp(q · p / τ)  +  Σ exp(q · n / τ)
         p∈P                   n∈N
```

| Term | Role | Why it's needed |
|---|---|---|
| **L2-normalized dot product** | Cosine similarity in disguise. | Inner products become bounded and scale-stable, and the train metric matches the serve metric exactly. |
| **Sum-of-exp in the numerator** | Sets the model's reward to "rank *any* positive above all negatives." | Sum-of-log penalizes the model whenever positives have unequal scores, even when the ranking is correct — wrong incentive for set-valued relevance. |
| **Temperature `τ = 0.2`** | Sharpens the softmax. | Too high → all docs treated as equally important and learning stalls. Too low → the loss becomes hinge-like and only the top item matters. `0.2` is the centre of mass in the dense-retrieval literature. |
| **8 random negatives per query** | Provides the contrast set in the denominator. | Random sampling is weak but zero-infrastructure; hard-negative mining is the obvious next upgrade. |

> **Why `lr = 1e-3`?** Unusually high for fine-tuning — but only the adapter moves. Pre-trained base weights would not survive `1e-3`; freshly-initialized rank-16 adapter matrices do, and they need it to escape LoRA's zero initialization of `B` quickly.

---

## Architecture

### Encoder · `text → ℝ⁷⁶⁸`

```
"<QRY> fasting and blood sugar </QRY>"
   ↓ tokenize  (three pairs of learned special tokens added to the vocab)
   ↓ ModernBERT-150M   (frozen base + LoRA on every linear layer)
   ↓ CLS pooling — h[0]
   ↓ L2 normalize
vector ∈ 𝕊⁷⁶⁷
```

### LoRA residual · `W' x = W x + (α/r) · B A x`

For every linear layer, a low-rank residual is trained while the base weight `W` stays frozen. `A ~ 𝒩(0, σ²)`, `B = 0`, so the model behaves identically to the base at step 0 and gradients flow through `B` from there.

| | Full fine-tuning | LoRA (`r = 16, α = 32`) |
|---|---:|---:|
| Trainable parameters | ~150 M | ~1.5 M |
| Optimizer state (AdamW, fp32) | ~1.2 GB | ~12 MB |
| Artifact size | ~600 MB | **~6 MB** |

### Why these design choices?

| Choice | Why |
|---|---|
| **ModernBERT-150M backbone** | Alternating local/global attention and RoPE give cheap long context; 150 M is the sweet spot between learning useful semantics and fine-tuning on a single GPU. Pre-trained with retrieval in mind, so the off-the-shelf representations are already reasonable. |
| **CLS pooling**, not mean | Mean pooling has a hidden `1/L` length bias that underweights long documents. CLS is the simplest aggregation and matches the head ModernBERT was pre-trained against. |
| **`<QRY>` / `<TLE>` / `<TXT>` learned tokens** | Queries and documents are asymmetric; title and body play different structural roles. Learned tokens give the model an explicit, unambiguous signal of which is which, in two embedding rows each. E5 and BGE use natural-language prefixes for the same purpose; learned tokens are the cleaner version — no risk of colliding with corpus vocabulary. |
| **Single `AddTokens` helper** for every entry point | If train and serve format the same string differently, the geometry breaks silently — same model, same weights, wrong answers. Centralizing the format is the cheapest insurance against this failure mode. |
| **LoRA on `all-linear`**, not attention only | Adapting only attention is a common shortcut, but the MLP projections carry a substantial fraction of the useful domain shift on retrieval. The parameter saving from skipping them is small. |
| **`r = 16, α = 32`** | Effective scaling `α/r = 2`. High enough that the adapter has meaningful influence early in training, low enough not to destabilize the optimization. |
| **Adapter dropout `0.065`** | Mild stochastic regularization on the bottleneck. With a small corpus (~3.6 k docs) and a high effective adapter LR, this materially helps generalization to the dev set. |

---

## Retrieval & live indexing

After training, every document is encoded once, L2-normalized, and stacked into a flat `(N, 768)` tensor saved alongside its PDF paths. Search is a single dot product against this matrix.

```python
q     = L2norm( encoder("<QRY> " + query + " </QRY>")[:, 0, :] )   # (1, 768)
sims  = (q @ doc_vecs.T).squeeze(0)                                 # (N,) in [-1, 1]
hits  = [(sims[i], pdf_paths[i]) for i in argsort(sims, descending=True)
                                  if sims[i] >= threshold][:top_k]
```

Both sides are unit-normalized, so the dot product *is* cosine similarity. The threshold (default `0.75`) is a **floor on semantic relevance**, not a tuning knob — it has a stable, interpretable meaning across queries because the encoder was trained with the same metric. Top-k (default `5`) is a UX cap.

> **Why a flat tensor, not FAISS?** NFCorpus is ~3.6 k docs. An `(N, 768)` float32 matmul is **bandwidth-bound** and finishes in ~200 µs on CPU — any ANN library's own indexing overhead would be larger. Same code scales cleanly to ~100 k documents (~300 MB index, ~10 ms scan); swap to `IndexFlatIP` past that, and the interface doesn't change.

The index isn't static. A new PDF + title + abstract is encoded with the same helper used at corpus-build time, appended to the in-memory `doc_vecs` under a `threading.Lock`, and persisted atomically — searchable on the very next query, with no restart and no re-encode of the existing corpus.

```python
new_vec = encode_doc(model, tokenizer, device, title, text)        # (1, 768), L2-normed
with _lock:
    doc_vecs = torch.cat([doc_vecs, new_vec], dim=0)
    pdf_paths.append(save_path)
    torch.save({'doc_vecs': doc_vecs.cpu(), 'pdf_paths': pdf_paths}, encoded_path)
```

<p align="center">
  <!-- TODO: replace with docs/interface.png -->
  <img src="docs/interface.png" width="80%" alt="SimpleRAG interface" />
  <br/><sub><i>The running interface — searching the corpus and adding new papers from the browser.</i></sub>
</p>

---

## Hyperparameters

All in [`src/config.py`](src/config.py) and [`src/train.py`](src/train.py).

<table>
<tr><th align="left">Training</th><th></th><th align="left">Model &amp; LoRA</th><th></th></tr>
<tr><td>Optimizer</td><td><code>AdamW</code></td><td>Backbone</td><td><code>ettin-encoder-150m</code></td></tr>
<tr><td>Learning rate</td><td><code>1e-3</code></td><td>Hidden dim <code>D</code></td><td><code>768</code></td></tr>
<tr><td>Schedule</td><td>Linear warmup (10 %) → decay</td><td>LoRA rank / scaling</td><td><code>r = 16</code>, <code>α = 32</code></td></tr>
<tr><td>Batch size (queries)</td><td><code>2</code></td><td>Adapter dropout</td><td><code>0.065</code></td></tr>
<tr><td>Epochs</td><td><code>200</code></td><td>Target modules</td><td><code>all-linear</code></td></tr>
<tr><td>Loss</td><td>Multi-positive InfoNCE</td><td>Trainable / total</td><td>~1.5 M / ~150 M</td></tr>
<tr><td>Temperature <code>τ</code></td><td><code>0.2</code></td><td>Adapter size on disk</td><td>~6 MB</td></tr>
<tr><td>Negatives / query</td><td><code>8</code> random</td><td>Max seq len (q &amp; doc)</td><td><code>256</code></td></tr>
</table>

---

## Project structure

```
.
├── src/
│   ├── config.py                 # paths, defaults, host/port
│   ├── model.py                  # encoder, tokenizer, special tokens, device pick
│   ├── utils.py                  # AddTokens — single source of truth for string format
│   ├── indexer.py                # encode_doc + thread-safe append_to_index
│   ├── train.py                  # NFCorpus dataset, multi-positive InfoNCE, LoRA loop
│   ├── encode_corpus.py          # corpus → corpus_encoded.pt
│   └── search.py                 # terminal REPL with OSC-8 clickable PDF links
├── app/
│   ├── app.py                    # web interface — search, add, PDF serving
│   └── templates/index.html      # single-page UI
├── adapter_config.json           # LoRA config (peft)
├── adapter_model.safetensors     # trained LoRA weights (~6 MB)
├── corpus_encoded.pt             # {doc_vecs: (N, 768) float32, pdf_paths: [str]}
└── data/nfcorpus/                # corpus, queries, qrels, PDFs
```

---

## How to run it

```bash
git clone https://github.com/franciszekparma/SimpleRAG.git
cd SimpleRAG
pip install torch transformers peft flask pandas tqdm
```

A pre-trained adapter and pre-encoded index ship with the repo, so steps 3–4 are optional for a first run.

**1 · Web interface**

```bash
python app/app.py
# → http://127.0.0.1:5000
```

**2 · Terminal search** *(Cmd / Ctrl-click results to open the PDF)*

```bash
python src/search.py <checkpoint_dir> corpus_encoded.pt 0.75
```

**3 · (Re-)train the adapter**

```bash
python src/train.py
```

**4 · (Re-)build the index**

```bash
python src/encode_corpus.py <checkpoint_dir> corpus_encoded.pt
```

> All commands run from the repo root.

---

## Dataset

**NFCorpus** (Boteva et al., 2016) — a full-text biomedical IR benchmark, ~3.6 k documents and ~3.2 k queries, with multi-relevance qrels. The multi-positive structure is exactly why the loss is multi-positive InfoNCE rather than the standard single-positive form.

Expected layout under `data/nfcorpus/`:

```
corpus.jsonl                # {"_id", "title", "text", ...}
queries.jsonl               # {"_id", "text"}
qrels/{train,dev,test}.tsv  # query-id ↔ corpus-id ↔ relevance
pdf_docs/                   # one .pdf per corpus entry, filename = title
```

---

## References

- Karpukhin et al. (2020). [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906).
- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- van den Oord et al. (2018). [Representation Learning with Contrastive Predictive Coding (InfoNCE)](https://arxiv.org/abs/1807.03748).
- Wang et al. (2022). [Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)](https://arxiv.org/abs/2212.03533).
- Warner et al. (2024). [ModernBERT: Smarter, Better, Faster, Longer](https://arxiv.org/abs/2412.13663).
- Boteva et al. (2016). [NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/).

---

<div align="center"><sub>MIT &copy; Franciszek Parma &amp; Jan Juszczyk</sub></div>
