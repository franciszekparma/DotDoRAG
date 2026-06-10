<div align="center">

# SimpleRAG

**A from-scratch PyTorch dense retriever — a 150 M-parameter ModernBERT encoder adapted to biomedical scientific literature with LoRA, served against a flat-tensor index over PDFs.**

<sub>Built by <b>Franciszek Parma</b> &amp; <b>Jan Juszczyk</b></sub>

</div>

<p align="center">
  <!-- TODO: replace with docs/search.png -->
  <img src="docs/search.png" width="80%" alt="Search interface" />
  <br/><sub><i>Searching the corpus — top hits above the cosine threshold, ranked by semantic similarity.</i></sub>
</p>

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
2. [The encoder](#the-encoder)
3. [Role-marker tokens](#role-marker-tokens)
4. [LoRA adaptation](#lora-adaptation)
5. [The training objective](#the-training-objective)
6. [Index construction](#index-construction)
7. [Search](#search)
8. [Live indexing](#live-indexing)
9. [Hyperparameters](#hyperparameters)
10. [Project structure](#project-structure)
11. [How to run it](#how-to-run-it)
12. [Dataset](#dataset)
13. [References](#references)

---

## Why a fine-tuned bi-encoder?

Lexical retrieval like BM25 matches on surface words. It fails the moment the query says *"fasting and blood sugar"* and the relevant paper is titled *"intermittent caloric restriction and glycemic response."*

A **bi-encoder** sidesteps this by mapping queries and documents independently into the same unit-norm vector space and ranking by cosine similarity. Documents are encoded once, offline; a query becomes a single forward pass plus one matmul against the cached index — sub-millisecond search, no re-ranker.

Off-the-shelf encoders are trained on generic text, so the vector geometry isn't tuned to *this* corpus's notion of relevance. **Fine-tuning** fixes that, and **LoRA** does it by moving ~1 % of the parameters: the base model stays intact and the adapter is a 6 MB diff.

> **Bottom line.** BM25 fails on paraphrase. A generic encoder gets most of the way. A LoRA-adapted encoder closes the rest of the gap at the cost of one training run.

---

## The encoder

The backbone is [`jhu-clsp/ettin-encoder-150m`](https://huggingface.co/jhu-clsp/ettin-encoder-150m) — a 150 M-parameter encoder built on the [ModernBERT](https://arxiv.org/abs/2412.13663) architecture.

| Property | Value | Why it matters |
|---|---|---|
| Parameters | 150 M | Fits comfortably on a single consumer GPU; small enough to fine-tune on a laptop. |
| Hidden size `D` | 768 | Wide enough to carry semantic structure; the index is `(N, 768)`. |
| Attention pattern | Alternating local / global | Quadratic only every few layers; long-context encoding stays cheap. |
| Positional encoding | RoPE | Stable extrapolation beyond training length. |

Sentence vectors are the **[CLS] token's** final hidden state, L2-normalized. CLS pooling was chosen over mean pooling because (i) ModernBERT was pre-trained with a CLS-style objective and the token already carries a sentence summary, and (ii) mean pooling has a hidden `1/L` length bias that underweights long documents. Device autoselect: CUDA → MPS → CPU.

---

## Role-marker tokens

Queries and documents are syntactically and semantically different objects. Document titles and body text play different roles. SimpleRAG injects this structure explicitly, with three pairs of **learned** special tokens added to the vocabulary:

```
<QRY> ... </QRY>     queries
<TLE> ... </TLE>     document titles
<TXT> ... </TXT>     document body text
```

E5 and BGE use natural-language prefixes (`"query: ..."`, `"passage: ..."`) for the same purpose. Learned tokens are the cleaner version of the idea: two embedding rows each, no risk of colliding with corpus vocabulary, role information unambiguous to the optimizer.

> **One source of truth.** Every string that enters the model — at train, encode, and serve time — goes through one `AddTokens` helper. If formatting drifts between phases, the geometry breaks silently: same model, same weights, wrong answers. Centralizing the format is the cheapest insurance against this failure mode.

---

## LoRA adaptation

Full fine-tuning would work. It would also need optimizer state that doesn't fit on most consumer GPUs and would produce a 600 MB artifact for every domain. [Low-Rank Adaptation (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) is the parameter-efficient alternative: for every targeted linear layer `W`, it adds a low-rank residual

```
W' x  =  W x  +  (α / r) · B A x
                            ↑   ↑
                       (d_out × r)  (r × d_in)
```

with `A ~ 𝒩(0, σ²)`, `B = 0` so the model behaves identically to the base at step 0.

| | Full fine-tuning | LoRA (`r = 16, α = 32`) |
|---|---:|---:|
| Trainable parameters | ~150 M | ~1.5 M |
| Optimizer state (AdamW, fp32) | ~1.2 GB | ~12 MB |
| Artifact size | ~600 MB | **~6 MB** |

| Choice | Why |
|---|---|
| **`target_modules = "all-linear"`** | Adapts attention (`Wqkv`, `Wo`) *and* MLP projections (`Wi`, `Wo`). Adapting only attention is a common shortcut, but the MLPs carry a substantial fraction of the useful domain shift on retrieval, and the parameter saving from skipping them is small. |
| **`r = 16, α = 32`** | Effective scaling `α/r = 2`. High enough that the adapter has meaningful influence early in training, low enough not to destabilize the optimization. |
| **`lora_dropout = 0.065`** | Mild regularization on the adapter activations. With a small corpus (~3.6 k docs) and a high effective adapter LR, this materially helps generalization to the dev set. |

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

| Choice | Why |
|---|---|
| **L2-normalize before the dot** | Inner products become cosine similarities; gradients stay scale-stable; the train metric matches the serve metric exactly. |
| **Sum-of-exp in the numerator** (log-of-sum, not sum-of-log) | Rewards ranking *any* positive above all negatives. Sum-of-log penalizes the model when positives have unequal scores even if the ranking is correct — wrong incentive for set-valued relevance. |
| **`τ = 0.2`** | Lower temperatures sharpen the softmax. Too high → all docs treated as equally important and learning stalls; too low → the loss becomes hinge-like and only the top item matters. `0.2` is the centre of mass in the dense-retrieval literature. |
| **8 random negatives per query** | Weak but zero-infrastructure. Hard-negative mining (sampling documents the current model scores high but are known to be irrelevant) is the obvious next upgrade. |

Optimizer: AdamW, `lr = 1e-3` (high, but only adapter weights move), linear warmup over 10 % of steps then linear decay. Best checkpoint by lowest dev loss.

> **Why `lr = 1e-3`?** Pre-trained base weights would not survive `1e-3`. Freshly-initialized rank-16 adapter matrices do — and they need it to escape LoRA's zero initialization of `B` quickly.

---

## Index construction

After training, `encode_corpus.py` layers the trained adapter on top of the base encoder and encodes every document with a matching PDF:

```python
for doc in corpus.jsonl:
    text = "<TLE> {title} </TLE> <TXT> {body} </TXT>"
    v    = L2norm( encoder(text)[:, 0, :] )         # (D,)
    doc_vecs.append(v); pdf_paths.append(pdf_path)

torch.save({'doc_vecs': torch.stack(doc_vecs),       # (N, 768)
            'pdf_paths': pdf_paths},                  # list[str], aligned by row
           'corpus_encoded.pt')
```

> **Why a flat tensor, not FAISS?** NFCorpus is ~3.6 k docs. An `(N, 768)` float32 matmul is **bandwidth-bound** and finishes in ~200 µs on CPU — any ANN library's own indexing overhead would be larger. The same code scales cleanly to ~100 k documents (~300 MB index, ~10 ms scan). Past that, swap `q @ doc_vecs.T` for FAISS `IndexFlatIP`; the interface doesn't change.

---

## Search

```python
q     = L2norm( encoder("<QRY> " + query + " </QRY>")[:, 0, :] )   # (1, 768)
sims  = (q @ doc_vecs.T).squeeze(0)                                 # (N,), in [-1, 1]
hits  = [(sims[i], pdf_paths[i]) for i in argsort(sims, descending=True)
                                  if sims[i] >= threshold][:top_k]
```

Both sides are unit-normalized, so the dot product *is* cosine similarity — the threshold (default `0.75`) has a stable, interpretable meaning across queries. It's a **floor on semantic relevance**, not a tuning knob. The top-k cap (default `5`) is a UX cap.

---

## Live indexing

The index isn't a static artifact. Upload a new PDF + title + abstract through the interface, and it's encoded with the same helper used at corpus-build time, appended to the in-memory `doc_vecs`, and persisted atomically — searchable on the very next query, no restart, no re-encode of the existing corpus.

```python
new_vec = encode_doc(model, tokenizer, device, title, text)        # (1, 768), L2-normed
with _lock:
    doc_vecs = torch.cat([doc_vecs, new_vec], dim=0)
    pdf_paths.append(save_path)
    torch.save({'doc_vecs': doc_vecs.cpu(), 'pdf_paths': pdf_paths}, encoded_path)
```

> The `threading.Lock` is the only thing standing between two concurrent uploads and a corrupted index file. Don't remove it.

<p align="center">
  <!-- TODO: replace with docs/add.png -->
  <img src="docs/add.png" width="80%" alt="Add document interface" />
  <br/><sub><i>Adding a new paper — the title and abstract are encoded and appended to the live index.</i></sub>
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
