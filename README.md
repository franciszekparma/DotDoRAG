<h1 align="center">DotDoRAG — A LoRA-Tuned Bi-Encoder for Clinical Literature Retrieval</h1>

<p align="center">
  <img src="images/rag_illustration.jpg" alt="SimpleRAG illustration" height="350" width="550">
</p>

<p align="center">
  <em>A small semantic search engine for medical / nutritional papers. A pretrained encoder is fine-tuned with LoRA so that queries and the documents that answer them land near each other in vector space, and retrieval at query time is a single matrix multiply against a pre-encoded corpus.</em>
</p>

<p align="center">
  <a href="#results"><img alt="NDCG@10" src="https://img.shields.io/badge/NDCG%4010-0.393-0ea5e9?style=flat-square"></a>
  <a href="#results"><img alt="vs BM25" src="https://img.shields.io/badge/vs%20BM25-%2B29%25-10b981?style=flat-square"></a>
  <a href="#results"><img alt="vs zero-shot" src="https://img.shields.io/badge/vs%20zero--shot%20BGE-%2B91%25-10b981?style=flat-square"></a>
  <img alt="Adapter size" src="https://img.shields.io/badge/adapter-13%20MB-94a3b8?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-94a3b8?style=flat-square">
</p>

---

## TL;DR

> Take a pre-trained transformer encoder, freeze it, glue on tiny LoRA adapters, and teach those adapters — with InfoNCE plus hard negatives plus teacher distillation — to map queries into the same neighborhood as their relevant documents. Encode the corpus *once*, search by a single matmul. On NFCorpus the result beats BM25 by **+29% NDCG@10** and nearly doubles zero-shot BGE.

**Dataset:** [BEIR / NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) — 3,633 medical/nutritional documents, 323 test queries, human-annotated relevance graded 0–2.

---

## Table of contents

1. [Why a bi-encoder](#why-a-bi-encoder)
2. [Architecture overview](#architecture-overview)
3. [Backbones](#backbones)
4. [Role tokens](#role-tokens)
5. [Pooling and normalization](#pooling-and-normalization)
6. [InfoNCE training objective](#infonce-training-objective)
7. [LoRA fine-tuning](#lora-fine-tuning)
8. [Hard negatives and teacher distillation](#hard-negatives-and-teacher-distillation)
9. [Corpus encoding and search](#corpus-encoding-and-search)
10. [Results](#results)
11. [Web UI — MedSearch](#web-ui--medsearch)
12. [Reproducing](#reproducing)
13. [Repository layout](#repository-layout)
14. [What's intentionally missing](#whats-intentionally-missing)

---

## Why a bi-encoder

Four options for *given a query and a corpus, return relevant documents*:

| Approach | Speed | Quality | Notes |
|---|---|---|---|
| BM25 / TF-IDF | Fast | OK | Lexical only collapses on paraphrase |
| Cross-encoder | Slow | Best | Re-runs the model per (query, doc) pair |
| **Bi-encoder** | **Fast** | **Good** | Encode once offline, matmul at query time |
| Generative RAG | — | — | Bi-encoder + LLM on top out of scope here |

Bi-encoder is the right trade-off at this scale: real transformer quality, but the heavy compute moves to a one-time encoding pass.

---

## Architecture overview

```
┌────────────┐                           ┌──────────────┐
│   query    │──► encoder ──► q ────┐    │  doc_vecs    │   pre-encoded
└────────────┘                      │    │  (N × d)     │   once
                                    ▼    └──────┬───────┘
                                 q · D.T        │
                                    │           │
                                    ▼           │
                              top-K cosine  ────┘
                                    │
                                    ▼
                                 results
```

At query time: encode the query once, compute one `(1 × d) @ (d × N)` matmul, sort, threshold. At ~3.6k docs this is sub-millisecond on CPU. For 10⁵+ docs you swap the matmul for FAISS/HNSW — the geometry stays the same.

---

## Backbones

Two encoders compared, both fine-tuned with LoRA:

- **BGE-small-en-v1.5** (`BAAI/bge-small-en-v1.5`) — ~33M params, already pretrained as a retrieval encoder on hundreds of millions of pairs. Strong zero-shot baseline.
- **Ettin-150M** (`jhu-clsp/ettin-encoder-150m`) — ModernBERT internals (rotary position embeddings, GeGLU, alternating local/global attention). Not pretrained specifically for retrieval we fine-tune it from a more general starting point.

Both are wrapped in a thin `nn.Module` (`src/model.py`) that runs the encoder on (query, positive, negative) batches and returns the first-position hidden state of each.

---

## Role tokens

Bi-encoders see two very different kinds of text — terse queries and long documents. Rather than letting the model infer that from surface form, we tell it directly with six new special tokens:

```
<QRY> ... </QRY>     for queries
<TLE> ... </TLE>     for titles
<TXT> ... </TXT>     for body text
```

The tokens are added to the tokenizer the embedding matrix is resized (`resize_token_embeddings`) the six new rows are random-init and learn during fine-tuning what each role should mean geometrically.

---

## Pooling and normalization

A variable-length sequence needs to become **one vector**. We use **CLS pooling** — `last_hidden_state[..., 0, :]` — because the contrastive loss trains that one slot to act as a summary register.

Then we **L2-normalize**. Two reasons:

1. **Dot product = cosine similarity** when both sides are unit norm:
   `cos(u, v) = (u · v) / (‖u‖ · ‖v‖) = u · v`. So `que_vec @ doc_vecs.T` *is* the cosine-similarity matrix.
2. **Better geometry.** The loss can only move *directions*, not magnitudes — the model can't cheat by inflating "important" vectors.

---

## InfoNCE training objective

For each query, sample one positive `p` from the qrels and `N` negatives `nᵢ`. With temperature `τ`:

$$
\mathcal{L} \=\ -\log \frac{\exp(q \cdot p / \tau)}{\exp(q \cdot p / \tau) + \sum_i \exp(q \cdot n_i / \tau)}
$$

Minimizing this simultaneously pushes `cos(q, p) → 1` (numerator up) and `cos(q, nᵢ) → −1` (denominator down).

The two distributions below are the actual `cos(q, d)` values measured on 120 sampled NFCorpus test queries — paired positives (green) against random negatives (red). The dashed lines mark the means.

<p align="center">
  <img src="plots/fig_similarity_dist.png" alt="Cosine similarity distributions of positive vs negative pairs, before and after fine-tuning" width="900">
</p>

The whole point of the loss is to widen `Δμ = mean(positive sim) − mean(negative sim)`. **Plain BGE gives `Δμ = +0.066` the LoRA-tuned model gives `Δμ = +0.095` — a 44% increase in the margin that drives ranking.**

**Temperature.** A smaller `τ` sharpens the softmax, amplifying similarity gaps. With **random** negatives we use a gentler `τ = 0.2` (Ettin recipe) with **hard-mined** negatives we drop to `τ = 0.02` (BGE recipe) since now even tiny margins are meaningful.

---

## LoRA fine-tuning

Full fine-tuning of 33M–150M params on 3,633 documents would overfit and steamroll the pretrained weights. **LoRA** freezes the backbone and adds trainable low-rank updates to each linear layer:

$$
y \=\ W x \+\ \frac{\alpha}{r} \cdot (B A)x
$$

- `A ∈ ℝ^{r×d_in}` is Gaussian-init, `B ∈ ℝ^{d_out×r}` is zero-init — so the adapter starts as a **no-op**.
- Empirically, fine-tuning updates `ΔW` tend to be intrinsically low-rank, so rank-16 or rank-32 is enough.
- Result: a **~13 MB** `adapter_model.safetensors` instead of a full checkpoint.

The high LR (1e-3) is normal for LoRA — adapters start at zero, only they get updated, and the frozen base can't be destabilized.

| | **Ettin LoRA** | **BGE LoRA** |
|---|---|---|
| Base model | Ettin-150M | BGE-small (~33M) |
| LoRA rank / α | 16 / 32 | 32 / 32 |
| Target modules | all-linear (Wi, Wo, Wqkv) | query / key / value / dense |
| Temperature `τ` | 0.2 | 0.02 |
| Negatives per query | 8 random | 7 hard + 1 random |
| Teacher distillation | — | KL from BGE-reranker-base |
| Effective batch | 2 | 64 × grad-accum 4 = 256 |
| Epochs | 200 | up to 30, patience 8 |

All hyperparameters live in `src/utils.py` so the recipes can be swapped without code edits.

---

## Hard negatives and teacher distillation

The BGE recipe stacks two upgrades on top of vanilla InfoNCE:

**Hard negatives.** Random negatives are trivially distinguishable from the positive — the loss saturates fast and the model never learns the *fine* distinctions. After each epoch, re-rank all non-positives by current similarity and resample the top-K as hard negatives. The mining is in `src/data.py`.

**Teacher distillation.** A cross-encoder (`BAAI/bge-reranker-base`) scores each (query, candidate) pair. Train the student's softmax over candidates to match the teacher's softmax via KL divergence:

```
loss = InfoNCE  +  KL_WEIGHT · KL(student_logits || teacher_logits)
```

with `KL_WEIGHT = 2.0`. The student learns *relative* candidate quality, not just "positive vs garbage."

The geometry change shows up clearly in 2-D too. Below is a t-SNE of real encoded queries (blue diamonds), their positives (green), and a pool of negatives (red), with each query joined to its positive by a thin line. The annotation reports the **actual** mean cosine similarity `cos(q, d⁺)` in the original embedding space (t-SNE coordinates themselves aren't comparable across panels):

<p align="center">
  <img src="plots/fig_embedding_tsne.png" alt="t-SNE of real query, positive, and negative embeddings before and after fine-tuning" width="900">
</p>

---

## Corpus encoding and search

`src/encode_corpus.py` runs every document through the trained model once, L2-normalizes, and saves:

```
corpus_encoded.pt  =  { doc_vecs: (N × d) tensor,  pdf_paths: [N] }
```

This is the whole point of a bi-encoder — trade one `O(N)` encoding pass for an `O(N·d)` matmul at every query.

`src/search.py`:

```python
que_vec = F.normalize(encoder(query), p=2, dim=1)   # (1, d)
sims    = (que_vec @ doc_vecs.T).squeeze(0)         # (N,)
hits    = [(s, p) for s, p in sorted(zip(sims, paths), reverse=True)
           if s >= threshold]
```

One matmul, one sort, one threshold filter.

---

## Results

NFCorpus test split — **323 queries · 3,633 corpus documents · top-10 evaluation.** Numbers come from `src/eval_compare.py` and are persisted in [`eval_results.json`](eval_results.json).

| Method | Base | Adapter | NDCG@10 | Recall@10 |
|---|---|---|---:|---:|
| **bge_lora** | BGE-small-en-v1.5 | `bge_lora/` | **0.3925** | **0.2013** |
| BM25 | — | — | 0.3052 | 0.1474 |
| plain_bge (zero-shot) | BGE-small-en-v1.5 | — | 0.2053 | 0.1011 |
| etin_lora | Ettin-150M | `etin_lora/` | 0.1016 | 0.0356 |
| plain_etin (zero-shot) | Ettin-150M | — | 0.0103 | 0.0015 |

<p align="center">
  <img src="plots/fig_metrics_bars.png" alt="NDCG vs Recall scatter" width="800">
  &nbsp;&nbsp;
  ---
  <img src="plots/fig_lift.png" alt="Lift over BM25" width="800">
</p>


**Takeaways**

- Fine-tuned BGE beats BM25 by **+8.7 NDCG points (+29%)** and nearly **2×** zero-shot BGE.
- NDCG@10 and Recall@10 move together — there is no precision-vs-recall trade-off here; better representations help both.
- Ettin had no retrieval pretraining; LoRA still gave a **~10×** lift, but the starting point matters more than any single training trick.
- The starting checkpoint dominates: *plain* BGE (33M params, retrieval-pretrained) beats *trained* Ettin (150M params, generic pretraining).

---

## Web UI — MedSearch

A Flask wrapper around the same encoder and the same `corpus_encoded.pt`. No new ML — just a UI, plus the ability to grow the index after training.

### Main page

<p align="center">
  <img src="imgs/main_page.png" alt="MedSearch main page" width="540">
</p>

Search bar with two knobs:

- **Top-K** — maximum number of results to return.
- **Threshold** — minimum cosine similarity to count as a hit.

### Searching

<p align="center">
  <img src="imgs/sample_search.png" alt="MedSearch search results" width="500">
</p>

Query → `<QRY>...</QRY>` → encoded → L2-normalized → `que_vec @ doc_vecs.T` → top-K. The score next to each result is raw cosine similarity. The query *"How to eat healthy"* pulls back *"Essentials of Healthy Eating: A Guide"* with only one lexical word in common — that is the encoder doing its job.

### Adding a document

<p align="center">
  <img src="imgs/doc_add.png" alt="MedSearch add-document form" width="540">
</p>

Three fields: title, abstract/text (this is what gets indexed; the PDF itself is **not** parsed), and the PDF file. On submit, `/add`:

1. Saves the PDF to `data/nfcorpus/pdf_docs/`.
2. Runs the same encoding path used at build time (`src/indexer.py`).
3. Appends the new vector to the in-memory matrix and persists `corpus_encoded.pt`, under a lock to avoid races.

From the next search onward the document is in the candidate pool. **No retraining** — the LoRA adapter stays put; we just extend the matrix the query is multiplied against.

---

## Reproducing

```bash
# 1) Install (CPU works; CUDA / Apple MPS auto-detected)
pip install -r requirements.txt

# 2) (optional) Re-train the BGE LoRA adapter
python src/train_bge_lora.py

# 3) (optional) Re-train the Ettin LoRA adapter
python src/train_etin_lora.py

# 4) Encode the corpus with the trained adapter
python src/encode_corpus.py

# 5) Run the full evaluation on NFCorpus test
python src/eval_compare.py     # writes eval_results.json

# 6) Regenerate README plots from eval_results.json
python src/make_plots.py       # writes plots/*.png

# 7) Launch the UI
python app/app.py              # http://127.0.0.1:5050
```

A pre-trained adapter (`bge_lora/`) and a pre-encoded corpus (`corpus_encoded.pt`) ship with the repo, so steps 2–4 are optional.

---

## Repository layout

```
illustrations/                    cover illustration
plots/                            research-style figures (generated)
imgs/                             UI screenshots

src/
  model.py                        Etin / BGE wrappers + tokenizer + role tokens
  data.py                         NFCorpusDataset, hard-negative mining
  utils.py                        all hyperparameters + paths (single source of truth)
  train_bge_lora.py               BGE LoRA + hard negatives + teacher KL distillation
  train_etin_lora.py              Ettin LoRA with random negatives
  encode_corpus.py                one-shot encoding pass → corpus_encoded.pt
  search.py                       CLI search with clickable PDF links
  indexer.py                      encode-and-append helper for the web UI
  eval_compare.py                 NDCG@10 / Recall@10 / ROC-AUC across methods
  make_plots.py                   generates the README figures

app/
  app.py                          Flask wrapper
  templates/                      MedSearch UI

bge_lora/                         trained BGE adapter (~13 MB)
etin_lora/                        trained Ettin adapter
corpus_encoded.pt                 { doc_vecs, pdf_paths }
eval_results.json                 latest eval numbers
data/nfcorpus/                    BEIR NFCorpus (corpus, queries, qrels, pdfs)
```

---

## What's intentionally missing

- **ANN index** (FAISS / HNSW). Needed past ~10⁵ documents — a plain matmul stops fitting in memory.
- **Cross-encoder reranker** over the top-K. Could close another chunk of the gap to the teacher.
- **PDF text extraction.** User-supplied abstracts beat blind page-1 slurps.
- **Domain-adaptive pretraining** of Ettin on biomedical text before LoRA. Probably the highest-leverage missing piece.

None of these change the skeleton — **encoder → contrastive loss → cached vectors → matmul.** That part is the same recipe most production retrieval stacks use.

---

<p align="center"><sub>Built on top of <a href="https://huggingface.co/BAAI/bge-small-en-v1.5">BGE-small</a>, <a href="https://huggingface.co/jhu-clsp/ettin-encoder-150m">Ettin</a>, <a href="https://github.com/huggingface/peft">PEFT</a>, and <a href="https://huggingface.co/datasets/BeIR/nfcorpus">BEIR NFCorpus</a>.</sub></p>
