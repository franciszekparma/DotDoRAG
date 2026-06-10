# SimpleRAG

**A from-scratch implementation of a domain-adapted dense retriever for scientific literature, written end-to-end in ~500 lines of PyTorch and Flask.**

SimpleRAG takes a generic 150M-parameter encoder, adapts it to the biomedical domain with a small LoRA module, encodes a corpus of PDFs into a flat tensor of unit vectors, and serves the whole system behind a Flask app that you can search, browse, and add new papers to in real time.

The project is small on purpose. Every component — the encoder choice, the role-marker tokens, the LoRA configuration, the contrastive loss, the index format, the serving path — is justified in this document. If you want to understand modern dense retrieval by reading one repository, this is meant to be that repository.

---

## Contents

1. [Motivation](#1-motivation)
2. [Problem formulation](#2-problem-formulation)
3. [The encoder](#3-the-encoder)
4. [Role-marker tokens](#4-role-marker-tokens)
5. [LoRA adaptation](#5-lora-adaptation)
6. [Training objective](#6-training-objective)
7. [Training procedure](#7-training-procedure)
8. [Index construction](#8-index-construction)
9. [Inference](#9-inference)
10. [The web application](#10-the-web-application)
11. [Quick start](#11-quick-start)
12. [Repository layout](#12-repository-layout)
13. [Hyperparameters](#13-hyperparameters)
14. [Limitations and what to improve next](#14-limitations-and-what-to-improve-next)
15. [References](#15-references)

---

## 1. Motivation

Information retrieval has a long lexical tradition. **BM25** and its relatives score a document by counting how often the query's tokens appear in it, weighted by inverse document frequency and length normalization. The recipe is simple, fast, and surprisingly hard to beat — *as long as the query and the document use the same words*.

In specialized corpora they often don't. A biomedical researcher might type:

> *"how does fasting affect blood sugar"*

while the relevant paper is titled:

> *"Intermittent caloric restriction and glycemic response in healthy adults."*

The vocabularies are disjoint; the meanings are identical. BM25 cannot bridge that gap on its own.

**Dense retrieval** does. The idea, made practical by [DPR (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906) and extended by [E5](https://arxiv.org/abs/2212.03533), [GTR](https://arxiv.org/abs/2112.07899), [BGE](https://arxiv.org/abs/2309.07597), and many others, is to learn a function `f : text → ℝᴰ` such that semantically related strings land near each other in `ℝᴰ`. Retrieval becomes nearest-neighbour search, and synonymy is handled implicitly by the geometry.

This sounds like neural-network solutionism, but two facts make it land:

1. **Documents are encoded once, offline.** A query is one forward pass plus a matrix multiply against a frozen tensor. A 100K-document index fits in a few hundred megabytes, and search is sub-millisecond on a single CPU.
2. **The function `f` does not need to be huge.** A well-trained 100–300M parameter encoder, specialized to the corpus through a small LoRA module, reaches retrieval quality competitive with models 10× the size.

SimpleRAG is the smallest end-to-end demonstration of that second fact that we could write down.

---

## 2. Problem formulation

We are given:

- A query string `q ∈ Σ*` from some vocabulary `Σ`.
- A corpus `D = {d₁, ..., dₙ}` of document strings.
- A relevance relation `R ⊂ Q × D` we want to approximate.

We seek an encoder `f_θ : Σ* → 𝕊ᴰ⁻¹` mapping strings to the unit sphere in `ℝᴰ` such that, for any query `q`,

```
                                                                         ⟨f_θ(q), f_θ(d)⟩
    arg max_{d ∈ D}  cos(f_θ(q), f_θ(d))    ≈    arg max_{d ∈ D}  P(d relevant to q)
```

Equivalently: we want `cos(f_θ(q), f_θ(d))` to rank documents in the same order as the (unknown) true relevance probability `P(R | q, d)`.

Two design decisions follow from this formulation:

- **Encoder symmetry.** A single shared `f_θ` is used for both queries and documents. This is the "bi-encoder" choice. A cross-encoder `g(q, d)` would be more accurate but quadratic at serve time; a two-tower model with separate `f_q, f_d` adds parameters without a clear win on small corpora.
- **Cosine on the unit sphere.** L2-normalizing both sides turns the inner product into the cosine, bounds it to `[-1, 1]`, and gives the similarity threshold a stable interpretation across queries. The loss must operate on the same normalized vectors that inference uses, otherwise the train and serve metrics diverge.

The asymmetry between queries (short, often interrogative) and documents (long, declarative) is not handled by separate encoders. It is handled by **role-marker tokens** — see §4.

---

## 3. The encoder

The backbone is [`jhu-clsp/ettin-encoder-150m`](https://huggingface.co/jhu-clsp/ettin-encoder-150m): a 150M-parameter encoder from Johns Hopkins' CLSP group, built on the [ModernBERT](https://arxiv.org/abs/2412.13663) architecture.

| Property | Value | Why it matters |
|---|---|---|
| Parameters | 150M | Fits comfortably in 8 GB of GPU memory with batch size 2–4 and gradient checkpointing off. |
| Hidden size `D` | 768 | Wide enough to carry semantic structure; the index is `(N, 768) × float32`. |
| Max position | 8192 | Lets full abstracts encode without truncation when needed. |
| Attention pattern | Alternating local/global | Quadratic only every few layers; the rest is windowed, so long context is cheap. |
| Positional encoding | RoPE | Stable extrapolation beyond training length. |

Sentence vectors are taken as the **[CLS] token's final hidden state**:

```
h     = encoder(tokens)              # (L, D)
e_cls = h[0]                         # (D,)
e     = e_cls / ‖e_cls‖₂             # (D,) — on 𝕊ᴰ⁻¹
```

Mean pooling is the common alternative and is sometimes marginally better. CLS pooling was chosen here for three reasons:

1. ModernBERT was pre-trained with a CLS-style objective; the token already carries a sentence-level summary.
2. CLS pooling has no length bias — mean pooling implicitly underweights long documents because each token contributes `1/L`.
3. It is the simplest possible aggregation, which matters when train and serve format any drift apart.

Device selection (`model.py`) is automatic: CUDA → MPS → CPU. All three are exercised in development.

---

## 4. Role-marker tokens

A query and a document are syntactically and semantically different objects. So is a document's title and its body. A naive bi-encoder ignores these distinctions: it sees only strings.

SimpleRAG injects them explicitly. Three pairs of special tokens are added to the tokenizer's vocabulary:

```
<QRY> ... </QRY>     queries
<TLE> ... </TLE>     document titles
<TXT> ... </TXT>     document body text
```

Concretely, the wrapping helper (`utils.AddTokens`) does:

```
query     →  "<QRY> how does fasting affect blood sugar </QRY>"
document  →  "<TLE> Intermittent caloric restriction </TLE> <TXT> ... </TXT>"
```

The embedding matrix is resized so the new tokens have their own learnable rows. The encoder then learns, during fine-tuning, to interpret a `<QRY>`-prefixed input differently from a `<TLE>`-prefixed input — even though the model architecture itself is symmetric.

This is the **prompted bi-encoder** trick. E5 and BGE use natural-language prefixes (`"query: ..."`, `"passage: ..."`) for the same purpose. Learned tokens are a cleaner version of the same idea: they cost two embedding rows each, they cannot collide with corpus vocabulary, and they make the role information unambiguous to the optimizer.

> **One source of truth.** Every string that enters the model goes through `AddTokens`. Training (`train.py`), corpus encoding (`encode_corpus.py`), and live indexing (`indexer.py`) all import the same helper. If train-time formatting and serve-time formatting drift apart, the index geometry breaks silently — same model, same weights, wrong answers. Centralizing the format is the cheapest insurance against this failure mode.

---

## 5. LoRA adaptation

Full fine-tuning of the 150M-parameter encoder would work. It would also (i) require optimizer state that does not fit in a typical consumer GPU, (ii) overwrite a perfectly good base model, and (iii) produce a 600 MB artifact for every domain you adapt to.

[Low-Rank Adaptation (LoRA; Hu et al., 2021)](https://arxiv.org/abs/2106.09685) is a parameter-efficient alternative. For every targeted linear layer `W ∈ ℝᵈ^out×ᵈ^in`, LoRA inserts a residual:

```
        W' x  =  W x  +  α/r · B A x
                            ↑      ↑
                         (d_out × r)  (r × d_in)
```

with `r ≪ min(d_in, d_out)`. The matrices `A, B` are trained; the original `W` is frozen. Initialization: `A ~ 𝒩(0, σ²)`, `B = 0`, so the model behaves identically to the base at step 0 and gradients flow through `B` from there.

The number of trainable parameters drops from `d_in · d_out` to `r · (d_in + d_out)` per layer. With `r = 16, α = 32` and `target_modules = "all-linear"`:

| | Full fine-tuning | LoRA |
|---|---:|---:|
| Trainable parameters | ~150M | ~1.5M |
| Optimizer state (AdamW, fp32) | ~1.2 GB | ~12 MB |
| Final artifact | ~600 MB | **~6 MB** |

Three configuration choices deserve commentary:

- **`target_modules = "all-linear"`.** Both attention projections (`Wqkv`, `Wo`) and MLP projections (`Wi`, `Wo`) are adapted. A common shortcut is to adapt only attention, but on retrieval the MLPs carry a substantial fraction of the useful semantic shift, and the parameter saving from skipping them is small.
- **`r = 16, α = 32`.** The effective scaling is `α/r = 2`. This is a standard middle ground: high enough that the adapter has meaningful influence early in training, low enough that it does not destabilize the optimization.
- **`lora_dropout = 0.065`.** Mild stochastic regularization on the adapter activations. With a small dataset like NFCorpus (~3.6K documents) and a high effective learning rate on the adapter, this materially helps generalization to the dev set.

The artifact at the end of training is a `peft`-compatible directory containing `adapter_config.json` and `adapter_model.safetensors`. At serve time the base model is loaded once and the adapter is layered on top with `PeftModel.from_pretrained` — no second copy of the weights, no extra memory.

---

## 6. Training objective

NFCorpus queries can have many relevant documents — in some cases 20 or more. The standard InfoNCE loss assumes exactly one positive per query, which makes it the wrong objective here. SimpleRAG uses a **multi-positive InfoNCE**, derived below.

### 6.1 Setup

For a batch of `B` queries, fix one query `q` with:

- Encoded vector `q ∈ 𝕊ᴰ⁻¹`.
- A set of positive documents with vectors `P = {p₁, ..., p_|P|} ⊂ 𝕊ᴰ⁻¹`.
- A set of negative documents with vectors `N = {n₁, ..., n_|N|} ⊂ 𝕊ᴰ⁻¹`.

All vectors are L2-normalized, so `⟨q, p⟩ = cos(q, p)`.

### 6.2 Standard InfoNCE (one positive)

With a single positive `p` and negatives `N`:

```
                exp(⟨q, p⟩ / τ)
L_InfoNCE = -log ──────────────────────────────
                exp(⟨q, p⟩ / τ) + Σ exp(⟨q, n⟩ / τ)
                                   n∈N
```

The expression inside the `log` is the softmax probability that the model assigns to `p` over the candidate set `{p} ∪ N`. Minimizing the negative log-likelihood pushes `⟨q, p⟩` up and `⟨q, n⟩` down.

### 6.3 Multi-positive generalization

When `|P| > 1`, two natural extensions exist:

- **Sum-of-log** (average the loss over positives): `−(1/|P|) Σ log softmax(p)` — treats each positive as a separate classification task.
- **Log-of-sum** (treat the union of positives as a single answer):

```
                Σ exp(⟨q, p⟩ / τ)
                p∈P
L = -log ──────────────────────────────────────────
         Σ exp(⟨q, p⟩ / τ)  +  Σ exp(⟨q, n⟩ / τ)
         p∈P                    n∈N
```

SimpleRAG uses the second form. The intuition: relevance is set-valued, so the model should be rewarded for ranking *any* positive above all negatives, not all of them. The sum-of-log version penalizes the model whenever positives have unequal scores even if the ranking is correct, which empirically hurts on NFCorpus-style multi-positive data.

### 6.4 Three details that matter

- **L2 normalize first, then dot.** Otherwise gradients scale with `‖q‖, ‖p‖`, which drift during training and destabilize the softmax. Normalizing also matches the inference metric exactly.
- **Temperature `τ = 0.2`.** Lower `τ` sharpens the softmax. At `τ → 0`, the loss becomes hinge-like and rewards only the top-scoring item; at `τ → ∞`, all items are treated as equally important and learning stalls. `τ ∈ [0.05, 0.5]` is the practical band; `0.2` is the centre of mass in the dense-retrieval literature.
- **`+ 1e-8` inside the log.** Pure numerical hygiene. Early in training, scores collapse and the softmax denominator can underflow. The epsilon costs nothing and avoids `log(0) = -∞`.

### 6.5 Negative sampling

SimpleRAG uses **8 in-batch-independent random negatives per query**, sampled uniformly from the corpus with rejection of any document in the query's positive set.

Random negatives are the weakest option. **Hard-negative mining** — sampling documents that the model currently scores high but are known to be irrelevant — gives a stronger learning signal and is the standard next upgrade. We omit it here because it adds a separate retrieval pass per epoch and complicates the training loop. The current loss is enough to demonstrate the method; the architecture supports the upgrade.

---

## 7. Training procedure

```
for each batch of (queries, positives, negatives):
    1. Tokenize each side with its role markers (<QRY>, <TLE>+<TXT>).
    2. Encode through the LoRA-adapted encoder.
    3. L2-normalize.
    4. Compute multi-positive InfoNCE.
    5. Backprop into adapter weights only (base is frozen).
    6. AdamW step + linear-warmup-then-decay scheduler step.
    7. On dev-loss improvement, save the adapter.
```

Concrete settings:

- **Optimizer:** AdamW, `lr = 1e-3`. The learning rate is unusually high for fine-tuning *because only the adapter moves*. Pre-trained base weights would not survive `1e-3`; freshly-initialized rank-16 matrices do, and they need it to escape the zero initialization of `B` quickly.
- **Schedule:** `get_linear_schedule_with_warmup` with `warmup_steps = 0.1 · total_steps`. Warmup matters more than usual because the adapter starts at zero and the gradients in the first hundred steps are mostly noise.
- **Batch size:** 2 queries per batch (each contributing 1 positive + 8 negatives, so 18 documents). Bi-encoder contrastive losses are bottlenecked by the number of negatives *seen*, not by the number of queries, and 200 epochs gives plenty of negative coverage.
- **Epochs:** 200, with best-on-dev checkpointing. The training set has ~110K (query, positive) pairs; convergence is well past the 50-epoch mark.
- **Checkpointing:** `model.enc.save_pretrained(...)` writes only the adapter — no base weights, no optimizer state. Resuming requires re-attaching the adapter with `PeftModel.from_pretrained`.

---

## 8. Index construction

Once the adapter is trained, `encode_corpus.py` produces the dense index:

```python
model.enc = PeftModel.from_pretrained(base_encoder, adapter_dir)
model.eval()

with torch.inference_mode():
    for batch in corpus_dataloader:
        text = "<TLE> {title} </TLE> <TXT> {body} </TXT>"
        e    = L2norm(model.enc(text).last_hidden_state[:, 0, :])    # (B, D)
        doc_vecs.append(e); pdf_paths.extend(batch.paths)

doc_vecs = torch.cat(doc_vecs).cpu()                                  # (N, D)
torch.save({'doc_vecs': doc_vecs, 'pdf_paths': pdf_paths},
           'corpus_encoded.pt')
```

### 8.1 Why a flat tensor, not FAISS

NFCorpus contains ~3.6K documents. A `(3600, 768)` float32 matrix is **11 MB**. A query–corpus matmul on CPU finishes in **~200 µs** and is bandwidth-bound, not compute-bound. Any approximate-nearest-neighbour library — FAISS, HNSW, ScaNN — would spend more time on its own indexing overhead than the brute-force scan takes.

The same code scales painlessly to ~100K documents (~300 MB index, ~10 ms scan on a single CPU core). At that point an `IndexFlatIP` from FAISS becomes the natural drop-in, and only one line changes:

```python
# from
sims = q @ doc_vecs.T

# to
sims, idxs = faiss_index.search(q.numpy(), top_k)
```

Beyond ~1M documents the brute-force scan stops being a good idea and an HNSW or IVF index is appropriate. SimpleRAG deliberately stays in the regime where the simplest data structure is also the fastest, and avoids dependencies that pay off only at much larger scale.

### 8.2 What's actually stored

```
corpus_encoded.pt
├── doc_vecs   : torch.float32, shape (N, 768), unit-norm rows
└── pdf_paths  : list[str], length N, aligned by row
```

Two parallel structures, one tensor for similarity and one list of strings for resolving a row index back to a file on disk. No metadata, no IDs — those live in `corpus.jsonl` and are joined on filename at serve time.

---

## 9. Inference

### 9.1 Scoring

```python
q     = "<QRY> " + user_query + " </QRY>"
q_vec = L2norm(encoder(q).last_hidden_state[:, 0, :])      # (1, D)
sims  = (q_vec @ doc_vecs.T).squeeze(0)                     # (N,)
order = torch.argsort(sims, descending=True)
```

The matrix multiply is the entire search. Because rows of `doc_vecs` are unit-norm and `q_vec` is unit-norm, `sims[i] = cos(q, dᵢ) ∈ [-1, 1]`.

### 9.2 Threshold and top-k

A returned hit must satisfy two conditions:

```
score ≥ threshold  AND  rank ≤ top_k
```

The threshold (default `0.75`) is a **floor on semantic relevance**, not a tuning knob — it has a stable meaning because both sides are unit-normalized and the encoder was trained with the same metric. Setting it too high returns no hits even when relevant documents exist; setting it too low fills the results with weakly related papers. Empirically `0.70–0.80` is the useful range for NFCorpus.

The top-k cap (default `5`) is a UX cap, not a quality cap. A user who wants every hit above the threshold can set `top_k = ∞`.

### 9.3 Two interfaces, one path

Both `search.py` (a terminal REPL with [OSC-8 hyperlinks](https://gist.github.com/egmontkob/eb114294efbcd5adb1944c9f3cb5feda) so PDF paths are Cmd/Ctrl-clickable in modern terminals) and `app.py` (the Flask server) go through the same encode → score → filter pipeline. They share the same `encode_doc` and `append_to_index` helpers from `indexer.py`. There is no parallel code path for the CLI vs the web app; both are thin wrappers around the same core.

---

## 10. The web application

`app.py` is a Flask app that loads the encoder, the adapter, and `corpus_encoded.pt` at startup and exposes five endpoints.

| Route | Method | Description |
|---|---|---|
| `/` | GET | Single-page search UI rendered from `templates/index.html`. |
| `/search` | POST | JSON `{query, top_k?, threshold?}` → JSON `{query, results: [{title, snippet, score, filename, pdf_url}]}`. |
| `/add` | POST | Multipart upload `{pdf, title?, text?}` → encodes the text, appends to the live index, returns the new `total_docs`. |
| `/pdf/<filename>` | GET | Serves a PDF from `PDF_DIR` with a path-traversal guard. |
| `/stats` | GET | `{total_docs: N}`. |

### 10.1 Live indexing

The interesting endpoint is `/add`. The handler:

1. Sanitizes the uploaded filename (`werkzeug.utils.secure_filename`) and disambiguates collisions with a numeric suffix.
2. Saves the PDF to `PDF_DIR`.
3. Encodes `<TLE> title </TLE> <TXT> text </TXT>` with the same `encode_doc` function used at corpus-build time.
4. Calls `append_to_index`, which under a `threading.Lock` appends the new row to the in-memory `doc_vecs`, appends the path to `pdf_paths`, and atomically rewrites `corpus_encoded.pt` to disk.
5. Updates the in-memory metadata index so the new document has a title and snippet immediately.

The lock guarantees that two simultaneous uploads cannot race on the file write. The new document is searchable on the very next `/search` call, with no restart and no re-encode of the existing corpus.

### 10.2 Security: path-traversal guard

`GET /pdf/<filename>` is the only endpoint that touches a user-supplied filename. Without protection, a request like `/pdf/../../../etc/passwd` would happily serve arbitrary files.

The guard is two checks:

```python
safe_dir = os.path.realpath(config.PDF_DIR)
target   = os.path.realpath(os.path.join(safe_dir, filename))
if not target.startswith(safe_dir + os.sep) or not os.path.isfile(target):
    abort(404)
```

`os.path.realpath` resolves symlinks so symlink-based attacks fail. The `startswith` check rejects any path that resolves outside `PDF_DIR`. The `isfile` check ensures we never serve a directory.

### 10.3 Upload cap

`MAX_CONTENT_LENGTH = 50 MB` rejects oversized uploads at the WSGI layer, before any code in the app runs. This prevents a trivial denial-of-service via large request bodies.

---

## 11. Quick start

### 11.1 Install

```bash
git clone https://github.com/franciszekparma/simplerag.git
cd simplerag
pip install torch transformers peft flask pandas tqdm
```

### 11.2 Data

NFCorpus, in its standard layout:

```
data/nfcorpus/
├── corpus.jsonl
├── queries.jsonl
├── qrels/
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
└── pdf_docs/
    ├── PMC1234567.pdf
    └── ...
```

### 11.3 Train the adapter

```bash
python src/train.py
# → checkpoints/epoch_K_train_X.XXXX_val_Y.YYYY/
```

### 11.4 Encode the corpus

```bash
python src/encode_corpus.py <checkpoint_dir> corpus_encoded.pt
```

### 11.5 Search

Terminal (Cmd/Ctrl-click results to open PDFs):

```bash
python src/search.py <checkpoint_dir> corpus_encoded.pt 0.75
```

Web app:

```bash
python app/app.py
# → http://127.0.0.1:5000
```

A pre-trained adapter and pre-encoded index are committed to the repository, so steps 11.3 and 11.4 are optional for a first run.

---

## 12. Repository layout

```
.
├── src/                        # all Python sources except the web app
│   ├── config.py               # paths, defaults, host/port
│   ├── model.py                # encoder, tokenizer, special tokens, device pick
│   ├── utils.py                # AddTokens — single source of truth for string format
│   ├── indexer.py              # encode_doc, thread-safe append_to_index
│   ├── train.py                # NFCorpus dataset, multi-positive InfoNCE, LoRA loop
│   ├── encode_corpus.py        # corpus → corpus_encoded.pt
│   └── search.py               # terminal REPL with OSC-8 clickable hyperlinks
├── app/                        # Flask web app
│   ├── app.py                  # /search, /add, /pdf, /stats
│   └── templates/index.html    # single-page UI
├── adapter_config.json         # LoRA config (peft)
├── adapter_model.safetensors   # trained LoRA weights (~6 MB)
├── corpus_encoded.pt           # {doc_vecs: (N, D) float32, pdf_paths: [str]}
└── data/nfcorpus/              # corpus, queries, qrels, PDFs
```

---

## 13. Hyperparameters

All tunables live in [`config.py`](config.py) and [`train.py`](train.py).

**Encoder**
| Parameter | Value |
|---|---|
| Base model | `jhu-clsp/ettin-encoder-150m` |
| Hidden size `D` | 768 |
| Pooling | CLS (`hidden[..., 0, :]`) |
| Max sequence length (query / doc) | 256 / 256 |

**LoRA**
| Parameter | Value |
|---|---|
| Rank `r` | 16 |
| Scaling `α` | 32 (effective `α/r = 2`) |
| Adapter dropout | 0.065 |
| Target modules | `all-linear` (Wqkv, Wo, Wi) |
| Bias mode | none |
| Trainable parameters | ~1.5M of ~150M |

**Training**
| Parameter | Value |
|---|---|
| Loss | Multi-positive InfoNCE (log-of-sum) |
| Temperature `τ` | 0.20 |
| Negatives per query | 8, sampled uniformly with rejection |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Schedule | Linear warmup 10% → linear decay |
| Batch size (queries) | 2 |
| Epochs | 200 |
| Checkpoint selection | Lowest dev loss |

**Retrieval**
| Parameter | Value |
|---|---|
| Similarity | Cosine (= dot on unit vectors) |
| Default threshold | 0.75 |
| Default top-k | 5 |
| Upload cap | 50 MB |

---

## 14. Limitations and what to improve next

The point of this section is to be honest about what SimpleRAG is *not*, and what it would take to go further.

- **Random negatives are the weakest contrastive signal.** Hard negatives — sampled by retrieving with the current model and rejecting known positives — measurably improve nDCG@10 in every published comparison. Adding a periodic re-mining pass to `train.py` is the highest-leverage upgrade.
- **NFCorpus only.** The encoder, the role tokens, and the loss are all corpus-agnostic. Swapping in MS MARCO, BEIR, or a private corpus is a dataset-class change in `train.py` and `encode_corpus.py`; nothing else needs to move.
- **No re-ranker.** Bi-encoders trade quality for speed. The standard fix is a small cross-encoder re-ranker on the top-50 hits — adds 50 forward passes per query (still fast) and recovers most of the gap to a pure cross-encoder.
- **Brute-force search.** Correct and fast up to ~100K documents. Past that, swap `q @ doc_vecs.T` for FAISS (`IndexFlatIP` first, `IndexHNSWFlat` if memory becomes tight).
- **No evaluation script.** Training reports train/dev loss but not nDCG, Recall@K, or MRR. A `evaluate.py` that runs the standard BEIR metrics on the test qrels is a worthwhile addition.
- **No chunking for long PDFs.** Each PDF is encoded by its title + abstract only. For full-text retrieval you would chunk the body into ~512-token windows, encode each, and either max-pool the scores per document or treat each chunk as an independent retrieval unit.

None of these are flaws in the *method*. They are scope decisions for a self-contained reference implementation.

---

## 15. References

**Dense retrieval.**
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- Wang, L., et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5).* [arXiv:2212.03533](https://arxiv.org/abs/2212.03533)
- Xiao, S., et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE).* [arXiv:2309.07597](https://arxiv.org/abs/2309.07597)

**Contrastive learning.**
- van den Oord, A., et al. (2018). *Representation Learning with Contrastive Predictive Coding (InfoNCE).* [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
- Khosla, P., et al. (2020). *Supervised Contrastive Learning* (multi-positive formulation). [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)

**Encoder architectures.**
- Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Warner, B., et al. (2024). *ModernBERT: Smarter, Better, Faster, Longer.* [arXiv:2412.13663](https://arxiv.org/abs/2412.13663)

**Parameter-efficient adaptation.**
- Hu, E., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

**Dataset.**
- Boteva, V., Gholipour, D., Sokolov, A., Riezler, S. (2016). *A Full-Text Learning to Rank Dataset for Medical Information Retrieval (NFCorpus).* [Paper](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

---

## License

MIT &copy; franciszekparma
