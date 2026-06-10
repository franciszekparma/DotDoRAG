# SimpleRAG

Semantic search over a personal library of scientific PDFs. You type a question, it returns the papers that actually answer it — not the ones that happen to share keywords.

The whole system is three artifacts: a **6 MB LoRA adapter** that teaches a small encoder to read biomedical text the way a researcher would, a **flat tensor of document vectors** that takes a single matmul to search, and a **Flask app** that ties them together and lets you drop new PDFs into the index without a restart.

---

## What a query looks like

```
> fasting and blood glucose regulation

  0.842   Effects of intermittent caloric restriction on glycemic response
  0.811   Time-restricted feeding and insulin sensitivity in adults
  0.793   Postprandial glucose dynamics under prolonged fasting
  0.778   Metabolic adaptations during alternate-day fasting
  ...
```

The word *fasting* doesn't appear in the first three titles. That's the whole point — lexical search would have missed them. A bi-encoder fine-tuned on a domain corpus learns that *fasting*, *caloric restriction*, *time-restricted feeding*, and *prolonged fasting* live in the same neighbourhood of the vector space.

---

## The shape of the system

```
                                  ┌─────────────────────┐
                                  │  base encoder       │
                                  │  ettin-150m (≈600MB)│
                                  └──────────┬──────────┘
                                             │
   train.py ─────────── LoRA fine-tune ──────┤
   (NFCorpus + qrels)                        │
                                             ▼
                                  ┌─────────────────────┐
                                  │  adapter (≈6 MB)    │ ◄── the only thing you ship from training
                                  └──────────┬──────────┘
                                             │
   encode_corpus.py ──────── encode docs ────┤
   (corpus.jsonl + PDFs)                     │
                                             ▼
                                  ┌─────────────────────┐
                                  │  corpus_encoded.pt  │   doc_vecs : (N, D)
                                  │                     │   pdf_paths: [str]·N
                                  └──────────┬──────────┘
                                             │
   app.py ───────────────── load + serve ────┤
                                             ▼
                              query  ─►  q · doc_vecsᵀ  ─►  top-k PDFs
```

Each box on the right is a file on disk. The left side is the code that produces it.

---

## Stage 1 — Training the retriever

The retriever is a **bi-encoder**: queries and documents go through the same model independently, producing two vectors that are scored by cosine similarity. Independence is the whole reason this is fast at serve time — documents get encoded *once*, into a frozen matrix, and a query is one vector against that matrix.

### Backbone

[`jhu-clsp/ettin-encoder-150m`](https://huggingface.co/jhu-clsp/ettin-encoder-150m) — a 150M-parameter ModernBERT-style encoder. Big enough to learn useful semantics, small enough to fine-tune on a laptop GPU, and built with retrieval in mind (alternating local/global attention, RoPE, long context).

Sentence vectors are the **CLS token's** final hidden state. One vector per input, no pooling decisions to second-guess at serve time.

### Role-marker tokens

Before tokenization, every string is wrapped in one of three learned special tokens:

```
<QRY> what role does TGF-beta play in fibrosis </QRY>
<TLE> TGF-β signalling in pulmonary fibrosis </TLE>  <TXT> ... </TXT>
```

These tokens are added to the vocabulary and the embedding matrix is resized. They give the model an explicit, learned signal of which side of the relationship a piece of text belongs to (query vs. document) and which structural slot it fills (title vs. body). E5 and BGE use natural-language prefixes for the same purpose; learned tokens are a cleaner version of the same idea.

All wrapping flows through a single `AddTokens` helper used by training, indexing, and serving alike. If train and serve format the same string differently, the geometry breaks silently — so the helper has exactly one source of truth.

### Why LoRA, not full fine-tuning

Full fine-tuning would work. It would also produce a 600 MB artifact, require an optimizer state that doesn't fit on most consumer GPUs, and overwrite a perfectly good base model.

LoRA adds a small low-rank residual `ΔW = BA` to every linear layer and trains only those matrices. Three consequences:

1. **~1% of the parameters move.** Optimizer state shrinks accordingly.
2. **The artifact is the diff.** The base encoder stays on Hugging Face; you ship a 6 MB `safetensors` file.
3. **Adapter hot-swapping.** You can train one adapter per domain (biomed, legal, code) and load the right one at serve time without touching the base weights.

Concrete config:

```python
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.065,
           target_modules="all-linear", bias="none")
```

`all-linear` adapts every projection — attention `Wqkv`/`Wo` *and* MLP `Wi`/`Wo`. Adapting only attention is a common shortcut that costs noticeable quality on retrieval; the MLPs carry a lot of the semantic shift you actually want.

### The loss: multi-positive InfoNCE

NFCorpus queries can have ten or more relevant documents. Standard InfoNCE assumes one. The fix is to put *all* known positives into the numerator:

```
            Σ exp(q·p / τ)
            p∈P
L = -log ─────────────────────────
         Σ exp(q·p / τ)  +  Σ exp(q·n / τ)
         p∈P                 n∈N
```

Three details that matter:

- **L2-normalize before the dot product.** Inner products become cosine similarities, gradients stay well-scaled, and the training metric matches the serving metric exactly.
- **Temperature `τ = 0.2`.** Sharpens the softmax; the model is pushed to make hard discriminations between positives and negatives.
- **8 random negatives per query.** Random is the weak option — hard-mined negatives would be better — but it requires zero extra infrastructure, which keeps the project a weekend project.

### Training loop in seven lines

```python
for q, P, N in dataloader:                 # q: query text, P: positives, N: negatives
    q_vec, p_vecs, n_vecs = model(q, P, N) # all CLS-pooled, all (..., D)
    loss = multi_positive_infonce(q_vec, p_vecs, n_vecs, tau=0.2)
    loss.backward()
    optimizer.step(); scheduler.step()
    optimizer.zero_grad()
    if dev_loss < best: model.enc.save_pretrained(ckpt_dir)
```

AdamW at `1e-3` (high, but only adapter weights move), linear warmup over 10% of steps then linear decay. The artifact at the end is a directory containing `adapter_config.json` and `adapter_model.safetensors`.

---

## Stage 2 — Building the index

`encode_corpus.py` loads the base encoder, layers the trained adapter on top with `PeftModel.from_pretrained`, and encodes every document in the corpus that has a matching PDF on disk:

```
for doc in corpus.jsonl:
    if PDF exists:
        text = <TLE>title</TLE> <TXT>body</TXT>
        v    = L2norm(encoder(text))             # (D,)
        doc_vecs.append(v); pdf_paths.append(path)

save({'doc_vecs': stack(doc_vecs),                # (N, D) float32
      'pdf_paths': pdf_paths},                    # list[str], aligned by row
     'corpus_encoded.pt')
```

### Why a flat tensor, not FAISS

NFCorpus has ~3.6k documents. A `(3600, 768)` float32 matrix is 11 MB. The matmul against a single query vector runs in microseconds on CPU and is bandwidth-bound, not compute-bound. Any ANN library would spend more time on its own overhead than the brute-force scan takes.

The same code scales cleanly to ~100k documents. Beyond that, swapping `q @ doc_vecs.T` for a FAISS `IndexFlatIP` or `IndexHNSWFlat` is a few lines — but the *interface* (`encode_query` → similarity → top-k) doesn't change.

---

## Stage 3 — Serving

`app.py` is a Flask app that holds `doc_vecs` and `pdf_paths` in memory and exposes five endpoints.

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Single-page search UI |
| `/search` | POST | Encode query, score against `doc_vecs`, return top-k above threshold as JSON |
| `/add` | POST | Multipart upload — saves the PDF, encodes title+text, appends to the live index, persists |
| `/pdf/<filename>` | GET | Serves a PDF, with a real-path check that pins it inside `PDF_DIR` |
| `/stats` | GET | `{"total_docs": N}` |

### Query path

```python
q     = L2norm(encoder("<QRY> " + query + " </QRY>"))   # (1, D)
sims  = (q @ doc_vecs.T).squeeze(0)                      # (N,)
order = sims.argsort(descending=True)

results = []
for i in order:
    if sims[i] < threshold: break
    results.append(pdf_paths[i])
    if len(results) >= top_k: break
```

Because both sides are unit-normalized, the dot product *is* the cosine similarity. The threshold (default `0.75`) has a stable, interpretable meaning across queries — it isn't a per-query calibration knob.

### Live index updates

`/add` takes a PDF + title + abstract, encodes the text the same way training did, and appends one row to the in-memory `doc_vecs` and one entry to `pdf_paths`. The updated tensor is then rewritten to `corpus_encoded.pt` under a `threading.Lock` so concurrent uploads can't race on the file. New documents are searchable on the very next query — no restart, no re-encode of the corpus.

### Path-traversal guard

The PDF serving route accepts a filename from the URL, which means it has to be hardened against `../../etc/passwd`-style inputs:

```python
safe_dir = os.path.realpath(config.PDF_DIR)
target   = os.path.realpath(os.path.join(safe_dir, filename))
if not target.startswith(safe_dir + os.sep):
    abort(404)
```

Real-path resolution defeats symlink games; the prefix check defeats traversal. Anything that lands outside `PDF_DIR` is a 404, not a leak.

---

## Run it

```bash
git clone https://github.com/franciszekparma/simplerag.git
cd simplerag
pip install torch transformers peft flask pandas tqdm
```

Drop the standard NFCorpus layout under `data/nfcorpus/`: `corpus.jsonl`, `queries.jsonl`, `qrels/{train,dev,test}.tsv`, and `pdf_docs/*.pdf`.

```bash
python train.py                                         # train LoRA adapter
python encode_corpus.py <ckpt_dir> corpus_encoded.pt    # build the index
python search.py <ckpt_dir> corpus_encoded.pt 0.75      # terminal search
python app.py                                           # web app → http://127.0.0.1:5000
```

A pre-trained adapter and pre-encoded index are committed to the repo, so you can skip the first two steps and go straight to `app.py`.

---

## Hyperparameters

All in [`config.py`](config.py) and [`train.py`](train.py).

**Encoder & retrieval**
| | |
|---|---|
| Base model | `jhu-clsp/ettin-encoder-150m` |
| Embedding dim | 768 |
| Pooling | CLS |
| Max sequence length | 256 (query and doc) |
| Similarity | Cosine (= dot product on unit vectors) |
| Default top-k / threshold | 5 / 0.75 |

**LoRA**
| | |
|---|---|
| Rank `r` / `alpha` | 16 / 32 |
| Dropout | 0.065 |
| Target modules | all-linear |

**Training**
| | |
|---|---|
| Loss | Multi-positive InfoNCE, `τ = 0.2` |
| Negatives | 8 random per query |
| Optimizer | AdamW, `lr = 1e-3` |
| Schedule | Linear warmup (10%) → linear decay |
| Batch size / epochs | 2 / 200 |

---

## Layout

```
.
├── model.py                    # encoder, tokenizer, special tokens, device selection
├── utils.py                    # AddTokens — the only place strings get wrapped
├── train.py                    # NFCorpus dataset, multi-positive InfoNCE, LoRA loop
├── encode_corpus.py            # corpus → corpus_encoded.pt
├── indexer.py                  # encode_doc + thread-safe append_to_index
├── search.py                   # terminal REPL with clickable OSC-8 PDF links
├── app.py                      # Flask app: search, /add, PDF serving
├── config.py                   # paths, defaults, host/port
├── templates/index.html        # single-page UI
├── adapter_config.json         # LoRA config
├── adapter_model.safetensors   # trained LoRA weights (≈ 6 MB)
├── corpus_encoded.pt           # {doc_vecs: (N, D), pdf_paths: [str]}
└── data/nfcorpus/              # corpus, queries, qrels, PDFs
```

---

## References

- Karpukhin et al. (2020). [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Wang et al. (2022). [Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)](https://arxiv.org/abs/2212.03533)
- Warner et al. (2024). [ModernBERT: Smarter, Better, Faster, Longer](https://arxiv.org/abs/2412.13663)
- Boteva et al. (2016). [NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

---

## License

MIT &copy; franciszekparma
