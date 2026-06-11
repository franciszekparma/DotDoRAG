# SimpleRAG

<!-- TODO: hero image goes here -->
<p align="center">
  <!-- Drop a banner / screenshot here, e.g.: -->
  <!-- <img src="imgs/banner.png" alt="SimpleRAG" width="800"> -->
</p>


A small, self-contained semantic retrieval system for clinical literature. A pretrained transformer encoder is fine-tuned with LoRA so that **questions and the documents that answer them land near each other in vector space**, and search at query time collapses to a single matrix multiplication against a pre-encoded corpus.

The dataset is [BEIR / NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) — about 3.6k medical / nutritional documents paired with natural-language queries and human-annotated relevance labels.
>>>>>>> af2015b (Added sample photos and README.md)

This README is written more as an explanation of *why* every piece of the pipeline looks the way it does than as a quickstart. It walks through the backbone, the role tokens, the contrastive objective, LoRA, the encoding pass, retrieval, and finally the Flask UI that wraps everything for end users. There is some math, but it is meant to make the code make sense, not to be rigorous.

---

## Table of contents

1. [The shape of the problem](#1-the-shape-of-the-problem)
2. [Why a bi-encoder (and not a cross-encoder, BM25, or generative RAG)](#2-why-a-bi-encoder-and-not-a-cross-encoder-bm25-or-generative-rag)
3. [The backbone: Ettin-150M / ModernBERT](#3-the-backbone-ettin-150m--modernbert)
4. [Role tokens: telling the model what it is reading](#4-role-tokens-telling-the-model-what-it-is-reading)
5. [Producing a single vector per text](#5-producing-a-single-vector-per-text)
6. [The training objective: multi-positive InfoNCE](#6-the-training-objective-multi-positive-infonce)
7. [Parameter-efficient fine-tuning with LoRA](#7-parameter-efficient-fine-tuning-with-lora)
8. [The training loop in detail](#8-the-training-loop-in-detail)
9. [Encoding the corpus once](#9-encoding-the-corpus-once)
10. [Search: cosine similarity as a matmul](#10-search-cosine-similarity-as-a-matmul)
11. [Repository layout](#11-repository-layout)
12. [Running it end to end](#12-running-it-end-to-end)
13. [Addition: the MedSearch web UI](#13-addition-the-medsearch-web-ui)
14. [Things that could be improved](#14-things-that-could-be-improved)

---

## 1. The shape of the problem

The task is **information retrieval**: given a natural-language query, return the documents from a fixed corpus that are most relevant to it. NFCorpus gives us three things we can use:

- A **corpus** of about 3.6k documents, each with a title and a body (`data/nfcorpus/corpus.jsonl`).
- A set of **queries** in plain English (`data/nfcorpus/queries.jsonl`).
- **qrels** — query-to-document relevance judgements made by humans, split into `train` / `dev` / `test` (`data/nfcorpus/qrels/*.tsv`).

A query like "How to eat healthy" should pull back things like "Essentials of Healthy Eating: A Guide" even though the surface words barely overlap. That is the whole point of *semantic* search: matching on meaning, not on lexical overlap.

The classical lexical baseline for this is BM25, which scores a document by how often the query terms appear in it, weighted by inverse document frequency. BM25 is shockingly hard to beat on short, keyword-shaped queries — but it fails on paraphrases, synonyms, and questions like "what helps people stop being depressed" vs. a document titled "Efficacy of SSRIs in major depressive disorder." That is the regime where dense retrieval shines, and it is the regime we are training for.

## 2. Why a bi-encoder (and not a cross-encoder, BM25, or generative RAG)

There are roughly four standard architectures for "given a query and a corpus, return the relevant documents":

1. **Lexical retrieval (BM25 / TF-IDF).** Fast, strong baseline, zero training. Bad at paraphrases and synonyms.
2. **Cross-encoder.** Feed `[CLS] query [SEP] document [SEP]` jointly into a transformer and score the pair. *Very* accurate, *very* slow — you would have to run the model once per (query, document) pair at query time. Unusable for corpus-wide search past a few hundred docs.
3. **Bi-encoder (dual encoder).** Encode the query and the document *separately* into fixed-size vectors, and use a cheap similarity (cosine / dot product) between them. The documents can be encoded once, offline, into a matrix; search at query time is a single matmul. This is what this project uses.
4. **Generative RAG.** A bi-encoder is used to retrieve, then a language model conditions on the retrieved passages and writes an answer. The "RAG" in the project's name is mostly aspirational here — this repo builds the retrieval half cleanly, and a generation head could be bolted on top without changing anything below.

The bi-encoder trade-off is the right one for this scale: we keep almost all the accuracy of a real transformer reading the text, while moving the heavy compute to a one-time encoding pass.

A common pipeline in production is **retrieve with a bi-encoder, then rerank the top 100 with a cross-encoder**. That is a natural next step but is intentionally out of scope here.

## 3. The backbone: Ettin-150M / ModernBERT

The encoder is [`jhu-clsp/ettin-encoder-150m`](https://huggingface.co/jhu-clsp/ettin-encoder-150m), a 150M-parameter ModernBERT-architecture model from JHU's CLSP group. The choice is deliberate:

- **It is small enough to fine-tune on a single GPU (or even MPS)** while still being competitive on retrieval benchmarks. Larger encoders exist, but the marginal accuracy is rarely worth the wall-clock cost on a corpus this size.
- **ModernBERT internals are a meaningful upgrade over original BERT.** Specifically:
  - **Rotary position embeddings (RoPE)** instead of learned absolute positions. RoPE rotates the query/key vectors by a position-dependent angle so that the dot product between two tokens depends on their *relative* offset, which generalizes better to lengths the model did not see during pretraining.
  - **GeGLU activations** in the feed-forward block (a gated variant of GELU) — better optimization, better quality per parameter.
  - **Alternating local / global attention.** Most layers only attend to a sliding window, with a few global-attention layers interspersed. This makes long-context inference cheaper without giving up much modeling power.
  - **Larger vocabulary and tokenizer choices** that are friendlier to scientific text.
- **It was pretrained with retrieval in mind.** Ettin is an encoder/decoder pair released together, and the encoder half was explicitly evaluated as a retrieval backbone, so its representations are already in the right neighborhood for what we want.

In code (`src/model.py`):

```python
class RAGEtin(nn.Module):
    def __init__(self, name="jhu-clsp/ettin-encoder-150m"):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)

    def forward(self, queries, positives, negatives):
        return (
            self.enc(**queries  ).last_hidden_state[..., 0, :],
            self.enc(**positives).last_hidden_state[..., 0, :],
            self.enc(**negatives).last_hidden_state[..., 0, :],
        )
```

The forward pass takes the three batches the contrastive loss needs (query, positive doc, negative docs), runs each through the same shared encoder, and returns the first-position hidden state of each — see §5 for why position 0 specifically.

## 4. Role tokens: telling the model what it is reading

A bi-encoder has a subtle problem: the same model is asked to encode two very different kinds of text — terse queries ("how to eat healthy") and long documents ("Essentials of Healthy Eating: A Guide. Background. The overconsumption of energy dense foods..."). If we just shove them in raw, the encoder has to *infer* from surface form whether it is reading a question or a passage, and whether a chunk of text is a title or a body.

We give it a hard signal instead, via six new special tokens (`src/utils.py`):

```python
new_tokens = {
    'query_begin': '<QRY>', 'query_end': '</QRY>',
    'title_begin': '<TLE>', 'title_end': '</TLE>',
    'text_begin':  '<TXT>', 'text_end':  '</TXT>',
}
```

Every input is wrapped before tokenization:

- Queries become `<QRY> how to eat healthy </QRY>`.
- Documents become `<TLE> Essentials of Healthy Eating </TLE> <TXT> Background. The overconsumption ... </TXT>`.

The six tokens are added to the tokenizer and the encoder's embedding matrix is resized to make room for them (`src/model.py`):

```python
tokenizer.add_tokens(list(token_adder.new_tokens.values()))
model = RAGEtin()
model.enc.resize_token_embeddings(len(tokenizer))
```

`resize_token_embeddings` extends the input embedding (and tied output embedding, if any) by six rows, initialized randomly. Those rows then learn during fine-tuning what "this is a query" and "this is a title" should *mean* in the model's representation space — which is exactly what we want a bi-encoder to specialize on.

This is a very common trick in retrieval. Some encoders use a different prefix (`"query: ..."` / `"passage: ..."`), some use a special token like `[Q]`. The mechanism is the same: condition the encoder on what role the input is playing.

## 5. Producing a single vector per text

We need to turn a variable-length sequence of token vectors into a single fixed-size vector per input. The options are:

- **CLS pooling.** Use the hidden state at position 0. Works well when the model was pretrained with a `[CLS]`-like sentinel and is the standard choice for BERT-family retrieval encoders.
- **Mean pooling.** Average the hidden states across all (non-padding) tokens. Often slightly better when the backbone was not pretrained with a CLS objective.
- **Max pooling.** Element-wise max. Less common.
- **Attention pooling.** A learned weighted sum. More parameters, marginal wins.

This project uses **CLS pooling**:

```python
self.enc(**queries).last_hidden_state[..., 0, :]
```

`last_hidden_state` has shape `(batch, seq_len, hidden_dim)`. Indexing `[..., 0, :]` selects position 0 across the whole batch, giving `(batch, hidden_dim)`. ModernBERT-style models have a CLS-equivalent first token, and CLS pooling has the nice property that it produces a vector that is *learned* end-to-end against the contrastive loss — the model can use that one slot as a summary register because it knows that is the only slot the loss will look at.

The vectors are then L2-normalized before being compared:

```python
F.normalize(vec, p=2, dim=1)
```

L2-normalizing puts every vector on the unit hypersphere ‖v‖₂ = 1, which has two big consequences:

1. **Dot products become cosine similarities.** `cos(u, v) = (u · v) / (‖u‖ · ‖v‖)`, and if both norms are 1 then `cos(u, v) = u · v`. So `que_vec @ doc_vecs.T` is *literally* the matrix of cosine similarities, no extra division required.
2. **The optimization geometry is nicer.** The loss only cares about direction, not magnitude, so the model cannot cheat by making "important" documents have a bigger norm.

## 6. The training objective: multi-positive InfoNCE

The loss lives in `src/train.py` as `MultiNCELoss`. The idea, in one sentence: **for every query, pull the relevant document(s) closer in cosine space and push the irrelevant ones away.**

### 6.1 What goes into a single training example

For each query `q` (from `NFCorpusDataset`):

1. Look up the set of documents the qrels mark as relevant — call them `P` (positives).
2. Randomly sample **one** positive `d⁺` from `P`.
3. Randomly sample **N = 8** documents from the rest of the corpus as negatives `d⁻₁, …, d⁻₈`. (Documents in `P` are excluded.)
4. Wrap each in role tokens, tokenize, and pass to the encoder.

This is the "**random negatives**" / "in-batch negatives" family of training signals. There are stronger variants (BM25-mined hard negatives, model-mined hard negatives, ANCE-style refreshed negatives) — those would be the obvious upgrade, but random negatives are cheap and good enough to get strong representations on a corpus this size.

### 6.2 The math

Let `q` be the L2-normalized query vector, `p` be the L2-normalized positive document vector, and `nᵢ` (for `i = 1..N`) be the L2-normalized negative document vectors. Since everything is unit norm, `q · v` *is* the cosine similarity `s(q, v) ∈ [-1, 1]`.

Define logits with temperature `τ`:

```
ℓ⁺  =  (q · p) / τ
ℓ⁻ᵢ =  (q · nᵢ) / τ
```

The per-query loss is the negative log of the softmax-like ratio

```
        exp(ℓ⁺)
L = -log ─────────────────────────────
         exp(ℓ⁺) + Σᵢ exp(ℓ⁻ᵢ)
```

This is **InfoNCE** (van den Oord et al., 2018), the workhorse loss of contrastive learning. Minimizing it does exactly two things at once:

- Pushes `cos(q, p)` toward 1 (numerator goes up).
- Pushes each `cos(q, nᵢ)` toward −1 (denominator goes down).

In words: relevant pair high, irrelevant pairs low. That is the whole game.

### 6.3 Temperature: why τ = 0.2

`τ` is a softmax temperature. Smaller `τ` makes the loss "sharper" — it cares much more about the *gap* between the positive and the hardest negative, because dividing by a small number blows small similarity differences up into large logit differences. Larger `τ` smooths things out, so the model gets less aggressive feedback per step.

Common values in dense retrieval sit between 0.01 and 0.1; this code uses 0.2, which is on the gentler end. Gentler temperature pairs naturally with random (rather than hard-mined) negatives — when the negatives are mostly trivial, an aggressive temperature would over-fit to easy contrasts and starve the model of useful signal.

### 6.4 Multi-positive generalization

The implementation is written to allow multiple positives per query, even though in practice we sample one. If `P = {p₁, …, p_K}`, the loss generalizes to

```
         Σⱼ exp(ℓ⁺ⱼ)
L = -log ────────────────────────────
        Σⱼ exp(ℓ⁺ⱼ) + Σᵢ exp(ℓ⁻ᵢ)
```

You can read this off `MultiNCELoss.calc_loss`:

```python
pos_exp = torch.exp(pos_logits)
neg_exp = torch.exp(neg_logits)
numer = pos_exp.sum(dim=-1)
denom = numer + neg_exp.sum(dim=-1)
loss  = -torch.log((numer + 1e-8) / (denom + 1e-8))
```

The `1e-8` is a numerical guard against log(0). The sum-over-positives in the numerator is the multi-positive generalization, which collapses to standard InfoNCE when `K = 1`.

> *Note:* a strictly more numerically stable formulation would use `logsumexp` instead of `log(sum(exp(...)))`. For `τ = 0.2` and unit-norm inputs the raw form is fine; with very small `τ` you would want `logsumexp`.

### 6.5 What this gets us, intuitively

Training pushes the encoder's output space into a shape where **semantic similarity = geometric closeness**. After enough steps, "how to eat healthy" and "Essentials of Healthy Eating: A Guide" end up pointing in nearly the same direction, while "Effects of resveratrol on metabolic syndrome" ends up pointing somewhere else, even though both documents contain medical English.

That geometry is the entire product. The retrieval step (§10) is just a way to read it back out.

## 7. Parameter-efficient fine-tuning with LoRA

Full fine-tuning of a 150M-parameter encoder on a 3.6k-document corpus is a bad idea for two reasons:

1. **Overfitting.** Too many parameters, too little data — the model would memorize the qrels instead of learning a general retrieval geometry.
2. **Catastrophic forgetting.** The pretrained Ettin weights already encode an enormous amount of language understanding. Full fine-tuning happily steamrolls over it.

The fix is **LoRA** (Low-Rank Adaptation, Hu et al., 2021), via the `peft` library.

### 7.1 What LoRA actually does

Take a linear layer `y = W x`, where `W ∈ ℝ^{d_out × d_in}`. LoRA *freezes* `W` and adds a trainable low-rank update:

```
y = W x + (B A) x         with  A ∈ ℝ^{r × d_in},  B ∈ ℝ^{d_out × r}
```

`r` is the rank, typically 4 to 32. The number of trainable parameters in this layer drops from `d_out · d_in` to `r · (d_in + d_out)`, which for a 768-dim hidden state and `r = 16` is a ~24× reduction. Across the whole network, the trainable-parameter count falls by orders of magnitude.

In practice the LoRA update is scaled by `α / r`:

```
y = W x + (α / r) · (B A) x
```

`α` controls how loudly the adapter speaks; `α / r` keeps that scale roughly constant if you change `r`. `A` is typically initialized to Gaussian noise and `B` to zero, so the adapter starts as a no-op and the model behaves identically to the frozen base on step 0.

### 7.2 Why low rank works at all

This is the surprising part. Why should a rank-16 update be enough to specialize a 768-dim layer for a new task? The empirical observation behind LoRA is that fine-tuning updates `ΔW = W_finetuned − W_pretrained` tend to be *intrinsically low-rank* — they live in a small subspace of all possible weight changes. So you do not need a full-rank `ΔW`; you just need its low-rank approximation, and `BA` is exactly that.

For task adaptation (as opposed to teaching the model fundamentally new knowledge), this works remarkably well across modalities and architectures.

### 7.3 The configuration used here

From `src/train.py`:

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.065,
    bias="none",
    task_type=None,
    init_lora_weights=True,
)
model.enc = get_peft_model(model.enc, peft_config)
```

- `r = 16` — rank of the adapter matrices.
- `lora_alpha = 32` — so the effective LoRA scale is `α / r = 2`. This is a common ratio.
- `target_modules = "all-linear"` — add adapters to *every* linear projection in the encoder. The saved `adapter_config.json` records that PEFT resolved this to `["Wi", "Wo", "Wqkv"]`, which are ModernBERT's three linear blocks (FFN input, FFN output, and the fused query/key/value projection). That covers both attention and feed-forward paths.
- `lora_dropout = 0.065` — small dropout on the adapter inputs. Light regularization on a small corpus.
- `bias = "none"` — biases stay frozen. They are tiny; adapting them rarely helps.

The result is a roughly **13 MB** `adapter_model.safetensors` instead of a multi-hundred-MB full checkpoint. At inference time, PEFT loads the base encoder and overlays the adapter on top with `PeftModel.from_pretrained`.

### 7.4 Why this pairs well with a high learning rate

You will see `lr = 1e-3` in the training loop — much higher than the `2e-5` you would use for full fine-tuning. That is not a typo. Only the small adapter matrices are being updated; the frozen backbone is not at risk of being thrown off. The adapters *want* to move quickly because they start near zero, and a small learning rate would waste a lot of training on slow warmup.

## 8. The training loop in detail

`src/train.py` puts everything together. Going through it section by section:

### 8.1 The dataset

```python
class NFCorpusDataset(Dataset):
    def __init__(self, ..., num_negatives=8, split='train'):
        ...
```

- Reads `queries.jsonl` into `{query_id: "<QRY>...</QRY>"}`.
- Reads `corpus.jsonl` into `{doc_id: "<TLE>...</TLE> <TXT>...</TXT>"}`.
- Reads the appropriate qrels TSV and groups by `query-id` so each row is `(query_id, [list of relevant doc_ids])`.

`__getitem__` then samples one positive from the relevant list and 8 random negatives that are *not* in the relevant set. The "not in the relevant set" check matters: if you accidentally hand the loss a negative that is actually relevant, you train the model to push it away from a query it should be near, which is direct anti-signal.

### 8.2 The collate function

`collate_fn` flattens the batch and runs the tokenizer:

```python
positives = [text for item in batch for text in item['positive_texts']]
negatives = [text for item in batch for text in item['negative_texts']]

return {
    'query':     tokenize([item['query_text'] for item in batch]),
    'positives': tokenize(positives),
    'negatives': tokenize(negatives),
}
```

Three separate tokenized batches are returned. The model then encodes each, and the loss reshapes the flat negatives back into `(batch_size, num_negatives, hidden_dim)` so each query is paired with its own negatives.

`max_length=256` is the truncation cap. NFCorpus passages are short, and 256 BPE tokens covers most documents comfortably.

### 8.3 Optimizer, schedule, and length

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn   = MultiNCELoss()
epochs    = 200

total_steps  = len(train_dl) * epochs
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
```

- **AdamW.** Adam with decoupled weight decay. The standard choice for transformer fine-tuning.
- **Linear warmup + linear decay.** Learning rate ramps up from 0 to 1e-3 over the first 10% of training, then decays linearly back to 0. Warmup matters because at step 0 the adapter matrices are near zero and the gradients can be large; jumping straight to `1e-3` can destabilize training. Linear decay is the simplest schedule that works well with AdamW.
- **200 epochs.** This sounds like a lot, but each epoch is small (the NFCorpus train split is on the order of 100-ish queries), so the total step count is still manageable. The best-validation-loss checkpoint is saved.

### 8.4 The step

```python
que_vecs, pos_vecs, neg_vecs = model(queries, positives, negatives)
loss = loss_fn.calc_loss(que_vecs, pos_vecs, neg_vecs)

optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()
```

Standard PyTorch. Because LoRA is wired in, `loss.backward()` only computes gradients for the adapter parameters (and the six new token embeddings, which are also trainable). Everything else is frozen.

### 8.5 Validation and checkpointing

After each epoch the model is switched to `eval()` and runs over the `dev` qrels with the same loss. If the dev loss improves, only the LoRA adapter is saved:

```python
model.enc.save_pretrained(checkpoint_path)
```

`save_pretrained` on a PEFT-wrapped module writes *just* the adapter (a small `adapter_config.json` plus `adapter_model.safetensors`). The base Ettin weights stay on the Hub.

## 9. Encoding the corpus once

After training, every document gets encoded **once** and the vectors are cached to disk. That is the whole point of choosing a bi-encoder — you trade a one-time `O(N)` encoding pass for an `O(N · d)` matrix multiplication at query time, instead of an `O(N · forward_pass_cost)` at every query.

`src/encode_corpus.py` does this:

```python
model.enc = PeftModel.from_pretrained(model.enc, checkpoint_path)
model.eval()

doc_vecs, pdf_paths = [], []
with torch.inference_mode():
    for X in corpus_dl:
        vecs = model.enc(**X['docs']).last_hidden_state[..., 0, :]
        doc_vecs.append(F.normalize(vecs, p=2, dim=1))
        pdf_paths.extend(X['pdf_path'])
    doc_vecs = torch.cat(doc_vecs, dim=0).cpu()

torch.save({'doc_vecs': doc_vecs, 'pdf_paths': pdf_paths}, out_path)
```

Three things worth noticing:

1. **`torch.inference_mode()` instead of `torch.no_grad()`.** Same effect for our purposes (no autograd), but `inference_mode` is a bit more aggressive about avoiding version-counter bookkeeping, which makes the encoding pass slightly faster.
2. **L2 normalization happens here, once.** That way the cached `doc_vecs` are unit-norm and search can rely on `dot product = cosine similarity` without re-normalizing every query.
3. **PDF paths are kept in lockstep with the vectors.** `doc_vecs[i]` corresponds to `pdf_paths[i]`. A single torch save bundles both, so loading is one line.

For NFCorpus the corpus only encodes documents that have a matching PDF on disk (`if not os.path.exists(pdf_path): continue`). That keeps the index aligned with what the UI can actually serve.

## 10. Search: cosine similarity as a matmul

Search is intentionally boring (`src/search.py`):

```python
que = tokenizer([ta.add_query_tokens(query)], ..., return_tensors='pt')
que_vec = model.enc(**que).last_hidden_state[..., 0, :]
que_vec = F.normalize(que_vec, p=2, dim=1)

sims = (que_vec @ doc_vecs.T).squeeze(0)
scores, idxs = torch.sort(sims, descending=True)

hits = [(s, pdf_paths[i]) for s, i in zip(scores, idxs) if s >= threshold]
```

Walking through it:

1. **Wrap and tokenize the query.** Same `<QRY>...</QRY>` wrapping as during training. The encoder needs to see exactly the same format.
2. **Encode and normalize.** Same CLS pooling and L2 normalization as the corpus pass.
3. **One matmul.** `que_vec` is `(1, d)`, `doc_vecs.T` is `(d, N)`, so `sims` is `(1, N)` — one cosine similarity per document. With ~3.6k documents and a ~768-dim hidden state, this is well under a millisecond.
4. **Sort and threshold.** Descending sort, then keep only documents whose similarity is above the user's threshold.

Cosine similarity ranges in `[-1, 1]`, but in practice a well-trained retrieval encoder produces scores in roughly `[0.3, 0.95]` for natural-language queries — random pairs hover near a fairly high baseline because the encoder has a "general English" subspace they all live in. The threshold (default `0.75` in `config.py`, lowered to `0.5` in the UI for friendlier defaults) is the cutoff between "this looks plausibly relevant" and "this is just baseline similarity, ignore it."

### Why this scales fine

For 3.6k documents a dense matmul on CPU is trivial. For millions of documents you would replace the matmul with an **approximate nearest neighbor** index (FAISS, HNSW, ScaNN). The geometry the model learned does not change; only the lookup data structure does. Everything in `src/` would work unmodified against a FAISS index — just swap the matmul for a `.search()` call.

## 11. Repository layout

```
src/
  model.py           Ettin encoder + tokenizer with the six role tokens added.
  utils.py           AddTokens — the <QRY>/<TLE>/<TXT> wrapping helper.
  train.py           NFCorpusDataset, MultiNCELoss, the training loop.
  encode_corpus.py   One-shot pass: every doc → vector → corpus_encoded.pt.
  search.py          CLI search: encode query, cosine sim, top-K with clickable PDFs.
  indexer.py         Encode-and-append helpers used by the web UI for live adds.
  config.py          Paths, defaults (top-K, threshold), max lengths, host/port.

app/
  app.py             Flask wrapper around the encoder + cached index.
  templates/
    index.html       The MedSearch UI.

adapter_config.json            The trained LoRA config (rank, alpha, targets).
adapter_model.safetensors      The trained LoRA weights (~13 MB).
corpus_encoded.pt              {'doc_vecs': Tensor[N, d], 'pdf_paths': [str]} — the cached index.
data/nfcorpus/                 BEIR NFCorpus (corpus, queries, qrels, PDFs).
```

## 12. Running it end to end

The repo ships with a pre-trained adapter and a pre-encoded corpus, so you can skip straight to step 3 if you just want to try the search.

```bash
# 1. (Optional) Re-train the LoRA adapter from scratch.
#    Writes checkpoints/epoch_<N>_train_<L>_val_<L>/ each time dev loss improves.
python src/train.py

# 2. (Optional) Re-encode the corpus with the chosen adapter checkpoint.
python src/encode_corpus.py path/to/checkpoint corpus_encoded.pt

# 3. CLI search. Args: checkpoint_dir, encoded_corpus, threshold.
python src/search.py . corpus_encoded.pt 0.5

# 4. Web UI (see next section).
python app/app.py
```

The CLI search loop reads queries from stdin, prints ranked hits with their cosine similarity score, and renders each title as an OSC-8 hyperlink that opens the PDF in your default viewer when you Cmd/Ctrl-click it. Entering a number opens that hit directly.

## 13. Addition: the MedSearch web UI

The Flask app in `app/` is a wrapper around exactly the same encoder and the same `corpus_encoded.pt`. It does not change any of the ML; it just makes the system usable without a terminal and adds a path for growing the index after training.

It exposes three endpoints:

- `GET  /`        — renders the single-page UI.
- `POST /search`  — `{query, top_k, threshold}` → ranked results.
- `POST /add`     — multipart form (PDF + title + text) → appends to the live index.
- `GET  /pdf/...` — serves the PDF blobs for the result links (with a realpath check so you cannot escape the PDF directory).

Start it with:

```bash
python app/app.py    # http://127.0.0.1:5000
```

### 13.1 The landing page

![Main page](imgs/main_page.png)

Just a search bar and two knobs:

- **Top-K** — the maximum number of results to return.
- **Threshold** — the minimum cosine similarity required to count as a hit. Lower values cast a wider net; higher values demand a tighter semantic match.

The collapsed "Add a document to the index" panel at the bottom is the upload form, covered in §13.3.

### 13.2 Searching

![Search results](imgs/search_sample.png)

A query (here, *"How to eat healthy"*) is sent as JSON to `/search`. On the server (`app/app.py`):

```python
que_vec = encode_query(query)              # same wrapping + CLS + L2 norm as training
sims = (que_vec @ doc_vecs.T).squeeze(0)
scores, idxs = torch.sort(sims, descending=True)
```

Then results above the threshold are paginated to top-K, and each one is joined with its title and a 320-char snippet from `corpus.jsonl` before being shipped back as JSON. The UI renders each card with its raw cosine similarity score (the 0.879 / 0.866 in the screenshot) and a "View PDF" button that hits `/pdf/<filename>`. Latency for the whole round trip is on the order of tens of milliseconds — the "52 ms" badge in the screenshot is the server-reported timing.

Notice how both results are about healthy eating despite the query not lexically matching "Essentials of Healthy Eating" beyond the word "healthy" — that is the encoder doing its job. A BM25 baseline would also catch this particular query because the overlap is real, but a more paraphrastic query ("foods that help you live longer", "ways to lose weight without dieting") is where the semantic model pulls ahead.

### 13.3 Adding a document

![Add document](imgs/doc_add.png)

The upload form takes three things:

- **Title.** Used as the display name and wrapped in `<TLE>...</TLE>` before encoding.
- **Abstract / text.** This is what the encoder actually indexes. The PDF itself is *not* parsed; the user provides the representative text. This is intentional — PDF extraction is its own can of worms, and giving the user explicit control over what gets indexed produces dramatically better retrieval than blindly slurping page 1.
- **PDF file.** Saved to `data/nfcorpus/pdf_docs/` so it can be served back from search results.

On the server (`/add` in `app/app.py`):

1. Sanitize the filename, de-duplicate against the PDF directory, and save the file.
2. Encode `(title, text)` with the **same** pipeline used at corpus-build time — see `src/indexer.py`:
   ```python
   doc_text = build_doc_text(title, text)              # <TLE>...</TLE> <TXT>...</TXT>
   enc = tokenizer([doc_text], padding=True, truncation=True,
                    max_length=config.MAX_DOC_LENGTH, return_tensors='pt')
   vec = model.enc(**enc).last_hidden_state[..., 0, :]
   return F.normalize(vec, p=2, dim=1)
   ```
3. Append the new vector to the in-memory `doc_vecs` tensor and the new path to `pdf_paths`, under a lock so concurrent uploads do not race:
   ```python
   with _lock:
       doc_vecs = torch.cat([doc_vecs, new_vec], dim=0)
       pdf_paths.append(new_path)
       torch.save({'doc_vecs': doc_vecs.cpu(), 'pdf_paths': pdf_paths}, encoded_path)
   ```
4. Persist the updated tensor + path list back to `corpus_encoded.pt` so the new document survives a restart.

From the next search onward the new document is in the candidate pool. There is no re-training — the LoRA adapter stays put; we are just extending the matrix that the query gets multiplied against. This is one of the practical wins of the bi-encoder architecture: **adding a document is a single forward pass**, not a model update.

### 13.4 What the app intentionally does *not* do

A few things were left out on purpose to keep the focus on the retrieval model:

- **No PDF text extraction.** As above, the user supplies the indexing text. Cleaner than guessing.
- **No re-ranking.** The matmul ordering is shown as-is. A cross-encoder reranker on the top-K would be a clean drop-in.
- **No ANN index.** A linear matmul is the right tool at this scale. FAISS / HNSW would only be needed past ~10⁵ documents.
- **No auth, no users, no rate limiting.** It is meant to be run locally.

## 14. Things that could be improved

If this were headed for production, the obvious next steps would be:

1. **Hard-negative mining.** Replace random negatives with BM25-mined or model-mined hard negatives — documents that look plausible but are actually irrelevant. This is the single highest-leverage upgrade for retrieval quality.
2. **In-batch negatives.** With a larger batch size, every other query's positive becomes a free additional negative for the current query. The current batch size of 2 leaves that on the table.
3. **Cross-encoder reranker.** A small reranker over the top-50 hits would noticeably tighten precision, especially on ambiguous queries.
4. **Lower temperature once negatives are harder.** With hard negatives, `τ ≈ 0.05` becomes a more natural setting.
5. **Evaluate with proper IR metrics.** NDCG@10 and Recall@k on the `test` qrels rather than just dev loss.
6. **Approximate nearest neighbors.** Swap the matmul for FAISS once the corpus grows past a few hundred thousand documents.

None of these change the shape of the system — they change what numbers come out the other end. The skeleton (encoder → contrastive loss → cached vectors → matmul) is the same one most production retrieval stacks use.