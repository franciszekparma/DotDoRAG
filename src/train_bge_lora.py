import os
import math
import random
import contextlib

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import (
  get_cosine_schedule_with_warmup,
  AutoTokenizer,
  AutoModelForSequenceClassification,
)

from tqdm.auto import tqdm

from data import NFCorpusDataset
from model import bge, bge_tokenizer, device


SEED = 42
QUERY_MAX_LEN = 96
DOC_MAX_LEN = 512
BATCH_SIZE = 64
GRAD_ACCUM = 4
NUM_HARD_NEG = 7
NUM_RAND_NEG = 1
LR = 7e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
EPOCHS = 30
PATIENCE = 8
GRAD_CLIP = 1.0
TEMP = 0.02
TEACHER_TEMP = 1.0
KL_WEIGHT = 2.0
REMINE_EVERY = 1
EVAL_K = 10
NUM_WORKERS = 4
CHECKPOINT_DIR = '../bge_lora_checkpoints'
TEACHER_NAME = 'BAAI/bge-reranker-base'
TEACHER_MAX_LEN = 512
TEACHER_CHUNK = 32
RERANK_POOL = 100
BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

_SPECIAL_TOKS = ('<QRY>', '</QRY>', '<TLE>', '</TLE>', '<TXT>', '</TXT>')


def _strip_special(text):
  for tok in _SPECIAL_TOKS:
    text = text.replace(tok, ' ')
  return ' '.join(text.split()).strip()


def normalize_dataset(ds):
  ds.queries = {qid: _strip_special(t) for qid, t in ds.queries.items()}
  ds.corpus = {cid: _strip_special(t) for cid, t in ds.corpus.items()}


class MultiPosDataset(NFCorpusDataset):
  def __getitem__(self, idx):
    qid = self.query_ids[idx]
    pos_set = self.pos_sets[qid]
    relevant = [cid for cid, s in self.graded[qid].items() if s > 0]
    if not relevant:
      relevant = list(self.graded[qid].keys())
    pos_id = self._rng.choice(relevant)

    negs = []
    chosen = set()
    hn = self.hard_negs.get(qid, [])
    if self.num_hard_neg > 0 and hn:
      sampled_hn = self._rng.sample(hn, min(self.num_hard_neg, len(hn)))
      negs.extend(sampled_hn)
      chosen.update(sampled_hn)

    target = self.num_hard_neg + self.num_rand_neg
    while len(negs) < target:
      sampled = self._rng.choice(self.corpus_ids)
      if sampled not in pos_set and sampled not in chosen:
        negs.append(sampled)
        chosen.add(sampled)

    return {
      'query_id': qid,
      'pos_id': pos_id,
      'neg_ids': negs,
      'relevant': set(relevant),
      'query_text': self.queries[qid],
      'positive_texts': [self.corpus[pos_id]],
      'negative_texts': [self.corpus[cid] for cid in negs],
    }


def collate_fn(batch):
  raw_queries = [item['query_text'] for item in batch]
  positives = [item['positive_texts'][0] for item in batch]
  negatives = [text for item in batch for text in item['negative_texts']]
  student_queries = [BGE_QUERY_PREFIX + q for q in raw_queries]

  return {
    'query_ids': [item['query_id'] for item in batch],
    'pos_ids': [item['pos_id'] for item in batch],
    'neg_ids': [item['neg_ids'] for item in batch],
    'relevant': [item['relevant'] for item in batch],
    'raw_query_texts': raw_queries,
    'pos_texts': positives,
    'neg_texts': negatives,
    'q': bge_tokenizer(student_queries, padding=True, truncation=True,
                       max_length=QUERY_MAX_LEN, return_tensors='pt'),
    'p': bge_tokenizer(positives, padding=True, truncation=True,
                       max_length=DOC_MAX_LEN, return_tensors='pt'),
    'n': bge_tokenizer(negatives, padding=True, truncation=True,
                       max_length=DOC_MAX_LEN, return_tensors='pt'),
  }


def build_pos_mask_full(pos_ids, neg_ids_list, relevant_list, num_neg):
  B = len(pos_ids)
  D = B + B * num_neg
  mask = torch.zeros(B, D, dtype=torch.bool)
  for i, rel_i in enumerate(relevant_list):
    for j, pid in enumerate(pos_ids):
      if pid in rel_i:
        mask[i, j] = True
    for j, negs_j in enumerate(neg_ids_list):
      base = B + j * num_neg
      for k, nid in enumerate(negs_j):
        if nid in rel_i:
          mask[i, base + k] = True
  return mask


def multi_pos_infonce(qv, pv, nv, pos_mask, temp):
  B = qv.size(0)
  qv = F.normalize(qv.float(), dim=-1)
  pv = F.normalize(pv.float(), dim=-1)
  nv = F.normalize(nv.float(), dim=-1)
  docs = torch.cat([pv, nv], dim=0)
  logits = (qv @ docs.T) / temp
  diag = torch.zeros_like(pos_mask)
  diag[:, :B] = torch.eye(B, dtype=torch.bool, device=pos_mask.device)
  keep_mask = pos_mask & ~diag
  logits = logits.masked_fill(keep_mask, float('-inf'))
  labels = torch.arange(B, device=qv.device)
  return F.cross_entropy(logits, labels)


def distill_kl(qv, pv, nv, teacher_logits, teacher_temp, student_temp, num_neg):
  B = qv.size(0)
  qv = F.normalize(qv.float(), dim=-1)
  pv = F.normalize(pv.float(), dim=-1)
  nv = F.normalize(nv.float(), dim=-1)
  own_pos = (qv * pv).sum(-1, keepdim=True)
  nv_grouped = nv.view(B, num_neg, -1)
  own_neg = torch.einsum('bh,bnh->bn', qv, nv_grouped)
  student_logits = torch.cat([own_pos, own_neg], dim=-1) / student_temp
  with torch.no_grad():
    teacher_dist = F.softmax(teacher_logits / teacher_temp, dim=-1)
  student_log = F.log_softmax(student_logits, dim=-1)
  return -(teacher_dist * student_log).sum(-1).mean()


@torch.inference_mode()
def teacher_score(teacher, tok, query_texts, pos_texts, neg_texts_flat,
                  B, num_neg, use_amp, chunk=TEACHER_CHUNK):
  pairs_q, pairs_d = [], []
  for i in range(B):
    pairs_q.append(query_texts[i])
    pairs_d.append(pos_texts[i])
    for k in range(num_neg):
      pairs_q.append(query_texts[i])
      pairs_d.append(neg_texts_flat[i * num_neg + k])
  amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
             if use_amp else contextlib.nullcontext())
  chunks = []
  for start in range(0, len(pairs_q), chunk):
    end = min(start + chunk, len(pairs_q))
    enc = tok(pairs_q[start:end], pairs_d[start:end],
              padding=True, truncation=True,
              max_length=TEACHER_MAX_LEN, return_tensors='pt')
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    with amp_ctx:
      out = teacher(**enc).logits.squeeze(-1)
    chunks.append(out.float())
  return torch.cat(chunks, dim=0).view(B, 1 + num_neg)


@torch.inference_mode()
def encode_texts(model, texts, batch_size=256, max_len=DOC_MAX_LEN, use_amp=True):
  vecs = []
  for i in range(0, len(texts), batch_size):
    chunk = texts[i:i + batch_size]
    b = bge_tokenizer(chunk, padding=True, truncation=True,
                      max_length=max_len, return_tensors='pt')
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
               if use_amp else contextlib.nullcontext())
    with amp_ctx:
      v = model.enc(**b).last_hidden_state[..., 0, :]
    vecs.append(F.normalize(v.float(), dim=-1).cpu())
  return torch.cat(vecs, dim=0)


def retrieval_metrics(model, ds, k=EVAL_K, use_amp=True):
  was_training = model.training
  model.eval()
  doc_emb = encode_texts(model, [ds.corpus[cid] for cid in ds.corpus_ids],
                         max_len=DOC_MAX_LEN, use_amp=use_amp)
  q_texts = [BGE_QUERY_PREFIX + ds.queries[qid] for qid in ds.query_ids]
  q_emb = encode_texts(model, q_texts, max_len=QUERY_MAX_LEN, use_amp=use_amp)
  sims = q_emb @ doc_emb.T
  topk = sims.topk(k, dim=-1).indices.numpy()

  ndcgs, recalls = [], []
  for qi, qid in enumerate(ds.query_ids):
    graded = ds.graded[qid]
    ranked = [graded.get(ds.corpus_ids[idx], 0) for idx in topk[qi]]
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ranked))
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    if idcg > 0:
      ndcgs.append(dcg / idcg)
    n_pos = sum(1 for v in graded.values() if v > 0)
    hits = sum(1 for r in ranked if r > 0)
    if n_pos > 0:
      recalls.append(hits / n_pos)

  if was_training:
    model.train()
  return {
    f'ndcg@{k}': sum(ndcgs) / max(len(ndcgs), 1),
    f'recall@{k}': sum(recalls) / max(len(recalls), 1),
  }


@torch.inference_mode()
def rerank_metrics(model, teacher, teacher_tok, ds, k=EVAL_K,
                   pool=RERANK_POOL, use_amp=True):
  was_training = model.training
  model.eval()
  doc_emb = encode_texts(model, [ds.corpus[cid] for cid in ds.corpus_ids],
                         max_len=DOC_MAX_LEN, use_amp=use_amp)
  q_texts = [BGE_QUERY_PREFIX + ds.queries[qid] for qid in ds.query_ids]
  q_emb = encode_texts(model, q_texts, max_len=QUERY_MAX_LEN, use_amp=use_amp)
  sims = q_emb @ doc_emb.T
  pool = min(pool, sims.size(1))
  pool_idx_all = sims.topk(pool, dim=-1).indices.numpy()

  amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
             if use_amp else contextlib.nullcontext())

  ndcgs, recalls = [], []
  for qi, qid in enumerate(tqdm(ds.query_ids, desc='rerank', leave=False)):
    pool_idx = pool_idx_all[qi].tolist()
    docs = [ds.corpus[ds.corpus_ids[idx]] for idx in pool_idx]
    raw_q = ds.queries[qid]
    enc = teacher_tok([raw_q] * len(docs), docs,
                      padding=True, truncation=True,
                      max_length=TEACHER_MAX_LEN, return_tensors='pt')
    enc = {kk: vv.to(device, non_blocking=True) for kk, vv in enc.items()}
    with amp_ctx:
      scores = teacher(**enc).logits.squeeze(-1).float()
    top_in_pool = scores.topk(min(k, len(docs))).indices.cpu().tolist()
    reranked = [pool_idx[i] for i in top_in_pool]

    graded = ds.graded[qid]
    ranked = [graded.get(ds.corpus_ids[idx], 0) for idx in reranked]
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ranked))
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    if idcg > 0:
      ndcgs.append(dcg / idcg)
    n_pos = sum(1 for v in graded.values() if v > 0)
    hits = sum(1 for r in ranked if r > 0)
    if n_pos > 0:
      recalls.append(hits / n_pos)

  if was_training:
    model.train()
  return {
    f'ndcg@{k}': sum(ndcgs) / max(len(ndcgs), 1),
    f'recall@{k}': sum(recalls) / max(len(recalls), 1),
  }


def remine_hard_negs(model, ds, k=100, pool_size=200, use_amp=True):
  was_training = model.training
  model.eval()
  doc_emb = encode_texts(model, [ds.corpus[cid] for cid in ds.corpus_ids],
                         max_len=DOC_MAX_LEN, use_amp=use_amp)
  q_texts = [BGE_QUERY_PREFIX + ds.queries[qid] for qid in ds.query_ids]
  q_emb = encode_texts(model, q_texts, max_len=QUERY_MAX_LEN, use_amp=use_amp)
  sims = q_emb @ doc_emb.T
  pool = min(pool_size, sims.size(1))
  top = sims.topk(pool, dim=-1).indices.numpy()
  for qi, qid in enumerate(ds.query_ids):
    pos = ds.pos_sets[qid]
    hn = []
    for idx in top[qi]:
      cid = ds.corpus_ids[idx]
      if cid not in pos:
        hn.append(cid)
      if len(hn) >= k:
        break
    ds.hard_negs[qid] = hn
  if was_training:
    model.train()


def main():
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

  train_ds = MultiPosDataset(
    split='TRAIN',
    num_hard_neg=NUM_HARD_NEG,
    num_rand_neg=NUM_RAND_NEG,
    mine_hard_negs=(NUM_HARD_NEG > 0),
    seed=SEED,
  )
  val_ds = MultiPosDataset(
    split='VAL',
    num_hard_neg=0,
    num_rand_neg=NUM_HARD_NEG + NUM_RAND_NEG,
    seed=SEED,
  )
  for ds in (train_ds, val_ds):
    normalize_dataset(ds)

  sample_qid = train_ds.query_ids[0]
  tqdm.write(f'[sanity] query[0]: {(BGE_QUERY_PREFIX + train_ds.queries[sample_qid])[:120]!r}')

  dl_kw = dict(
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=(device == 'cuda'),
    persistent_workers=(NUM_WORKERS > 0),
  )
  train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw)

  peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["query", "key", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=None,
  )
  model = bge
  model.enc = get_peft_model(model.enc, peft_config)
  model.enc.print_trainable_parameters()

  use_amp = (device == 'cuda')
  tqdm.write(f'[load] teacher: {TEACHER_NAME}')
  teacher_tok = AutoTokenizer.from_pretrained(TEACHER_NAME)
  teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME)
  teacher = teacher.to(device).eval().requires_grad_(False)
  if use_amp:
    teacher = teacher.to(dtype=torch.bfloat16)

  trainable = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.AdamW(
    trainable, lr=LR, weight_decay=WEIGHT_DECAY,
    fused=(device == 'cuda'),
  )

  opt_steps_per_epoch = max(1, len(train_dl) // GRAD_ACCUM)
  total_steps = opt_steps_per_epoch * EPOCHS
  warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
  scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

  os.makedirs(CHECKPOINT_DIR, exist_ok=True)

  best_ndcg = -1.0
  epochs_since_best = 0

  for epoch in range(EPOCHS):
    if epoch > 0 and epoch % REMINE_EVERY == 0:
      tqdm.write(f'[remine] epoch {epoch + 1}: refreshing hard negatives with current model')
      remine_hard_negs(model, train_ds, k=100, pool_size=200, use_amp=use_amp)

    model.train()
    train_losses, kl_losses, nce_losses = [], [], []
    optimizer.zero_grad(set_to_none=True)
    accum_count = 0

    pbar = tqdm(train_dl, desc=f'Epoch {epoch + 1}/{EPOCHS} train', unit='batch')
    for step, X in enumerate(pbar):
      q = {k: v.to(device, non_blocking=True) for k, v in X['q'].items()}
      p = {k: v.to(device, non_blocking=True) for k, v in X['p'].items()}
      n = {k: v.to(device, non_blocking=True) for k, v in X['n'].items()}
      pos_mask = build_pos_mask_full(
        X['pos_ids'], X['neg_ids'], X['relevant'],
        num_neg=NUM_HARD_NEG + NUM_RAND_NEG,
      ).to(device, non_blocking=True)

      B = len(X['pos_ids'])
      N = NUM_HARD_NEG + NUM_RAND_NEG
      teacher_logits = teacher_score(
        teacher, teacher_tok,
        X['raw_query_texts'], X['pos_texts'], X['neg_texts'],
        B=B, num_neg=N, use_amp=use_amp,
      ).to(device)

      amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                 if use_amp else contextlib.nullcontext())
      with amp_ctx:
        qv, pv, nv = model(q, p, n)
      nce = multi_pos_infonce(qv, pv, nv, pos_mask, TEMP)
      kl = distill_kl(qv, pv, nv, teacher_logits, TEACHER_TEMP, TEMP, num_neg=N)
      loss = nce + KL_WEIGHT * kl

      (loss / GRAD_ACCUM).backward()
      accum_count += 1
      train_losses.append(loss.item())
      kl_losses.append(kl.item())
      nce_losses.append(nce.item())

      if accum_count >= GRAD_ACCUM or step == len(train_dl) - 1:
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0
      pbar.set_postfix(loss=f'{loss.item():.3f}',
                       nce=f'{nce.item():.3f}',
                       kl=f'{kl.item():.3f}')

    train_loss = sum(train_losses) / len(train_losses)
    train_nce = sum(nce_losses) / len(nce_losses)
    train_kl = sum(kl_losses) / len(kl_losses)

    val_metrics = retrieval_metrics(model, val_ds, k=EVAL_K, use_amp=use_amp)
    val_ndcg = val_metrics[f'ndcg@{EVAL_K}']
    val_recall = val_metrics[f'recall@{EVAL_K}']

    saved = ''
    if val_ndcg > best_ndcg:
      best_ndcg = val_ndcg
      epochs_since_best = 0
      ckpt = os.path.join(
        CHECKPOINT_DIR,
        f'epoch_{epoch + 1}_train_{train_loss:.4f}_ndcg_{val_ndcg:.4f}'
      )
      model.enc.save_pretrained(ckpt)
      saved = f' !!!Saved best at: {ckpt}!!!'
    else:
      epochs_since_best += 1

    tqdm.write(
      f'Epoch {epoch + 1}/{EPOCHS}: train_loss={train_loss:.4f} '
      f'(nce={train_nce:.4f} kl={train_kl:.4f}) '
      f'val_ndcg@{EVAL_K}={val_ndcg:.4f} val_recall@{EVAL_K}={val_recall:.4f} '
      f'(best_ndcg={best_ndcg:.4f}){saved}'
    )

    if epochs_since_best >= PATIENCE:
      tqdm.write(f'Early stopping: no nDCG improvement for {PATIENCE} epochs.')
      break

  test_ds = MultiPosDataset(
    split='TEST',
    num_hard_neg=0,
    num_rand_neg=0,
    seed=SEED,
  )
  normalize_dataset(test_ds)
  test_metrics = retrieval_metrics(model, test_ds, k=EVAL_K, use_amp=use_amp)
  tqdm.write(
    f'\nFinal held-out test (dense): ndcg@{EVAL_K}={test_metrics[f"ndcg@{EVAL_K}"]:.4f} '
    f'recall@{EVAL_K}={test_metrics[f"recall@{EVAL_K}"]:.4f}'
  )
  rerank_test = rerank_metrics(model, teacher, teacher_tok, test_ds,
                               k=EVAL_K, pool=RERANK_POOL, use_amp=use_amp)
  tqdm.write(
    f'Final held-out test (dense+rerank@{RERANK_POOL}): '
    f'ndcg@{EVAL_K}={rerank_test[f"ndcg@{EVAL_K}"]:.4f} '
    f'recall@{EVAL_K}={rerank_test[f"recall@{EVAL_K}"]:.4f}'
  )
  rerank_val = rerank_metrics(model, teacher, teacher_tok, val_ds,
                              k=EVAL_K, pool=RERANK_POOL, use_amp=use_amp)
  tqdm.write(
    f'Final val (dense+rerank@{RERANK_POOL}): '
    f'ndcg@{EVAL_K}={rerank_val[f"ndcg@{EVAL_K}"]:.4f} '
    f'recall@{EVAL_K}={rerank_val[f"recall@{EVAL_K}"]:.4f}'
  )


if __name__ == '__main__':
  main()