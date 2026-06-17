import os
import math
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from tqdm.auto import tqdm

from data import NFCorpusDataset
from model import bge, bge_tokenizer, device


SEED = 42
QUERY_MAX_LEN = 64
DOC_MAX_LEN = 256
BATCH_SIZE = 128
NUM_HARD_NEG = 4
NUM_RAND_NEG = 4
LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
EPOCHS = 25
PATIENCE = 4
GRAD_CLIP = 1.0
TEMP = 0.02
EVAL_K = 10
NUM_WORKERS = 4
CHECKPOINT_DIR = '../bge_lora_checkpoints'


def collate_fn(batch):
  queries = [item['query_text'] for item in batch]
  positives = [item['positive_texts'][0] for item in batch]
  negatives = [text for item in batch for text in item['negative_texts']]

  return {
    'query_ids': [item['query_id'] for item in batch],
    'pos_ids': [item['pos_id'] for item in batch],
    'graded_positives': [item['graded_positives'] for item in batch],
    'q': bge_tokenizer(queries, padding=True, truncation=True,
                       max_length=QUERY_MAX_LEN, return_tensors='pt'),
    'p': bge_tokenizer(positives, padding=True, truncation=True,
                       max_length=DOC_MAX_LEN, return_tensors='pt'),
    'n': bge_tokenizer(negatives, padding=True, truncation=True,
                       max_length=DOC_MAX_LEN, return_tensors='pt'),
  }


def build_pos_mask(pos_ids, graded_positives_list):
  B = len(pos_ids)
  mask = torch.zeros(B, B, dtype=torch.bool)
  for i, gp_i in enumerate(graded_positives_list):
    for j, pid in enumerate(pos_ids):
      if pid in gp_i:
        mask[i, j] = True
  return mask


def in_batch_nce_loss(qv, pv, nv, pos_masks, temp):
  B = qv.size(0)
  qv = F.normalize(qv.float(), dim=-1)
  pv = F.normalize(pv.float(), dim=-1)
  nv = F.normalize(nv.float(), dim=-1)
  docs = torch.cat([pv, nv], dim=0)
  logits = (qv @ docs.T) / temp
  diag = torch.eye(B, dtype=torch.bool, device=qv.device)
  off_diag_pos = pos_masks & ~diag
  logits[:, :B] = logits[:, :B].masked_fill(off_diag_pos, float('-inf'))
  labels = torch.arange(B, device=qv.device)
  return F.cross_entropy(logits, labels)


@torch.inference_mode()
def encode_texts(model, texts, batch_size=256, max_len=DOC_MAX_LEN, use_amp=True):
  vecs = []
  for i in range(0, len(texts), batch_size):
    chunk = texts[i:i + batch_size]
    b = bge_tokenizer(chunk, padding=True, truncation=True,
                      max_length=max_len, return_tensors='pt')
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
      v = model.enc(**b).last_hidden_state[..., 0, :]
    vecs.append(F.normalize(v.float(), dim=-1).cpu())
  return torch.cat(vecs, dim=0)


def retrieval_metrics(model, ds, k=EVAL_K, use_amp=True):
  was_training = model.training
  model.eval()
  doc_emb = encode_texts(model, [ds.corpus[cid] for cid in ds.corpus_ids],
                         max_len=DOC_MAX_LEN, use_amp=use_amp)
  q_emb = encode_texts(model, [ds.queries[qid] for qid in ds.query_ids],
                       max_len=QUERY_MAX_LEN, use_amp=use_amp)
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


def main():
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

  train_ds = NFCorpusDataset(
    split='TRAIN',
    num_hard_neg=NUM_HARD_NEG,
    num_rand_neg=NUM_RAND_NEG,
    mine_hard_negs=(NUM_HARD_NEG > 0),
    seed=SEED,
  )
  val_ds = NFCorpusDataset(
    split='VAL',
    num_hard_neg=0,
    num_rand_neg=NUM_HARD_NEG + NUM_RAND_NEG,
    seed=SEED,
  )

  dl_kw = dict(
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=(device == 'cuda'),
    persistent_workers=(NUM_WORKERS > 0),
  )
  train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw)

  peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type=None,
    init_lora_weights=True,
    modules_to_save=["word_embeddings"],
  )
  model = bge
  model.enc = get_peft_model(model.enc, peft_config)
  model.enc.print_trainable_parameters()

  trainable = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.AdamW(
    trainable, lr=LR, weight_decay=WEIGHT_DECAY,
    fused=(device == 'cuda'),
  )

  total_steps = len(train_dl) * EPOCHS
  warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
  scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

  use_amp = (device == 'cuda')
  os.makedirs(CHECKPOINT_DIR, exist_ok=True)

  best_ndcg = -1.0
  best_state = None
  epochs_since_best = 0

  for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for X in tqdm(train_dl, desc=f'Epoch {epoch + 1}/{EPOCHS} train', unit='batch'):
      q = {k: v.to(device, non_blocking=True) for k, v in X['q'].items()}
      p = {k: v.to(device, non_blocking=True) for k, v in X['p'].items()}
      n = {k: v.to(device, non_blocking=True) for k, v in X['n'].items()}
      pos_masks = build_pos_mask(X['pos_ids'], X['graded_positives']).to(device, non_blocking=True)

      optimizer.zero_grad(set_to_none=True)
      with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
        qv, pv, nv = model(q, p, n)
      loss = in_batch_nce_loss(qv, pv, nv, pos_masks, TEMP)

      loss.backward()
      torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
      optimizer.step()
      scheduler.step()
      train_losses.append(loss.item())

    train_loss = sum(train_losses) / len(train_losses)
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
      best_state = {n: p.detach().cpu().clone()
                    for n, p in model.named_parameters() if p.requires_grad}
      saved = f' !!!Saved best at: {ckpt}!!!'
    else:
      epochs_since_best += 1

    tqdm.write(
      f'Epoch {epoch + 1}/{EPOCHS}: train_loss={train_loss:.4f} '
      f'val_ndcg@{EVAL_K}={val_ndcg:.4f} val_recall@{EVAL_K}={val_recall:.4f} '
      f'(best_ndcg={best_ndcg:.4f}){saved}'
    )

    if epochs_since_best >= PATIENCE:
      tqdm.write(f'Early stopping: no nDCG improvement for {PATIENCE} epochs.')
      break

  if best_state is not None:
    with torch.no_grad():
      own = dict(model.named_parameters())
      for n, p in best_state.items():
        own[n].data.copy_(p.to(own[n].device))

  test_ds = NFCorpusDataset(
    split='TEST',
    num_hard_neg=0,
    num_rand_neg=0,
    seed=SEED,
  )
  test_metrics = retrieval_metrics(model, test_ds, k=EVAL_K, use_amp=use_amp)
  tqdm.write(
    f'\nFinal held-out test: ndcg@{EVAL_K}={test_metrics[f"ndcg@{EVAL_K}"]:.4f} '
    f'recall@{EVAL_K}={test_metrics[f"recall@{EVAL_K}"]:.4f}'
  )


if __name__ == '__main__':
  main()