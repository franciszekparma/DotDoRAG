import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from tqdm.auto import tqdm
import json
import math
import random
import numpy as np
import pandas as pd

from utils import AddTokens
from model import etin, etin_tokenizer, device


SEED = 42
QUERY_MAX_LEN = 64
DOC_MAX_LEN = 512
BATCH_SIZE = 128
NUM_HARD_NEG = 16
NUM_RAND_NEG = 16
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
EPOCHS = 60
PATIENCE = 8
GRAD_CLIP = 1.0
TEMP = 0.05
EVAL_K = 10


class NFCorpusDataset(Dataset):
  def __init__(self,
    qrels_path='data/nfcorpus/qrels/',
    queries_path='data/nfcorpus/queries.jsonl',
    corpus_path='data/nfcorpus/corpus.jsonl',
    num_hard_neg=NUM_HARD_NEG,
    num_rand_neg=NUM_RAND_NEG,
    split='train',
  ):
    super().__init__()
    if split.upper() == 'TRAIN':
      qrels_path += 'train.tsv'
    elif split.upper() == 'TEST':
      qrels_path += 'test.tsv'
    elif split.upper() == 'DEV':
      qrels_path += 'dev.tsv'

    self.ta = AddTokens()
    self.num_hard_neg = num_hard_neg
    self.num_rand_neg = num_rand_neg
    self.split = split.upper()

    self.queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
      for line in f:
        item = json.loads(line)
        self.queries[item['_id']] = self.ta.add_query_tokens(item['text'])

    self.corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
      for line in f:
        item = json.loads(line)
        doc = ""
        if 'title' in item: doc += self.ta.add_title_tokens(item['title']) + " "
        if 'text' in item: doc += self.ta.add_text_tokens(item['text'])
        self.corpus[item['_id']] = doc.strip()

    self.corpus_ids = list(self.corpus.keys())

    self.q_enc = {
      qid: etin_tokenizer(text, truncation=True, max_length=QUERY_MAX_LEN)
      for qid, text in self.queries.items()
    }
    self.c_enc = {
      cid: etin_tokenizer(text, truncation=True, max_length=DOC_MAX_LEN)
      for cid, text in self.corpus.items()
    }

    qrels = pd.read_csv(qrels_path, sep='\t')
    self.graded = {
      qid: dict(zip(g['corpus-id'], g['score']))
      for qid, g in qrels.groupby('query-id')
    }
    self.query_ids = [qid for qid in self.graded if qid in self.queries]
    self.pos_ids = {qid: list(self.graded[qid].keys()) for qid in self.query_ids}

    self.hard_negs = {qid: [] for qid in self.query_ids}
    if self.split == 'TRAIN' and num_hard_neg > 0:
      self._mine_hard_negatives()

  def _mine_hard_negatives(self):
    try:
      from rank_bm25 import BM25Okapi
    except ImportError as e:
      raise RuntimeError("Install rank_bm25 to mine hard negatives: pip install rank_bm25") from e

    corpus_tok = [self.corpus[cid].lower().split() for cid in self.corpus_ids]
    bm25 = BM25Okapi(corpus_tok)
    n = len(self.corpus_ids)
    pool_size = min(200, n)
    keep_per_query = 100
    for qid in tqdm(self.query_ids, desc='BM25 hard-neg mining'):
      pos = set(self.pos_ids[qid])
      scores = bm25.get_scores(self.queries[qid].lower().split())
      top = np.argpartition(-scores, kth=pool_size - 1)[:pool_size]
      top = top[np.argsort(-scores[top])]
      hn = []
      for i in top:
        cid = self.corpus_ids[i]
        if cid not in pos:
          hn.append(cid)
        if len(hn) >= keep_per_query:
          break
      self.hard_negs[qid] = hn

  def __len__(self):
    return len(self.query_ids)

  def __getitem__(self, idx):
    qid = self.query_ids[idx]
    pos = self.pos_ids[qid]
    pos_id = random.choice(pos)
    pos_set = set(pos)

    negs = []
    hn = self.hard_negs.get(qid, [])
    if self.num_hard_neg > 0 and hn:
      negs.extend(random.sample(hn, min(self.num_hard_neg, len(hn))))
    target = self.num_hard_neg + self.num_rand_neg
    chosen = set(negs)
    while len(negs) < target:
      sampled = random.choice(self.corpus_ids)
      if sampled not in pos_set and sampled not in chosen:
        negs.append(sampled)
        chosen.add(sampled)

    return {
      'q': self.q_enc[qid],
      'p': self.c_enc[pos_id],
      'n': [self.c_enc[cid] for cid in negs],
    }


def collate_fn(batch):
  def pad(encs):
    return etin_tokenizer.pad(encs, padding=True, return_tensors='pt')
  return {
    'q': pad([b['q'] for b in batch]),
    'p': pad([b['p'] for b in batch]),
    'n': pad([e for b in batch for e in b['n']]),
  }


class InBatchNCELoss(nn.Module):
  def __init__(self, temp=TEMP):
    super().__init__()
    self.temp = temp

  def forward(self, q, p, n):
    B = q.size(0)
    q = F.normalize(q.float(), dim=-1)
    p = F.normalize(p.float(), dim=-1)
    n = F.normalize(n.float(), dim=-1)
    docs = torch.cat([p, n], dim=0)
    logits = (q @ docs.T) / self.temp
    labels = torch.arange(B, device=q.device)
    return F.cross_entropy(logits, labels)


def unfreeze_qk(model):
  for p in model.parameters():
    p.requires_grad = False

  hidden = model.enc.config.hidden_size
  qk_params = []

  def mask_v(grad):
    grad = grad.clone()
    grad[2 * hidden:] = 0
    return grad

  for module in model.enc.modules():
    wqkv = getattr(module, 'Wqkv', None)
    if isinstance(wqkv, nn.Linear):
      wqkv.weight.requires_grad = True
      wqkv.weight.register_hook(mask_v)
      qk_params.append(wqkv.weight)
      if wqkv.bias is not None:
        wqkv.bias.requires_grad = True
        wqkv.bias.register_hook(mask_v)
        qk_params.append(wqkv.bias)

  return qk_params


@torch.inference_mode()
def encode_all(model, encs, batch_size=128, use_amp=True):
  out = []
  for i in range(0, len(encs), batch_size):
    chunk = encs[i:i + batch_size]
    b = etin_tokenizer.pad(chunk, padding=True, return_tensors='pt')
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
      v = model.enc(**b).last_hidden_state[..., 0, :]
    out.append(F.normalize(v.float(), dim=-1).cpu())
  return torch.cat(out, dim=0)


def ndcg_at_k(model, ds, k=EVAL_K, use_amp=True):
  model.eval()
  doc_emb = encode_all(model, [ds.c_enc[cid] for cid in ds.corpus_ids], use_amp=use_amp)
  q_emb = encode_all(model, [ds.q_enc[qid] for qid in ds.query_ids], use_amp=use_amp)
  sims = q_emb @ doc_emb.T
  topk = sims.topk(k, dim=-1).indices.numpy()

  ndcgs = []
  for qi, qid in enumerate(ds.query_ids):
    graded = ds.graded[qid]
    ranked = [graded.get(ds.corpus_ids[idx], 0) for idx in topk[qi]]
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ranked))
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    if idcg > 0:
      ndcgs.append(dcg / idcg)
  return sum(ndcgs) / max(len(ndcgs), 1)


def main():
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

  train_ds = NFCorpusDataset(split='train', num_hard_neg=NUM_HARD_NEG, num_rand_neg=NUM_RAND_NEG)
  val_ds = NFCorpusDataset(split='dev', num_hard_neg=0, num_rand_neg=NUM_HARD_NEG + NUM_RAND_NEG)

  dl_kw = dict(
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=(device == 'cuda'),
    persistent_workers=True,
  )
  train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **dl_kw)

  model = etin
  if hasattr(model.enc, 'gradient_checkpointing_enable'):
    model.enc.gradient_checkpointing_enable()
  qk_params = unfreeze_qk(model)

  optimizer = torch.optim.AdamW(qk_params, lr=LR, weight_decay=WEIGHT_DECAY)
  loss_fn = InBatchNCELoss(temp=TEMP)

  total_steps = len(train_dl) * EPOCHS
  warmup_steps = int(WARMUP_RATIO * total_steps)
  scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

  use_amp = (device == 'cuda')

  best_ndcg = -1.0
  epochs_since_improve = 0

  for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for X in tqdm(train_dl, desc=f'Epoch {epoch + 1}/{EPOCHS} train', unit='batch'):
      q = {k: v.to(device, non_blocking=True) for k, v in X['q'].items()}
      p = {k: v.to(device, non_blocking=True) for k, v in X['p'].items()}
      n = {k: v.to(device, non_blocking=True) for k, v in X['n'].items()}

      with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
        qv, pv, nv = model(q, p, n)
      loss = loss_fn(qv, pv, nv)

      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(qk_params, GRAD_CLIP)
      optimizer.step()
      scheduler.step()
      train_losses.append(loss.item())

    dev_ndcg = ndcg_at_k(model, val_ds, k=EVAL_K, use_amp=use_amp)

    train_loss = sum(train_losses) / len(train_losses)
    saved = ''
    if dev_ndcg > best_ndcg:
      best_ndcg = dev_ndcg
      checkpoint_path = f'checkpoints/epoch_{epoch + 1}_train_{train_loss:.4f}_ndcg_{dev_ndcg:.4f}'
      model.enc.save_pretrained(checkpoint_path)
      saved = f' !!!Saved the best model at: {checkpoint_path}!!!'
      epochs_since_improve = 0
    else:
      epochs_since_improve += 1

    tqdm.write(f'Epoch {epoch + 1}/{EPOCHS}: train={train_loss:.4f} dev_ndcg@{EVAL_K}={dev_ndcg:.4f} (best={best_ndcg:.4f}){saved}')

    if epochs_since_improve >= PATIENCE:
      tqdm.write(f'Early stopping at epoch {epoch + 1} (no NDCG improvement for {PATIENCE} epochs).')
      break


if __name__ == '__main__':
  main()