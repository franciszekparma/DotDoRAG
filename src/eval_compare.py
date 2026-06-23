import os
import math
import contextlib
import gc

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from rank_bm25 import BM25Okapi

from data import NFCorpusDataset
from utils import (
  AddTokens,
  BGE_NAME,
  ETIN_NAME,
  BGE_QUERY_PREFIX,
  SPECIAL_TOKS,
  MAX_QUERY_LENGTH as QUERY_MAX_LEN,
  MAX_DOC_LENGTH as DOC_MAX_LEN,
  EVAL_K as K,
  EVAL_BATCH_SIZE,
  EVAL_ETIN_MAX_LEN,
)

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')
use_amp = (device == 'cuda')


def strip_special(text):
  for tok in SPECIAL_TOKS:
    text = text.replace(tok, ' ')
  return ' '.join(text.split()).strip()


@torch.inference_mode()
def encode(enc, tok, texts, batch_size, max_len):
  vecs = []
  for i in range(0, len(texts), batch_size):
    chunk = texts[i:i + batch_size]
    b = tok(chunk, padding=True, truncation=True,
            max_length=max_len, return_tensors='pt')
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
               if use_amp else contextlib.nullcontext())
    with amp_ctx:
      v = enc(**b).last_hidden_state[..., 0, :]
    vecs.append(F.normalize(v.float(), dim=-1).cpu())
  return torch.cat(vecs, dim=0)


def score_from_topk(topk_idx, corpus_ids, graded_per_q, query_ids, k):
  ndcgs, recalls = [], []
  for qi, qid in enumerate(query_ids):
    graded = graded_per_q[qid]
    ranked = [graded.get(corpus_ids[i], 0) for i in topk_idx[qi]]
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ranked))
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    if idcg > 0:
      ndcgs.append(dcg / idcg)
    n_pos = sum(1 for v in graded.values() if v > 0)
    hits = sum(1 for r in ranked if r > 0)
    if n_pos > 0:
      recalls.append(hits / n_pos)
  return {
    f'ndcg@{k}': sum(ndcgs) / max(len(ndcgs), 1),
    f'recall@{k}': sum(recalls) / max(len(recalls), 1),
  }


def eval_roc_auc(scores, corpus_ids, graded, query_ids):
  aucs = []
  for qi, qid in enumerate(query_ids):
    y = np.array([1 if graded[qid].get(corpus_ids[j], 0) > 0 else 0
                  for j in range(len(corpus_ids))])
    if len(np.unique(y)) < 2:
      continue
    aucs.append(roc_auc_score(y, scores[qi]))
  return {'roc_auc': sum(aucs) / max(len(aucs), 1)}


def eval_dense(enc, tok, queries_list, corpus_list, corpus_ids, graded, query_ids,
               q_prefix='', qmax=QUERY_MAX_LEN, dmax=DOC_MAX_LEN, k=K):
  doc_emb = encode(enc, tok, corpus_list, EVAL_BATCH_SIZE, dmax)
  q_emb = encode(enc, tok, [q_prefix + q for q in queries_list], EVAL_BATCH_SIZE, qmax)
  sims = q_emb @ doc_emb.T
  topk = sims.topk(k, dim=-1).indices.numpy()
  m = score_from_topk(topk, corpus_ids, graded, query_ids, k)
  m.update(eval_roc_auc(sims.numpy(), corpus_ids, graded, query_ids))
  return m


def eval_bm25(queries_list, corpus_list, corpus_ids, graded, query_ids, k=K):
  tokenized_corpus = [doc.lower().split() for doc in corpus_list]
  bm25 = BM25Okapi(tokenized_corpus)
  topk = np.zeros((len(queries_list), k), dtype=np.int64)
  all_scores = np.zeros((len(queries_list), len(corpus_list)))
  for qi, q in enumerate(queries_list):
    scores = bm25.get_scores(q.lower().split())
    all_scores[qi] = scores
    topk[qi] = np.argpartition(-scores, kth=k - 1)[:k]
    topk[qi] = topk[qi][np.argsort(-scores[topk[qi]])]
  m = score_from_topk(topk, corpus_ids, graded, query_ids, k)
  m.update(eval_roc_auc(all_scores, corpus_ids, graded, query_ids))
  return m


def load_bge_with_adapter(adapter_dir, bge_tok):
  enc = AutoModel.from_pretrained(BGE_NAME)
  enc.resize_token_embeddings(len(bge_tok))
  enc = enc.to(device).eval()
  enc = PeftModel.from_pretrained(enc, adapter_dir).to(device).eval()
  return enc


def load_etin_with_adapter(adapter_dir, etin_tok):
  enc = AutoModel.from_pretrained(ETIN_NAME)
  enc.resize_token_embeddings(len(etin_tok))
  enc = enc.to(device).eval()
  enc = PeftModel.from_pretrained(enc, adapter_dir).to(device).eval()
  return enc


def main():
  ds = NFCorpusDataset(split='TEST', num_hard_neg=0, num_rand_neg=0, seed=42)
  print(f'[data] test queries={len(ds.query_ids)} corpus={len(ds.corpus_ids)}')

  query_ids = ds.query_ids
  corpus_ids = ds.corpus_ids
  graded = ds.graded

  raw_queries = [strip_special(ds.queries[q]) for q in query_ids]
  raw_corpus = [strip_special(ds.corpus[c]) for c in corpus_ids]

  tagged_queries = [ds.queries[q] for q in query_ids]
  tagged_corpus = [ds.corpus[c] for c in corpus_ids]

  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

  results = []

  print('\n=== BM25 (keyword search) ===')
  m = eval_bm25(raw_queries, raw_corpus, corpus_ids, graded, query_ids, k=K)
  print(f'BM25: ndcg@{K}={m[f"ndcg@{K}"]:.4f} recall@{K}={m[f"recall@{K}"]:.4f} roc_auc={m["roc_auc"]:.4f}')
  results.append(('BM25 (keyword)', m))

  bge_tok = AutoTokenizer.from_pretrained(BGE_NAME)
  bge_tok.add_tokens(list(AddTokens().new_tokens.values()))

  print('\n=== plain BGE (no adapter) ===')
  plain_bge = AutoModel.from_pretrained(BGE_NAME)
  plain_bge.resize_token_embeddings(len(bge_tok))
  plain_bge = plain_bge.to(device).eval()
  m = eval_dense(plain_bge, bge_tok, raw_queries, raw_corpus, corpus_ids, graded,
                 query_ids, q_prefix=BGE_QUERY_PREFIX, k=K)
  print(f'plain_bge: ndcg@{K}={m[f"ndcg@{K}"]:.4f} recall@{K}={m[f"recall@{K}"]:.4f} roc_auc={m["roc_auc"]:.4f}')
  results.append(('plain BGE (zero-shot)', m))
  del plain_bge
  gc.collect()
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()

  print('\n=== bge_lora ===')
  bge_enc = load_bge_with_adapter(os.path.join(repo_root, 'bge_lora'), bge_tok)
  m = eval_dense(bge_enc, bge_tok, raw_queries, raw_corpus, corpus_ids, graded,
                 query_ids, q_prefix=BGE_QUERY_PREFIX, k=K)
  print(f'bge_lora: ndcg@{K}={m[f"ndcg@{K}"]:.4f} recall@{K}={m[f"recall@{K}"]:.4f} roc_auc={m["roc_auc"]:.4f}')
  results.append(('bge_lora', m))
  del bge_enc
  gc.collect()
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()

  etin_tok = AutoTokenizer.from_pretrained(ETIN_NAME)
  etin_tok.add_tokens(list(AddTokens().new_tokens.values()))

  print('\n=== plain Ettin (no adapter) ===')
  plain_etin = AutoModel.from_pretrained(ETIN_NAME)
  plain_etin.resize_token_embeddings(len(etin_tok))
  plain_etin = plain_etin.to(device).eval()
  m = eval_dense(plain_etin, etin_tok, raw_queries, raw_corpus, corpus_ids,
                 graded, query_ids, q_prefix='', qmax=EVAL_ETIN_MAX_LEN, dmax=EVAL_ETIN_MAX_LEN, k=K)
  print(f'plain_etin: ndcg@{K}={m[f"ndcg@{K}"]:.4f} recall@{K}={m[f"recall@{K}"]:.4f} roc_auc={m["roc_auc"]:.4f}')
  results.append(('plain Ettin (zero-shot)', m))
  del plain_etin
  gc.collect()
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()

  print('\n=== etin_lora ===')
  etin_enc = load_etin_with_adapter(os.path.join(repo_root, 'etin_lora'), etin_tok)
  m = eval_dense(etin_enc, etin_tok, tagged_queries, tagged_corpus, corpus_ids,
                 graded, query_ids, q_prefix='', qmax=EVAL_ETIN_MAX_LEN, dmax=EVAL_ETIN_MAX_LEN, k=K)
  print(f'etin_lora: ndcg@{K}={m[f"ndcg@{K}"]:.4f} recall@{K}={m[f"recall@{K}"]:.4f} roc_auc={m["roc_auc"]:.4f}')
  results.append(('etin_lora', m))

  print('\n=== SUMMARY (TEST set) ===')
  print(f'{"method":24s} {"nDCG@10":>10s} {"Recall@10":>10s} {"ROC-AUC":>10s}')
  for name, m in results:
    print(f'{name:24s} {m[f"ndcg@{K}"]:>10.4f} {m[f"recall@{K}"]:>10.4f} {m["roc_auc"]:>10.4f}')
  best = max(results, key=lambda r: r[1][f'ndcg@{K}'])
  print(f'\nBest by nDCG@{K}: {best[0]} ({best[1][f"ndcg@{K}"]:.4f})')


if __name__ == '__main__':
  main()