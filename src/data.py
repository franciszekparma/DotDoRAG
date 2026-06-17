import json
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils import AddTokens


SPLIT_FILES = {
  'TRAIN': 'train.tsv',
  'VAL':   'dev.tsv',
  'DEV':   'dev.tsv',
  'TEST':  'test.tsv',
}


class NFCorpusDataset(Dataset):
  def __init__(self,
    qrels_path='data/nfcorpus/qrels/',
    queries_path='data/nfcorpus/queries.jsonl',
    corpus_path='data/nfcorpus/corpus.jsonl',
    split='train',
    num_hard_neg=0,
    num_rand_neg=8,
    mine_hard_negs=False,
    num_negatives=None,
    seed=None,
  ):
    super().__init__()
    split = split.upper()
    if split not in SPLIT_FILES:
      raise ValueError(f"split must be one of {sorted(SPLIT_FILES)}, got {split!r}")
    if num_negatives is not None:
      num_rand_neg = num_negatives
      num_hard_neg = 0

    self.split = split
    self.ta = AddTokens()
    self.num_hard_neg = num_hard_neg
    self.num_rand_neg = num_rand_neg
    self._rng = random.Random(seed)

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

    qrels = pd.read_csv(qrels_path + SPLIT_FILES[split], sep='\t')
    self.graded = {
      qid: dict(zip(g['corpus-id'], g['score']))
      for qid, g in qrels.groupby('query-id')
    }
    self.query_ids = [qid for qid in self.graded if qid in self.queries]
    self.pos_ids = {qid: list(self.graded[qid].keys()) for qid in self.query_ids}
    self.pos_sets = {qid: set(ids) for qid, ids in self.pos_ids.items()}

    self.hard_negs = {qid: [] for qid in self.query_ids}
    if mine_hard_negs and self.num_hard_neg > 0:
      self._mine_hard_negatives()

  def _mine_hard_negatives(self, pool_size=200, keep_per_query=100):
    try:
      from rank_bm25 import BM25Okapi
    except ImportError as e:
      raise RuntimeError(
        "Install rank_bm25 to mine hard negatives: pip install rank_bm25"
      ) from e

    corpus_tok = [self.corpus[cid].lower().split() for cid in self.corpus_ids]
    bm25 = BM25Okapi(corpus_tok)
    pool_size = min(pool_size, len(self.corpus_ids))
    for qid in tqdm(self.query_ids, desc=f'BM25 hard-neg mining ({self.split})'):
      pos = self.pos_sets[qid]
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
    pos_ids = self.pos_ids[qid]
    pos_set = self.pos_sets[qid]
    pos_id = self._rng.choice(pos_ids)

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
      'graded_positives': pos_set,
      'query_text': self.queries[qid],
      'positive_texts': [self.corpus[pos_id]],
      'negative_texts': [self.corpus[cid] for cid in negs],
    }


def make_splits(num_hard_neg=0, num_rand_neg=7, mine_hard_negs=False, seed=None, **kwargs):
  total_neg = max(1, num_hard_neg + num_rand_neg)
  train = NFCorpusDataset(
    split='TRAIN',
    num_hard_neg=num_hard_neg,
    num_rand_neg=num_rand_neg,
    mine_hard_negs=mine_hard_negs,
    seed=seed,
    **kwargs,
  )
  val = NFCorpusDataset(
    split='VAL',
    num_hard_neg=0,
    num_rand_neg=total_neg,
    mine_hard_negs=False,
    seed=seed,
    **kwargs,
  )
  test = NFCorpusDataset(
    split='TEST',
    num_hard_neg=0,
    num_rand_neg=total_neg,
    mine_hard_negs=False,
    seed=seed,
    **kwargs,
  )
  return train, val, test


def make_train_val(num_negatives=8, **kwargs):
  train, val, _ = make_splits(num_hard_neg=0, num_rand_neg=num_negatives, **kwargs)
  return train, val
