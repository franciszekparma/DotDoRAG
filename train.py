import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import json
import random
import pandas as pd
import torch

from utils import AddTokens
from model import device

class NFCorpusDataset(Dataset):
	def __init__(self, qrels_path, queries_path, corpus_path, num_negatives):
		self.ta = AddTokens()
		self.num_negatives = num_negatives
		
		self.queries = {}
		with open(queries_path, 'r') as f:
			for line in f:
				item = json.loads(line)
				self.queries[item['_id']] = self.ta.add_query_tokens(item['text'])
				
		self.corpus = {}
		with open(corpus_path, 'r') as f:
			for line in f:
				item = json.loads(line)
				doc = ""
				if 'title' in item: doc += self.ta.add_title_tokens(item['title']) + " "
				if 'text' in item: doc += self.ta.add_text_tokens(item['text'])
				self.corpus[item['_id']] = doc.strip()
				
		self.all_corpus_ids = list(self.corpus.keys())
		
		qrels = pd.read_csv(qrels_path, sep='\t')
		self.data = qrels.groupby('query-id')['corpus-id'].apply(list).reset_index()
		
	def __len__(self):
		return len(self.data)
		
	def __getitem__(self, idx):
		row = self.data.iloc[idx]
		query_id = row['query-id']
		pos_ids = set(row['corpus-id'])
		
		neg_ids = []
		while len(neg_ids) < self.num_negatives:
			sampled = random.choice(self.all_corpus_ids)
			if sampled not in pos_ids:
				neg_ids.append(sampled)
				
		return {
			'query_id': query_id,
			'query_text': self.queries[query_id],
			'positive_texts': [self.corpus[cid] for cid in pos_ids],
			'negative_texts': [self.corpus[cid] for cid in neg_ids]
		}


class MultiNCELoss(nn.Module):
  def __init__(self, temp=0.3):
    super().__init__()
    
    self.temp = temp
    
  def calc_loss(self, que_vec, pos_vecs, neg_vecs):
    que_vec = (F.normalize(que_vec, p=2, dim=1)).unsqueeze(1)
    pos_vecs = F.normalize(pos_vecs, p=2, dim=2)
    neg_vecs = F.normalize(neg_vecs, p=2, dim=2)
    
    pos_logits = (que_vec * pos_vecs).sum(dim=-1) / self.temp
    neg_logits = (que_vec * neg_vecs).sum(dim=-1) / self.temp
    
    pos_exp = torch.exp(pos_logits)
    neg_exp = torch.exp(neg_logits)
    
    numer = pos_exp.sum(dim=-1)
    denom = numer + neg_exp.sum(dim=-1)
    
    loss = -torch.log((numer + 1e-8) / (denom + 1e-8))
    
    return torch.mean(loss)