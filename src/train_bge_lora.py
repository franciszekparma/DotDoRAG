import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup

from tqdm.auto import tqdm
import json
import random
import pandas as pd

from utils import AddTokens
from model import bge, bge_tokenizer, device


class NFCorpusDataset(Dataset):
  def __init__(self,
    qrels_path='data/nfcorpus/qrels/',
    queries_path='data/nfcorpus/queries.jsonl',
    corpus_path='data/nfcorpus/corpus.jsonl',
    num_negatives=8,
    split='train'
  ):
    super().__init__()
    if split.upper() == 'TRAIN':
      qrels_path += 'train.tsv'
    elif split.upper() == 'TEST':
      qrels_path += 'test.tsv'
    elif split.upper() == 'DEV':
      qrels_path += 'dev.tsv'

    self.ta = AddTokens()
    self.num_negatives = num_negatives

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

    pos_id = random.choice(list(pos_ids))

    return {
      'query_id': query_id,
      'query_text': self.queries[query_id],
      'positive_texts': [self.corpus[pos_id]],
      'negative_texts': [self.corpus[cid] for cid in neg_ids]
    }


class MultiNCELoss(nn.Module):
  def __init__(self, temp=0.2):
    super().__init__()
    
    self.temp = temp
    
  def calc_loss(self, que_vec, pos_vecs, neg_vecs):
    B = que_vec.size(0)
    
    if pos_vecs.dim() == 2:
      pos_vecs = pos_vecs.unsqueeze(1)
    if neg_vecs.dim() == 2:
      neg_vecs = neg_vecs.view(B, -1, neg_vecs.size(-1))
      
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


def collate_fn(batch):
  queries = [item['query_text'] for item in batch]
  positives = [text for item in batch for text in item['positive_texts']]
  negatives = [text for item in batch for text in item['negative_texts']]

  def tokenize(texts):
    return bge_tokenizer(
      texts,
      padding=True,
      truncation=True,
      max_length=256,
      return_tensors='pt'
    )

  return {
    'query_id': [item['query_id'] for item in batch],
    'query': tokenize(queries),
    'positives': tokenize(positives),
    'negatives': tokenize(negatives),
  }



def main():
  train_ds = NFCorpusDataset(split='train')
  val_ds = NFCorpusDataset(split='dev')
  dl_kw = dict(batch_size=2, collate_fn=collate_fn)
  train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw)
  val_dl = DataLoader(val_ds, shuffle=False, drop_last=False, **dl_kw)

  peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.065,
    bias="none",
    task_type=None,
    init_lora_weights=True
  )
  model = bge
  model.enc = get_peft_model(model.enc, peft_config)

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
  loss_fn = MultiNCELoss()
  
  epochs = 200
  
  total_steps = len(train_dl) * epochs
  warmup_steps = int(0.1 * total_steps)

  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
  )
  
  best_val = float('inf')
  
  for epoch in range(epochs):
    model.train()
    train_losses = []
    for X in tqdm(train_dl, desc=f'Epoch {epoch + 1}/{epochs} train', unit='batch'):
      queries, positives, negatives = X['query'], X['positives'], X['negatives']
      queries, positives, negatives = {k: v.to(device) for k, v in queries.items()}, {k: v.to(device) for k, v in positives.items()}, {k: v.to(device) for k, v in negatives.items()}
      
      que_vecs, pos_vecs, neg_vecs = model(queries, positives, negatives)
      
      loss = loss_fn.calc_loss(que_vecs, pos_vecs, neg_vecs)
      
      train_losses.append(loss.item())
      optimizer.zero_grad()
      
      loss.backward()
      optimizer.step()
      scheduler.step()

    val_losses = []
    
    model.eval()
    with torch.inference_mode():
      for X in tqdm(val_dl, desc=f'Epoch {epoch + 1}/{epochs} val', leave=False, unit='batch'):
        queries, positives, negatives = X['query'], X['positives'], X['negatives']
        queries, positives, negatives = {k: v.to(device) for k, v in queries.items()}, {k: v.to(device) for k, v in positives.items()}, {k: v.to(device) for k, v in negatives.items()}
        
        que_vecs, pos_vecs, neg_vecs = model(queries, positives, negatives)
        
        val_losses.append(loss_fn.calc_loss(que_vecs, pos_vecs, neg_vecs).item())

    train_loss, val_loss = sum(train_losses) / len(train_losses), sum(val_losses) / len(val_losses)
    
    saved = ''
    if val_loss < best_val:
      best_val = val_loss
      checkpoint_path = f'..bge_lora_checkpoints/epoch_{epoch+1}_train_{train_loss:.4f}_val_{val_loss:.4f}'
      model.enc.save_pretrained(checkpoint_path)
      saved = f'!!!Saved the best model at: {checkpoint_path}!!!'
      
    tqdm.write(f'Epoch {epoch + 1}/{epochs}: train={train_loss:.4f} val={val_loss:.4f} (best val={best_val:.4f}){saved}')
  

if __name__ == '__main__':
  main()