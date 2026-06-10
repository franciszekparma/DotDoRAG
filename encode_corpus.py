import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel

from tqdm.auto import tqdm
import json
import os
import sys

from utils import AddTokens
from model import model, tokenizer, device


class NFCorpusDocs(Dataset):
  def __init__(self,
    corpus_path='data/nfcorpus/corpus.jsonl',
    pdf_dir='data/nfcorpus/pdf_docs'
  ):
    super().__init__()
    self.ta = AddTokens()

    self.pdf_paths = []
    self.texts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
      for line in f:
        item = json.loads(line)
        title = item.get('title', '')
        pdf_path = os.path.join(pdf_dir, title + '.pdf')
        if not os.path.exists(pdf_path):
          continue
        doc = ""
        if 'title' in item: doc += self.ta.add_title_tokens(item['title']) + " "
        if 'text' in item: doc += self.ta.add_text_tokens(item['text'])
        self.pdf_paths.append(pdf_path)
        self.texts.append(doc.strip())

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    return {
      'pdf_path': self.pdf_paths[idx],
      'doc_text': self.texts[idx],
    }


def collate_fn(batch):
  texts = [item['doc_text'] for item in batch]

  def tokenize(texts):
    return tokenizer(
      texts,
      padding=True,
      truncation=True,
      max_length=256,
      return_tensors='pt'
    )

  return {
    'pdf_path': [item['pdf_path'] for item in batch],
    'docs': tokenize(texts),
  }


def main():
  checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else '.'
  out_path = sys.argv[2] if len(sys.argv) > 2 else 'corpus_encoded.pt'

  corpus_ds = NFCorpusDocs()
  dl_kw = dict(batch_size=16, collate_fn=collate_fn)
  corpus_dl = DataLoader(corpus_ds, shuffle=False, drop_last=False, **dl_kw)

  model.enc = PeftModel.from_pretrained(model.enc, checkpoint_path)
  model.to(device)

  doc_vecs = []
  pdf_paths = []

  model.eval()
  with torch.inference_mode():
    for X in tqdm(corpus_dl, desc='encoding corpus', unit='batch'):
      docs = {k: v.to(device) for k, v in X['docs'].items()}

      vecs = model.enc(**docs).last_hidden_state[..., 0, :]

      doc_vecs.append(F.normalize(vecs, p=2, dim=1))
      pdf_paths.extend(X['pdf_path'])

    doc_vecs = torch.cat(doc_vecs, dim=0).cpu()

  torch.save({'doc_vecs': doc_vecs, 'pdf_paths': pdf_paths}, out_path)

  tqdm.write(f'Saved {len(pdf_paths)} encoded docs to {out_path}')


if __name__ == '__main__':
  main()