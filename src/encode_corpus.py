import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel

from tqdm.auto import tqdm
import json
import os
import sys

import utils
from model import model, tokenizer, device
from indexer import build_doc_text


class NFCorpusDocs(Dataset):
  def __init__(self,
    corpus_path=utils.CORPUS_JSONL,
    pdf_dir=utils.PDF_DIR
  ):
    super().__init__()

    self.pdf_paths = []
    self.texts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
      for line in f:
        item = json.loads(line)
        title = item.get('title', '')
        pdf_path = os.path.join(pdf_dir, title + '.pdf')
        if not os.path.exists(pdf_path):
          continue
        self.pdf_paths.append(pdf_path)
        self.texts.append(build_doc_text(item.get('title', ''), item.get('text', '')))

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
      max_length=utils.MAX_DOC_LENGTH,
      return_tensors='pt'
    )

  return {
    'pdf_path': [item['pdf_path'] for item in batch],
    'docs': tokenize(texts),
  }


def main():
  checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else utils.CHECKPOINT_PATH
  out_path = sys.argv[2] if len(sys.argv) > 2 else utils.ENCODED_PATH

  corpus_ds = NFCorpusDocs()
  dl_kw = dict(batch_size=utils.ENCODE_BATCH_SIZE, collate_fn=collate_fn)
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