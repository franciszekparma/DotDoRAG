import torch
import torch.nn.functional as F
from peft import PeftModel

import sys

from utils import AddTokens
from model import model, tokenizer, device


def main():
  checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else '.'
  encoded_path = sys.argv[2] if len(sys.argv) > 2 else 'corpus_encoded.pt'
  threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85

  ta = AddTokens()

  cached = torch.load(encoded_path, map_location=device)
  doc_vecs, pdf_paths = cached['doc_vecs'].to(device), cached['pdf_paths']

  model.enc = PeftModel.from_pretrained(model.enc, checkpoint_path)
  model.to(device)

  query = input('query> ').strip()

  model.eval()
  with torch.inference_mode():
    que = tokenizer(
      [ta.add_query_tokens(query)],
      padding=True,
      truncation=True,
      max_length=256,
      return_tensors='pt'
    )
    que = {k: v.to(device) for k, v in que.items()}

    que_vec = model.enc(**que).last_hidden_state[..., 0, :]
    que_vec = F.normalize(que_vec, p=2, dim=1)

    sims = (que_vec @ doc_vecs.T).squeeze(0)
    scores, idxs = torch.sort(sims, descending=True)

  hits = [pdf_paths[i.item()] for s, i in zip(scores, idxs) if s.item() >= threshold]

  if not hits:
    print('No matching papers found. Try rephrasing your query.')
    return

  print(f'Found {len(hits)} matching paper(s):')
  for path in hits:
    print(path)


if __name__ == '__main__':
  main()