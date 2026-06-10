import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging

from utils import AddTokens

logging.set_verbosity_error()

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu'
)

class RAGEtin(nn.Module):
  def __init__(self, name="jhu-clsp/ettin-encoder-150m"):
    super().__init__()
    
    self.enc = AutoModel.from_pretrained(name)
    
  def forward(self, queries, positives, negatives):
    return self.enc(**queries).last_hidden_state[..., 0, :], self.enc(**positives).last_hidden_state[..., 0, :], self.enc(**negatives).last_hidden_state[..., 0, :]

token_adder = AddTokens()


tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")
tokenizer.add_tokens(list(token_adder.new_tokens.values()))

model = RAGEtin()
model.enc.resize_token_embeddings(len(tokenizer))
model = model.to(device)