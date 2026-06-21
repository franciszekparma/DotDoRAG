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

class Etin(nn.Module):
  def __init__(self, name="jhu-clsp/ettin-encoder-150m"):
    super().__init__()
    
    self.enc = AutoModel.from_pretrained(name)
    
  def forward(self, queries, positives, negatives):
    return self.enc(**queries).last_hidden_state[..., 0, :], self.enc(**positives).last_hidden_state[..., 0, :], self.enc(**negatives).last_hidden_state[..., 0, :]

class BGE(nn.Module):
  def __init__(self, name="BAAI/bge-small-en-v1.5"):
    super().__init__()
    
    self.enc = AutoModel.from_pretrained(name)
    
  def forward(self, queries, positives, negatives):
    return self.enc(**queries).last_hidden_state[..., 0, :], self.enc(**positives).last_hidden_state[..., 0, :], self.enc(**negatives).last_hidden_state[..., 0, :]


etin = Etin().to(device)
bge = BGE().to(device)

token_adder = AddTokens()

etin_tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")
etin_tokenizer.add_tokens(list(token_adder.new_tokens.values()))
etin.enc.resize_token_embeddings(len(etin_tokenizer))

bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
bge_tokenizer.add_tokens(list(token_adder.new_tokens.values()))
bge.enc.resize_token_embeddings(len(bge_tokenizer))

model = bge
tokenizer = bge_tokenizer