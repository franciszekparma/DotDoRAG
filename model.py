import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from utils import AddTokens

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu'
)

class Encoder(nn.Module):
  def __init__(self, name="jhu-clsp/ettin-encoder-150m"):
    super().__init__()
    
    self.enc = AutoModel.from_pretrained(name)
    self.enc.final_layer_norm = nn.Identity()
    
  def forward(self, X):
    return self.enc(**X).last_hidden_state[..., 0, :]

token_adder = AddTokens()


tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")
tokenizer.add_tokens(list(token_adder.new_tokens.values()))

model = Encoder().to(device)
model.enc.resize_token_embeddings(len(tokenizer))