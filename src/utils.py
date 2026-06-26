import os


# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, 'data', 'nfcorpus', 'pdf_docs')
CORPUS_JSONL = os.path.join(BASE_DIR, 'data', 'nfcorpus', 'corpus.jsonl')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'adapters', 'bge_lora')
ENCODED_PATH = os.path.join(BASE_DIR, 'corpus_encoded.pt')

# ===== Model names =====
BGE_NAME = 'BAAI/bge-small-en-v1.5'
ETTIN_NAME = 'jhu-clsp/ettin-encoder-150m'
TEACHER_NAME = 'BAAI/bge-reranker-base'

# ===== Prefixes / special tokens =====
BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '
SPECIAL_TOKS = ('<QRY>', '</QRY>', '<TLE>', '</TLE>', '<TXT>', '</TXT>')

# ===== Serve (app + CLI search) =====
HOST = '127.0.0.1'
PORT = 5050
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.55
MAX_QUERY_LENGTH = 96
MAX_DOC_LENGTH = 512

# ===== Corpus encoding =====
ENCODE_BATCH_SIZE = 16

# ===== Evaluation =====
EVAL_K = 10
EVAL_BATCH_SIZE = 128
EVAL_ETTIN_MAX_LEN = 256

# ===== Common training =====
SEED = 42

# ===== BGE LoRA training (train_bge_lora.py) =====
BGE_TRAIN_QUERY_MAX_LEN = 96
BGE_TRAIN_DOC_MAX_LEN = 512
BGE_TRAIN_BATCH_SIZE = 64
BGE_TRAIN_GRAD_ACCUM = 4
BGE_TRAIN_NUM_HARD_NEG = 7
BGE_TRAIN_NUM_RAND_NEG = 1
BGE_TRAIN_LR = 7e-4
BGE_TRAIN_WEIGHT_DECAY = 0.01
BGE_TRAIN_WARMUP_RATIO = 0.10
BGE_TRAIN_EPOCHS = 30
BGE_TRAIN_PATIENCE = 8
BGE_TRAIN_GRAD_CLIP = 1.0
BGE_TRAIN_TEMP = 0.02
BGE_TRAIN_TEACHER_TEMP = 1.0
BGE_TRAIN_KL_WEIGHT = 2.0
BGE_TRAIN_REMINE_EVERY = 1
BGE_TRAIN_NUM_WORKERS = 4
BGE_TRAIN_CHECKPOINT_DIR = os.path.join(BASE_DIR, 'bge_lora_checkpoints')
BGE_TRAIN_TEACHER_MAX_LEN = 512
BGE_TRAIN_TEACHER_CHUNK = 32
BGE_TRAIN_RERANK_POOL = 100

# ===== Ettin LoRA training (train_ettin_lora.py) =====
ETTIN_LORA_MAX_LEN = 256
ETTIN_LORA_BATCH_SIZE = 2
ETTIN_LORA_RANK = 16
ETTIN_LORA_ALPHA = 32
ETTIN_LORA_DROPOUT = 0.065
ETTIN_LORA_LR = 1e-3
ETTIN_LORA_EPOCHS = 200
ETTIN_LORA_TEMP = 0.2
ETTIN_LORA_WARMUP_RATIO = 0.10
ETTIN_LORA_CHECKPOINT_DIR = os.path.join(BASE_DIR, 'ettin_lora_checkpoints')

# ===== Ettin full fine-tune (train_ettin_finetune.py) =====
ETTIN_FT_QUERY_MAX_LEN = 64
ETTIN_FT_DOC_MAX_LEN = 512
ETTIN_FT_BATCH_SIZE = 128
ETTIN_FT_NUM_HARD_NEG = 16
ETTIN_FT_NUM_RAND_NEG = 16
ETTIN_FT_LR = 5e-5
ETTIN_FT_WEIGHT_DECAY = 0.01
ETTIN_FT_WARMUP_RATIO = 0.06
ETTIN_FT_EPOCHS = 60
ETTIN_FT_PATIENCE = 8
ETTIN_FT_GRAD_CLIP = 1.0
ETTIN_FT_TEMP = 0.05
ETTIN_FT_EVAL_K = 10

# ===== Adapter weight downloads =====
DOWNLOAD_WEIGHTS = True
ETTIN_LORA_GDRIVE_ID = '1Z992gM2ub-igVBXsQ1JySVttjCInzZTo'
BGE_LORA_GDRIVE_ID = '1NP1wKzl5KNoIF4K6YFgfNOAAU_I6DuZ9'


def maybe_download_weights():
  if not DOWNLOAD_WEIGHTS:
    return
  import gdown
  ettin_weights = os.path.join(BASE_DIR, 'adapters', 'ettin_lora', 'adapter_model.safetensors')
  if not os.path.exists(ettin_weights):
    print('Downloading Ettin LoRA weights...')
    gdown.download(id=ETTIN_LORA_GDRIVE_ID, output=ettin_weights, quiet=False)
  bge_weights = os.path.join(BASE_DIR, 'adapters', 'bge_lora', 'adapter_model.safetensors')
  if not os.path.exists(bge_weights):
    print('Downloading BGE LoRA weights...')
    gdown.download(id=BGE_LORA_GDRIVE_ID, output=bge_weights, quiet=False)


class AddTokens():
  def __init__(self):
    self.new_tokens = {
      'query_begin': '<QRY>',
      'query_end': '</QRY>',
      'title_begin': '<TLE>',
      'title_end': '</TLE>',
      'text_begin': '<TXT>',
      'text_end': '</TXT>'
    }

  def add_query_tokens(self, query):
    return self.new_tokens['query_begin'] + query + self.new_tokens['query_end']

  def add_title_tokens(self, title):
    return self.new_tokens['title_begin'] + title + self.new_tokens['title_end']

  def add_text_tokens(self, text):
    return self.new_tokens['text_begin'] + text + self.new_tokens['text_end']