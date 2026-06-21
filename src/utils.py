import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_DIR = os.path.join(BASE_DIR, 'data', 'nfcorpus', 'pdf_docs')
CORPUS_JSONL = os.path.join(BASE_DIR, 'data', 'nfcorpus', 'corpus.jsonl')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'bge_lora')
ENCODED_PATH = os.path.join(BASE_DIR, 'corpus_encoded.pt')

DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.55
MAX_QUERY_LENGTH = 96
MAX_DOC_LENGTH = 512

BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

HOST = "127.0.0.1"
PORT = 5050


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