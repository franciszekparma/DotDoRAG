import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_DIR = os.path.join(BASE_DIR, 'data', 'nfcorpus', 'pdf_docs')
CORPUS_JSONL = os.path.join(BASE_DIR, 'data', 'nfcorpus', 'corpus.jsonl')
CHECKPOINT_PATH = BASE_DIR
ENCODED_PATH = os.path.join(BASE_DIR, 'corpus_encoded.pt')

DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.75
MAX_QUERY_LENGTH = 256
MAX_DOC_LENGTH = 256

HOST = "127.0.0.1"
PORT = 5000