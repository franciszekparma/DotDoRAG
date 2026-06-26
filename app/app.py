import os
import sys
import json
from urllib.parse import quote

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from peft import PeftModel
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer
from werkzeug.utils import secure_filename

import utils
from model import model, tokenizer, device
from indexer import build_doc_text, encode_doc, append_to_index

MODELS = ('bge', 'bge_lora', 'ettin', 'ettin_lora', 'bm25')
_ta = utils.AddTokens()
_doc_vecs_cache = {}
_enc_cache = {}


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

utils.maybe_download_weights()

print('Loading encoded corpus...')
_cached = torch.load(utils.ENCODED_PATH, map_location=device)
pdf_paths = list(_cached['pdf_paths'])
_doc_vecs_cache['bge_lora'] = _cached['doc_vecs'].to(device)

print('Loading adapter weights...')
model.enc = PeftModel.from_pretrained(model.enc, utils.CHECKPOINT_PATH)
model.to(device)
model.eval()
_enc_cache['bge_lora'] = (model.enc, tokenizer)

def load_metadata():
  meta = {}
  if not os.path.exists(utils.CORPUS_JSONL):
    return meta
  with open(utils.CORPUS_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
      item = json.loads(line)
      title = item.get('title', '')
      if not title:
        continue
      entry = {'title': title, 'snippet': (item.get('text', '') or '')[:320]}
      meta[title] = entry
  return meta


META_BY_KEY = load_metadata()


def filename_to_title(filename):
  return os.path.splitext(os.path.basename(filename))[0]


def _load_corpus_texts():
  plain, tagged = {}, {}
  if not os.path.exists(utils.CORPUS_JSONL):
    return plain, tagged
  with open(utils.CORPUS_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
      item = json.loads(line)
      title = item.get('title', '')
      if not title:
        continue
      text = item.get('text', '') or ''
      plain[title] = build_doc_text(title, text)
      doc = ''
      if title:
        doc += _ta.add_title_tokens(title) + ' '
      if text:
        doc += _ta.add_text_tokens(text)
      tagged[title] = doc.strip()
  return plain, tagged


_plain_by_title, _tagged_by_title = _load_corpus_texts()
doc_texts = [_plain_by_title.get(filename_to_title(os.path.basename(p)), '') for p in pdf_paths]
tagged_doc_texts = [_tagged_by_title.get(filename_to_title(os.path.basename(p)), '') for p in pdf_paths]
bm25_index = BM25Okapi([t.lower().split() for t in doc_texts])


@torch.inference_mode()
def _encode_texts(enc, tok, texts, max_len):
  vecs = []
  for i in range(0, len(texts), utils.ENCODE_BATCH_SIZE):
    batch = tok(
      texts[i:i + utils.ENCODE_BATCH_SIZE],
      padding=True, truncation=True, max_length=max_len, return_tensors='pt',
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    v = enc(**batch).last_hidden_state[..., 0, :]
    vecs.append(F.normalize(v, p=2, dim=1))
  return torch.cat(vecs, dim=0)


def _get_encoder(name):
  if name in _enc_cache:
    return _enc_cache[name]
  tok = AutoTokenizer.from_pretrained(
    utils.BGE_NAME if name.startswith('bge') else utils.ETTIN_NAME
  )
  tok.add_tokens(list(_ta.new_tokens.values()))
  enc = AutoModel.from_pretrained(
    utils.BGE_NAME if name.startswith('bge') else utils.ETTIN_NAME
  )
  enc.resize_token_embeddings(len(tok))
  if name.endswith('_lora'):
    adapter = os.path.join(utils.BASE_DIR, 'adapters', name)
    enc = PeftModel.from_pretrained(enc, adapter)
  _enc_cache[name] = (enc.to(device).eval(), tok)
  return _enc_cache[name]


def _get_doc_vecs(name):
  if name not in _doc_vecs_cache:
    enc, tok = _get_encoder(name)
    texts = tagged_doc_texts if name == 'ettin_lora' else doc_texts
    max_len = utils.EVAL_ETTIN_MAX_LEN if name.startswith('ettin') else utils.MAX_DOC_LENGTH
    _doc_vecs_cache[name] = _encode_texts(enc, tok, texts, max_len)
  return _doc_vecs_cache[name]


@torch.inference_mode()
def _encode_query(name, query):
  enc, tok = _get_encoder(name)
  if name.startswith('bge'):
    text = utils.BGE_QUERY_PREFIX + query
    max_len = utils.MAX_QUERY_LENGTH
  elif name == 'ettin_lora':
    text = _ta.add_query_tokens(query)
    max_len = utils.EVAL_ETTIN_MAX_LEN
  else:
    text = query
    max_len = utils.EVAL_ETTIN_MAX_LEN
  batch = tok([text], padding=True, truncation=True, max_length=max_len, return_tensors='pt')
  batch = {k: v.to(device) for k, v in batch.items()}
  vec = enc(**batch).last_hidden_state[..., 0, :]
  return F.normalize(vec, p=2, dim=1)


def _search_scores(name, query):
  if name == 'bm25':
    return bm25_index.get_scores(query.lower().split())
  que_vec = _encode_query(name, query)
  return (que_vec @ _get_doc_vecs(name).T).squeeze(0)


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
  payload = request.get_json(silent=True) or {}
  query = (payload.get('query') or '').strip()
  top_k = int(payload.get('top_k') or utils.DEFAULT_TOP_K)
  threshold = float(payload.get('threshold') if payload.get('threshold') is not None else utils.DEFAULT_THRESHOLD)
  model_name = (payload.get('model') or 'bge_lora').strip()
  if model_name not in MODELS:
    return jsonify({'error': f'Unknown model: {model_name}'}), 400

  if not query:
    return jsonify({'error': 'Empty query'}), 400

  sims = _search_scores(model_name, query)
  if model_name == 'bm25':
    idxs = np.argsort(-sims)
    scores = sims[idxs]
  else:
    scores, idxs = torch.sort(sims, descending=True)
    scores, idxs = scores.tolist(), idxs.tolist()

  results = []
  for s, i in zip(scores, idxs):
    s = float(s)
    i = int(i)
    if model_name != 'bm25' and s < threshold:
      break
    path = pdf_paths[i]
    filename = os.path.basename(path)
    stem = filename_to_title(filename)
    meta = META_BY_KEY.get(filename) or META_BY_KEY.get(stem) or {}
    results.append({
      'title': meta.get('title') or stem,
      'snippet': meta.get('snippet', ''),
      'score': round(s, 4),
      'filename': filename,
      'pdf_url': f'/pdf/{quote(filename)}',
    })
    if len(results) >= top_k:
      break

  return jsonify({'query': query, 'model': model_name, 'results': results})


@app.route('/add', methods=['POST'])
def add_document():

  uploaded = request.files.get('pdf')
  title = (request.form.get('title') or '').strip()
  text = (request.form.get('text') or '').strip()

  if not uploaded or uploaded.filename == '':
    return jsonify({'error': 'PDF file is required'}), 400
  if not title and not text:
    return jsonify({'error': 'Provide at least a title or some text to index'}), 400

  safe_name = secure_filename(uploaded.filename)
  if not safe_name.lower().endswith('.pdf'):
    safe_name = safe_name + '.pdf'

  if not title:
    title = filename_to_title(safe_name)

  os.makedirs(utils.PDF_DIR, exist_ok=True)
  save_path = os.path.join(utils.PDF_DIR, safe_name)

  base, ext = os.path.splitext(safe_name)
  i = 1
  while os.path.exists(save_path):
    safe_name = f'{base}_{i}{ext}'
    save_path = os.path.join(utils.PDF_DIR, safe_name)
    i += 1

  uploaded.save(save_path)

  new_vec = encode_doc(model, tokenizer, device, title, text)
  _doc_vecs_cache['bge_lora'] = append_to_index(
    _doc_vecs_cache['bge_lora'], pdf_paths, new_vec, save_path,
  )
  doc_texts.append(build_doc_text(title, text))
  tagged = _ta.add_title_tokens(title)
  if text:
    tagged = (tagged + ' ' + _ta.add_text_tokens(text)).strip()
  tagged_doc_texts.append(tagged)
  global bm25_index
  bm25_index = BM25Okapi([t.lower().split() for t in doc_texts])
  _doc_vecs_cache.pop('bge', None)
  _doc_vecs_cache.pop('ettin', None)
  _doc_vecs_cache.pop('ettin_lora', None)

  entry = {'title': title, 'snippet': text[:320]}
  META_BY_KEY[safe_name] = entry
  META_BY_KEY[filename_to_title(safe_name)] = entry

  return jsonify({
    'ok': True,
    'title': title,
    'filename': safe_name,
    'total_docs': len(pdf_paths),
    'pdf_url': f'/pdf/{quote(safe_name)}',
  })


@app.route('/pdf/<path:filename>')
def serve_pdf(filename):
  safe_dir = os.path.realpath(utils.PDF_DIR)
  target = os.path.realpath(os.path.join(safe_dir, filename))
  if not target.startswith(safe_dir + os.sep) or not os.path.isfile(target):
    abort(404)
  return send_from_directory(safe_dir, filename, mimetype='application/pdf')


@app.route('/stats')
def stats():
  return jsonify({'total_docs': len(pdf_paths), 'models': list(MODELS)})


if __name__ == '__main__':
  port = int(os.environ.get('PORT', utils.PORT))
  app.run(host=utils.HOST, port=port, debug=False)