import os
import json
from urllib.parse import quote

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from peft import PeftModel
from werkzeug.utils import secure_filename

import config
from utils import AddTokens
from model import model, tokenizer, device
from indexer import encode_doc, append_to_index


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB upload cap

print('Loading encoded corpus...')
_cached = torch.load(config.ENCODED_PATH, map_location=device)
doc_vecs = _cached['doc_vecs'].to(device)
pdf_paths = list(_cached['pdf_paths'])

print('Loading adapter weights...')
model.enc = PeftModel.from_pretrained(model.enc, config.CHECKPOINT_PATH)
model.to(device)
model.eval()

ta = AddTokens()


def load_metadata():
    """Build a metadata index keyed by both the title and the filename stem so we can
    look up by either."""
    meta = {}
    if not os.path.exists(config.CORPUS_JSONL):
        return meta
    with open(config.CORPUS_JSONL, 'r', encoding='utf-8') as f:
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


@torch.inference_mode()
def encode_query(query):
    enc = tokenizer(
        [ta.add_query_tokens(query)],
        padding=True,
        truncation=True,
        max_length=config.MAX_QUERY_LENGTH,
        return_tensors='pt',
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    vec = model.enc(**enc).last_hidden_state[..., 0, :]
    return F.normalize(vec, p=2, dim=1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    payload = request.get_json(silent=True) or {}
    query = (payload.get('query') or '').strip()
    top_k = int(payload.get('top_k') or config.DEFAULT_TOP_K)
    threshold = float(payload.get('threshold') if payload.get('threshold') is not None else config.DEFAULT_THRESHOLD)

    if not query:
        return jsonify({'error': 'Empty query'}), 400

    que_vec = encode_query(query)
    sims = (que_vec @ doc_vecs.T).squeeze(0)
    scores, idxs = torch.sort(sims, descending=True)

    results = []
    for s, i in zip(scores.tolist(), idxs.tolist()):
        if s < threshold:
            break
        path = pdf_paths[i]
        filename = os.path.basename(path)
        stem = filename_to_title(filename)
        meta = META_BY_KEY.get(filename) or META_BY_KEY.get(stem) or {}
        results.append({
            'title': meta.get('title') or stem,
            'snippet': meta.get('snippet', ''),
            'score': round(float(s), 4),
            'filename': filename,
            'pdf_url': f'/pdf/{quote(filename)}',
        })
        if len(results) >= top_k:
            break

    return jsonify({'query': query, 'results': results})


@app.route('/add', methods=['POST'])
def add_document():
    global doc_vecs

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

    os.makedirs(config.PDF_DIR, exist_ok=True)
    save_path = os.path.join(config.PDF_DIR, safe_name)

    base, ext = os.path.splitext(safe_name)
    i = 1
    while os.path.exists(save_path):
        safe_name = f'{base}_{i}{ext}'
        save_path = os.path.join(config.PDF_DIR, safe_name)
        i += 1

    uploaded.save(save_path)

    new_vec = encode_doc(model, tokenizer, device, title, text)
    doc_vecs = append_to_index(doc_vecs, pdf_paths, new_vec, save_path)

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
    safe_dir = os.path.realpath(config.PDF_DIR)
    target = os.path.realpath(os.path.join(safe_dir, filename))
    if not target.startswith(safe_dir + os.sep) or not os.path.isfile(target):
        abort(404)
    return send_from_directory(safe_dir, filename, mimetype='application/pdf')


@app.route('/stats')
def stats():
    return jsonify({'total_docs': len(pdf_paths)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', config.PORT))
    app.run(host=config.HOST, port=port, debug=False)