"""Shared helpers for encoding documents and appending them to the index."""
import os
import threading

import torch
import torch.nn.functional as F

import config
from utils import AddTokens


_lock = threading.Lock()
_ta = AddTokens()


def build_doc_text(title, text):
    parts = []
    if title:
        parts.append(_ta.add_title_tokens(title))
    if text:
        parts.append(_ta.add_text_tokens(text))
    return ' '.join(parts).strip()


@torch.inference_mode()
def encode_doc(model, tokenizer, device, title, text):
    doc_text = build_doc_text(title, text)
    enc = tokenizer(
        [doc_text],
        padding=True,
        truncation=True,
        max_length=config.MAX_DOC_LENGTH,
        return_tensors='pt',
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    vec = model.enc(**enc).last_hidden_state[..., 0, :]
    return F.normalize(vec, p=2, dim=1)


def append_to_index(doc_vecs, pdf_paths, new_vec, new_path, encoded_path=config.ENCODED_PATH):
    """Append a vector + path to the in-memory tensors and persist to disk.
    Returns the updated doc_vecs tensor (may be a new object).
    """
    with _lock:
        doc_vecs = torch.cat([doc_vecs, new_vec], dim=0)
        pdf_paths.append(new_path)
        torch.save(
            {'doc_vecs': doc_vecs.detach().cpu(), 'pdf_paths': pdf_paths},
            encoded_path,
        )
    return doc_vecs