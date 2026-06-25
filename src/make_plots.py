"""Generate the figures used in README.md.

Two kinds of plots:
  1. From eval_results.json (cheap):
       fig_metrics_bars.png  fig_lift.png   — all four metrics per method
  2. From real model outputs (loads BGE base + LoRA, encodes a sample):
       fig_similarity_dist.png  fig_embedding_tsne.png

Run from repo root:
    python src/make_plots.py
"""
import contextlib
import gc
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import NFCorpusDataset
from utils import (
  AddTokens, BGE_NAME, BGE_QUERY_PREFIX, SPECIAL_TOKS,
  MAX_QUERY_LENGTH, MAX_DOC_LENGTH,
)


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'plots')
os.makedirs(OUT, exist_ok=True)

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')

# ---------- shared style ----------
plt.rcParams.update({
  'font.family': 'DejaVu Sans',
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.grid': True,
  'grid.alpha': 0.22,
  'grid.linestyle': '--',
  'axes.titlepad': 12,
  'axes.titleweight': 'bold',
})

PALETTE = {
  'plain_etin': '#94a3b8',
  'etin_lora':  '#64748b',
  'plain_bge':  '#7dd3fc',
  'BM25':       '#f59e0b',
  'bge_lora':   '#0ea5e9',
}
PRETTY = {
  'plain_etin': 'plain Ettin',
  'etin_lora':  'Ettin + LoRA',
  'plain_bge':  'plain BGE',
  'BM25':       'BM25',
  'bge_lora':   'BGE + LoRA',
}


# ============================================================
# Cheap plots from eval_results.json
# ============================================================
def plots_from_eval_json():
  with open(os.path.join(BASE, 'eval_results.json')) as f:
    data = json.load(f)
  order = ['plain_etin', 'etin_lora', 'plain_bge', 'BM25', 'bge_lora']
  results = sorted(data['results'], key=lambda r: order.index(r['method']))
  labels = [r['method'] for r in results]
  pretty = [PRETTY[m] for m in labels]
  metric_keys = ['ndcg@10', 'recall@10', 'roc_auc', 'auc_pr']
  metric_labels = ['NDCG@10', 'Recall@10', 'ROC-AUC', 'AUC-PR']
  hatches = [None, '///', '..', 'xx']
  colors = [PALETTE[m] for m in labels]

  # ---------- Figure: grouped metric bars ----------
  fig, ax = plt.subplots(figsize=(12, 5.4))
  x = np.arange(len(labels))
  n_m = len(metric_keys)
  w = 0.17
  bars_all = []
  for mi, (key, mlbl, hatch) in enumerate(zip(metric_keys, metric_labels, hatches)):
    vals = np.array([r[key] for r in results])
    offset = (mi - (n_m - 1) / 2) * w
    bars = ax.bar(
      x + offset, vals, w, color=colors, edgecolor='#0f172a', linewidth=0.7,
      label=mlbl, alpha=0.95 if mi % 2 == 0 else 0.55, hatch=hatch,
    )
    bars_all.append(bars)
    for bar, val in zip(bars, vals):
      ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006, f'{val:.3f}',
              ha='center', va='bottom', fontsize=7.5, fontweight='bold' if mi == 0 else 'normal')
  ax.set_xticks(x); ax.set_xticklabels(pretty)
  ax.set_ylabel('Score')
  ax.set_title('NFCorpus test — retrieval quality by method')
  ax.set_ylim(0, max(r[k] for r in results for k in metric_keys) * 1.22)
  from matplotlib.patches import Patch
  legend_handles = [
    Patch(facecolor='#cbd5e1', edgecolor='#0f172a', label='NDCG@10'),
    Patch(facecolor='#cbd5e1', edgecolor='#0f172a', hatch='///', alpha=0.55, label='Recall@10'),
    Patch(facecolor='#cbd5e1', edgecolor='#0f172a', hatch='..', label='ROC-AUC'),
    Patch(facecolor='#cbd5e1', edgecolor='#0f172a', hatch='xx', label='AUC-PR'),
  ]
  ax.legend(handles=legend_handles, loc='upper left', frameon=False, ncol=2)
  plt.tight_layout()
  plt.savefig(os.path.join(OUT, 'fig_metrics_bars.png'), dpi=180, bbox_inches='tight')
  plt.close()

  # ---------- Figure: lift over BM25 ----------
  bm25 = next(r for r in results if r['method'] == 'BM25')
  methods_no_bm25 = [m for m in labels if m != 'BM25']
  pretty_nb = [PRETTY[m] for m in methods_no_bm25]
  lift_keys = ['ndcg@10', 'recall@10', 'roc_auc', 'auc_pr']
  lift_labels = ['ΔNDCG@10', 'ΔRecall@10', 'ΔROC-AUC', 'ΔAUC-PR']
  lift_hatches = [None, '///', '..', 'xx']

  fig, ax = plt.subplots(figsize=(12, 5.4))
  x = np.arange(len(methods_no_bm25))
  n_m = len(lift_keys)
  w = 0.17
  for mi, (key, mlbl, hatch) in enumerate(zip(lift_keys, lift_labels, lift_hatches)):
    base = bm25[key]
    lifts = np.array([
      (next(r[key] for r in results if r['method'] == m) - base) / base * 100
      for m in methods_no_bm25
    ])
    offset = (mi - (n_m - 1) / 2) * w
    c2 = [PALETTE[m] for m in methods_no_bm25]
    ax.bar(x + offset, lifts, w, color=c2, edgecolor='#0f172a', linewidth=0.7,
           alpha=0.95 if mi % 2 == 0 else 0.55, hatch=hatch)
    for i, v in enumerate(lifts):
      off = 3 if v >= 0 else -10
      ax.text(i + offset, v + off, f'{v:+.0f}%', ha='center', fontsize=7.5,
              fontweight='bold' if mi == 0 else 'normal')
  ax.axhline(0, color='#0f172a', linewidth=1)
  ax.set_xticks(x); ax.set_xticklabels(pretty_nb)
  ax.set_ylabel('Relative change vs BM25  (%)')
  ax.set_title('Lift / drop relative to BM25 baseline')
  from matplotlib.patches import Patch
  legend_handles = [
    Patch(facecolor='#cbd5e1', edgecolor='#0f172a', label='ΔNDCG@10'),
    Patch(facecolor='#cbd5e1', edgecolor='#0f172a', hatch='///', alpha=0.55, label='ΔRecall@10'),
    Patch(facecolor='#cbd5e1', edgecolor='#0f172a', hatch='..', label='ΔROC-AUC'),
    Patch(facecolor='#cbd5e1', edgecolor='#0f172a', hatch='xx', label='ΔAUC-PR'),
  ]
  ax.legend(handles=legend_handles, loc='lower right', frameon=False, ncol=2)
  plt.tight_layout()
  plt.savefig(os.path.join(OUT, 'fig_lift.png'), dpi=180, bbox_inches='tight')
  plt.close()


# ============================================================
# Real plots from BGE base vs BGE+LoRA
# ============================================================
def _strip_special(text):
  for tok in SPECIAL_TOKS:
    text = text.replace(tok, ' ')
  return ' '.join(text.split()).strip()


@torch.inference_mode()
def _encode(enc, tok, texts, batch_size=32, max_len=512, q_prefix=''):
  vecs = []
  for i in range(0, len(texts), batch_size):
    chunk = [q_prefix + t for t in texts[i:i + batch_size]]
    b = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    b = {k: v.to(device) for k, v in b.items()}
    out = enc(**b).last_hidden_state[:, 0, :]
    out = F.normalize(out, p=2, dim=1)
    vecs.append(out.cpu())
  return torch.cat(vecs, 0)


def _load_bge(adapter=None):
  tok = AutoTokenizer.from_pretrained(BGE_NAME)
  tok.add_tokens(list(AddTokens().new_tokens.values()))
  enc = AutoModel.from_pretrained(BGE_NAME)
  enc.resize_token_embeddings(len(tok))
  enc = enc.to(device).eval()
  if adapter is not None:
    enc = PeftModel.from_pretrained(enc, adapter).to(device).eval()
  return enc, tok


def plots_from_model():
  # Sample test queries with at least one positive doc.
  ds = NFCorpusDataset(qrels_path=os.path.join(BASE, 'data/nfcorpus/qrels/'),
                       queries_path=os.path.join(BASE, 'data/nfcorpus/queries.jsonl'),
                       corpus_path=os.path.join(BASE, 'data/nfcorpus/corpus.jsonl'),
                       split='TEST', num_hard_neg=0, num_rand_neg=0, seed=42)
  rng = random.Random(0)
  query_ids = [q for q in ds.query_ids
               if any(s > 0 for s in ds.graded[q].values())]
  rng.shuffle(query_ids)
  query_ids = query_ids[:120]
  pos_ids = []
  for q in query_ids:
    rel = [c for c, s in ds.graded[q].items() if s > 0]
    pos_ids.append(rng.choice(rel))
  # random negatives — corpus docs not in any positive set for the sample
  all_pos = set(pos_ids)
  neg_pool = [c for c in ds.corpus_ids if c not in all_pos]
  rng.shuffle(neg_pool)
  neg_ids = neg_pool[:240]

  q_texts = [_strip_special(ds.queries[q]) for q in query_ids]
  p_texts = [_strip_special(ds.corpus[c]) for c in pos_ids]
  n_texts = [_strip_special(ds.corpus[c]) for c in neg_ids]

  # ---- encode with plain BGE
  print('[plot] encoding with plain BGE...')
  enc, tok = _load_bge(adapter=None)
  q_v_plain = _encode(enc, tok, q_texts, max_len=MAX_QUERY_LENGTH, q_prefix=BGE_QUERY_PREFIX)
  p_v_plain = _encode(enc, tok, p_texts, max_len=MAX_DOC_LENGTH)
  n_v_plain = _encode(enc, tok, n_texts, max_len=MAX_DOC_LENGTH)
  del enc; gc.collect()
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()

  # ---- encode with BGE + LoRA
  print('[plot] encoding with BGE + LoRA...')
  enc, tok = _load_bge(adapter=os.path.join(BASE, 'bge_lora'))
  q_v_lora = _encode(enc, tok, q_texts, max_len=MAX_QUERY_LENGTH, q_prefix=BGE_QUERY_PREFIX)
  p_v_lora = _encode(enc, tok, p_texts, max_len=MAX_DOC_LENGTH)
  n_v_lora = _encode(enc, tok, n_texts, max_len=MAX_DOC_LENGTH)
  del enc; gc.collect()
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()

  # ---- Similarity distributions (paired positive sims + per-query negative sims) ----
  def pos_neg_sims(qv, pv, nv):
    pos_sim = (qv * pv).sum(1).numpy()                     # (Q,)
    neg_sim = (qv @ nv.T).numpy().reshape(-1)              # (Q*Nneg,)
    return pos_sim, neg_sim

  pos_p, neg_p = pos_neg_sims(q_v_plain, p_v_plain, n_v_plain)
  pos_l, neg_l = pos_neg_sims(q_v_lora,  p_v_lora,  n_v_lora)

  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
  bins = np.linspace(min(neg_p.min(), neg_l.min(), pos_p.min(), pos_l.min()) - 0.02,
                     max(pos_p.max(), pos_l.max()) + 0.02, 50)
  for ax, (pos, neg, title) in zip(axes, [
    (pos_p, neg_p, f'Plain BGE  (zero-shot)\nΔμ = {pos_p.mean() - neg_p.mean():+.3f}'),
    (pos_l, neg_l, f'BGE + LoRA  (fine-tuned)\nΔμ = {pos_l.mean() - neg_l.mean():+.3f}'),
  ]):
    ax.hist(neg, bins=bins, color='#ef4444', alpha=0.55, label='negative (q, d⁻)',
            edgecolor='white', linewidth=0.4, density=True)
    ax.hist(pos, bins=bins, color='#10b981', alpha=0.75, label='positive (q, d⁺)',
            edgecolor='white', linewidth=0.4, density=True)
    ax.axvline(pos.mean(), color='#065f46', linestyle='--', linewidth=1.2)
    ax.axvline(neg.mean(), color='#7f1d1d', linestyle='--', linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel('cosine similarity')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
  axes[0].set_ylabel('density')
  fig.suptitle(
    'Cosine similarity of (query, document) pairs — what fine-tuning actually changes',
    fontsize=13, fontweight='bold', y=1.02,
  )
  plt.tight_layout()
  plt.savefig(os.path.join(OUT, 'fig_similarity_dist.png'), dpi=180, bbox_inches='tight')
  plt.close()

  # ---- t-SNE: real embeddings, before vs after ----
  print('[plot] running t-SNE...')
  # Subsample for clarity
  K = 60
  idx = np.arange(len(query_ids))[:K]
  jdx = np.arange(len(neg_ids))[:120]

  def tsne_panel(qv, pv, nv):
    pts = np.vstack([qv[idx].numpy(), pv[idx].numpy(), nv[jdx].numpy()])
    emb = TSNE(n_components=2, perplexity=20, init='pca', learning_rate='auto',
               random_state=0, metric='cosine').fit_transform(pts)
    return emb[:K], emb[K:2*K], emb[2*K:]

  q2_p, p2_p, n2_p = tsne_panel(q_v_plain, p_v_plain, n_v_plain)
  q2_l, p2_l, n2_l = tsne_panel(q_v_lora,  p_v_lora,  n_v_lora)

  # Compute *cosine* similarity of paired q→d⁺ in the original (high-dim) space —
  # this is the honest "before/after" signal. t-SNE distances are not comparable.
  cos_p = (q_v_plain[idx] * p_v_plain[idx]).sum(1).mean().item()
  cos_l = (q_v_lora[idx]  * p_v_lora[idx]).sum(1).mean().item()

  fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharex=False, sharey=False)
  for ax, (q2, p2, n2), title in [
    (axes[0], (q2_p, p2_p, n2_p), f'Plain BGE   (zero-shot)\nmean cos(q, d⁺) = {cos_p:.3f}'),
    (axes[1], (q2_l, p2_l, n2_l), f'BGE + LoRA   (fine-tuned)\nmean cos(q, d⁺) = {cos_l:.3f}'),
  ]:
    for i in range(K):
      ax.plot([q2[i, 0], p2[i, 0]], [q2[i, 1], p2[i, 1]],
              color='#94a3b8', alpha=0.45, linewidth=0.7, zorder=1)
    ax.scatter(n2[:, 0], n2[:, 1], s=34, color='#ef4444', alpha=0.55,
               label='negative docs', edgecolor='white', linewidth=0.4, zorder=2)
    ax.scatter(p2[:, 0], p2[:, 1], s=46, color='#10b981',
               label='positive docs', edgecolor='white', linewidth=0.5, zorder=3)
    ax.scatter(q2[:, 0], q2[:, 1], s=46, color='#0ea5e9', marker='D',
               label='queries', edgecolor='white', linewidth=0.5, zorder=4)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
      spine.set_edgecolor('#cbd5e1')
  axes[0].legend(loc='upper right', frameon=True, fontsize=10)
  fig.suptitle(
    't-SNE of real query / positive / negative embeddings',
    fontsize=14, fontweight='bold', y=1.01,
  )
  plt.tight_layout()
  plt.savefig(os.path.join(OUT, 'fig_embedding_tsne.png'), dpi=180, bbox_inches='tight')
  plt.close()


if __name__ == '__main__':
  plots_from_eval_json()
  plots_from_model()
  # Clean up old/stale figures so README never references something that no longer exists.
  for stale in ('fig_scatter.png', 'fig_ablation.png', 'fig_embedding_geometry.png'):
    p = os.path.join(OUT, stale)
    if os.path.exists(p):
      os.remove(p)
  print('Wrote:')
  for n in sorted(os.listdir(OUT)):
    print(' ', os.path.join('plots', n))
