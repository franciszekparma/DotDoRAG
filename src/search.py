import os
import sys
import subprocess
from urllib.parse import quote

import torch
import torch.nn.functional as F
from peft import PeftModel

import config
from utils import AddTokens
from model import model, tokenizer, device


def osc8_link(text, url):
    """Wrap `text` as a clickable hyperlink in modern terminals (iTerm2, Terminal.app,
    VSCode, kitty). Falls back to plain text in terminals that ignore the sequence."""
    return f'\033]8;;{url}\033\\{text}\033]8;;\033\\'


def path_to_file_url(path):
    abs_path = os.path.abspath(path)
    return 'file://' + quote(abs_path)


def open_path(path):
    if not os.path.exists(path):
        print(f'  ! file not found: {path}')
        return
    if sys.platform == 'darwin':
        subprocess.run(['open', path], check=False)
    elif sys.platform.startswith('linux'):
        subprocess.run(['xdg-open', path], check=False)
    elif sys.platform == 'win32':
        os.startfile(path)
    else:
        print(f'  open manually: {path}')


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else config.CHECKPOINT_PATH
    encoded_path = sys.argv[2] if len(sys.argv) > 2 else config.ENCODED_PATH
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else config.DEFAULT_THRESHOLD

    ta = AddTokens()

    cached = torch.load(encoded_path, map_location=device)
    doc_vecs = cached['doc_vecs'].to(device)
    pdf_paths = cached['pdf_paths']

    model.enc = PeftModel.from_pretrained(model.enc, checkpoint_path)
    model.to(device)
    model.eval()

    while True:
        try:
            query = input('\nquery> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            continue
        if query.lower() in ('exit', 'quit'):
            return

        with torch.inference_mode():
            que = tokenizer(
                [ta.add_query_tokens(query)],
                padding=True,
                truncation=True,
                max_length=config.MAX_QUERY_LENGTH,
                return_tensors='pt',
            )
            que = {k: v.to(device) for k, v in que.items()}
            que_vec = model.enc(**que).last_hidden_state[..., 0, :]
            que_vec = F.normalize(que_vec, p=2, dim=1)

            sims = (que_vec @ doc_vecs.T).squeeze(0)
            scores, idxs = torch.sort(sims, descending=True)

        hits = [
            (s.item(), pdf_paths[i.item()])
            for s, i in zip(scores, idxs)
            if s.item() >= threshold
        ]

        if not hits:
            print('No matching papers found. Try rephrasing or lowering the threshold.')
            continue

        print(f'\nFound {len(hits)} matching paper(s) - Cmd/Ctrl-click a title to open, or enter its number:\n')
        for i, (score, path) in enumerate(hits, start=1):
            title = os.path.splitext(os.path.basename(path))[0]
            url = path_to_file_url(path)
            link = osc8_link(title, url)
            print(f'  [{i:>2}] {score:.3f}  {link}')

        try:
            choice = input('\nopen> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not choice:
            continue
        if not choice.isdigit():
            print('  (skipped — enter a number next time)')
            continue
        n = int(choice)
        if 1 <= n <= len(hits):
            open_path(hits[n - 1][1])
        else:
            print(f'  out of range (1–{len(hits)})')


if __name__ == '__main__':
    main()
