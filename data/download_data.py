import os
import urllib.request
import zipfile

DATA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NFCORPUS_DIR = os.path.join(DATA_DIR, "nfcorpus")
ZIP_PATH = os.path.join(DATA_DIR, "nfcorpus.zip")


def download():
  if os.path.exists(os.path.join(NFCORPUS_DIR, "corpus.jsonl")):
    print("NFCorpus already present, skipping download.")
    return

  os.makedirs(DATA_DIR, exist_ok=True)

  print(f"Downloading NFCorpus...")
  print(f"Source: {DATA_URL}")
  print("Note: NFCorpus is free for academic/non-commercial use only.")
  print("      See https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/ for terms.\n")

  urllib.request.urlretrieve(DATA_URL, ZIP_PATH, reporthook=_progress)
  print()

  print("Extracting...")
  with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(DATA_DIR)
  os.remove(ZIP_PATH)

  print(f"Done. Dataset saved to {NFCORPUS_DIR}")


def _progress(count, block_size, total_size):
  if total_size > 0:
    pct = min(count * block_size / total_size * 100, 100)
    print(f"\r  {pct:.1f}%", end="", flush=True)


if __name__ == "__main__":
  download()
