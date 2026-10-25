"""
Build the pretrained word-embedding table used by the 'embedding' feature group.
Run from backend/:  python ml/build_embeddings.py [path/to/glove-wiki-gigaword-100.gz]

Source: GloVe 6B 100-d (Wikipedia + Gigaword, Stanford NLP), via the gensim-data GitHub release.
Keeps the 100,000 most frequent words as float16 -> ml/embeddings/glove100_100k.npz (~20 MB).
Why GloVe and not MiniLM: sentence-transformers weights are hosted on Hugging Face, which was not reachable
from the development environment. Swap in MiniLM later by replacing Embedder.encode (features.py).
"""
import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"
OUT = Path(__file__).resolve().parent / "embeddings" / "glove100_100k.npz"
TOP = 100_000

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("glove-wiki-gigaword-100.gz")
if not src.exists():
    print("Downloading", URL)
    urllib.request.urlretrieve(URL, src)
words, vecs = [], []
with gzip.open(src, "rt", encoding="utf8") as f:
    next(f)  # header: "400000 100"
    for i, line in enumerate(f):
        if i >= TOP:
            break
        parts = line.rstrip().split(" ")
        words.append(parts[0])
        vecs.append(np.asarray(parts[1:], dtype=np.float32))
OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(OUT, words=np.array(words), vectors=np.stack(vecs).astype(np.float16))
print(f"Saved {len(words)} x {len(vecs[0])} to {OUT}")
