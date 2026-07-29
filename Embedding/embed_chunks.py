from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config


def load_chunks(path: str = config.CHUNKS_PATH) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"[embed_chunks] loaded {len(chunks)} chunks from {path}")
    return chunks


def embed_passages(chunks: list[dict], model: SentenceTransformer | None = None) -> np.ndarray:
    """Embed chunk text as passages -- NO instruction prefix here.
    BGE is asymmetric: the prefix belongs on the query side only
    (see query_embed.py). Prefixing passages too will hurt recall, not help it.
    """
    model = model or SentenceTransformer(config.EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,  # required -- Qdrant collection uses COSINE distance
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    assert embeddings.shape[1] == config.EMBEDDING_DIM, (
        f"Model produced {embeddings.shape[1]}-dim vectors, "
        f"but config/collection expect {config.EMBEDDING_DIM}. "
        f"Did EMBEDDING_MODEL change without updating EMBEDDING_DIM and the "
        f"Qdrant collection's vector size?"
    )
    return embeddings


def embed_and_save(
    chunks_path: str = config.CHUNKS_PATH,
    out_path: str = config.EMBEDDINGS_CACHE_PATH,
) -> tuple[list[dict], np.ndarray]:
    """Embed once, cache to disk. Re-embedding 3,536 chunks is cheap on CPU
    but there's no reason to redo it every time you tweak the DB-loading step.
    If a cache already exists and matches the chunk count, reuse it instead
    of re-embedding -- makes iterating on db.py/run_phase3.py fast.
    """
    chunks = load_chunks(chunks_path)

    cache = Path(out_path)
    if cache.exists():
        cached = np.load(cache)
        if cached.shape[0] == len(chunks) and cached.shape[1] == config.EMBEDDING_DIM:
            print(f"[embed_chunks] reusing cached embeddings at {out_path} ({cached.shape})")
            return chunks, cached
        print(
            f"[embed_chunks] cache at {out_path} has shape {cached.shape}, "
            f"expected ({len(chunks)}, {config.EMBEDDING_DIM}) -- re-embedding"
        )

    embeddings = embed_passages(chunks)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings)
    print(f"[embed_chunks] wrote {embeddings.shape} embeddings -> {out_path}")

    return chunks, embeddings


if __name__ == "__main__":
    embed_and_save()
