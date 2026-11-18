"""
OpenAI text embeddings for the replication script, embedded once and stored.

An API embedding model is frozen: the same text always gives the same vector.
So instead of calling the API inside the training loop (one request per batch,
paid again every epoch because shuffled batches never repeat), every unique
text is embedded once and stored; training then reads the stored vectors as
pre-computed inputs.

Store layout, under ``<cache_root>/<model>/``:

    shard_00000.npy        float32 [n, dim], raw API vectors (unit length)
    shard_00000.keys.txt   one sha1(text) per row, written last

A shard counts only once its keys file exists, so a run that dies mid-write
loses at most one shard. Shards are never rewritten: a later run embeds only
the texts whose key is in no shard, so it resumes, and a changed dataset costs
only its new texts.

Empty texts are never sent (the API rejects them); they get a zero vector.
Texts longer than the model's 8,191-token limit are cut to it.

The key is read from OPENAI_API_KEY. Nothing here imports TensorFlow.
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# model: vector length
MODELS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
# US dollars per million tokens: OpenAI's list prices as known in mid-2026.
# Used only for the printed estimate; the bill is whatever OpenAI charges.
PRICE_PER_MILLION: Dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

MAX_TOKENS_PER_TEXT = 8191       # model limit per input
MAX_TEXTS_PER_REQUEST = 2048     # API limit per request
MAX_TOKENS_PER_REQUEST = 200_000  # API limit is 300,000; leave headroom
SHARD_TEXTS = 10_000             # texts per stored shard
SPLITS = ("train", "val", "test")


def read_texts(data_dir: str, split: str, *, prefix: str = "no_prop_overlap_",
               text_column: str = "text") -> List[str]:
    """The split's texts, row for row, exactly as ``load_splits`` reads them."""
    df = pd.read_csv(f"{data_dir}/{prefix}{split}.csv", usecols=[text_column])
    return ["" if pd.isna(x) else str(x) for x in df[text_column]]


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def is_empty(text: str) -> bool:
    return not text.strip()


def store_dir(cache_root: str, model: str) -> str:
    if model not in MODELS:
        raise ValueError(f"Unknown OpenAI embedding model {model!r}; known: {list(MODELS)}")
    return os.path.join(cache_root, model)


def _shards(directory: str) -> List[Tuple[str, List[str]]]:
    """[(npy path, keys)] for every finished shard, in order."""
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".keys.txt"):
            npy = os.path.join(directory, name[: -len(".keys.txt")] + ".npy")
            if os.path.exists(npy):
                with open(os.path.join(directory, name)) as f:
                    out.append((npy, f.read().split()))
    return out


def stored_keys(cache_root: str, model: str) -> set:
    return {k for _, keys in _shards(store_dir(cache_root, model)) for k in keys}


def _encoding():
    import tiktoken

    # The tokenizer of the text-embedding-3 models.
    return tiktoken.get_encoding("cl100k_base")


def tokenize(texts: Sequence[str]) -> List[List[int]]:
    """Token ids per text, cut to the model limit. Special-token spellings in
    the text (e.g. "<|endoftext|>") are encoded as ordinary characters."""
    enc = _encoding()
    return [ids[:MAX_TOKENS_PER_TEXT] for ids in enc.encode_ordinary_batch(list(texts))]


def plan(texts: Iterable[str], model: str, cache_root: str) -> dict:
    """What embedding ``texts`` would take: unique texts, how many are already
    stored, and the tokens and list-price cost of the rest. No API call."""
    unique = sorted({t for t in texts if not is_empty(t)})
    have = stored_keys(cache_root, model)
    missing = [t for t in unique if text_key(t) not in have]
    lengths = [len(ids) for ids in tokenize(missing)]
    tokens = int(sum(lengths))
    return {
        "model": model,
        "unique_texts": len(unique),
        "already_stored": len(unique) - len(missing),
        "to_embed": len(missing),
        "tokens": tokens,
        "cut_to_limit": int(sum(n >= MAX_TOKENS_PER_TEXT for n in lengths)),
        "est_cost_usd": tokens / 1e6 * PRICE_PER_MILLION[model],
        "missing": missing,
    }


def _requests(token_lists: Sequence[List[int]]) -> List[Tuple[int, int]]:
    """[start, stop) ranges, each within the API's per-request limits."""
    out, start, tokens = [], 0, 0
    for i, ids in enumerate(token_lists):
        if i > start and (i - start >= MAX_TEXTS_PER_REQUEST
                          or tokens + len(ids) > MAX_TOKENS_PER_REQUEST):
            out.append((start, i))
            start, tokens = i, 0
        tokens += len(ids)
    if start < len(token_lists):
        out.append((start, len(token_lists)))
    return out


def embed_missing(texts: Iterable[str], model: str, cache_root: str, *,
                  limit: int = 0, workers: int = 4,
                  after_shard: Optional[Callable[[], None]] = None) -> dict:
    """Embed every non-empty unique text not yet stored (the first ``limit`` of
    them, in sorted order, when ``limit`` > 0) and store it. ``after_shard`` is
    called after each shard is written (e.g. to commit a Modal volume).
    Returns the plan plus the tokens OpenAI reported billing."""
    from openai import OpenAI

    info = plan(texts, model, cache_root)
    missing = info.pop("missing")
    if limit > 0:
        missing = missing[:limit]
    info["embedding_now"] = len(missing)
    info["billed_tokens"] = 0
    if not missing:
        return info

    directory = store_dir(cache_root, model)
    os.makedirs(directory, exist_ok=True)
    dim = MODELS[model]
    # max_retries: the client backs off and retries on rate limits (429) and
    # server errors, honouring the wait the API asks for.
    client = OpenAI(max_retries=10, timeout=120)

    def call(token_lists: List[List[int]]) -> Tuple[np.ndarray, int]:
        r = client.embeddings.create(model=model, input=token_lists, encoding_format="float")
        rows = sorted(r.data, key=lambda d: d.index)
        vecs = np.asarray([d.embedding for d in rows], dtype=np.float32)
        if vecs.shape != (len(token_lists), dim) or not np.isfinite(vecs).all():
            raise RuntimeError(f"Unexpected embedding response: shape {vecs.shape}, "
                               f"wanted {(len(token_lists), dim)}")
        return vecs, int(r.usage.total_tokens)

    next_shard = len({name.split(".")[0] for name in os.listdir(directory)
                      if name.startswith("shard_")})
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for lo in range(0, len(missing), SHARD_TEXTS):
            chunk = missing[lo: lo + SHARD_TEXTS]
            token_lists = tokenize(chunk)
            ranges = _requests(token_lists)
            parts = list(pool.map(lambda r: call(token_lists[r[0]: r[1]]), ranges))
            vecs = np.concatenate([p[0] for p in parts])
            info["billed_tokens"] += sum(p[1] for p in parts)

            stem = os.path.join(directory, f"shard_{next_shard:05d}")
            while os.path.exists(stem + ".npy") or os.path.exists(stem + ".keys.txt"):
                next_shard += 1
                stem = os.path.join(directory, f"shard_{next_shard:05d}")
            np.save(stem + ".npy", vecs)
            with open(stem + ".keys.txt", "w") as f:  # last: marks the shard finished
                f.write("\n".join(text_key(t) for t in chunk) + "\n")
            next_shard += 1
            if after_shard is not None:
                after_shard()
            done += len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  embedded {done:,} / {len(missing):,} texts "
                  f"({info['billed_tokens']:,} tokens billed, {rate:,.0f} texts/s)", flush=True)
    info["billed_cost_usd"] = info["billed_tokens"] / 1e6 * PRICE_PER_MILLION[model]
    return info


def load_embeddings(texts: Sequence[str], model: str, cache_root: str) -> np.ndarray:
    """float32 [len(texts), dim]: the stored vector of each text, row for row
    (zeros for empty texts). Raises if any text has not been embedded yet."""
    dim = MODELS[model]
    keys = [None if is_empty(t) else text_key(t) for t in texts]
    rows_of: Dict[str, List[int]] = {}
    for i, k in enumerate(keys):
        if k is not None:
            rows_of.setdefault(k, []).append(i)
    out = np.zeros((len(texts), dim), dtype=np.float32)
    found = set()
    for npy, shard_keys in _shards(store_dir(cache_root, model)):
        hits = [(j, k) for j, k in enumerate(shard_keys) if k in rows_of and k not in found]
        if not hits:
            continue
        vecs = np.load(npy)
        for j, k in hits:
            out[rows_of[k]] = vecs[j]
            found.add(k)
    absent = len(rows_of) - len(found)
    if absent:
        raise FileNotFoundError(
            f"{absent:,} of {len(rows_of):,} unique texts have no stored {model} embedding "
            f"under {cache_root}. Embed them first:\n"
            f"  modal run modal_embed.py::embed --model {model} --data-dir <data dir>")
    return out
