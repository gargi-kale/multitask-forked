"""
Modal harness for the `multitask` library — OpenAI embeddings, embedded once.

    modal run modal_embed.py::embed --estimate-only     # count tokens, no API call
    modal run modal_embed.py::embed --limit 200         # a small paid test
    modal run modal_embed.py::embed                     # everything not yet stored

Embeds every unique text of the train / val / test CSVs with an OpenAI
embedding model and stores the vectors on the `multitask-cache` volume under
EMBEDDING_CACHE. `modal_replicate.py::replicate --encoder openai-small` then
trains on the stored vectors and needs no key and no network access to OpenAI.
Re-running embeds only what is missing, so it resumes after a failure.

This is the only function that sees the OpenAI key, which is why it lives in
its own module: `modal run` validates every function registered in the module
it loads, so the secret below would otherwise be required for every
entrypoint in `modal_replicate.py`.

Setup (once): store the key as a Modal secret named `openai-api`,

    modal secret create openai-api OPENAI_API_KEY=sk-...
    # or, from a local .env holding OPENAI_API_KEY:
    modal secret create openai-api --from-dotenv .env

CPU only: nothing here trains.
"""

import sys

import modal

from modal_common import EMBEDDING_CACHE, REPO, VOLUMES, cache_vol, embed_image

app = modal.App("multitask-embed")


@app.function(
    image=embed_image,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("openai-api")],
    timeout=60 * 60 * 6,
    cpu=4.0,
    memory=8192,
)
def embed(
    model: str = "text-embedding-3-small",
    data_dir: str = "/data/paper",
    estimate_only: bool = False,
    limit: int = 0,
    workers: int = 4,
) -> dict:
    """Embed the unique texts of `data_dir`'s three CSVs that are not stored yet.

    `estimate_only` prints the token count and list-price cost and stops
    before any API call. `limit` > 0 embeds only that many texts (a paid
    plumbing test). `workers` is the number of parallel requests; the client
    waits and retries on rate limits, so a low account limit slows the run
    down instead of failing it.
    """
    sys.path.insert(0, f"{REPO}/replication")
    import openai_embeddings as oe

    texts = [t for split in oe.SPLITS for t in oe.read_texts(data_dir, split)]
    info = oe.plan(texts, model, EMBEDDING_CACHE)
    info.pop("missing")
    print(f"{model} on {data_dir}: {len(texts):,} rows, {info['unique_texts']:,} unique "
          f"non-empty texts, {info['already_stored']:,} already stored.", flush=True)
    print(f"To embed: {info['to_embed']:,} texts, {info['tokens']:,} tokens "
          f"({info['cut_to_limit']:,} cut to the {oe.MAX_TOKENS_PER_TEXT:,}-token limit), "
          f"about ${info['est_cost_usd']:.2f} at list price.", flush=True)
    if estimate_only:
        return info

    info = oe.embed_missing(texts, model, EMBEDDING_CACHE, limit=limit, workers=workers,
                            after_shard=cache_vol.commit)
    cache_vol.commit()
    if info["embedding_now"]:
        print(f"Embedded {info['embedding_now']:,} texts; OpenAI reported "
              f"{info['billed_tokens']:,} tokens, about ${info['billed_cost_usd']:.4f} "
              "at list price.", flush=True)
    left = info["to_embed"] - info["embedding_now"]
    print("Every text is stored." if not left else f"{left:,} texts still to embed.", flush=True)
    return info
