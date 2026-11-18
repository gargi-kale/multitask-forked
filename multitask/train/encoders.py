"""Encoder presets: small BERT, ELECTRA (full and cut), OpenAI embeddings.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from multitask.config import EncoderConfig, EncoderIntegration, EncoderInputType


# ---------------------------------------------------------------------------
# Encoder configs
# ---------------------------------------------------------------------------

ecfg_distilbert_uncased = EncoderConfig(
    encoder_identifier="distilbert-base-uncased",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE,
    trainable_encoder=True,
)

ecfg_tfhub_bert = EncoderConfig(
    encoder_identifier="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
    encoder_input_type=EncoderInputType.TFHUB_TOKENS,
    encoder_integration=EncoderIntegration.TFHUB,
    trainable_encoder=True,
)

ecfg_small_bert = EncoderConfig(
    encoder_identifier="google/bert_uncased_L-2_H-128_A-2",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE,
    trainable_encoder=True,
    from_pt=True
)

# ELECTRA-small's discriminator: 12 layers, hidden 256, ~13.5M parameters. It
# uses BERT's uncased vocabulary (the vocab files are identical), so the
# tokenizer in load_splits and the special ids in mask_tokens apply unchanged.
# It has no pooler, so the model reads the [CLS] position of the last layer,
# and no masked-word head (see NoMLMHead).
ecfg_electra_small = EncoderConfig(
    encoder_identifier="google/electra-small-discriminator",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE,
    trainable_encoder=True,
)


def shortened_electra_small(num_layers: int) -> EncoderConfig:
    """ELECTRA-small cut down to its embeddings and first `num_layers` of 12
    transformer layers, which keep their pretrained weights; the layers above
    are dropped. There is no official ELECTRA checkpoint smaller than
    ELECTRA-small. Parameters: 2 layers ~5.6M (the small BERT has ~4.4M),
    4 layers ~7.2M, 6 layers ~8.7M, all 12 ~13.5M."""
    return EncoderConfig(
        encoder_identifier="google/electra-small-discriminator",
        encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
        encoder_integration=EncoderIntegration.HUGGINGFACE,
        trainable_encoder=True,
        encoder_kwargs={"num_hidden_layers": num_layers},
    )


# OpenAI embedding models (paper_replication only). The model is frozen, so its
# vectors are embedded once, stored (openai_embeddings.py, filled by
# `modal run modal_embed.py::embed`) and fed to the model as pre-computed
# inputs: there is no encoder in the model, and nothing below the shared layers
# trains. --encoder name: OpenAI model.
OPENAI_EMBEDDING_MODELS = {
    "openai-small": "text-embedding-3-small",  # 1,536 numbers per text
    "openai-large": "text-embedding-3-large",  # 3,072 numbers per text
}
OPENAI_EMBEDDING_DIMS = {"openai-small": 1536, "openai-large": 3072}


def precomputed_embeddings(embedding_dim: int) -> EncoderConfig:
    return EncoderConfig(
        encoder_input_type=EncoderInputType.PRECOMPUTED,
        embedding_dim=embedding_dim,
        trainable_encoder=False,
    )


def openai_embedding_source(encoder: str, cache_root: str):
    """texts -> float32 [n, dim] from the stored vectors of ``encoder``.

    OpenAI vectors have length 1, so each of their ``dim`` numbers is tiny
    (about 1 / sqrt(dim) = 0.026 for openai-small), far below the roughly
    unit-sized BERT outputs the dense layers' initialisation expects. The
    vectors are multiplied by sqrt(dim) so their numbers are about unit-sized;
    this rescales every text alike and changes no direction or similarity."""
    from multitask.train.openai_embeddings import load_embeddings

    model = OPENAI_EMBEDDING_MODELS[encoder]
    scale = np.float32(np.sqrt(OPENAI_EMBEDDING_DIMS[encoder]))

    def source(texts: Sequence[str]) -> np.ndarray:
        return load_embeddings(texts, model, cache_root) * scale

    return source


ENCODER_CONFIGS = {
    "distilbert": ecfg_distilbert_uncased,
    "tfhub-bert": ecfg_tfhub_bert,
    "small-bert": ecfg_small_bert,
    "electra-small": ecfg_electra_small,
    "electra-small-L2": shortened_electra_small(2),
    "electra-small-L4": shortened_electra_small(4),
    "electra-small-L6": shortened_electra_small(6),
    **{name: precomputed_embeddings(dim) for name, dim in OPENAI_EMBEDDING_DIMS.items()},
}
