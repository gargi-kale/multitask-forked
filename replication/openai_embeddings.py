"""Moved to multitask/train/openai_embeddings.py; this alias keeps
`import openai_embeddings` (modal_embed.py, old scripts) working."""
import sys

from multitask.train import openai_embeddings as _module

sys.modules[__name__] = _module
