from abc import abstractmethod
import tensorflow as tf
import numpy as np
import os
import requests
import diskcache

from huggingface_hub import InferenceClient

# Optionally load environment variables (e.g., API keys) from a .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is an optional dependency; skip if not installed
    pass

class APIEncoder(tf.keras.layers.Layer):
    """Base class for API encoders. Embeddings are cached on disk so the same
    inputs are not re-embedded across runs or epochs.
    """
    def __init__(self, model_name, embedding_dim, cache_dir=None, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.cache_dir = cache_dir or f".embedding_cache/{model_name.replace('/', '_')}"
        self._cache = diskcache.Cache(self.cache_dir)

    @abstractmethod
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Call the API and return embeddings

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        raise NotImplementedError

    def call(self, inputs):
        """
        Process string tensor through the API

        Args:
            inputs: tf.Tensor of dtype tf.string, shape (batch_size,)

        Returns:
            tf.Tensor of shape (batch_size, embedding_dim)
        """
        # Convert tf.string Tensor to Python string

        embeddings = tf.py_function(
            func = self._embed_texts_wrapper,
            inp=[inputs],
            Tout=tf.float32
        )
        embeddings.set_shape((None, self.embedding_dim))
        return embeddings

    def _embed_texts_wrapper(self, inputs_tensor):
        """
        Wrapper to convert tensor to list, check disk cache, and call API if needed.
        """
        texts = [t.decode('utf-8') for t in inputs_tensor.numpy()]
        key = tuple(texts)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        embeddings = self._embed_texts(texts).astype(np.float32)
        self._cache[key] = embeddings
        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'cache_dir': self.cache_dir,
        })
        return config


class HFInferenceEncoder(APIEncoder):
    """
    Encoder using HuggingFace Inference (InferenceClient / Inference Providers).

    Uses the huggingface_hub InferenceClient, which routes to Inference Providers
    (the legacy serverless api-inference.huggingface.co endpoint returns 410 Gone).
    Uses HF_API_TOKEN or HF_TOKEN from the environment.
    """
    EMBEDDING_DIMS = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/paraphrase-MiniLM-L3-v2": 384,
        "sentence-transformers/all-MiniLM-L12-v2": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "intfloat/e5-small-v2": 384,
        "thenlper/gte-small": 384,
    }

    def __init__(self, model_name: str, **kwargs):
        embedding_dim = self.EMBEDDING_DIMS.get(model_name)
        if embedding_dim is None:
            raise ValueError(
                f"Unknown HuggingFace model: {model_name}. "
                f"Known models: {list(self.EMBEDDING_DIMS.keys())}"
            )
        super().__init__(model_name=model_name, embedding_dim=embedding_dim, **kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
            self._client = InferenceClient(token=token)
        return self._client

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        # feature_extraction accepts str or list of str; pass list for one batch
        result = self.client.feature_extraction(
            texts,
            model=self.model_name,
        )
        arr = np.array(result, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

class OpenAIEncoder(APIEncoder):
    """Encoder using OpenAI's API"""
    EMBEDDING_DIMS = {
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'text-embedding-ada-002': 1536,
    }

    def __init__(self, model_name: str = 'text-embedding-3-small', **kwargs):
        embedding_dim = self.EMBEDDING_DIMS.get(model_name)
        if embedding_dim is None:
            raise ValueError(f"Unknown OpenAI model: {model_name}. \
                Known models: {list(self.EMBEDDING_DIMS.keys())}")
        super().__init__(model_name=model_name, embedding_dim=embedding_dim, **kwargs)
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(f"OpenAI library required. Install with: pip install openai.")


            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for OpenAIEncoder."
                )

            self._client = OpenAI(api_key=api_key) # uses OPENAI_API_KEY environment variable
        return self._client
    
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

class CohereEncoder(APIEncoder):
    """Encoder using Cohere's Embed API. Uses COHERE_API_KEY environment variable."""
    EMBEDDING_DIMS = {
        'embed-english-v3.0': 1024,
        'embed-english-light-v3.0': 384,
        'embed-multilingual-v3.0': 1024,
        'embed-multilingual-light-v3.0': 384,
        'embed-english-v2.0': 4096,
        'embed-english-light-v2.0': 1024,
        'embed-multilingual-v2.0': 768,
        'embed-v4.0': 1024,
    }

    def __init__(self, model_name: str = 'embed-english-v3.0', **kwargs):
        embedding_dim = self.EMBEDDING_DIMS.get(model_name)
        if embedding_dim is None:
            raise ValueError(
                f"Unknown Cohere model: {model_name}. "
                f"Known models: {list(self.EMBEDDING_DIMS.keys())}"
            )
        super().__init__(model_name=model_name, embedding_dim=embedding_dim, **kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from cohere import ClientV2
            except ImportError:
                raise ImportError(
                    "Cohere library required. Install with: pip install cohere."
                )
            api_key = os.environ.get("COHERE_API_KEY")
            if not api_key:
                raise ValueError(
                    "COHERE_API_KEY environment variable is required for CohereEncoder."
                )
            self._client = ClientV2(api_key=api_key)
        return self._client

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        kwargs = {
            "texts": texts,
            "model": self.model_name,
            "input_type": "search_document",
            "embedding_types": ["float"],
        }
        # output_dimension is only supported for embed-v4.0
        if self.model_name == "embed-v4.0":
            kwargs["output_dimension"] = self.embedding_dim
        res = self.client.embed(**kwargs)
        return np.array(res.embeddings.float, dtype=np.float32)


class VoyageEncoder(APIEncoder):
    """Encoder using Voyage AI's API. Uses VOYAGE_API_KEY environment variable."""
    EMBEDDING_DIMS = {
        'voyage-4-large': 1024,
        'voyage-4': 1024,
        'voyage-4-lite': 1024,
        'voyage-code-3': 1024,
        'voyage-context-3': 1024,
        'voyage-finance-2': 1024,
        'voyage-law-2': 1024,
        'voyage-code-2': 1536,
        'voyage-3': 1024,
        'voyage-large-2': 1024,
        'voyage-2': 1024,
    }

    def __init__(self, model_name: str = 'voyage-3', **kwargs):
        embedding_dim = self.EMBEDDING_DIMS.get(model_name)
        if embedding_dim is None:
            raise ValueError(
                f"Unknown Voyage model: {model_name}. "
                f"Known models: {list(self.EMBEDDING_DIMS.keys())}"
            )
        super().__init__(model_name=model_name, embedding_dim=embedding_dim, **kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import voyageai
            except ImportError:
                raise ImportError(
                    "Voyage AI library required. Install with: pip install voyageai."
                )
            api_key = os.environ.get("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError(
                    "VOYAGE_API_KEY environment variable is required for VoyageEncoder."
                )
            self._client = voyageai.Client(api_key=api_key)
        return self._client

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        result = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type="document",
        )
        return np.array(result.embeddings, dtype=np.float32)