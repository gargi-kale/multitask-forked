from abc import abstractmethod
import tensorflow as tf
import numpy as np
import os
import requests
import diskcache

from huggingface_hub import InferenceClient

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
    Encoder using HuggingFace's API.

    Supported models: https://huggingface.co/inference/models
    """
    def __init__(self, model_name: str, **kwargs):
        embedding_dim = self.EMBEDDING_DIMS.get(model_name)
        if embedding_dim is None:
            raise ValueError(f"Unknown HuggingFace model: {model_name}. \
                Known models: {list(self.EMBEDDING_DIMS.keys())}")
        super().__init__(model_name=model_name, embedding_dim=embedding_dim, **kwargs)
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        headers = {}
        token = os.environ.get("HF_API_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        return np.array(response.json())

    @staticmethod
    def _check_available(model_name: str) -> bool:
        import requests
        response = requests.get(
            f"https://api-inference.huggingface.co/status/{model_name}"
        )
        return response.status_code == 200 and response.json().get("loaded", False)

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

            self._client = OpenAI() # uses OPENAI_API_KEY environment variable
            # or maybe require user to set this value
        return self._client
    
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

class CohereEncoder(APIEncoder):
    # TODO: Implement this.
    pass

class VoyageEncoder(APIEncoder):
    # TODO: Implement this.
    pass