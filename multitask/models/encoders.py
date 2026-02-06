from abc import abstractmethod
import tensorflow as tf
import numpy as np

class APIEncoder(tf.keras.layers.Layer):
    """Base class for API encoders"""
    def __init__(self, model_name: str, embedding_dim: int, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.model_name = model_name
        self.embedding_dim = embedding_dim

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
        Wrapper to convert tensor to list and call API.
        """
        texts = [t.decode('utf-8') for t in inputs_tensor.numpy()]
        embeddings = self._embed_texts(texts)
        return embeddings.astype(np.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        })
        return config

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
        response = self._client.embeddings.create(
            model = self.model_name,
            input = texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

class CohereEncoder(APIEncoder):
    # TODO: Implement this.
    pass

class VoyageEncoder(APIEncoder):
    # TODO: Implement this.
    pass