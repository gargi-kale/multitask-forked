from abc import abstractmethod
import tensorflow as tf
import numpy as np
import os
import requests

from huggingface_hub import InferenceClient

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

class PyTorchFrozenEncoder(APIEncoder):
    """
    Wrap a PyTorch model as a frozen TensorFlow layer.

    Lets you use a PyTorch model inside a TensorFlow/Keras pipeline, but without 
    gradient flow (hence "frozen"—the weights never update).
    """

    def __init__(self, model_name: str, **kwargs):
        # Get embedding dim from config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        embedding_dim = config.hidden_size 

        super().__init__(model_name=model_name, embedding_dim=embedding_dim, **kwargs)
        self._torch_model = None
        self._tokenizer = None

    @property
    def torch_model(self):
        if self._torch_model is None:
            import torch
            from transformers import AutoModel
            self._torch_model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self._torch_model.eval()

        return self._torch_model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Llama-based models don't have a pad token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        
        return self._tokenizer
    
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch
        
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt' # PyTorch tensors
        )

        with torch.no_grad():
            outputs = self.torch_model(**batch)

        # Mean pooling over sequence length
        attention_mask = batch['attention_mask'].unsqueeze(-1)
        hidden_states = outputs.last_hidden_state
        masked = hidden_states * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        embeddings = summed / counts

        return embeddings.numpy()


class HFInferenceEncoder(APIEncoder):
    """
    Encoder using HuggingFace's API.

    Supported models: https://huggingface.co/inference/models
    """
    EMBEDDING_DIMS = {
        'knowledgator/Llama-encoder-1.0B': 2048,
        'intfloat/e5-mistral-7b-instruct': 4096,
        'Salesforce/SFR-Embedding-Mistral': 4096
    }

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