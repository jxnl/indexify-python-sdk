from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from indexify.extractor_sdk.data import Feature
from indexify.extractor_sdk.extractor import Extractor, Feature


class SentenceTransformersEmbedding:
    def __init__(self, model_name) -> None:
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            f"sentence-transformers/{model_name}"
        )
        self._model = AutoModel.from_pretrained(
            f"sentence-transformers/{model_name}", torchscript=True
        )
        self._model.eval()

    def embed_batch(self, inputs: List[str]) -> List[List[float]]:
        result = self._embed(inputs)
        return result.tolist()

    def embed(self, query: str) -> List[float]:
        result = self._embed([query])
        return result[0].tolist()

    def _embed(self, inputs: List[str]) -> torch.Tensor:
        encoded_input = self._tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        )
        sentence_embeddings = self._model(**encoded_input)
        return F.normalize(sentence_embeddings, p=2, dim=1)


class BasicSentenceTransformerModels(Extractor):
    def __init__(self, model: str):
        super().__init__()
        self.model = SentenceTransformersEmbedding(model)

    def extract(self, input: str) -> List[Feature]:
        embeddings = self.model.embed(input)
        return [Feature.embedding(values=embeddings)]


class BasicHFTransformerEmbeddingModels(Extractor):
    def __init__(self, model: str):
        super().__init__()
        self._model = AutoModel.from_pretrained(model, trust_remote_code=True)

    def extract(self, input: str) -> List[Feature]:
        embeddings = self.model.embed_query(input)
        return [Feature.embedding(values=embeddings)]
