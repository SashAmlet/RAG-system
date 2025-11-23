from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from src.models import ProcessorResult, EmbedderResult, TextChunk

# ----- Strategy interface -----


class IEmbedder(ABC):

    @abstractmethod
    def embed(self, text_chunk: TextChunk) -> EmbedderResult:
        """Векторизує ОДИН чанк"""
        pass

    def embed_batch(self, chunks: List[TextChunk]) -> List[EmbedderResult]:
        """Векторизує множину чанків"""
        return [self.embed(chunk) for chunk in chunks]


# ----- Concrete strategies -----


class BOWEmbedder(IEmbedder):

    def __init__(self, vocabulary: List[str]):
        self.vocab = vocabulary

    def embed(self, result: ProcessorResult) -> EmbedderResult:
        # Використовуємо токени з моделі або розбиваємо текст самі
        tokens = result.tokens if result.tokens else result.processed_text.split(
        )

        counter = Counter(tokens)
        vector = [counter.get(word, 0) for word in self.vocab]
        return EmbedderResult(vector=vector,
                              metadata={
                                  "method": "BOW",
                                  "dim": len(vector)
                              })


class TFIDFEmbedder(IEmbedder):

    def __init__(self, documents: List[str]):
        # У реальному RAG це проблематично, бо TF-IDF треба тренувати на всьому корпусі заздалегідь.
        # Але для навчального проєкту ок.
        self.vectorizer = TfidfVectorizer()
        if documents:
            self.vectorizer.fit(documents)
        else:
            # Fallback щоб не падало, якщо документів 0
            self.vectorizer.fit(["dummy text"])

    def embed(self, result: ProcessorResult) -> EmbedderResult:
        text = result.processed_text
        vector = self.vectorizer.transform([text]).toarray()[0].tolist()
        return EmbedderResult(vector=vector,
                              metadata={
                                  "method": "TF-IDF",
                                  "dim": len(vector)
                              })


class Word2VecEmbedder(IEmbedder):

    def __init__(self, model):
        self.model = model

    def embed(self, result: ProcessorResult) -> EmbedderResult:
        tokens = result.tokens if result.tokens else result.processed_text.split(
        )

        vectors = [
            self.model.wv[word] for word in tokens if word in self.model.wv
        ]
        if not vectors:
            avg_vector = [0.0] * self.model.vector_size
        else:
            avg_vector = np.mean(vectors, axis=0).tolist()

        return EmbedderResult(vector=avg_vector,
                              metadata={
                                  "method": "Word2Vec",
                                  "dim": len(avg_vector)
                              })


class SentenceBERTEmbedder(IEmbedder):

    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, result: ProcessorResult) -> EmbedderResult:
        # SBERT працює з чистим текстом
        text = result.processed_text
        vector = self.model.encode(text).tolist()
        return EmbedderResult(vector=vector,
                              metadata={
                                  "method": "Sentence-BERT",
                                  "dim": len(vector)
                              })


class OpenAIEmbedder(IEmbedder):

    def __init__(self, client, model="text-embedding-ada-002"):
        self.client = client
        self.model = model

    def embed(self, result: ProcessorResult) -> EmbedderResult:
        text = result.processed_text
        # Замінюємо переноси рядків, це рекомендація OpenAI
        text = text.replace("\n", " ")

        response = self.client.embeddings.create(model=self.model, input=text)
        # OpenAI v1.x повертає об'єкт, а не словник
        vector = response.data[0].embedding
        return EmbedderResult(vector=vector,
                              metadata={
                                  "method": "OpenAI",
                                  "dim": len(vector)
                              })


# ----- Factory -----


class EmbedderFactory:

    @staticmethod
    def create(method: str, **kwargs) -> IEmbedder:
        method = method.lower()
        if method == "bow":
            return BOWEmbedder(kwargs["vocabulary"])
        elif method == "tfidf":
            return TFIDFEmbedder(kwargs["documents"])
        elif method == "word2vec":
            return Word2VecEmbedder(kwargs["model"])
        elif method == "sbert":
            return SentenceBERTEmbedder(
                kwargs.get("model_name",
                           "sentence-transformers/all-MiniLM-L6-v2"))
        elif method == "openai":
            return OpenAIEmbedder(
                kwargs["client"], kwargs.get("model",
                                             "text-embedding-ada-002"))
        else:
            raise ValueError(f"Unknown embedder method: {method}")
