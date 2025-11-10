from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# ----- Models -----

@dataclass
class PreprocessorResult:
    tokens: List[str]

@dataclass
class EmbedderResult:
    vector: List[float]
    metadata: Dict[str, Any]  # например {"method": "TF-IDF", "dim": 300}

# ----- Strategy interface -----

class IEmbedder(ABC):
    @abstractmethod
    def embed(self, preprocessed: PreprocessorResult) -> EmbedderResult:
        pass

# ----- Concrete strategies -----

class BOWEmbedder(IEmbedder):
    def __init__(self, vocabulary: List[str]):
        self.vocab = vocabulary

    def embed(self, preprocessed: PreprocessorResult) -> EmbedderResult:
        counter = Counter(preprocessed.tokens)
        vector = [counter.get(word, 0) for word in self.vocab]
        return EmbedderResult(
            vector=vector,
            metadata={"method": "BOW", "dim": len(vector)}
        )


class TFIDFEmbedder(IEmbedder):
    def __init__(self, documents: List[str]):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(documents)

    def embed(self, preprocessed: PreprocessorResult) -> EmbedderResult:
        joined_text = " ".join(preprocessed.tokens)
        vector = self.vectorizer.transform([joined_text]).toarray()[0].tolist()
        return EmbedderResult(
            vector=vector,
            metadata={"method": "TF-IDF", "dim": len(vector)}
        )


class Word2VecEmbedder(IEmbedder):
    def __init__(self, model):
        self.model = model  # gensim Word2Vec или FastText модель

    def embed(self, preprocessed: PreprocessorResult) -> EmbedderResult:
        vectors = [self.model.wv[word] for word in preprocessed.tokens if word in self.model.wv]
        if not vectors:
            avg_vector = [0.0] * self.model.vector_size
        else:
            avg_vector = np.mean(vectors, axis=0).tolist()
        return EmbedderResult(
            vector=avg_vector,
            metadata={"method": "Word2Vec", "dim": len(avg_vector)}
        )


class SentenceBERTEmbedder(IEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, preprocessed: PreprocessorResult) -> EmbedderResult:
        text = " ".join(preprocessed.tokens)
        vector = self.model.encode(text).tolist()
        return EmbedderResult(
            vector=vector,
            metadata={"method": "Sentence-BERT", "dim": len(vector)}
        )


class OpenAIEmbedder(IEmbedder):
    def __init__(self, client, model="text-embedding-ada-002"):
        self.client = client
        self.model = model

    def embed(self, preprocessed: PreprocessorResult) -> EmbedderResult:
        text = " ".join(preprocessed.tokens)
        response = self.client.embeddings.create(model=self.model, input=text)
        vector = response.data[0].embedding
        return EmbedderResult(
            vector=vector,
            metadata={"method": "OpenAI", "dim": len(vector)}
        )

# ----- Factory -----

class EmbedderFactory:
    @staticmethod
    def create(method: str, **kwargs) -> IEmbedder:
        if method == "bow":
            return BOWEmbedder(kwargs["vocabulary"])
        elif method == "tfidf":
            return TFIDFEmbedder(kwargs["documents"])
        elif method == "word2vec":
            return Word2VecEmbedder(kwargs["model"])
        elif method == "sbert":
            return SentenceBERTEmbedder(kwargs.get("model_name"))
        elif method == "openai":
            return OpenAIEmbedder(kwargs["client"], kwargs.get("model", "text-embedding-ada-002"))
        else:
            raise ValueError(f"Unknown embedder method: {method}")


docs = ["hello world", "machine learning is fun", "hello machine"]
preprocessed = PreprocessorResult(tokens=["hello world", "machine"])

# Создать TF-IDF embedder через фабрику
embedder = EmbedderFactory.create("bow", vocabulary=docs)
# embedder = EmbedderFactory.create("sbert", model_name="sentence-transformers/all-MiniLM-L6-v2")
result = embedder.embed(preprocessed)

print(result.metadata)  # {'method': 'TF-IDF', 'dim': ...}
print(len(result.vector))
print(result.vector)
# print(embedder.vectorizer.get_feature_names_out())
