from abc import ABC, abstractmethod
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging

from src.models import TextChunk, EmbedderResult

logger = logging.getLogger(__name__)


class IEmbedder(ABC):
    """Базовий інтерфейс для embedder'ів"""

    @abstractmethod
    def embed(self, chunk: TextChunk) -> EmbedderResult:
        """Векторизує ОДИН текстовий чанк."""
        pass

    @abstractmethod
    def embed_batch(self, chunks: List[TextChunk]) -> List[EmbedderResult]:
        """Векторизує множину чанків (оптимізовано)."""
        pass


class SentenceBERTEmbedder(IEmbedder):
    """
    Embedder на базі Sentence-BERT.
    Використовує трансформерні моделі для створення якісних семантичних векторів.
    """

    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 batch_size: int = 32):
        """
        Args:
            model_name: Назва моделі з HuggingFace
            device: 'cuda', 'cpu' або None (авто)
            batch_size: Розмір батча для batch_encode
        """
        logger.info(f"Завантаження SBERT моделі: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Модель завантажена. Розмірність: {self.embedding_dim}")

    def embed(self, chunk: TextChunk) -> EmbedderResult:
        """
        Векторизує один чанк.
        Для множини чанків використовуйте embed_batch() (швидше).
        """
        vector = self.model.encode(chunk.text,
                                   convert_to_numpy=True,
                                   show_progress_bar=False).tolist()

        return EmbedderResult(vector=vector,
                              chunk_id=chunk.chunk_id,
                              document_id=chunk.document_id,
                              metadata={
                                  "method": "Sentence-BERT",
                                  "model": self.model_name,
                                  "chunk_index": chunk.chunk_index,
                                  "text_length": len(chunk.text)
                              })

    def embed_batch(self, chunks: List[TextChunk]) -> List[EmbedderResult]:
        """
        Векторизує множину чанків за один прохід.
        Використовує batch encoding для оптимізації.
        """
        if not chunks:
            return []

        logger.info(
            f"Векторизація {len(chunks)} чанків (batch_size={self.batch_size})"
        )

        # Витягуємо тексти
        texts = [chunk.text for chunk in chunks]

        # Batch векторизація (швидше в 5-10 разів)
        vectors = self.model.encode(texts,
                                    batch_size=self.batch_size,
                                    convert_to_numpy=True,
                                    show_progress_bar=len(chunks) > 50)

        # Створюємо результати
        results = []
        for chunk, vector in zip(chunks, vectors):
            results.append(
                EmbedderResult(vector=vector.tolist(),
                               chunk_id=chunk.chunk_id,
                               document_id=chunk.document_id,
                               metadata={
                                   "method": "Sentence-BERT",
                                   "model": self.model_name,
                                   "chunk_index": chunk.chunk_index,
                                   "text_length": len(chunk.text)
                               }))

        logger.info(
            f"Векторизація завершена. Розмірність: {len(results[0].vector)}")
        return results


class EmbedderFactory:
    """Фабрика для створення embedder'ів"""

    @staticmethod
    def create(method: str = "sbert", **kwargs) -> IEmbedder:
        """
        Створює embedder.
        
        Args:
            method: Тип embedder'а (наразі лише 'sbert')
            **kwargs: Параметри для embedder'а
            
        Returns:
            Екземпляр IEmbedder
        """
        method = method.lower()

        if method == "sbert":
            return SentenceBERTEmbedder(model_name=kwargs.get(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                                        device=kwargs.get("device", None),
                                        batch_size=kwargs.get(
                                            "batch_size", 32))
        else:
            raise ValueError(
                f"Unknown embedder method: {method}. Available: 'sbert'")
