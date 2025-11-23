from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
import faiss
import pickle
import logging
from pathlib import Path

from src.models import EmbedderResult, TextChunk, SearchResult

logger = logging.getLogger(__name__)


class IStorage(ABC):
    """Базовий інтерфейс для векторного сховища"""

    @abstractmethod
    def add(self, embeddings: List[EmbedderResult]) -> None:
        """Додає вектори в індекс"""
        pass

    @abstractmethod
    def search(self,
               query_vector: List[float],
               top_k: int = 5) -> List[SearchResult]:
        """Знаходить top_k найбільш схожих чанків"""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Зберігає індекс на диск"""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Завантажує індекс з диску"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Очищає індекс"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """Повертає статистику індексу"""
        pass


class FAISSStorage(IStorage):
    """
    Storage на базі FAISS (Facebook AI Similarity Search).
    """

    def __init__(self, dimension: int = 384, normalize_vectors: bool = True):
        """
        Args:
            dimension: Розмірність векторів (384 для all-MiniLM-L6-v2)
            normalize_vectors: Нормалізувати вектори для cosine similarity
        """
        self.dimension = dimension
        self.normalize_vectors = normalize_vectors

        # FAISS індекс
        self.index = faiss.IndexFlatIP(self.dimension)

        # Metadata storage
        self.metadata_store: Dict[int, TextChunk] = {}  # faiss_id -> TextChunk
        self.chunk_id_to_faiss_id: Dict[str, int] = {}  # chunk_id -> faiss_id

        # Лічильник для нових ID
        self.next_id = 0

        logger.info(f"FAISSStorage ініціалізовано: dim={dimension}")

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 нормалізація векторів для cosine similarity"""
        if not self.normalize_vectors:
            return vectors

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Уникаємо ділення на 0
        return vectors / norms

    def add(self, embeddings: List[EmbedderResult]) -> None:
        """
        Додає вектори в індекс (інкрементально).
        
        Args:
            embeddings: Список векторних представлень чанків
        """
        if not embeddings:
            logger.warning("Отримано порожній список embeddings")
            return

        logger.info(f"Додавання {len(embeddings)} векторів в індекс")

        # Конвертуємо в numpy array
        vectors = np.array([emb.vector for emb in embeddings],
                           dtype=np.float32)

        # Нормалізуємо
        vectors = self._normalize(vectors)

        # Додаємо в FAISS
        self.index.add(vectors)

        # Зберігаємо metadata
        for i, emb in enumerate(embeddings):
            faiss_id = self.next_id + i

            # Створюємо TextChunk з metadata
            chunk = TextChunk(
                text=emb.metadata.get('text', ''),  # Якщо текст був збережений
                chunk_id=emb.chunk_id,
                document_id=emb.document_id,
                chunk_index=emb.metadata.get('chunk_index', 0),
                metadata=emb.metadata)

            self.metadata_store[faiss_id] = chunk
            self.chunk_id_to_faiss_id[emb.chunk_id] = faiss_id

        self.next_id += len(embeddings)

        logger.info(
            f"Додано успішно. Всього векторів в індексі: {self.index.ntotal}")

    def search(self,
               query_vector: List[float],
               top_k: int = 5) -> List[SearchResult]:
        """
        Знаходить top_k найбільш схожих чанків.
        """
        if self.index.ntotal == 0:
            logger.warning("Індекс порожній, неможливо виконати пошук")
            return []

        # Конвертуємо в numpy
        query = np.array([query_vector], dtype=np.float32)
        query = self._normalize(query)

        # Обмежуємо top_k до кількості векторів в індексі
        top_k = min(top_k, self.index.ntotal)

        # Пошук в FAISS
        distances, indices = self.index.search(query, top_k)

        # Конвертуємо в SearchResult
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS повертає -1 якщо не знайдено
                continue

            chunk = self.metadata_store.get(idx)
            if chunk is None:
                logger.warning(f"Метадані для індексу {idx} не знайдено")
                continue

            # Конвертуємо distance (inner product) в similarity score (0-1)
            # Після нормалізації: inner product ∈ [-1, 1]
            # Конвертуємо в [0, 1]: (inner_product + 1) / 2
            score = float((dist + 1) / 2)

            results.append(
                SearchResult(chunk=chunk,
                             score=score,
                             document_id=chunk.document_id,
                             chunk_id=chunk.chunk_id))

        logger.info(f"Знайдено {len(results)} результатів")
        return results

    def save(self, filepath: str) -> None:
        """
        Зберігає індекс + metadata на диск.
        
        Args:
            filepath: Шлях без розширення (додасться .faiss та .pkl)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        faiss_path = str(filepath) + ".faiss"
        metadata_path = str(filepath) + ".pkl"

        faiss.write_index(self.index, faiss_path)

        # Зберігаємо metadata
        metadata = {
            'metadata_store': self.metadata_store,
            'chunk_id_to_faiss_id': self.chunk_id_to_faiss_id,
            'next_id': self.next_id,
            'dimension': self.dimension
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Індекс збережено: {faiss_path}, {metadata_path}")

    def load(self, filepath: str) -> None:
        """
        Завантажує індекс + metadata з диску.
        
        Args:
            filepath: Шлях без розширення
        """
        faiss_path = str(filepath) + ".faiss"
        metadata_path = str(filepath) + ".pkl"

        # Перевірка існування файлів
        if not Path(faiss_path).exists():
            raise FileNotFoundError(f"FAISS індекс не знайдено: {faiss_path}")
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata не знайдено: {metadata_path}")

        # Завантажуємо FAISS індекс
        self.index = faiss.read_index(faiss_path)

        # Завантажуємо metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.metadata_store = metadata['metadata_store']
        self.chunk_id_to_faiss_id = metadata['chunk_id_to_faiss_id']
        self.next_id = metadata['next_id']

        # Перевірка розмірності
        if metadata['dimension'] != self.dimension:
            logger.warning(f"Розмірність індексу ({metadata['dimension']}) "
                           f"не співпадає з очікуваною ({self.dimension})")

        logger.info(f"Індекс завантажено: {self.index.ntotal} векторів")

    def clear(self) -> None:
        """Очищає індекс та metadata"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store.clear()
        self.chunk_id_to_faiss_id.clear()
        self.next_id = 0
        logger.info("Індекс очищено")

    def get_stats(self) -> Dict[str, int]:
        """Повертає статистику індексу"""
        unique_docs = len(
            set(chunk.document_id for chunk in self.metadata_store.values()))
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'unique_documents': unique_docs,
            'normalize_vectors': self.normalize_vectors,
            'metadata_count': len(self.metadata_store)
        }


class StorageFactory:
    """Фабрика для створення storage"""

    @staticmethod
    def create(storage_type: str = "faiss", **kwargs) -> IStorage:
        """
        Створює storage.
        
        Args:
            storage_type: Тип storage ('faiss')
            **kwargs: Параметри для storage
            
        Returns:
            Екземпляр IStorage
        """
        storage_type = storage_type.lower()

        if storage_type == "faiss":
            return FAISSStorage(dimension=kwargs.get("dimension", 384),
                                normalize_vectors=kwargs.get(
                                    "normalize_vectors", True))
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
