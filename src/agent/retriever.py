import logging
from typing import List

from src.models import TextChunk, SearchResult
from src.storage.storage import IStorage
from src.embeddings.embedder import IEmbedder

logger = logging.getLogger(__name__)


class Retriever:
    """
    Пошук релевантних фрагментів для запиту користувача.
    """

    def __init__(
        self, storage: IStorage, embedder: IEmbedder, min_similarity: float = 0.3
    ):
        """
        Args:
            storage: Векторне сховище
            embedder: Embedder для векторизації запитів
            min_similarity: Мінімальний score для фільтрації (0-1)
        """
        self.storage = storage
        self.embedder = embedder
        self.min_similarity = min_similarity

    def retrieve(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """
        Знаходить релевантні фрагменти для запиту.

        Args:
            query: Текст запиту
            top_k: Кількість результатів

        Returns:
            Список SearchResult, відсортований за релевантністю
        """
        logger.info(f"Пошук релевантних чанків для запиту: '{query[:50]}...'")

        # 1. Векторизуємо запит
        query_chunk = TextChunk(
            text=query, chunk_id="query", document_id="query", chunk_index=0
        )
        query_embedding = self.embedder.embed(query_chunk)

        # 2. Шукаємо в storage
        results = self.storage.search(query_embedding.vector, top_k=top_k)

        # 3. Фільтруємо за min_similarity
        filtered_results = [r for r in results if r.score >= self.min_similarity]

        logger.info(
            f"Знайдено {len(results)} результатів, "
            f"після фільтрації: {len(filtered_results)}"
        )

        return filtered_results
