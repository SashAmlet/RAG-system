import logging
import time
from typing import Optional

from src.models import AgentResponse, SearchResult
from src.storage.storage import IStorage
from src.embeddings.embedder import IEmbedder
from src.agent.retriever import Retriever
from src.agent.prompt_builder import PromptBuilder
from src.agent.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AIAgent:
    """
    AI Agent для обробки запитів у RAG системі.
    
    Workflow:
    1. Retrieve - знаходить релевантні чанки
    2. Build prompt - формує промпт з контекстом
    3. Generate - отримує відповідь від LLM
    4. Format - структурує відповідь
    """

    def __init__(self,
                 storage: IStorage,
                 embedder: IEmbedder,
                 llm_client: LLMClient,
                 top_k: int = 5,
                 min_similarity: float = 0.3,
                 temperature: float = 0.1,
                 max_tokens: int = 500,
                 language: str = "uk"):
        """
        Args:
            storage: Векторне сховище
            embedder: Embedder для векторизації запитів
            llm_client: Клієнт для LLM
            top_k: Кількість чанків для retrieval
            min_similarity: Мінімальний score для фільтрації
            temperature: Креативність LLM (0-2)
            max_tokens: Макс. довжина відповіді
            language: Мова відповідей
        """
        self.retriever = Retriever(storage, embedder, min_similarity)
        self.prompt_builder = PromptBuilder(language)
        self.llm_client = llm_client

        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(
            f"AIAgent ініціалізовано (top_k={top_k}, temp={temperature})")

    def answer(self, query: str) -> AgentResponse:
        """
        Обробляє запит користувача та генерує відповідь.
        
        Args:
            query: Запитання користувача
            
        Returns:
            AgentResponse з відповіддю та джерелами
        """
        logger.info(f"Обробка запиту: '{query}'")
        start_time = time.time()

        try:
            # 1. Retrieve релевантних чанків
            search_results = self.retriever.retrieve(query, top_k=self.top_k)

            if not search_results:
                logger.warning("Не знайдено релевантних документів")
                return self._handle_no_context(query)

            # 2. Формуємо промпт
            system_prompt, user_prompt = self.prompt_builder.build_qa_prompt(
                query, search_results)

            # 3. Генеруємо відповідь
            answer = self.llm_client.generate(system_prompt=system_prompt,
                                              user_prompt=user_prompt,
                                              temperature=self.temperature,
                                              max_tokens=self.max_tokens)

            # 4. Формуємо результат
            duration = time.time() - start_time

            response = AgentResponse(answer=answer,
                                     sources=search_results,
                                     query=query,
                                     metadata={
                                         "duration_seconds":
                                         round(duration, 2),
                                         "num_sources":
                                         len(search_results),
                                         "avg_similarity":
                                         round(
                                             sum(r.score
                                                 for r in search_results) /
                                             len(search_results), 3)
                                     })

            logger.info(f"Відповідь згенеровано за {duration:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Помилка при обробці запиту: {str(e)}")
            return AgentResponse(
                answer=f"Вибачте, виникла помилка при обробці запиту: {str(e)}",
                sources=[],
                query=query,
                metadata={"error": str(e)})

    def _handle_no_context(self, query: str) -> AgentResponse:
        """
        Обробляє випадок коли не знайдено релевантних документів.
        """
        system_prompt, user_prompt = self.prompt_builder.build_no_context_prompt(
            query)

        try:
            answer = self.llm_client.generate(system_prompt=system_prompt,
                                              user_prompt=user_prompt,
                                              temperature=0.1,
                                              max_tokens=150)
        except Exception:
            answer = "Вибачте, не знайдено релевантної інформації в документах для відповіді на ваше запитання."

        return AgentResponse(answer=answer,
                             sources=[],
                             query=query,
                             metadata={"no_context": True})
