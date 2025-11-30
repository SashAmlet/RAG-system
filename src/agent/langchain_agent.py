from typing import Optional
import logging

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage

from src.storage.langchain_adapter import FAISSLangChainAdapter

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    LangChain агент для RAG системи.
    Використовує Ollama (Qwen 2.5) як LLM та FAISS як knowledge base.
    """

    def __init__(
        self,
        faiss_index_path: str,
        model: str = "qwen2.5:7b",
        temperature: float = 0.1,
        top_k: int = 4,
    ):
        """
        Args:
            faiss_index_path: Шлях до FAISS індексу
            model: Ollama модель (qwen2.5:7b, llama3.1:8b, etc.)
            temperature: Креативність (0-1, 0.1 для точності)
            top_k: Кількість релевантних чанків для retrieval
        """
        logger.info(f"Ініціалізація RAG Agent з моделлю {model}")

        self.top_k = top_k

        # 1. LLM (Ollama)
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_ctx=4096,  # Context window
        )
        logger.info("✅ Ollama LLM підключено")

        # 2. Завантажуємо FAISS vectorstore
        self.vectorstore = FAISSLangChainAdapter.load_faiss(faiss_index_path)
        logger.info(f"✅ FAISS індекс завантажено: {faiss_index_path}")

        # 3. Створюємо retrieval tool
        retriever_tool = Tool(
            name="search_knowledge_base",
            description=(
                "Searches the knowledge base for relevant information. "
                "Use this tool to find context before answering questions. "
                "Input should be a search query (in Ukrainian or English)."
            ),
            func=self._search_knowledge_base,
        )

        # 4. Створюємо агента (LangChain v1.0+)
        system_prompt = self._get_system_prompt()

        self.agent = create_agent(
            model=self.llm, tools=[retriever_tool], system_prompt=system_prompt
        )
        logger.info("✅ LangChain Agent створено")

    def _search_knowledge_base(self, query: str) -> str:
        """
        Tool function для пошуку в knowledge base.

        Args:
            query: Пошуковий запит

        Returns:
            Контекст з релевантних документів
        """
        logger.info(f"Пошук в knowledge base: '{query[:50]}...'")

        # Пошук релевантних документів
        docs = self.vectorstore.similarity_search(query, k=self.top_k)

        if not docs:
            return "Релевантної інформації не знайдено в документах."

        # Формуємо контекст
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Джерело {i}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)
        logger.info(f"Знайдено {len(docs)} релевантних фрагментів")

        return context

    def _get_system_prompt(self) -> str:
        """Системний промпт для агента (українською)"""
        return """Ти — експертний асистент, який відповідає на запитання на основі наданих документів.

ПРАВИЛА:
1. ЗАВЖДИ використовуй інструмент search_knowledge_base для пошуку інформації перед відповіддю
2. Давай точні відповіді ЛИШЕ на основі знайденого контексту
3. Якщо інформації недостатньо, чесно скажи про це
4. НЕ вигадуй факти, яких немає в документах
5. Відповідай УКРАЇНСЬКОЮ мовою
6. Посилайся на джерела через [Джерело 1], [Джерело 2] тощо
7. Структуруй відповідь логічно та зрозуміло

Приклад хорошої відповіді:
"Машинне навчання — це підгалузь штучного інтелекту [Джерело 1]. 
Воно дозволяє комп'ютерам вчитися на даних без явного програмування [Джерело 2]."
"""

    def query(self, question: str) -> dict:
        """
        Обробляє запит користувача.

        Args:
            question: Запитання користувача

        Returns:
            Dict з відповіддю та метаданими
        """
        logger.info(f"Обробка запиту: '{question}'")

        try:
            # Викликаємо агента
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )

            # Витягуємо відповідь
            # response має структуру: {"messages": [...]}
            messages = response.get("messages", [])

            if not messages:
                return {
                    "answer": "Вибачте, не вдалося згенерувати відповідь.",
                    "error": "No messages in response",
                }

            # Останнє повідомлення = відповідь агента
            answer = (
                messages[-1].content
                if hasattr(messages[-1], "content")
                else str(messages[-1])
            )

            logger.info("✅ Відповідь згенеровано")

            return {
                "answer": answer,
                "question": question,
                "num_messages": len(messages),
            }

        except Exception as e:
            logger.error(f"Помилка при обробці запиту: {e}")
            return {
                "answer": f"Помилка: {str(e)}",
                "error": str(e),
                "question": question,
            }

    def stream_query(self, question: str):
        """
        Streaming відповідь (для майбутньої інтеграції з UI).

        Args:
            question: Запитання користувача

        Yields:
            Частини відповіді
        """
        try:
            for chunk in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]}
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Помилка при streaming: {e}")
            yield {"error": str(e)}
