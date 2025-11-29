import logging
import time
from typing import Optional, List, Dict, Any, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END

from src.models import AgentResponse, SearchResult
from src.storage.storage import IStorage
from src.embeddings.embedder import IEmbedder
from src.agent.retriever import Retriever
from src.agent.prompt_builder import PromptBuilder
from src.agent.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    query: str
    documents: List[SearchResult]
    answer: str
    system_prompt: str
    user_prompt: str


class AIAgent:
    """
    AI Agent для обробки запитів у RAG системі.
    Використовує LangGraph для оркестрації.
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
        
        self.retriever = Retriever(storage, embedder, min_similarity)
        self.prompt_builder = PromptBuilder(language)
        self.llm_client = llm_client

        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Ініціалізація графа
        self.workflow = self._build_graph()
        logger.info(f"AIAgent (LangGraph) ініціалізовано")

    def _build_graph(self) -> StateGraph:
        """Створює граф обробки запиту"""
        workflow = StateGraph(AgentState)

        # Додаємо вузли
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("build_prompt", self._build_prompt_node)
        workflow.add_node("generate", self._generate_node)

        # Визначаємо ребра
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "build_prompt")
        workflow.add_edge("build_prompt", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def _retrieve_node(self, state: AgentState) -> Dict:
        """Вузол пошуку документів"""
        query = state["query"]
        results = self.retriever.retrieve(query, top_k=self.top_k)
        return {"documents": results}

    def _build_prompt_node(self, state: AgentState) -> Dict:
        """Вузол створення промпту"""
        query = state["query"]
        documents = state["documents"]
        
        if not documents:
            sys_prompt, user_prompt = self.prompt_builder.build_no_context_prompt(query)
        else:
            sys_prompt, user_prompt = self.prompt_builder.build_qa_prompt(query, documents)
            
        return {"system_prompt": sys_prompt, "user_prompt": user_prompt}

    def _generate_node(self, state: AgentState) -> Dict:
        """Вузол генерації відповіді"""
        answer = self.llm_client.generate(
            system_prompt=state["system_prompt"],
            user_prompt=state["user_prompt"],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return {"answer": answer}

    def answer(self, query: str) -> AgentResponse:
        """
        Обробляє запит користувача через граф.
        """
        logger.info(f"Обробка запиту: '{query}'")
        start_time = time.time()

        try:
            # Запуск графа
            initial_state = {"query": query, "documents": [], "answer": "", "system_prompt": "", "user_prompt": ""}
            final_state = self.workflow.invoke(initial_state)

            duration = time.time() - start_time
            
            # Формування відповіді
            response = AgentResponse(
                answer=final_state["answer"],
                sources=final_state["documents"],
                query=query,
                metadata={
                    "duration_seconds": round(duration, 2),
                    "num_sources": len(final_state["documents"]),
                    "avg_similarity": round(sum(r.score for r in final_state["documents"]) / len(final_state["documents"]), 3) if final_state["documents"] else 0
                }
            )

            logger.info(f"Відповідь згенеровано за {duration:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Помилка при обробці запиту: {str(e)}")
            return AgentResponse(
                answer=f"Вибачте, виникла помилка: {str(e)}",
                sources=[],
                query=query,
                metadata={"error": str(e)}
            )
