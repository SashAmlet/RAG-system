import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
import pprint

from src.embeddings.embedder import EmbedderFactory
from src.storage.storage import FAISSStorage
from src.agent.agent import AIAgent
from src.agent.llm_client import LLMClientFactory

# Завантаження змінних середовища
load_dotenv()

# Ініціалізація компонентів
embedder = EmbedderFactory.create(
    method="sbert",
    model_name=os.getenv("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    batch_size=int(os.getenv("EMBEDDER_BATCH_SIZE", 32)),
)

storage = FAISSStorage(dimension=384)
index_path = "data/indexes/knowledge_base"
if os.path.exists(f"{index_path}.faiss"):
    storage.load(index_path)
else:
    print("Warning: Index not found. Please run 'python main.py --mode index' first.")

# Використовуємо Ollama
llm_client = LLMClientFactory.create(
    provider="ollama",
    model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
    temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
)

agent = AIAgent(
    storage=storage,
    embedder=embedder,
    llm_client=llm_client,
    top_k=int(os.getenv("TOP_K", 5)),
    min_similarity=float(os.getenv("MIN_SIMILARITY", 0.3)),
    temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", 800)),
    language="uk",
)

# Створення FastAPI додатку
app = FastAPI(
    title="RAG System API",
    version="1.0",
    description="API for RAG System using LangChain & Ollama",
)


# Адаптер для LangServe
def run_agent(input_data: dict) -> str:
    """
    Обгортка для виклику агента.
    Обробляє різні формати вхідних даних від LangServe.
    """
    print(f"DEBUG: Input data type: {type(input_data)}")
    print("DEBUG: Input data:")
    pprint.pprint(input_data)

    raw_query = None

    # 1. Визначаємо де лежить запит/повідомлення
    if isinstance(input_data, dict):
        if "input" in input_data:
            raw_query = input_data["input"]
        elif "messages" in input_data:
            raw_query = input_data["messages"]
        elif "question" in input_data:
            raw_query = input_data["question"]
        elif "undefined" in input_data:
            # LangServe Playground іноді надсилає дані з ключем 'undefined'
            raw_query = input_data["undefined"]
        # Fallback: якщо це dict, але немає відомих ключів
        elif not raw_query:
            if "content" in input_data:
                raw_query = input_data["content"]
    elif isinstance(input_data, list):
        raw_query = input_data

    # 2. Витягуємо текст запиту
    final_query = ""

    if isinstance(raw_query, str):
        final_query = raw_query
    elif isinstance(raw_query, list):
        print(f"DEBUG: Processing list of {len(raw_query)} items")
        # Якщо це список повідомлень, шукаємо останнє повідомлення від користувача
        for i, msg in enumerate(reversed(raw_query)):
            print(f"DEBUG: Checking item {i}: {type(msg)}")
            # LangChain Message object
            if hasattr(msg, "content"):
                msg_type = getattr(msg, "type", "")
                print(f"DEBUG: Item is object with content. Type: {msg_type}")
                if msg_type == "human" or msg_type == "user":
                    final_query = msg.content
                    break
            # Dictionary representation
            elif isinstance(msg, dict):
                msg_type = msg.get("type") or msg.get("role")
                print(f"DEBUG: Item is dict. Type: {msg_type}")
                if msg_type in ["human", "user"]:
                    final_query = msg.get("content", "")
                    break
            # String
            elif isinstance(msg, str):
                print("DEBUG: Item is string")
                final_query = msg
                break

    print(f"DEBUG: Extracted query: '{final_query}'")

    if not final_query:
        return "Error: Could not extract query from input. Please check server logs."

    # Викликаємо метод answer
    response = agent.answer(final_query)
    return response.answer


# Створюємо Runnable з функції
chain = RunnableLambda(run_agent)

add_routes(app, chain, path="/rag", playground_type="chat")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
