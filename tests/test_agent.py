"""
Тести для AIAgent
"""
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.agent import AIAgent
from src.storage.storage import FAISSStorage
from src.embeddings.embedder import EmbedderFactory
from src.agent.llm_client import LLMClientFactory
from src.models import EmbedderResult, TextChunk
import numpy as np


def test_agent_with_mock_data():
    """Тест AIAgent з симульованими даними"""
    print("=" * 70)
    print("ТЕСТ AIAGENT")
    print("=" * 70)

    # ВАЖЛИВО: Встановіть ваш API ключ
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "your-api-key-here")

    if PERPLEXITY_API_KEY == "your-api-key-here":
        print("⚠️  УВАГА: Встановіть PERPLEXITY_API_KEY environment variable!")
        print("export PERPLEXITY_API_KEY='your-key'")
        return

    # 1. Створюємо компоненти
    print("\n1️⃣ Створення компонентів...")

    storage = FAISSStorage(dimension=384)
    embedder = EmbedderFactory.create(method="sbert")
    llm_client = LLMClientFactory.create(
        provider="perplexity",
        api_key=PERPLEXITY_API_KEY,
        model="sonar"  # Економна модель
    )

    # 2. Додаємо тестові документи
    print("2️⃣ Індексація тестових документів...")

    test_chunks = [
        TextChunk(
            text=
            "Машинне навчання — це підгалузь штучного інтелекту, яка дозволяє комп'ютерам вчитися на даних без явного програмування. Системи ML аналізують патерни та приймають рішення.",
            chunk_id="chunk_1",
            document_id="doc_ml",
            chunk_index=0),
        TextChunk(
            text=
            "Python є найпопулярнішою мовою для data science та машинного навчання. Бібліотеки як NumPy, Pandas та Scikit-learn надають потужні інструменти для аналізу даних.",
            chunk_id="chunk_2",
            document_id="doc_python",
            chunk_index=0),
        TextChunk(
            text=
            "Трансформери революціонізували обробку природної мови. Механізм attention дозволяє моделі фокусуватися на релевантних частинах вхідних даних.",
            chunk_id="chunk_3",
            document_id="doc_transformers",
            chunk_index=0)
    ]

    # Векторизуємо і додаємо в storage
    embeddings = embedder.embed_batch(test_chunks)
    storage.add(embeddings)

    print(f"   ✅ Додано {len(test_chunks)} документів")

    # 3. Створюємо AIAgent
    print("3️⃣ Ініціалізація AIAgent...")

    agent = AIAgent(storage=storage,
                    embedder=embedder,
                    llm_client=llm_client,
                    top_k=3,
                    min_similarity=0.2,
                    temperature=0.1,
                    max_tokens=300,
                    language="uk")

    print("   ✅ AIAgent готовий\n")

    # 4. Тестові запити
    test_queries = [
        "Що таке машинне навчання?", "Яка мова програмування найкраща для ML?",
        "Як працюють трансформери?"
    ]

    print("4️⃣ Обробка тестових запитів:\n")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Запит {i}: {query}")
        print("-" * 70)

        # Отримуємо відповідь
        response = agent.answer(query)

        # Виводимо результат
        print(f"\n💡 Відповідь:\n{response.answer}\n")

        print(f"📚 Джерела ({len(response.sources)}):")
        for j, source in enumerate(response.sources, 1):
            print(
                f"  [{j}] Score: {source.score:.3f} | {source.chunk.text[:60]}..."
            )

        print(f"\n📊 Метадані: {response.metadata}")
        print("=" * 70)

    print("\n🎉 Тест завершено успішно!")


if __name__ == "__main__":
    test_agent_with_mock_data()
