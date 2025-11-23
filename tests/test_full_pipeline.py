import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.preprocessor import Preprocessor
from src.embeddings.embedder import EmbedderFactory
from src.storage.storage import FAISSStorage
from src.models import ProcessorResult, TextChunk


def test_full_pipeline():
    """Тест повного RAG pipeline"""
    print("=" * 70)
    print("ІНТЕГРАЦІЙНИЙ ТЕСТ: Preprocessor -> Chunker -> Embedder -> Storage")
    print("=" * 70)

    # 1. Створюємо компоненти
    preprocessor = Preprocessor()
    embedder = EmbedderFactory.create(method="sbert")
    storage = FAISSStorage(dimension=384)

    print("✅ Компоненти створено\n")

    file_name = "introduction_to_microservices_galkin_shkilniak"
    path = f"data\\raw\\{file_name}.pdf"
    result = preprocessor.process_document(path)

    # 3. Векторизуємо
    embeddings = embedder.embed_batch(result.chunks)
    print(
        f"🔢 Створено {len(embeddings)} векторів (dim={len(embeddings[0].vector)})\n"
    )

    # 4. Додаємо в storage
    storage.add(embeddings)
    stats = storage.get_stats()
    print(f"💾 Індекс створено:")
    print(f"   - Векторів: {stats['total_vectors']}")
    print(f"   - Документів: {stats['unique_documents']}")
    print(f"   - Розмірність: {stats['dimension']}\n")

    # 5. Зберігаємо на диск
    storage.save("test_knowledge_base")
    print("💾 Індекс збережено на диск\n")

    # 6. Виконуємо пошук
    query_text = "Бізнес-можливості?"
    print(f"🔍 Запит: '{query_text}'")

    # Векторизуємо запит
    query_chunk = TextChunk(text=query_text,
                            chunk_id="query",
                            document_id="query")
    query_embedding = embedder.embed(query_chunk)

    # Шукаємо
    results = storage.search(query_embedding.vector, top_k=3)

    print(f"\n📊 Знайдено {len(results)} релевантних фрагментів:\n")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.score:.4f}")
        print(f"   Text: {result.chunk.text[:80]}...")
        print(f"   Chunk ID: {result.chunk_id}")
        print(f"   Doc: {result.document_id}\n")

    # 7. Завантажуємо індекс з диску
    storage2 = FAISSStorage()
    storage2.load("test_knowledge_base")
    results2 = storage2.search(query_embedding.vector, top_k=3)
    assert len(results2) == len(results)
    print("✅ Індекс успішно завантажено з диску\n")

    # Очищаємо
    import os
    os.remove("test_knowledge_base.faiss")
    os.remove("test_knowledge_base.pkl")

    print("=" * 70)
    print("🎉 ІНТЕГРАЦІЙНИЙ ТЕСТ ПРОЙДЕНО УСПІШНО!")
    print("=" * 70)


if __name__ == "__main__":
    test_full_pipeline()
