import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedder import EmbedderFactory
from src.preprocessing.preprocessor import Preprocessor
from src.storage.storage import FAISSStorage
from src.models import EmbedderResult, SearchResult
import tempfile


def test_storage_basic():
    """Базовий тест: add, search, save, load"""

    print("=" * 60)
    print("ТЕСТ 1: Базовий функціонал Storage")
    print("=" * 60)

    # 1. Створюємо storage
    storage = FAISSStorage(dimension=384)
    print(f"✅ Storage створено: {storage.get_stats()}\n")

    prep = Preprocessor()

    file_name = "introduction_to_microservices_galkin_shkilniak"
    path = f"data\\raw\\{file_name}.pdf"
    res = prep.process_document(path)

    embedder = EmbedderFactory.create("sbert")
    vector_res = embedder.embed_batch(res.chunks)

    # 3. Додаємо в storage
    storage.add(vector_res)
    stats = storage.get_stats()
    print(f"✅ Вектори додано в індекс:")
    print(f"   Total vectors: {stats['total_vectors']}")

    # 4. Пошук
    query_vector = vector_res[0].vector  # Шукаємо перший вектор
    results = storage.search(query_vector, top_k=3)

    print(f"✅ Пошук виконано, знайдено {len(results)} результатів:")
    for i, result in enumerate(results):
        print(f"   {i+1}. Score: {result.score:.4f}, Chunk: {result.chunk_id}")
    print()

    # 5. Збереження
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_index"
        storage.save(str(save_path))
        print(f"✅ Індекс збережено: {save_path}\n")

        # 6. Завантаження
        new_storage = FAISSStorage(dimension=384)
        new_storage.load(str(save_path))
        print(f"✅ Індекс завантажено")
        print(f"   Stats: {new_storage.get_stats()}\n")

        # 7. Перевірка після завантаження
        results2 = new_storage.search(query_vector, top_k=3)
        print(f"✅ Пошук після завантаження:")
        for i, result in enumerate(results2):
            print(
                f"   {i+1}. Score: {result.score:.4f}, Chunk: {result.chunk_id}"
            )
        print()

        # Перевірка що результати однакові
        assert len(results) == len(
            results2), "Кількість результатів не співпадає!"
        for r1, r2 in zip(results, results2):
            assert r1.chunk_id == r2.chunk_id, "Chunk ID не співпадає!"
            assert abs(r1.score - r2.score) < 0.001, "Score не співпадає!"

        print("✅ Результати після завантаження співпадають!")

    print("\n🎉 Тест пройдено успішно!\n")


def test_incremental_adding():
    """Тест інкрементального додавання"""

    print("=" * 60)
    print("ТЕСТ 2: Інкрементальне додавання документів")
    print("=" * 60)

    storage = FAISSStorage(dimension=384)

    # Додаємо документи по черзі
    for doc_num in range(3):
        embeddings = []
        for i in range(5):
            vector = np.random.randn(384).tolist()
            emb = EmbedderResult(
                vector=vector,
                chunk_id=f"doc{doc_num}_chunk{i}",
                document_id=f"document_{doc_num}",
                metadata={"text": f"Doc {doc_num}, chunk {i}"})
            embeddings.append(emb)

        storage.add(embeddings)
        print(f"✅ Додано документ {doc_num}: {len(embeddings)} чанків")
        print(f"   Всього в індексі: {storage.get_stats()['total_vectors']}")

    print(
        f"\n✅ Всього в індексі: {storage.get_stats()['total_vectors']} векторів"
    )
    print("🎉 Тест пройдено!\n")


if __name__ == "__main__":
    test_storage_basic()
    test_incremental_adding()
