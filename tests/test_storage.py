import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedder import EmbedderFactory
from src.preprocessing.preprocessor_factory import PreprocessorFactory
from src.storage.storage import FAISSStorage
import tempfile

from config.logging_config import configure_logging

configure_logging()


def recursive_files_scan(directory: Path):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_file():
            yield item
        elif item.is_dir():
            yield from recursive_files_scan(item)


def test_storage_basic():
    """Базовий тест: add, search, save, load"""

    print("=" * 60)
    print("ТЕСТ 1: Базовий функціонал Storage")
    print("=" * 60)

    # 1. Створюємо storage
    storage = FAISSStorage(dimension=384)
    print(f"✅ Storage створено: {storage.get_stats()}\n")

    prep = PreprocessorFactory.create(worker="minimal",
                                      default_parser="pdf_marker")
    embedder = EmbedderFactory.create("sbert")

    file_name = "JIRACORESERVER0"
    path = f"data\\raw\\{file_name}.pdf"

    folder = "python-3.13.8-docs-text"

    # for file in recursive_files_scan("data\\raw\\" + folder):
    # res = prep.process_document(file)
    # vector_res = embedder.embed_batch(res.chunks)
    # storage.add(vector_res, res.chunks)

    res = prep.process_document(path)
    vector_res = embedder.embed_batch(res.chunks)
    storage.add(vector_res, res.chunks)

    # 3. Додаємо в storage
    stats = storage.get_stats()
    print(f"✅ Вектори додано в індекс:")
    print(f"   Total vectors: {stats['total_vectors']}")

    # 4. Пошук
    query_vector = vector_res[0].vector  # Шукаємо перший вектор
    results = storage.search(query_vector, top_k=3)

    print(f"✅ Пошук виконано, знайдено {len(results)} результатів:")
    for i, result in enumerate(results):
        print(f"   {i+1}. Score: {result.score:.4f}, Chunk: {result.chunk_id}")
        print(f"       Text: {result.chunk.text[:80]}...")
    print()

    # 5. Збереження
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path("data\\processed\\") / "test_index"
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
            print(f"       Text: {result.chunk.text[:80]}...")
        print()

        # Перевірка що результати однакові
        assert len(results) == len(
            results2), "Кількість результатів не співпадає!"
        for r1, r2 in zip(results, results2):
            assert r1.chunk_id == r2.chunk_id, "Chunk ID не співпадає!"
            assert abs(r1.score - r2.score) < 0.001, "Score не співпадає!"

        print("✅ Результати після завантаження співпадають!")

    print("\n🎉 Тест пройдено успішно!\n")


if __name__ == "__main__":
    test_storage_basic()
