import argparse
import logging
from pathlib import Path

from src.preprocessing.preprocessor import Preprocessor
from src.embeddings.embedder import EmbedderFactory
from src.storage.storage import FAISSStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def index_documents(input_path: str, output_path: str):
    """Індексація документів"""
    logger.info(f"Початок індексації документів з {input_path}")

    # Ініціалізація компонентів
    preprocessor = Preprocessor()
    embedder = EmbedderFactory.create(method="sbert")
    storage = FAISSStorage(dimension=384)

    # Обробка документів
    pdf_files = list(Path(input_path).glob("*.pdf"))
    logger.info(f"Знайдено {len(pdf_files)} PDF файлів")

    for pdf_file in pdf_files:
        try:
            logger.info(f"Обробка: {pdf_file.name}")

            # Preprocessor + Chunking
            result = preprocessor.process_document(file_path=str(pdf_file),
                                                   enable_chunking=True,
                                                   chunk_size=500,
                                                   chunk_overlap=100)

            if not result.chunks:
                logger.warning(f"Немає чанків для {pdf_file.name}")
                continue

            # Embedding
            embeddings = embedder.embed_batch(result.chunks)

            # Storage
            storage.add(embeddings)

            logger.info(f"✅ Оброблено: {len(result.chunks)} чанків")

        except Exception as e:
            logger.error(f"Помилка обробки {pdf_file.name}: {e}")

    # Збереження індексу
    storage.save(output_path)
    stats = storage.get_stats()
    logger.info(f"📊 Індекс збережено: {stats}")


def interactive_mode(index_path: str):
    """Інтерактивний режим запитів"""
    logger.info("Запуск інтерактивного режиму")

    # Завантаження індексу
    embedder = EmbedderFactory.create(method="sbert")
    storage = FAISSStorage()
    storage.load(index_path)

    stats = storage.get_stats()
    logger.info(f"📊 Завантажено індекс: {stats}")

    print("\n" + "=" * 60)
    print("RAG СИСТЕМА - ІНТЕРАКТИВНИЙ РЕЖИМ")
    print("=" * 60)
    print("Введіть запитання (або 'exit' для виходу)\n")

    from src.models import TextChunk

    while True:
        query = input("Ваш запит: ").strip()

        if query.lower() in ['exit', 'quit', 'вихід']:
            break

        if not query:
            continue

        try:
            # Векторизація запиту
            query_chunk = TextChunk(text=query,
                                    chunk_id="query",
                                    document_id="query")
            query_embedding = embedder.embed(query_chunk)

            # Пошук
            results = storage.search(query_embedding.vector, top_k=3)

            print(f"\n🔍 Знайдено {len(results)} релевантних фрагментів:\n")
            for i, result in enumerate(results):
                print(f"{i+1}. Score: {result.score:.4f}")
                print(f"   {result.chunk.text[:200]}...")
                print(f"   (Документ: {result.document_id})\n")

        except Exception as e:
            logger.error(f"Помилка: {e}")

    print("До побачення!")


def main():
    parser = argparse.ArgumentParser(description="RAG System")
    parser.add_argument("--mode",
                        choices=["index", "interactive"],
                        default="interactive",
                        help="Режим роботи")
    parser.add_argument("--input",
                        default="./data/documents",
                        help="Шлях до PDF документів")
    parser.add_argument("--index-path",
                        default="./data/indexes/knowledge_base",
                        help="Шлях до індексу")

    args = parser.parse_args()

    if args.mode == "index":
        index_documents(args.input, args.index_path)
    elif args.mode == "interactive":
        interactive_mode(args.index_path)


if __name__ == "__main__":
    main()
