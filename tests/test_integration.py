import os
import sys
from pathlib import Path

# Додаємо src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedder import EmbedderFactory
from src.preprocessing.preprocessor import Preprocessor


def test_full_pipeline():
    """Тест повного pipeline: текст -> чанки -> вектори"""

    prep = Preprocessor()

    file_name = "introduction_to_microservices_galkin_shkilniak"
    path = f"data\\raw\\{file_name}.pdf"
    res = prep.process_document(path)

    for i, chunk in enumerate(res.chunks):
        print(f"   Chunk {i}: {len(chunk.text)} chars, ID={chunk.chunk_id}")
    print()

    # 2. Embedding
    embedder = EmbedderFactory.create("sbert")
    vector_res = embedder.embed_batch(res.chunks)

    print(f"Текст: {len(res.processed_text)} символів")
    print(f"Розмірність вектору: {len(vector_res[0].vector)}")
    print("Test Passed! ✅")


if __name__ == "__main__":
    test_full_pipeline()
