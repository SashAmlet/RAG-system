import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import TextChunk

from src.embeddings.embedder import EmbedderFactory

from src.storage.storage import FAISSStorage


def main():

    save_path = Path("data\\raw\\") / "test_index"
    embeder = EmbedderFactory.create("sbert")

    new_storage = FAISSStorage(dimension=384)
    new_storage.load(save_path)
    print(f"✅ Індекс завантажено")
    print(f"   Stats: {new_storage.get_stats()}\n")

    query = "What is Jira core?"
    query = TextChunk(text=query, chunk_id="query", document_id="query")
    query_vector = embeder.embed(query)
    results = new_storage.search(query_vector.vector, top_k=5)

    for i, result in enumerate(results):
        print(f"   {i+1}. Score: {result.score:.4f}, Chunk: {result.chunk_id}")
        print(f"       Text: {result.chunk.text[:80]}...")


if __name__ == "__main__":
    main()
