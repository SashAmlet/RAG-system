import os
import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from pathlib import Path

from src.preprocessing.preprocessor_factory import PreprocessorFactory
from src.embeddings.embedder import EmbedderFactory
from src.storage.storage import FAISSStorage
from src.agent.agent import AIAgent
from src.agent.llm_client import LLMClientFactory

# Завантажуємо змінні середовища
load_dotenv()

console = Console()


def index_documents(preprocessor, embedder, storage, data_dir="data/raw"):
    """Індексує всі документи з директорії"""
    console.print(f"\n[bold blue]📚 Індексація документів з {data_dir}[/bold blue]\n")

    data_path = Path(data_dir)
    if not data_path.exists():
        console.print("[red]Директорія не знайдена![/red]")
        return

    files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.txt"))

    if not files:
        console.print("[yellow]Документи не знайдено![/yellow]")
        return

    all_chunks = []

    for file_path in files:
        console.print(f"📄 Обробка: {file_path.name}")

        # Обробка документа
        result = preprocessor.process_document(
            str(file_path),
            enable_chunking=True,
            chunk_size=int(os.getenv("CHUNK_SIZE", 800)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 150)),
        )

        console.print(f"   ✅ Створено {len(result.chunks)} чанків")
        all_chunks.extend(result.chunks)

    # Векторизація
    console.print(f"\n🔢 Векторизація {len(all_chunks)} чанків...")
    embeddings = embedder.embed_batch(all_chunks)

    # Збереження в storage
    console.print("💾 Збереження в векторну БД...")
    storage.add(embeddings, all_chunks)
    storage.save("data/indexes/knowledge_base")

    stats = storage.get_stats()
    console.print(f"\n[bold green]✅ Індексація завершена![/bold green]")
    console.print(f"   Векторів: {stats['total_vectors']}")
    console.print(f"   Документів: {stats['unique_documents']}\n")


def query_mode(agent):
    """Режим одиночного запиту"""
    question = input("\n💬 Ваше запитання: ")

    console.print("\n[yellow]🤔 Обробка...[/yellow]\n")
    response = agent.answer(question)

    # Виводимо відповідь
    console.print("[bold green]💡 Відповідь:[/bold green]")
    console.print(Markdown(response.answer))

    # Виводимо джерела
    console.print(f"\n[bold blue]📚 Джерела ({len(response.sources)}):[/bold blue]")
    for i, src in enumerate(response.sources, 1):
        console.print(f"[cyan]{i}.[/cyan] Score: {src.score:.3f}")
        console.print(f"   {src.chunk.text[:100]}...\n")

    # Метадані
    console.print(f"[dim]⏱️  Час: {response.metadata.get('duration_seconds')}s[/dim]")


def interactive_mode(agent):
    """Інтерактивний режим діалогу"""
    console.print("\n[bold green]🤖 Інтерактивний режим[/bold green]")
    console.print("[dim]Введіть 'exit' або 'quit' для виходу[/dim]\n")

    while True:
        try:
            question = input("💬 Вы: ")

            if question.lower() in ["exit", "quit", "вихід"]:
                console.print("\n[yellow]👋 До побачення![/yellow]")
                break

            if not question.strip():
                continue

            console.print("\n[yellow]🤔 Обробка...[/yellow]\n")
            response = agent.answer(question)

            console.print("[bold green]🤖 Асистент:[/bold green]")
            console.print(Markdown(response.answer))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]👋 До побачення![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Помилка: {e}[/red]\n")


def main():
    parser = argparse.ArgumentParser(description="RAG System")
    parser.add_argument(
        "--mode",
        choices=["index", "query", "interactive"],
        default="interactive",
        help="Режим роботи",
    )
    parser.add_argument(
        "--data-dir", default="data/raw", help="Директорія з документами"
    )
    parser.add_argument("--question", help="Запитання (для mode=query)")

    args = parser.parse_args()

    console.print("[bold blue]🚀 RAG System[/bold blue]\n")

    # Ініціалізація компонентів
    console.print("⚙️  Ініціалізація компонентів...")

    preprocessor = PreprocessorFactory.create(worker="minimal", default_parser="auto")

    embedder = EmbedderFactory.create(
        method="sbert",
        model_name=os.getenv(
            "EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        batch_size=int(os.getenv("EMBEDDER_BATCH_SIZE", 32)),
    )

    storage = FAISSStorage(dimension=384)

    if args.mode == "index":
        # Режим індексації
        index_documents(preprocessor, embedder, storage, args.data_dir)

    else:
        # Завантажуємо існуючий індекс
        index_path = "data/indexes/knowledge_base"
        if not Path(f"{index_path}.faiss").exists():
            console.print(
                "[red]❌ Індекс не знайдено! Спочатку запустіть --mode index[/red]"
            )
            return

        console.print("💾 Завантаження індексу...")
        storage.load(index_path)

        # Створюємо LLM client
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            console.print("[red]❌ PERPLEXITY_API_KEY не встановлено![/red]")
            return

        llm_client = LLMClientFactory.create(
            provider=os.getenv("LLM_PROVIDER", "perplexity"),
            api_key=api_key,
            model=os.getenv("LLM_MODEL", "sonar"),
        )

        # Створюємо AI Agent
        agent = AIAgent(
            storage=storage,
            embedder=embedder,
            llm_client=llm_client,
            top_k=int(os.getenv("TOP_K", 4)),
            min_similarity=float(os.getenv("MIN_SIMILARITY", 0.3)),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 800)),
            language="uk",
        )

        console.print("[green]✅ Система готова![/green]\n")

        if args.mode == "query":
            # Режим одиночного запиту
            if not args.question:
                args.question = input("💬 Ваше запитання: ")
            query_mode(agent)
        else:
            # Інтерактивний режим
            interactive_mode(agent)


if __name__ == "__main__":
    main()
