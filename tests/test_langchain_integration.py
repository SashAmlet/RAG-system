import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.langchain_orchestrator import LangChainOrchestrator


def test_ollama_connection():
    """Тест підключення до Ollama"""
    print("=" * 70)
    print("ТЕСТ 1: Підключення до Ollama")
    print("=" * 70)

    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="qwen2.5:7b")
    response = llm.invoke("Привіт! Напиши одне речення українською.")

    print(f"Відповідь: {response.content}")
    assert len(response.content) > 0
    print("✅ Ollama працює!\n")


def test_full_rag_pipeline():
    """Тест повного RAG pipeline"""
    print("=" * 70)
    print("ТЕСТ 2: Повний RAG Pipeline")
    print("=" * 70)

    # Створюємо orchestrator
    orchestrator = LangChainOrchestrator(
        index_path="data/indexes/test_langchain", model="qwen2.5:7b")

    # Індексуємо тестовий текст (створимо файл)
    test_file = Path("test_document.txt")
    test_content = """
    Машинне навчання — це підгалузь штучного інтелекту, яка дозволяє 
    комп'ютерам вчитися на даних без явного програмування.
    
    Python є найпопулярнішою мовою для data science та машинного навчання.
    Бібліотеки як NumPy, Pandas та Scikit-learn надають потужні інструменти.
    
    Трансформери революціонізували обробку природної мови завдяки механізму attention.
    """
    test_file.write_text(test_content, encoding='utf-8')

    try:
        # Індексація
        print("\n1️⃣ Індексація документа...")
        num_chunks = orchestrator.index_document(str(test_file))
        print(f"   ✅ Створено {num_chunks} чанків")

        # Запит
        print("\n2️⃣ Обробка запиту...")
        question = "Що таке машинне навчання?"
        response = orchestrator.query(question)

        print(f"\n📝 Запит: {question}")
        print(f"💡 Відповідь:\n{response['answer']}\n")

        assert response['success']
        assert len(response['answer']) > 0

        print("✅ RAG Pipeline працює!\n")

    finally:
        # Очищаємо
        test_file.unlink()
        import shutil
        if Path("data/indexes/test_langchain").exists():
            shutil.rmtree("data/indexes/test_langchain")


if __name__ == "__main__":
    test_ollama_connection()
    test_full_rag_pipeline()

    print("=" * 70)
    print("🎉 ВСІ ТЕСТИ ПРОЙДЕНО УСПІШНО!")
    print("=" * 70)
