"""
Тести для FastAPI сервера.
Запустіть сервер перед тестуванням: python server.py
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Тест health endpoint"""
    print("\n=== Testing /health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(
        f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}"
    )
    assert response.status_code == 200


def test_stats():
    """Тест stats endpoint"""
    print("\n=== Testing /stats ===")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(
        f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}"
    )


def test_query():
    """Тест query endpoint"""
    print("\n=== Testing /query ===")

    payload = {"question": "Що таке машинне навчання?"}

    response = requests.post(f"{BASE_URL}/query", json=payload)

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nПитання: {payload['question']}")
        print(f"\nВідповідь:\n{result.get('answer', 'N/A')}")
    else:
        print(f"Error: {response.text}")


def test_langserve_invoke():
    """Тест LangServe invoke endpoint"""
    print("\n=== Testing /rag-agent/invoke ===")

    payload = {
        "input": {
            "messages": [{
                "role": "user",
                "content": "Розкажи про Python"
            }]
        }
    }

    response = requests.post(f"{BASE_URL}/rag-agent/invoke", json=payload)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(
            f"Response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}..."
        )
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    print("🧪 Тестування RAG API Server")
    print("Переконайтесь що сервер запущено: python server.py\n")

    try:
        test_health()
        test_stats()
        test_query()
        test_langserve_invoke()

        print("\n✅ Всі тести пройдено!")
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
