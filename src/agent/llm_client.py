from abc import ABC, abstractmethod
import logging
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Базовий інтерфейс для LLM клієнтів"""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> str:
        """Генерує відповідь від LLM"""
        pass


class PerplexityClient(LLMClient):
    """
    Клієнт для Perplexity API.
    Підтримує різні моделі Llama, Claude, GPT через Perplexity.
    """

    def __init__(self, api_key: str, model: str = "sonar", timeout: int = 30):
        """
        Args:
            api_key: Perplexity API ключ
            model: Назва моделі
                   - sonar (дешево, якісно)
                   - llama-3.1-sonar-large-128k-online (дорожче, краще)
                   - llama-3.1-sonar-huge-128k-online (найкраще)
            timeout: Таймаут запиту (секунди)
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.api_url = "https://api.perplexity.ai/chat/completions"

        logger.info(f"PerplexityClient ініціалізовано. Модель: {model}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> str:
        """
        Генерує відповідь через Perplexity API.

        Args:
            system_prompt: Системний промпт
            user_prompt: Промпт користувача
            temperature: Креативність (0-2, рекомендовано 0.1 для точності)
            max_tokens: Максимальна довжина відповіді

        Returns:
            Текст відповіді

        Raises:
            Exception: При помилках API
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "stream": False,
        }

        logger.info(f"Відправка запиту до Perplexity API (model={self.model})")
        start_time = time.time()

        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            answer = result["choices"][0]["message"]["content"]

            # Логуємо статистику
            usage = result.get("usage", {})
            duration = time.time() - start_time

            logger.info(
                f"Відповідь отримано за {duration:.2f}s. "
                f"Токени: {usage.get('total_tokens', 'N/A')}"
            )

            return answer.strip()

        except requests.exceptions.Timeout:
            logger.error(f"Таймаут запиту до Perplexity API ({self.timeout}s)")
            raise Exception("LLM API таймаут. Спробуйте ще раз.")

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP помилка: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Помилка LLM API: {e.response.status_code}")

        except Exception as e:
            logger.error(f"Несподівана помилка: {str(e)}")
            raise Exception(f"Помилка при зверненні до LLM: {str(e)}")


class OllamaClient(LLMClient):
    """
    Клієнт для Ollama (локальні моделі).
    Використовує langchain_ollama.
    """

    def __init__(self, model: str = "qwen2.5:7b", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.llm = ChatOllama(model=model, temperature=temperature)
        logger.info(f"OllamaClient ініціалізовано. Модель: {model}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> str:
        """
        Генерує відповідь через Ollama.
        """
        logger.info(f"Відправка запиту до Ollama (model={self.model})")
        start_time = time.time()

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            # Оновлюємо температуру якщо змінилась
            if temperature != self.temperature:
                self.llm.temperature = temperature

            response = self.llm.invoke(messages)
            answer = response.content

            duration = time.time() - start_time
            logger.info(f"Відповідь отримано за {duration:.2f}s")

            return answer.strip()

        except Exception as e:
            logger.error(f"Помилка Ollama: {str(e)}")
            raise Exception(f"Помилка при зверненні до Ollama: {str(e)}")


class LLMClientFactory:
    """Фабрика для створення LLM клієнтів"""

    @staticmethod
    def create(provider: str, **kwargs) -> LLMClient:
        """
        Створює LLM клієнт.

        Args:
            provider: 'perplexity', 'openai', 'ollama'
            **kwargs: Параметри для клієнта

        Returns:
            Екземпляр LLMClient
        """
        provider = provider.lower()

        if provider == "perplexity":
            return PerplexityClient(
                api_key=kwargs["api_key"],
                model=kwargs.get("model", "sonar"),
                timeout=kwargs.get("timeout", 30),
            )
        elif provider == "ollama":
            return OllamaClient(
                model=kwargs.get("model", "qwen2.5:7b"),
                temperature=kwargs.get("temperature", 0.1),
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
