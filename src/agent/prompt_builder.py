from typing import List
from src.models import SearchResult


class PromptBuilder:
    """
    Створює промпти для різних типів запитів.
    """

    def __init__(self, language: str = "uk"):
        """
        Args:
            language: Мова відповідей ('uk', 'en')
        """
        self.language = language
        self.system_prompts = {
            "uk":
            """Ти — експертний асистент, який відповідає на запитання на основі наданих документів.

Правила:
- Давай точні та стислі відповіді українською мовою
- Використовуй ЛИШЕ інформацію з наданого контексту
- Якщо інформації недостатньо, чесно скажи про це
- Не вигадуй факти, яких немає в контексті
- Структуруй відповідь логічно та зрозуміло
- Посилайся на контекст через номери [1], [2] тощо"""
        }

    def build_qa_prompt(self, query: str,
                        search_results: List[SearchResult]) -> tuple[str, str]:
        """
        Створює промпт для question-answering.
        
        Args:
            query: Запит користувача
            search_results: Релевантні фрагменти
            
        Returns:
            Tuple (system_prompt, user_prompt)
        """
        # System prompt
        system_prompt = self.system_prompts.get(self.language,
                                                self.system_prompts["uk"])

        # Формуємо контекст
        context_parts = []
        for i, result in enumerate(search_results, 1):
            # Додаємо номер, score та текст
            context_parts.append(
                f"[{i}] (Релевантність: {result.score:.2f})\n{result.chunk.text.strip()}"
            )

        context_text = "\n\n".join(context_parts)

        # User prompt
        user_prompt = f"""Контекст з документів:
{context_text}

Запитання: {query}

Відповідь:"""

        return system_prompt, user_prompt

    def build_no_context_prompt(self, query: str) -> tuple[str, str]:
        """
        Промпт коли не знайдено релевантних документів.
        
        Returns:
            Tuple (system_prompt, user_prompt)
        """
        system_prompt = """Ти — чесний асистент. Якщо у тебе немає інформації в документах, скажи про це прямо."""

        user_prompt = f"""Запитання: {query}

Інформація: У наявних документах не знайдено інформації для відповіді на це запитання.

Відповідь:"""

        return system_prompt, user_prompt
