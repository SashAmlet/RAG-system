from abc import ABC, abstractmethod
from typing import List
from pathlib import Path


class IDocumentParser(ABC):
    """Базовий інтерфейс для всіх парсерів документів"""

    @abstractmethod
    def parse(self, file_path: str | Path) -> str:
        """
        Парсить документ і повертає текст.
        
        Args:
            file_path: Шлях до файлу
            
        Returns:
            Витягнутий текст
        """
        pass

    @abstractmethod
    def supports(self, file_path: str | Path) -> bool:
        """
        Перевіряє чи парсер підтримує даний формат.
        
        Args:
            file_path: Шлях до файлу
            
        Returns:
            True якщо підтримує
        """
        pass

    def get_supported_extensions(self) -> List[str]:
        """Повертає список підтримуваних розширень"""
        return []
