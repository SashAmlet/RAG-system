from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple


class DocumentParser(ABC):
    """
    Базовий інтерфейс для парсерів документів.
    """

    # Розширення файлів, які підтримує парсер
    supported_extensions: Tuple[str, ...] = ()

    def supports(self, file_path: str | Path) -> bool:
        suffix = Path(file_path).suffix.lower()
        return suffix in self.supported_extensions

    @abstractmethod
    def parse(self, file_path: str | Path) -> str:
        """
        Повертає сирий текст із документа.
        """
        raise NotImplementedError
