from src.preprocessing.parsers.document_parser import IDocumentParser
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class TXTParser(IDocumentParser):
    """Парсер для звичайних текстових файлів"""

    def __init__(self, encoding: str = "utf-8"):
        """
        Args:
            encoding: Кодування файлу (utf-8, cp1251, etc.)
        """
        self.encoding = encoding

    def parse(self, file_path: str | Path) -> str:
        """Читає текстовий файл"""
        try:
            with open(file_path, "r", encoding=self.encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Спроба з іншим кодуванням
            logger.warning(f"Помилка з {self.encoding}, спроба cp1251")
            with open(file_path, "r", encoding="cp1251") as f:
                return f.read()

    def supports(self, file_path: str | Path) -> bool:
        """Підтримує .txt файли"""
        return Path(file_path).suffix.lower() in [".txt", ".text"]

    def get_supported_extensions(self) -> List[str]:
        return [".txt", ".text"]
