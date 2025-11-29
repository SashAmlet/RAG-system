# src/preprocessing/parsers/parser_factory.py

from typing import Optional, Dict, Type
from pathlib import Path
import logging

from src.preprocessing.parsers.document_parser import IDocumentParser
from src.preprocessing.parsers.pdf_parser import PDFParser
from src.preprocessing.parsers.marker_pdf_parser import MarkerPDFParser
from src.preprocessing.parsers.txt_parser import TXTParser

logger = logging.getLogger(__name__)


class ParserFactory:
    """Фабрика для автоматичного вибору парсера за типом файлу"""

    # Реєстр доступних парсерів
    _parsers: Dict[str, Type[IDocumentParser]] = {
        'pdf': PDFParser,
        'txt': TXTParser,
        'pdf_marker': MarkerPDFParser
    }

    @classmethod
    def create(cls, parser_type: str, **kwargs) -> IDocumentParser:
        """
        Створює парсер за типом.
        
        Args:
            parser_type: 'pdf', 'txt', 'docx', 'html', 'marker_pdf'
            **kwargs: Параметри для парсера
            
        Returns:
            Екземпляр IDocumentParser
        """
        parser_class = cls._parsers.get(parser_type.lower())

        if parser_class is None:
            raise ValueError(f"Unknown parser type: {parser_type}. "
                             f"Available: {list(cls._parsers.keys())}")

        return parser_class(**kwargs)

    @classmethod
    def auto_detect(cls, file_path: str | Path) -> IDocumentParser:
        """
        Автоматично визначає парсер за розширенням файлу.
        
        Args:
            filepath: Шлях до файлу
            
        Returns:
            Найкращий парсер для цього файлу
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        # Перебираємо всі парсери і шукаємо підходящий
        for parser_name, parser_class in cls._parsers.items():
            parser = parser_class()
            if parser.supports(str(file_path)):
                logger.info(
                    f"Auto-detected parser '{parser_name}' for {extension}")
                return parser

        raise ValueError(f"No parser found for file: {file_path}. "
                         f"Extension: {extension}")

    @classmethod
    def register(cls, name: str, parser_class: Type[IDocumentParser]):
        """
        Реєструє новий парсер.
        Дозволяє додавати custom парсери ззовні.
        """
        cls._parsers[name] = parser_class
        logger.info(f"Registered custom parser: {name}")
