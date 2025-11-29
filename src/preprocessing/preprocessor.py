# src/preprocessing/preprocessor.py

from typing import List, Optional, Union
from pathlib import Path
import logging

from src.preprocessing.worker import Worker
from src.preprocessing.parsers.parser_factory import ParserFactory
from src.preprocessing.parsers.document_parser import IDocumentParser
from src.models import ProcessorResult

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocessor з підтримкою різних форматів документів.
    """

    def __init__(
            self,
            workers: Optional[List[Worker]] = None,
            default_parser: str = "auto"  # 'pdf', 'txt', 'auto'
    ):
        """
        Args:
            workers: Список workers для обробки тексту
            default_parser: Дефолтний парсер або 'auto' для автовизначення
        """
        self.default_parser = default_parser

        # Workers pipeline
        if workers is None:
            # Дефолтний pipeline
            from src.preprocessing.worker import (UnicodeNormalizer,
                                                  FixHyphenUk, TextCleaner,
                                                  ParagraphFixer, EscapeFixer,
                                                  RemovePageNumbers,
                                                  SingleLineifier)
            self.workers = [
                UnicodeNormalizer(),
                FixHyphenUk(),
                TextCleaner(),
                ParagraphFixer(),
                EscapeFixer(),
                RemovePageNumbers(),
                SingleLineifier()
            ]
        else:
            self.workers = workers

    def process_document(self,
                         file_path: str,
                         parser: Optional[Union[str, IDocumentParser]] = None,
                         enable_chunking: bool = True,
                         **chunking_kwargs) -> ProcessorResult:
        """
        Обробляє документ будь-якого підтримуваного формату.
        
        Args:
            file_path: Шлях до файлу
            parser: Парсер ('pdf', 'txt', 'auto') або екземпляр IDocumentParser
            enable_chunking: Чи розбивати на чанки
            **chunking_kwargs: Параметри для chunking
            
        Returns:
            ProcessorResult з обробленим текстом та чанками
        """
        logger.info(f"Обробка документа: {file_path}")

        # 1. Визначаємо парсер
        if parser is None:
            parser = self.default_parser

        if isinstance(parser, str):
            if parser == "auto":
                parser_instance = ParserFactory.auto_detect(file_path)
            else:
                parser_instance = ParserFactory.create(parser)
        else:
            parser_instance = parser

        # 2. Парсимо документ
        logger.info(
            f"Використовується парсер: {parser_instance.__class__.__name__}")
        raw_text = parser_instance.parse(file_path)
        logger.info(f"Витягнуто {len(raw_text)} символів")

        # 3. Обробка через workers
        processed_text = raw_text
        applied_workers = []

        for worker in self.workers:
            try:
                processed_text = worker.process(processed_text)
                applied_workers.append(worker.__class__.__name__)
            except Exception as e:
                logger.error(
                    f"Помилка в worker {worker.__class__.__name__}: {e}")

        # 4. Створюємо результат
        result = ProcessorResult(processed_text=processed_text,
                                 original_filename=Path(file_path).name,
                                 metadata={
                                     "file_size":
                                     Path(file_path).stat().st_size,
                                     "parser":
                                     parser_instance.__class__.__name__
                                 },
                                 processing_info={
                                     "text_length": len(processed_text),
                                     "cleaning_steps_applied": applied_workers
                                 })

        # 5. Chunking
        if enable_chunking:
            from src.preprocessing.chunker import ChunkerFactory, ChunkingConfig

            chunking_strategy = chunking_kwargs.pop('chunking_strategy',
                                                    'semantic')
            config = ChunkingConfig(
                chunk_size=chunking_kwargs.get('chunk_size', 800),
                chunk_overlap=chunking_kwargs.get('chunk_overlap', 150))

            chunker = ChunkerFactory.create(chunking_strategy, config)
            result.chunks = chunker.chunk(processed_text, result.document_id)
            logger.info(f"Створено {len(result.chunks)} чанків")

        return result
