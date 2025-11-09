from typing import List, Optional, Any
from pathlib import Path
import logging
import asyncio

from src.preprocessing.worker import *
from src.preprocessing.pdf_parser import PDFParser, PdfParserConfig
from src.models import ProcessorResult

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Клас Preprocessor виконує послідовну обробку тексту.
    """

    def __init__(self,
                 workers: Optional[List[Worker]] = None,
                 marker_pdf_config: Optional[Any] = None,
                 pdf_config: Optional[PdfParserConfig] = None,
                 use_marker_by_default: bool = False):

        # Ініціалізація MarkerPDFParser створимо при першому запиті.
        self._marker_pdf_config = marker_pdf_config
        self._marker_pdf_parser = None

        # Ініціалізуємо звичайний PDF-парсер одразу (швидший)
        self.simple_pdf_parser = PDFParser(pdf_config)
        self.use_marker_by_default = use_marker_by_default

        if workers is None:
            # Default pipeline: normalize unicode, fix hyphenation, clean text,
            # fix paragraphing and remove obvious page numbers.
            self.workers = [
                UnicodeNormalizer(),
                FixHyphenUk(),
                TextCleaner(preserve_tables=True, preserve_math=True),
                ParagraphFixer(),
                EscapeFixer(),
                RemovePageNumbers(aggressive=False),
                SingleLineifier()
            ]
        else:
            self.workers = workers

    def add_worker(self, worker: Worker) -> None:
        """Додає нового воркера до списку обробників"""
        self.workers.append(worker)
        logger.info(f"Додано воркер: {worker.__class__.__name__}")

    def remove_worker(self, worker_class: type) -> bool:
        """Видаляє воркер за типом класу"""
        for i, worker in enumerate(self.workers):
            if isinstance(worker, worker_class):
                removed = self.workers.pop(i)
                logger.info(f"Видалено воркер: {removed.__class__.__name__}")
                return True
        return False

    def process_text(self, text: str) -> str:
        """Послідовно пропускає текст через усі воркери"""
        logger.info(
            f"Початок обробки тексту через {len(self.workers)} воркерів")

        for i, worker in enumerate(self.workers):
            logger.debug(
                f"Обробка воркером {i+1}/{len(self.workers)}: {worker.__class__.__name__}"
            )
            text = worker.process(text)

        logger.info("Обробка тексту завершена")
        return text

    async def process_text_async(self, text: str) -> str:
        """Асинхронна версія process_text, використовує worker.process_async."""
        logger.info(
            f"(async) Початок обробки тексту через {len(self.workers)} воркерів"
        )

        for i, worker in enumerate(self.workers):
            logger.debug(
                f"(async) Обробка воркером {i+1}/{len(self.workers)}: {worker.__class__.__name__}"
            )
            # Виклик асинхронного wrapper-а кожного воркера
            text = await worker.process_async(text)

        logger.info("(async) Обробка тексту завершена")
        return text

    def _ensure_marker_parser(self):
        """Імпортує і створює MarkerPDFParser при першому використанні.

        Якщо пакет `marker-pdf` відсутній або імпорт призводить до помилки,
        ця функція підніме ImportError/Exception згідно з поведінкою модуля.
        """
        if self._marker_pdf_parser is None:
            try:
                # Локальний імпорт, щоб уникнути важкої ініціалізації на імпорті модуля
                from src.preprocessing.marker_pdf_parser import MarkerPDFParser

                logger.info("Лінива ініціалізація MarkerPDFParser...")
                self._marker_pdf_parser = MarkerPDFParser(
                    self._marker_pdf_config)  # type: ignore
                logger.info("MarkerPDFParser успішно ініціалізовано (lazy)")
            except Exception as e:
                logger.error(f"Не вдалося ініціалізувати MarkerPDFParser: {e}")
                raise

    def process_document(self,
                         file_path: str,
                         use_marker: Optional[bool] = None) -> ProcessorResult:
        """
        Парсить документ і обробляє через воркерів.

        Args:
            file_path: Шлях до файлу
            use_marker: Чи використовувати MarkerPDFParser. Якщо None, використовується значення за замовчуванням
        """
        logger.info(f"Початок обробки документа: {file_path}")

        # Визначаємо, який парсер використовувати
        should_use_marker = use_marker if use_marker is not None else self.use_marker_by_default
        if should_use_marker:
            # Ініціалізуємо marker-парсер тільки при потребі
            self._ensure_marker_parser()
            parser = self._marker_pdf_parser
            if parser is None:
                # Явна помилка коли marker-парсер не доступний після спроби ініціалізації
                raise RuntimeError("MarkerPDFParser is not available")
        else:
            parser = self.simple_pdf_parser

        try:
            raw_text = parser.parse(file_path)

            parser_type = "MarkerPDFParser" if should_use_marker else "PDFParser"
            logger.info(
                f"PDF парсинг завершено ({parser_type}). Отримано {len(raw_text)} символів"
            )

            # Обробляємо текст через воркерів
            processed_text = self.process_text(raw_text)

            # Отримуємо метадані
            if should_use_marker and self._marker_pdf_parser is not None:
                pdf_metadata = self._marker_pdf_parser.get_metadata()
            else:
                pdf_metadata = {}

            # Зберігаємо оброблений текст

            # Створюємо результат
            result = ProcessorResult(
                processed_text=processed_text,
                metadata={
                    "source":
                    file_path,
                    "pdf_metadata":
                    pdf_metadata,
                    "parser_type":
                    parser_type,
                    "original_text_length":
                    len(raw_text),
                    "processed_text_length":
                    len(processed_text),
                    "workers_used":
                    [worker.__class__.__name__ for worker in self.workers],
                    "formulas_supported":
                    should_use_marker
                })

            logger.info(
                f"Обробка документа завершена. Фінальний текст: {len(processed_text)} символів"
            )

            return result

        except Exception as e:
            logger.error(f"Помилка обробки документа {file_path}: {str(e)}")
            raise

    async def process_document_async(
            self,
            file_path: str,
            use_marker: Optional[bool] = None) -> ProcessorResult:
        """
        Асинхронна версія process_document.

        Використовує asyncio.to_thread для блокуючих операцій (парсинг, запис файлу),
        та викликає асинхронну обробку воркерів.
        """
        logger.info(f"(async) Початок обробки документа: {file_path}")

        # Визначаємо, який парсер використовувати
        should_use_marker = use_marker if use_marker is not None else self.use_marker_by_default
        if should_use_marker:
            # Ініціалізуємо marker-парсер тільки при потребі
            self._ensure_marker_parser()
            parser = self._marker_pdf_parser
            if parser is None:
                # Явна помилка коли marker-парсер не доступний після спроби ініціалізації
                raise RuntimeError("MarkerPDFParser is not available")
        else:
            parser = self.simple_pdf_parser

        try:
            # Викликаємо блокуючий парсер у threadpool
            raw_text = await asyncio.to_thread(parser.parse, file_path)

            parser_type = "MarkerPDFParser" if should_use_marker else "PDFParser"
            logger.info(
                f"(async) PDF парсинг завершено ({parser_type}). Отримано {len(raw_text)} символів"
            )

            # Асинхронна обробка тексту через воркерів
            processed_text = await self.process_text_async(raw_text)

            # Отримуємо метадані (якщо використовується marker)
            if should_use_marker and self._marker_pdf_parser is not None:
                pdf_metadata = self._marker_pdf_parser.get_metadata()
            else:
                pdf_metadata = {}

            # Створюємо результат
            result = ProcessorResult(
                processed_text=processed_text,
                metadata={
                    "source":
                    file_path,
                    "pdf_metadata":
                    pdf_metadata,
                    "parser_type":
                    parser_type,
                    "original_text_length":
                    len(raw_text),
                    "processed_text_length":
                    len(processed_text),
                    "workers_used":
                    [worker.__class__.__name__ for worker in self.workers],
                    "formulas_supported":
                    should_use_marker
                })

            logger.info(
                f"(async) Обробка документа завершена. Фінальний текст: {len(processed_text)} символів"
            )

            return result

        except Exception as e:
            logger.error(
                f"(async) Помилка обробки документа {file_path}: {str(e)}")
            raise

    def save_document(self, file_path: str, content: str) -> None:
        """Зберігає оброблений текст у файл."""
        try:
            file_stem = Path(file_path).stem
            out_dir = Path(f"data/processed/{file_stem}")
            out_file = out_dir / 'processed1.txt'

            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_file, "w", encoding="utf-8") as doc:
                doc.write(content)

            logger.info(f"Збережено оброблений текст у: {out_file}")
        except Exception as e:
            logger.warning(
                f"Не вдалося автоматично зберегти оброблений текст: {e}")

    async def save_document_async(self, file_path: str, content: str) -> None:
        """Асинхронно зберігає оброблений текст у файл."""
        try:
            file_stem = Path(file_path).stem
            out_dir = Path(f"data/processed/{file_stem}")
            out_file = out_dir / 'processed1.txt'

            out_dir.mkdir(parents=True, exist_ok=True)

            # Файлові операції в треді
            async def write_file(path: Path, content: str):
                await asyncio.to_thread(path.write_text, content, "utf-8")

            await write_file(out_file, content)
            logger.info(f"(async) Збережено оброблений текст у: {out_file}")
        except Exception as e:
            logger.warning(
                f"(async) Не вдалося автоматично зберегти оброблений текст: {e}"
            )

    def set_marker_pdf_config(self, config: Optional[object]) -> None:
        """Оновлює конфігурацію Marker PDF парсера.

        Замість негайної ініціалізації парсера, оновлюємо збережену конфіг
        і чистимо існуючий екземпляр, щоб він був створений ліниво при наступному
        виклику, якщо це потрібно.
        """
        self._marker_pdf_config = config
        # Скидаємо існуючий екземпляр — він буде створений під час першого використання
        self._marker_pdf_parser = None
        logger.info("Конфігурація Marker PDF парсера оновлена (lazy)")

    def set_pdf_config(self, config: PdfParserConfig) -> None:
        """Оновлює конфігурацію звичайного PDF парсера"""
        self.simple_pdf_parser = PDFParser(config)
        logger.info("Конфігурація звичайного PDF парсера оновлена")

    def get_supported_formats(self) -> List[str]:
        """Повертає список підтримуваних форматів файлів"""
        return ['.pdf']

    def get_pdf_metadata(self) -> dict:
        """Повертає метадані активного PDF парсера.

        Якщо використовується marker за замовчуванням, намагаємось ліниво
        ініціалізувати парсер і повернути його метадані. Якщо ініціалізація
        не вдається, повертаємо порожній словник.
        """
        if self.use_marker_by_default:
            try:
                self._ensure_marker_parser()
                if self._marker_pdf_parser is not None:
                    return self._marker_pdf_parser.get_metadata()
            except Exception:
                logger.warning(
                    "Marker parser unavailable when fetching metadata")
        return {}
