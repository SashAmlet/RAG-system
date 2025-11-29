from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging
from .document_parser import IDocumentParser

logger = logging.getLogger(__name__)


@dataclass
class MarkerPdfParserConfig:
    """
    Налаштування парсера PDF з marker-pdf.
    
    - max_pages: максимум сторінок для парсингу (None = всі).
    - page_separator: роздільник між сторінками (None щоб не додавати).
    - extract_images: чи витягувати зображення окремо.
    """
    max_pages: Optional[int] = None
    page_separator: Optional[str] = None
    extract_images: bool = True
    device: str = 'auto'
    processed_dir: str = "data/processed"
    images_subdir: str = "images"
    processed_text_filename: str = "processed.txt"


class MarkerPDFParser(IDocumentParser):
    """
    Парсер PDF-документів на базі marker-pdf з підтримкою формул.
    Автоматично розпізнає формули та конвертує їх у LaTeX.
    """

    supported_extensions = (".pdf", )

    def __init__(self, config: Optional[MarkerPdfParserConfig] = None) -> None:
        self.config = config or MarkerPdfParserConfig()

        # Перевіряємо наявність бібліотеки лише при ініціалізації класу
        try:
            global pdf, create_model_dict, text_from_rendered
            import marker.converters.pdf as pdf
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
        except ImportError:
            raise ImportError("Бібліотека 'marker-pdf' не встановлена. "
                              "Встановіть її командою: pip install marker-pdf")

        logger.info("Ініціалізація Marker PDF converter...")
        try:
            self.converter = pdf.PdfConverter(
                artifact_dict=create_model_dict())
            logger.info("Marker PDF converter успішно ініціалізовано")
        except Exception as e:
            logger.error(f"Помилка ініціалізації Marker: {e}")
            raise

    def parse(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not self.supports(path):
            raise ValueError(
                f"Unsupported file extension for MarkerPDFParser: {path.suffix}"
            )

        logger.info(f"Початок парсингу PDF з marker-pdf: {path}")

        try:
            # Основна обробка через marker
            logger.debug("Запуск marker converter...")
            rendered = self.converter(str(path))

            logger.debug("Вилучення тексту та зображень...")
            text, _, images = text_from_rendered(rendered)

            # Обробка відповідно до конфігурації
            # Передаємо шлях файлу, щоб зберігати зображення поряд із файлом
            processed_text = self._process_text_output(text, images, path)

            logger.info(
                f"PDF парсинг завершено. Отримано {len(processed_text)} символів"
            )
            return processed_text

        except Exception as e:
            logger.error(f"Помилка парсингу PDF {path}: {e}")
            raise

    def _process_text_output(self, text: str, images: dict,
                             source_path: Path) -> str:
        """
        Обробляє вихідний текст відповідно до конфігурації
        """
        # Основний текст (вже містить формули в LaTeX)
        text_lines = text.split('\n')

        # Обмеження по сторінкам (приблизно)
        if self.config.max_pages:
            max_lines = self.config.max_pages * 50
            text_lines = text_lines[:max_lines]

        # Додаємо роздільники сторінок (якщо потрібно)
        if self.config.page_separator:
            page_size = 50
            processed_lines = []

            for i, line in enumerate(text_lines):
                processed_lines.append(line)

                if (i + 1) % page_size == 0 and i < len(text_lines) - 1:
                    processed_lines.append(self.config.page_separator)

            text = '\n'.join(processed_lines)
        else:
            text = '\n'.join(text_lines)

        # Зберігаємо інформацію про зображення
        if self.config.extract_images and images:
            self._save_images_info(images, source_path)

        return text

    def _save_images_info(self, images: dict, source_path: Path) -> None:
        """Зберігає зображення в папку поруч з `source_path`.

        Папка: <same_dir>/<source_stem>_images
        """
        try:
            from PIL import Image
            PIL_AVAILABLE = True
        except Exception:
            Image = None  # type: ignore
            PIL_AVAILABLE = False

        logger.info(
            f"Знайдено {len(images)} зображень. Збереження у папку: {self.config.processed_dir}/{source_path.stem}/{self.config.images_subdir}"
        )

        # Папка: <processed_dir>/<file_stem>/<images_subdir>
        output_dir = Path(self.config.processed_dir
                          ) / source_path.stem / self.config.images_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        # Normalize items: dict->items(), list/iterable->enumerated (string keys)
        if isinstance(images, dict):
            items = list(images.items())
        else:
            items = [(str(i), v) for i, v in enumerate(images)]

        for idx, (name, val) in enumerate(items):
            # Compose a safe filename base
            try:
                key_name = str(name)
            except Exception:
                key_name = f"img{idx}"

            file_bytes = None
            ext = None

            safe_name = f"{key_name}_{idx}.png"
            out_path = output_dir / safe_name
            try:
                val.save(out_path)
                saved.append(str(out_path))
                logger.debug(f"Збережено (PIL): {out_path}")
                continue
            except Exception as e:
                logger.warning(
                    f"Не вдалося зберегти PIL image {key_name}: {e}")

            # fallback: try to convert to bytes
            if file_bytes:
                # ensure bytes-like
                if not isinstance(file_bytes, (bytes, bytearray)):
                    logger.warning(
                        f"Не вдалося перетворити в bytes зображення '{key_name}' (index {idx}), пропускаємо"
                    )
                    continue

                if not ext:
                    ext = "png"

                safe_name = f"{key_name}_{idx}.{ext}"
                out_path = output_dir / safe_name
                try:
                    with open(out_path, "wb") as fh:
                        fh.write(file_bytes)
                    saved.append(str(out_path))
                    logger.debug(f"Збережено: {out_path}")
                except Exception as e:
                    logger.warning(f"Не вдалося записати файл {out_path}: {e}")
            else:
                logger.warning(
                    f"Не вдалося розпізнати формат зображення '{key_name}' (index {idx}), пропускаємо"
                )

        logger.info(f"Збережено {len(saved)} зображень у: {output_dir}")

    def get_metadata(self) -> dict:
        """Повертає метадані про парсер"""
        return {
            "parser_type": "marker-pdf",
            "supports_formulas": True,
            "supports_tables": True,
            "supports_images": True,
            "config": {
                "max_pages": self.config.max_pages,
                "page_separator": self.config.page_separator,
                "extract_images": self.config.extract_images,
                "device": self.config.device,
                "processed_dir": self.config.processed_dir,
                "images_subdir": self.config.images_subdir,
                "processed_text_filename": self.config.processed_text_filename
            }
        }

    def supports(self, file_path: str | Path) -> bool:
        """Перевіряє чи це PDF"""
        return Path(file_path).suffix.lower() == '.pdf'

    def get_supported_extensions(self) -> List[str]:
        return ['.pdf']
