from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
# import base64
from typing import Any

from .document_parser import DocumentParser

# Імпорти для marker-pdf
try:
    import marker.converters.pdf as pdf
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError as e:
    MARKER_AVAILABLE = False
    raise e

logger = logging.getLogger(__name__)


@dataclass
class MarkerPdfParserConfig:
    """
    Налаштування парсера PDF з marker-pdf.
    
    - max_pages: максимум сторінок для парсингу (None = всі).
    - page_separator: роздільник між сторінками (None щоб не додавати).
    - extract_images: чи витягувати зображення окремо.
    - device: пристрій для обчислень ('auto', 'cpu', 'cuda', 'mps').
    """
    max_pages: Optional[int] = None
    page_separator: Optional[str] = None
    extract_images: bool = True
    device: str = 'auto'
    processed_dir: str = "data/processed"
    images_subdir: str = "images"
    processed_text_filename: str = "processed.txt"


class MarkerPDFParser(DocumentParser):
    """
    Парсер PDF-документів на базі marker-pdf з підтримкою формул.
    Автоматично розпізнає формули та конвертує їх у LaTeX.
    """

    supported_extensions = (".pdf", )

    def __init__(self, config: Optional[MarkerPdfParserConfig] = None) -> None:
        if not MARKER_AVAILABLE:
            raise ImportError(
                "marker-pdf не встановлено. Встановіть: pip install marker-pdf"
            )

        self.config = config or MarkerPdfParserConfig()

        # Ініціалізуємо converter один раз для оптимізації
        logger.info("Ініціалізація Marker PDF converter...")
        try:
            self.converter = pdf.PdfConverter(
                artifact_dict=create_model_dict(), )
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
            # Приблизна оцінка: ~50 рядків на сторінку
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

        # def detect_ext_from_bytes(b: Any) -> Optional[str]:
        #     """Try to detect image extension from bytes using Pillow or simple signatures."""
        #     # Coerce to bytes if possible
        #     try:
        #         if not isinstance(b, (bytes, bytearray)):
        #             b = bytes(b)
        #     except Exception:
        #         return None

        #     # Try Pillow if available
        #     if PIL_AVAILABLE:
        #         try:
        #             from io import BytesIO
        #             fmt = Image.open(BytesIO(b)).format  # type: ignore[attr-defined]
        #             if fmt:
        #                 fmt = fmt.lower()
        #                 if fmt == 'jpeg':
        #                     return 'jpg'
        #                 return fmt
        #         except Exception:
        #             pass

        #     # Basic signature checks
        #     if b.startswith(b"\xff\xd8\xff"):
        #         return 'jpg'
        #     if b.startswith(b"\x89PNG\r\n\x1a\n"):
        #         return 'png'
        #     if b[:6] in (b'GIF87a', b'GIF89a'):
        #         return 'gif'
        #     if b.startswith(b'BM'):
        #         return 'bmp'
        #     return None

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

            # # bytes
            # if isinstance(val, (bytes, bytearray)):
            #     file_bytes = bytes(val)
            #     ext = detect_ext_from_bytes(file_bytes)

            # # BytesIO-like
            # elif hasattr(val, "getvalue") and callable(val.getvalue):
            #     try:
            #         file_bytes = val.getvalue()
            #         ext = detect_ext_from_bytes(file_bytes)
            #     except Exception:
            #         file_bytes = None

            # # PIL Image
            # elif Image is not None and isinstance(val, Image.Image):
            #     # save via PIL
            #     safe_name = f"{key_name}_{idx}.png"
            #     out_path = output_dir / safe_name
            #     try:
            #         val.save(out_path)
            #         saved.append(str(out_path))
            #         logger.debug(f"Збережено (PIL): {out_path}")
            #         continue
            #     except Exception as e:
            #         logger.warning(f"Не вдалося зберегти PIL image {key_name}: {e}")

            # # dict wrappers
            # elif isinstance(val, dict):
            #     # try common keys
            #     for k in ("bytes", "data", "image", "content", "raw"):
            #         if k in val and isinstance(val[k], (bytes, bytearray)):
            #             file_bytes = bytes(val[k])
            #             ext = detect_ext_from_bytes(file_bytes)
            #             break

            #     # sometimes nested PIL
            #     if file_bytes is None:
            #         for k in ("pil", "image_obj"):
            #             if k in val and Image is not None and isinstance(val[k], Image.Image):
            #                 safe_name = f"{key_name}_{idx}.png"
            #                 out_path = output_dir / safe_name
            #                 try:
            #                     val[k].save(out_path)
            #                     saved.append(str(out_path))
            #                     logger.debug(f"Збережено (nested PIL): {out_path}")
            #                     file_bytes = None
            #                     break
            #                 except Exception:
            #                     pass

            # # string - maybe base64
            # elif isinstance(val, str):
            #     # try base64
            #     try:
            #         file_bytes = base64.b64decode(val)
            #         ext = detect_ext_from_bytes(file_bytes)
            #     except Exception:
            #         file_bytes = None

            # fallback: try to convert to bytes
            if file_bytes:
                # ensure bytes-like
                if not isinstance(file_bytes, (bytes, bytearray)):
                    logger.warning(
                        f"Не вдалося перетворити в bytes зображення '{key_name}' (index {idx}), пропускаємо"
                    )
                    continue

                if not ext:
                    # default to png if we can't detect
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


def parse_pdf_with_marker(
        file_path: str | Path,
        config: Optional[MarkerPdfParserConfig] = None) -> str:
    """Зручна функція для швидкого парсингу PDF"""
    return MarkerPDFParser(config).parse(file_path)
