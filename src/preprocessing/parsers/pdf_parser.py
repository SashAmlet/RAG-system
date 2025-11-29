from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import pdfplumber

from .document_parser import IDocumentParser


@dataclass
class PdfParserConfig:
    """
    Налаштування парсера PDF.
    - include_tables: чи витягати таблиці окремо та додавати як текст.
    - page_separator: роздільник між сторінками (None щоб не додавати).
    - max_pages: максимум сторінок для парсингу (None = всі).
    """
    include_tables: bool = False
    page_separator: Optional[str] = None
    max_pages: Optional[int] = None


class PDFParser(IDocumentParser):
    """
    Парсер PDF-документів на базі pdfplumber.
    """
    supported_extensions = (".pdf", )

    def __init__(self, config: Optional[PdfParserConfig] = None) -> None:
        self.config = config or PdfParserConfig()

    def parse(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not self.supports(path):
            raise ValueError(
                f"Unsupported file extension for PDFParser: {path.suffix}")

        parts: List[str] = []

        with pdfplumber.open(str(path)) as pdf:
            num_pages = len(pdf.pages)
            limit = self.config.max_pages or num_pages

            for i in range(min(num_pages, limit)):
                page = pdf.pages[i]

                # Основний текст сторінки
                page_text = page.extract_text(layout=True) or ""
                # Деякі PDF повертатимуть None — нормалізуємо до порожнього рядка

                # Витяг таблиць як текст
                if self.config.include_tables:
                    tables_text = self._extract_tables_as_text(page)
                    if tables_text:
                        if page_text and not page_text.endswith("\n"):
                            page_text += "\n"
                        page_text += tables_text

                # Додаємо в загальний буфер
                parts.append(page_text.strip())

                # Розділювач між сторінками, якщо задано
                if self.config.page_separator and i < min(num_pages,
                                                          limit) - 1:
                    parts.append(self.config.page_separator)

        # Повертаємо сирий текст; подальшу чистку робитимуть воркери
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _extract_tables_as_text(page) -> str:
        """
        Перетворення знайдених таблиць у просте текстове представлення.
        Позначаємо блоки як [TABLE] ... [/TABLE] для подальших воркерів.
        """
        try:
            tables = page.extract_tables() or []
        except Exception:
            # Якщо pdfminer не зміг виділити таблиці — ігноруємо, повертаємо порожній текст
            return ""

        blocks: List[str] = []
        for tbl in tables:
            if not tbl:
                continue

            rows_as_text = []
            for row in tbl:
                # В деяких випадках клітинки можуть бути None
                cells = [(c or "").strip() for c in row]
                # Простий пайп-сепаратор
                rows_as_text.append(" | ".join(cells))

            if rows_as_text:
                blocks.append("[TABLE]\n" + "\n".join(rows_as_text) +
                              "\n[/TABLE]")

        return "\n\n".join(blocks)

    def supports(self, file_path: str | Path) -> bool:
        """Перевіряє чи це PDF"""
        return Path(file_path).suffix.lower() == '.pdf'

    def get_supported_extensions(self) -> List[str]:
        return ['.pdf']
