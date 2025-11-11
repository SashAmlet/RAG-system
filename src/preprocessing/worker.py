from abc import ABC, abstractmethod
import re
from typing import List, Optional, ClassVar
import asyncio

import unicodedata


class Worker(ABC):
    """Абстрактний клас воркера для обробки тексту"""

    @abstractmethod
    def process(self, text: str) -> str:
        pass

    async def process_async(self, text: str) -> str:
        """
        Асинхронний wrapper для сумісності: за замовчуванням запускає
        синхронний метод `process` в окремому потоці через asyncio.to_thread.

        Реалізації, які роблять IO або будуть тривалими, можуть
        перевизначити цей метод асинхронно.
        """
        # Делегуємо до sync-процесу у треді, щоб не блокувати цикл подій
        return await asyncio.to_thread(self.process, text)


class TextCleaner(Worker):
    """
    Розширене очищення тексту після marker-pdf:
    - видаляє зайві спецсимволи
    - нормалізує пунктуацію й пробіли
    - прибирає пусті LaTeX-команди
    """

    def __init__(self, preserve_tables: bool = True, preserve_math: bool = True):
        self.preserve_tables = preserve_tables
        self.preserve_math = preserve_math

    def process(self, text: str) -> str:
        import re

        # Placeholder mechanism to protect blocks we don't want altered
        preserves = []

        def _store_and_replace(m):
            preserves.append(m.group(0))
            return f"__PRESERVE_{len(preserves)-1}__"

        # Preserve [TABLE]...[/TABLE] blocks produced by PDF parser
        if self.preserve_tables:
            text = re.sub(r'\[TABLE\].*?\[/TABLE\]', _store_and_replace, text, flags=re.S)

        # Preserve common LaTeX/math blocks so cleaning doesn't mangle formulas
        if self.preserve_math:
            # $$...$$ blocks
            text = re.sub(r'\$\$.*?\$\$', _store_and_replace, text, flags=re.S)
            # \[ ... \]
            text = re.sub(r'\\\[.*?\\\]', _store_and_replace, text, flags=re.S)
            # inline $...$
            text = re.sub(r'(?<!\$)\$[^\n\$]+\$(?!\$)', _store_and_replace, text, flags=re.S)
            # \begin{...} ... \end{...}
            text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', _store_and_replace, text, flags=re.S)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove URLs (keep parentheses-aware stopping)
        text = re.sub(r'https?://[^\s)]+|www\.[^\s)]+', '', text)

        # Remove emails (simple heuristic)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}', '', text)

        # Remove empty LaTeX commands like \label{...} \ref{...}
        text = re.sub(r'\\(label|ref|cite)\{.*?\}', '', text)

        # Replace common non-breaking spaces and thin spaces
        text = text.replace('\u2009', ' ').replace('\u202F', ' ').replace('\xa0', ' ')

        # Normalize quotes and apostrophes
        text = text.replace('“', '"').replace('”', '"').replace('’', "'")

        # Normalize multiple punctuation and spacing around dashes
        text = re.sub(r'([?!])\1+', r'\1', text)
        text = re.sub(r'\s*[-–—]\s*', ' – ', text)

        # Collapse many spaces but preserve newlines
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'(\n\s*){3,}', '\n\n', text)

        # Trim spaces around newlines
        text = re.sub(r'\s*\n\s*', '\n', text)

        # Restore preserved blocks
        for i, original in enumerate(preserves):
            token = f"__PRESERVE_{i}__"
            text = text.replace(token, original)

        return text.strip()


class UnicodeNormalizer(Worker):
    """
    Нормалізує Unicode:
    - Прибирає невидимі символи
    - Вирівнює кодування (NFC)
    """

    def process(self, text: str) -> str:
        # NFC normalization to canonical composed form
        text = unicodedata.normalize('NFC', text)

        # Remove invisible/formatting chars but keep common whitespace
        # Allow newline, carriage return and tab to remain.
        allowed_whitespace = {'\n', '\r', '\t'}

        filtered_chars = []
        for ch in text:
            cat = unicodedata.category(ch)
            # Categories that start with 'C' are control/other. We drop them
            # except for a few useful whitespace characters.
            if cat and cat[0] == 'C':
                if ch in allowed_whitespace:
                    filtered_chars.append(ch)
                # else: drop the character (e.g., ZERO WIDTH JOINER, other invisibles)
            else:
                filtered_chars.append(ch)

        return ''.join(filtered_chars)

class ParagraphFixer(Worker):
    """
    Нормалізує структуру параграфів:
    - об’єднує занадто розділені рядки
    - додає розриви абзаців після крапок
    """

    def process(self, text: str) -> str:
        import re

        # Merge lines that were broken mid-sentence (not followed by an empty line)
        # Use actual newline in pattern (not the literal backslash-n)
        text = re.sub(r'(?<![.!?])\n(?!\n)', ' ', text)

        # Add paragraph break after sentence-ending punctuation when followed by
        # a capital letter (Latin or Cyrillic). Use a callable replacement to avoid
        # backreference escape pitfalls.
        uppercase = 'A-ZА-ЯЁЄІЇҐ'
        pattern = re.compile(rf'([\.\!\?])\s+(?=[{uppercase}])')

        def _para_repl(m: re.Match) -> str:
            return m.group(1) + "\n\n"

        text = pattern.sub(_para_repl, text)

        # Collapse multiple empty lines to max two
        text = re.sub(r'(\n\s*){3,}', '\n\n', text)

        return text.strip()


class EscapeFixer(Worker):
    """
    Cleans up stray escape sequences introduced by earlier replacements or
    by upstream parsers that left literal backslash sequences like "\n" or "\1".
    - converts literal '\\n' into real newlines
    - removes isolated backslash-number artifacts like '\\1' that often appear
      when replacement strings were double-escaped
    """

    def process(self, text: str) -> str:
        # Convert literal backslash-n sequences into real newlines
        if '\\n' in text:
            text = text.replace('\\n', '\n')

        # Remove artifacts like \1, \2 that are not part of LaTeX (we preserved math blocks earlier)
        text = re.sub(r'\\\d+\b', '', text)

        # Also remove stray backslash followed by whitespace/newline markers
        text = re.sub(r'\\+\s*', '', text)

        return text


class RemovePageNumbers(Worker):
    """
    Воркер для видалення номерів сторінок.
    Припускає, що номера сторінок стоять на окремому рядку, можливі варіанти:
    лише цифри або "Сторінка 12" тощо.
    """

    # Простий regex для рядка з номером сторінки
    # Matches lines like: "12", "Page 12", "Сторінка 12", "p. 12", "12 / 120"
    _page_num_regex: ClassVar[re.Pattern] = re.compile(
        r"^\s*(?:page|p\.?|сторінка|ст\.?|стор\.?|стр\.?|s\.?|pagenumber)?\s*[\d]{1,4}(?:\s*[\\/|]\s*[\d]{1,4})?\s*$",
        re.I | re.U)

    def __init__(self, aggressive: bool = True, min_digits: int = 1):
        """aggressive: if True, will remove more patterns (default True).
        min_digits: minimum digit length to consider as page number (default 1).
        """
        self.aggressive = aggressive
        self.min_digits = min_digits

    def process(self, text: str) -> str:
        lines = text.split('\n')
        filtered_lines = []

        for i, line in enumerate(lines):
            s = line.strip()
            # If the line is very short and contains only digits (likely a page number), drop it
            if s.isdigit() and len(s) >= self.min_digits and (not self.aggressive and len(s) <= 4 or self.aggressive):
                prev_empty = (i == 0) or (lines[i-1].strip() == '')
                next_empty = (i == len(lines)-1) or (lines[i+1].strip() == '')
                if prev_empty or next_empty:
                    continue

            # Drop lines that match common page-number patterns
            if self._page_num_regex.match(s):
                continue

            filtered_lines.append(line)

        return '\n'.join(filtered_lines)



class FixHyphenUk(Worker):
    """
    Окремий воркер для виправлення переносу слів через дефіс в українському тексті.
    Об'єднує слова розбиті дефісом на кінці рядка з наступним рядком.
    """

    def process(self, text: str) -> str:
        # Remove soft-hyphen characters (U+00AD)
        text = text.replace('\u00AD', '')

        lines = text.split('\n')
        processed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            s = line.rstrip()
            # If line ends with a hyphen (or an em/–/— followed by optional space), join with next line
            if i + 1 < len(lines):
                if re.search(r'[-\u2010-\u2015]\s*$', s):
                    next_line = lines[i+1].lstrip()
                    # remove the hyphen and join without extra space
                    joined = re.sub(r'[-\u2010-\u2015]\s*$', '', s) + next_line
                    processed_lines.append(joined)
                    i += 2
                    continue

            processed_lines.append(line)
            i += 1

        return '\n'.join(processed_lines)


class SingleLineifier(Worker):
    """
    Collapses all line breaks into single spaces and collapses multiple
    whitespace runs into a single space, yielding a single-line output.

    Use with caution: this removes paragraph boundaries and other structural
    cues; useful for generating single-line inputs for embeddings or tests.
    """

    def __init__(self, preserve_sentence_breaks: bool = False):
        # preserve_sentence_breaks is reserved for future; currently ignored
        self.preserve_sentence_breaks = preserve_sentence_breaks

    def process(self, text: str) -> str:
        # Replace any sequence of whitespace (including newlines, tabs) with a single space
        # First, normalize CRLF to LF
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove any sequences like '\n' (literal backslash-n) as well
        text = text.replace('\\n', ' ')

        # Collapse all whitespace to single space
        text = re.sub(r"\s+", ' ', text)

        # Trim
        return text.strip()

