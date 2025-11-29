"""
Text chunking strategies для RAG системи.
Підтримує різні методи розбиття тексту на семантичні фрагменти.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import re
from dataclasses import dataclass
import logging

from src.models import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Конфігурація для chunker"""
    chunk_size: int = 500  # символів
    chunk_overlap: int = 150  # перекриття між chunks
    min_chunk_size: int = 100  # мінімальний розмір chunk
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True

    def __post_init__(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap має бути меншим за chunk_size")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size має бути меншим за chunk_size")


class BaseChunker(ABC):
    """Базовий абстрактний клас для chunkers"""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    @abstractmethod
    def chunk(self, text: str, document_id: str) -> List[TextChunk]:
        """
        Розбиває текст на chunks
        
        Args:
            text: текст для розбиття
            document_id: ID документа
            
        Returns:
            List[TextChunk]: список chunks
        """
        pass

    def _create_chunk(self,
                      text: str,
                      chunk_index: int,
                      document_id: str,
                      start_char: int,
                      end_char: int,
                      metadata: Optional[dict] = None) -> TextChunk:
        """Допоміжний метод для створення chunk"""
        chunk_id = f"{document_id}_chunk_{chunk_index}"

        base_metadata = {
            'chunk_index': chunk_index,
            'char_count': len(text),
            'start_char': start_char,
            'end_char': end_char,
        }

        if metadata:
            base_metadata.update(metadata)

        return TextChunk(text=text.strip(),
                         chunk_id=chunk_id,
                         document_id=document_id,
                         metadata=base_metadata,
                         start_char=start_char,
                         end_char=end_char,
                         chunk_index=chunk_index)


class FixedSizeChunker(BaseChunker):
    """
    Простий chunker з фіксованим розміром.
    Швидкий, але не враховує семантику.
    """

    def chunk(self, text: str, document_id: str) -> List[TextChunk]:
        chunks = []
        chunk_index = 0
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.config.chunk_size, text_length)
            chunk_text = text[start:end]

            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                chunks.append(
                    self._create_chunk(text=chunk_text,
                                       chunk_index=chunk_index,
                                       document_id=document_id,
                                       start_char=start,
                                       end_char=end))
                chunk_index += 1

            start += self.config.chunk_size - self.config.chunk_overlap

        logger.info(
            f"Created {len(chunks)} fixed-size chunks for document {document_id}"
        )
        return chunks


class SentenceChunker(BaseChunker):
    """
    Chunker що поважає межі речень.
    Оптимальний баланс між швидкістю та якістю.
    """

    # Регулярки для розпізнавання кінця речення
    SENTENCE_ENDINGS = re.compile(
        r'([.!?…]+[\s\n]+)|'  # Стандартні розділові знаки
        r'([.!?…]+$)'  # Розділові знаки в кінці тексту
    )

    # Винятки - скорочення які не є кінцем речення
    ABBREVIATIONS = {
        'др',
        'проф',
        'акад',
        'інж',
        'ст',
        'мол',
        'р',  # українські
        'р.',
        'ст.',
        'див.',
        'напр.',
        'т.д.',
        'т.п.',  # з крапками
        'Mr',
        'Mrs',
        'Dr',
        'Prof',
        'Inc',
        'Ltd',
        'etc',  # англійські
        'Ph.D',
        'M.D',
        'B.A',
        'M.A'
    }

    def _split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Розбиває текст на речення, зберігаючи позиції
        
        Returns:
            List[(sentence, start_pos, end_pos)]
        """
        sentences = []
        current_start = 0

        for match in self.SENTENCE_ENDINGS.finditer(text):
            end_pos = match.end()
            sentence = text[current_start:end_pos].strip()

            # Перевіряємо чи це не скорочення
            if self._is_real_sentence_end(sentence):
                if sentence:
                    sentences.append((sentence, current_start, end_pos))
                current_start = end_pos

        # Додаємо залишок тексту якщо є
        if current_start < len(text):
            remaining = text[current_start:].strip()
            if remaining:
                sentences.append((remaining, current_start, len(text)))

        return sentences

    def _is_real_sentence_end(self, sentence: str) -> bool:
        """Перевіряє чи це справді кінець речення, а не скорочення"""
        # Проста евристика: перевіряємо останнє слово перед крапкою
        words = sentence.split()
        if not words:
            return True

        last_word = words[-1].rstrip('.!?…')
        return last_word.lower() not in self.ABBREVIATIONS

    def chunk(self, text: str, document_id: str) -> List[TextChunk]:
        sentences = self._split_into_sentences(text)
        chunks = []
        chunk_index = 0

        current_chunk_sentences = []
        current_chunk_size = 0
        current_chunk_start = 0

        for i, (sentence, start_pos, end_pos) in enumerate(sentences):
            sentence_length = len(sentence)

            # Якщо одне речення більше за chunk_size - розбиваємо його
            if sentence_length > self.config.chunk_size:
                # Зберігаємо попередній chunk якщо є
                if current_chunk_sentences:
                    # ✅ Витягуємо тільки текст для об'єднання
                    chunk_text = ' '.join(
                        [s[0] for s in current_chunk_sentences])
                    chunks.append(
                        self._create_chunk(text=chunk_text,
                                           chunk_index=chunk_index,
                                           document_id=document_id,
                                           start_char=current_chunk_start,
                                           end_char=start_pos,
                                           metadata={
                                               'sentence_count':
                                               len(current_chunk_sentences)
                                           }))
                    chunk_index += 1
                    current_chunk_sentences = []
                    current_chunk_size = 0

                # Розбиваємо довге речення як fixed-size
                for sub_start in range(
                        0, sentence_length,
                        self.config.chunk_size - self.config.chunk_overlap):
                    sub_end = min(sub_start + self.config.chunk_size,
                                  sentence_length)
                    sub_chunk = sentence[sub_start:sub_end]

                    if len(sub_chunk.strip()) >= self.config.min_chunk_size:
                        chunks.append(
                            self._create_chunk(
                                text=sub_chunk,
                                chunk_index=chunk_index,
                                document_id=document_id,
                                start_char=start_pos + sub_start,
                                end_char=start_pos + sub_end,
                                metadata={'is_long_sentence': True}))
                        chunk_index += 1

                current_chunk_start = end_pos
                continue

            # Перевіряємо чи додавання речення не перевищить chunk_size
            if (current_chunk_size + sentence_length > self.config.chunk_size
                    and current_chunk_sentences):

                # Зберігаємо поточний chunk
                # ✅ Витягуємо тільки текст
                chunk_text = ' '.join([s[0] for s in current_chunk_sentences])
                chunk_end = current_chunk_sentences[-1][
                    2]  # ✅ Беремо end_pos останнього речення

                chunks.append(
                    self._create_chunk(text=chunk_text,
                                       chunk_index=chunk_index,
                                       document_id=document_id,
                                       start_char=current_chunk_start,
                                       end_char=chunk_end,
                                       metadata={
                                           'sentence_count':
                                           len(current_chunk_sentences)
                                       }))
                chunk_index += 1

                # Створюємо overlap - беремо останні речення з попереднього chunk
                overlap_sentences = []
                overlap_size = 0
                overlap_start = current_chunk_start

                for prev_sent, prev_start, prev_end in reversed(
                        current_chunk_sentences[-3:]):
                    if overlap_size + len(
                            prev_sent) <= self.config.chunk_overlap:
                        overlap_sentences.insert(
                            0, (prev_sent, prev_start, prev_end))
                        overlap_size += len(prev_sent)
                        overlap_start = prev_start
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_chunk_size = overlap_size
                current_chunk_start = overlap_start if overlap_sentences else start_pos

            # Додаємо кортеж
            current_chunk_sentences.append((sentence, start_pos, end_pos))
            current_chunk_size += sentence_length

        # Додаємо останній chunk
        if current_chunk_sentences:
            # Витягуємо тільки текст
            chunk_text = ' '.join([s[0] for s in current_chunk_sentences])
            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                chunks.append(
                    self._create_chunk(text=chunk_text,
                                       chunk_index=chunk_index,
                                       document_id=document_id,
                                       start_char=current_chunk_start,
                                       end_char=len(text),
                                       metadata={
                                           'sentence_count':
                                           len(current_chunk_sentences)
                                       }))

        logger.info(
            f"Created {len(chunks)} sentence-aware chunks for document {document_id}"
        )
        return chunks


class SemanticChunker(BaseChunker):
    """
    Просунутий chunker що враховує семантику через параграфи та секції.
    Найкраща якість для RAG, але повільніший.
    """

    # Розділові знаки для параграфів
    PARAGRAPH_SEPARATOR = re.compile(r'\n\s*\n+')

    # Маркери секцій (заголовки)
    SECTION_MARKERS = re.compile(
        r'^(#{1,6}\s+.+|'  # Markdown headers
        r'\d+\.\s+[А-ЯІЇЄҐA-Z].+|'  # Нумеровані заголовки
        r'[А-ЯІЇЄҐA-Z][А-ЯІЇЄҐA-Z\s]{3,}|'  # ВЕЛИКІ БУКВИ заголовки
        r'Розділ\s+\d+|'  # "Розділ 1"
        r'Chapter\s+\d+|'  # "Chapter 1"
        r'Глава\s+\d+)'  # "Глава 1"
        ,
        re.MULTILINE)

    def _split_into_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """Розбиває текст на параграфи"""
        paragraphs = []
        last_end = 0

        for match in self.PARAGRAPH_SEPARATOR.finditer(text):
            para_text = text[last_end:match.start()].strip()
            if para_text:
                paragraphs.append((para_text, last_end, match.start()))
            last_end = match.end()

        # Останній параграф
        if last_end < len(text):
            para_text = text[last_end:].strip()
            if para_text:
                paragraphs.append((para_text, last_end, len(text)))

        return paragraphs

    def _extract_section_title(self, text: str) -> Optional[str]:
        """Витягує заголовок секції якщо є"""
        match = self.SECTION_MARKERS.match(text)
        if match:
            return match.group(0).strip('#').strip()
        return None

    def chunk(self, text: str, document_id: str) -> List[TextChunk]:
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        chunk_index = 0

        current_chunk_paras = []
        current_chunk_size = 0
        current_chunk_start = 0
        current_section = None

        for i, (para, start_pos, end_pos) in enumerate(paragraphs):
            # Перевіряємо чи це заголовок секції
            section_title = self._extract_section_title(para)
            if section_title:
                current_section = section_title

            para_length = len(para)

            # Якщо параграф занадто великий - використовуємо sentence chunker
            if para_length > self.config.chunk_size:
                # Зберігаємо попередній chunk
                if current_chunk_paras:
                    chunk_text = '\n\n'.join(current_chunk_paras)
                    chunks.append(
                        self._create_chunk(text=chunk_text,
                                           chunk_index=chunk_index,
                                           document_id=document_id,
                                           start_char=current_chunk_start,
                                           end_char=start_pos,
                                           metadata={
                                               'paragraph_count':
                                               len(current_chunk_paras),
                                               'section':
                                               current_section
                                           }))
                    chunk_index += 1
                    current_chunk_paras = []
                    current_chunk_size = 0

                # Розбиваємо великий параграф через SentenceChunker
                sentence_chunker = SentenceChunker(self.config)
                sub_chunks = sentence_chunker.chunk(para,
                                                    f"{document_id}_para{i}")

                for sub_chunk in sub_chunks:
                    # Оновлюємо метадані
                    sub_chunk.chunk_index = chunk_index
                    sub_chunk.chunk_id = f"{document_id}_chunk_{chunk_index}"
                    sub_chunk.document_id = document_id
                    sub_chunk.metadata['section'] = current_section
                    sub_chunk.metadata['from_long_paragraph'] = True
                    sub_chunk.start_char = start_pos + sub_chunk.start_char
                    sub_chunk.end_char = start_pos + sub_chunk.end_char

                    chunks.append(sub_chunk)
                    chunk_index += 1

                current_chunk_start = end_pos
                continue

            # Перевіряємо чи додавання параграфа не перевищить розмір
            if current_chunk_size + para_length > self.config.chunk_size and current_chunk_paras:
                # Зберігаємо chunk
                chunk_text = '\n\n'.join(current_chunk_paras)
                chunks.append(
                    self._create_chunk(text=chunk_text,
                                       chunk_index=chunk_index,
                                       document_id=document_id,
                                       start_char=current_chunk_start,
                                       end_char=start_pos,
                                       metadata={
                                           'paragraph_count':
                                           len(current_chunk_paras),
                                           'section':
                                           current_section
                                       }))
                chunk_index += 1

                # Overlap - останній параграф
                if self.config.chunk_overlap > 0 and len(
                        current_chunk_paras) > 0:
                    last_para = current_chunk_paras[-1]
                    if len(last_para) <= self.config.chunk_overlap:
                        current_chunk_paras = [last_para]
                        current_chunk_size = len(last_para)
                        # Знаходимо start позицію останнього параграфа
                        if i > 0:
                            current_chunk_start = paragraphs[i - 1][1]
                    else:
                        current_chunk_paras = []
                        current_chunk_size = 0
                        current_chunk_start = start_pos
                else:
                    current_chunk_paras = []
                    current_chunk_size = 0
                    current_chunk_start = start_pos

            current_chunk_paras.append(para)
            current_chunk_size += para_length

        # Останній chunk
        if current_chunk_paras:
            chunk_text = '\n\n'.join(current_chunk_paras)
            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                chunks.append(
                    self._create_chunk(text=chunk_text,
                                       chunk_index=chunk_index,
                                       document_id=document_id,
                                       start_char=current_chunk_start,
                                       end_char=len(text),
                                       metadata={
                                           'paragraph_count':
                                           len(current_chunk_paras),
                                           'section':
                                           current_section
                                       }))

        logger.info(
            f"Created {len(chunks)} semantic chunks for document {document_id}"
        )
        return chunks


# Factory для створення chunker
class ChunkerFactory:
    """Factory для створення chunkers"""

    _chunkers = {
        'fixed': FixedSizeChunker,
        'sentence': SentenceChunker,
        'semantic': SemanticChunker,
    }

    @classmethod
    def create(cls,
               chunker_type: str = 'semantic',
               config: Optional[ChunkingConfig] = None) -> BaseChunker:
        """
        Створює chunker заданого типу
        
        Args:
            chunker_type: тип chunker ('fixed', 'sentence', 'semantic')
            config: конфігурація chunker
            
        Returns:
            BaseChunker: екземпляр chunker
        """
        if chunker_type not in cls._chunkers:
            raise ValueError(f"Unknown chunker type: {chunker_type}. "
                             f"Available: {list(cls._chunkers.keys())}")

        chunker_class = cls._chunkers[chunker_type]
        return chunker_class(config)

    @classmethod
    def register_chunker(cls, name: str, chunker_class: type):
        """Реєструє новий тип chunker"""
        cls._chunkers[name] = chunker_class
