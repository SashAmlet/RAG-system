from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class ProcessorResult:
    """
    Результат Preprocessor - весь оброблений документ.
    """
    processed_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_filename: Optional[str] = None
    processing_info: Dict[str, Any] = field(default_factory=dict)
    chunks: List['TextChunk'] = field(default_factory=list)
    document_id: Optional[str] = None

    def __post_init__(self):
        if not self.processing_info:
            self.processing_info = {
                'text_length': len(self.processed_text),
                'cleaning_steps_applied': []
            }
        # Генеруємо document_id якщо не вказано
        if not self.document_id and self.original_filename:
            import hashlib
            self.document_id = hashlib.md5(
                self.original_filename.encode()).hexdigest()[:16]


@dataclass
class TextChunk:
    text: str
    chunk_id: str
    document_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0

    def __post_init__(self):
        # Автоматично додаємо базові метадані
        if 'char_count' not in self.metadata:
            self.metadata['char_count'] = len(self.text)
        if 'start_char' not in self.metadata:
            self.metadata['start_char'] = self.start_char
        if 'end_char' not in self.metadata:
            self.metadata['end_char'] = self.end_char


@dataclass
class EmbedderResult:
    vector: List[float]
    chunk_id: str
    document_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Додаємо розмірність вектора
        if 'dim' not in self.metadata:
            self.metadata['dim'] = len(self.vector)


@dataclass
class SearchResult:
    """
    Результат пошуку в векторній базі.
    """
    chunk: TextChunk  # Текстовий чанк
    score: float  # Similarity score (0-1, вище = краще)
    document_id: str  # ID документа
    chunk_id: str  # ID чанка
    metadata: Dict[str,
                   Any] = field(default_factory=dict)  # Додаткові метадані

    def __post_init__(self):
        # Обмежуємо score діапазоном [0, 1]
        if self.score > 1.0:
            self.score = 1.0
        elif self.score < 0.0:
            self.score = 0.0

        # Копіюємо метадані з чанка
        if not self.metadata and self.chunk:
            self.metadata = self.chunk.metadata.copy()
