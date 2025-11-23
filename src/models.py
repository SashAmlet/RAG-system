from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class ProcessorResult:
    """
    Модель для зберігання результатів обробЦчки документа.
    
    """
    processed_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_filename: Optional[str] = None
    processing_info: Dict[str, Any] = field(default_factory=dict)
    chunks: List['TextChunk'] = field(default_factory=list)

    def __post_init__(self):
        if not self.processing_info:
            self.processing_info = {
                'text_length': len(self.processed_text),
                'cleaning_steps_applied': []
            }


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
    metadata: Dict[str, Any]
