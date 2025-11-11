from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class ProcessorResult:
    """
    Модель для зберігання результатів обробки документа.
    
    Attributes:
        processed_text (str): Оброблений текст документа
        metadata (Dict[str, Any]): Метадані про оригінальний документ
        original_filename (Optional[str]): Ім'я оригінального файлу
        processing_info (Dict[str, Any]): Інформація про кроки обробки
    """
    processed_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_filename: Optional[str] = None
    processing_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ініціалізація processing_info якщо порожній."""
        if not self.processing_info:
            self.processing_info = {
                'text_length': len(self.processed_text),
                'word_count': len(self.processed_text.split()) if self.processed_text else 0,
                'cleaning_steps_applied': []
            }

@dataclass
class PreprocessorResult:
    tokens: List[str]

@dataclass
class EmbedderResult:
    vector: List[float]
    metadata: Dict[str, Any]
