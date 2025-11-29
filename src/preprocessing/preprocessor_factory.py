from typing import Optional, List
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.worker import *


class PreprocessorFactory:
    """Фабрика для створення Preprocessor з різними конфігураціями"""

    @staticmethod
    def create(worker: str = "none",
               default_parser: str = "auto",
               custom_workers: Optional[List[Worker]] = None) -> Preprocessor:
        """
        Створює Preprocessor з обраним профілем.
        
        Args:
            profile: "none", "minimal", "aggressive"
            default_parser: "pdf", "txt", "auto"
            custom_workers: Власний список workers
        """
        if custom_workers is not None:
            return Preprocessor(workers=custom_workers,
                                default_parser=default_parser)

        if worker == "minimal":
            workers = [UnicodeNormalizer(), TextCleaner(), SingleLineifier()]
        elif worker == "aggressive":
            workers = [
                UnicodeNormalizer(),
                FixHyphenUk(),
                TextCleaner(preserve_tables=True),
                ParagraphFixer(),
                EscapeFixer(),
                RemovePageNumbers(aggressive=True),
                SingleLineifier()
            ]
        else:  # default
            workers = None  # Дефолтний pipeline

        return Preprocessor(workers=workers, default_parser=default_parser)
