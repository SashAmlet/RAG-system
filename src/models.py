"""Data models for the application"""
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PreprocessorResult:
    tokens: List[str]

@dataclass
class EmbedderResult:
    vector: List[float]
    metadata: Dict[str, Any]
