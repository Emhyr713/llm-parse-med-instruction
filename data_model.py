# data_model.py
from dataclasses import dataclass, field
from typing import  Dict,  Optional

# ==================== МОДЕЛИ ДАННЫХ ====================

@dataclass
class Sentence:
    """Класс для хранения информации о предложении"""
    text: str
    index: int
    start_char: int
    end_char: int
    length: int
    category: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    section: Optional[str] = None  # 'pd' или 'pk'

@dataclass
class ClassificationStats:
    """Статистика классификации"""
    total_sentences: int = 0
    pd_sentences: int = 0
    pk_sentences: int = 0
    other_sentences: int = 0
    avg_confidence: float = 0.0
    total_time: float = 0.0
    by_category: Dict[str, int] = field(default_factory=dict)
