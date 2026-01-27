"""
src.nlp - NLP модули для обнаружения PII (персональных данных)

Модули:
- patterns.py - Regex паттерны для российских документов
- pii_detector.py - Multi-layer детектор PII (regex + spaCy)
"""

from src.nlp.pii_detector import PIIDetector

__all__ = ["PIIDetector"]