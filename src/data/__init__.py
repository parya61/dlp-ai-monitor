"""
src.data - Модули для работы с данными.

Модули:
- generator.py - Генерация синтетических DLP-инцидентов
- loader.py - Загрузка данных из файлов (CSV, Parquet, Excel)
- preprocessor.py - Preprocessing для ML (создадим позже)
"""

from src.data.generator import DLPIncidentGenerator
from src.data.loader import DataLoader

__all__ = [
    "DLPIncidentGenerator",
    "DataLoader",
]