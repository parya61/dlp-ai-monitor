"""
src.ml - Machine Learning модули для классификации DLP-инцидентов

Модули:
- features.py - Feature engineering (извлечение признаков)
- train.py - Обучение CatBoost модели
- evaluate.py - Оценка качества модели
- predict.py - Inference (предсказания)
"""

from src.ml.features import FeatureExtractor
from src.ml.train import DLPClassifier
from src.ml.predict import DLPPredictor

__all__ = [
    "FeatureExtractor",
    "DLPClassifier", 
    "DLPPredictor",
]