"""
src/ml/train.py

ЧТО: Обучение ML модели для классификации DLP-инцидентов
ЗАЧЕМ: Автоматически определять тип инцидента и критичность

АРХИТЕКТУРА:
- DLPClassifier - главный класс
- train() - обучение модели
- save() - сохранение модели
- load() - загрузка модели

ИСПОЛЬЗОВАНИЕ:
    from src.ml import DLPClassifier
    
    classifier = DLPClassifier()
    classifier.train(df_train)
    classifier.save("models/dlp_classifier.pkl")
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from src.config import get_config
from src.ml.features import FeatureExtractor
from src.utils import get_logger, timer

# Инициализация
logger = get_logger(__name__)
config = get_config()


class DLPClassifier:
    """
    ML классификатор для DLP-инцидентов.
    
    Обучает две модели:
    1. incident_type: тип инцидента (email, usb, cloud, printer)
    2. severity: критичность (Low, Medium, High, Critical)
    
    Attributes:
        feature_extractor: FeatureExtractor для извлечения признаков
        model_incident_type: CatBoost модель для типа инцидента
        model_severity: CatBoost модель для критичности
    """
    
    def __init__(
        self,
        max_tfidf_features: int = 100,
        use_pii: bool = True,
        catboost_iterations: Optional[int] = None,
        catboost_learning_rate: Optional[float] = None,
        catboost_depth: Optional[int] = None,
    ):
        """
        Инициализация классификатора.
        
        Args:
            max_tfidf_features: Максимум TF-IDF признаков
            use_pii: Использовать ли PII признаки
            catboost_iterations: Итерации CatBoost (из конфига если None)
            catboost_learning_rate: Learning rate (из конфига если None)
            catboost_depth: Глубина деревьев (из конфига если None)
        """
        logger.info("Initializing DLPClassifier...")
        
        # Feature Extractor
        self.feature_extractor = FeatureExtractor(
            max_tfidf_features=max_tfidf_features,
            use_pii=use_pii
        )
        
        # CatBoost параметры (из конфига или переданные)
        self.catboost_params = {
            "iterations": catboost_iterations or config.CATBOOST_ITERATIONS,
            "learning_rate": catboost_learning_rate or config.CATBOOST_LEARNING_RATE,
            "depth": catboost_depth or config.CATBOOST_DEPTH,
            "loss_function": "MultiClass",  # многоклассовая классификация
            "verbose": False,               # не показывать логи обучения
            "random_seed": 42,              # для воспроизводимости
        }
        
        # Модели (создадим при обучении)
        self.model_incident_type = None
        self.model_severity = None
        
        # Метрики (заполнятся после обучения)
        self.metrics = {}
        
        logger.info(f"DLPClassifier initialized with params: {self.catboost_params}")
    
    @timer
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        target_columns: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Обучает модели на данных.
        
        ПРОЦЕСС:
        1. Разделяем на train/test
        2. Извлекаем признаки через FeatureExtractor
        3. Обучаем CatBoost для incident_type
        4. Обучаем CatBoost для severity
        5. Оцениваем качество на test
        
        Args:
            df: DataFrame с данными
            test_size: Размер test выборки (default: 0.2 = 20%)
            target_columns: Названия целевых колонок
                {"incident_type": "incident_type", "severity": "severity"}
        
        Returns:
            Dict с метриками качества
        
        Example:
            classifier = DLPClassifier()
            metrics = classifier.train(df)
            print(metrics)  # {'incident_type_accuracy': 0.95, ...}
        """
        logger.info(f"Starting training on {len(df)} samples...")
        
        # Целевые колонки
        if target_columns is None:
            target_columns = {
                "incident_type": "incident_type",
                "severity": "severity"
            }
        
        # Проверяем наличие целевых колонок
        for key, col in target_columns.items():
            if col not in df.columns:
                raise ValueError(f"Target column '{col}' not found in DataFrame")
        
        # =====================================================================
        # ШАГ 1: РАЗДЕЛЕНИЕ НА TRAIN/TEST
        # =====================================================================
        
        logger.info(f"Splitting data: train={1-test_size:.0%}, test={test_size:.0%}")
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df[target_columns["incident_type"]]  # сохраняем пропорции классов
        )
        
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        # =====================================================================
        # ШАГ 2: ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
        # =====================================================================
        
        logger.info("Extracting features...")
        
        # Обучаем feature extractor на train данных
        X_train = self.feature_extractor.fit_transform(train_df)
        
        # Применяем к test данным
        X_test = self.feature_extractor.transform(test_df)
        
        logger.info(f"Features extracted: {X_train.shape[1]} features")
        
        # Целевые переменные
        y_train_type = train_df[target_columns["incident_type"]]
        y_test_type = test_df[target_columns["incident_type"]]
        
        y_train_severity = train_df[target_columns["severity"]]
        y_test_severity = test_df[target_columns["severity"]]
        
        # =====================================================================
        # ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ ДЛЯ incident_type
        # =====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING: Incident Type Classifier")
        logger.info("=" * 80)
        
        self.model_incident_type = CatBoostClassifier(**self.catboost_params)
        
        self.model_incident_type.fit(
            X_train,
            y_train_type,
            eval_set=(X_test, y_test_type),
            verbose=False
        )
        
        # Оценка на test
        accuracy_type = self.model_incident_type.score(X_test, y_test_type)
        logger.info(f"Incident Type Accuracy: {accuracy_type:.4f}")
        
        # =====================================================================
        # ШАГ 4: ОБУЧЕНИЕ МОДЕЛИ ДЛЯ severity
        # =====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING: Severity Classifier")
        logger.info("=" * 80)
        
        self.model_severity = CatBoostClassifier(**self.catboost_params)
        
        self.model_severity.fit(
            X_train,
            y_train_severity,
            eval_set=(X_test, y_test_severity),
            verbose=False
        )
        
        # Оценка на test
        accuracy_severity = self.model_severity.score(X_test, y_test_severity)
        logger.info(f"Severity Accuracy: {accuracy_severity:.4f}")
        
        # =====================================================================
        # ШАГ 5: СОХРАНЕНИЕ МЕТРИК
        # =====================================================================
        
        self.metrics = {
            "incident_type_accuracy": accuracy_type,
            "severity_accuracy": accuracy_severity,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "n_features": X_train.shape[1],
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Incident Type Accuracy: {accuracy_type:.4f}")
        logger.info(f"Severity Accuracy: {accuracy_severity:.4f}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        logger.info("=" * 80)
        
        return self.metrics
    
    def predict(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает тип и критичность инцидентов.
        
        Args:
            df: DataFrame с инцидентами
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - predictions_type: предсказанные типы
                - predictions_severity: предсказанная критичность
        
        Example:
            types, severities = classifier.predict(df_new)
        """
        if self.model_incident_type is None or self.model_severity is None:
            raise ValueError("Models not trained! Call train() first.")
        
        logger.info(f"Predicting on {len(df)} samples...")
        
        # Извлекаем признаки
        X = self.feature_extractor.transform(df)
        
        # Предсказания
        predictions_type = self.model_incident_type.predict(X)
        predictions_severity = self.model_severity.predict(X)
        
        return predictions_type, predictions_severity
    
    def save(self, filepath: str | Path) -> None:
        """
        Сохраняет обученный классификатор.
        
        Сохраняет:
        - feature_extractor
        - model_incident_type
        - model_severity
        - metrics
        
        Args:
            filepath: Путь к файлу для сохранения
        
        Example:
            classifier.save("models/dlp_classifier.pkl")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving classifier to {filepath}...")
        
        # Собираем объект для сохранения
        save_dict = {
            "feature_extractor": self.feature_extractor,
            "model_incident_type": self.model_incident_type,
            "model_severity": self.model_severity,
            "metrics": self.metrics,
            "catboost_params": self.catboost_params,
        }
        
        # Сохраняем через pickle
        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Classifier saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str | Path) -> "DLPClassifier":
        """
        Загружает обученный классификатор.
        
        Args:
            filepath: Путь к файлу
        
        Returns:
            DLPClassifier: Загруженный классификатор
        
        Example:
            classifier = DLPClassifier.load("models/dlp_classifier.pkl")
            predictions = classifier.predict(df_new)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading classifier from {filepath}...")
        
        # Загружаем
        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)
        
        # Создаём новый объект
        classifier = cls()
        classifier.feature_extractor = save_dict["feature_extractor"]
        classifier.model_incident_type = save_dict["model_incident_type"]
        classifier.model_severity = save_dict["model_severity"]
        classifier.metrics = save_dict["metrics"]
        classifier.catboost_params = save_dict["catboost_params"]
        
        logger.info(f"Classifier loaded from {filepath}")
        logger.info(f"Metrics: {classifier.metrics}")
        
        return classifier


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    from src.data import DataLoader
    
    logger.info("=" * 80)
    logger.info("DLP CLASSIFIER - TRAINING DEMO")
    logger.info("=" * 80)
    
    # Загружаем данные
    loader = DataLoader()
    csv_path = config.get_data_path("incidents_30k.csv", subdir="synthetic")
    
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        logger.error("Run 'python -m src.data.generator' first!")
    else:
        df = loader.load_csv(csv_path)
        logger.info(f"Loaded {len(df)} incidents")
        
        # Создаём классификатор
        classifier = DLPClassifier(
            max_tfidf_features=50,
            use_pii=True
        )
        
        # Обучаем
        metrics = classifier.train(df, test_size=0.2)
        
        # Сохраняем модель
        model_path = config.get_model_path("dlp_classifier_demo.pkl")
        classifier.save(model_path)
        
        logger.info("\n" + "=" * 80)
        logger.info("Training complete! Model saved.")
        logger.info("=" * 80)