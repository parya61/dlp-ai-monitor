"""
src/ml/predict.py

ЧТО: Inference (предсказания) на новых DLP-инцидентах
ЗАЧЕМ: Использовать обученную модель в production

КЛАСС:
- DLPPredictor - обёртка для удобного использования модели

ИСПОЛЬЗОВАНИЕ:
    from src.ml import DLPPredictor
    
    predictor = DLPPredictor.from_file("models/dlp_classifier.pkl")
    
    incident = {
        "description": "Карта: 1234 5678 9012 3456",
        "department": "Sales",
        "user": "Иванов И.И."
    }
    
    result = predictor.predict_one(incident)
    print(result)  # {'incident_type': 'email', 'severity': 'High', ...}
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import get_config
from src.ml.train import DLPClassifier
from src.utils import get_logger

# Инициализация
logger = get_logger(__name__)
config = get_config()


class DLPPredictor:
    """
    Predictor для DLP-инцидентов.
    
    Удобная обёртка над обученным DLPClassifier для inference.
    
    Attributes:
        classifier: Загруженный DLPClassifier
    """
    
    def __init__(self, classifier: DLPClassifier):
        """
        Инициализация predictor'а.
        
        Args:
            classifier: Обученный DLPClassifier
        """
        if classifier.model_incident_type is None:
            raise ValueError("Classifier not trained! Train or load a model first.")
        
        self.classifier = classifier
        logger.info("DLPPredictor initialized")
    
    @classmethod
    def from_file(cls, filepath: str | Path) -> "DLPPredictor":
        """
        Загружает predictor из файла.
        
        Args:
            filepath: Путь к сохранённому классификатору
        
        Returns:
            DLPPredictor: Готовый predictor
        
        Example:
            predictor = DLPPredictor.from_file("models/dlp_classifier.pkl")
        """
        logger.info(f"Loading predictor from {filepath}...")
        classifier = DLPClassifier.load(filepath)
        return cls(classifier)
    
    def predict_one(self, incident: Dict) -> Dict:
        """
        Предсказывает тип и критичность для одного инцидента.
        
        Args:
            incident: Dict с полями инцидента
                - description: текст описания (обязательно)
                - department: отдел (опционально)
                - user: пользователь (опционально)
                - ... другие поля
        
        Returns:
            Dict с предсказаниями:
                - incident_type: предсказанный тип
                - severity: предсказанная критичность
                - confidence_type: уверенность модели (0-1)
                - confidence_severity: уверенность модели (0-1)
        
        Example:
            incident = {
                "description": "Карта: 1234 5678 9012 3456",
                "department": "Finance"
            }
            result = predictor.predict_one(incident)
            print(result)
        """
        # Проверяем наличие description
        if "description" not in incident:
            raise ValueError("Incident must have 'description' field")
        
        # Превращаем в DataFrame (модель ожидает DataFrame)
        df = pd.DataFrame([incident])
        
        # Предсказываем
        type_pred, severity_pred = self.classifier.predict(df)
        
        # Получаем вероятности (confidence)
        X = self.classifier.feature_extractor.transform(df)
        
        # Вероятности для каждого класса
        type_proba = self.classifier.model_incident_type.predict_proba(X)[0]
        severity_proba = self.classifier.model_severity.predict_proba(X)[0]
        
        # Максимальная вероятность = confidence
        confidence_type = float(type_proba.max())
        confidence_severity = float(severity_proba.max())
        
        result = {
            "incident_type": type_pred[0],
            "severity": severity_pred[0],
            "confidence_type": confidence_type,
            "confidence_severity": confidence_severity,
        }
        
        return result
    
    def predict_batch(self, incidents: List[Dict]) -> pd.DataFrame:
        """
        Предсказывает для нескольких инцидентов.
        
        Args:
            incidents: Список Dict'ов с инцидентами
        
        Returns:
            pd.DataFrame: Таблица с предсказаниями
        
        Example:
            incidents = [
                {"description": "Карта: 1234..."},
                {"description": "Паспорт: 4567..."}
            ]
            results = predictor.predict_batch(incidents)
        """
        logger.info(f"Predicting batch of {len(incidents)} incidents...")
        
        # Превращаем в DataFrame
        df = pd.DataFrame(incidents)
        
        # Предсказываем
        type_pred, severity_pred = self.classifier.predict(df)
        
        # Получаем вероятности
        X = self.classifier.feature_extractor.transform(df)
        type_proba = self.classifier.model_incident_type.predict_proba(X)
        severity_proba = self.classifier.model_severity.predict_proba(X)
        
        # Формируем результат
        results = df.copy()
        results["predicted_incident_type"] = type_pred
        results["predicted_severity"] = severity_pred
        results["confidence_type"] = type_proba.max(axis=1)
        results["confidence_severity"] = severity_proba.max(axis=1)
        
        logger.info(f"Batch prediction complete!")
        
        return results
    
    def predict_from_csv(
        self,
        input_csv: str | Path,
        output_csv: str | Path
    ) -> pd.DataFrame:
        """
        Загружает инциденты из CSV, предсказывает, сохраняет результат.
        
        ЗАЧЕМ: Для batch processing больших файлов.
        
        Args:
            input_csv: Путь к входному CSV
            output_csv: Путь к выходному CSV
        
        Returns:
            pd.DataFrame: Результаты
        
        Example:
            results = predictor.predict_from_csv(
                "data/new_incidents.csv",
                "data/predictions.csv"
            )
        """
        logger.info(f"Loading incidents from {input_csv}...")
        
        from src.data import DataLoader
        loader = DataLoader()
        df = loader.load_csv(input_csv)
        
        logger.info(f"Loaded {len(df)} incidents")
        
        # Предсказываем
        type_pred, severity_pred = self.classifier.predict(df)
        
        # Добавляем предсказания в DataFrame
        df["predicted_incident_type"] = type_pred
        df["predicted_severity"] = severity_pred
        
        # Сохраняем
        logger.info(f"Saving results to {output_csv}...")
        loader.save_csv(df, output_csv)
        
        logger.info(f"Predictions saved to {output_csv}")
        
        return df
    
    def get_model_info(self) -> Dict:
        """
        Возвращает информацию о модели.
        
        Returns:
            Dict с информацией:
                - metrics: метрики качества
                - n_features: количество признаков
                - catboost_params: параметры CatBoost
        """
        info = {
            "metrics": self.classifier.metrics,
            "n_features": len(self.classifier.feature_extractor.get_feature_names()),
            "catboost_params": self.classifier.catboost_params,
        }
        
        return info


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_predictor(model_path: str | Path = None) -> DLPPredictor:
    """
    Удобная функция для загрузки predictor'а.
    
    Args:
        model_path: Путь к модели (если None, использует default)
    
    Returns:
        DLPPredictor: Готовый predictor
    
    Example:
        predictor = load_predictor()
        result = predictor.predict_one(incident)
    """
    if model_path is None:
        # Используем default путь из конфига
        model_path = config.get_model_path("dlp_classifier.pkl")
    
    return DLPPredictor.from_file(model_path)


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("DLP PREDICTOR - DEMO")
    logger.info("=" * 80)
    
    # Проверяем наличие обученной модели
    model_path = config.get_model_path("dlp_classifier_demo.pkl")
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Train a model first: python -m src.ml.train")
    else:
        # Загружаем predictor
        predictor = DLPPredictor.from_file(model_path)
        
        # Информация о модели
        info = predictor.get_model_info()
        logger.info(f"\nModel Info:")
        logger.info(f"  Features: {info['n_features']}")
        logger.info(f"  Incident Type Accuracy: {info['metrics']['incident_type_accuracy']:.4f}")
        logger.info(f"  Severity Accuracy: {info['metrics']['severity_accuracy']:.4f}")
        
        # =====================================================================
        # ТЕСТОВЫЕ ПРЕДСКАЗАНИЯ
        # =====================================================================
        
        print("\n" + "=" * 80)
        print("TEST PREDICTIONS")
        print("=" * 80)
        
        # Тест 1: Email с картой и паспортом
        incident1 = {
            "description": "Добрый день! Карта: 1234 5678 9012 3456, паспорт: 4567 123456",
            "department": "Sales",
            "user": "Иванов И.И."
        }
        
        print("\n📧 Test 1: Email with card and passport")
        print(f"Description: {incident1['description'][:50]}...")
        result1 = predictor.predict_one(incident1)
        print(f"Predicted Type: {result1['incident_type']} (confidence: {result1['confidence_type']:.2f})")
        print(f"Predicted Severity: {result1['severity']} (confidence: {result1['confidence_severity']:.2f})")
        
        # Тест 2: USB инцидент
        incident2 = {
            "description": "Попытка копирования файла 'Клиенты.xlsx' на USB-накопитель",
            "department": "IT",
            "user": "Петров П.П."
        }
        
        print("\n💾 Test 2: USB incident")
        print(f"Description: {incident2['description']}")
        result2 = predictor.predict_one(incident2)
        print(f"Predicted Type: {result2['incident_type']} (confidence: {result2['confidence_type']:.2f})")
        print(f"Predicted Severity: {result2['severity']} (confidence: {result2['confidence_severity']:.2f})")
        
        # Тест 3: Cloud инцидент
        incident3 = {
            "description": "Загрузка документа в Google Drive",
            "department": "Marketing",
            "user": "Сидорова С.С."
        }
        
        print("\n☁️  Test 3: Cloud incident")
        print(f"Description: {incident3['description']}")
        result3 = predictor.predict_one(incident3)
        print(f"Predicted Type: {result3['incident_type']} (confidence: {result3['confidence_type']:.2f})")
        print(f"Predicted Severity: {result3['severity']} (confidence: {result3['confidence_severity']:.2f})")
        
        print("\n" + "=" * 80)
        logger.info("Prediction demo complete!")
        logger.info("=" * 80)