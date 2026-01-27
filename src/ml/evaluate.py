"""
src/ml/evaluate.py

ЧТО: Оценка качества ML модели для DLP-инцидентов
ЗАЧЕМ: Детальный анализ ошибок, важности признаков, метрик

ФУНКЦИИ:
- evaluate_classifier() - полная оценка модели
- plot_confusion_matrix() - матрица ошибок
- get_classification_report() - precision, recall, F1
- get_feature_importance() - важность признаков

ИСПОЛЬЗОВАНИЕ:
    from src.ml.evaluate import evaluate_classifier
    
    metrics = evaluate_classifier(classifier, df_test)
    print(metrics)
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import get_config
from src.utils import get_logger

# Инициализация
logger = get_logger(__name__)
config = get_config()


def evaluate_classifier(
    classifier,
    df_test: pd.DataFrame,
    target_columns: Dict[str, str] = None
) -> Dict:
    """
    Полная оценка классификатора.
    
    Вычисляет:
    - Accuracy, Precision, Recall, F1 для обоих классов
    - Confusion Matrix
    - Classification Report
    - Feature Importance
    
    Args:
        classifier: Обученный DLPClassifier
        df_test: Test данные
        target_columns: Названия целевых колонок
    
    Returns:
        Dict с метриками и отчётами
    
    Example:
        metrics = evaluate_classifier(classifier, df_test)
        print(f"Accuracy: {metrics['incident_type']['accuracy']}")
    """
    logger.info(f"Evaluating classifier on {len(df_test)} test samples...")
    
    if target_columns is None:
        target_columns = {
            "incident_type": "incident_type",
            "severity": "severity"
        }
    
    # Истинные метки
    y_true_type = df_test[target_columns["incident_type"]]
    y_true_severity = df_test[target_columns["severity"]]
    
    # Предсказания
    y_pred_type, y_pred_severity = classifier.predict(df_test)
    
    # =========================================================================
    # МЕТРИКИ ДЛЯ INCIDENT_TYPE
    # =========================================================================
    
    logger.info("\nEvaluating Incident Type Classifier...")
    
    metrics_type = {
        "accuracy": accuracy_score(y_true_type, y_pred_type),
        "precision_macro": precision_score(y_true_type, y_pred_type, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_type, y_pred_type, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_type, y_pred_type, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_type, y_pred_type),
        "classification_report": classification_report(
            y_true_type, y_pred_type, output_dict=True, zero_division=0
        ),
    }
    
    logger.info(f"Incident Type - Accuracy: {metrics_type['accuracy']:.4f}")
    logger.info(f"Incident Type - F1 (macro): {metrics_type['f1_macro']:.4f}")
    
    # =========================================================================
    # МЕТРИКИ ДЛЯ SEVERITY
    # =========================================================================
    
    logger.info("\nEvaluating Severity Classifier...")
    
    metrics_severity = {
        "accuracy": accuracy_score(y_true_severity, y_pred_severity),
        "precision_macro": precision_score(y_true_severity, y_pred_severity, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_severity, y_pred_severity, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_severity, y_pred_severity, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_severity, y_pred_severity),
        "classification_report": classification_report(
            y_true_severity, y_pred_severity, output_dict=True, zero_division=0
        ),
    }
    
    logger.info(f"Severity - Accuracy: {metrics_severity['accuracy']:.4f}")
    logger.info(f"Severity - F1 (macro): {metrics_severity['f1_macro']:.4f}")
    
    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================
    
    logger.info("\nExtracting feature importance...")
    
    feature_importance_type = get_feature_importance(
        classifier.model_incident_type,
        classifier.feature_extractor.get_feature_names()
    )
    
    feature_importance_severity = get_feature_importance(
        classifier.model_severity,
        classifier.feature_extractor.get_feature_names()
    )
    
    # =========================================================================
    # РЕЗУЛЬТАТ
    # =========================================================================
    
    results = {
        "incident_type": metrics_type,
        "severity": metrics_severity,
        "feature_importance_type": feature_importance_type,
        "feature_importance_severity": feature_importance_severity,
    }
    
    logger.info("\nEvaluation complete!")
    
    return results


def get_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Извлекает важность признаков из CatBoost модели.
    
    ЗАЧЕМ: Понять, какие признаки влияют на предсказания.
    Например: "has_card" может быть самым важным для severity.
    
    Args:
        model: Обученная CatBoost модель
        feature_names: Список имён признаков
        top_n: Показать топ-N важных признаков
    
    Returns:
        pd.DataFrame: Таблица с важностью признаков
    
    Example:
        importance = get_feature_importance(model, feature_names, top_n=10)
        print(importance)
    """
    # Получаем важность из CatBoost
    importance = model.get_feature_importance()
    
    # Создаём DataFrame
    df_importance = pd.DataFrame({
        "feature": feature_names[:len(importance)],
        "importance": importance
    })
    
    # Сортируем по убыванию важности
    df_importance = df_importance.sort_values("importance", ascending=False)
    
    # Топ-N
    df_importance = df_importance.head(top_n)
    
    return df_importance


def print_evaluation_report(metrics: Dict) -> None:
    """
    Красиво выводит отчёт об оценке модели.
    
    Args:
        metrics: Результаты evaluate_classifier()
    
    Example:
        metrics = evaluate_classifier(classifier, df_test)
        print_evaluation_report(metrics)
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION REPORT")
    print("=" * 80)
    
    # =========================================================================
    # INCIDENT TYPE
    # =========================================================================
    
    print("\n📊 INCIDENT TYPE CLASSIFIER")
    print("-" * 80)
    
    type_metrics = metrics["incident_type"]
    
    print(f"Accuracy:  {type_metrics['accuracy']:.4f}")
    print(f"Precision: {type_metrics['precision_macro']:.4f} (macro)")
    print(f"Recall:    {type_metrics['recall_macro']:.4f} (macro)")
    print(f"F1 Score:  {type_metrics['f1_macro']:.4f} (macro)")
    
    print("\n📋 Classification Report:")
    report = type_metrics['classification_report']
    for class_name, metrics_dict in report.items():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        print(f"  {class_name:15s} - Precision: {metrics_dict['precision']:.3f}, "
              f"Recall: {metrics_dict['recall']:.3f}, "
              f"F1: {metrics_dict['f1-score']:.3f}")
    
    print("\n🔝 Top 10 Important Features:")
    importance_type = metrics["feature_importance_type"]
    for idx, row in importance_type.head(10).iterrows():
        print(f"  {row['feature']:30s} - {row['importance']:.2f}")
    
    # =========================================================================
    # SEVERITY
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("⚠️  SEVERITY CLASSIFIER")
    print("-" * 80)
    
    sev_metrics = metrics["severity"]
    
    print(f"Accuracy:  {sev_metrics['accuracy']:.4f}")
    print(f"Precision: {sev_metrics['precision_macro']:.4f} (macro)")
    print(f"Recall:    {sev_metrics['recall_macro']:.4f} (macro)")
    print(f"F1 Score:  {sev_metrics['f1_macro']:.4f} (macro)")
    
    print("\n📋 Classification Report:")
    report = sev_metrics['classification_report']
    for class_name, metrics_dict in report.items():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        print(f"  {class_name:15s} - Precision: {metrics_dict['precision']:.3f}, "
              f"Recall: {metrics_dict['recall']:.3f}, "
              f"F1: {metrics_dict['f1-score']:.3f}")
    
    print("\n🔝 Top 10 Important Features:")
    importance_sev = metrics["feature_importance_severity"]
    for idx, row in importance_sev.head(10).iterrows():
        print(f"  {row['feature']:30s} - {row['importance']:.2f}")
    
    print("\n" + "=" * 80)


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    from src.data import DataLoader
    from src.ml import DLPClassifier
    
    logger.info("=" * 80)
    logger.info("MODEL EVALUATION - DEMO")
    logger.info("=" * 80)
    
    # Загружаем данные
    loader = DataLoader()
    csv_path = config.get_data_path("incidents_sample.csv", subdir="synthetic")
    
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        logger.error("Run 'python -m src.data.generator' first!")
    else:
        df = loader.load_csv(csv_path)
        logger.info(f"Loaded {len(df)} incidents")
        
        # Создаём и обучаем классификатор
        logger.info("\nTraining classifier...")
        classifier = DLPClassifier(max_tfidf_features=50, use_pii=True)
        classifier.train(df, test_size=0.3)  # 30% для test
        
        # Оцениваем
        logger.info("\nEvaluating classifier...")
        from sklearn.model_selection import train_test_split
        _, df_test = train_test_split(df, test_size=0.3, random_state=42)
        
        metrics = evaluate_classifier(classifier, df_test)
        
        # Выводим отчёт
        print_evaluation_report(metrics)
        
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation complete!")
        logger.info("=" * 80)