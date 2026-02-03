"""
API Dependencies для DLP AI Monitor

ЧТО: Загрузка ML модели и её переиспользование
ЗАЧЕМ: Загружаем модель ОДИН РАЗ при старте API, не при каждом запросе

Автор: DLP AI Monitor
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ (загружаются один раз)
# =============================================================================

# Путь к модели
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "dlp_classifier_demo.pkl"

# Хранилище для загруженной модели
_classifier = None


# =============================================================================
# ЗАГРУЗКА МОДЕЛИ
# =============================================================================

def load_classifier():
    """
    Загружает ML модель из pickle файла.
    
    ЧТО: Загружает обученный DLPClassifier
    ЗАЧЕМ: Модель нужна для предсказаний
    
    КАК РАБОТАЕТ:
    1. Проверяем что модель ещё не загружена (singleton pattern)
    2. Загружаем из pickle файла
    3. Сохраняем в глобальную переменную
    4. При следующем вызове возвращаем уже загруженную модель
    
    Returns:
        dict: Словарь с компонентами модели
        {
            'feature_extractor': FeatureExtractor,
            'model_incident_type': CatBoostClassifier,
            'model_severity': CatBoostClassifier,
            'metrics': dict,
            'catboost_params': dict
        }
    
    Raises:
        FileNotFoundError: Если модель не найдена
        Exception: Если ошибка при загрузке
    """
    global _classifier
    
    # Если модель уже загружена - возвращаем её (singleton)
    if _classifier is not None:
        logger.debug("Returning cached classifier")
        return _classifier
    
    # Проверяем существование файла
    if not MODEL_PATH.exists():
        error_msg = f"Model file not found: {MODEL_PATH}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info(f"Loading classifier from {MODEL_PATH}...")
        
        # Загружаем из pickle
        with open(MODEL_PATH, 'rb') as f:
            classifier_data = pickle.load(f)
        
        # Проверяем структуру
        required_keys = ['feature_extractor', 'model_incident_type', 'model_severity']
        missing_keys = [k for k in required_keys if k not in classifier_data]
        
        if missing_keys:
            raise ValueError(f"Model file missing keys: {missing_keys}")
        
        # Сохраняем в глобальную переменную
        _classifier = classifier_data
        
        logger.info("✅ Classifier loaded successfully")
        logger.info(f"   - Incident Type Accuracy: {classifier_data.get('metrics', {}).get('incident_type_accuracy', 'N/A')}")
        logger.info(f"   - Severity Accuracy: {classifier_data.get('metrics', {}).get('severity_accuracy', 'N/A')}")
        
        return _classifier
        
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        raise


def get_classifier():
    """
    Dependency для FastAPI endpoints.
    
    ЗАЧЕМ: FastAPI автоматически вызовет эту функцию и передаст модель в endpoint
    
    КАК ИСПОЛЬЗОВАТЬ:
        @app.post("/predict")
        def predict(
            request: PredictRequest,
            classifier = Depends(get_classifier)  # ← Автоматически!
        ):
            result = classifier['model_severity'].predict(...)
            return result
    
    Returns:
        dict: Загруженная модель
    """
    return load_classifier()


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def is_model_loaded() -> bool:
    """
    Проверяет загружена ли модель.
    
    ЗАЧЕМ: Для health check endpoint
    
    Returns:
        bool: True если модель загружена
    """
    return _classifier is not None


def get_model_info() -> dict:
    """
    Возвращает информацию о модели.
    
    ЗАЧЕМ: Для мониторинга и отладки
    
    Returns:
        dict: Информация о модели
    """
    if _classifier is None:
        return {
            "loaded": False,
            "path": str(MODEL_PATH),
            "exists": MODEL_PATH.exists()
        }
    
    return {
        "loaded": True,
        "path": str(MODEL_PATH),
        "metrics": _classifier.get('metrics', {}),
        "features_count": len(_classifier['feature_extractor'].get_feature_names()),
        "catboost_params": _classifier.get('catboost_params', {})
    }


def reload_classifier():
    """
    Перезагружает модель.
    
    ЗАЧЕМ: Если обновили модель, можно перезагрузить без перезапуска API
    
    Returns:
        dict: Новая загруженная модель
    """
    global _classifier
    _classifier = None
    logger.info("Classifier cache cleared, reloading...")
    return load_classifier()


# =============================================================================
# ПРЕДЗАГРУЗКА ПРИ СТАРТЕ (опционально)
# =============================================================================

def preload_model():
    """
    Предзагружает модель при старте API.
    
    ЗАЧЕМ: Первый запрос не будет ждать загрузки модели
    
    КАК ИСПОЛЬЗОВАТЬ:
        В main.py добавить:
        
        @app.on_event("startup")
        async def startup_event():
            preload_model()
    """
    try:
        load_classifier()
        logger.info("Model preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload model: {e}")
        # Не падаем, API запустится без модели
        # Health check покажет что модель не загружена