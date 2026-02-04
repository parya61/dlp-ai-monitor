"""
FastAPI Application для DLP AI Monitor

ЧТО: REST API для ML модели классификации DLP-инцидентов
ЗАЧЕМ: Позволяет использовать модель через HTTP запросы

КАК ЗАПУСТИТЬ:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

ENDPOINTS:
    GET  /              - Информация об API
    GET  /health        - Health check
    POST /api/v1/predict - Классификация инцидента
    GET  /docs          - Swagger UI (автоматическая документация)
    GET  /redoc         - ReDoc (альтернативная документация)

Автор: DLP AI Monitor
Версия: 1.0.0
"""

import sys
from pathlib import Path
import time
import logging
import numpy as np 
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Добавляем корень проекта в PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Импорты из проекта
from api.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse
)
from api.dependencies import (
    get_classifier,
    is_model_loaded,
    get_model_info,
    preload_model
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="DLP AI Monitor API",
    description="""
    DLP AI Monitor  - ML система для автоматической классификации DLP-инцидентов
    
    Возможности:
      Классификация типа инцидента (email/usb/cloud/printer) - 100% accuracy
      Определение критичности (Low/Medium/High/Critical) - 53% accuracy
      Обнаружение PII (персональных данных)
      Рекомендации по действиям
    
    Технологии:
     CatBoost для классификации
     TF-IDF + PII features (64 признака)
     Class weights для балансировки
     Обучено на 30,000 инцидентов
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# =============================================================================
# MIDDLEWARE (CORS, логирование)
# =============================================================================

# CORS - разрешаем запросы с любых доменов (для frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production указать конкретные домены!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request, call_next):
    """
    Логирует все HTTP запросы.
    
    ЗАЧЕМ: Для мониторинга и отладки
    """
    start_time = time.time()
    
    # Обрабатываем запрос
    response = await call_next(request)
    
    # Считаем время обработки
    process_time = (time.time() - start_time) * 1000  # в миллисекундах
    
    # Логируем
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} "
        f"- {process_time:.2f}ms"
    )
    
    # Добавляем заголовок с временем обработки
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    
    return response


# =============================================================================
# STARTUP / SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Выполняется при запуске API.
    
    ЗАЧЕМ: Предзагружаем ML модель чтобы первый запрос был быстрым
    """
    logger.info("=" * 80)
    logger.info(" Starting DLP AI Monitor API")
    logger.info("=" * 80)
    
    # Предзагружаем модель
    try:
        preload_model()
        logger.info(" Model preloaded successfully")
    except Exception as e:
        logger.error(f" Failed to preload model: {e}")
        logger.error("API will start but /predict endpoint will fail!")
    
    logger.info("=" * 80)
    logger.info(" API Documentation:")
    logger.info("   Swagger UI: http://localhost:8000/docs")
    logger.info("   ReDoc:      http://localhost:8000/redoc")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Выполняется при остановке API"""
    logger.info(" Shutting down DLP AI Monitor API")


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Информация об API"])
def root():
    """
    Корневой endpoint - информация об API.
    """
    return {
        "name": "DLP AI Monitor API",
        "version": "1.0.0",
        "description": "ML API для классификации DLP-инцидентов",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/api/v1/predict"
        },
        "model_info": get_model_info(),
        "documentation": "https://github.com/parya61/dlp-ai-monitor"
    }


@app.get("/health", response_model=HealthResponse, tags=["Состояние API"])
def health_check():
    """
    Health check endpoint.
    
    ЗАЧЕМ: Для мониторинга состояния API (Kubernetes, Docker)
    
    Returns:
        HealthResponse: Статус API и загруженных компонентов
    """
    model_loaded = is_model_loaded()
    
    # Определяем статус
    if model_loaded:
        status = "healthy"
    else:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        version="1.0.0"
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_incident(
    request: PredictRequest,
    classifier = Depends(get_classifier)
):
    """
    Классифицирует DLP-инцидент.
    
    ЧТО ДЕЛАЕТ:
    1. Принимает описание инцидента
    2. Извлекает признаки (PII, TF-IDF, metadata)
    3. Предсказывает тип инцидента (email/usb/cloud/printer)
    4. Предсказывает критичность (Low/Medium/High/Critical)
    5. Возвращает результаты + рекомендации
    """
    start_time = time.time()
    
    try:
        # Импортируем pandas для создания DataFrame
        import pandas as pd
        
        # Формируем DataFrame для модели
        # ВАЖНО: Модель ожидает те же колонки что и при обучении
        
        # Объединяем subject + body в description (как модель ожидает)
        full_text = f"{request.subject}. {request.body}"
        
        # Определяем department из email домена (если не указан)
        department = request.department if request.department else "Unknown"
        
        # Проверяем внешний получатель (не @company.com = внешний)
        is_external = not request.recipient_email.endswith("@company.com")
        
        data = {
            'description': [full_text],
            'department': [department],
            'timestamp': [request.timestamp],
            'is_external_recipient': [is_external],
            'contains_pii': [None]  # PII detector определит автоматически
        }
        
        df = pd.DataFrame(data)
        
        # Извлекаем признаки
        feature_extractor = classifier['feature_extractor']
        X = feature_extractor.transform(df)
        
        # Предсказываем тип инцидента
        model_type = classifier['model_incident_type']
        pred_type_raw = model_type.predict(X)

        # Конвертируем numpy array в string
        if isinstance(pred_type_raw, np.ndarray):
            pred_type = str(pred_type_raw.ravel()[0])
        else:
            pred_type = str(pred_type_raw[0])
        
        # Получаем уверенность (вероятности всех классов)
        try:
            proba_type = model_type.predict_proba(X)[0]
            confidence_type = float(max(proba_type))
        except:
            confidence_type = 0.95  # Fallback
        
        # Предсказываем критичность
        model_severity = classifier['model_severity']
        pred_severity_raw = model_severity.predict(X)

        # Конвертируем numpy array в string
        if isinstance(pred_severity_raw, np.ndarray):
            pred_severity = str(pred_severity_raw.ravel()[0])
        else:
            pred_severity = str(pred_severity_raw[0])
        
        # Получаем уверенность
        try:
            proba_severity = model_severity.predict_proba(X)[0]
            confidence_severity = float(max(proba_severity))
        except:
            confidence_severity = 0.60  # Fallback
        
        # Обнаруживаем PII в тексте письма
        from src.nlp.pii_detector import PIIDetector
        pii_detector = PIIDetector()
        pii_result = pii_detector.detect(full_text)
        
        # Извлекаем типы PII
        detected_pii = []
        if len(pii_result['cards']) > 0:
            detected_pii.append('card')
        if len(pii_result['passports']) > 0:
            detected_pii.append('passport')
        if len(pii_result['inn']) > 0:
            detected_pii.append('inn')
        if len(pii_result['snils']) > 0:
            detected_pii.append('snils')
        if len(pii_result['phones']) > 0:
            detected_pii.append('phone')
        if len(pii_result['emails']) > 0:
            detected_pii.append('email')
        
        pii_count = pii_result['pii_count']
        
        # Определяем risk level (на основе severity)
        risk_level = pred_severity
        
        # Генерируем рекомендацию
        recommendation = generate_recommendation(
            pred_type,
            pred_severity,
            pii_count,
            is_external  # Используем вычисленное значение
        )
        
        # Считаем время обработки
        processing_time = (time.time() - start_time) * 1000  # миллисекунды
        
        # Формируем ответ
        response = PredictResponse(
            incident_type=pred_type,
            incident_type_confidence=confidence_type,
            severity=pred_severity,
            severity_confidence=confidence_severity,
            risk_level=risk_level,
            detected_pii=detected_pii,
            pii_count=pii_count,
            recommendation=recommendation,
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(
            f"Prediction: type={pred_type} ({confidence_type:.2f}), "
            f"severity={pred_severity} ({confidence_severity:.2f}), "
            f"pii_count={pii_count}, time={processing_time:.2f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def generate_recommendation(
    incident_type: str,
    severity: str,
    pii_count: int,
    is_external: bool
) -> str:
    """
    Генерирует рекомендацию по действиям.
    
    ЗАЧЕМ: Подсказать аналитику что делать с инцидентом
    
    Args:
        incident_type: Тип инцидента
        severity: Критичность
        pii_count: Количество PII
        is_external: Отправлено наружу?
    
    Returns:
        str: Рекомендация
    """
    recommendations = {
        'Critical': "НЕМЕДЛЕННО: Заблокировать отправку, уведомить службу безопасности, создать инцидент в SIEM",
        'High': "СРОЧНО: Записать в лог высокого приоритета, уведомить администратора, проверить в течение часа",
        'Medium': "ВАЖНО: Зафиксировать инцидент, уведомить ответственного, проверить в течение дня",
        'Low': "ИНФОРМАЦИЯ: Записать для статистики, проверка при плановом аудите"
    }
    
    base_recommendation = recommendations.get(severity, "Проверить инцидент")
    
    # Добавляем детали
    details = []
    
    if pii_count > 0:
        details.append(f"Обнаружено {pii_count} PII")
    
    if is_external:
        details.append("Отправка внешнему получателю - повышенный риск")
    
    if incident_type == 'email' and severity in ['High', 'Critical']:
        details.append("Рекомендуется проверить получателя и содержимое вложений")
    
    if details:
        return f"{base_recommendation}. {'. '.join(details)}."
    
    return base_recommendation


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Глобальный обработчик ошибок.
    
    ЗАЧЕМ: Ловим все необработанные ошибки и возвращаем понятный JSON
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "status_code": 500
        }
    )


# =============================================================================
# MAIN (для запуска через python api/main.py)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload при изменении кода
        log_level="info"
    )