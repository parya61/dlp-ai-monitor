"""
API Schemas для DLP AI Monitor

ЧТО: Pydantic модели для валидации входящих/исходящих данных
ЗАЧЕМ: FastAPI автоматически проверяет типы и возвращает 422 если данные неверные
"""

from pydantic import BaseModel, Field
from typing import Optional


# REQUEST SCHEMAS
class PredictRequest(BaseModel):
    """Запрос на классификацию DLP-инцидента"""
    description: str = Field(..., min_length=10, max_length=1000)
    department: str
    timestamp: Optional[str] = None
    is_external_recipient: Optional[bool] = False
    contains_pii: Optional[bool] = None


# RESPONSE SCHEMAS  
class PredictResponse(BaseModel):
    """Ответ API с предсказанием"""
    incident_type: str
    incident_type_confidence: float
    severity: str
    severity_confidence: float
    risk_level: str
    detected_pii: list[str] = []
    pii_count: int = 0
    recommendation: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check ответ"""
    status: str
    model_loaded: bool
    version: str = "1.0.0"