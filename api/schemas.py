"""
API Schemas для DLP AI Monitor

ЧТО: Pydantic модели для валидации входящих/исходящих данных
ЗАЧЕМ: FastAPI автоматически проверяет типы и возвращает 422 если данные неверные
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# REQUEST SCHEMAS
class PredictRequest(BaseModel):
    """
    Запрос на классификацию DLP-инцидента
    
    ЧТО: Полная информация об инциденте для анализа
    ЗАЧЕМ: ML модель использует все эти признаки для классификации
    """
    # Email информация
    user_email: str = Field(..., description="Email отправителя")
    recipient_email: str = Field(..., description="Email получателя")
    subject: str = Field(..., min_length=1, max_length=500, description="Тема письма")
    body: str = Field(..., min_length=1, max_length=10000, description="Текст письма")
    
    # Вложения
    attachment_names: List[str] = Field(default=[], description="Список имён файлов")
    attachment_types: List[str] = Field(default=[], description="MIME типы файлов")
    attachment_sizes: List[int] = Field(default=[], description="Размеры файлов в байтах")
    
    # Временная метка
    timestamp: str = Field(..., description="Время инцидента в формате ISO 8601")
    
    # Опциональные метаданные
    department: Optional[str] = Field(None, description="Отдел отправителя")
    user_role: Optional[str] = Field(None, description="Роль пользователя")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_email": "employee@company.com",
                "recipient_email": "competitor@external.com",
                "subject": "Confidential: Q4 Financial Report",
                "body": "Please find attached our Q4 financial data. Revenue: $5M, Expenses: $3M.",
                "attachment_names": ["Q4_Report.xlsx", "Budget_2024.pdf"],
                "attachment_types": ["application/vnd.ms-excel", "application/pdf"],
                "attachment_sizes": [2500000, 1800000],
                "timestamp": "2024-01-15T14:30:00",
                "department": "Finance",
                "user_role": "Analyst"
            }
        }


# RESPONSE SCHEMAS  
class PredictResponse(BaseModel):
    """
    Ответ API с предсказанием
    
    ЧТО: Результат классификации инцидента
    ЗАЧЕМ: Клиент получает тип инцидента, серьёзность, рекомендации
    """
    incident_type: str = Field(..., description="Тип инцидента (email/file_transfer/web_upload)")
    incident_type_confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели (0-1)")
    
    severity: str = Field(..., description="Серьёзность (Low/Medium/High/Critical)")
    severity_confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели (0-1)")
    
    risk_level: str = Field(..., description="Уровень риска (Low/Medium/High/Critical)")
    
    detected_pii: List[str] = Field(default=[], description="Найденные типы PII")
    pii_count: int = Field(default=0, ge=0, description="Количество найденных PII")
    
    recommendation: str = Field(..., description="Рекомендация по действию")
    processing_time_ms: float = Field(..., ge=0, description="Время обработки в миллисекундах")
    
    class Config:
        json_schema_extra = {
            "example": {
                "incident_type": "email",
                "incident_type_confidence": 0.9876,
                "severity": "Critical",
                "severity_confidence": 0.8543,
                "risk_level": "Critical",
                "detected_pii": ["EMAIL", "FINANCIAL_DATA"],
                "pii_count": 2,
                "recommendation": "BLOCK - Critical severity incident with external recipient and financial data",
                "processing_time_ms": 18.5
            }
        }


class HealthResponse(BaseModel):
    """Health check ответ"""
    status: str = Field(..., description="Статус API (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Загружена ли ML модель")
    version: str = Field(default="1.0.0", description="Версия API")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }