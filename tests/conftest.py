"""
Общие фикстуры для pytest тестов.

ЧТО: Переиспользуемые тестовые данные (фикстуры)
ЗАЧЕМ: Избежать дублирования кода в тестах

ФИКСТУРЫ:
- sample_incident: Простой тестовый инцидент
- sample_incident_with_pii: Инцидент с PII данными
- sample_incidents_list: Список инцидентов для batch тестирования
"""

import pytest
import pandas as pd
from datetime import datetime


@pytest.fixture
def sample_incident() -> dict:
    """
    Простой тестовый инцидент без PII.
    
    Использование:
        def test_something(sample_incident):
            result = process(sample_incident)
            assert result is not None
    """
    return {
        "incident_id": 1,
        "timestamp": "2024-01-15 10:30:00",
        "incident_type": "email",
        "severity": "Medium",
        "user": "Иван Петров",
        "department": "IT",
        "description": "Отправка файла коллеге",
        "source_ip": "192.168.1.100",
        "destination": "colleague@company.com",
        "file_name": "report.xlsx",
        "file_size_kb": 125.5,
        "action": "send",
        "is_blocked": False,
        "detection_method": "rule_based",
        "confidence_score": 0.75
    }


@pytest.fixture
def sample_incident_with_pii() -> dict:
    """
    Инцидент с персональными данными (PII).
    
    Содержит:
    - Номер банковской карты
    - Номер паспорта
    - Телефон
    - Email
    """
    return {
        "incident_id": 2,
        "timestamp": "2024-01-15 14:20:00",
        "incident_type": "email",
        "severity": "Critical",
        "user": "Мария Сидорова",
        "department": "Finance",
        "description": """
        Отправлен файл с конфиденциальными данными:
        Карта: 1234 5678 9012 3456
        Паспорт: 4567 123456
        Телефон: +7 900 123-45-67
        Email: client@example.com
        """,
        "source_ip": "192.168.1.101",
        "destination": "external@gmail.com",
        "file_name": "client_data.xlsx",
        "file_size_kb": 450.0,
        "action": "send",
        "is_blocked": True,
        "detection_method": "ml_based",
        "confidence_score": 0.95
    }


@pytest.fixture
def sample_incidents_list(sample_incident, sample_incident_with_pii) -> list:
    """
    Список из нескольких инцидентов для batch тестирования.
    """
    # Добавляем третий инцидент
    usb_incident = {
        "incident_id": 3,
        "timestamp": "2024-01-15 16:45:00",
        "incident_type": "usb",
        "severity": "High",
        "user": "Алексей Смирнов",
        "department": "Sales",
        "description": "Копирование файлов на USB накопитель",
        "source_ip": "192.168.1.102",
        "destination": "USB_Device_XYZ",
        "file_name": "sales_report.pdf",
        "file_size_kb": 2500.0,
        "action": "copy",
        "is_blocked": False,
        "detection_method": "rule_based",
        "confidence_score": 0.68
    }
    
    return [sample_incident, sample_incident_with_pii, usb_incident]


@pytest.fixture
def sample_dataframe(sample_incidents_list) -> pd.DataFrame:
    """
    DataFrame с тестовыми инцидентами.
    
    Использование в тестах:
        def test_dataframe_processing(sample_dataframe):
            assert len(sample_dataframe) == 3
            assert "incident_type" in sample_dataframe.columns
    """
    return pd.DataFrame(sample_incidents_list)


@pytest.fixture
def sample_text_with_pii() -> str:
    """
    Текст с различными типами PII для тестирования детектора.
    """
    return """
    Конфиденциальная информация клиента:
    
    ФИО: Иванов Иван Иванович
    Паспорт: 1234 567890
    ИНН: 123456789012
    СНИЛС: 123-456-789 01
    Телефон: +7 (900) 123-45-67
    Email: ivanov@example.com
    Банковская карта: 5469 1234 5678 9012
    
    Дополнительные контакты:
    Телефон 2: 8-800-555-35-35
    Email 2: support@bank.ru
    """