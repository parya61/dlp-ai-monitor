"""
Тесты для PII детектора (src/nlp/pii_detector.py).

ЧТО ТЕСТИРУЕМ:
1. Обнаружение банковских карт
2. Обнаружение паспортов РФ
3. Обнаружение ИНН
4. Обнаружение СНИЛС
5. Обнаружение телефонов
6. Обнаружение email
7. Расчёт risk_level
8. Работа с пустым текстом

АРХИТЕКТУРА ТЕСТОВ:
- Используем фикстуры из conftest.py
- Каждый тест проверяет одну функцию
- Тесты независимы друг от друга
"""

import pytest
from src.nlp.pii_detector import PIIDetector


class TestPIIDetector:
    """Набор тестов для PIIDetector класса."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Запускается перед каждым тестом.
        
        ЗАЧЕМ: Создаём детектор один раз для всех тестов в классе.
        autouse=True означает, что фикстура применяется автоматически.
        """
        self.detector = PIIDetector(use_ner=False)  # Без NER для скорости
    
    def test_detect_card_numbers(self):
        """
        Проверяем обнаружение номеров банковских карт.
        
        Тестируем разные форматы:
        - С пробелами: 1234 5678 9012 3456
        - С дефисами: 1234-5678-9012-3456
        - Без разделителей: 1234567890123456
        """
        # Arrange - готовим тестовые данные
        text = """
        Карта 1: 1234 5678 9012 3456
        Карта 2: 5469-1234-5678-9012
        Карта 3: 4111111111111111
        """
        
        # Act - вызываем детектор
        result = self.detector.detect(text)
        
        # Assert - проверяем результат
        assert result["has_card"] is True, "Должны обнаружить карты"
        assert len(result["cards"]) == 3, "Должны найти 3 карты"
        assert "risk_level" in result, "Должен быть risk_level"
    
    def test_detect_passport_numbers(self):
        """Проверяем обнаружение паспортов РФ."""
        text = "Паспорт: 1234 567890"
        
        result = self.detector.detect(text)
        
        assert result["has_passport"] is True
        assert len(result["passports"]) == 1
    
    def test_detect_inn(self):
        """Проверяем обнаружение ИНН (12 цифр)."""
        text = "ИНН: 123456789012"
        
        result = self.detector.detect(text)
        
        assert result["has_inn"] is True
        assert len(result["inn"]) == 1
    
    def test_detect_snils(self):
        """
        Проверяем обнаружение СНИЛС.
        
        Форматы:
        - С дефисами: 123-456-789 01
        - Без дефисов: 12345678901
        """
        text = "СНИЛС: 123-456-789 01"
        
        result = self.detector.detect(text)
        
        assert result["has_snils"] is True
        assert len(result["snils"]) == 1
    
    def test_detect_phones(self):
        """
        Проверяем обнаружение телефонов.
        
        Форматы:
        - +7 (900) 123-45-67
        - 8-800-555-35-35
        - +79001234567
        """
        text = """
        Телефон 1: +7 (900) 123-45-67
        Телефон 2: 8-800-555-35-35
        Телефон 3: +79001234567
        """
        
        result = self.detector.detect(text)
        
        assert result["has_phone"] is True
        assert len(result["phones"]) >= 2, "Должны найти минимум 2 телефона"
    
    def test_detect_emails(self):
        """Проверяем обнаружение email адресов."""
        text = "Контакты: user@example.com, support@company.ru"
        
        result = self.detector.detect(text)
        
        assert result["has_email"] is True
        assert len(result["emails"]) == 2
    
    def test_risk_level_calculation(self, sample_incident_with_pii):
        """
        Проверяем расчёт уровня риска (risk_level).
        
        ЛОГИКА:
        - 0-1 PII: Low
        - 2-3 PII: Medium
        - 4-5 PII: High
        - 6+ PII: Critical
        """
        text = sample_incident_with_pii["description"]
        
        result = self.detector.detect(text)
        
        # В тексте есть: карта, паспорт, телефон, email = 4 типа PII
        assert result["risk_level"] in ["High", "Critical"]
        assert result["pii_count"] >= 4
    
    def test_empty_text(self):
        """Проверяем работу с пустым текстом."""
        result = self.detector.detect("")
        
        # Все флаги должны быть False
        assert result["has_card"] is False
        assert result["has_passport"] is False
        assert result["has_inn"] is False
        assert result["has_snils"] is False
        assert result["has_phone"] is False
        assert result["has_email"] is False
        assert result["pii_count"] == 0
        assert result["risk_level"] == "Low"
    
    def test_no_pii_detected(self):
        """Проверяем текст без PII."""
        text = "Обычный текст без конфиденциальных данных."
        
        result = self.detector.detect(text)
        
        assert result["pii_count"] == 0
        assert result["risk_level"] == "Low"
    
    def test_multiple_same_type_pii(self):
        """
        Проверяем обнаружение нескольких PII одного типа.
        
        Например, несколько карт в одном тексте.
        """
        text = """
        Карта 1: 1234 5678 9012 3456
        Карта 2: 5469 1234 5678 9012
        Карта 3: 4111 1111 1111 1111
        """
        
        result = self.detector.detect(text)
        
        assert len(result["cards"]) == 3
        assert result["pii_count"] >= 3


class TestPIIDetectorWithFixtures:
    """Тесты с использованием фикстур из conftest.py."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.detector = PIIDetector(use_ner=False)
    
    def test_incident_with_pii(self, sample_incident_with_pii):
        """Тестируем на реалистичном инциденте с PII."""
        text = sample_incident_with_pii["description"]
        
        result = self.detector.detect(text)
        
        # Инцидент содержит критичные данные
        assert result["has_card"] is True
        assert result["has_passport"] is True
        assert result["risk_level"] in ["High", "Critical"]
    
    def test_incident_without_pii(self, sample_incident):
        """Тестируем на инциденте без PII."""
        text = sample_incident["description"]
        
        result = self.detector.detect(text)
        
        # Обычное описание, без PII
        assert result["pii_count"] == 0 or result["risk_level"] == "Low"
    
    def test_full_text_sample(self, sample_text_with_pii):
        """Тестируем на полном образце текста с разными PII."""
        result = self.detector.detect(sample_text_with_pii)
        
        # В тексте есть почти все типы PII
        assert result["has_passport"] is True
        assert result["has_inn"] is True
        assert result["has_snils"] is True
        assert result["has_phone"] is True
        assert result["has_email"] is True
        assert result["has_card"] is True
        assert result["pii_count"] >= 6
        assert result["risk_level"] == "Critical"


# Параметризованные тесты (запускаются с разными данными)
@pytest.mark.parametrize("card_number,expected", [
    ("1234 5678 9012 3456", True),   # С пробелами
    ("1234-5678-9012-3456", True),   # С дефисами
    ("1234567890123456", True),      # Без разделителей
    ("1234 5678", False),            # Слишком короткий
    ("abcd efgh ijkl mnop", False),  # Не цифры
])
def test_card_patterns(card_number, expected):
    """
    Параметризованный тест для различных форматов карт.
    
    ЗАЧЕМ: Проверить много вариантов одним тестом.
    pytest запустит этот тест 5 раз с разными параметрами.
    """
    detector = PIIDetector(use_ner=False)
    result = detector.detect(f"Номер карты: {card_number}")
    
    assert result["has_card"] is expected


if __name__ == "__main__":
    # Запуск тестов напрямую (для отладки)
    pytest.main([__file__, "-v"])
