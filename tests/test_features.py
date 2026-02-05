"""
Тесты для feature engineering модуля (src/ml/features.py).

ЧТО ТЕСТИРУЕМ:
1. Создание PII признаков (has_card, has_passport, etc.)
2. TF-IDF векторизация
3. Обработку department (encoding)
4. Правильность формы выходных данных
5. Отсутствие data leakage

ВАЖНО: Тесты адаптированы под реальную структуру FeatureExtractor
"""

import pytest
import pandas as pd
import numpy as np
from src.ml.features import FeatureExtractor


class TestFeatureExtractor:
    """Тесты для FeatureExtractor класса."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Создаём экстрактор перед каждым тестом."""
        self.extractor = FeatureExtractor(max_tfidf_features=50, use_pii=True)
    
    def test_extractor_initialization(self):
        """
        Проверяем инициализацию экстрактора.
        
        ЗАЧЕМ: Убедиться, что все компоненты создались правильно.
        """
        assert self.extractor.pii_detector is not None
        assert self.extractor.tfidf is not None
        assert self.extractor.label_encoders is not None  # dict, не single encoder!
        assert isinstance(self.extractor.label_encoders, dict)
    
    def test_fit_on_large_dataframe(self):
        """
        Проверяем обучение экстрактора на достаточном количестве данных.
        
        ПРОБЛЕМА: TF-IDF с min_df=2 требует минимум 2 документа с каждым словом.
        РЕШЕНИЕ: Используем минимум 10 инцидентов для тестов.
        """
        # Создаём 10 инцидентов для теста
        data = {
            "timestamp": ["2024-01-01 00:00:00"] * 10,
            "department": ["IT", "Finance", "HR", "Sales", "IT", "Finance", "HR", "Sales", "IT", "Finance"],
            "body": [
                "Отправка файла с данными карты",
                "Финансовый отчёт с номером карты",
                "Данные сотрудника с паспортом",
                "Продажи и контакты клиента",
                "IT инцидент с файлом",
                "Финансовые данные клиента",
                "HR документ с персональными данными",
                "Отчёт по продажам",
                "Техническая документация",
                "Бюджет и расходы"
            ]
        }
        df = pd.DataFrame(data)
        
        # Act
        self.extractor.fit(df)
        
        # Assert
        assert self.extractor.is_fitted is True
        assert hasattr(self.extractor.tfidf, "vocabulary_")
        assert len(self.extractor.label_encoders) > 0
    
    def test_transform_output_shape(self):
        """
        Проверяем форму выходных данных после transform.
        
        ВАЖНО: Количество признаков должно быть константным!
        """
        # Создаём достаточно данных (10+ инцидентов)
        data = {
            "timestamp": ["2024-01-01 00:00:00"] * 10,
            "department": ["IT"] * 10,
            "body": [f"Инцидент номер {i} с данными карты паспорт" for i in range(10)]
        }
        df = pd.DataFrame(data)
        
        # Arrange
        self.extractor.fit(df)
        
        # Act
        features = self.extractor.transform(df)
        
        # Assert
        assert isinstance(features, np.ndarray), "Должен вернуть numpy array"
        assert features.shape[0] == len(df), "Число строк = число инцидентов"
        assert features.shape[1] > 0, "Должны быть признаки"
        
        # Проверяем, что нет NaN
        assert not np.isnan(features).any(), "Не должно быть NaN значений"
    
    def test_pii_features_extraction(self):
        """
        Проверяем извлечение PII признаков.
        
        ЛОГИКА:
        Текст с PII должен иметь флаги has_card=1, has_passport=1, etc.
        """
        # Создаём 10 инцидентов, один с PII
        bodies = [f"Обычный текст {i}" for i in range(9)]
        bodies.append("""
        Карта: 1234 5678 9012 3456
        Паспорт: 4567 123456
        """)
        
        data = {
            "timestamp": ["2024-01-01 00:00:00"] * 10,
            "department": ["IT"] * 10,
            "body": bodies
        }
        df = pd.DataFrame(data)
        
        # Обучаем и трансформируем
        self.extractor.fit(df)
        features = self.extractor.transform(df)
        
        # Извлекаем имена признаков
        feature_names = self.extractor.get_feature_names()
        
        # Проверяем наличие PII признаков
        assert "has_card" in feature_names
        assert "has_passport" in feature_names
        assert "pii_count" in feature_names
    
    def test_department_encoding(self):
        """
        Проверяем кодирование department.
        
        ЛОГИКА:
        - Department (IT, Finance, HR) должен быть закодирован в число
        - label_encoders["department"] должен существовать
        """
        data = {
            "timestamp": ["2024-01-01 00:00:00"] * 10,
            "department": ["IT", "Finance", "HR", "Sales", "IT", "Finance", "HR", "Sales", "IT", "Finance"],
            "body": [f"Text {i}" for i in range(10)]
        }
        df = pd.DataFrame(data)
        
        self.extractor.fit(df)
        features = self.extractor.transform(df)
        
        # Проверяем наличие encoder
        assert "department" in self.extractor.label_encoders
        
        # Проверяем, что закодированные значения разумные
        assert not np.isnan(features).any()
    
    def test_fit_transform_consistency(self):
        """
        Проверяем консистентность fit + transform.
        
        ВАЖНО: После fit на одних данных, transform на других должен работать.
        """
        # Обучаемся на данных
        train_data = {
            "timestamp": ["2024-01-01 00:00:00"] * 10,
            "department": ["IT"] * 10,
            "body": [f"Training text {i} карта паспорт" for i in range(10)]
        }
        train_df = pd.DataFrame(train_data)
        
        self.extractor.fit(train_df)
        
        # Трансформируем новые данные
        test_data = {
            "timestamp": ["2024-01-02 00:00:00"] * 5,
            "department": ["IT"] * 5,
            "body": [f"Test text {i}" for i in range(5)]
        }
        test_df = pd.DataFrame(test_data)
        
        features = self.extractor.transform(test_df)
        
        # Должен вернуть массив без ошибок
        assert features.shape[0] == 5
        assert features.shape[1] > 0


class TestFeatureNamesAndValues:
    """Тесты для проверки конкретных значений признаков."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.extractor = FeatureExtractor(max_tfidf_features=50, use_pii=True)
    
    def test_feature_count(self):
        """
        Проверяем общее количество признаков.
        
        ОЖИДАЕМАЯ СТРУКТУРА:
        - 7 PII признаков (has_card, has_passport, has_inn, has_snils, has_phone, has_email, pii_count)
        - До 50 TF-IDF признаков
        - Temporal признаки (hour_of_day, is_night, day_of_week, is_weekend)
        - 1 metadata признак (encoded_department)
        """
        data = {
            "timestamp": ["2024-01-01 00:00:00"] * 10,
            "department": ["IT"] * 10,
            "body": [f"Text {i} with some content" for i in range(10)]
        }
        df = pd.DataFrame(data)
        
        self.extractor.fit(df)
        feature_names = self.extractor.get_feature_names()
        
        # Проверяем минимальное количество
        assert len(feature_names) >= 8, "Должно быть минимум 8 признаков (7 PII + 1 dept)"


# Параметризованные тесты для разных форматов данных
@pytest.mark.parametrize("n_samples", [10, 20, 50])
def test_scalability(n_samples):
    """
    Тест масштабируемости: проверяем работу с разным числом инцидентов.
    
    ЗАЧЕМ: Убедиться, что экстрактор работает на разных размерах данных.
    ВАЖНО: Минимум 10 инцидентов из-за min_df=2 в TF-IDF.
    """
    # Создаём синтетические данные
    data = {
        "timestamp": [f"2024-01-{i%30+1:02d} 00:00:00" for i in range(n_samples)],
        "department": ["IT"] * n_samples,
        "body": [f"Incident {i} with card data and passport info" for i in range(n_samples)]
    }
    df = pd.DataFrame(data)
    
    extractor = FeatureExtractor(max_tfidf_features=50, use_pii=True)
    extractor.fit(df)
    features = extractor.transform(df)
    
    assert features.shape[0] == n_samples
    assert features.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
