"""
Тесты для FastAPI endpoints (api/main.py).

ЧТО ТЕСТИРУЕМ:
1. GET / - информация об API
2. GET /health - проверка здоровья
3. POST /api/v1/predict - предсказание
4. Валидация входных данных (Pydantic)
5. Коды ответов (200, 422, 500)

АРХИТЕКТУРА:
- Используем TestClient от FastAPI
- Тесты НЕ запускают реальный сервер
- Модель загружается один раз для всех тестов (фикстура)
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    """
    Создаём тестовый клиент для FastAPI.
    
    ЗАЧЕМ: scope="module" - создаётся один раз для всех тестов в этом файле.
    Экономит время (не нужно загружать модель перед каждым тестом).
    """
    return TestClient(app)


class TestRootEndpoint:
    """Тесты для корневого эндпоинта (GET /)."""
    
    def test_root_returns_200(self, client):
        """Проверяем, что корневой эндпоинт возвращает 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_json(self, client):
        """Проверяем, что ответ в формате JSON."""
        response = client.get("/")
        data = response.json()
        
        assert isinstance(data, dict)
        assert "name" in data
        assert "version" in data
    
    def test_root_has_correct_info(self, client):
        """Проверяем содержимое ответа."""
        response = client.get("/")
        data = response.json()
        
        assert data["name"] == "DLP AI Monitor API"
        assert "version" in data


class TestHealthEndpoint:
    """Тесты для health check эндпоинта (GET /health)."""
    
    def test_health_returns_200(self, client):
        """Проверяем, что health check возвращает 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_status_ok(self, client):
        """
        Проверяем, что статус = "healthy".
        
        ЗАЧЕМ: В production мониторинг пингует /health каждые N секунд.
        Если status != "healthy" - алерт!
        """
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"
    
    def test_health_has_model_status(self, client):
        """Проверяем наличие информации о модели."""
        response = client.get("/health")
        data = response.json()
        
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


class TestPredictEndpoint:
    """Тесты для предсказания (POST /api/v1/predict)."""
    
    @pytest.fixture
    def valid_request(self):
        """
        Валидный запрос для предсказания.
        
        СТРУКТУРА: Соответствует PredictRequest из api/schemas.py
        """
        return {
            "user_email": "employee@company.com",
            "recipient_email": "colleague@company.com",
            "subject": "Project Report",
            "body": "Please find attached the quarterly report with financial data.",
            "attachment_names": ["report.xlsx"],
            "attachment_types": ["application/vnd.ms-excel"],
            "attachment_sizes": [125500],
            "timestamp": "2024-01-15T10:30:00",
            "department": "IT",
            "user_role": "Analyst"
        }
    
    def test_predict_returns_200(self, client, valid_request):
        """Проверяем успешное предсказание."""
        response = client.post("/api/v1/predict", json=valid_request)
        assert response.status_code == 200
    
    def test_predict_response_structure(self, client, valid_request):
        """
        Проверяем структуру ответа.
        
        ОЖИДАЕТСЯ:
        {
            "incident_type": "email",
            "incident_type_confidence": 0.95,
            "severity": "Medium",
            "severity_confidence": 0.75,
            "processing_time_ms": 18.5
        }
        """
        response = client.post("/api/v1/predict", json=valid_request)
        data = response.json()
        
        # Проверяем наличие всех ключей
        assert "incident_type" in data
        assert "incident_type_confidence" in data
        assert "severity" in data
        assert "severity_confidence" in data
        assert "processing_time_ms" in data
    
    def test_predict_incident_type_valid(self, client, valid_request):
        """
        Проверяем, что предсказанный тип инцидента валидный.
        
        Допустимые типы: email, usb, cloud, printer
        """
        response = client.post("/api/v1/predict", json=valid_request)
        data = response.json()
        
        valid_types = ["email", "usb", "cloud", "printer"]
        assert data["incident_type"] in valid_types
    
    def test_predict_severity_valid(self, client, valid_request):
        """
        Проверяем, что предсказанная критичность валидная.
        
        Допустимые уровни: Low, Medium, High, Critical
        """
        response = client.post("/api/v1/predict", json=valid_request)
        data = response.json()
        
        valid_severities = ["Low", "Medium", "High", "Critical"]
        assert data["severity"] in valid_severities
    
    def test_predict_confidence_range(self, client, valid_request):
        """
        Проверяем, что confidence в диапазоне [0, 1].
        
        ВАЖНО: CatBoost predict_proba возвращает вероятности от 0 до 1.
        """
        response = client.post("/api/v1/predict", json=valid_request)
        data = response.json()
        
        assert 0 <= data["incident_type_confidence"] <= 1
        assert 0 <= data["severity_confidence"] <= 1
    
    def test_predict_processing_time_reasonable(self, client, valid_request):
        """
        Проверяем, что время обработки разумное (<1000ms).
        
        ЗАЧЕМ: В production нужен быстрый ответ.
        Если processing_time > 1000ms - что-то не так!
        """
        response = client.post("/api/v1/predict", json=valid_request)
        data = response.json()
        
        assert data["processing_time_ms"] < 1000, "Слишком медленная обработка!"
    
    def test_predict_with_pii(self, client):
        """
        Тестируем предсказание для инцидента с PII.
        
        ОЖИДАНИЕ: Должен предсказать высокую критичность.
        """
        request = {
            "user_email": "maria@company.com",
            "recipient_email": "external@gmail.com",
            "subject": "Confidential Client Data",
            "body": """
            Client information:
            Card: 1234 5678 9012 3456
            Passport: 4567 123456
            Phone: +7 900 123-45-67
            Email: client@example.com
            """,
            "attachment_names": ["client_data.xlsx"],
            "attachment_types": ["application/vnd.ms-excel"],
            "attachment_sizes": [450000],
            "timestamp": "2024-01-15T14:20:00",
            "department": "Finance"
        }
        
        response = client.post("/api/v1/predict", json=request)
        data = response.json()
        
        # Инцидент с PII должен быть критичным
        assert data["severity"] in ["High", "Critical"]


class TestAPIValidation:
    """Тесты для валидации входных данных (Pydantic)."""
    
    def test_missing_required_field(self, client):
        """
        Проверяем ошибку при отсутствии обязательного поля.
        
        Должен вернуть 422 Unprocessable Entity.
        """
        # Запрос без subject (обязательное поле)
        invalid_request = {
            "user_email": "test@company.com",
            "recipient_email": "test@test.com",
            # "subject": отсутствует!
            "body": "Test body",
            "timestamp": "2024-01-01T00:00:00"
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_invalid_field_type(self, client):
        """
        Проверяем ошибку при неправильном типе поля.
        
        Например, attachment_sizes должен быть List[int], не string.
        """
        invalid_request = {
            "user_email": "test@company.com",
            "recipient_email": "test@test.com",
            "subject": "Test",
            "body": "Test body",
            "attachment_sizes": "NOT_A_LIST",  # Ошибка типа!
            "timestamp": "2024-01-01T00:00:00"
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_empty_description(self, client):
        """
        Проверяем работу с пустым body (минимум 1 символ по схеме).
        
        Должно вернуть 422 (body требует min_length=1).
        """
        request = {
            "user_email": "test@company.com",
            "recipient_email": "test@test.com",
            "subject": "Test",
            "body": "",  # Пустое тело (нарушает min_length=1)
            "timestamp": "2024-01-01T00:00:00"
        }
        
        response = client.post("/api/v1/predict", json=request)
        # Должен вернуть 422 (валидация не пройдена)
        assert response.status_code == 422


class TestAPIPerformance:
    """Тесты производительности API."""
    
    def test_concurrent_requests(self, client, valid_request):
        """
        Тестируем несколько одновременных запросов.
        
        ЗАЧЕМ: В production API должен обрабатывать много запросов параллельно.
        """
        # Отправляем 10 запросов подряд
        responses = []
        for _ in range(10):
            response = client.post("/api/v1/predict", json=valid_request)
            responses.append(response)
        
        # Все должны вернуть 200
        for response in responses:
            assert response.status_code == 200


# Маркер для медленных тестов (запускаются опционально)
@pytest.mark.slow
def test_load_testing(client, valid_request):
    """
    Load testing: отправляем 100 запросов.
    
    ЗАЧЕМ: Проверить, что API не падает под нагрузкой.
    
    Запуск: pytest -m slow
    """
    for i in range(100):
        response = client.post("/api/v1/predict", json=valid_request)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
