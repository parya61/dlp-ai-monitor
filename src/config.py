"""
src/config.py

ЧТО: Центральная конфигурация проекта DLP AI Monitor
ЗАЧЕМ: Все настройки в одном месте, легко менять без изменения кода

АРХИТЕКТУРА:
- Config класс читает .env файл через pydantic-settings
- Автоматически определяет пути относительно корня проекта
- Предоставляет type-safe доступ к настройкам

ИСПОЛЬЗОВАНИЕ:
    from src.config import get_config
    
    config = get_config()
    print(config.CLAUDE_API_KEY)  # Читает из .env
    print(config.DATA_DIR)  # pathlib.Path объект
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Конфигурация DLP AI Monitor.
    
    Автоматически загружает переменные из .env файла.
    Если переменной нет в .env, использует значение по умолчанию.
    """
    
    # =========================================================================
    # ПУТИ К ДИРЕКТОРИЯМ
    # =========================================================================
    # Эти пути определяются автоматически относительно корня проекта
    
    # Корень проекта (родитель папки src/)
    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.resolve()
    )
    
    # Папка с данными
    DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data"
    )
    
    # Папка с обученными моделями
    MODELS_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "models"
    )
    
    # Папка с логами
    LOGS_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "logs"
    )
    
    # =========================================================================
    # CLAUDE API (для RAG модуля)
    # =========================================================================
    CLAUDE_API_KEY: str = Field(
        default="",
        description="API ключ для Claude (получить на console.anthropic.com)"
    )
    
    CLAUDE_MODEL: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Модель Claude для RAG"
    )
    
    # =========================================================================
    # MLFLOW (отслеживание экспериментов)
    # =========================================================================
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        description="URI для MLflow tracking server"
    )
    
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="dlp-ai-monitor",
        description="Название эксперимента в MLflow"
    )
    
    # =========================================================================
    # ML MODEL SETTINGS (параметры модели)
    # =========================================================================
    MODEL_TYPE: Literal["catboost", "randomforest", "xgboost"] = Field(
        default="catboost",
        description="Тип ML модели для классификации"
    )
    
    # CatBoost параметры
    CATBOOST_ITERATIONS: int = Field(
        default=1000,
        description="Количество итераций обучения CatBoost"
    )
    
    CATBOOST_LEARNING_RATE: float = Field(
        default=0.03,
        description="Learning rate для CatBoost"
    )
    
    CATBOOST_DEPTH: int = Field(
        default=6,
        description="Глубина деревьев CatBoost"
    )
    
    # =========================================================================
    # NLP SETTINGS (обнаружение PII)
    # =========================================================================
    SPACY_MODEL: str = Field(
        default="ru_core_news_lg",
        description="Модель spaCy для русского языка"
    )
    
    BERT_MODEL: str = Field(
        default="DeepPavlov/rubert-base-cased",
        description="BERT модель для обнаружения PII"
    )
    
    # =========================================================================
    # RAG SETTINGS (векторный поиск похожих инцидентов)
    # =========================================================================
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="Модель для создания эмбеддингов"
    )
    
    VECTOR_STORE_TYPE: Literal["faiss", "chroma"] = Field(
        default="faiss",
        description="Тип векторного хранилища"
    )
    
    TOP_K_SIMILAR: int = Field(
        default=5,
        description="Сколько похожих инцидентов искать"
    )
    
    # =========================================================================
    # LOGGING (логирование)
    # =========================================================================
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Уровень логирования"
    )
    
    # =========================================================================
    # API SETTINGS (FastAPI)
    # =========================================================================
    API_HOST: str = Field(
        default="0.0.0.0",
        description="Хост для FastAPI"
    )
    
    API_PORT: int = Field(
        default=8000,
        description="Порт для FastAPI"
    )
    
    API_RELOAD: bool = Field(
        default=True,
        description="Auto-reload для разработки"
    )
    
    # =========================================================================
    # SYNTHETIC DATA GENERATION (генерация данных)
    # =========================================================================
    SYNTHETIC_DATA_SIZE: int = Field(
        default=10000,
        description="Количество синтетических инцидентов для генерации"
    )
    
    # =========================================================================
    # PYDANTIC SETTINGS CONFIG
    # =========================================================================
    # Эта секция говорит pydantic, откуда читать настройки
    model_config = SettingsConfigDict(
        env_file=".env",  # Читать из .env файла
        env_file_encoding="utf-8",
        case_sensitive=True,  # Чувствительность к регистру
        extra="ignore",  # Игнорировать лишние поля в .env
    )
    
    def __init__(self, **kwargs):
        """
        Инициализация конфигурации.
        Создаёт необходимые директории, если их нет.
        """
        super().__init__(**kwargs)
        
        # Создаём директории, если их нет
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "raw").mkdir(exist_ok=True)
        (self.DATA_DIR / "processed").mkdir(exist_ok=True)
        (self.DATA_DIR / "synthetic").mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    def validate_api_keys(self) -> bool:
        """
        Проверяет, что все необходимые API ключи заданы.
        
        Returns:
            bool: True если все ключи заданы, иначе False
        """
        if not self.CLAUDE_API_KEY:
            return False
        return True
    
    def get_model_path(self, model_name: str) -> Path:
        """
        Возвращает полный путь к файлу модели.
        
        Args:
            model_name: Название модели (например, "catboost_classifier.cbm")
            
        Returns:
            Path: Полный путь к файлу модели
        """
        return self.MODELS_DIR / model_name
    
    def get_data_path(self, filename: str, subdir: str = "processed") -> Path:
        """
        Возвращает полный путь к файлу данных.
        
        Args:
            filename: Имя файла (например, "incidents.csv")
            subdir: Поддиректория в data/ ("raw", "processed", "synthetic")
            
        Returns:
            Path: Полный путь к файлу данных
        """
        return self.DATA_DIR / subdir / filename


# =============================================================================
# SINGLETON PATTERN - один экземпляр Config на всё приложение
# =============================================================================

_config_instance: Config | None = None


def get_config() -> Config:
    """
    Возвращает singleton экземпляр конфигурации.
    
    ЗАЧЕМ: Config создаётся один раз и используется везде.
    Это экономит ресурсы и гарантирует консистентность настроек.
    
    Returns:
        Config: Экземпляр конфигурации
        
    Usage:
        from src.config import get_config
        config = get_config()
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config()
    
    return _config_instance


# =============================================================================
# CONVENIENCE FUNCTION - для удобства
# =============================================================================

def reset_config() -> None:
    """
    Сбрасывает singleton конфигурации.
    Полезно для тестов.
    """
    global _config_instance
    _config_instance = None


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (для тестирования)
# =============================================================================

if __name__ == "__main__":
    # Получаем конфигурацию
    config = get_config()
    
    # Выводим основные параметры
    print("=" * 60)
    print("DLP AI Monitor - Configuration")
    print("=" * 60)
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"Data Directory: {config.DATA_DIR}")
    print(f"Models Directory: {config.MODELS_DIR}")
    print(f"Logs Directory: {config.LOGS_DIR}")
    print("-" * 60)
    print(f"Claude API Key Set: {'Yes' if config.CLAUDE_API_KEY else 'No'}")
    print(f"Claude Model: {config.CLAUDE_MODEL}")
    print("-" * 60)
    print(f"ML Model Type: {config.MODEL_TYPE}")
    print(f"CatBoost Iterations: {config.CATBOOST_ITERATIONS}")
    print(f"Log Level: {config.LOG_LEVEL}")
    print("=" * 60)
    
    # Проверяем API ключи
    if not config.validate_api_keys():
        print("\n⚠️  WARNING: Claude API key not set!")
        print("Create .env file and add: CLAUDE_API_KEY=your_key_here")