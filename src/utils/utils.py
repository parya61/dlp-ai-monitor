"""
src/utils.py

ЧТО: Общие утилиты для всего проекта
ЗАЧЕМ: Избегать дублирования кода (DRY principle)

СОДЕРЖИТ:
- setup_logging() - настройка логирования для модуля
- get_logger() - получение logger объекта
- load_json() - загрузка JSON файла
- save_json() - сохранение в JSON
- load_pickle() - загрузка pickle файла
- save_pickle() - сохранение в pickle
- timer() - декоратор для измерения времени выполнения функции

ИСПОЛЬЗОВАНИЕ:
    from src.utils import get_logger, timer
    
    logger = get_logger(__name__)
    
    @timer
    def my_function():
        logger.info("Doing something...")
"""

import json
import logging
import pickle
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from src.config import get_config


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logging(
    name: str = "dlp_ai_monitor",
    log_file: str | None = None,
) -> logging.Logger:
    """
    Настраивает логирование для модуля.
    
    ЧТО ДЕЛАЕТ:
    - Создаёт logger с заданным именем
    - Настраивает формат вывода
    - Логи идут в консоль и (опционально) в файл
    - Уровень логирования берётся из config
    
    Args:
        name: Имя logger'а (обычно __name__ модуля)
        log_file: Путь к файлу для сохранения логов (опционально)
    
    Returns:
        logging.Logger: Настроенный logger
    
    Example:
        logger = setup_logging(__name__)
        logger.info("Application started")
    """
    config = get_config()
    
    # Создаём logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Если уже есть обработчики, не добавляем новые
    # (чтобы избежать дублирования при повторных вызовах)
    if logger.handlers:
        return logger
    
    # Формат сообщений
    # Пример: 2024-01-26 10:30:45 - INFO - src.data.loader - Loading data...
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (вывод в консоль)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (сохранение в файл) - опционально
    if log_file:
        file_path = config.LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Удобная функция для получения logger'а.
    
    ЗАЧЕМ: Чтобы в каждом модуле просто писать:
        from src.utils import get_logger
        logger = get_logger(__name__)
    
    Args:
        name: Имя logger'а (обычно __name__)
    
    Returns:
        logging.Logger: Настроенный logger
    """
    return setup_logging(name)


# =============================================================================
# FILE I/O UTILITIES
# =============================================================================

def load_json(filepath: str | Path) -> dict | list:
    """
    Загружает данные из JSON файла.
    
    Args:
        filepath: Путь к JSON файлу
    
    Returns:
        dict | list: Загруженные данные
    
    Raises:
        FileNotFoundError: Если файл не найден
        json.JSONDecodeError: Если файл не валидный JSON
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: dict | list,
    filepath: str | Path,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Сохраняет данные в JSON файл.
    
    Args:
        data: Данные для сохранения (dict или list)
        filepath: Путь к файлу для сохранения
        indent: Отступы для красивого форматирования (default: 2)
        ensure_ascii: Если False, сохраняет UTF-8 символы (кириллицу)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def load_pickle(filepath: str | Path) -> Any:
    """
    Загружает объект из pickle файла.
    
    ЗАЧЕМ: Pickle используется для сохранения обученных моделей,
    preprocessor'ов и других Python объектов.
    
    Args:
        filepath: Путь к pickle файлу
    
    Returns:
        Any: Загруженный объект
    
    Example:
        model = load_pickle("data/models/catboost_model.pkl")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, filepath: str | Path) -> None:
    """
    Сохраняет объект в pickle файл.
    
    Args:
        obj: Объект для сохранения
        filepath: Путь к файлу для сохранения
    
    Example:
        save_pickle(model, "data/models/catboost_model.pkl")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


# =============================================================================
# PERFORMANCE UTILITIES
# =============================================================================

def timer(func: Callable) -> Callable:
    """
    Декоратор для измерения времени выполнения функции.
    
    ЗАЧЕМ: Полезно для профилирования кода и поиска узких мест.
    
    Usage:
        @timer
        def train_model():
            # обучение модели
            pass
        
        train_model()  # Выведет: Function 'train_model' took 10.5 seconds
    
    Args:
        func: Функция для измерения
    
    Returns:
        Callable: Обёрнутая функция
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        start_time = time.time()
        logger.info(f"Starting '{func.__name__}'...")
        
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(
                f"Finished '{func.__name__}' in {elapsed_time:.2f} seconds"
            )
            return result
        
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                f"Error in '{func.__name__}' after {elapsed_time:.2f} seconds: {e}"
            )
            raise
    
    return wrapper


# =============================================================================
# DATA UTILITIES
# =============================================================================

def ensure_dir(directory: str | Path) -> Path:
    """
    Создаёт директорию, если её нет.
    
    Args:
        directory: Путь к директории
    
    Returns:
        Path: Path объект директории
    
    Example:
        output_dir = ensure_dir("data/output")
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_size(filepath: str | Path) -> str:
    """
    Возвращает размер файла в человеко-читаемом формате.
    
    Args:
        filepath: Путь к файлу
    
    Returns:
        str: Размер файла (например, "1.5 MB")
    
    Example:
        size = get_file_size("data/incidents.csv")
        print(f"File size: {size}")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return "File not found"
    
    size_bytes = filepath.stat().st_size
    
    # Конвертируем в KB, MB, GB
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} PB"


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (для тестирования)
# =============================================================================

if __name__ == "__main__":
    # Тестируем логирование
    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("\n" + "=" * 60)
    
    # Тестируем timer декоратор
    @timer
    def example_function():
        """Пример функции для теста."""
        logger.info("Doing some work...")
        time.sleep(2)  # Имитация работы
        return "Done!"
    
    result = example_function()
    print(f"Result: {result}")
    
    print("=" * 60)
    
    # Тестируем работу с файлами
    test_data = {"name": "Test", "value": 42}
    test_file = Path("data/test.json")
    
    # Сохраняем
    save_json(test_data, test_file)
    logger.info(f"Saved test data to {test_file}")
    
    # Загружаем
    loaded_data = load_json(test_file)
    logger.info(f"Loaded data: {loaded_data}")
    
    # Размер файла
    size = get_file_size(test_file)
    logger.info(f"File size: {size}")
    
    # Удаляем тестовый файл
    test_file.unlink()
    logger.info("Cleaned up test file")