"""
src/data/loader.py

ЧТО: Загрузка данных из разных форматов
ЗАЧЕМ: Единый интерфейс для работы с CSV, Parquet, Excel, JSON

ОСНОВНЫЕ ВОЗМОЖНОСТИ:
- Загрузка из разных форматов
- Автоматическое определение формата по расширению
- Валидация данных
- Получение информации о датасете

ИСПОЛЬЗОВАНИЕ:
    from src.data import DataLoader
    
    loader = DataLoader()
    
    # Загрузить CSV
    df = loader.load_csv("data/synthetic/incidents.csv")
    
    # Автоматически определить формат
    df = loader.load_auto("data/incidents.parquet")
    
    # Получить информацию
    info = loader.get_data_info(df)
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import get_config
from src.utils import get_logger

# Инициализация
logger = get_logger(__name__)
config = get_config()


class DataLoader:
    """
    Класс для загрузки данных из разных форматов.
    
    Поддерживаемые форматы:
    - CSV (.csv)
    - Parquet (.parquet)
    - Excel (.xlsx, .xls)
    - JSON (.json)
    """
    
    def __init__(self):
        """Инициализация загрузчика данных."""
        logger.info("Initialized DataLoader")
        
        # Поддерживаемые форматы
        self.supported_formats = [".csv", ".parquet", ".xlsx", ".xls", ".json"]
    
    def load_csv(
        self,
        filepath: str | Path,
        encoding: str = "utf-8",
        **kwargs
    ) -> pd.DataFrame:
        """
        Загружает данные из CSV файла.
        
        Args:
            filepath: Путь к CSV файлу
            encoding: Кодировка файла (default: utf-8)
            **kwargs: Дополнительные параметры для pd.read_csv()
        
        Returns:
            pd.DataFrame: Загруженные данные
        
        Example:
            df = loader.load_csv("data/synthetic/incidents.csv")
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading CSV from {filepath}...")
        
        try:
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        
        except UnicodeDecodeError:
            # Пробуем другую кодировку (utf-8-sig для файлов с BOM)
            logger.warning(f"Failed with {encoding}, trying utf-8-sig...")
            df = pd.read_csv(filepath, encoding="utf-8-sig", **kwargs)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def load_parquet(
        self,
        filepath: str | Path,
        **kwargs
    ) -> pd.DataFrame:
        """
        Загружает данные из Parquet файла.
        
        Parquet - это колоночный формат хранения данных.
        Преимущества: быстрее чем CSV, меньше места, сохраняет типы данных.
        
        Args:
            filepath: Путь к Parquet файлу
            **kwargs: Дополнительные параметры для pd.read_parquet()
        
        Returns:
            pd.DataFrame: Загруженные данные
        
        Example:
            df = loader.load_parquet("data/processed/incidents.parquet")
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading Parquet from {filepath}...")
        
        try:
            df = pd.read_parquet(filepath, **kwargs)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load Parquet: {e}")
            raise
    
    def load_excel(
        self,
        filepath: str | Path,
        sheet_name: str | int = 0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Загружает данные из Excel файла.
        
        Args:
            filepath: Путь к Excel файлу (.xlsx или .xls)
            sheet_name: Название или индекс листа (default: 0 - первый лист)
            **kwargs: Дополнительные параметры для pd.read_excel()
        
        Returns:
            pd.DataFrame: Загруженные данные
        
        Example:
            df = loader.load_excel("data/incidents.xlsx", sheet_name="Sheet1")
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading Excel from {filepath}, sheet: {sheet_name}...")
        
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load Excel: {e}")
            raise
    
    def load_json(
        self,
        filepath: str | Path,
        orient: str = "records",
        **kwargs
    ) -> pd.DataFrame:
        """
        Загружает данные из JSON файла.
        
        Args:
            filepath: Путь к JSON файлу
            orient: Формат JSON ('records', 'split', 'index', 'columns', 'values')
            **kwargs: Дополнительные параметры для pd.read_json()
        
        Returns:
            pd.DataFrame: Загруженные данные
        
        Example:
            df = loader.load_json("data/incidents.json")
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading JSON from {filepath}...")
        
        try:
            df = pd.read_json(filepath, orient=orient, **kwargs)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            raise
    
    def load_auto(self, filepath: str | Path, **kwargs) -> pd.DataFrame:
        """
        Автоматически определяет формат файла и загружает данные.
        
        ЗАЧЕМ: Не нужно помнить, какой метод использовать для каждого формата.
        
        Args:
            filepath: Путь к файлу
            **kwargs: Дополнительные параметры для соответствующего метода
        
        Returns:
            pd.DataFrame: Загруженные данные
        
        Example:
            # Автоматически определит формат
            df = loader.load_auto("data/incidents.csv")
            df = loader.load_auto("data/incidents.parquet")
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Определяем формат по расширению
        suffix = filepath.suffix.lower()
        
        if suffix == ".csv":
            return self.load_csv(filepath, **kwargs)
        elif suffix == ".parquet":
            return self.load_parquet(filepath, **kwargs)
        elif suffix in [".xlsx", ".xls"]:
            return self.load_excel(filepath, **kwargs)
        elif suffix == ".json":
            return self.load_json(filepath, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {self.supported_formats}"
            )
    
    @staticmethod
    def get_data_info(df: pd.DataFrame) -> Dict:
        """
        Возвращает информацию о датасете.
        
        ЗАЧЕМ: Быстро понять, что за данные загружены.
        
        Args:
            df: DataFrame для анализа
        
        Returns:
            Dict с информацией:
                - n_rows: количество строк
                - n_columns: количество колонок
                - columns: список колонок
                - dtypes: типы данных
                - missing_values: количество пропусков
                - memory_usage: использование памяти
        
        Example:
            info = loader.get_data_info(df)
            print(f"Rows: {info['n_rows']}, Columns: {info['n_columns']}")
        """
        # Количество пропусков по колонкам
        missing = df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()
        
        # Использование памяти
        memory_bytes = df.memory_usage(deep=True).sum()
        memory_mb = memory_bytes / (1024 * 1024)
        
        info = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": missing_dict if missing_dict else "No missing values",
            "memory_usage_mb": round(memory_mb, 2),
        }
        
        return info
    
    @staticmethod
    def save_csv(
        df: pd.DataFrame,
        filepath: str | Path,
        encoding: str = "utf-8-sig",
        index: bool = False,
        **kwargs
    ) -> None:
        """
        Сохраняет DataFrame в CSV файл.
        
        Args:
            df: DataFrame для сохранения
            filepath: Путь к файлу
            encoding: Кодировка (default: utf-8-sig для совместимости с Excel)
            index: Сохранять ли индекс (default: False)
            **kwargs: Дополнительные параметры для df.to_csv()
        
        Example:
            loader.save_csv(df, "data/output/result.csv")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving CSV to {filepath}...")
        df.to_csv(filepath, encoding=encoding, index=index, **kwargs)
        logger.info(f"Saved {len(df)} rows to {filepath}")
    
    @staticmethod
    def save_parquet(
        df: pd.DataFrame,
        filepath: str | Path,
        **kwargs
    ) -> None:
        """
        Сохраняет DataFrame в Parquet файл.
        
        Args:
            df: DataFrame для сохранения
            filepath: Путь к файлу
            **kwargs: Дополнительные параметры для df.to_parquet()
        
        Example:
            loader.save_parquet(df, "data/processed/result.parquet")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving Parquet to {filepath}...")
        df.to_parquet(filepath, **kwargs)
        logger.info(f"Saved {len(df)} rows to {filepath}")


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    # Создаём загрузчик
    loader = DataLoader()
    
    print("=" * 80)
    print("DataLoader - Example Usage")
    print("=" * 80)
    
    # Пример 1: Загрузка CSV (если файл существует)
    csv_path = config.get_data_path("incidents_sample.csv", subdir="synthetic")
    
    if csv_path.exists():
        print(f"\n📁 Loading CSV from: {csv_path}")
        df = loader.load_csv(csv_path)
        
        # Получаем информацию о данных
        info = loader.get_data_info(df)
        
        print(f"\n📊 Dataset Info:")
        print(f"  Rows: {info['n_rows']}")
        print(f"  Columns: {info['n_columns']}")
        print(f"  Memory: {info['memory_usage_mb']} MB")
        print(f"\n  Column names: {', '.join(info['columns'][:5])}...")
        
        # Показываем первые строки
        print(f"\n📋 First 3 rows:")
        print(df.head(3).to_string())
    
    else:
        print(f"\n⚠️  File not found: {csv_path}")
        print("Run 'python -m src.data.generator' first to generate sample data")
    
    print("\n" + "=" * 80)