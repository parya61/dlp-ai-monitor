"""
src/data/augmentation.py

ЧТО: Улучшенная генерация DLP-инцидентов с реалистичными паттернами
ЗАЧЕМ: Создать production-like датасет для обучения ML модели

ВОЗМОЖНОСТИ:
- Реалистичные email шаблоны
- Временные паттерны (рабочие часы, сезонность)
- Вариативность текстов
- Опечатки и естественный шум
- Разные форматы PII

ИСПОЛЬЗОВАНИЕ:
    from src.data.augmentation import EnhancedDLPGenerator
    
    generator = EnhancedDLPGenerator()
    df = generator.generate(n_incidents=30000)
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from faker import Faker

from src.config import get_config
from src.data.generator import DLPIncidentGenerator
from src.utils import get_logger

# Инициализация
logger = get_logger(__name__)
config = get_config()


class EnhancedDLPGenerator(DLPIncidentGenerator):
    """
    Улучшенный генератор DLP-инцидентов.
    
    Добавляет реализм через:
    - Разнообразные email шаблоны
    - Временные паттерны (больше инцидентов в конце недели)
    - Опечатки и вариации
    - Реалистичные имена файлов и путей
    """
    
    def __init__(self, seed: int = 42):
        """Инициализация улучшенного генератора."""
        super().__init__(seed=seed)
        
        # Расширенные шаблоны email'ов
        self.email_templates_extended = [
            # Деловые
            "Добрый день, {name}! Направляю запрошенные документы в приложении.",
            "Здравствуйте! Высылаю информацию по клиенту {client}.",
            "Коллеги, во вложении отчёт за {month} месяц.",
            "Отправляю данные для проверки. Срочно нужен ответ.",
            
            # Неформальные (риск утечки выше)
            "Привет! Скинул тебе базу клиентов на личную почту.",
            "Смотри, что нашёл в папке с зарплатами 😄",
            "Держи файлик, который просил. Никому не говори!",
            
            # Формальные
            "Уважаемые коллеги, направляем сводку по персональным данным сотрудников.",
            "В соответствии с вашим запросом направляю выгрузку из CRM.",
            "Информирую о выявленных несоответствиях в документах {doc_type}.",
            
            # Короткие (реальный стиль)
            "Документы во вложении.",
            "Смотри файл.",
            "Отправил как просил.",
            "База в аттаче.",
        ]
        
        # Реалистичные имена файлов
        self.realistic_filenames = [
            "Клиенты_{date}.xlsx",
            "База_CRM_экспорт.csv",
            "Зарплаты_{month}_{year}.xlsx",
            "Договоры_архив.zip",
            "Паспортные_данные.docx",
            "Персональные_данные_сотрудников.xlsx",
            "Confidential_{random}.pdf",
            "НЕ_УДАЛЯТЬ_ВАЖНО.xlsx",
            "Финансовый_отчёт_Q{quarter}.xlsx",
            "backup_{timestamp}.sql",
        ]
        
        logger.info("EnhancedDLPGenerator initialized with realistic patterns")
    
    def _generate_realistic_timestamp(self) -> datetime:
        """
        Генерирует реалистичные временные метки.
        
        ПАТТЕРНЫ:
        - Больше инцидентов в конце недели (пятница)
        - Меньше инцидентов ночью
        - Пики в 10:00-12:00 и 14:00-16:00
        
        Returns:
            datetime: Реалистичная временная метка
        """
        # Дата за последний год
        days_ago = random.randint(1, 365)
        base_date = datetime.now() - timedelta(days=days_ago)
        
        # Вероятность инцидента по дням недели
        # 0 = понедельник, 6 = воскресенье
        weekday = base_date.weekday()
        
        # Больше инцидентов в пятницу (люди спешат, халатны)
        if weekday == 4:  # Пятница
            if random.random() < 0.3:  # 30% шанс пересоздать
                days_ago = random.randint(1, 365)
                base_date = datetime.now() - timedelta(days=days_ago)
        
        # Меньше в выходные
        if weekday >= 5:  # Суббота/воскресенье
            if random.random() < 0.7:  # 70% шанс пересоздать
                days_ago = random.randint(1, 365)
                base_date = datetime.now() - timedelta(days=days_ago)
        
        # Рабочие часы (8:00 - 19:00)
        # Пики: 10:00-12:00 и 14:00-16:00
        hour_weights = {
            8: 5, 9: 10, 10: 20, 11: 25, 12: 15,
            13: 8, 14: 20, 15: 22, 16: 18, 17: 12,
            18: 7, 19: 3
        }
        
        hour = random.choices(
            list(hour_weights.keys()),
            weights=list(hour_weights.values())
        )[0]
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return base_date.replace(hour=hour, minute=minute, second=second)
    
    def _generate_realistic_filename(self) -> str:
        """
        Генерирует реалистичное имя файла.
        
        Returns:
            str: Имя файла с реалистичными паттернами
        """
        template = random.choice(self.realistic_filenames)
        
        # Заполняем плейсхолдеры
        filename = template.format(
            date=datetime.now().strftime("%d.%m.%Y"),
            month=random.choice(["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
                                "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]),
            year=random.randint(2023, 2026),
            quarter=random.randint(1, 4),
            random=random.randint(1000, 9999),
            timestamp=int(datetime.now().timestamp())
        )
        
        return filename
    
    def _add_typos(self, text: str, typo_rate: float = 0.02) -> str:
        """
        Добавляет опечатки в текст для реализма.
        
        Args:
            text: Исходный текст
            typo_rate: Вероятность опечатки на символ
        
        Returns:
            str: Текст с опечатками
        """
        # Частые опечатки в русском языке
        typo_map = {
            'а': 'о', 'о': 'а', 'е': 'и', 'и': 'е',
            'т': 'т', 'п': 'р', 'р': 'п', 'л': 'д',
        }
        
        result = []
        for char in text:
            if char.lower() in typo_map and random.random() < typo_rate:
                # Опечатка
                result.append(typo_map[char.lower()])
            else:
                result.append(char)
        
        return ''.join(result)
    
    def _generate_email_incident(self) -> Dict:
        """
        Генерирует реалистичный email инцидент.
        
        Использует родительский метод + улучшения.
        
        Returns:
            Dict: Инцидент с улучшенным описанием
        """
        # Используем родительский метод
        incident = super()._generate_email_incident()
        
        # Улучшаем описание
        template = random.choice(self.email_templates_extended)
        
        # Простые плейсхолдеры
        description = template.replace("{name}", "коллега")
        description = description.replace("{client}", "клиента")
        description = description.replace("{month}", random.choice(["январь", "февраль", "март"]))
        description = description.replace("{doc_type}", random.choice(["договоры", "акты"]))
        
        # Добавляем PII иногда
        if random.random() < 0.6:  # 60% содержат PII
            pii_elements = []
            
            if random.random() < 0.4:
                card = f"{random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}"
                pii_elements.append(f"Карта: {card}")
            
            if random.random() < 0.3:
                passport = f"{random.randint(1000, 9999)} {random.randint(100000, 999999)}"
                pii_elements.append(f"Паспорт: {passport}")
            
            if pii_elements:
                description += " " + ", ".join(pii_elements) + "."
        
        # Добавляем опечатки иногда
        if random.random() < 0.15:  # 15% с опечатками
            description = self._add_typos(description, typo_rate=0.01)
        
        # Обновляем инцидент
        incident['description'] = description
        incident['timestamp'] = self._generate_realistic_timestamp()
        
        return incident
    
    def _generate_usb_incident(self) -> Dict:
        """Генерирует реалистичный USB инцидент."""
        incident = super()._generate_usb_incident()
        
        # Реалистичное имя файла
        filename = self._generate_realistic_filename()
        
        actions = [
            f"Копирование файла '{filename}' на USB-накопитель",
            f"Попытка записи '{filename}' на внешний носитель",
            f"Обнаружена передача файла '{filename}' через USB",
        ]
        
        incident['description'] = random.choice(actions)
        incident['timestamp'] = self._generate_realistic_timestamp()
        
        return incident
    
    def _generate_cloud_incident(self) -> Dict:
        """Генерирует реалистичный cloud инцидент."""
        incident = super()._generate_cloud_incident()
        
        filename = self._generate_realistic_filename()
        services = ["Google Drive", "Яндекс.Диск", "OneDrive", "Dropbox"]
        
        actions = [
            f"Загрузка '{filename}' в {random.choice(services)}",
            f"Синхронизация '{filename}' с {random.choice(services)}",
        ]
        
        incident['description'] = random.choice(actions)
        incident['timestamp'] = self._generate_realistic_timestamp()
        
        return incident
    
    def generate(self, n_incidents: int = 1000, show_progress: bool = True) -> pd.DataFrame:
        """
        Генерирует улучшенный датасет DLP-инцидентов.
        
        Args:
            n_incidents: Количество инцидентов
            show_progress: Показывать прогресс
        
        Returns:
            pd.DataFrame: Датасет инцидентов
        """
        logger.info(f"Generating {n_incidents} enhanced DLP incidents...")
        
        incidents = []
        
        # Типы и веса из родительского класса
        incident_types = ["email", "usb", "cloud", "printer"]
        incident_weights = [0.4, 0.25, 0.2, 0.15]  # email чаще всего
        
        # Используем tqdm если доступен
        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_incidents), desc="Generating incidents")
        except ImportError:
            iterator = range(n_incidents)
            if show_progress:
                logger.info("Install tqdm for progress bar: pip install tqdm")
        
        for i in iterator:
            # Выбираем тип инцидента
            incident_type = random.choices(
                incident_types,
                weights=incident_weights
            )[0]
            
            # Генерируем инцидент нужного типа
            if incident_type == "email":
                incident = self._generate_email_incident()
            elif incident_type == "usb":
                incident = self._generate_usb_incident()
            elif incident_type == "cloud":
                incident = self._generate_cloud_incident()
            else:  # printer
                incident = self._generate_printer_incident()
            
            incidents.append(incident)
        
        df = pd.DataFrame(incidents)
        
        logger.info(f"Generated {len(df)} incidents")
        logger.info(f"Types distribution: {df['incident_type'].value_counts().to_dict()}")
        logger.info(f"Severity distribution: {df['severity'].value_counts().to_dict()}")
        
        return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_large_dataset(
    n_incidents: int = 30000,
    output_path: str = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Удобная функция для генерации большого датасета.
    
    Args:
        n_incidents: Количество инцидентов (default: 30000)
        output_path: Путь для сохранения CSV (опционально)
        seed: Random seed для воспроизводимости
    
    Returns:
        pd.DataFrame: Сгенерированный датасет
    
    Example:
        df = generate_large_dataset(n_incidents=50000)
        df.to_csv("incidents_50k.csv", index=False)
    """
    logger.info("=" * 80)
    logger.info(f"GENERATING LARGE DATASET: {n_incidents} incidents")
    logger.info("=" * 80)
    
    generator = EnhancedDLPGenerator(seed=seed)
    df = generator.generate(n_incidents=n_incidents)
    
    if output_path:
        from src.data import DataLoader
        loader = DataLoader()
        loader.save_csv(df, output_path)
        logger.info(f"Dataset saved to {output_path}")
    
    return df


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("ENHANCED DLP GENERATOR - DEMO")
    logger.info("=" * 80)
    
    # Генерируем небольшую выборку для демонстрации
    generator = EnhancedDLPGenerator(seed=42)
    df = generator.generate(n_incidents=100)
    
    # Показываем примеры
    logger.info("\nExample incidents:")
    for idx, row in df.head(5).iterrows():
        logger.info(f"\n{idx+1}. {row['incident_type'].upper()} - {row['severity']}")
        logger.info(f"   {row['description'][:100]}...")
        logger.info(f"   Time: {row['timestamp']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Demo complete!")
    logger.info("=" * 80)