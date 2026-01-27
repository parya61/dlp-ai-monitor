"""
src/data/generator.py

ЧТО: Генератор синтетических DLP-инцидентов
ЗАЧЕМ: Создать датасет для обучения ML модели (реальные данные использовать нельзя)

ГЕНЕРИРУЕТ:
- Инциденты разных типов (email, USB, облако, принтер)
- Тексты с российскими PII (ФИО, номера карт, паспорта, ИНН, СНИЛС)
- Метаданные (timestamp, user, severity, department)

ИСПОЛЬЗОВАНИЕ:
    from src.data.generator import DLPIncidentGenerator
    
    generator = DLPIncidentGenerator()
    df = generator.generate(n_incidents=1000)
    df.to_csv("data/synthetic/incidents.csv", index=False)
"""

import random
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
from faker import Faker

from src.config import get_config
from src.utils import get_logger, timer

# Инициализация
logger = get_logger(__name__)
config = get_config()
fake = Faker("ru_RU")  # Русская локализация для генерации российских данных


class DLPIncidentGenerator:
    """
    Генератор синтетических DLP-инцидентов.
    
    Создаёт реалистичные инциденты утечек данных для обучения ML модели.
    """
    
    def __init__(self, seed: int = 42):
        """
        Инициализация генератора.
        
        Args:
            seed: Random seed для воспроизводимости результатов
        """
        random.seed(seed)
        Faker.seed(seed)
        
        logger.info(f"Initialized DLPIncidentGenerator with seed={seed}")
        
        # Типы инцидентов с их весами (вероятностями)
        # Email инциденты встречаются чаще всего
        self.incident_types = {
            "email": 0.50,      # 50% всех инцидентов
            "usb": 0.20,        # 20%
            "cloud": 0.20,      # 20%
            "printer": 0.10,    # 10%
        }
        
        # Уровни критичности
        self.severity_levels = ["Low", "Medium", "High", "Critical"]
        
        # Отделы компании
        self.departments = [
            "IT", "HR", "Finance", "Legal", "Sales", 
            "Marketing", "Operations", "R&D", "Support"
        ]
        
        # Доменные зоны для email
        self.email_domains = [
            "company.ru", "corp.ru", "example.ru", 
            "business.com", "enterprise.ru"
        ]
    
    def _generate_russian_name(self) -> str:
        """Генерирует российское ФИО."""
        return fake.name()
    
    def _generate_russian_phone(self) -> str:
        """Генерирует российский номер телефона."""
        return fake.phone_number()
    
    def _generate_card_number(self) -> str:
        """
        Генерирует номер банковской карты.
        Формат: 4 группы по 4 цифры (XXXX XXXX XXXX XXXX)
        """
        return " ".join([str(random.randint(1000, 9999)) for _ in range(4)])
    
    def _generate_passport(self) -> str:
        """
        Генерирует номер паспорта РФ.
        Формат: XXXX XXXXXX (серия 4 цифры, номер 6 цифр)
        """
        series = random.randint(1000, 9999)
        number = random.randint(100000, 999999)
        return f"{series} {number}"
    
    def _generate_inn(self) -> str:
        """
        Генерирует ИНН (Идентификационный Номер Налогоплательщика).
        Физическое лицо: 12 цифр
        """
        return "".join([str(random.randint(0, 9)) for _ in range(12)])
    
    def _generate_snils(self) -> str:
        """
        Генерирует СНИЛС (Страховой Номер Индивидуального Лицевого Счёта).
        Формат: XXX-XXX-XXX YY
        """
        parts = [str(random.randint(100, 999)) for _ in range(3)]
        checksum = random.randint(10, 99)
        return f"{parts[0]}-{parts[1]}-{parts[2]} {checksum}"
    
    def _generate_pii_data(self) -> Dict[str, str]:
        """
        Генерирует набор персональных данных.
        
        Returns:
            Dict с ключами: name, phone, card, passport, inn, snils
        """
        return {
            "name": self._generate_russian_name(),
            "phone": self._generate_russian_phone(),
            "card": self._generate_card_number(),
            "passport": self._generate_passport(),
            "inn": self._generate_inn(),
            "snils": self._generate_snils(),
        }
    
    def _generate_email_incident(self) -> Dict:
        """
        Генерирует инцидент утечки через email.
        
        СЦЕНАРИИ:
        - Отправка конфиденциальных данных на личный email
        - Отправка данных клиентов внешним получателям
        - Пересылка паспортов, договоров с PII
        
        Returns:
            Dict с полями инцидента
        """
        pii = self._generate_pii_data()
        
        # Генерируем отправителя (сотрудника)
        sender_name = self._generate_russian_name()
        sender_email = f"{fake.user_name()}@{random.choice(self.email_domains)}"
        
        # Генерируем получателя
        # С вероятностью 70% получатель внешний (gmail, yandex и т.д.)
        is_external = random.random() < 0.7
        if is_external:
            recipient_email = f"{fake.user_name()}@{random.choice(['gmail.com', 'yandex.ru', 'mail.ru', 'outlook.com'])}"
        else:
            recipient_email = f"{fake.user_name()}@{random.choice(self.email_domains)}"
        
        # Генерируем текст письма с PII
        email_templates = [
            f"Добрый день! Высылаю данные клиента: {pii['name']}, паспорт {pii['passport']}, ИНН {pii['inn']}.",
            f"Информация по договору: ФИО {pii['name']}, телефон {pii['phone']}, карта {pii['card']}.",
            f"Данные для оформления: {pii['name']}, СНИЛС {pii['snils']}, паспорт {pii['passport']}.",
            f"Документы клиента: паспорт {pii['passport']}, ИНН {pii['inn']}, телефон {pii['phone']}.",
        ]
        
        email_body = random.choice(email_templates)
        
        # Определяем критичность
        # Если внешний получатель + много PII = высокая критичность
        has_passport = "паспорт" in email_body.lower()
        has_card = "карта" in email_body.lower()
        
        if is_external and (has_passport or has_card):
            severity = random.choice(["High", "Critical"])
        elif is_external:
            severity = random.choice(["Medium", "High"])
        else:
            severity = random.choice(["Low", "Medium"])
        
        return {
            "incident_type": "email",
            "description": email_body,
            "sender": sender_email,
            "recipient": recipient_email,
            "is_external_recipient": is_external,
            "contains_pii": True,
            "pii_types": "passport,inn,phone,card",  # какие типы PII найдены
            "severity": severity,
            "department": random.choice(self.departments),
            "user": sender_name,
        }
    
    def _generate_usb_incident(self) -> Dict:
        """
        Генерирует инцидент утечки через USB.
        
        СЦЕНАРИИ:
        - Копирование конфиденциальных файлов на USB
        - Попытка вынести базу данных клиентов
        - Копирование документов с паспортными данными
        
        Returns:
            Dict с полями инцидента
        """
        pii = self._generate_pii_data()
        
        user_name = self._generate_russian_name()
        
        # Типы файлов
        file_types = ["xlsx", "docx", "pdf", "csv", "txt"]
        file_type = random.choice(file_types)
        
        # Имена файлов
        file_names = [
            f"Клиенты_база_2024.{file_type}",
            f"Договоры_конфиденциальные.{file_type}",
            f"Персональные_данные.{file_type}",
            f"Паспорта_сотрудников.{file_type}",
            f"Реестр_физлиц.{file_type}",
        ]
        
        file_name = random.choice(file_names)
        
        usb_templates = [
            f"Попытка копирования файла '{file_name}' на USB-накопитель. Файл содержит данные: {pii['name']}, паспорт {pii['passport']}.",
            f"Обнаружено копирование на USB: '{file_name}'. Обнаружены ПД: ФИО {pii['name']}, ИНН {pii['inn']}, СНИЛС {pii['snils']}.",
            f"Попытка вывоза информации: файл '{file_name}' с данными {pii['name']}, телефон {pii['phone']}, карта {pii['card']}.",
        ]
        
        description = random.choice(usb_templates)
        
        # USB инциденты обычно критичные (попытка вынести данные физически)
        severity = random.choice(["High", "Critical"])
        
        return {
            "incident_type": "usb",
            "description": description,
            "file_name": file_name,
            "file_type": file_type,
            "contains_pii": True,
            "pii_types": "passport,inn,snils,phone,card",
            "severity": severity,
            "department": random.choice(self.departments),
            "user": user_name,
        }
    
    def _generate_cloud_incident(self) -> Dict:
        """
        Генерирует инцидент утечки через облачные сервисы.
        
        СЦЕНАРИИ:
        - Загрузка документов в Google Drive
        - Загрузка файлов в Dropbox
        - Синхронизация конфиденциальных данных с OneDrive
        
        Returns:
            Dict с полями инцидента
        """
        pii = self._generate_pii_data()
        
        user_name = self._generate_russian_name()
        
        # Облачные сервисы
        cloud_services = ["Google Drive", "Dropbox", "OneDrive", "Yandex.Disk"]
        cloud_service = random.choice(cloud_services)
        
        # Типы файлов
        file_types = ["xlsx", "docx", "pdf"]
        file_type = random.choice(file_types)
        
        file_names = [
            f"Финансовые_данные_Q4.{file_type}",
            f"Список_клиентов.{file_type}",
            f"Конфиденциальный_отчет.{file_type}",
            f"Персональные_данные_сотрудников.{file_type}",
        ]
        
        file_name = random.choice(file_names)
        
        cloud_templates = [
            f"Загрузка файла '{file_name}' в {cloud_service}. Файл содержит: {pii['name']}, паспорт {pii['passport']}, ИНН {pii['inn']}.",
            f"Обнаружена синхронизация с {cloud_service}: '{file_name}'. Данные: {pii['name']}, телефон {pii['phone']}, СНИЛС {pii['snils']}.",
            f"Попытка загрузки в {cloud_service}: '{file_name}' с информацией {pii['name']}, карта {pii['card']}.",
        ]
        
        description = random.choice(cloud_templates)
        
        # Облачные инциденты средней/высокой критичности
        severity = random.choice(["Medium", "High", "High"])
        
        return {
            "incident_type": "cloud",
            "description": description,
            "cloud_service": cloud_service,
            "file_name": file_name,
            "file_type": file_type,
            "contains_pii": True,
            "pii_types": "passport,inn,snils,phone,card",
            "severity": severity,
            "department": random.choice(self.departments),
            "user": user_name,
        }
    
    def _generate_printer_incident(self) -> Dict:
        """
        Генерирует инцидент утечки через принтер.
        
        СЦЕНАРИИ:
        - Печать конфиденциальных документов
        - Печать паспортных данных
        - Печать договоров с PII
        
        Returns:
            Dict с полями инцидента
        """
        pii = self._generate_pii_data()
        
        user_name = self._generate_russian_name()
        
        # Типы документов
        document_types = [
            "Договор",
            "Анкета",
            "Заявление",
            "Согласие на обработку ПД",
            "Личное дело",
        ]
        
        document_type = random.choice(document_types)
        
        printer_templates = [
            f"Печать документа '{document_type}' с данными: {pii['name']}, паспорт {pii['passport']}, ИНН {pii['inn']}.",
            f"Обнаружена печать: '{document_type}'. Содержит ПД: {pii['name']}, телефон {pii['phone']}, СНИЛС {pii['snils']}.",
            f"Попытка печати '{document_type}' с информацией: {pii['name']}, карта {pii['card']}, паспорт {pii['passport']}.",
        ]
        
        description = random.choice(printer_templates)
        
        # Принтер инциденты обычно низкой/средней критичности
        severity = random.choice(["Low", "Medium", "Medium"])
        
        return {
            "incident_type": "printer",
            "description": description,
            "document_type": document_type,
            "contains_pii": True,
            "pii_types": "passport,inn,snils,phone,card",
            "severity": severity,
            "department": random.choice(self.departments),
            "user": user_name,
        }
    
    @timer
    def generate(self, n_incidents: int = 1000) -> pd.DataFrame:
        """
        Генерирует датасет DLP-инцидентов.
        
        Args:
            n_incidents: Количество инцидентов для генерации
        
        Returns:
            pd.DataFrame с инцидентами
        
        Example:
            generator = DLPIncidentGenerator()
            df = generator.generate(n_incidents=5000)
            df.to_csv("data/synthetic/incidents.csv", index=False)
        """
        logger.info(f"Starting generation of {n_incidents} incidents...")
        
        incidents = []
        
        # Генерируем инциденты согласно заданным пропорциям
        for i in range(n_incidents):
            # Выбираем тип инцидента с учётом весов
            incident_type = random.choices(
                list(self.incident_types.keys()),
                weights=list(self.incident_types.values()),
                k=1
            )[0]
            
            # Генерируем инцидент соответствующего типа
            if incident_type == "email":
                incident = self._generate_email_incident()
            elif incident_type == "usb":
                incident = self._generate_usb_incident()
            elif incident_type == "cloud":
                incident = self._generate_cloud_incident()
            elif incident_type == "printer":
                incident = self._generate_printer_incident()
            
            # Добавляем общие поля
            # Временная метка в последние 90 дней
            days_ago = random.randint(0, 90)
            timestamp = datetime.now() - timedelta(days=days_ago)
            
            incident.update({
                "incident_id": f"INC-{i+1:06d}",
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "detected_by": "DLP System",
            })
            
            incidents.append(incident)
            
            # Логируем прогресс каждые 1000 инцидентов
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i+1}/{n_incidents} incidents...")
        
        # Создаём DataFrame
        df = pd.DataFrame(incidents)
        
        logger.info(f"Generation complete! Created DataFrame with shape {df.shape}")
        logger.info(f"Incident types distribution:\n{df['incident_type'].value_counts()}")
        logger.info(f"Severity distribution:\n{df['severity'].value_counts()}")
        
        return df


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    # Создаём генератор
    generator = DLPIncidentGenerator(seed=42)
    
    # Генерируем 100 инцидентов для примера
    df = generator.generate(n_incidents=100)
    
    # Сохраняем в CSV
    output_path = config.get_data_path("incidents_sample.csv", subdir="synthetic")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    logger.info(f"Saved sample incidents to {output_path}")
    
    # Показываем примеры
    print("\n" + "=" * 80)
    print("SAMPLE INCIDENTS")
    print("=" * 80)
    print(df[["incident_id", "incident_type", "severity", "user", "description"]].head(10).to_string())
    print("=" * 80)