"""
src/nlp/pii_detector.py

ЧТО: Multi-layer детектор персональных данных (PII)
ЗАЧЕМ: Автоматическое обнаружение утечек конфиденциальной информации в DLP-системе

АРХИТЕКТУРА:
- Layer 1: REGEX - быстрый поиск структурированных данных (карты, паспорта)
- Layer 2: spaCy NER - поиск имён, организаций, локаций через ML
- Layer 3: (будущее) BERT - контекстный анализ для сложных случаев

ИСПОЛЬЗОВАНИЕ:
    from src.nlp import PIIDetector
    
    detector = PIIDetector()
    result = detector.detect("Карта: 1234 5678 9012 3456, ФИО: Иванов Иван")
    
    print(result["has_pii"])  # True
    print(result["cards"])     # ['1234 5678 9012 3456']
    print(result["persons"])   # ['Иванов Иван']
"""

from typing import Dict, List, Optional
import warnings

from src.config import get_config
from src.nlp import patterns
from src.utils import get_logger, timer

# Инициализация
logger = get_logger(__name__)
config = get_config()

# Попытка загрузить spaCy (может отсутствовать)
try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("spaCy is available")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. NER layer will be disabled. Install: pip install spacy")


class PIIDetector:
    """
    Multi-layer детектор персональных данных.
    
    Использует гибридный подход:
    1. Regex для быстрого поиска структурированных данных
    2. spaCy NER для поиска имён и других entity
    
    Attributes:
        use_ner: Использовать ли spaCy NER (если установлен)
        nlp: spaCy модель (если загружена)
    """
    
    def __init__(self, use_ner: bool = True):
        """
        Инициализация детектора PII.
        
        Args:
            use_ner: Использовать ли spaCy NER для поиска имён (default: True)
        """
        logger.info("Initializing PIIDetector...")
        
        self.use_ner = use_ner and SPACY_AVAILABLE
        self.nlp = None
        
        # Загружаем spaCy модель, если нужно
        if self.use_ner:
            try:
                # Пробуем загрузить модель из конфига
                model_name = config.SPACY_MODEL
                logger.info(f"Loading spaCy model: {model_name}")
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            
            except OSError:
                # Модель не установлена
                logger.warning(
                    f"spaCy model '{model_name}' not found. "
                    f"Install it: python -m spacy download {model_name}"
                )
                logger.warning("NER layer disabled. Only regex will be used.")
                self.use_ner = False
            
            except Exception as e:
                logger.error(f"Failed to load spaCy model: {e}")
                self.use_ner = False
        
        logger.info(
            f"PIIDetector initialized. "
            f"Regex: ✓, NER: {'✓' if self.use_ner else '✗'}"
        )
    
    # =========================================================================
    # LAYER 1: REGEX DETECTION (быстро, для структурированных данных)
    # =========================================================================
    
    def _detect_cards(self, text: str) -> List[str]:
        """Находит номера банковских карт через regex."""
        return patterns.CARD_PATTERN.findall(text)
    
    def _detect_passports(self, text: str) -> List[str]:
        """Находит номера паспортов РФ через regex."""
        return patterns.PASSPORT_PATTERN.findall(text)
    
    def _detect_inn(self, text: str) -> List[str]:
        """Находит ИНН через regex."""
        return patterns.INN_PATTERN.findall(text)
    
    def _detect_snils(self, text: str) -> List[str]:
        """Находит СНИЛС через regex."""
        return patterns.SNILS_PATTERN.findall(text)
    
    def _detect_phones(self, text: str) -> List[str]:
        """Находит телефоны через regex."""
        matches = patterns.PHONE_PATTERN.findall(text)
        # PHONE_PATTERN возвращает tuples вида ('+7', '...')
        # Берём первый элемент или всю строку
        if matches and isinstance(matches[0], tuple):
            # Восстанавливаем полный номер
            return [match[0] if isinstance(match, tuple) else match for match in matches]
        return matches
    
    def _detect_emails(self, text: str) -> List[str]:
        """Находит email адреса через regex."""
        return patterns.EMAIL_PATTERN.findall(text)
    
    # =========================================================================
    # LAYER 2: SPACY NER (для имён, организаций, локаций)
    # =========================================================================
    
    def _detect_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Находит named entities через spaCy NER.
        
        ЗАЧЕМ: Regex не может найти ФИО (нет строгого формата).
        spaCy ML модель понимает, что "Иванов Иван" - это имя человека.
        
        Args:
            text: Текст для анализа
        
        Returns:
            Dict с найденными entities:
                - persons: список ФИО
                - orgs: список организаций
                - locations: список локаций
        """
        if not self.use_ner or self.nlp is None:
            return {"persons": [], "orgs": [], "locations": []}
        
        try:
            doc = self.nlp(text)
            
            # Извлекаем entities по типам
            # PER = Person (ФИО)
            # ORG = Organization (компании)
            # LOC = Location (места)
            entities = {
                "persons": [ent.text for ent in doc.ents if ent.label_ == "PER"],
                "orgs": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
                "locations": [ent.text for ent in doc.ents if ent.label_ == "LOC"],
            }
            
            return entities
        
        except Exception as e:
            logger.error(f"NER detection failed: {e}")
            return {"persons": [], "orgs": [], "locations": []}
    
    # =========================================================================
    # MAIN DETECTION METHOD
    # =========================================================================
    
    def detect(self, text: str) -> Dict:
        """
        Обнаруживает все типы PII в тексте.
        
        Использует multi-layer подход:
        1. Regex для структурированных данных (быстро)
        2. spaCy NER для имён (точно)
        
        Args:
            text: Текст для анализа
        
        Returns:
            Dict с результатами:
                - cards: список номеров карт
                - passports: список паспортов
                - inn: список ИНН
                - snils: список СНИЛС
                - phones: список телефонов
                - emails: список email
                - persons: список ФИО (через NER)
                - orgs: список организаций (через NER)
                - locations: список локаций (через NER)
                - has_pii: bool - есть ли PII
                - pii_count: количество найденных PII
                - risk_level: уровень риска (Low/Medium/High/Critical)
        
        Example:
            detector = PIIDetector()
            result = detector.detect("Карта: 1234 5678 9012 3456")
            print(result["has_pii"])  # True
            print(result["cards"])     # ['1234 5678 9012 3456']
        """
        if not text:
            return self._empty_result()
        
        # LAYER 1: Regex detection
        cards = self._detect_cards(text)
        passports = self._detect_passports(text)
        inn = self._detect_inn(text)
        snils = self._detect_snils(text)
        phones = self._detect_phones(text)
        emails = self._detect_emails(text)
        
        # LAYER 2: NER detection
        entities = self._detect_entities(text)
        
        # Собираем результат
        result = {
            # Regex results
            "cards": cards,
            "passports": passports,
            "inn": inn,
            "snils": snils,
            "phones": phones,
            "emails": emails,
            
            # NER results
            "persons": entities["persons"],
            "orgs": entities["orgs"],
            "locations": entities["locations"],
        }
        
        # Подсчитываем общее количество PII
        pii_count = sum([
            len(cards),
            len(passports),
            len(inn),
            len(snils),
            len(phones),
            len(emails),
            len(entities["persons"]),
        ])
        
        # Определяем уровень риска
        risk_level = self._calculate_risk_level(result)
        
        # Добавляем метаданные
        result.update({
            "has_pii": pii_count > 0,
            "pii_count": pii_count,
            "risk_level": risk_level,
            "detection_method": "hybrid (regex + ner)" if self.use_ner else "regex only",
        })
        
        return result
    
    def _calculate_risk_level(self, result: Dict) -> str:
        """
        Вычисляет уровень риска на основе найденных PII.
        
        Логика:
        - Critical: паспорта + карты (самые критичные данные)
        - High: паспорта или карты + другие PII
        - Medium: несколько типов PII без паспортов/карт
        - Low: один тип PII
        
        Args:
            result: Dict с результатами детекции
        
        Returns:
            str: "Critical", "High", "Medium", "Low"
        """
        has_cards = len(result.get("cards", [])) > 0
        has_passports = len(result.get("passports", [])) > 0
        has_inn = len(result.get("inn", [])) > 0
        has_snils = len(result.get("snils", [])) > 0
        has_persons = len(result.get("persons", [])) > 0
        
        # Подсчитываем количество типов PII
        pii_types_count = sum([
            has_cards, has_passports, has_inn, 
            has_snils, has_persons
        ])
        
        # Определяем риск
        if has_passports and has_cards:
            return "Critical"  # паспорт + карта = очень опасно
        
        elif has_passports or has_cards:
            return "High"  # паспорт или карта
        
        elif pii_types_count >= 2:
            return "Medium"  # несколько типов PII
        
        elif pii_types_count == 1:
            return "Low"  # только один тип
        
        else:
            return "Low"  # на всякий случай
    
    def _empty_result(self) -> Dict:
        """Возвращает пустой результат."""
        return {
            "cards": [],
            "passports": [],
            "inn": [],
            "snils": [],
            "phones": [],
            "emails": [],
            "persons": [],
            "orgs": [],
            "locations": [],
            "has_pii": False,
            "pii_count": 0,
            "risk_level": "Low",
            "detection_method": "hybrid (regex + ner)" if self.use_ner else "regex only",
        }
    
    @timer
    def detect_batch(self, texts: List[str]) -> List[Dict]:
        """
        Обнаруживает PII в нескольких текстах (batch processing).
        
        ЗАЧЕМ: Обработка большого датасета (например, 10000 инцидентов).
        
        Args:
            texts: Список текстов для анализа
        
        Returns:
            List[Dict]: Результаты для каждого текста
        
        Example:
            texts = ["Карта: 1234...", "Паспорт: 4567..."]
            results = detector.detect_batch(texts)
        """
        logger.info(f"Processing batch of {len(texts)} texts...")
        
        results = []
        for i, text in enumerate(texts):
            result = self.detect(text)
            results.append(result)
            
            # Логируем прогресс каждые 1000 текстов
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i+1}/{len(texts)} texts...")
        
        logger.info(f"Batch processing complete!")
        return results


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    # Создаём детектор
    detector = PIIDetector(use_ner=True)
    
    print("\n" + "=" * 80)
    print("PII DETECTOR - DEMO")
    print("=" * 80)
    
    # Тестовый текст с разными типами PII
    test_text = """
    Добрый день! Высылаю данные клиента Иванова Ивана Ивановича.
    
    Паспорт: 4567 123456
    Карта: 1234 5678 9012 3456
    ИНН: 123456789012
    СНИЛС: 123-456-789 12
    Телефон: +79991234567
    Email: ivan.ivanov@example.com
    
    Компания: ООО "Пример"
    Адрес: Москва, ул. Ленина, д. 1
    """
    
    print("\n📄 Test text:")
    print(test_text)
    
    # Обнаруживаем PII
    print("\n🔍 Detecting PII...")
    result = detector.detect(test_text)
    
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    
    print(f"\n✅ Has PII: {result['has_pii']}")
    print(f"📊 PII Count: {result['pii_count']}")
    print(f"⚠️  Risk Level: {result['risk_level']}")
    print(f"🔧 Detection Method: {result['detection_method']}")
    
    print("\n" + "-" * 80)
    print("FOUND PII:")
    print("-" * 80)
    
    if result["cards"]:
        print(f"\n💳 Cards ({len(result['cards'])}):")
        for card in result["cards"]:
            print(f"   - {card}")
    
    if result["passports"]:
        print(f"\n📕 Passports ({len(result['passports'])}):")
        for passport in result["passports"]:
            print(f"   - {passport}")
    
    if result["inn"]:
        print(f"\n🔢 INN ({len(result['inn'])}):")
        for inn in result["inn"]:
            print(f"   - {inn}")
    
    if result["snils"]:
        print(f"\n📋 SNILS ({len(result['snils'])}):")
        for snils in result["snils"]:
            print(f"   - {snils}")
    
    if result["phones"]:
        print(f"\n📱 Phones ({len(result['phones'])}):")
        for phone in result["phones"]:
            print(f"   - {phone}")
    
    if result["emails"]:
        print(f"\n📧 Emails ({len(result['emails'])}):")
        for email in result["emails"]:
            print(f"   - {email}")
    
    if result["persons"]:
        print(f"\n👤 Persons (NER) ({len(result['persons'])}):")
        for person in result["persons"]:
            print(f"   - {person}")
    
    if result["orgs"]:
        print(f"\n🏢 Organizations (NER) ({len(result['orgs'])}):")
        for org in result["orgs"]:
            print(f"   - {org}")
    
    if result["locations"]:
        print(f"\n📍 Locations (NER) ({len(result['locations'])}):")
        for loc in result["locations"]:
            print(f"   - {loc}")
    
    print("\n" + "=" * 80)