"""
src/ml/features.py

ЧТО: Feature Engineering для DLP-инцидентов
ЗАЧЕМ: Превращаем текст и метаданные в числовые признаки для ML модели

АРХИТЕКТУРА:
- FeatureExtractor - главный класс
- extract_pii_features() - признаки из PII (карты, паспорта)
- extract_text_features() - TF-IDF векторизация текста
- extract_metadata_features() - признаки из метаданных
- fit() - обучение на тренировочных данных
- transform() - преобразование новых данных

ИСПОЛЬЗОВАНИЕ:
    from src.ml import FeatureExtractor
    
    extractor = FeatureExtractor()
    extractor.fit(df_train)
    X = extractor.transform(df_train)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.config import get_config
from src.nlp import PIIDetector
from src.utils import get_logger

# Инициализация
logger = get_logger(__name__)
config = get_config()


class FeatureExtractor:
    """
    Feature Engineering для DLP-инцидентов.
    
    Извлекает признаки трёх типов:
    1. PII features - обнаруженные персональные данные
    2. Text features - TF-IDF векторы из текста описания
    3. Metadata features - отдел, пользователь, время
    
    Attributes:
        pii_detector: Детектор PII для анализа текста
        tfidf: TF-IDF векторизатор для текста
        label_encoders: Энкодеры для категориальных признаков
    """
    
    def __init__(
        self,
        max_tfidf_features: int = 100,
        use_pii: bool = True
    ):
        """
        Инициализация Feature Extractor.
        
        Args:
            max_tfidf_features: Максимум TF-IDF признаков (default: 100)
            use_pii: Использовать ли PII детектор (default: True)
        """
        logger.info("Initializing FeatureExtractor...")
        
        self.max_tfidf_features = max_tfidf_features
        self.use_pii = use_pii
        
        # PII детектор (если нужен)
        self.pii_detector = None
        if self.use_pii:
            try:
                self.pii_detector = PIIDetector(use_ner=False)  # только regex (быстрее)
                logger.info("PII detector initialized")
            except Exception as e:
                logger.warning(f"Failed to init PII detector: {e}. PII features disabled.")
                self.use_pii = False
        
        # TF-IDF векторизатор для текста
        # ЗАЧЕМ: Превращает текст в числовой вектор
        # "карта паспорт" -> [0.5, 0.3, 0.0, 0.8, ...]
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,  # топ-100 самых важных слов
            min_df=2,                         # слово должно быть минимум в 2 документах
            ngram_range=(1, 2),               # unigrams + bigrams ("карта", "номер карты")
            lowercase=True,                   # приводим к lowercase
            strip_accents='unicode'           # убираем акценты
        )
        
        # Label Encoders для категориальных признаков
        # ЗАЧЕМ: "Sales" -> 0, "IT" -> 1, "HR" -> 2
        self.label_encoders = {}
        
        # Список имён признаков (заполнится после fit)
        self.feature_names = []
        
        # Флаг обучения
        self.is_fitted = False
        
        logger.info(
            f"FeatureExtractor initialized. "
            f"TF-IDF features: {max_tfidf_features}, PII: {self.use_pii}"
        )
    
    # =========================================================================
    # ИЗВЛЕЧЕНИЕ PII ПРИЗНАКОВ
    # =========================================================================
    
    def _extract_pii_features(self, text: str) -> Dict[str, int]:
        """
        Извлекает признаки из PII (персональных данных).
        
        ЗАЧЕМ: Наличие паспорта/карты влияет на критичность инцидента.
        
        Args:
            text: Текст описания инцидента
        
        Returns:
            Dict с бинарными признаками:
                - has_card: есть ли номер карты (0/1)
                - has_passport: есть ли паспорт (0/1)
                - has_inn: есть ли ИНН (0/1)
                - has_snils: есть ли СНИЛС (0/1)
                - has_phone: есть ли телефон (0/1)
                - has_email: есть ли email (0/1)
                - pii_count: общее количество PII
        """
        if not self.use_pii or not self.pii_detector:
            return {
                "has_card": 0,
                "has_passport": 0,
                "has_inn": 0,
                "has_snils": 0,
                "has_phone": 0,
                "has_email": 0,
                "pii_count": 0,
            }
        
        try:
            # Детектируем PII
            result = self.pii_detector.detect(text)
            
            # Бинарные признаки (есть/нет)
            features = {
                "has_card": int(len(result["cards"]) > 0),
                "has_passport": int(len(result["passports"]) > 0),
                "has_inn": int(len(result["inn"]) > 0),
                "has_snils": int(len(result["snils"]) > 0),
                "has_phone": int(len(result["phones"]) > 0),
                "has_email": int(len(result["emails"]) > 0),
                "pii_count": result["pii_count"],
            }
            
            return features
        
        except Exception as e:
            logger.error(f"PII extraction failed: {e}")
            return {
                "has_card": 0,
                "has_passport": 0,
                "has_inn": 0,
                "has_snils": 0,
                "has_phone": 0,
                "has_email": 0,
                "pii_count": 0,
            }
    
    # =========================================================================
    # ИЗВЛЕЧЕНИЕ ТЕКСТОВЫХ ПРИЗНАКОВ
    # =========================================================================
    
    def _extract_text_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        Извлекает TF-IDF признаки из текста.
        
        TF-IDF = Term Frequency - Inverse Document Frequency
        ЗАЧЕМ: Важные слова получают больший вес.
        
        Пример:
            "карта" встречается часто в инцидентах -> высокий вес
            "и", "в", "на" встречаются везде -> низкий вес
        
        Args:
            texts: Список текстов
            fit: Обучать ли TF-IDF (только на train данных)
        
        Returns:
            np.ndarray: Матрица TF-IDF признаков (n_samples x n_features)
        """
        if fit:
            # Обучаем TF-IDF на тренировочных данных
            logger.info(f"Fitting TF-IDF on {len(texts)} texts...")
            tfidf_matrix = self.tfidf.fit_transform(texts)
            logger.info(f"TF-IDF vocabulary size: {len(self.tfidf.vocabulary_)}")
        else:
            # Применяем уже обученный TF-IDF
            tfidf_matrix = self.tfidf.transform(texts)
        
        return tfidf_matrix.toarray()
    
    # =========================================================================
    # ИЗВЛЕЧЕНИЕ МЕТАДАННЫХ
    # =========================================================================
    
    def _extract_metadata_features(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> np.ndarray:
        """
        Извлекает признаки из метаданных.
        
        ЗАЧЕМ: Отдел и тип инцидента связаны:
            - Finance часто работает с финансовыми данными
            - IT редко использует USB (есть права админа)
        
        ВАЖНО: Не используем incident_type, т.к. это целевая переменная!
        
        Args:
            df: DataFrame с метаданными
            fit: Обучать ли Label Encoders
        
        Returns:
            np.ndarray: Матрица метаданных (n_samples x n_features)
        """
        features = []
        
        # Категориальные признаки для кодирования
        # ВАЖНО: не используем incident_type, т.к. это целевая переменная (data leakage!)
        categorical_features = ["department"]
        
        for col in categorical_features:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue
            
            if fit:
                # Обучаем Label Encoder
                self.label_encoders[col] = LabelEncoder()
                encoded = self.label_encoders[col].fit_transform(df[col].fillna("unknown"))
            else:
                # Применяем уже обученный
                if col in self.label_encoders:
                    # Handle unknown categories
                    le = self.label_encoders[col]
                    values = df[col].fillna("unknown")
                    encoded = np.array([
                        le.transform([v])[0] if v in le.classes_ else -1
                        for v in values
                    ])
                else:
                    encoded = np.zeros(len(df))
            
            features.append(encoded.reshape(-1, 1))
        
        # Объединяем все признаки
        if features:
            return np.hstack(features)
        else:
            return np.zeros((len(df), 0))
    
    # =========================================================================
    # ГЛАВНЫЕ МЕТОДЫ
    # =========================================================================
    
    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        """
        Обучает Feature Extractor на тренировочных данных.
        
        ЗАЧЕМ: Нужно обучить TF-IDF и Label Encoders на train данных,
        чтобы потом применить их к test данным.
        
        Args:
            df: DataFrame с тренировочными данными
        
        Returns:
            self: для chaining (extractor.fit(df).transform(df))
        
        Example:
            extractor = FeatureExtractor()
            extractor.fit(df_train)
        """
        logger.info(f"Fitting FeatureExtractor on {len(df)} samples...")
        
        # Проверяем наличие колонки description
        if "description" not in df.columns:
            raise ValueError("DataFrame must have 'description' column")
        
        # Извлекаем текст
        texts = df["description"].fillna("").tolist()
        
        # Обучаем TF-IDF
        _ = self._extract_text_features(texts, fit=True)
        
        # Обучаем Label Encoders
        _ = self._extract_metadata_features(df, fit=True)
        
        # Формируем список имён признаков
        self._build_feature_names()
        
        self.is_fitted = True
        logger.info(f"FeatureExtractor fitted. Total features: {len(self.feature_names)}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Преобразует данные в признаки.
        
        ВАЖНО: Нужно сначала вызвать fit() на train данных!
        
        Args:
            df: DataFrame для преобразования
        
        Returns:
            np.ndarray: Матрица признаков (n_samples x n_features)
        
        Example:
            X_train = extractor.fit(df_train).transform(df_train)
            X_test = extractor.transform(df_test)
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor not fitted! Call fit() first.")
        
        logger.info(f"Transforming {len(df)} samples...")
        
        all_features = []
        
        # 1. PII признаки
        if self.use_pii:
            logger.info("Extracting PII features...")
            pii_features_list = []
            for text in df["description"].fillna(""):
                pii_feats = self._extract_pii_features(text)
                pii_features_list.append(list(pii_feats.values()))
            
            pii_features = np.array(pii_features_list)
            all_features.append(pii_features)
            logger.info(f"PII features shape: {pii_features.shape}")
        
        # 2. TF-IDF текстовые признаки
        logger.info("Extracting text features...")
        texts = df["description"].fillna("").tolist()
        text_features = self._extract_text_features(texts, fit=False)
        all_features.append(text_features)
        logger.info(f"Text features shape: {text_features.shape}")
        
        # 3. Метаданные
        logger.info("Extracting metadata features...")
        metadata_features = self._extract_metadata_features(df, fit=False)
        if metadata_features.shape[1] > 0:
            all_features.append(metadata_features)
            logger.info(f"Metadata features shape: {metadata_features.shape}")
        
        # Объединяем все признаки
        X = np.hstack(all_features)
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        
        return X
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Обучает и преобразует данные за один вызов.
        
        Эквивалентно: extractor.fit(df).transform(df)
        
        Args:
            df: DataFrame для обучения и преобразования
        
        Returns:
            np.ndarray: Матрица признаков
        """
        return self.fit(df).transform(df)
    
    def _build_feature_names(self):
        """Формирует список имён признаков."""
        names = []
        
        # PII признаки
        if self.use_pii:
            names.extend([
                "has_card", "has_passport", "has_inn",
                "has_snils", "has_phone", "has_email", "pii_count"
            ])
        
        # TF-IDF признаки
        if hasattr(self.tfidf, 'get_feature_names_out'):
            tfidf_names = [f"tfidf_{name}" for name in self.tfidf.get_feature_names_out()]
            names.extend(tfidf_names)
        
        # Метаданные
        for col in self.label_encoders.keys():
            names.append(f"encoded_{col}")
        
        self.feature_names = names
    
    def get_feature_names(self) -> List[str]:
        """
        Возвращает список имён признаков.
        
        Returns:
            List[str]: Имена признаков
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor not fitted! Call fit() first.")
        
        return self.feature_names


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    from src.data import DataLoader
    
    logger.info("=" * 80)
    logger.info("FEATURE EXTRACTOR - DEMO")
    logger.info("=" * 80)
    
    # Загружаем данные
    loader = DataLoader()
    csv_path = config.get_data_path("incidents_sample.csv", subdir="synthetic")
    
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        logger.error("Run 'python -m src.data.generator' first!")
    else:
        df = loader.load_csv(csv_path)
        logger.info(f"Loaded {len(df)} incidents")
        
        # Создаём feature extractor
        extractor = FeatureExtractor(max_tfidf_features=50, use_pii=True)
        
        # Извлекаем признаки
        X = extractor.fit_transform(df)
        
        logger.info("\n" + "=" * 80)
        logger.info("RESULTS:")
        logger.info("=" * 80)
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Number of features: {len(extractor.get_feature_names())}")
        
        # Показываем первые признаки
        feature_names = extractor.get_feature_names()
        logger.info(f"\nFirst 10 features:")
        for i, name in enumerate(feature_names[:10]):
            logger.info(f"  {i+1}. {name}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Feature extraction complete!")
        logger.info("=" * 80)