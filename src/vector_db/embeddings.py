"""
src/vector_db/embeddings.py

ЧТО: Создание эмбеддингов для DLP-инцидентов
ЗАЧЕМ: Преобразовать текст в векторы для поиска похожих случаев

ТЕХНОЛОГИИ:
- Sentence-Transformers (multi-qa-MiniLM-L6-cos-v1)
- 384-dimensional vectors
- Multilingual support (русский + английский)

ИСПОЛЬЗОВАНИЕ:
    from src.vector_db import IncidentEmbedder
    
    embedder = IncidentEmbedder()
    
    # Один текст
    vector = embedder.encode_one("Утечка данных через email")
    
    # Batch
    vectors = embedder.encode(df['description'].tolist())
"""

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_config
from src.utils import get_logger

# Инициализация
logger = get_logger(__name__)
config = get_config()


class IncidentEmbedder:
    """
    Создание эмбеддингов для текстов DLP-инцидентов.
    
    Использует Sentence-Transformers для преобразования текста в векторы.
    Модель: multi-qa-MiniLM-L6-cos-v1
    - 384 dimensions
    - Multilingual (русский + английский)
    - Быстрая (CPU friendly)
    - Хорошо для similarity search
    
    Attributes:
        model: Sentence-Transformer модель
        dimension: Размерность векторов
    """
    
    def __init__(self, model_name: str = None):
        """
        Инициализация embedder'а.
        
        Args:
            model_name: Название модели (default: multi-qa-MiniLM-L6-cos-v1)
        """
        if model_name is None:
            # Используем multilingual модель для русского+английского
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            # Альтернативы:
            # "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" - для английского
            # "sentence-transformers/distiluse-base-multilingual-cased-v1" - больше, точнее
        
        logger.info(f"Loading Sentence-Transformer model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Please install: pip install sentence-transformers")
            raise
    
    def encode_one(self, text: str, show_progress: bool = False) -> np.ndarray:
        """
        Создаёт эмбеддинг для одного текста.
        
        Args:
            text: Текст для кодирования
            show_progress: Показывать прогресс (для длинных текстов)
        
        Returns:
            np.ndarray: Вектор размерности (dimension,)
        
        Example:
            vector = embedder.encode_one("Утечка через email")
            print(vector.shape)  # (384,)
        """
        if not text or not text.strip():
            # Пустой текст → нулевой вектор
            logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self.dimension, dtype=np.float32)
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        
        return embedding.astype(np.float32)
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Создаёт эмбеддинги для списка текстов (batch processing).
        
        Args:
            texts: Список текстов
            batch_size: Размер batch'а для обработки
            show_progress: Показывать прогресс-бар
        
        Returns:
            np.ndarray: Матрица векторов (n_texts, dimension)
        
        Example:
            texts = df['description'].tolist()
            vectors = embedder.encode(texts, batch_size=64)
            print(vectors.shape)  # (10000, 384)
        """
        if not texts:
            logger.warning("Empty texts list provided")
            return np.zeros((0, self.dimension), dtype=np.float32)
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        # Заменяем пустые тексты на placeholder
        texts_clean = [
            text if text and text.strip() else "[EMPTY]"
            for text in texts
        ]
        
        # Кодируем
        embeddings = self.model.encode(
            texts_clean,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"Encoded {len(embeddings)} vectors, shape: {embeddings.shape}")
        
        return embeddings.astype(np.float32)
    
    def get_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Вычисляет cosine similarity между двумя векторами.
        
        Args:
            vec1: Первый вектор
            vec2: Второй вектор
        
        Returns:
            float: Similarity score (0 to 1, где 1 = идентичные)
        
        Example:
            v1 = embedder.encode_one("Утечка через email")
            v2 = embedder.encode_one("Отправка письма с данными")
            similarity = embedder.get_similarity(v1, v2)
            print(f"Similarity: {similarity:.3f}")
        """
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_embeddings_for_dataframe(
    df,
    text_column: str = 'description',
    model_name: str = None
) -> np.ndarray:
    """
    Удобная функция для создания эмбеддингов из DataFrame.
    
    Args:
        df: DataFrame с инцидентами
        text_column: Название колонки с текстом
        model_name: Модель для эмбеддингов (опционально)
    
    Returns:
        np.ndarray: Матрица эмбеддингов
    
    Example:
        vectors = create_embeddings_for_dataframe(df, text_column='description')
    """
    embedder = IncidentEmbedder(model_name=model_name)
    texts = df[text_column].fillna("").tolist()
    vectors = embedder.encode(texts)
    return vectors


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("INCIDENT EMBEDDER - DEMO")
    logger.info("=" * 80)
    
    # Создаём embedder
    embedder = IncidentEmbedder()
    
    # Тестовые тексты
    texts = [
        "Отправка email с данными клиентов на личную почту",
        "Копирование файла с паспортными данными на USB",
        "Загрузка конфиденциального документа в Google Drive",
        "Печать документа с номерами банковских карт",
        "Утечка персональных данных через облачное хранилище",
    ]
    
    logger.info(f"\nEncoding {len(texts)} test texts...")
    
    # Кодируем
    vectors = embedder.encode(texts, show_progress=False)
    
    logger.info(f"Vectors shape: {vectors.shape}")
    logger.info(f"Vector dimension: {embedder.dimension}")
    
    # Проверяем similarity
    logger.info("\nSimilarity between texts:")
    logger.info(f"Text 1: {texts[0][:50]}...")
    logger.info(f"Text 2: {texts[1][:50]}...")
    
    sim = embedder.get_similarity(vectors[0], vectors[1])
    logger.info(f"Similarity: {sim:.3f}")
    
    # Similarity с самим собой (должно быть ~1.0)
    sim_self = embedder.get_similarity(vectors[0], vectors[0])
    logger.info(f"\nSimilarity with itself: {sim_self:.3f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Demo complete!")
    logger.info("=" * 80)