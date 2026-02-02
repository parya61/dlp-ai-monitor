"""
src/vector_db/faiss_store.py

ЧТО: FAISS хранилище для векторов DLP-инцидентов
ЗАЧЕМ: Быстрый поиск похожих инцидентов (similarity search)

ТЕХНОЛОГИИ:
- FAISS (Facebook AI Similarity Search)
- IndexFlatL2 для точного поиска
- Поддержка сохранения/загрузки

ИСПОЛЬЗОВАНИЕ:
    from src.vector_db import FAISSStore
    
    store = FAISSStore(dimension=384)
    
    # Добавляем векторы
    store.add(vectors, metadata)
    
    # Ищем похожие
    similar = store.search(query_vector, k=5)
    
    # Сохраняем
    store.save("data/vector_db/incidents.faiss")
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.config import get_config
from src.utils import get_logger

# Инициализация
logger = get_logger(__name__)
config = get_config()


class FAISSStore:
    """
    FAISS векторное хранилище для поиска похожих инцидентов.
    
    Использует FAISS IndexFlatL2 для точного поиска ближайших соседей.
    Хранит metadata (ID инцидентов, тексты) отдельно.
    
    Attributes:
        dimension: Размерность векторов
        index: FAISS индекс
        metadata: Метаданные инцидентов (ID, тексты и т.д.)
    """
    
    def __init__(self, dimension: int = 384):
        """
        Инициализация FAISS хранилища.
        
        Args:
            dimension: Размерность векторов (default: 384 для MiniLM)
        """
        self.dimension = dimension
        
        # Создаём FAISS индекс
        # IndexFlatL2 - точный поиск через L2 distance
        self.index = faiss.IndexFlatL2(dimension)
        
        # Метаданные (ID, тексты, и т.д.)
        self.metadata: List[Dict] = []
        
        logger.info(f"FAISSStore initialized with dimension={dimension}")
    
    def add(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Добавляет векторы в индекс.
        
        Args:
            vectors: Матрица векторов (n_samples, dimension)
            metadata: Список метаданных для каждого вектора
        
        Example:
            vectors = embedder.encode(texts)
            metadata = [{"id": i, "text": t} for i, t in enumerate(texts)]
            store.add(vectors, metadata)
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )
        
        n_vectors = len(vectors)
        
        # Добавляем векторы в FAISS
        self.index.add(vectors)
        
        # Добавляем метаданные
        if metadata is None:
            # Создаём пустые метаданные
            metadata = [{"id": i} for i in range(n_vectors)]
        
        if len(metadata) != n_vectors:
            raise ValueError(
                f"Metadata length {len(metadata)} does not match "
                f"vectors length {n_vectors}"
            )
        
        self.metadata.extend(metadata)
        
        logger.info(f"Added {n_vectors} vectors to index. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> List[Dict]:
        """
        Ищет k ближайших соседей для query вектора.
        
        Args:
            query_vector: Вектор запроса (dimension,)
            k: Количество ближайших соседей
        
        Returns:
            List[Dict]: Список похожих инцидентов с расстояниями
                [{
                    "index": int,
                    "distance": float,
                    "metadata": dict
                }, ...]
        
        Example:
            query = embedder.encode_one("Утечка через email")
            similar = store.search(query, k=5)
            
            for item in similar:
                print(f"Distance: {item['distance']:.3f}")
                print(f"Text: {item['metadata']['text']}")
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no results")
            return []
        
        # Убеждаемся что query_vector правильной формы
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Ищем k ближайших
        distances, indices = self.index.search(query_vector, k)
        
        # Формируем результаты
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):  # Проверка на валидность индекса
                results.append({
                    "index": int(idx),
                    "distance": float(dist),
                    "similarity": float(self._distance_to_similarity(dist)),
                    "metadata": self.metadata[idx]
                })
        
        return results
    
    def search_batch(
        self,
        query_vectors: np.ndarray,
        k: int = 5
    ) -> List[List[Dict]]:
        """
        Ищет k ближайших соседей для batch запросов.
        
        Args:
            query_vectors: Матрица векторов запросов (n_queries, dimension)
            k: Количество ближайших соседей для каждого запроса
        
        Returns:
            List[List[Dict]]: Список результатов для каждого запроса
        
        Example:
            queries = embedder.encode(["query1", "query2"])
            results = store.search_batch(queries, k=5)
            
            for i, similar in enumerate(results):
                print(f"Query {i}: {len(similar)} results")
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no results")
            return [[] for _ in range(len(query_vectors))]
        
        # Ищем
        distances, indices = self.index.search(query_vectors, k)
        
        # Формируем результаты для каждого запроса
        all_results = []
        for query_distances, query_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(query_distances, query_indices):
                if idx < len(self.metadata):
                    results.append({
                        "index": int(idx),
                        "distance": float(dist),
                        "similarity": float(self._distance_to_similarity(dist)),
                        "metadata": self.metadata[idx]
                    })
            all_results.append(results)
        
        return all_results
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Конвертирует L2 distance в similarity score (0 to 1).
        
        Args:
            distance: L2 distance
        
        Returns:
            float: Similarity (1 = идентичные, 0 = очень разные)
        """
        # Простая экспоненциальная нормализация
        # Можно настроить коэффициент для разных случаев
        similarity = np.exp(-distance / 10.0)
        return float(similarity)
    
    def save(self, filepath: str | Path) -> None:
        """
        Сохраняет FAISS индекс и метаданные.
        
        Args:
            filepath: Путь к файлу (без расширения)
        
        Example:
            store.save("data/vector_db/incidents")
            # Создаст файлы:
            #   - incidents.faiss (FAISS индекс)
            #   - incidents_metadata.pkl (метаданные)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Убираем расширение если есть
        base_path = filepath.with_suffix('')
        
        # Сохраняем FAISS индекс
        faiss_path = str(base_path) + ".faiss"
        faiss.write_index(self.index, faiss_path)
        logger.info(f"FAISS index saved to {faiss_path}")
        
        # Сохраняем метаданные
        metadata_path = str(base_path) + "_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "dimension": self.dimension
            }, f)
        logger.info(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, filepath: str | Path) -> "FAISSStore":
        """
        Загружает FAISS индекс и метаданные.
        
        Args:
            filepath: Путь к файлу (без расширения)
        
        Returns:
            FAISSStore: Загруженное хранилище
        
        Example:
            store = FAISSStore.load("data/vector_db/incidents")
            similar = store.search(query_vector, k=5)
        """
        filepath = Path(filepath)
        base_path = filepath.with_suffix('')
        
        # Загружаем метаданные
        metadata_path = str(base_path) + "_metadata.pkl"
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
        
        metadata = data["metadata"]
        dimension = data["dimension"]
        
        # Создаём новый объект
        store = cls(dimension=dimension)
        
        # Загружаем FAISS индекс
        faiss_path = str(base_path) + ".faiss"
        store.index = faiss.read_index(faiss_path)
        store.metadata = metadata
        
        logger.info(f"Loaded FAISSStore from {base_path}")
        logger.info(f"Total vectors: {store.index.ntotal}")
        
        return store
    
    def get_stats(self) -> Dict:
        """
        Возвращает статистику хранилища.
        
        Returns:
            Dict: Статистика (количество векторов, размерность и т.д.)
        """
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "metadata_count": len(self.metadata),
            "index_type": type(self.index).__name__
        }


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("FAISS STORE - DEMO")
    logger.info("=" * 80)
    
    # Создаём хранилище
    dimension = 384
    store = FAISSStore(dimension=dimension)
    
    # Генерируем тестовые векторы
    n_vectors = 100
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    
    # Метаданные
    metadata = [
        {
            "id": i,
            "text": f"Incident {i}: Sample description",
            "type": np.random.choice(["email", "usb", "cloud", "printer"])
        }
        for i in range(n_vectors)
    ]
    
    # Добавляем в хранилище
    logger.info(f"\nAdding {n_vectors} vectors...")
    store.add(vectors, metadata)
    
    # Статистика
    stats = store.get_stats()
    logger.info(f"\nStore stats: {stats}")
    
    # Поиск похожих
    logger.info("\nSearching for similar vectors...")
    query = vectors[0]  # Берём первый вектор как запрос
    
    similar = store.search(query, k=5)
    
    logger.info(f"\nFound {len(similar)} similar incidents:")
    for i, item in enumerate(similar):
        logger.info(
            f"{i+1}. Index: {item['index']}, "
            f"Distance: {item['distance']:.3f}, "
            f"Similarity: {item['similarity']:.3f}, "
            f"Type: {item['metadata']['type']}"
        )
    
    # Сохранение
    save_path = config.get_data_path("test_faiss", subdir="vector_db")
    logger.info(f"\nSaving to {save_path}...")
    store.save(save_path)
    
    # Загрузка
    logger.info(f"\nLoading from {save_path}...")
    loaded_store = FAISSStore.load(save_path)
    
    loaded_stats = loaded_store.get_stats()
    logger.info(f"Loaded store stats: {loaded_stats}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Demo complete!")
    logger.info("=" * 80)