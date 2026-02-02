"""
src/vector_db/similarity.py

ЧТО: Высокоуровневый API для поиска похожих DLP-инцидентов
ЗАЧЕМ: Удобная обёртка над FAISS для работы с DataFrame

ИСПОЛЬЗОВАНИЕ:
    from src.vector_db import find_similar_incidents
    
    # Поиск похожих инцидентов
    similar = find_similar_incidents(
        query_text="Утечка через email",
        df=incidents_df,
        k=5
    )
"""

from typing import List, Optional

import pandas as pd

from src.config import get_config
from src.utils import get_logger
from src.vector_db.embeddings import IncidentEmbedder
from src.vector_db.faiss_store import FAISSStore

# Инициализация
logger = get_logger(__name__)
config = get_config()


def find_similar_incidents(
    query_text: str,
    df: pd.DataFrame,
    embedder: Optional[IncidentEmbedder] = None,
    store: Optional[FAISSStore] = None,
    k: int = 5,
    text_column: str = 'description'
) -> pd.DataFrame:
    """
    Находит k похожих инцидентов для текста запроса.
    
    ЗАЧЕМ: На собесе покажешь - "Система автоматически находит похожие 
    инциденты из истории и показывает аналитику как их обрабатывали"
    
    Args:
        query_text: Текст запроса (описание инцидента)
        df: DataFrame с инцидентами
        embedder: IncidentEmbedder (создастся если None)
        store: FAISSStore (создастся если None)
        k: Количество похожих инцидентов
        text_column: Колонка с текстом
    
    Returns:
        pd.DataFrame: DataFrame с похожими инцидентами + similarity scores
    
    Example:
        df = pd.read_csv("incidents.csv")
        
        similar = find_similar_incidents(
            query_text="Отправка email с картами клиентов",
            df=df,
            k=5
        )
        
        print(similar[['description', 'similarity', 'incident_type']])
    """
    logger.info(f"Finding {k} similar incidents for query...")
    
    # Создаём embedder если нужно
    if embedder is None:
        embedder = IncidentEmbedder()
    
    # Создаём store если нужно
    if store is None:
        logger.info("Creating FAISS index from DataFrame...")
        
        # Создаём эмбеддинги для всех инцидентов
        texts = df[text_column].fillna("").tolist()
        vectors = embedder.encode(texts)
        
        # Создаём метаданные
        metadata = [
            {
                "id": i,
                "index": i,
                **row.to_dict()
            }
            for i, row in df.iterrows()
        ]
        
        # Создаём store и добавляем векторы
        store = FAISSStore(dimension=embedder.dimension)
        store.add(vectors, metadata)
    
    # Кодируем query
    query_vector = embedder.encode_one(query_text)
    
    # Ищем похожие
    results = store.search(query_vector, k=k)
    
    # Конвертируем в DataFrame
    similar_data = []
    for item in results:
        data = item['metadata'].copy()
        data['similarity'] = item['similarity']
        data['distance'] = item['distance']
        similar_data.append(data)
    
    similar_df = pd.DataFrame(similar_data)
    
    logger.info(f"Found {len(similar_df)} similar incidents")
    
    return similar_df


def build_similarity_index(
    df: pd.DataFrame,
    save_path: str,
    text_column: str = 'description',
    embedder: Optional[IncidentEmbedder] = None
) -> FAISSStore:
    """
    Строит и сохраняет FAISS индекс для DataFrame.
    
    ЗАЧЕМ: Построить индекс один раз, потом быстро загружать для поиска.
    
    Args:
        df: DataFrame с инцидентами
        save_path: Путь для сохранения индекса
        text_column: Колонка с текстом
        embedder: IncidentEmbedder (создастся если None)
    
    Returns:
        FAISSStore: Построенное хранилище
    
    Example:
        df = pd.read_csv("incidents_30k.csv")
        
        store = build_similarity_index(
            df=df,
            save_path="data/vector_db/incidents_30k"
        )
        
        # Потом можно загрузить:
        # store = FAISSStore.load("data/vector_db/incidents_30k")
    """
    logger.info(f"Building FAISS index for {len(df)} incidents...")
    
    # Создаём embedder
    if embedder is None:
        embedder = IncidentEmbedder()
    
    # Создаём эмбеддинги
    texts = df[text_column].fillna("").tolist()
    vectors = embedder.encode(texts, batch_size=64, show_progress=True)
    
    # Создаём метаданные
    logger.info("Creating metadata...")
    metadata = []
    for i, row in df.iterrows():
        meta = {
            "id": i,
            "index": i,
            **row.to_dict()
        }
        metadata.append(meta)
    
    # Создаём store
    store = FAISSStore(dimension=embedder.dimension)
    store.add(vectors, metadata)
    
    # Сохраняем
    logger.info(f"Saving index to {save_path}...")
    store.save(save_path)
    
    logger.info("Index built and saved successfully!")
    
    return store


def get_incident_clusters(
    df: pd.DataFrame,
    n_clusters: int = 10,
    text_column: str = 'description'
) -> pd.DataFrame:
    """
    Кластеризует инциденты по похожести.
    
    ЗАЧЕМ: На собесе покажешь - "Система автоматически группирует 
    похожие инциденты для анализа паттернов"
    
    Args:
        df: DataFrame с инцидентами
        n_clusters: Количество кластеров
        text_column: Колонка с текстом
    
    Returns:
        pd.DataFrame: DataFrame с добавленной колонкой 'cluster'
    
    Example:
        df = pd.read_csv("incidents.csv")
        df_clustered = get_incident_clusters(df, n_clusters=10)
        
        # Анализ кластеров
        print(df_clustered.groupby('cluster')['incident_type'].value_counts())
    """
    logger.info(f"Clustering {len(df)} incidents into {n_clusters} clusters...")
    
    from sklearn.cluster import KMeans
    
    # Создаём эмбеддинги
    embedder = IncidentEmbedder()
    texts = df[text_column].fillna("").tolist()
    vectors = embedder.encode(texts, show_progress=True)
    
    # KMeans кластеризация
    logger.info("Running KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(vectors)
    
    # Добавляем в DataFrame
    df_result = df.copy()
    df_result['cluster'] = clusters
    
    logger.info("Clustering complete!")
    logger.info(f"Cluster distribution:\n{pd.Series(clusters).value_counts().sort_index()}")
    
    return df_result


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    from src.data import DataLoader
    
    logger.info("=" * 80)
    logger.info("SIMILARITY SEARCH - DEMO")
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
        
        # Тестовый запрос
        query_text = "Отправка email с данными клиентов"
        
        logger.info(f"\nQuery: {query_text}")
        logger.info("-" * 80)
        
        # Ищем похожие
        similar = find_similar_incidents(
            query_text=query_text,
            df=df,
            k=5
        )
        
        # Показываем результаты
        logger.info("\nTop 5 similar incidents:")
        for idx, row in similar.iterrows():
            logger.info(f"\n{idx+1}. Similarity: {row['similarity']:.3f}")
            logger.info(f"   Type: {row['incident_type']}, Severity: {row['severity']}")
            logger.info(f"   {row['description'][:100]}...")
        
        # Кластеризация
        logger.info("\n" + "=" * 80)
        logger.info("CLUSTERING DEMO")
        logger.info("=" * 80)
        
        df_clustered = get_incident_clusters(df, n_clusters=5)
        
        logger.info("\nCluster analysis:")
        for cluster_id in range(5):
            cluster_df = df_clustered[df_clustered['cluster'] == cluster_id]
            logger.info(f"\nCluster {cluster_id}: {len(cluster_df)} incidents")
            logger.info(f"  Types: {cluster_df['incident_type'].value_counts().to_dict()}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Demo complete!")
        logger.info("=" * 80)   