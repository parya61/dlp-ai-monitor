"""
src.vector_db - Векторная база данных для поиска похожих инцидентов

Модули:
- embeddings.py - Создание эмбеддингов через Sentence-Transformers
- faiss_store.py - FAISS хранилище для векторов
- similarity.py - Поиск похожих инцидентов

Использование:
    from src.vector_db import IncidentEmbedder, FAISSStore
    
    # Создаём эмбеддинги
    embedder = IncidentEmbedder()
    vectors = embedder.encode(incidents)
    
    # Сохраняем в FAISS
    store = FAISSStore()
    store.add(vectors, metadata)
    
    # Ищем похожие
    similar = store.search(query_vector, k=5)
"""

from src.vector_db.embeddings import IncidentEmbedder
from src.vector_db.faiss_store import FAISSStore
from src.vector_db.similarity import find_similar_incidents

__all__ = [
    "IncidentEmbedder",
    "FAISSStore",
    "find_similar_incidents",
]