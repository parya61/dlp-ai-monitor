"""
scripts/generate_large_dataset.py

ЧТО: Скрипт для генерации большого датасета DLP-инцидентов + векторная БД
ЗАЧЕМ: Production-ready данные для обучения ML модели

ЗАПУСК:
    python scripts/generate_large_dataset.py --n_incidents 30000

ЧТО СОЗДАЁТСЯ:
1. data/synthetic/incidents_30k.csv - датасет инцидентов
2. data/vector_db/incidents_30k.faiss - FAISS индекс
3. data/vector_db/incidents_30k_metadata.pkl - метаданные
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.data import DataLoader
from src.data.augmentation import EnhancedDLPGenerator
from src.utils import get_logger
from src.vector_db import IncidentEmbedder, FAISSStore

# Инициализация
logger = get_logger(__name__)
config = get_config()


def generate_dataset(
    n_incidents: int = 30000,
    output_name: str = None,
    seed: int = 42
):
    """
    Генерирует большой датасет + векторную БД.
    
    Args:
        n_incidents: Количество инцидентов
        output_name: Название файлов (default: incidents_{n}k)
        seed: Random seed
    """
    logger.info("=" * 80)
    logger.info(f"GENERATING LARGE DATASET: {n_incidents} incidents")
    logger.info("=" * 80)
    
    # Определяем имя файлов
    if output_name is None:
        n_k = n_incidents // 1000
        output_name = f"incidents_{n_k}k"
    
    # Пути
    csv_path = config.get_data_path(f"{output_name}.csv", subdir="synthetic")
    faiss_path = config.get_data_path(output_name, subdir="vector_db")
    
    # =========================================================================
    # ШАГ 1: ГЕНЕРАЦИЯ ДАТАСЕТА
    # =========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Generating incidents")
    logger.info("=" * 80)
    
    generator = EnhancedDLPGenerator(seed=seed)
    df = generator.generate(n_incidents=n_incidents, show_progress=True)
    
    # Сохраняем CSV
    loader = DataLoader()
    loader.save_csv(df, csv_path)
    
    logger.info(f"\n✅ Dataset saved: {csv_path}")
    logger.info(f"   Shape: {df.shape}")
    logger.info(f"   Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Статистика
    logger.info("\n📊 Dataset statistics:")
    logger.info(f"   Incident types: {df['incident_type'].value_counts().to_dict()}")
    logger.info(f"   Severity: {df['severity'].value_counts().to_dict()}")
    logger.info(f"   Departments: {df['department'].nunique()} unique")
    logger.info(f"   Users: {df['user'].nunique()} unique")
    
    # =========================================================================
    # ШАГ 2: СОЗДАНИЕ ЭМБЕДДИНГОВ
    # =========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Creating embeddings")
    logger.info("=" * 80)
    
    embedder = IncidentEmbedder()
    
    texts = df['description'].fillna("").tolist()
    vectors = embedder.encode(texts, batch_size=64, show_progress=True)
    
    logger.info(f"\n✅ Embeddings created")
    logger.info(f"   Shape: {vectors.shape}")
    logger.info(f"   Dimension: {embedder.dimension}")
    logger.info(f"   Memory: {vectors.nbytes / 1024 / 1024:.2f} MB")
    
    # =========================================================================
    # ШАГ 3: СОЗДАНИЕ FAISS ИНДЕКСА
    # =========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Building FAISS index")
    logger.info("=" * 80)
    
    # Создаём метаданные
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
    store.save(faiss_path)
    
    logger.info(f"\n✅ FAISS index saved: {faiss_path}")
    
    # Статистика
    stats = store.get_stats()
    logger.info(f"\n📊 FAISS stats:")
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")
    
    # =========================================================================
    # ШАГ 4: ТЕСТИРОВАНИЕ ПОИСКА
    # =========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Testing similarity search")
    logger.info("=" * 80)
    
    # Тестовый запрос
    test_query = "Отправка email с данными клиентов на личную почту"
    logger.info(f"\nTest query: {test_query}")
    
    query_vector = embedder.encode_one(test_query)
    similar = store.search(query_vector, k=5)
    
    logger.info(f"\nTop 5 similar incidents:")
    for i, item in enumerate(similar):
        meta = item['metadata']
        logger.info(f"\n{i+1}. Similarity: {item['similarity']:.3f}")
        logger.info(f"   Type: {meta.get('incident_type')}, Severity: {meta.get('severity')}")
        logger.info(f"   {meta.get('description', '')[:100]}...")
    
    # =========================================================================
    # ФИНАЛ
    # =========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ GENERATION COMPLETE!")
    logger.info("=" * 80)
    
    logger.info("\n📁 Created files:")
    logger.info(f"   1. {csv_path}")
    logger.info(f"   2. {faiss_path}.faiss")
    logger.info(f"   3. {faiss_path}_metadata.pkl")
    
    logger.info("\n🚀 Next steps:")
    logger.info("   1. Train ML model: python -m src.ml.train")
    logger.info("   2. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    logger.info("   3. Test similarity: python -m src.vector_db.similarity")
    
    return df, store


def main():
    """Main функция для CLI."""
    parser = argparse.ArgumentParser(
        description="Generate large DLP incidents dataset with vector DB"
    )
    
    parser.add_argument(
        "--n_incidents",
        type=int,
        default=30000,
        help="Number of incidents to generate (default: 30000)"
    )
    
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output file name (default: incidents_{n}k)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Генерируем
    generate_dataset(
        n_incidents=args.n_incidents,
        output_name=args.output_name,
        seed=args.seed
    )


if __name__ == "__main__":
    main()