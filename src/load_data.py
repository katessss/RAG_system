from pathlib import Path
import torch
import json

from src.processing.pdf_parser import parse_pdf
from src.processing.chunker import process_docling_to_chunks, create_embeddings
from src.databases.chroma import save_to_chroma
from src.databases.sqlite import save_to_sqlite

from logger_config import setup_logger
logger = setup_logger(__name__)
import os
from dotenv import load_dotenv
load_dotenv()
CUR_MODEL_TYPE = os.getenv("MODEL_TYPE")



def check_databases(model_type: str):
    """
    Проверяет, существуют ли файлы SQLite и папка ChromaDB для конкретной модели.
    """
    db_dir = Path("DB")
    sqlite_path = db_dir / "FTS_search.db"
    chroma_path = db_dir / f"semantic_search_db_{model_type}"

    if not db_dir.exists():
        return False

    if not sqlite_path.exists() or sqlite_path.stat().st_size < 1024:
        return False

    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        return False

    return True


def load_data_for_rag(folder_path: str = "data", model_type=CUR_MODEL_TYPE):
    if check_databases(model_type):
        logger.info(f"Базы данных для модели {model_type} уже существуют и не пустые. Пропускаем этап загрузки и обработки данных.")
        return 
        
    logger.info(f"Загрузка данных для модели {model_type}")
    # Загружаем и парсим PDF файлы из папки
    results = parse_pdf(folder_path)

    # Преобразуем результаты парсинга в чанки и создаем эмбеддинги
    all_chunks = []
    for name, result in results.items():
        chunks = process_docling_to_chunks(result=result, max_text_len = 500, min_merge_threshold = 300, file_name = name)
        all_chunks.extend(chunks)
    
    chunks_with_vectors = create_embeddings(all_chunks, model_type, batch_size=4)

    Path("DB").mkdir(parents=True, exist_ok=True)
    
    # Сохраняем чанки с эмбеддингами в ChromaDB
    save_to_chroma(chunks_with_vectors, collection_name="vipnet_docs", db_path=f"DB/semantic_search_db_{model_type}")

    sqlite_path=f"DB/FTS_search.db"
    if not  Path(sqlite_path).exists() or Path(sqlite_path).stat().st_size < 1024:
        logger.info("Сохранение данных в SQLite FTS...")
    # Сохраняем чанки в SQLite для полнотекстового поиска
        save_to_sqlite(all_chunks, db_path=sqlite_path)

    logger.info("Все этапы загрузки данных завершены успешно.")
    return



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", default="data")
    parser.add_argument("--model", default=CUR_MODEL_TYPE)
    args = parser.parse_args()

    load_data_for_rag(folder_path=args.folder, model_type=args.model )    