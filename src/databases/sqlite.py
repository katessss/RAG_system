import sqlite3
import json
import os
from pathlib import Path
from logger_config import setup_logger
logger = setup_logger(__name__)

from src.processing.cleaners import normalize_for_fts

def init_sqlite_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Создаем таблицу с поддержкой полнотекстового поиска FTS5
    # original_content - для выдачи результата
    # stemmed_content - для самого поиска (здесь будет текст после стеммера)
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
            id,
            original_content,
            stemmed_content,
            metadata
        )
    ''')
    conn.commit()
    return conn

def save_to_sqlite(chunks, db_path):
    db_dir = os.path.dirname(db_path)
    if db_dir:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
    
    conn = init_sqlite_db(db_path)
    cursor = conn.cursor()
    
    data_to_insert = []
    for i, chunk in enumerate(chunks):
        # Готовим текст для поиска
        stemmed = normalize_for_fts(chunk["content"])
        
        data_to_insert.append((
            f"id_{i}",
            chunk["content"],
            stemmed,
            json.dumps(chunk["metadata"], ensure_ascii=False)
        ))
    
    # Очищаем старое и вставляем новое
    cursor.execute("DELETE FROM docs_fts")
    cursor.executemany(
        "INSERT INTO docs_fts (id, original_content, stemmed_content, metadata) VALUES (?, ?, ?, ?)",
        data_to_insert
    )
    
    conn.commit()
    conn.close()
    logger.info("Данные успешно сохранены в SQLite FTS5.")