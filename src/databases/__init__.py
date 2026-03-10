import chromadb
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()
CUR_MODEL_TYPE = os.getenv("MODEL_TYPE")
from logger_config import setup_logger
logger = setup_logger(__name__)

def get_chroma_collection(db_path=None, collection_name="vipnet_docs"):
    if db_path is None:
        db_path = f"DB/semantic_search_db_{CUR_MODEL_TYPE}"
    try:
        client = chromadb.PersistentClient(path=db_path)
        coll = client.get_or_create_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Ошибка при подключении к ChromaDB: {e}")
        raise e
    return coll

def get_sqlite_conn(db_path=f"DB/FTS_search.db"):
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
    except Exception as e:
        logger.error(f"Ошибка при подключении к SQLite: {e}")
        raise e
    return conn