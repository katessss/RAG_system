import chromadb
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()
CUR_MODEL_TYPE = os.getenv("MODEL_TYPE")

def get_chroma_collection(db_path=f"DB/semantic_search_db_{CUR_MODEL_TYPE}", collection_name="vipnet_docs"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name)


def get_sqlite_conn(db_path=f"DB/FTS_search.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn