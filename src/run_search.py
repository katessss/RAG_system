from logger_config import setup_logger
logger = setup_logger(__name__)

from src.utils.retrivers import hybrid_search, search_with_rerank
from src.databases import get_sqlite_conn, get_chroma_collection
from src.models.embedder import load_e5_model
from src.models.reranker import load_reranker


if __name__=="__main__":
    
    
    
    # Выводим для проверки
    for i, chunk in enumerate(results):
        print(f"[{i+1}] Method: {chunk['method']} | Page: {chunk['page']} | Score: {chunk['score']} | RRF Score: {chunk['rrf_score']} | Reranker score {chunk['reranker_score']}")
        print(f"Content: {chunk['content'][:350]}...\n")
