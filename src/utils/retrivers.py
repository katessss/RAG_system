import sqlite3
import json
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
CUR_MODEL_TYPE = os.getenv("MODEL_TYPE")

from logger_config import setup_logger
logger = setup_logger(__name__)

from src.processing.cleaners import prepare_query_from_natural_language, format_text
    

def get_search_results(query, collection, sqlite_conn, model, model_type, top_k=5):
    fts_results = fts_retrieve(query=query, connection=sqlite_conn, type_of_search="AND", top_k=top_k)
    semantic_results = semantic_retrieve(query=query, collection=collection, model=model, model_type=model_type, top_k=top_k)
    
    # Если по AND ничего не нашли, пробуем более мягкий поиск через OR
    if not fts_results:
        fts_results = fts_retrieve(query=query, connection=sqlite_conn, type_of_search="OR", top_k=top_k)

    return semantic_results, fts_results



def fts_retrieve(query, connection, type_of_search="AND", top_k=5):
    """
    Выполняет полнотекстовый поиск в SQLite.
    Возвращает список самых подходящих чанков.
    """
    # список стемов
    stems = prepare_query_from_natural_language(query)
    
    if not stems:
        logger.warning(f"Пустой запрос после обработки: '{query}'")
        return []

    # форматриуем для FTS5: добавляем '*' к каждому слову и соединяем через AND/OR
    formatted_query = f" {type_of_search} ".join([f"{s}*" for s in stems])
    
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    try:
        # MATCH
        # В SQLite чем число МЕНЬШЕ, тем ЛУЧШЕ совпадение
        search_sql = """
            SELECT 
                original_content, 
                metadata, 
                bm25(docs_fts) as rank
            FROM docs_fts 
            WHERE stemmed_content MATCH ? 
            ORDER BY rank 
            LIMIT ?
        """
        
        cursor.execute(search_sql, (formatted_query, top_k))
        rows = cursor.fetchall()

    except sqlite3.OperationalError as e:
        # Если в запросе есть спецсимволы, которые SQLite не понял
        logger.error(f"Ошибка в синтаксисе MATCH: {e} | Запрос: {formatted_query}")
        return []

    # форматируем результат
    results = []
    for row in rows:
        meta = json.loads(row["metadata"])
        results.append({
            "content": row["original_content"],
            "page": meta.get("page", 0),
            "context": meta.get("context", ""),
            "source": meta.get("source", ""),
            "score": round(-row["rank"], 4) # чтобы чем больше, тем лучше
        })

    return results


    
def semantic_retrieve(query, collection, model, model_type=CUR_MODEL_TYPE, top_k=5):
    """
    Выполняет поиск по смыслу: превращает вопрос в вектор и находит самые похожие куски текста в ChromaBD.
    """
    try:
        prepared_query = format_text(query, model_type, "QUERY")

        if model_type=="e5" or model_type=="user2":
            query_vector = model.encode(
                [prepared_query], 
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()

        elif model_type=="giga":
            task = "Дан вопрос, необходимо найти абзац текста с ответом"
            prompt = f"Instruct: {task}\nQuery: "
            query_vector = model.encode(
                [prepared_query],
                prompt=prompt,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32).tolist()

        else:
            logger.error(f"Модель {model_type} не поддерживается!")
            raise ValueError(f"Unsupported model type: {model_type}")

        
        results = collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results["ids"] or len(results["ids"][0]) == 0:
            logger.warning(f"По запросу '{query}' ничего не найдено в векторной базе.")
            return []

        # Форматирование результатов
        formatted_results = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            
            formatted_results.append({
                "content": results["documents"][0][i],
                "page": metadata.get("page", 0),
                "context": metadata.get("context", "Не указан"),
                "source": metadata.get("source", "Неизвестный файл"), 
                "score": round(1 - results["distances"][0][i] / 2, 4) # ChromaDB по умолчанию использует L2 distance, используем формулу L2 = 2 - 2 cosine 
            })

        return formatted_results

    except Exception as e:
        logger.error(f"Ошибка при выполнении семантического поиска: {e}")
        return []


