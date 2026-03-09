from src.databases import get_chroma_collection, get_sqlite_conn
from src.models.embedders import load_embedder
from src.models.reranker import load_reranker
from src.models.llm import load_qwen_model
from src.utils.retrivers import get_search_results
from src.utils.combinations import search_with_rerank
from src.utils.generate_answer import generate_qwen_answer
from src.load_data import load_data_for_rag

import os
import argparse
from dotenv import load_dotenv
load_dotenv()
CUR_MODEL_TYPE = os.getenv("MODEL_TYPE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Вопрос к RAG системы ViPNet")

    parser.add_argument("--folder", default="data")
    parser.add_argument("--model", default=CUR_MODEL_TYPE)
    args = parser.parse_args()
    model_type=args.model
    folder_path=args.folder

    load_data_for_rag(folder_path=folder_path, model_type=model_type)      

    # Инициализация
    collection = get_chroma_collection(db_path=f"DB/semantic_search_db_{model_type}")
    connection = get_sqlite_conn(db_path=f"DB/FTS_search.db")
    embedder = load_embedder(model_type)
    reranker = load_reranker()
    llm, tokenizer = load_qwen_model()
    
    while True:
        query = input("Введите ваш вопрос: ")
        # query = "Как настроить туннельный режим на ViPNet Coordinator HW1000?"
        
        # Поиск
        semantic_res, fts_res = get_search_results(query, collection, connection, embedder, model_type, top_k=10)

        # Объединение и реранкинг
        all_res = {r['content']: r for r in semantic_res + fts_res}.values()
        top = search_with_rerank(query, list(all_res), reranker, top_k=5)

        # Генерация ответа
        context = "\n\n".join([f"[Источник: стр. {c['page']}, раздел '{c['context']}']\n{c['content']}" for c in top])

        answer = generate_qwen_answer(query, context, llm, tokenizer)
        print("-" * 50)
        print(f"\nОТВЕТ:\n{answer}")
        print("\n" + "-" * 50)
