from src.databases import get_chroma_collection, get_sqlite_conn
from src.models.embedders import load_embedder
from src.models.reranker import load_reranker
from src.models.llm import load_qwen_model
from src.utils.retrivers import get_search_results
from src.utils.combinations import search_with_rerank
from src.utils.generate_answer import generate_qwen_answer

if __name__ == "__main__":
    query = input("Введите ваш вопрос: ")
    # query = "Как настроить туннельный режим на ViPNet Coordinator HW1000?"
    
    # Инициализация
    collection = get_chroma_collection()
    connection = get_sqlite_conn()
    embedder = load_embedder("giga")
    reranker = load_reranker()
    llm, tokenizer = load_qwen_model()

    # Поиск
    semantic_res, fts_res = get_search_results(query, collection, connection, embedder, top_k=10)

    # Объединение и реранкинг
    all_res = {r['content']: r for r in semantic_res + fts_res}.values()
    top = search_with_rerank(query, list(all_res), reranker, top_k=5)

    # Генерация ответа
    context = "\n\n".join([f"[стр. {c['page']}]\n{c['content']}" for c in top])
    answer = generate_qwen_answer(query, context, llm, tokenizer)
    print(answer)