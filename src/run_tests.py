import time
import torch
import json
import os
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from logger_config import setup_logger

# Загрузка конфигурации
logger = setup_logger(__name__)
load_dotenv()

from src.utils.retrivers import fts_retrieve, semantic_retrieve
from src.utils.combinations import base_search, search_with_rerank, rrf_combination
from src.databases import get_chroma_collection, get_sqlite_conn
from src.models.embedders import load_embedder
from src.models.reranker import load_reranker


def calculate_retrieval_metrics(results, target_content, top_k):
    """Вычисляет ранг и попадание в топ-K для одного запроса"""
    rank = -1
    if results:
        for i, res in enumerate(results[:top_k]):
            if res['content'].strip() == target_content:
                rank = i + 1
                break
    return rank


def get_hybrid_results(query, score_type, top_k, collection, connection, model, reranker, model_type):
    """Универсальный интерфейс для разных стратегий поиска"""
    semantic_res = semantic_retrieve(query=query, collection=collection, model=model, model_type=model_type, top_k=top_k)
    fts_res = fts_retrieve(query=query, connection=connection, type_of_search="AND", top_k=top_k)
    if not fts_res: fts_res = fts_retrieve(query, connection, type_of_search="OR", top_k=top_k)
    
    if score_type == "base":
        return base_search(semantic_res, fts_res, top_k=top_k)
    
    if score_type == "rrf":
        return rrf_combination(semantic_res, fts_res, k=60, top_k=top_k)
    
    if score_type == "reranker":
        # Дедупликация перед реранкером
        seen = set()
        unique_res = []
        for res in (semantic_res + fts_res):
            content = res['content'].strip()
            if content not in seen:
                unique_res.append(res)
                seen.add(content)
        return search_with_rerank(query, unique_res, reranker, top_k=top_k)
    
    return []


def run_metrics_benchmark(benchmark_data, search_logic_func, top_k=10, desc=""):
    """Универсальное ядро для запуска тестов и сбора метрик"""
    hits = {1: 0, 5: 0, 10: 0}
    mrr_sum = 0
    total_time = 0
    total_queries = len(benchmark_data)

    logger.info(f"Запуск бенчмарка: {desc} ({total_queries} запросов)")

    for item in tqdm(benchmark_data, desc=desc):
        query = item['question']
        target_content = item['original_content'].strip()

        start_time = time.perf_counter()
        results = search_logic_func(query)
        total_time += (time.perf_counter() - start_time)

        rank = calculate_retrieval_metrics(results, target_content, top_k)
        
        if rank > 0:
            if rank == 1: hits[1] += 1
            if rank <= 5: hits[5] += 1
            hits[10] += 1
            mrr_sum += (1.0 / rank)

    avg_time = (total_time / total_queries) * 1000
        
    print(f"\nРезультаты [{desc}]:")
    print("\n" + "="*40)
    print(f"РЕЗУЛЬТАТЫ БЕНЧМАРКА ДЛЯ {model_type}")
    print("="*40)
    print(f"Hit@1:  {(hits[1]/total_queries)*100:>6.2f}% )")
    print(f"Hit@5:  {(hits[5]/total_queries)*100:>6.2f}%")
    print(f"Hit@10: {(hits[10]/total_queries)*100:>6.2f}%")
    print(f"MRR:    {mrr_sum/total_queries:>8.4f}")
    print(f"Avg Time: {avg_time:>6.2f} ms")
    print("="*40)
    
    
    return {
        "Hit@1": round((hits[1]/total_queries)*100, 2),
        "Hit@5": round((hits[5]/total_queries)*100, 2),
        "Hit@10": round((hits[10]/total_queries)*100, 2),
        "MRR": round(mrr_sum/total_queries, 4),
        "Avg_Time_ms": round(avg_time, 2)
    }


def evaluate_strategies_for_model(benchmark_data, model_type, device="cuda:2"):
    """Сравнивает Base vs RRF vs Reranker для конкретной модели"""
    model = load_embedder(model_type, device)
    collection = get_chroma_collection(db_path=f"DB/semantic_search_db_{model_type}")
    connection = get_sqlite_conn()
    reranker = load_reranker() 
    
    results = {}
    for strategy in ["base", "rrf", "reranker"]:
        search_callback = lambda q: get_hybrid_results(q, strategy, 10, collection, connection, model, reranker, model_type)
        results[strategy] = run_metrics_benchmark(benchmark_data, search_callback, 10, f"{model_type}_{strategy}")
    
    del model, reranker
    torch.cuda.empty_cache()
    return results


def evaluate_model(benchmark_data, model_type, top_k=10):
    model = load_embedder(model_type)
    collection = get_chroma_collection(db_path=f"DB/semantic_search_db_{model_type}")
    hits = {1: 0, 5: 0, 10: 0}
    mrr_sum = 0
    total_time = 0
    total_queries = len(benchmark_data)

    for item in tqdm(benchmark_data, desc=f"Testing {model_type}"):
        query = item['question']
        target_content = item['original_content'].strip()

        start_time = time.perf_counter()
        results = semantic_retrieve(query, collection, model, model_type=model_type, top_k=top_k)
        total_time += (time.perf_counter() - start_time)

        rank = calculate_retrieval_metrics(results, target_content, top_k)
        
        if rank > 0:
            if rank == 1: hits[1] += 1
            if rank <= 5: hits[5] += 1
            hits[10] += 1
            mrr_sum += (1.0 / rank)

        avg_time = (total_time / total_queries) * 1000

    print("\n" + "="*40)
    print(f"РЕЗУЛЬТАТЫ БЕНЧМАРКА ДЛЯ {model_type}")
    print("="*40)
    print(f"Hit@1:  {(hits[1]/total_queries)*100:>6.2f}% )")
    print(f"Hit@5:  {(hits[5]/total_queries)*100:>6.2f}%")
    print(f"Hit@10: {(hits[10]/total_queries)*100:>6.2f}%")
    print(f"MRR:    {mrr_sum/total_queries:>8.4f}")
    print(f"Avg Time: {avg_time:>6.2f} ms")
    print("="*40)
        

    return {
        "Hit@1": round((hits[1]/total_queries)*100, 2),
        "Hit@5": round((hits[5]/total_queries)*100, 2),
        "Hit@10": round((hits[10]/total_queries)*100, 2),
        "MRR": round(mrr_sum/total_queries, 4),
        "Time": round(avg_time, 2)
    }


if __name__ == "__main__":
    BENCHMARK_PATH = "tests/benchmark.json"
    if not Path(BENCHMARK_PATH).exists():
        logger.error("Файл бенчмарка не найден!")
        exit()

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Глобальный отчет
    report = {}

    # Запускаем цикл по всем интересующим моделям
    for m_type in ["e5", "user2", "giga"]:
        data= evaluate_model(queries, m_type) 
    # # Итоговое сравнение в консоли
    # print("\n" + "="*50)
    # print("ИТОГОВАЯ ТАБЛИЦА (Hit@1)")
    # print("="*50)
    # for m, strats in report.items():
    #     line = f"{m.upper():<8} | "
    #     line += " | ".join([f"{s}: {data['Hit@1']}%" for s, data in strats.items()])
    #     print(line)
    # print("="*50)