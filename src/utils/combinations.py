import sqlite3
import json
from logger_config import setup_logger
logger = setup_logger(__name__)


def base_search(semantic_results, fts_results, top_k=5):    
    # Объединяем результаты и удаляем дубликаты
    combined_dict = {}
    
    for res in fts_results:
        content_hash = res['content'].strip()
        res['method'] = 'fts'
        combined_dict[content_hash] = res

    for res in semantic_results:
        content_hash = res['content'].strip()
        if content_hash in combined_dict:
            # Если чанк найден обоими методами, повышаем его приоритет
            combined_dict[content_hash]['score'] += res['score']
            combined_dict[content_hash]['method'] = 'hybrid'
        else:
            res['method'] = 'semantic'
            combined_dict[content_hash] = res

    final_results = sorted(
        combined_dict.values(), 
        key=lambda x: x['score'], 
        reverse=True
    )

    top_results = final_results
    
    return top_results


def search_with_rerank(query, search_results, reranker_model, top_k=5):
    
    if not search_results:
        return []

    rerank_pairs = []
    for res in search_results:
        rerank_pairs.append([query, res['content']])
    
    scores = reranker_model.predict(rerank_pairs)

    # обновляем score и сортируем заново
    for i, res in enumerate(search_results):
        res['reranker_score'] = float(scores[i]) 
    
    # Сортируем от большего к меньшему (у реранкера score может быть и отрицательным, и > 1)
    final_results = sorted(search_results, key=lambda x: x['reranker_score'], reverse=True)

    return final_results[:top_k]


def rrf_combination(semantic_results, fts_results, k=60, top_k=5):
    rrf_scores = {}
    info_map = {}
    
    for rank, res in enumerate(semantic_results):
        content = res['content']
        rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (k + rank + 1)

        if content not in info_map:
            info_map[content] = res
    
    for rank, res in enumerate(fts_results):
        content = res['content']
        rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (k + rank + 1)

        if content not in info_map:
            info_map[content] = res

    sorted_content = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    final_results = []
    for content in sorted_content[:top_k]:
        merged_res = info_map[content].copy()
        merged_res['rrf_score'] = round(rrf_scores[content], 5)
        final_results.append(merged_res)
        
    return final_results
        