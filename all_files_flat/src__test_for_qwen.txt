import json
import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

from src.databases import get_chroma_collection, get_sqlite_conn
from src.models.embedders import load_embedder
from src.models.reranker import load_reranker
from src.models.llm import load_qwen_model
from src.utils.combinations import search_with_rerank
from src.utils.generate_answer import generate_qwen_answer
from src.utils.retrivers import get_search_results

from logger_config import setup_logger
logger = setup_logger(__name__)

load_dotenv()
CUR_MODEL_TYPE = os.getenv("MODEL_TYPE")
DEVICE=os.getenv("DEVICE", "cuda")

def evaluate_llm(benchmark_data):  
    results_report = []
    
    # загружаем что надо
    collection = get_chroma_collection(db_path=f"DB/semantic_search_db_{CUR_MODEL_TYPE}")
    connection = get_sqlite_conn()
    print(CUR_MODEL_TYPE)
    model = load_embedder(CUR_MODEL_TYPE)
    reranker = load_reranker() 
    llm, token = load_qwen_model()

    for item in tqdm(benchmark_data, desc="LLM Benchmarking"):
        query = item['question']
        semantic_res, fts_res = get_search_results(query, collection, connection, model, top_k=10)

        # запускам реранкер 
        seen = set()
        unique_res = []
        for res in (semantic_res + fts_res):
            content = res['content'].strip()
            if content not in seen:
                unique_res.append(res)
                seen.add(content)
        contexts = search_with_rerank(query, unique_res, reranker, top_k=5)
        full_context = "\n\n".join([
            f"[Источник: стр. {c['page']}, раздел {c['context']}]\n{c['content']}" 
            for c in contexts
        ])
        
        # ответ от qwen
        answer = generate_qwen_answer(query=query, context=full_context, model=llm, tokenizer=token)

        results_report.append({
            "question": query,
            "answer": answer,
            "truth": item.get('original_content'),
            "true_page": item.get('expected_page', "unknown"),
            "used_context": full_context, 
            })

    return results_report

if __name__=="__main__":
    BENCHMARK_PATH = "tests/benchmark.json"
    if not Path(BENCHMARK_PATH).exists():
        logger.error("Файл бенчмарка не найден!")
        exit()

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = evaluate_llm(data)

    output_folder = Path("tests/results")
    output_folder.mkdir(exist_ok=True)
    
    csv_file = output_folder / f"llm_eval_{CUR_MODEL_TYPE}.csv"
    
    df = pd.DataFrame(report)
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    
    logger.info(f"Тест завершен! Результаты сохранены в CSV: {csv_file}")    