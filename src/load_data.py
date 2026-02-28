from pathlib import Path
import torch
import json

from src.processing.pdf_parser import parse_pdf
from src.processing.chunker import process_docling_to_chunks, create_embeddings
from src.databases.chroma import save_to_chroma
from src.databases.sqlite import save_to_sqlite

import os
from dotenv import load_dotenv
load_dotenv()
CUR_MODEL_TYPE = "giga"#os.getenv("MODEL_TYPE")
# EMMBEDER_MODEL = load_e5_model("intfloat/multilingual-e5-large", "cuda:2")
# LLM_MODEL = load_model("Qwen/Qwen2.5-7B-Instruct", "cuda" if torch.cuda.is_available() else "cpu")

def load_data_for_rag(folder_path: str = "data"):
    # Загружаем и парсим PDF файлы из папки
    # results = parse_pdf(folder_path)

    # Преобразуем результаты парсинга в чанки и создаем эмбеддинги
    # all_chunks = []
    # for name, result in results.items():
    #     chunks = process_docling_to_chunks(result=result, max_text_len = 500, min_merge_threshold = 300, file_name = name)
    #     all_chunks.extend(chunks)
    file_path = "benchmarks_generation/exported_chunks.json"
    with open(file_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)        
    print(f"Успешно загружено {len(all_chunks)} чанков из файла {file_path}")
    
    chunks_with_vectors = create_embeddings(all_chunks, CUR_MODEL_TYPE, batch_size=4)

    # Path("DB").mkdir(parents=True, exist_ok=True)
    
    # Сохраняем чанки с эмбеддингами в ChromaDB
    save_to_chroma(chunks_with_vectors, collection_name="vipnet_docs", db_path=f"DB/semantic_search_db_{CUR_MODEL_TYPE}")

    # Сохраняем чанки в SQLite для полнотекстового поиска
    # save_to_sqlite(all_chunks, db_path=f"DB/FTS_search.db")

if __name__=='__main__':
    load_data_for_rag()