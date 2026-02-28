import re
from typing import List, Dict, Any
from docling_core.types.doc import TableItem, PictureItem, TextItem, DocItemLabel
from logger_config import setup_logger
logger = setup_logger(__name__)
import numpy as np

from src.models.embedders import load_embedder
from src.processing.cleaners import clean_content_noise, is_junk_text, format_text
    
def process_docling_to_chunks(result, max_text_len: int = 500, min_merge_threshold: int = 300, file_name: str = '') -> List[Dict[str, Any]]:
    chunks = []
    text_buffer = []
    current_len = 0
    current_header = "Общая информация"
    pending_table_name = "" # Сюда будем ловить текст типа "Таблица 1..."

    for element, _ in result.iterate_items():
        label = element.label
        element_text = getattr(element, "text", "").strip()
        #Пропускаем оглавление 
        if label == DocItemLabel.DOCUMENT_INDEX or "Содержание" in element_text:
            continue

        # ОТСЛЕЖИВАНИЕ ЗАГОЛОВКОВ
        if label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]:
            # Если в буфере что-то было - сохраняем перед новым разделом
            if text_buffer:
                content = clean_content_noise(f"Раздел: {current_header}\n" + "\n".join(text_buffer))
                if not is_junk_text(content):
                    chunks.append({
                        "type": "text",
                        "content": content,
                        "metadata": {
                            "page": getattr(element.prov[0], 'page_no', None) if element.prov else None,
                            "file": file_name
                        }

                    })
                text_buffer, current_len = [], 0
            current_header = element_text
            continue

        # ЛОВИМ НАЗВАНИЯ ТАБЛИЦ
        # Если текст начинается со слова "Таблица" и он короткий
        if isinstance(element, TextItem) and re.match(r"^Таблица\s+\d+", element.text.strip()):
            pending_table_name = element.text.strip()
            continue

        #  ОБРАБОТКА ТАБЛИЦ
        if isinstance(element, TableItem):
            table_md = element.export_to_markdown(doc=result)
            content = f"Таблица: {pending_table_name}\n{table_md}" if pending_table_name else table_md
        
            chunks.append({
                "type": "table",
                "content": clean_content_noise(content),
                "metadata": {
                    "context": current_header,
                    "page": getattr(element.prov[0], 'page_no', None) if element.prov else None,
                    "file": file_name
                }
            })
            pending_table_name = "" # Очищаем
            continue


        # ОБРАБОТКА КАРТИНОК 
        elif isinstance(element, PictureItem):
            # Пробуем найти подпись. Если её нет, используем pending_table_name (если там было "Рисунок...")
            caption = ""
            if hasattr(element, 'caption') and element.caption:
                caption = element.caption.text.strip()
            elif pending_table_name and "Рисунок" in pending_table_name:
                caption = pending_table_name

            # Если подпись так и не нашли - пропускаем, чтобы не плодить мусор
            if not caption:
                continue

            if text_buffer: # Сбрасываем текст перед картинкой
                chunks.append({
                    "type": "text",
                    "content": clean_content_noise(f"Раздел: {current_header}\n" + "\n".join(text_buffer)),
                    "metadata": {
                        "context": current_header,
                        "page": getattr(element.prov[0], 'page_no', None) if element.prov else None,
                        "file": file_name
                    }
                })
                text_buffer, current_len = [], 0

            chunks.append({
                "type": "picture",
                "content": f"Раздел: {current_header}\nНайдена схема/рисунок: {caption}",
                "metadata": {
                    "context": current_header,
                    "page": getattr(element.prov[0], 'page_no', None) if element.prov else None,
                    "file": file_name
                }
            })
            pending_table_name = ""
            continue

            
        # ОБРАБОТКА ТЕКСТА
        elif isinstance(element, TextItem):
            content = element.text.strip()
            if not content: continue
            
            text_buffer.append(content)
            current_len += len(content)

            if current_len >= max_text_len:
                content = clean_content_noise(f"Раздел: {current_header}\n" + "\n".join(text_buffer))
                if not is_junk_text(content):
                    chunks.append({
                        "type": "text",
                        "content": content,
                        "metadata": {
                            "page": getattr(element.prov[0], 'page_no', None) if element.prov else None,
                            "file": file_name
                        }
                    })
                text_buffer, current_len = [], 0
    # Хвост
    if text_buffer:
        content = clean_content_noise(f"Раздел: {current_header}\n" + "\n".join(text_buffer))
        if not is_junk_text(content):
            chunks.append({"type": "text", "content": content, "metadata": {"page": None, "file": file_name}})

    # ФИНАЛЬНАЯ СКЛЕЙКА МЕЛКИХ ЧАНКОВ
    optimized_chunks = []
    for chunk in chunks:
        if not optimized_chunks:
            optimized_chunks.append(chunk)
            continue
        
        prev = optimized_chunks[-1]
        
        # Условие склейки: предыд. чанк - текст и он слишком короткий (<300 симв)
        if prev["type"] == "text" and chunk["type"] == "text" and len(prev["content"]) < min_merge_threshold:
            # Приклеиваем текущий чанк к предыдущему
            prev["content"] += "\n\n" + chunk["content"]
            # Обновляем метаданные (страницу берем из более крупного куска)
            if not prev["metadata"]["page"]:
                prev["metadata"]["page"] = chunk["metadata"]["page"]
        else:
            optimized_chunks.append(chunk)

    return optimized_chunks



def create_embeddings(data_list, model_type, batch_size=64):
    """
    Создает эмбеддинги для списка чанков.
    Использует GPU для пакетной обработки (batch processing).
    """
    if not data_list:
        print("Список данных пуст.")
        return []

    logger.info(f"Подготовка текстов для {len(data_list)} чанков...")
    model = load_embedder(model_type)
    
    # Используем 'context' из метаданных как заголовок для эмбеддинга
    prepared_texts = []
    for item in data_list:
        context = item.get("metadata", {}).get("context", "Общая информация")
        text = item.get("content", "")
        formated_text = format_text(f"{context}. {text}", model_type, "PASSAGE")
        prepared_texts.append(formated_text)

    print(f"Начинаем расчет векторов на {model.device} (batch_size={batch_size})...")
    
    all_vectors = model.encode(
        prepared_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    final_list = []
    for i, item in enumerate(data_list):
        new_item = item.copy()
        new_item["vectors"] = all_vectors[i].tolist()
        final_list.append(new_item)

    print("Все эмбеддинги созданы успешно.")
    return final_list