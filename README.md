# RAG-система для поиска по документации ViPNet

Система гибридного поиска по технической документации ViPNet Coordinator HW 5. Объединяет семантический поиск (ChromaDB) и полнотекстовый поиск (SQLite FTS5) с переранжированием через cross-encoder. В качестве генератора ответов используется Qwen2.5-7B-Instruct.

---

## Требования

- Python 3.10+
- CUDA-совместимая GPU (рекомендуется от 16 ГБ VRAM)
- 30+ ГБ свободного места на диске (модели + индексы)

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone <repo_url>
cd <repo_name>
```

### 2. Создать виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

> Если GPU недоступна, замените `onnxruntime-gpu` на `onnxruntime` в `requirements.txt` перед установкой.

### 4. Настроить переменные окружения

Создать файл `.env` в корне проекта:

```env
MODEL_TYPE=giga        # e5 | user2 | giga
DEVICE=cuda            # cuda | cpu | cuda:0 | cuda:1
```

---

## Структура проекта

```
├── data/                        # PDF-файлы документации (положить сюда)
├── DB/                          # Векторные и FTS-индексы (создаются автоматически) - хрняить БД в репозитории - дурной тон, но здесь они намеренно, для удобства воспроизведения
│   ├── FTS_search.db
│   ├── semantic_search_db_e5/
│   ├── semantic_search_db_user2/
│   └── semantic_search_db_giga/
├── research/                    # Эксперменты и ноутбуки
├── benchmarks_generation/
│   ├── generate_sintetic.py     # Генерация тестовых вопросов через GPT-4
│   ├── exported_chunks.json     # Экспортированные чанки
│   └── benchmark.json           # Готовый бенчмарк
├── src/
│   ├── databases/
│   │   ├── __init__.py          # Фабричные функции подключения к БД
│   │   ├── chroma.py            # Сохранение в ChromaDB
│   │   └── sqlite.py            # Сохранение в SQLite FTS5
│   ├── models/
│   │   ├── embedders.py         # Загрузка моделей эмбеддингов
│   │   ├── reranker.py          # Загрузка cross-encoder реранкера
│   │   └── llm.py               # Загрузка Qwen
│   ├── processing/
│   │   ├── pdf_parser.py        # Парсинг PDF через Docling
│   │   ├── chunker.py           # Нарезка на чанки + создание эмбеддингов
│   │   └── cleaners.py          # Очистка текста, стемминг, форматирование
│   ├── utils/
│   │   ├── retrivers.py         # Семантический и FTS-поиск
│   │   ├── combinations.py      # Base / RRF / Reranker стратегии
│   │   └── generate_answer.py   # Генерация ответа через Qwen
│   ├── load_data.py             # Точка входа: индексирование документов
│   ├── tests_for_rag.py         # Бенчмарк retrieval-части
│   └── test_for_qwen.py         # Бенчмарк LLM-части
├── temp/                        # Кэш результатов парсинга PDF
├── tests/                       # Кэш результатов парсинга PDF
│   ├── results/
│   │   ├── llm_eval_e5.csv      # Резульататы ответов llm
│   └── benchmark.json           # Синтетический бенчмарк
├── logger_config.py
├── requirements.txt
└── .env
```

---

## Запуск

### Шаг 1 — Положить PDF в папку `data/`

```
data/
├── 01 ViPNet Coordinator HW 5. Подготовка к работе.pdf
├── 02 ViPNet Coordinator HW 5. Настройка в CLI.pdf
└── ...
```

### Шаг 2 — Проиндексировать документы

```bash
python -m src.load_data
```

Скрипт автоматически:
1. Проверит, существуют ли индексы — если да, пропустит индексирование
2. Распарсит PDF через Docling (результаты закэшируются в `temp/`)
3. Нарежет документы на чанки
4. Создаст эмбеддинги и сохранит в ChromaDB
5. Сохранит чанки в SQLite FTS5

> Первый запуск занимает значительное время — Docling загружает свои модели и обрабатывает PDF. Повторные запуски используют кэш из `temp/` и работают быстро.

### Шаг 3 — Запустить поиск / получить ответ

Пример использования в коде:

```python
from src.databases import get_chroma_collection, get_sqlite_conn
from src.models.embedders import load_embedder
from src.models.reranker import load_reranker
from src.models.llm import load_qwen_model
from src.utils.retrivers import get_search_results
from src.utils.combinations import search_with_rerank
from src.utils.generate_answer import generate_qwen_answer

# Инициализация
collection = get_chroma_collection()
connection = get_sqlite_conn()
embedder = load_embedder("giga")
reranker = load_reranker()
llm, tokenizer = load_qwen_model()

# Поиск
query = "Как настроить туннельный режим на ViPNet Coordinator HW1000?"
semantic_res, fts_res = get_search_results(query, collection, connection, embedder, top_k=10)

# Объединение и реранкинг
all_res = {r['content']: r for r in semantic_res + fts_res}.values()
top = search_with_rerank(query, list(all_res), reranker, top_k=5)

# Генерация ответа
context = "\n\n".join([f"[стр. {c['page']}]\n{c['content']}" for c in top])
answer = generate_qwen_answer(query, context, llm, tokenizer)
print(answer)
```

---

## Тестирование и бенчмарк

### Бенчмарк retrieval-части (поиск без LLM)

Тестирует все комбинации моделей × стратегий и выводит Hit@1/5/10 и MRR:

```bash
python -m src.tests_for_rag
```

Результаты выводятся в консоль. Файл бенчмарка должен лежать в `tests/benchmark.json`.

### Бенчмарк LLM-части (полный пайплайн)

Прогоняет вопросы через весь пайплайн и сохраняет ответы модели в CSV:

```bash
python -m src.test_for_qwen
```

Результаты сохраняются в `tests/results/llm_eval_{MODEL_TYPE}.csv` с колонками: вопрос, ответ модели, эталонный чанк, использованный контекст.

### Генерация тестовых вопросов

Если нужно сгенерировать новый бенчмарк из своих чанков (требуется OpenAI API key):

```bash
python -m benchmarks_generation.generate_sintetic
```

Вопросы генерируются моделью GPT-4.1 на основе случайной выборки из `exported_chunks.json` и сохраняются в `benchmark_tasks.json`.

---

## Выбор модели и стратегии

Модель задаётся через `.env` (`MODEL_TYPE`) или передаётся явно в функции. Сравнение по результатам бенчмарка (409 запросов):

| Конфигурация | Hit@1 | MRR | Avg Time |
|---|---|---|---|
| E5 + Reranker | 73.59% | 0.821 | ~252 мс |
| USER2 + Reranker | 74.08% | 0.829 | ~270 мс |
| **GIGA + Reranker** | **74.08%** | **0.829** | ~290 мс |
| GIGA + RRF | 71.15% | 0.809 | ~46 мс |
| E5 + Base | 63.08% | 0.747 | ~21 мс |

**Рекомендации:**
- Максимальное качество → `GIGA` + Reranker
- Баланс качество/скорость → `GIGA` + RRF  
- Минимальные ресурсы → `E5` + Base

---

## Возможные проблемы

**`CUDA out of memory`** — уменьшить `batch_size` в `load_data.py` (параметр `create_embeddings(..., batch_size=4)`).

**Docling долго инициализируется** — это нормально при первом запуске, модели скачиваются автоматически.

**FTS возвращает пустой результат** — система автоматически переключается с AND на OR режим. Если оба пустые — запрос состоит только из стоп-слов.

**`MODEL_TYPE` не задан в `.env`** — по умолчанию используется значение `None`, что вызовет ошибку. Убедитесь, что `.env` создан и заполнен.
