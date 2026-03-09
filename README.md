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
git clone <https://github.com/katessss/RAG_system.git>
cd <RAG_system>
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
│   ├── run_query.py             # Запуск одного вопроса
│   └── test_for_qwen.py         # Бенчмарк LLM-части
├── temp/                        # Кэш результатов парсинга PDF
├── tests/                       
│   ├── results/
│   │   ├── llm_eval_e5.csv      # Резульататы ответов llm
│   │   ├── benchmark_report_for_e5.csv            # Резульататы бенчамарка для модеи e5
│   │   ├── benchmark_report_all.csv               # Резульататы бенчамарка все комбинации моделей × стратегий
│   │   ├── benchmark_report_for_semantic.csv      # Резульататы бенчамарка только сравнения семантического поиска
│   └── benchmark.json           # Синтетический бенчмарк
├── logger_config.py
├── requirements.txt
└── .env
```

---

## Запуск

### Шаг 1 - Положить PDF в папку `data/`

```
data/
├── 01 ViPNet Coordinator HW 5. Подготовка к работе.pdf
├── 02 ViPNet Coordinator HW 5. Настройка в CLI.pdf
└── ...
```

Можно использовать другую папку - тогда передавать путь через аргумент `--folder`.

### Шаг 2 - Проиндексировать документы

```bash
python -m src.load_data
python -m src.load_data --folder path/to/docs --model giga
```

Скрипт автоматически:
1. Проверит, существуют ли индексы - если да, пропустит индексирование
2. Распарсит PDF через Docling (результаты закэшируются в `temp/`)
3. Нарежет документы на чанки
4. Создаст эмбеддинги и сохранит в ChromaDB
5. Сохранит чанки в SQLite FTS5

> Первый запуск занимает значительное время - Docling загружает свои модели и обрабатывает PDF. Повторные запуски используют кэш из `temp/` и работают быстро.
>
> Все остальные точки входа сами проверяют наличие нужных индексов и при необходимости запускают индексирование - можно не вызывать `load_data.py` отдельно.

### Шаг 3 - Интерактивный режим (чат)

```bash
python -m src.run_query
python -m src.run_query --folder path/to/docs --model giga
```

Работает в режиме бесконечного цикла. Для выхода - `Ctrl+C`.

---

## Тестирование и бенчмарк

### Бенчмарк retrieval-части (поиск без LLM)

```bash
python -m src.tests_for_rag
python -m src.tests_for_rag --model giga --type_of_test all
```

| Аргумент | Описание | По умолчанию |
|---|---|---|
| `--path_to_bench` | Путь к файлу бенчмарка | `tests/benchmark.json` |
| `--folder` | Путь к папке с PDF | `data` |
| `--model` | Модель эмбеддингов | из `.env` |
| `--type_of_test` | Режим тестирования | `cur_model` |

Режимы `--type_of_test`:

- `cur_model` - все три стратегии (Base / RRF / Reranker) для модели из `.env`
- `for_semantic` - сравнение только эмбеддеров (e5, user2, giga), без FTS
- `all` - полный прогон: все модели × все стратегии

Результаты выводятся в консоль и сохраняются в `tests/results/benchmark_report_[type_of_test].json`.

> **Примечание:** В репозитории уже лежат готовые индексы для всех трёх моделей (`DB/semantic_search_db_e5`, `_user2`, `_giga`), поэтому тестировать можно сразу без переиндексации.

### Бенчмарк LLM-части (полный пайплайн)

```bash
python -m src.test_for_qwen
python -m src.test_for_qwen --model giga --path_to_bench tests/benchmark.json
```

| Аргумент | Описание | По умолчанию |
|---|---|---|
| `--path_to_bench` | Путь к файлу бенчмарка | `tests/benchmark.json` |
| `--folder` | Путь к папке с PDF | `data` |
| `--model` | Модель эмбеддингов | из `.env` |

Результат сохраняется в `tests/results/llm_eval_[model].csv` с колонками: вопрос, ответ ИИ, эталонный текст, номер страницы.

---

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

**`CUDA out of memory`** - уменьшить `batch_size` в `load_data.py` (параметр `create_embeddings(..., batch_size=4)`).

**Docling долго инициализируется** - это нормально при первом запуске, модели скачиваются автоматически.

**FTS возвращает пустой результат** - система автоматически переключается с AND на OR режим. Если оба пустые - запрос состоит только из стоп-слов.

**`MODEL_TYPE` не задан в `.env`** - по умолчанию используется значение `None`, что вызовет ошибку. Убедитесь, что `.env` создан и заполнен.
