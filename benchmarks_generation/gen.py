import openai
import json

def generate_benchmark_data(chunk_content, chunk_metadata, api_key):
    client = openai.OpenAI(api_key=api_key)

    system_prompt = """Вы — эксперт по тестированию поисковых систем (QA/RAG). 
Ваша задача: прочитать фрагмент технической документации и составить на его основе 1-2 пары "Вопрос-Ответ".

Требования к вопросам:
1. Вопрос должен быть конкретным и техническим (например, про характеристики, команды или настройки).
2. На вопрос можно дать ПОЛНЫЙ ответ, используя ТОЛЬКО этот фрагмент текста.
3. Избегайте общих фраз типа "О чем этот текст?".

Требования к ответам:
1. Ответ должен быть точным и лаконичным.
2. Если в тексте есть таблица, используйте данные из неё.

Выходной формат: строго JSON список объектов с полями 'question' и 'ground_truth'."""

    user_prompt = f"""КОНТЕКСТ (Раздел: {chunk_metadata.get('context')}, Стр: {chunk_metadata.get('page')}):
{chunk_content}

Сгенерируй вопросы для бенчмарка:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" }, # Гарантирует получение JSON
            temperature=0.5
        )
        
        # Получаем результат
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Ошибка API: {e}")
        return None