import openai
import json
import random
from pydantic import BaseModel, Field
from typing import List

api_key = ""
client = openai.OpenAI(api_key=api_key) # обошлось в 30 центов


class QuestionList(BaseModel):
    questions: List[str] = Field(description="1-2 конкретных технических вопросов по текст")

def generate_questions(chunk_content):
    system_prompt = """Вы — эксперт по тестированию поисковых систем (QA). 
    Ваша задача: прочитать фрагмент технической документации и составить на его основе 1-2 вопроса.

    Требования к вопросам:
    1. Вопрос должен быть таким, чтобы ответом на него был ИМЕННО этот фрагмент текста.
    2. Используйте конкретные названия моделей (HW1000), команд (iplir...) или цифры из таблиц.
    3. НИКАКИХ ответов присылать не нужно. Только вопросы.
    4. Избегайте общих вопросов типа "О чем этот раздел?"."""

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Создай вопросы к тексту:\n\n{chunk_content}"}
            ],
            response_format=QuestionList,
            temperature=0.5
        )
        
        return response.choices[0].message.parsed.questions
    except Exception as e:
        print(f"Ошибка: {e}")
        return []


if __name__=="__main__"
    file_path = "benchmarks_generation/exported_chunks.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
            
    print(f"Успешно загружено {len(data)} чанков из файла {file_path}")
        
    test_chunks = random.sample(data, 200)
    
    benchmark_tasks = []
    for chunk in test_chunks:
        qs = generate_questions(chunk['content'])
        print(f"{qs}\n\n")
        for q in qs:
            benchmark_tasks.append({
                "question": q,
                "expected_page": chunk["metadata"].get("page"),
                "expected_context": chunk["metadata"].get("context"),
                "original_content": chunk["content"]
            })
    
    with open("benchmarks_generation/benchmark_tasks.json", "w", encoding="utf-8") as f:
            json.dump(benchmark_tasks, f, ensure_ascii=False, indent=2)