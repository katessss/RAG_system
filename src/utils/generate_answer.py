def generate_qwen_answer(query: str, context: str, model, tokenizer, max_context_length=6000):
    """
    Принимает вопрос и найденный контекст, возвращает ответ от qwen
    """

    if len(context) > max_context_length * 3:
        context = context[:max_context_length * 3] + "...\n[Текст обрезан]"

    messages = [
        {"role": "system", "content": "Ты инженер технической поддержки ViPNet. Отвечай только на основе предоставленного текста."},
        {"role": "user", "content": f"Контекст: {context}\n\nВопрос: {query}"}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response