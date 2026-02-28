from transformers import AutoModelForCausalLM, AutoTokenizer


def load_qwen_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct", DEVICE: "cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        device=DEVICE
    )