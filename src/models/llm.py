from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv
load_dotenv()
DEVICE=os.getenv("DEVICE", "cuda")

def load_qwen_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct", device=DEVICE):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device
    )
    return model, tokenizer