from sentence_transformers import SentenceTransformer
import torch
import os
from dotenv import load_dotenv
load_dotenv()
DEVICE=os.getenv("DEVICE", "cuda")
from logger_config import setup_logger
logger = setup_logger(__name__)

models = ["e5", "giga", "user2"]

def load_embedder(model_type: str, device=DEVICE):
    if model_type=="e5": 
        return SentenceTransformer("intfloat/multilingual-e5-large", device=device)

    elif model_type=="user2":
        return  SentenceTransformer("deepvk/USER2-base", device=device)
        
    elif model_type=="giga":
        return SentenceTransformer(
            "ai-sage/Giga-Embeddings-instruct", 
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
            },
            config_kwargs={"trust_remote_code": True},
            device=device
        )
        
    else:
        logger.error(f"Модель {model_type} не поддерживается")
        exit()
        