from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv
load_dotenv()
DEVICE=os.getenv("DEVICE", "cuda")

def load_reranker(model_name="BAAI/bge-reranker-v2-m3", device=DEVICE):
    return CrossEncoder(model_name, max_length=512, device=device)