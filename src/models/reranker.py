from sentence_transformers import CrossEncoder

def load_reranker(model_name="BAAI/bge-reranker-v2-m3", DEVICE='cuda:2'):
    return CrossEncoder(model_name, max_length=512, device=DEVICE)