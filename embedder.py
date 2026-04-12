from sentence_transformers import SentenceTransformer

# Multilingual model, supports Chinese + English, ~90MB download on first run
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading embedding model (first time only)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
