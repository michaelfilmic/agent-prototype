import os
import chromadb
from chromadb.config import Settings

from embedder import embed_texts, embed_query

DB_PATH       = "C:/Users/micha/Desktop/agent_prototype/knowledge_db"
DOCS_PATH     = "C:/Users/micha/Desktop/agent_prototype/docs"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
RELEVANCE_THRESHOLD = 1.0  # distance below this = relevant (lower = more similar)


# ── ChromaDB ──────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name="knowledge")


# ── Chunking ──────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Indexing ──────────────────────────────────────────────────

def index_documents():
    """Load all .md and .txt files from docs/ and index into ChromaDB."""
    os.makedirs(DOCS_PATH, exist_ok=True)
    collection = get_collection()

    files = [f for f in os.listdir(DOCS_PATH) if f.endswith((".md", ".txt"))]
    if not files:
        print(f"No .md or .txt files found in {DOCS_PATH}")
        return

    all_chunks, all_ids, all_metadata = [], [], []

    for filename in files:
        filepath = os.path.join(DOCS_PATH, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}-chunk-{i}")
            all_metadata.append({"source": filename, "chunk": i})

        print(f"  Indexed: {filename} ({len(chunks)} chunks)")

    embeddings = embed_texts(all_chunks)
    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata,
    )
    print(f"\nDone. Total chunks indexed: {len(all_chunks)}")


# ── Querying ──────────────────────────────────────────────────

def query_knowledge(question: str, top_k: int = 3) -> list[dict]:
    """Return the most relevant chunks. Returns [] if nothing relevant found."""
    collection = get_collection()

    # Return empty if collection has no documents
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]
        if distance < RELEVANCE_THRESHOLD:
            chunks.append({
                "content":  results["documents"][0][i],
                "source":   results["metadatas"][0][i]["source"],
                "distance": distance,
            })

    return chunks


def format_for_llm(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    sections = [
        f"[{i+1}] From: {c['source']}\n{c['content']}"
        for i, c in enumerate(chunks)
    ]
    return "\n\n".join(sections)
