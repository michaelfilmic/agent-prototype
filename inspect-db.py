import chromadb
from chromadb.config import Settings

DB_PATH = "C:/Users/micha/Desktop/agent_prototype/knowledge_db"

client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_or_create_collection("knowledge")

print(f"Total chunks: {collection.count()}")
print("=" * 60)

# Show all chunks with source and content preview
results = collection.get(include=["documents", "metadatas"])

sources = {}
for doc, meta in zip(results["documents"], results["metadatas"]):
    src = meta["source"]
    sources.setdefault(src, []).append(doc)

for source, chunks in sources.items():
    print(f"\n📄 {source} ({len(chunks)} chunks)")
    print("-" * 40)
    for i, chunk in enumerate(chunks):
        preview = chunk[:100].replace("\n", " ")
        print(f"  [{i}] {preview}...")
