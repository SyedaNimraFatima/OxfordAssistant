# scripts/embed_index.py

import os
import pickle
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Paths
CHUNKED_DOCS_PATH = r"C:\Users\zbook\OneDrive\Desktop\Language Assistant\data\processed\chunked_documents.pkl"
FAISS_INDEX_PATH = r"C:\Users\zbook\OneDrive\Desktop\Language Assistant\data\index.faiss"
PKL_DOCS_PATH = r"C:\Users\zbook\OneDrive\Desktop\Language Assistant\data\index.pkl"
METADATA_PATH = r"C:\Users\zbook\OneDrive\Desktop\Language Assistant\data\metadata.json"

def load_chunked_documents(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_metadata(docs, output_path):
    metadata_list = []
    for i, doc in enumerate(docs):
        meta = doc.metadata.copy()
        meta["doc_id"] = i
        meta["preview"] = doc.page_content[:100].replace("\n", " ")  # optional preview
        metadata_list.append(meta)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"📑 Metadata saved to: {output_path}")

if __name__ == "__main__":
    print("📦 Loading chunked documents...")
    docs = load_chunked_documents(CHUNKED_DOCS_PATH)
    print(f"📄 Loaded {len(docs)} documents.")

    print("🔍 Initializing HuggingFace embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("📚 Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embedding_model)

    print(f"💾 Saving FAISS index to {FAISS_INDEX_PATH}")
    vectorstore.save_local(os.path.splitext(FAISS_INDEX_PATH)[0])  # Remove .faiss before saving

    print(f"💾 Saving original documents to {PKL_DOCS_PATH}")
    with open(PKL_DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    print("🧾 Generating metadata file...")
    save_metadata(docs, METADATA_PATH)

    print("✅ Embedding + Indexing complete.")