# scripts/build_chunks.py
import json
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

INPUT_PATH = r"C:\Users\zbook\OneDrive\Desktop\Language Assistant\data\processed\DictionaryToText.json"
OUTPUT_PATH = r"C:\Users\zbook\OneDrive\Desktop\Language Assistant\data\processed\chunked_documents.pkl"

def load_ocr_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def split_entries(text_data):
    """Split raw OCR page text into dictionary entries by word."""
    entries = []
    for page in text_data:
        text = page["text"]
        words = text.split("\n\n")  # crude per-entry split
        for w in words:
            if w.strip() and len(w.split()) > 1:
                entries.append({
                    "text": w.strip(),
                    "page": page["page"]
                })
    return entries

def create_documents(entries):
    docs = []
    for entry in entries:
        docs.append(Document(
            page_content=entry["text"],
            metadata={"source": f"page_{entry['page']}"}
        ))
    return docs

def chunk_documents(documents, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

if __name__ == "__main__":
    print("📚 Loading OCR JSON...")
    text_data = load_ocr_data(INPUT_PATH)

    print("🔍 Extracting dictionary entries...")
    entries = split_entries(text_data)
    print(f"📖 Total entries found: {len(entries)}")

    print("📄 Creating LangChain Document objects...")
    docs = create_documents(entries)

    print("✂️ Chunking documents...")
    chunked_docs = chunk_documents(docs)

    print(f"✅ Chunked into {len(chunked_docs)} total documents.")

    import pickle
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(chunked_docs, f)

    print(f"💾 Saved chunked documents to: {OUTPUT_PATH}")
