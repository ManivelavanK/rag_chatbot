# database.py — File-based storage (replaces PostgreSQL)
# Stores users, conversations, and messages in data/store.json
# fetch_all_chunks / fetch_chunks_by_ids read from ChromaDB directly

import json
from datetime import datetime
from pathlib import Path

import chromadb
from config import CHROMA_PATH, COLLECTION_NAME

STORE_PATH = Path("data/store.json")

def _load() -> dict:
    if STORE_PATH.exists():
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"users": {}, "conversations": {}, "messages": {}}

def _save(store: dict):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, default=str)

def init_db():
    if not STORE_PATH.exists():
        _save({"users": {}, "conversations": {}, "messages": {}})

def get_session():
    return _JsonSession()

class _JsonSession:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, *_):
        pass  # no-op, each operation saves immediately


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


def fetch_all_chunks(session) -> list[dict]:
    """Return all chunks from ChromaDB for BM25 index building."""
    col = _get_collection()
    results = col.get(include=["documents", "metadatas"])
    chunks = []
    for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        chunks.append({
            "chunk_id":   cid,
            "chunk_text": doc,
            "source":     meta.get("source", ""),
            "section":    meta.get("section", "General"),
            "doc_type":   meta.get("doc_type", "general"),
        })
    return chunks


def fetch_chunks_by_ids(session, chunk_ids: list[str]) -> dict[str, dict]:
    """Fetch specific chunks from ChromaDB by their IDs."""
    if not chunk_ids:
        return {}
    col = _get_collection()
    results = col.get(ids=chunk_ids, include=["documents", "metadatas"])
    return {
        cid: {
            "chunk_text": doc,
            "source":     meta.get("source", ""),
            "section":    meta.get("section", "General"),
            "doc_type":   meta.get("doc_type", "general"),
            "chunk_index": meta.get("chunk_index", 0),
        }
        for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
    }
