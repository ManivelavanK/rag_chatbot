# ingest.py — ingestion pipeline (PostgreSQL + ChromaDB)
#
# What changed vs the old version:
#   - Chunks are now inserted into PostgreSQL (chunks table) before embedding.
#   - ChromaDB still stores the embedding + lightweight metadata.
#   - Duplicate detection uses PostgreSQL chunk_ids (single source of truth).
#   - _doc_type() and _sliding_chunks() are completely unchanged.
#   - ingest_file() accepts a db session and document_id in addition to the
#     ChromaDB collection, so both stores are written atomically per file.

import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PATH, COLLECTION_NAME, DATA_DIR,
    EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
)
from database import (
    get_session,
    get_or_create_document,
    get_existing_chunk_ids,
    Chunk as DBChunk,
    init_db,
)
from logger import get_logger

log = get_logger("ingest")


# ── helpers (unchanged) ───────────────────────────────────────────────────────

def _doc_type(filename: str) -> str:
    name = filename.lower()
    if "leave"       in name: return "leave_rule"
    if "recruitment" in name: return "recruitment_rule"
    if "conduct"     in name: return "conduct_rule"
    if "moe"         in name: return "moe_circular"
    if "policy"      in name: return "nit_policy"
    return "general"


def _sliding_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + size]))
        start += size - overlap
    return chunks


# ── per-file ingestion ────────────────────────────────────────────────────────

def ingest_file(
    filepath: str,
    collection,
    model: SentenceTransformer,
    existing_ids: set,          # chunk_ids already in PostgreSQL
):
    filename = os.path.basename(filepath)
    doc_type = _doc_type(filename)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    added = skipped = 0

    # All inserts for one file share one transaction — rolls back cleanly on error
    with get_session() as session:
        document_id = get_or_create_document(session, filename, doc_type)

        for item in data:
            section     = item.get("section", "General")
            raw_content = item.get("content", "").strip()
            base_id     = item.get("chunk_id", f"{filename}_{section}")

            if not raw_content:
                continue

            sub_chunks = _sliding_chunks(raw_content, CHUNK_SIZE, CHUNK_OVERLAP)

            for idx, chunk_text in enumerate(sub_chunks):
                chunk_id = f"{base_id}_{idx}" if len(sub_chunks) > 1 else base_id

                if chunk_id in existing_ids:
                    skipped += 1
                    continue

                # ── 1. Insert into PostgreSQL ─────────────────────────────────
                db_chunk = DBChunk(
                    chunk_id     = chunk_id,
                    document_id  = document_id,
                    source       = filename,
                    section      = section,
                    chunk_text   = chunk_text,
                    chunk_index  = idx,
                    total_chunks = len(sub_chunks),
                )
                session.add(db_chunk)

                # ── 2. Embed + insert into ChromaDB ───────────────────────────
                embed_text = f"Section: {section}\n\n{chunk_text}"
                embedding  = model.encode(embed_text).tolist()

                collection.add(
                    ids        = [chunk_id],
                    documents  = [chunk_text],
                    embeddings = [embedding],
                    metadatas  = [{
                        "source":       filename,
                        "section":      section,
                        "doc_type":     doc_type,
                        "chunk_index":  idx,
                        "total_chunks": len(sub_chunks),
                    }],
                )

                existing_ids.add(chunk_id)
                added += 1

        # session.commit() is called automatically by get_session()'s __exit__

    log.info(f"{filename}: {added} added, {skipped} skipped")
    return added


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("Starting ingestion pipeline (PostgreSQL + ChromaDB)")

    # Create tables if they don't exist yet
    init_db()

    model      = SentenceTransformer(EMBED_MODEL)
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Use PostgreSQL as the single source of truth for existing chunk_ids
    with get_session() as session:
        existing = get_existing_chunk_ids(session)
    log.info(f"Existing chunks in PostgreSQL: {len(existing)}")

    total = 0
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".json"):
            total += ingest_file(
                os.path.join(DATA_DIR, fname),
                collection, model, existing,
            )

    log.info(f"Ingestion complete. Total new chunks added: {total}")
    log.info(f"ChromaDB collection size: {collection.count()}")


if __name__ == "__main__":
    main()
