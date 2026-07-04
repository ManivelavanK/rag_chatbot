# retriever.py — hybrid retrieval + reranking engine  (PostgreSQL-backed)
#
# What changed vs the old version:
#   - _build_bm25_index() now calls fetch_all_chunks(session) from PostgreSQL
#     instead of reading from ChromaDB's get(). ChromaDB is now used ONLY for
#     vector similarity search, not as a document store.
#   - _vector_search() calls ChromaDB for chunk_ids + distances, then enriches
#     each hit with the full text + metadata from PostgreSQL in a single IN query.
#   - Everything else — query expansion, BM25 scoring, reranking, public API —
#     is completely unchanged.
#
# Pipeline per query:
#   query → embed → ChromaDB (ids + distances)
#          → fetch_chunks_by_ids(session, ids) from PostgreSQL
#          → BM25 (in-memory, built from PostgreSQL on startup)
#          → merge → cross-encoder rerank → top-N chunks → LLM

from __future__ import annotations
import re
import time
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from ollama import chat

from config import (
    CHROMA_PATH, COLLECTION_NAME,
    EMBED_MODEL, RERANKER_MODEL, LLM_MODEL,
    VECTOR_TOP_K, BM25_TOP_K, FINAL_TOP_K,
    DISTANCE_CUTOFF, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
)
from database import get_session, fetch_all_chunks, fetch_chunks_by_ids
from prompts import build_query_expansion_prompt
from logger import get_logger

log = get_logger("retriever")


# ── data model (unchanged) ────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:     str
    content:      str
    source:       str
    section:      str
    doc_type:     str
    vector_dist:  float = 1.0
    bm25_score:   float = 0.0
    rerank_score: float = 0.0
    confidence:   str   = "Low"


# ── retriever class ───────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(self):
        log.info("Loading embedding model …")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        log.info("Loading cross-encoder reranker …")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        log.info("Connecting to ChromaDB …")
        self.client     = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)

        self._bm25:       BM25Okapi | None = None
        self._all_chunks: list[Chunk]      = []
        self._build_bm25_index()

    # ── BM25 index — now built from PostgreSQL ────────────────────────────────

    def _build_bm25_index(self):
        log.info("Building BM25 index from PostgreSQL …")
        with get_session() as session:
            rows = fetch_all_chunks(session)

        self._all_chunks = [
            Chunk(
                chunk_id = r["chunk_id"],
                content  = r["chunk_text"],
                source   = r["source"],
                section  = r["section"],
                doc_type = r["doc_type"],
            )
            for r in rows
        ]

        tokenized  = [c.content.lower().split() for c in self._all_chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        log.info(f"BM25 index built: {len(self._all_chunks)} chunks from PostgreSQL")

    def rebuild_index(self):
        """Call after re-ingestion to refresh the in-memory BM25 index."""
        self._build_bm25_index()

    # ── query expansion (unchanged) ───────────────────────────────────────────

    def _expand_query(self, question: str) -> list[str]:
        words = question.strip().split()
        if len(words) <= 2:
            log.info(f"Query expansion skipped (short query): {question}")
            return [question]
        try:
            prompt   = build_query_expansion_prompt(question)
            response = chat(
                model    = LLM_MODEL,
                messages = [{"role": "user", "content": prompt}],
            )
            # Support both ollama response formats
            if hasattr(response, "message"):
                raw = response.message.content.strip()
            elif isinstance(response, dict):
                raw = response["message"]["content"].strip()
            else:
                raw = str(response).strip()

            lines = raw.splitlines()

            # Keep only lines that look like real questions or statements.
            # Filter out: empty lines, lines starting with numbers/bullets,
            # lines that are too short (<10 chars) or too long (>200 chars),
            # and lines identical to the original question.
            variants = []
            for line in lines:
                line = line.strip()
                # Strip leading numbering like "1." "1)" "- " "* "
                line = re.sub(r'^[\d]+[.)\s]+', '', line).strip()
                line = re.sub(r'^[-*•]\s*', '', line).strip()
                if not line:
                    continue
                if len(line) < 10 or len(line) > 200:
                    continue
                if line.lower() == question.lower():
                    continue
                variants.append(line)
                if len(variants) == 2:   # we only need 2
                    break

            queries = [question] + variants
            log.info(f"Query expansion: {queries}")
            return queries
        except Exception as e:
            log.warning(f"Query expansion failed ({e}), using original only")
            return [question]

    # ── vector search — ChromaDB for ids, PostgreSQL for text ─────────────────

    def _vector_search(self, queries: list[str]) -> dict[str, Chunk]:
        # Step 1: collect (chunk_id, distance) pairs from ChromaDB
        id_dist: dict[str, float] = {}
        for q in queries:
            emb     = self.embedder.encode(f"Section: \n\n{q}").tolist()
            results = self.collection.query(
                query_embeddings=[emb],
                n_results=VECTOR_TOP_K,
                include=["distances"],          # no need to fetch documents from Chroma
            )
            for cid, dist in zip(results["ids"][0], results["distances"][0]):
                if dist > DISTANCE_CUTOFF:
                    continue
                # Keep the best (lowest) distance across multiple query variants
                if cid not in id_dist or dist < id_dist[cid]:
                    id_dist[cid] = dist

        if not id_dist:
            return {}

        # Step 2: fetch full chunk text + metadata from PostgreSQL in one query
        with get_session() as session:
            pg_rows = fetch_chunks_by_ids(session, list(id_dist.keys()))

        seen: dict[str, Chunk] = {}
        for cid, dist in id_dist.items():
            row = pg_rows.get(cid)
            if row is None:
                log.warning(f"chunk_id {cid!r} found in ChromaDB but missing in PostgreSQL")
                continue
            seen[cid] = Chunk(
                chunk_id    = cid,
                content     = row["chunk_text"],
                source      = row["source"],
                section     = row["section"],
                doc_type    = row["doc_type"],
                vector_dist = dist,
            )
        return seen

    # ── BM25 search (unchanged logic) ─────────────────────────────────────────

    def _bm25_search(self, question: str) -> dict[str, Chunk]:
        if self._bm25 is None or not self._all_chunks:
            log.warning("BM25 search skipped because no chunks are loaded from PostgreSQL")
            return {}

        tokens    = question.lower().split()
        scores    = self._bm25.get_scores(tokens)
        top_i     = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_TOP_K]
        max_score = max((scores[i] for i in top_i), default=1.0)

        seen: dict[str, Chunk] = {}
        for i in top_i:
            if scores[i] <= 0:
                continue
            c = self._all_chunks[i]
            seen[c.chunk_id] = Chunk(
                chunk_id   = c.chunk_id,
                content    = c.content,
                source     = c.source,
                section    = c.section,
                doc_type   = c.doc_type,
                bm25_score = scores[i] / max_score,
            )
        return seen

    # ── reranking ─────────────────────────────────────────────────────────────

    def _rerank(self, question: str, candidates: list[Chunk]) -> list[Chunk]:
        if not candidates:
            return []
        pairs  = [(question, c.content) for c in candidates]
        scores = self.reranker.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)
            if score >= CONFIDENCE_HIGH:
                chunk.confidence = "High"
            elif score >= CONFIDENCE_MEDIUM:
                chunk.confidence = "Medium"
            else:
                chunk.confidence = "Low"

        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:FINAL_TOP_K]

    # ── sub-question splitting ────────────────────────────────────────────────

    def _split_subquestions(self, question: str) -> list[str]:
        """
        Detect compound questions joined by 'and' and split them so each
        sub-question gets its own vector + BM25 search pass.
        Returns [question] unchanged if no clear split point found.
        """
        parts = re.split(r',?\s+and\s+', question, flags=re.IGNORECASE)
        if len(parts) == 2 and len(parts[0]) > 15 and len(parts[1]) > 15:
            return [p.strip().rstrip('?') + '?' for p in parts]
        return [question]

    # ── public API ─────────────────────────────────────────────────────────────

    def retrieve(self, question: str, diagnostics: bool = False) -> tuple[list[Chunk], dict]:
        t0 = time.time()

        subquestions = self._split_subquestions(question)
        all_vec_hits:  dict[str, Chunk] = {}
        all_bm25_hits: dict[str, Chunk] = {}
        sq_candidates: list[tuple[str, list[Chunk]]] = []

        for sq in subquestions:
            queries   = self._expand_query(sq)
            vec_hits  = self._vector_search(queries)
            bm25_hits = self._bm25_search(sq)

            for cid, chunk in vec_hits.items():
                if cid not in all_vec_hits or chunk.vector_dist < all_vec_hits[cid].vector_dist:
                    all_vec_hits[cid] = chunk
            for cid, chunk in bm25_hits.items():
                if cid not in all_bm25_hits or chunk.bm25_score > all_bm25_hits[cid].bm25_score:
                    all_bm25_hits[cid] = chunk

            merged_sq: dict[str, Chunk] = {**bm25_hits, **vec_hits}
            for cid in merged_sq:
                if cid in bm25_hits:
                    merged_sq[cid].bm25_score = bm25_hits[cid].bm25_score
            sq_candidates.append((sq, list(merged_sq.values())))

        # Rerank per sub-question; guarantee at least 1 slot per sub-question
        # using vector distance as fallback when all reranker scores are negative
        slots_per_sq = max(1, FINAL_TOP_K // len(subquestions))
        final_chunks: dict[str, Chunk] = {}

        for sq, candidates in sq_candidates:
            ranked = self._rerank(sq, candidates)
            top = ranked[:slots_per_sq]

            # If every top chunk has a very negative reranker score (<-3),
            # fall back to the best vector-distance hits for this sub-question
            if top and all(c.rerank_score < -3.0 for c in top):
                vec_fallback = sorted(
                    [c for c in candidates if c.vector_dist < DISTANCE_CUTOFF],
                    key=lambda c: c.vector_dist
                )[:slots_per_sq]
                if vec_fallback:
                    top = vec_fallback

            for c in top:
                if c.chunk_id not in final_chunks:
                    final_chunks[c.chunk_id] = c

        # Fill remaining FINAL_TOP_K slots with global rerank results
        all_merged: dict[str, Chunk] = {**all_bm25_hits, **all_vec_hits}
        for cid in all_merged:
            if cid in all_bm25_hits:
                all_merged[cid].bm25_score = all_bm25_hits[cid].bm25_score

        global_ranked = self._rerank(question, list(all_merged.values()))
        for c in global_ranked:
            if c.chunk_id not in final_chunks:
                final_chunks[c.chunk_id] = c
            if len(final_chunks) >= FINAL_TOP_K:
                break

        final   = list(final_chunks.values())[:FINAL_TOP_K]
        elapsed = time.time() - t0

        diag = {
            "queries":           subquestions,
            "vector_candidates": len(all_vec_hits),
            "bm25_candidates":   len(all_bm25_hits),
            "total_candidates":  len(all_merged),
            "final_chunks":      len(final),
            "retrieval_ms":      round(elapsed * 1000),
        }
        if diagnostics:
            log.info(f"Retrieval diagnostics: {diag}")

        return final, diag
