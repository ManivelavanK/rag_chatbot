-- schema.sql — NIT Calicut RAG  •  PostgreSQL schema
-- Run once:  psql -U postgres -d nit_rag -f schema.sql
-- (create the database first: CREATE DATABASE nit_rag;)

-- ── documents ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    document_id  SERIAL        PRIMARY KEY,
    filename     VARCHAR(255)  NOT NULL UNIQUE,   -- e.g. "NIT_policy.json"
    title        VARCHAR(512),
    doc_type     VARCHAR(64)   NOT NULL DEFAULT 'general',
    upload_date  TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ── chunks ────────────────────────────────────────────────────────────────────
-- chunk_id is the SAME string stored in ChromaDB — the join key between the two DBs
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     VARCHAR(512)  PRIMARY KEY,
    document_id  INT           NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    source       VARCHAR(255)  NOT NULL,
    section      VARCHAR(512)  NOT NULL DEFAULT 'General',
    page_number  INT,                             -- NULL when not available from JSON
    chunk_text   TEXT          NOT NULL,
    chunk_index  INT           NOT NULL DEFAULT 0,
    total_chunks INT           NOT NULL DEFAULT 1,
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ── indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source      ON chunks(source);
-- Full-text search index (used by BM25 fallback SQL path)
CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks USING gin(to_tsvector('english', chunk_text));

-- ── users ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL        PRIMARY KEY,
    username      VARCHAR(50)   NOT NULL UNIQUE,
    password_hash VARCHAR(255)  NOT NULL,
    created_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ── conversations ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id            SERIAL        PRIMARY KEY,
    user_id       INT           NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title         VARCHAR(255)  NOT NULL DEFAULT 'New Chat',
    created_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);

-- ── chat_messages ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_messages (
    id                SERIAL        PRIMARY KEY,
    conversation_id   INT           NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role              VARCHAR(20)   NOT NULL,
    content           TEXT          NOT NULL,
    retrieved_sources JSONB,
    confidence_score  FLOAT,
    diagnostics       JSONB,
    timestamp         TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat_messages(conversation_id);
