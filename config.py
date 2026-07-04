# config.py — centralized configuration for the entire RAG pipeline

# PostgreSQL — actual credentials live in .env, loaded by database.py
# This constant is here only for documentation; database.py reads os.getenv() directly.
DB_ENV_FILE       = ".env"

CHROMA_PATH       = "./embeddings/chroma_db"
COLLECTION_NAME   = "nit_documents"
DATA_DIR          = "./data"
LOG_DIR           = "./logs"

EMBED_MODEL       = "all-MiniLM-L6-v2"
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL         = "gemma3:1b"

# Chunking
CHUNK_SIZE        = 300   # words
CHUNK_OVERLAP     = 50    # words

# Retrieval
VECTOR_TOP_K      = 10    # cast wide net before reranking
BM25_TOP_K        = 10    # cast wide net before reranking
FINAL_TOP_K       = 4     # top 4 after reranking sent to LLM
DISTANCE_CUTOFF   = 1.5   # permissive — let reranker do the filtering

# Confidence thresholds (cross-encoder scores)
CONFIDENCE_HIGH   = 7.0
CONFIDENCE_MEDIUM = 2.5

# Abbreviations that must never be hallucinated
KNOWN_ABBR = {
    "CCL":   "Child Care Leave",
    "WRIIL": "Work Related Illness and Injury Leave",
    "MACP":  "Modified Assured Career Progression",
    "EL":    "Earned Leave",
    "HPL":   "Half Pay Leave",
    "CL":    "Casual Leave",
    "RH":    "Restricted Holiday",
    "OD":    "On Duty",
    "DPC":   "Departmental Promotion Committee",
    "CCS":   "Central Civil Services",
    "NIT":   "National Institute of Technology",
    "NITSER":"National Institutes of Technology, Science Education and Research",
    "MoE":   "Ministry of Education",
    "REC":   "Regional Engineering College",
    "CAPF":  "Central Armed Police Forces",
    "CPC":   "Central Pay Commission",
    "DSS":   "Document Self Service",
}
