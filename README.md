# NIT Calicut Policy Assistant 🏛️

A RAG (Retrieval-Augmented Generation) based chatbot that helps NIT Calicut staff and faculty instantly find answers from official policy documents.

## What It Does

Instead of manually searching through lengthy PDFs, just ask a question and get the most relevant policy excerpts retrieved instantly.

**Covers:**
- Leave Rules (EL, CL, HPL, CCL, Restricted Holidays, On Duty)
- Recruitment Rules & Promotion (DPC, MACP)
- NIT Calicut Institutional Policies
- Ministry of Education (MoE) Directives
- Central Civil Services (CCS) Rules

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Embedding Model | all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Gemma3:1b via Ollama |
| Vector Store | ChromaDB |
| Database | PostgreSQL |
| Search | Hybrid (BM25 + Vector) |

## How It Works

1. Query comes in → expanded into variants using LLM
2. Vector search (ChromaDB) + BM25 keyword search run in parallel
3. Results merged and reranked using a cross-encoder
4. Top chunks returned as the answer

## Project Structure
├── app.py # Streamlit UI
├── retriever.py # Hybrid retrieval + reranking engine
├── ingest.py # Data ingestion into ChromaDB + PostgreSQL
├── database.py # PostgreSQL models and session management
├── prompts.py # Prompt templates
├── config.py # Centralized configuration
├── logger.py # Logging setup
├── history.py # Conversation history
├── style.css # UI styling
├── data/ # Source policy JSON files
├── embeddings/ # ChromaDB persistent storage
└── .env # Environment variables (not committed)


## Setup & Installation

### Prerequisites
- Python 3.10+
- PostgreSQL running locally
- Ollama installed and running

### 1. Clone the repo
git clone https://github.com/your-username/nit-calicut-policy-assistant.git
cd nit-calicut-policy-assistant

Copy
2. Install dependencies
pip install -r requirements.txt

Copy
3. Configure environment
Create a .env file:

DB_HOST=localhost
DB_PORT=5432
DB_NAME=nitcalicut
DB_USER=your_user
DB_PASSWORD=your_password

Copy
4. Pull the LLM model
ollama pull gemma3:1b

Copy
bash
5. Ingest documents
python ingest.py

Copy
bash
6. Run the app
streamlit run app.py

Copy
bash
Open http://localhost:8501 in your browser.


developed by
Manivelavan K
