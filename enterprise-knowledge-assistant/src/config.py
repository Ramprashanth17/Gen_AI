"""Configuration Settings for Enterprise Knowledge Assistant"""

# Paths
DATA_DIR = "data"
DOCUMENTS_DIR = "data/documents"
VECTOR_DB_DIR = "data/vector_db"

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 32

# Chunking Settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# Vector Database
COLLECTION_NAME = "sap_knowledge"

#LLM Settings
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 250
TOP_K_RESULTS = 5

# System prompt for RAG
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context from company documents.

Rules:
- Answer based ONLY on the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer this question."
- Always cite which source document your answer comes from
- Be concise and clear (2-3 paragraphs maximum)
- Use professional business language"""
