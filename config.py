"""
Configuration settings for the RAG pipeline
"""

# Qdrant Settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_USE_MEMORY = True  # Set to False for persistent storage
COLLECTION_NAME = "documents"

# Ollama Settings
OLLAMA_MODEL = "tinyllama"
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Document Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# RAG Settings
TOP_K_RESULTS = 3
TEMPERATURE = 0.8
