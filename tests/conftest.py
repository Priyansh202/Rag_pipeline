import os

os.environ.pop("DATABASE_URL", None)
os.environ.pop("DATABASE_URL_PRIVATE", None)
os.environ.pop("PINECONE_API_KEY", None)
os.environ.setdefault("VECTOR_BACKEND", "pgvector")
os.environ.setdefault("LLM_PROVIDER", "none")
os.environ.setdefault("JWT_SECRET", "test-secret-must-be-at-least-32-bytes")
