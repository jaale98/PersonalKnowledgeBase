import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL   = os.getenv("DATABASE_URL", "postgresql+psycopg://pkb:pkb@localhost:5432/pkb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1536"))
ALLOW_CORS_ALL = os.getenv("ALLOW_CORS_ALL", "true").lower() == "true"
ENABLE_FTS = os.getenv("ENABLE_FTS", "false").lower() == "true"
ANN_LISTS = int(os.getenv("ANN_LISTS", "100"))
RESULT_LIMIT_DEFAULT = int(os.getenv("RESULT_LIMIT_DEFAULT", "20"))
ANN_PROBES = int(os.getenv("ANN_PROBES", "10"))
ENABLE_FTS = os.getenv("ENABLE_FTS", "true").lower() in {"1","true","yes","on"}
FUSION_ALPHA = float(os.getenv("FUSION_ALPHA", "0.70"))