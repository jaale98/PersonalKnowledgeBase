from openai import OpenAI
from .config import OPENAI_API_KEY, EMBEDDING_MODEL, VECTOR_DIM

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_embedding(text: str) -> list[float]:
    """Generate a vector embedding for a text string."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.strip()
    )

    vector = response.data[0].embedding
    if len(vector) != VECTOR_DIM:
        raise ValueError(f"Expected {VECTOR_DIM} dims, got {len(vector)}")

    return vector