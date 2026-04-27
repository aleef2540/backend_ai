import os
from openai import OpenAI
from qdrant_client import QdrantClient

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COURSE_COLLECTION", "course_scripts_hybrid")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def embed_text_openai(text: str):
    res = openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text
    )
    return res.data[0].embedding


async def search_courses_from_qdrant(query: str, limit: int = 5):
    vector = embed_text_openai(query)

    hits = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=limit,
        with_payload=True,
    )

    results = []

    for hit in hits.points:
        payload = hit.payload or {}

        results.append({
            "score": hit.score,
            "course_no": payload.get("course_id") or payload.get("course_no"),
            "course_name": payload.get("course_name") or payload.get("course"),
            "summary": payload.get("summary") or payload.get("retrieval_text"),
            "payload": payload,
        })

    return results