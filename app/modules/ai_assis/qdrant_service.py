import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COURSE_COLLECTION", "course_objects")
QDRANT_COURSE_COLLECTION = os.getenv("QDRANT_COURSE_COLLECTION_EN", "en_course")

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


async def check_topic_exists_in_qdrant(
    topic: str,
    limit: int = 5,
    min_score: float = 0.35
):
    if not topic or not topic.strip():
        return [], None

    vector = embed_text_openai(topic.strip())

    hits = qdrant_client.query_points(
        collection_name=QDRANT_COURSE_COLLECTION,
        query=vector,
        limit=limit,
        with_payload=True,
    )

    if not hits.points:
        return [], None

    best_score = hits.points[0].score or 0 
    print( f"TOPIC CHECK | topic={topic} | score={best_score}", flush=True )

    matched_courses = []

    for hit in hits.points:
        course_name = hit.payload.get('course_name', 'ไม่มีข้อมูลชื่อหลักสูตร')
        payload = hit.payload or {}
        score = hit.score or 0

        print(f" - Found Course: {course_name}") 
        print(f" Score: {score}", flush=True)

        if score < min_score:
            continue

        course_id = (
            payload.get("course_id")
            or payload.get("ICourse_no")
            or payload.get("course_no")
            or payload.get("id")
        )

        course_name = (
            payload.get("course_name")
            or payload.get("title")
            or ""
        )

        if not course_id:
            continue

        matched_courses.append({
            "course_id": course_id,
            "course_name": course_name,
            "score": score,
            "payload": payload,
        })

    if not matched_courses:
        return [], None

    return matched_courses, matched_courses[0]["course_id"]

async def search_courses_from_qdrant(query: str, limit: int = 3, excluded_courses: list = []):
    vector = embed_text_openai(query)
    qdrant_client.create_payload_index(
    collection_name=QDRANT_COLLECTION,
    field_name="course_no",
    field_schema=PayloadSchemaType.KEYWORD
)
    # กรอง course_no ที่ต้องการออกจากการ search
    hits = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=limit,
        with_payload=True,
        query_filter={
            "must_not": [
                {
                    "key": "course_no",
                    "match": {
                        "any": excluded_courses
                    }
                }
            ]
        }
    )

    best_by_course = {}

    for hit in hits.points:
        payload = hit.payload or {}

        course_id = (
            payload.get("course_id")
            or payload.get("course_no")
        )

        if not course_id:
            continue

        # ถ้ายังไม่เคยมี หรือ score ดีกว่า → replace
        if course_id not in best_by_course or hit.score > best_by_course[course_id]["score"]:
            best_by_course[course_id] = {
                "score": hit.score,
                "course_no": course_id,
                "course_name": payload.get("course_name") or payload.get("course"),
                "summary": payload.get("summary") or payload.get("retrieval_text"),
                "payload": payload,
            }

    # sort ตาม score ใหม่
    results = sorted(
        best_by_course.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:limit]