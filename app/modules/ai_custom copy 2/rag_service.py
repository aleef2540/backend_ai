import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


"""
ไฟล์ใหม่สำหรับ RAG โดยเฉพาะ

หน้าที่:
1. รับคำถามจาก user
2. ทำ embedding ด้วย OpenAI text-embedding-3-large
3. Search ใน Qdrant
4. Filter เฉพาะ course_no ที่ frontend ส่งมา
5. ส่ง context กลับไปให้ LLM ตอบ

แนะนำให้วางไฟล์นี้ที่:
app/modules/ai_custom/rag_service.py
"""


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ai_vdo1")

EMBED_MODEL = "text-embedding-3-large"


openai_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def embed_query(query: str) -> List[float]:
    query = str(query or "").strip()

    if not query:
        raise ValueError("query is empty")

    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
    )

    return response.data[0].embedding


def normalize_course_nos(course_nos: List[Any]) -> List[str]:
    result = []

    for item in course_nos or []:
        value = str(item).strip()
        if value:
            result.append(value)

    return result


def extract_points(query_result: Any) -> List[Any]:
    """
    รองรับ qdrant-client หลาย version
    บาง version return object.points
    บาง version return tuple/list
    """

    if hasattr(query_result, "points"):
        return query_result.points

    if isinstance(query_result, tuple):
        for part in query_result:
            if isinstance(part, list):
                return part

    if isinstance(query_result, list):
        return query_result

    return []


def normalize_point(item: Any) -> Optional[Dict[str, Any]]:
    """
    แปลงผลลัพธ์จาก Qdrant ให้เป็น dict กลาง
    """

    if isinstance(item, tuple):
        point = item[0]
        score = item[1] if len(item) > 1 else None
        payload = getattr(point, "payload", {}) or {}

        return {
            "score": score,
            "payload": payload,
        }

    payload = getattr(item, "payload", {}) or {}
    score = getattr(item, "score", None)

    return {
        "score": score,
        "payload": payload,
    }


def search_rag(
    user_message: str,
    course_nos: List[Any],
    limit: int = 5,
    score_threshold: float = 0.35,
) -> List[Dict[str, Any]]:

    course_no_values = normalize_course_nos(course_nos)

    print("\n========== RAG DEBUG START ==========", flush=True)
    print("[RAG] user_message =", repr(user_message), flush=True)
    print("[RAG] raw course_nos =", course_nos, flush=True)
    print("[RAG] normalized course_no_values =", course_no_values, flush=True)
    print("[RAG] collection =", QDRANT_COLLECTION, flush=True)
    print("[RAG] limit =", limit, flush=True)
    print("[RAG] score_threshold =", score_threshold, flush=True)

    if not course_no_values:
        print("[RAG] no course_no_values -> return []", flush=True)
        print("========== RAG DEBUG END ==========\n", flush=True)
        return []

    query_vector = embed_query(user_message)

    print("[RAG] embedding vector size =", len(query_vector), flush=True)
    print("[RAG] embedding first 5 =", query_vector[:5], flush=True)

    qdrant_filter = Filter(
        should=[
            FieldCondition(
                key="OCourse_no",
                match=MatchValue(value=str(course_no)),
            )
            for course_no in course_no_values
        ]
    )

    print("[RAG] filter =", qdrant_filter, flush=True)

    query_result = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=limit,
        with_payload=True,
    )

    raw_points = extract_points(query_result)

    print("[RAG] raw_points count =", len(raw_points), flush=True)

    results = []

    for index, item in enumerate(raw_points, start=1):
        normalized = normalize_point(item)

        if not normalized:
            print(f"[RAG] item {index}: normalize failed", flush=True)
            continue

        score = normalized["score"]
        payload = normalized["payload"] or {}

        # print(f"\n[RAG] result {index}", flush=True)
        print("  score =", score, flush=True)
        print("  OCourse_no =", payload.get("OCourse_no"), flush=True)
        print("  course =", payload.get("course"), flush=True)
        print("  vdo_id =", payload.get("vdo_id"), flush=True)
        print("  vdo_name =", payload.get("vdo_name"), flush=True)
        print("  youtube_id =", payload.get("youtube_id"), flush=True)
        print("  chunk_index =", payload.get("chunk_index"), flush=True)
        print("  text_preview =", repr((payload.get("text") or "")[:300]), flush=True)

        if score is not None and score < score_threshold:
            print("  SKIP: score below threshold", flush=True)
            continue

        results.append({
            "score": score,
            "course_no": payload.get("OCourse_no"),
            "course": payload.get("course"),
            "vdo_id": payload.get("vdo_id"),
            "vdo_name": payload.get("vdo_name"),
            "youtube_id": payload.get("youtube_id"),
            "chunk_index": payload.get("chunk_index"),
            "text": payload.get("text") or "",
            "payload": payload,
        })

    print("\n[RAG] final results count =", len(results), flush=True)
    print("========== RAG DEBUG END ==========\n", flush=True)

    return results


def build_rag_context(rag_results: List[Dict[str, Any]]) -> str:
    """
    รวมผลลัพธ์จาก Qdrant เป็น context สำหรับส่งให้ LLM
    """

    blocks = []

    for index, item in enumerate(rag_results, start=1):
        blocks.append(
            f"""
[แหล่งข้อมูล {index}]
score: {item.get("score")}
course_no: {item.get("course_no")}
course: {item.get("course")}
video: {item.get("vdo_name")}
youtube_id: {item.get("youtube_id")}
chunk_index: {item.get("chunk_index")}

เนื้อหา:
{item.get("text")}
""".strip()
        )

    return "\n\n".join(blocks)


def build_active_video_from_rag(rag_item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rag_item:
        return None

    youtube_id = str(rag_item.get("youtube_id") or "").strip()

    if not youtube_id:
        return None

    return {
        "video_part": rag_item.get("chunk_index"),
        "video_name": rag_item.get("vdo_name"),
        "video_id": youtube_id,
        "embed_url": f"https://www.youtube.com/embed/{youtube_id}",
    }


async def answer_by_rag(
    user_message: str,
    course_nos: List[Any],
    reply_func,
    state=None,
    limit: int = 5,
    score_threshold: float = 0.20,
) -> Tuple[str, Optional[Dict[str, Any]], List[Dict[str, Any]]]:

    try:
        rag_results = search_rag(
            user_message=user_message,
            course_nos=course_nos,
            limit=limit,
            score_threshold=score_threshold,
        )

    except Exception as e:
        print("[RAG ERROR]", type(e), repr(e), flush=True)

        reply = "ระบบค้นหาเนื้อหาขัดข้องชั่วคราวครับ"

        if state is not None:
            state.mode = "error"
            state.last_answer = reply
            state.last_intent = "rag_exception"
            state.last_answer_type = "rag_exception"

        return reply, None, []

    if not rag_results:
        reply = "ผมยังไม่พบเนื้อหาที่เกี่ยวข้องในหลักสูตรที่เลือกไว้ครับ"

        if state is not None:
            state.mode = "discover"
            state.active_course_no = None
            state.last_answer = reply
            state.last_intent = "rag_not_found"
            state.last_answer_type = "rag_not_found"

        return reply, None, []

    best = rag_results[0]
    topic = best.get("vdo_name") or best.get("course") or "เนื้อหาที่เกี่ยวข้อง"
    rag_context = build_rag_context(rag_results)

    rag_message = f"""
คำถามผู้เรียน:
{user_message}

คำสั่ง:
- ตอบจากเนื้อหาใน RAG_CONTEXT เท่านั้น
- ถ้าไม่มีข้อมูลพอ ให้บอกว่าเนื้อหาไม่เพียงพอ
- ตอบภาษาไทย สุภาพ กระชับ เข้าใจง่าย
- ห้ามแต่งข้อมูลนอกเหนือจาก context

RAG_CONTEXT:
{rag_context}
""".strip()

    reply = await reply_func(
        rag_message,
        topic,
        rag_context,
    )

    active_video = build_active_video_from_rag(best)

    if state is not None:
        state.mode = "learning"
        state.topic = topic
        state.active_course_no = best.get("course_no")
        state.last_answer = reply
        state.last_intent = "rag_answer"
        state.last_answer_type = "rag_answered"

    return reply, active_video, rag_results
