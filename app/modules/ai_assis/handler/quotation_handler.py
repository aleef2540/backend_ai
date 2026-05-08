import os
import json
import re
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

from app.modules.ai_assis.service import (
    build_conversation_context,
    _stream_text_response,
)
from app.core.database import run_query_bridge

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

REQUIRED_QUOTATION_FIELDS = [
    "course_name",
    "contact_name",
    "company_name",
    "position",
    "phone",
    "email",
    "additional_requirements",
]

FIELD_LABELS = {
    "course_name": "ชื่อหลักสูตร",
    "contact_name": "ชื่อผู้ติดต่อ",
    "company_name": "บริษัท",
    "position": "ตำแหน่ง",
    "phone": "เบอร์โทร",
    "email": "email",
    "additional_requirements": "ความต้องการเพิ่มเติมของลูกค้า เช่น มีจำนวนผู้เข้าอบรมกี่คน สถานที่ ช่วงเวลา",
}

async def insert_quotation_request(state):
    quotation_context = normalize_quotation_context(
        getattr(state, "quotation_context", {}) or {}
    )

    chat_id = (
        getattr(state, "chat_id", "")
        or quotation_context.get("chat_id", "")
    )

    sql = """
        INSERT INTO ai_quotation_requests (
            chat_id,
            course_id,
            course_name,
            contact_name,
            company_name,
            position,
            phone,
            email,
            additional_requirements,
            quotation_context_json,
            status,
            source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = [
        chat_id,
        quotation_context.get("course_id") or None,
        quotation_context.get("course_name", ""),
        quotation_context.get("contact_name", ""),
        quotation_context.get("company_name", ""),
        quotation_context.get("position", ""),
        quotation_context.get("phone", ""),
        quotation_context.get("email", ""),
        quotation_context.get("additional_requirements", ""),
        json.dumps(quotation_context, ensure_ascii=False),
        "new",
        "ai_assistant",
    ]

    result = run_query_bridge(sql, params)

    return result

def embed_text_openai(text: str):
    res = openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text
    )
    return res.data[0].embedding


async def check_topic_exists_in_qdrant(topic: str, limit: int = 5, min_score: float = 0.35):
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

    matched_courses = []

    for hit in hits.points:
        payload = hit.payload or {}
        score = hit.score or 0

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
            or payload.get("course")
            or ""
        )

        if not course_id or not course_name:
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


async def search_courses_from_qdrant(query: str, limit: int = 3, excluded_courses: list = None):
    if excluded_courses is None:
        excluded_courses = []

    vector = embed_text_openai(query)

    try:
        qdrant_client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="course_no",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception:
        pass

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

        if course_id not in best_by_course or hit.score > best_by_course[course_id]["score"]:
            best_by_course[course_id] = {
                "score": hit.score,
                "course_no": course_id,
                "course_id": course_id,
                "course_name": payload.get("course_name") or payload.get("course"),
                "summary": payload.get("summary") or payload.get("retrieval_text"),
                "payload": payload,
            }

    results = sorted(
        best_by_course.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:limit]


def normalize_quotation_context(ctx):
    if not isinstance(ctx, dict):
        ctx = {}

    for field in REQUIRED_QUOTATION_FIELDS:
        ctx.setdefault(field, "")

    ctx.setdefault("course_id", "")
    ctx.setdefault("course_confirmed", False)
    ctx.setdefault("course_candidates", [])
    ctx.setdefault("course_confirmation_pending", False)

    return ctx


def get_missing_quotation_fields(ctx):
    missing = []

    for field in REQUIRED_QUOTATION_FIELDS:
        value = str(ctx.get(field) or "").strip()

        if field == "course_name":
            if not value or not ctx.get("course_confirmed"):
                missing.append(field)
        elif not value:
            missing.append(field)

    return missing


def is_valid_email(value):
    value = (value or "").strip()
    return bool(re.fullmatch(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        value
    ))


def normalize_phone(value):
    digits = re.sub(r"\D", "", value or "")

    if digits.startswith("66"):
        digits = "0" + digits[2:]

    return digits


def is_valid_phone(value):
    digits = normalize_phone(value)
    return len(digits) in [9, 10] and digits.startswith("0")


def looks_like_bad_value(value):
    value = str(value or "").strip()

    if not value:
        return True

    if len(value) < 2:
        return True

    if re.fullmatch(r"\d+", value):
        return True

    return False


def merge_valid_fields(quotation_context, updated_fields):
    old_course_name = quotation_context.get("course_name", "")

    for field in REQUIRED_QUOTATION_FIELDS:
        value = str(updated_fields.get(field) or "").strip()

        if not value:
            continue

        if field == "email":
            if is_valid_email(value):
                quotation_context[field] = value
            continue

        if field == "phone":
            if is_valid_phone(value):
                quotation_context[field] = normalize_phone(value)
            continue

        if not looks_like_bad_value(value):
            quotation_context[field] = value

    new_course_name = quotation_context.get("course_name", "")

    if new_course_name and new_course_name != old_course_name:
        quotation_context["course_id"] = ""
        quotation_context["course_confirmed"] = False
        quotation_context["course_candidates"] = []
        quotation_context["course_confirmation_pending"] = False

    return quotation_context


async def ai_analyze_quotation(user_message, state):
    quotation_context = normalize_quotation_context(
        getattr(state, "quotation_context", {}) or {}
    )

    conversation_context = build_conversation_context(state.conversation_history)
    missing_fields = get_missing_quotation_fields(quotation_context)

    system_prompt = """
คุณคือ AI วิเคราะห์ข้อมูลสำหรับขอใบเสนอราคาอบรม

หน้าที่:
- อ่านบทสนทนาและข้อความล่าสุดของผู้ใช้
- วิเคราะห์ว่าผู้ใช้ให้ข้อมูลอะไรมา
- เติมเฉพาะข้อมูลที่ผู้ใช้ให้มาชัดเจนเท่านั้น
- ห้ามเดาข้อมูลเอง
- ถ้าข้อความเป็นแค่ intent เช่น ขอใบเสนอราคา, อยากทราบราคา, สนใจอบรม ให้ไม่ต้องเติม field ใด
- ถ้า email ไม่ใช่อีเมลจริง ห้ามเติม email
- ถ้า phone ไม่ใช่เบอร์โทรจริง ห้ามเติม phone
- วิเคราะห์ด้วยว่า field ใดควรถามต่อจากข้อมูลที่ยังขาด
- เลือกถามเพียง 1 field เท่านั้น
- ตอบเป็น JSON เท่านั้น
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

ข้อมูลเดิม:
{json.dumps(quotation_context, ensure_ascii=False)}

ข้อมูลที่ยังขาด:
{json.dumps([FIELD_LABELS[f] for f in missing_fields], ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

field ที่ระบบต้องการ:
{json.dumps(FIELD_LABELS, ensure_ascii=False)}

ตอบ JSON เท่านั้น:
{{
  "updated_fields": {{
    "course_name": "",
    "contact_name": "",
    "company_name": "",
    "position": "",
    "phone": "",
    "email": "",
    "additional_requirements": ""
  }},
  "invalid_answer": false,
  "invalid_field": "",
  "invalid_reason": "",
  "next_field_to_ask": "",
  "reason": ""
}}
""".strip()

    result_text = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0,
    ):
        if item.get("type") == "chunk":
            result_text += item.get("text", "")
        elif item.get("type") == "done":
            result_text = item.get("content") or result_text

    try:
        analysis = json.loads(result_text)
    except Exception:
        analysis = {
            "updated_fields": {},
            "invalid_answer": False,
            "invalid_field": "",
            "invalid_reason": "json_parse_failed",
            "next_field_to_ask": "",
            "reason": "",
        }

    updated_fields = analysis.get("updated_fields") or {}
    quotation_context = merge_valid_fields(quotation_context, updated_fields)

    missing_fields = get_missing_quotation_fields(quotation_context)

    next_field = str(analysis.get("next_field_to_ask") or "").strip()
    if next_field not in missing_fields:
        next_field = missing_fields[0] if missing_fields else ""

    analysis["next_field_to_ask"] = next_field
    state.quotation_context = quotation_context

    return quotation_context, analysis


async def ai_analyze_course_confirmation(user_message, course_candidates, conversation_context):
    system_prompt = """
คุณคือ AI วิเคราะห์คำตอบยืนยันหลักสูตร

หน้าที่:
- วิเคราะห์ว่าผู้ใช้ยืนยันหลักสูตรที่เสนอหรือไม่
- ถ้าผู้ใช้ตอบว่าใช่, ถูกต้อง, เอาอันนี้, สนใจหลักสูตรนี้ ให้ decision เป็น yes
- ถ้าผู้ใช้ตอบว่าไม่ใช่, ไม่เอา, คนละหลักสูตร ให้ decision เป็น no
- ถ้าผู้ใช้เลือกจากรายการ ให้ใส่ selected_course_id
- ถ้าไม่ชัดเจน ให้ decision เป็น unclear
- ตอบเป็น JSON เท่านั้น
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

รายการหลักสูตรที่เสนอ:
{json.dumps(course_candidates, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ตอบ JSON เท่านั้น:
{{
  "decision": "yes",
  "selected_course_id": "",
  "reason": ""
}}
""".strip()

    result_text = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0,
    ):
        if item.get("type") == "chunk":
            result_text += item.get("text", "")
        elif item.get("type") == "done":
            result_text = item.get("content") or result_text

    try:
        return json.loads(result_text)
    except Exception:
        return {
            "decision": "unclear",
            "selected_course_id": "",
            "reason": "json_parse_failed",
        }


async def prepare_course_confirmation_if_needed(quotation_context):
    course_name = str(quotation_context.get("course_name") or "").strip()

    if not course_name:
        return quotation_context

    if quotation_context.get("course_confirmed"):
        return quotation_context

    if quotation_context.get("course_confirmation_pending"):
        return quotation_context

    matched_courses, best_course_id = await check_topic_exists_in_qdrant(
        topic=course_name,
        limit=5,
        min_score=0.35,
    )

    if not matched_courses:
        quotation_context["course_candidates"] = []
        quotation_context["course_confirmation_pending"] = False
        return quotation_context

    quotation_context["course_candidates"] = matched_courses
    quotation_context["course_confirmation_pending"] = True
    quotation_context["course_id"] = best_course_id or ""

    return quotation_context


async def ai_generate_course_confirmation_reply(user_message, state):
    quotation_context = normalize_quotation_context(
        getattr(state, "quotation_context", {}) or {}
    )

    course_candidates = quotation_context.get("course_candidates") or []
    best_course = course_candidates[0] if course_candidates else {}

    system_prompt = """
คุณคือ AI Assistant ชื่อ En-Assistant ของเว็บไซต์บริษัทฝึกอบรม En-Training เพศชาย

หน้าที่:
- แจ้งผู้ใช้อย่างเป็นธรรมชาติว่าระบบพบหลักสูตรที่ใกล้เคียง
- ขอให้ผู้ใช้ยืนยันว่าใช่หลักสูตรนี้ไหม
- ถ้ามีหลายหลักสูตร อาจพูดถึงตัวเลือกหลักอย่างกระชับ
- ห้ามถามข้อมูลอื่นในรอบนี้
- ห้ามใช้ markdown
- ตอบ 1-3 ประโยค
""".strip()

    user_prompt = f"""
ข้อความล่าสุดของผู้ใช้:
{user_message}

หลักสูตรที่ค้นเจอ:
{json.dumps(course_candidates[:3], ensure_ascii=False)}

หลักสูตรที่ใกล้เคียงที่สุด:
{json.dumps(best_course, ensure_ascii=False)}

ให้ถามยืนยันว่าหลักสูตรนี้ใช่สิ่งที่ผู้ใช้ต้องการไหม
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text
            yield {
                "type": "chunk",
                "text": text,
            }

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    return


async def ai_generate_quotation_reply(user_message, state, analysis):
    quotation_context = normalize_quotation_context(
        getattr(state, "quotation_context", {}) or {}
    )

    missing_fields = get_missing_quotation_fields(quotation_context)
    conversation_context = build_conversation_context(state.conversation_history)

    system_prompt = """
คุณคือ AI Assistant ชื่อ En-Assistant ของเว็บไซต์บริษัทฝึกอบรม En-Training เพศชาย

หน้าที่:
- ช่วยผู้ใช้ขอใบเสนอราคาอบรม
- ใช้ข้อมูลที่เก็บได้แล้วประกอบการตอบ
- ถ้าข้อมูลยังขาด ให้ถามข้อมูลที่ยังขาดเพียง 1 อย่าง
- ให้ถามอย่างเป็นธรรมชาติ ไม่ใช่ภาษาฟอร์มแข็ง ๆ
- ถ้าคำตอบผู้ใช้ผิดรูปแบบ ให้บอกสั้น ๆ และขอข้อมูลที่ถูกต้องใหม่
- ห้ามถามหลายข้อพร้อมกัน
- ห้ามแต่งราคาเอง
- ห้ามบอกเวลาติดต่อกลับแบบแน่นอน
- ห้ามใช้ markdown
- ตอบ 2-4 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ผลวิเคราะห์ล่าสุด:
{json.dumps(analysis, ensure_ascii=False)}

ข้อมูลใบเสนอราคาที่เก็บได้แล้ว:
{json.dumps(quotation_context, ensure_ascii=False)}

ข้อมูลที่ยังขาด:
{json.dumps([FIELD_LABELS[f] for f in missing_fields], ensure_ascii=False)}

field ที่ควรถามต่อ:
{FIELD_LABELS.get(analysis.get("next_field_to_ask", ""), "")}

ให้ตอบผู้ใช้แบบเป็นธรรมชาติ
ถ้ายังขาดข้อมูล ให้ถามเฉพาะ field ที่ควรถามต่อ 1 อย่าง
ถ้าข้อมูลครบแล้ว ให้สรุปข้อมูลสั้น ๆ และแจ้งว่าจะส่งข้อมูลให้ทีมงานดำเนินการต่อ
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.35,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text
            yield {
                "type": "chunk",
                "text": text,
            }

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    return


async def handle_pending_course_confirmation(user_message, state):
    quotation_context = normalize_quotation_context(
        getattr(state, "quotation_context", {}) or {}
    )

    conversation_context = build_conversation_context(state.conversation_history)
    course_candidates = quotation_context.get("course_candidates") or []

    confirmation = await ai_analyze_course_confirmation(
        user_message=user_message,
        course_candidates=course_candidates,
        conversation_context=conversation_context,
    )

    decision = str(confirmation.get("decision") or "").strip().lower()
    selected_course_id = str(confirmation.get("selected_course_id") or "").strip()

    selected_course = None

    if selected_course_id:
        for course in course_candidates:
            if str(course.get("course_id")) == selected_course_id or str(course.get("course_no")) == selected_course_id:
                selected_course = course
                break

    if not selected_course and decision == "yes" and course_candidates:
        selected_course = course_candidates[0]

    if selected_course:
        quotation_context["course_id"] = selected_course.get("course_id") or selected_course.get("course_no") or ""
        quotation_context["course_name"] = selected_course.get("course_name") or ""
        quotation_context["course_confirmed"] = True
        quotation_context["course_confirmation_pending"] = False
        quotation_context["course_candidates"] = []

        state.quotation_context = quotation_context

        return quotation_context, confirmation, None

    if decision == "no":
        quotation_context["course_id"] = ""
        quotation_context["course_name"] = ""
        quotation_context["course_confirmed"] = False
        quotation_context["course_confirmation_pending"] = False
        quotation_context["course_candidates"] = []

        state.quotation_context = quotation_context

        reply = "ขอบคุณครับ งั้นรบกวนแจ้งชื่อหลักสูตรหรือหัวข้ออบรมที่ต้องการอีกครั้งได้ไหมครับ ผมจะช่วยค้นหาหลักสูตรที่ตรงกว่าให้ครับ"
        return quotation_context, confirmation, reply

    reply = "ขออนุญาตยืนยันอีกครั้งครับ หลักสูตรที่ระบบพบใช่หลักสูตรที่คุณต้องการขอใบเสนอราคาไหมครับ"
    return quotation_context, confirmation, reply


async def handle_quotation(req, state):
    user_message = (req.user_message or "").strip()

    quotation_context = normalize_quotation_context(
        getattr(state, "quotation_context", {}) or {}
    )

    state.quotation_context = quotation_context

    state.conversation_history.append({
        "role": "user",
        "content": user_message
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    if quotation_context.get("course_confirmation_pending"):
        quotation_context, confirmation, direct_reply = await handle_pending_course_confirmation(
            user_message=user_message,
            state=state,
        )

        if direct_reply:
            yield {
                "type": "chunk",
                "text": direct_reply
            }

            state.mode = "collecting_info"
            state.pending_action = "quotation"
            state.last_answer = direct_reply
            state.last_step = "quotation"
            state.quotation_context = quotation_context

            state.conversation_history.append({
                "role": "assistant",
                "content": direct_reply
            })

            yield {
                "type": "done",
                "reply": direct_reply,
                "status": "collecting_info",
                "reason": "course_confirmation",
                "course_confirmation": confirmation,
                "missing_fields": get_missing_quotation_fields(quotation_context),
                "quotation_context": quotation_context,
                "state": state,
                "source": "ai_assistant",
            }

            return

    quotation_context, analysis = await ai_analyze_quotation(
        user_message=user_message,
        state=state,
    )

    quotation_context = await prepare_course_confirmation_if_needed(quotation_context)
    state.quotation_context = quotation_context

    if quotation_context.get("course_confirmation_pending"):
        reply = ""

        async for item in ai_generate_course_confirmation_reply(
            user_message=user_message,
            state=state,
        ):
            if item.get("type") == "chunk":
                reply += item.get("text", "")
                yield item

        state.mode = "collecting_info"
        state.pending_action = "quotation"
        state.last_answer = reply
        state.last_step = "quotation"
        state.quotation_context = quotation_context

        state.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        yield {
            "type": "done",
            "reply": reply,
            "status": "collecting_info",
            "reason": "course_confirmation_required",
            "analysis": analysis,
            "matched_courses": quotation_context.get("course_candidates", []),
            "missing_fields": get_missing_quotation_fields(quotation_context),
            "quotation_context": quotation_context,
            "state": state,
            "source": "ai_assistant",
        }

        return

    reply = ""

    async for item in ai_generate_quotation_reply(
        user_message=user_message,
        state=state,
        analysis=analysis,
    ):
        if item.get("type") == "chunk":
            reply += item.get("text", "")
            yield item

    missing_fields = get_missing_quotation_fields(quotation_context)
    status = "collecting_info" if missing_fields else "quotation_ready"

    insert_result = None

    if status == "quotation_ready" and not quotation_context.get("quotation_inserted"):
        insert_result = await insert_quotation_request(state)
        quotation_context["quotation_inserted"] = True
        quotation_context["insert_result"] = insert_result
        state.quotation_context = quotation_context
        
    state.mode = status
    state.pending_action = "quotation"
    state.last_answer = reply
    state.last_step = "quotation"
    state.quotation_context = quotation_context

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": status,
        "reason": "quotation",
        "analysis": analysis,
        "missing_fields": missing_fields,
        "quotation_context": quotation_context,
        "state": state,
        "source": "ai_assistant",
        "insert_result": insert_result,
    }

    return