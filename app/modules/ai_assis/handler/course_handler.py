import json
import re
from urllib.parse import quote

from app.modules.ai_assis.service import (
    extract_requirements,
    calc_missing_requirements,
    build_next_question,
    build_search_query,
    build_next_question_topic,
    build_more_courses_reply,
    detect_post_recommend_intent,
    build_conversation_context,
    _stream_text_response,
    build_inhouse_discovery_reply,
    build_inhouse_topic_not_found_reply,
    classify_inhouse_need,
    fetch_inhouse_course_detail,
)

from app.modules.ai_assis.qdrant_service import (
    search_courses_from_qdrant,
    check_topic_exists_in_qdrant,
)

from app.modules.ai_assis.public_course_service import (
    fetch_public_course_context,
    fetch_public_course_detail,
)

from app.shared.ai.openai_client import (
    call_openai_chat_full,
)

def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", "", text)
    return text

async def resolve_inhouse_course_from_qdrant(
    user_message: str,
    state,
    limit: int = 5,
) -> dict | None:

    if not state.course_context:
        state.course_context = {}

    need = await classify_inhouse_need(
        user_message=user_message,
        course_context=state.course_context,
        conversation_history=state.conversation_history,
    )

    search_query = (need.get("search_query") or "").strip()

    if not search_query:
        search_query = await build_search_query(state.course_context)

    courses = await search_courses_from_qdrant(
        search_query,
        limit=limit,
        excluded_courses=[]
    )

    if not courses:
        return None

    matched_course = courses[0]
    payload = get_course_payload(matched_course)

    course_id = (
        payload.get("course_no")
        or payload.get("OCourse_no")
        or payload.get("id")
    )

    course_name = (
        payload.get("course_name")
        or payload.get("vdo_name")
        or payload.get("Course_name")
        or payload.get("title")
    )

    resolved = {
        "course_id": course_id,
        "course_name": course_name,
        "course": matched_course,
        "payload": payload,
        "search_query": search_query,
        "need": need,
    }

    state.course_context["last_inhouse_course"] = matched_course
    state.course_context["last_inhouse_course_id"] = course_id
    state.course_context["last_inhouse_course_name"] = course_name
    state.course_context["last_inhouse_search_query"] = search_query

    state.matched_course = matched_course
    state.matched_course_id = course_id

    return resolved

async def match_public_course_by_ai(
    user_message: str,
    courses: list[dict],
    conversation_history: list | None = None,
    state=None
) -> dict | None:

    conversation_context = build_conversation_context(conversation_history)

    compact_courses = []

    for i, course in enumerate(courses):
        compact_courses.append({
            "index": i,
            "course_name": course.get("course_name", ""),
            "course_name_en": course.get("course_name_en", ""),
            "course_date": course.get("course_date", ""),
            "month": course.get("month", ""),
            "price": course.get("price", ""),
            "badge": course.get("badge", ""),
        })

    system_prompt = """
คุณคือ Course Matcher

หน้าที่:
- เลือกหลักสูตร Public Training ที่ผู้ใช้หมายถึงจาก course_candidates เท่านั้น
- ถ้าผู้ใช้พูดว่า "คอร์สนี้", "อันนี้", "หลักสูตรนี้" ให้ดูบทสนทนาก่อนหน้าและ last_public_course
- ถ้าไม่มั่นใจ ให้ตอบ matched_index = null
- ห้ามแต่งชื่อหลักสูตร
- ตอบ JSON เท่านั้น

รูปแบบ JSON:
{
  "matched_index": 0,
  "confidence": 0.0,
  "reason": ""
}
""".strip()

    last_public_course = {}
    if state is not None:
        course_context = getattr(state, "course_context", {}) or {}
        last_public_course = course_context.get("last_public_course") or {}

    user_prompt = f"""
ข้อความล่าสุดของผู้ใช้:
{user_message}

บทสนทนาก่อนหน้า:
{conversation_context}

last_public_course:
{json.dumps(last_public_course, ensure_ascii=False)}

course_candidates:
{json.dumps(compact_courses, ensure_ascii=False)}

เลือก course ที่ตรงที่สุด
""".strip()

    result = await call_openai_chat_full(
        model="gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
    )

    text = (result.get("content") or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
        matched_index = data.get("matched_index", None)
        confidence = float(data.get("confidence", 0))

        if matched_index is None:
            return None

        matched_index = int(matched_index)

        if confidence < 0.55:
            return None

        if matched_index < 0 or matched_index >= len(courses):
            return None

        return courses[matched_index]

    except Exception:
        return None

def get_course_payload(course):
    if isinstance(course, dict):
        payload = course.get("payload") or course
        return payload if isinstance(payload, dict) else {}

    return {}

def get_course_id(course):
    payload = get_course_payload(course)

    return (
        payload.get("course_no")
        or payload.get("OCourse_no")
        or payload.get("id")
    )

def build_public_course_url(course_name: str) -> str:
    course_name = (course_name or "").strip()

    if not course_name:
        return ""

    # หน้าแผนอบรมรวมทั้งหมด
    if (
        "ตารางอบรม" in course_name
        or "Public Course" in course_name
        or "ทั้งหมด" in course_name
        or "Entraining" in course_name
    ):
        return "https://www.entraining.net/public-course/plan/all/"

    slug = re.sub(r"\s+", "-", course_name)
    slug = re.sub(r"-+", "-", slug)

    return f"https://www.entraining.net/public-course/{slug}"

async def handle_inhouse_course_detail(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    if not state.course_context:
        state.course_context = {}

    course_context = state.course_context or {}

    matched_course = (
        course_context.get("last_inhouse_course")
        or course_context.get("matched_course")
        or getattr(state, "matched_course", None)
    )

    if isinstance(matched_course, list):
        matched_course = matched_course[0] if matched_course else None

    course_payload = get_course_payload(matched_course)

    course_id = (
        course_payload.get("course_no")
        or course_payload.get("OCourse_no")
        or course_payload.get("id")
        or course_context.get("last_inhouse_course_id")
        or getattr(state, "matched_course_id", None)
    )

    course_name = (
        course_payload.get("course_name")
        or course_payload.get("vdo_name")
        or course_payload.get("Course_name")
        or course_payload.get("title")
        or course_context.get("last_inhouse_course_name")
    )

    if not course_id and not course_name:

        resolved = await resolve_inhouse_course_from_qdrant(
            user_message=user_message,
            state=state,
            limit=5,
        )

        if resolved:
            matched_course = resolved.get("course")
            course_payload = resolved.get("payload") or {}
            course_id = resolved.get("course_id")
            course_name = resolved.get("course_name")

            state.course_context["last_inhouse_course"] = matched_course
            state.course_context["last_inhouse_course_id"] = course_id
            state.course_context["last_inhouse_course_name"] = course_name
            state.course_context["last_inhouse_search_query"] = resolved.get("search_query")

    if not course_id and not course_name:

        reply = ""

        async for item in build_inhouse_discovery_reply(
            user_message=user_message,
            requirements=course_context,
            conversation_history=state.conversation_history,
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

        state.mode = "course_discovery"
        state.last_answer = reply
        state.last_step = "inhouse_detail_need_course"

        state.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        if len(state.conversation_history) > 10:
            state.conversation_history = state.conversation_history[-10:]

        yield {
            "type": "done",
            "reply": reply,
            "status": "collecting_info",
            "reason": "inhouse_detail_need_course",
            "state": state,
            "source": "ai_assistant_inhouse_detail",
        }

        return

    course_detail = await fetch_inhouse_course_detail(
        course_id=course_id,
        course_name=course_name,
    )

    merged_course_detail = {
        **course_payload,
        **(course_detail or {}),
    }

    if course_id:
        merged_course_detail["course_id"] = course_id

    if course_name:
        merged_course_detail["course_name"] = course_name

    state.course_context["course_type"] = "inhouse"
    state.course_context["course_action"] = "detail"
    state.course_context["last_inhouse_course"] = matched_course
    state.course_context["last_inhouse_course_id"] = course_id
    state.course_context["last_inhouse_course_name"] = course_name
    state.course_context["last_inhouse_course_detail"] = merged_course_detail

    system_prompt = """
คุณคือ AI Sales Consultant ของเว็บไซต์ En-Training

หน้าที่:
- ตอบรายละเอียดหลักสูตร In-house Training จาก inhouse_course_detail เท่านั้น
- ใช้ข้อมูลชื่อหลักสูตร วัตถุประสงค์ เนื้อหา กลุ่มเป้าหมาย และรายละเอียดที่มีใน context
- ถ้า detail ยังมีน้อย ให้ตอบจากข้อมูลที่มี และถามต่ออย่างสุภาพว่าต้องการปรับหลักสูตรให้เหมาะกับกลุ่มผู้เรียนแบบไหน
- ห้ามแต่งข้อมูลที่ไม่มีใน context
- ห้ามบอกว่าจะติดต่อกลับภายในเวลาที่แน่นอน
- ห้ามใช้ markdown
- ตอบแบบธรรมชาติ เหมือนที่ปรึกษาฝึกอบรม
- ตอบ 3-6 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

inhouse_course_detail:
{json.dumps(merged_course_detail, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบรายละเอียดหลักสูตร In-house จาก inhouse_course_detail เท่านั้น
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
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

    state.mode = "inhouse_course_detail"
    state.last_answer = reply
    state.last_step = "inhouse_course_detail"

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": "answered",
        "reason": "inhouse_course_detail",
        "state": state,
        "source": "ai_assistant_inhouse_detail",
    }

    return

async def handle_public_course_detail(req, state):
    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    if not state.course_context:
        state.course_context = {}

    public_courses = state.course_context.get("public_context")

    if not public_courses:
        public_courses = await fetch_public_course_context()
        state.course_context["public_context"] = public_courses

    matched_course = await match_public_course_by_ai(
        user_message=user_message,
        courses=public_courses,
        state=state
    )

    print("matched_course =", matched_course, flush=True)

    if matched_course:
        state.course_context["last_public_course"] = matched_course

    if not matched_course:
        reply = "อยากทราบรายละเอียดของหลักสูตรไหนครับ รบกวนพิมพ์ชื่อหลักสูตร หรือบอกหัวข้อที่สนใจเพิ่มเติมได้เลยครับ"

        state.mode = "public_course_detail"
        state.last_answer = reply
        state.last_step = "public_course_detail_need_course"

        yield {
            "type": "done",
            "reply": reply,
            "status": "collecting_info",
            "reason": "public_course_detail_need_course",
            "state": state,
            "source": "entraining_public_course_detail",
        }

        return

    course_url = (matched_course.get("course_url") or "").strip()
    course_url = course_url.replace(
    "https://www.entraining.net/public-course/",
    "https://entstaffs.entraining.net/api/public-course/"
)

    if not course_url:
        reply = "ยังไม่พบลิงก์รายละเอียดของหลักสูตรนี้ครับ"

        yield {
            "type": "done",
            "reply": reply,
            "status": "not_found",
            "reason": "public_course_url_not_found",
            "state": state,
            "source": "entraining_public_course_detail",
        }

        return

    course_detail_context = await fetch_public_course_detail(course_url)

    original_course_url = course_url.replace(
    "https://entstaffs.entraining.net/api/public-course/",
    "https://www.entraining.net/public-course/"
    )
    course_detail_context["course_url"] = original_course_url

    merged_course_detail = {
        **matched_course,
        **course_detail_context,
    }


    state.course_context["course_type"] = "public"
    state.course_context["course_action"] = "detail"
    state.course_context["last_public_course"] = matched_course
    state.course_context["last_public_course_detail"] = merged_course_detail

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์ En-Training

หน้าที่:
- ตอบรายละเอียดหลักสูตร Public Training จาก public_course_detail เท่านั้น
- ใช้ข้อมูลชื่อหลักสูตร วันที่ ราคา ลิงก์สมัคร โบรชัวร์ และ course_outline ประกอบการตอบ
- ห้ามแต่งข้อมูลที่ไม่มีใน context
- ถ้าพูดถึงชื่อหลักสูตร ให้ทำเป็นลิงก์ HTML จาก course_url ที่ให้มา
- รูปแบบลิงก์หลักสูตรต้องเป็น:
<a href="URL" target="_blank" style="color:#004AAD;font-weight:700;text-decoration:none;border-bottom:1px solid #287CED;">ชื่อหลักสูตร</a>
- ห้ามสร้าง URL เอง ถ้าไม่มี course_url
- ห้ามนำชื่อ "ตารางอบรม Public Course เดือน ทั้งหมด 2569 | Entraining" ไปสร้างเป็นลิงก์หลักสูตร
- ใช้ลิงก์ /public-course/{course_name} เฉพาะเมื่อเป็นชื่อหลักสูตรจริงเท่านั้น
- ถ้ามี register_url ให้แนบลิงก์สมัครอบรม โดยใช้คำว่า:
<a href="URL" target="_blank" style="color:#004AAD;font-weight:700;text-decoration:none;border-bottom:1px solid #287CED;">สมัครอบรม</a>
- ถ้ามี brochure_url ให้แนบลิงก์ดาวน์โหลดโบรชัวร์ โดยใช้คำว่า:
<a href="URL" target="_blank" style="color:#004AAD;font-weight:700;text-decoration:none;border-bottom:1px solid #287CED;">ดาวน์โหลดโบรชัวร์</a>
- ถ้าพูดถึงตารางอบรมรวมทั้งหมด หรือภาพรวม Public Training ทั้งหมด ให้ใช้ลิงก์:
<a href="https://www.entraining.net/public-course/plan/all/" target="_blank" style="color:#004AAD;font-weight:700;text-decoration:none;border-bottom:1px solid #287CED;">ตารางอบรม Public Training ทั้งหมด</a>
- ห้ามใช้ markdown
- ตอบแบบธรรมชาติ เหมือนเจ้าหน้าที่ช่วยแนะนำ
- ตอบ 3-6 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

public_course_detail:
{json.dumps(merged_course_detail, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบรายละเอียดหลักสูตรนี้จาก public_course_detail เท่านั้น
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text

            yield {
                "type": "chunk",
                "text": text
            }

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    state.mode = "public_course_detail"
    state.last_answer = reply
    state.last_step = "public_course_detail"

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": "answered",
        "reason": "public_course_detail",
        "state": state,
        "source": "entraining_public_course_detail",
    }

    return

async def handle_inhouse_course_search(req, state):

    user_message = (req.user_message or "").strip()

    if not state.course_context:
        state.course_context = {}

    state.course_context["course_type"] = "inhouse"

    need = await classify_inhouse_need(
        user_message=user_message,
        course_context=state.course_context,
        conversation_history=state.conversation_history,
    )

    print("INHOUSE NEED =", need, flush=True)

    need_type = need.get("need_type", "unclear")
    should_search = need.get("should_search", False)

    if need.get("topic"):
        state.course_context["topic"] = need.get("topic")

    if need.get("pain_point"):
        state.course_context["pain_point"] = need.get("pain_point")

    if need.get("target_group"):
        state.course_context["target_group"] = need.get("target_group")

    state.course_context["inhouse_need_type"] = need_type

    # 1) ยังไม่ควร search ให้ AI ชวนคุยก่อน
    if not should_search:

        reply = ""

        async for item in build_inhouse_discovery_reply(
            user_message=user_message,
            requirements=state.course_context,
            conversation_history=state.conversation_history,
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

        state.mode = "course_discovery"
        state.last_answer = reply
        state.last_step = "inhouse_discovery"
        state.course_context["course_stage"] = "discover"

        state.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        if len(state.conversation_history) > 10:
            state.conversation_history = state.conversation_history[-10:]

        yield {
            "type": "done",
            "reply": reply,
            "status": "collecting_info",
            "reason": "inhouse_discovery",
            "state": state,
            "source": "ai_assistant_inhouse",
        }

        return

    # 2) AI บอกว่าควร search แล้ว
    search_query = (need.get("search_query") or "").strip()

    if not search_query:
        search_query = await build_search_query(state.course_context)

    state.course_context["search_query"] = search_query

    print("INHOUSE SEARCH QUERY =", search_query, flush=True)

    topic_check_courses, course_id = await check_topic_exists_in_qdrant(
        search_query,
        limit=5
    )

    # 3) ไม่พบหลักสูตร ให้ AI ถาม refine
    if not topic_check_courses:

        reply = ""

        async for item in build_inhouse_topic_not_found_reply(
            user_message=user_message,
            requirements=state.course_context,
            conversation_history=state.conversation_history,
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

        state.mode = "course_discovery"
        state.last_answer = reply
        state.last_step = "inhouse_topic_not_found"
        state.course_context["course_stage"] = "refine"

        state.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        if len(state.conversation_history) > 10:
            state.conversation_history = state.conversation_history[-10:]

        yield {
            "type": "done",
            "reply": reply,
            "status": "collecting_info",
            "reason": "inhouse_topic_not_found",
            "state": state,
            "source": "ai_assistant_inhouse",
        }

        return

    matched_courses = (
        topic_check_courses
        if isinstance(topic_check_courses, list)
        else [topic_check_courses]
    )

    matched_courses = [c for c in matched_courses if c]
    matched_courses = matched_courses[:2]
    print("matched_course =", matched_courses, flush=True)
    state.course_context["matched_course"] = matched_courses
    state.course_context["course_stage"] = "matched"
    state.recommended_courses = matched_courses
    state.recommended_course_cta = []

    course_cta = []

    for course in matched_courses:
        payload = get_course_payload(course)

        course_no = (
            payload.get("course_no")
            or payload.get("OCourse_no")
            or payload.get("id")
        )

        course_name = (
            payload.get("course_name")
            or payload.get("vdo_name")
            or payload.get("Course_name")
            or payload.get("title")
        )

        if course_no and course_name:
            course_cta.append({
                "course_no": course_no,
                "course_name": course_name,
            })

    state.recommended_course_cta = course_cta

    reply = ""

    async for item in build_next_question_topic(
        requirements=state.course_context,
        missing=[],
        conversation_history=state.conversation_history,
        matched_course=matched_courses
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

    state.last_answer = reply
    state.matched_course = matched_courses
    state.matched_course_id = course_id
    state.last_step = "inhouse_course_matched"
    state.mode = "course_post_recommend"

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": "recommended",
        "reason": "inhouse_course_matched",
        "state": state,
        "source": "ai_assistant_course_qdrant",
    }

    return

async def handle_public_course_search(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    public_course_context = await fetch_public_course_context()

    public_course_context_with_links = []

    for item in public_course_context:
        if isinstance(item, dict):
            course_name = (item.get("course_name") or "").strip()
            course_url = (item.get("course_url") or "").strip()
            course_name_en = (item.get("course_name_en") or "").strip()
            course_date = (item.get("course_date") or "").strip()
            month = (item.get("month") or "").strip()
            price = (item.get("price") or "").strip()
            badge = (item.get("badge") or "").strip()
            image_url = (item.get("image_url") or "").strip()
            register_url = (item.get("register_url") or "").strip()
            brochure_url = (item.get("brochure_url") or "").strip()
        else:
            course_name = str(item or "").strip()
            course_url = build_public_course_url(course_name)
            course_name_en = ""
            course_date = ""
            month = ""
            price = ""
            badge = ""
            image_url = ""
            register_url = ""
            brochure_url = ""

        if not course_name:
            continue

        public_course_context_with_links.append({
            "course_name": course_name,
            "course_name_en": course_name_en,
            "course_url": course_url,
            "course_date": course_date,
            "month": month,
            "price": price,
            "badge": badge,
            "image_url": image_url,
            "register_url": register_url,
            "brochure_url": brochure_url,
        })

    if not state.course_context:
        state.course_context = {}

    state.course_context["course_type"] = "public"
    state.course_context["public_source_url"] = "https://www.entraining.net/public-course/plan/all/"
    state.course_context["public_context"] = public_course_context_with_links
    state.course_context["last_public_courses"] = public_course_context_with_links[:5]

    if public_course_context_with_links:
        state.course_context["last_public_course"] = public_course_context_with_links[0]
    

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์ En-Training

หน้าที่:
- ตอบคำถามเกี่ยวกับหลักสูตรแบบ Public Training จาก public_course_context เท่านั้น
- สรุปและแนะนำหลักสูตรที่เกี่ยวข้องกับสิ่งที่ผู้ใช้ถาม
- ถ้าพูดถึงชื่อหลักสูตร ให้ทำเป็นลิงก์ HTML จาก course_url ที่ให้มา
- รูปแบบลิงก์ต้องเป็น <a href="URL"
target="_blank"
style="
color:#004AAD;
font-weight:700;
text-decoration:none;
border-bottom:1px solid #287CED;
">
ชื่อหลักสูตร
</a>

- ห้ามสร้าง URL เอง ถ้าไม่มี course_url
- ห้ามแต่งชื่อหลักสูตร วันที่ ราคา หรือรายละเอียดที่ไม่มีใน context
- public_course_context อาจมี field course_name_en, course_date, month, price, badge, register_url, brochure_url ให้ใช้ข้อมูลเหล่านี้ประกอบการตอบ
- ถ้าผู้ใช้ถามวันที่อบรม ให้ดูจาก course_date
- ถ้าผู้ใช้ถามเดือนที่เปิดอบรม ให้ดูจาก month
- ถ้าผู้ใช้ต้องการสมัคร ให้ทำลิงก์จาก register_url โดยใช้คำว่า สมัครอบรม
- ถ้าผู้ใช้ต้องการโบรชัวร์ ให้ทำลิงก์จาก brochure_url โดยใช้คำว่า ดาวน์โหลดโบรชัวร์
- ถ้าผู้ใช้ถามราคา ให้ดูจาก price
- ถ้าผู้ใช้ต้องการสมัคร ให้ใช้ register_url ถ้ามี
- ถ้าผู้ใช้ต้องการโบรชัวร์ ให้ใช้ brochure_url ถ้ามี
- ถ้าพูดถึงตารางอบรมรวมทั้งหมด หรือภาพรวม Public Training ทั้งหมด ให้ใช้ลิงก์ https://www.entraining.net/public-course/plan/all/
- ห้ามนำชื่อ "ตารางอบรม Public Course เดือน ทั้งหมด 2569 | Entraining" ไปสร้างเป็นลิงก์หลักสูตร
- ใช้ลิงก์ /public-course/{course_name} เฉพาะเมื่อเป็นชื่อหลักสูตรจริงเท่านั้น
- ถ้า context ไม่มีข้อมูลตรงกับสิ่งที่ถาม ให้บอกอย่างสุภาพว่าไม่พบข้อมูลชัดเจนในรอบ public ที่มีอยู่
- ถ้ามีข้อมูลหลายหลักสูตร ให้แนะนำเฉพาะตัวที่เกี่ยวข้องที่สุด
- ตอบแบบธรรมชาติ เหมือนเจ้าหน้าที่ช่วยแนะนำ
- ห้ามคัดลอก context มาทั้งดุ้น
- ห้ามใช้ markdown
- ตอบ 2-5 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

public_course_context:
{json.dumps(public_course_context_with_links, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบคำถามหรือแนะนำหลักสูตร Public Training จาก public_course_context เท่านั้น
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.4,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text

            yield {
                "type": "chunk",
                "text": text
            }

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    state.mode = "public_course_search"
    state.last_answer = reply
    state.last_step = "public_course_search"

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": "answered",
        "reason": "public_course_search",
        "state": state,
        "source": "entraining_public_course_page",
    }

    return

async def handle_course_search(req, state):

    course_context = getattr(state, "course_context", {}) or {}
    course_type = course_context.get("course_type") or "unknown"
    course_action = course_context.get("course_action") or "overview"

    if course_type == "public":

        if course_action == "detail":
            async for item in handle_public_course_detail(
                req=req,
                state=state
            ):
                yield item
            return

        async for item in handle_public_course_search(
            req=req,
            state=state
        ):
            yield item
        return

    if course_type == "inhouse":

        if course_action == "detail":
            async for item in handle_inhouse_course_detail(
                req=req,
                state=state
            ):
                yield item
            return

        async for item in handle_inhouse_course_search(
            req=req,
            state=state
        ):
            yield item
        return

    async for item in ask_course_type(
        req=req,
        state=state
    ):
        yield item

    return

async def ask_course_type(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    course_context = getattr(state, "course_context", {}) or {}

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์บริษัทฝึกอบรม

หน้าที่:
- ช่วยผู้ใช้ที่สนใจหลักสูตร แต่ยังไม่ชัดว่าอยากดูแบบ Public Training หรือ In-house Training
- ห้ามถามแข็ง ๆ เหมือนเลือกเมนู
- ให้สะท้อนหัวข้อที่ผู้ใช้สนใจก่อน ถ้ามี
- อธิบายความต่างแบบสั้น ๆ:
  Public Training = รอบอบรมทั่วไปที่เปิดให้สมัคร
  In-house Training = จัดอบรมภายในองค์กรและปรับให้เหมาะกับทีม
- ถามต่อ 1 คำถามอย่างเป็นธรรมชาติ
- ห้ามใช้ markdown
- ตอบ 2-4 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

course_context:
{json.dumps(course_context or {}, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบกลับเพื่อถามต่ออย่างเป็นธรรมชาติว่าอยากดูแบบ Public หรือ In-house
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text
            yield {
                "type": "chunk",
                "text": text
            }

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    state.mode = "course_select_type"
    state.last_answer = reply
    state.last_step = "ask_course_type"

    if not state.course_context:
        state.course_context = {}

    state.course_context["course_type"] = "unknown"

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": "collecting_info",
        "reason": "ask_course_type",
        "state": state,
        "source": "ai_assistant_course_type",
    }

    return