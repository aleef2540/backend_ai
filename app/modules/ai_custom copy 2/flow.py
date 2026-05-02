from app.modules.ai_custom.course_service import get_course_data_by_nos_bridge
import json

from app.modules.ai_custom.service import (
    extract_requirements,
    calc_missing_requirements,
    build_next_question,
    build_search_query,
    reply_ask_concept_with_topic_stream,
    build_irrelevant_content_reply,
)

from app.modules.ai_custom.schema import ChatState_aicustom
from app.modules.ai_custom.rag_service import search_rag, build_rag_context, build_active_video_from_rag
import random




def build_video_payload(video):
    if not video:
        return None

    video_url = str(video.get("video_url") or "").strip()

    return {
        "video_part": video.get("video_part"),
        "video_name": video.get("video_name"),
        "video_id": video_url,
        "embed_url": f"https://www.youtube.com/embed/{video_url}" if video_url else None
    }

def build_course_name_context(course_data) -> str:
    names = []
    seen = set()

    for row in course_data:
        course_name = str(row.get("course_name") or "").strip()

        if not course_name:
            continue

        if course_name in seen:
            continue

        seen.add(course_name)
        names.append(course_name)

    return ", ".join(names)


def find_script_by_topic(course_data, topic: str) -> str:
    topic_clean = str(topic or "").strip().lower()

    if not topic_clean or topic_clean == "unknown":
        return ""

    for row in course_data:
        course_name = str(row.get("course_name") or "").strip()
        script = str(row.get("script") or "").strip()

        if course_name.lower() == topic_clean:
            return script

    for row in course_data:
        course_name = str(row.get("course_name") or "").strip().lower()
        script = str(row.get("script") or "").strip()

        if topic_clean in course_name or course_name in topic_clean:
            return script

    return ""

def find_course_by_topic(course_data, topic: str):
    topic_clean = str(topic or "").strip().lower()

    if not topic_clean or topic_clean == "unknown":
        return None

    for row in course_data:
        course_name = str(row.get("course_name") or "").strip()
        if course_name.lower() == topic_clean:
            return row

    for row in course_data:
        course_name = str(row.get("course_name") or "").strip().lower()
        if topic_clean in course_name or course_name in topic_clean:
            return row

    return None

def find_course_by_no(course_data, course_no):
    if not course_no:
        return None

    try:
        target_no = int(course_no)
    except Exception:
        return None

    for row in course_data:
        try:
            row_no = int(row.get("course_no"))
        except Exception:
            row_no = None

        if row_no == target_no:
            return row

    return None


def detect_followup_type(message: str) -> str:
    text = str(message or "").strip().lower()

    summary_keywords = [
        "สรุป", "ย่อ", "เอาแบบสั้น", "แบบสั้น", "สั้นๆ", "สั้น ๆ", "short summary"
    ]
    example_keywords = [
        "ตัวอย่าง", "ยกตัวอย่าง", "example", "มีเคสไหม", "มีกรณีไหม"
    ]
    application_keywords = [
        "เอาไปใช้", "นำไปใช้", "ใช้ยังไง", "ทำยังไง", "ควรทำยังไง",
        "ปรับใช้", "ใช้กับงาน", "ใช้กับลูกทีม", "ประยุกต์ใช้"
    ]
    comparison_keywords = [
        "ต่างจาก", "แตกต่าง", "เปรียบเทียบ", "เทียบกับ", "ดีกว่า", "เหมือนกันไหม"
    ]
    expand_keywords = [
        "ขยาย", "เพิ่ม", "เพิ่มอีก", "อีกหน่อย", "เพิ่มเติม", "เล่าเพิ่ม", "อธิบายเพิ่ม"
    ]

    if any(k in text for k in summary_keywords):
        return "summary"

    if any(k in text for k in example_keywords):
        return "example"

    if any(k in text for k in application_keywords):
        return "application"

    if any(k in text for k in comparison_keywords):
        return "comparison"

    if any(k in text for k in expand_keywords):
        return "expand"

    return "expand"


def map_followup_answer_type(followup_type: str) -> str:
    mapping = {
        "summary": "summary_given",
        "example": "example_given",
        "application": "application_given",
        "comparison": "comparison_given",
        "expand": "followup_expanded",
    }
    return mapping.get(followup_type, "followup_expanded")

def build_application_message(user_message: str, state) -> str:
    return (
        f"{user_message}\n\n"
        f"[ANSWER_MODE]: application\n"
        f"[PREVIOUS_ANSWER]: {state.last_answer or ''}\n"
        f"[PREVIOUS_INTENT]: {state.last_intent or ''}\n"
        f"[PREVIOUS_ANSWER_TYPE]: {state.last_answer_type or ''}\n"
        f"[INSTRUCTION]: ช่วยตอบในเชิงการนำไปใช้จริง วิธีเริ่มต้น ขั้นตอน หรือแนวทางปฏิบัติที่นำไปใช้กับงานได้"
    )

def build_summary_message(user_message: str, state) -> str:
    return (
        f"{user_message}\n\n"
        f"[ANSWER_MODE]: summary\n"
        f"[PREVIOUS_ANSWER]: {state.last_answer or ''}\n"
        f"[PREVIOUS_INTENT]: {state.last_intent or ''}\n"
        f"[PREVIOUS_ANSWER_TYPE]: {state.last_answer_type or ''}\n"
        f"[INSTRUCTION]: ช่วยสรุปให้กระชับ เข้าใจง่าย เน้นใจความสำคัญ ไม่ต้องยาว"
    )


def load_allowed_course_context(course_use):
    """Load allowed course names from course_no list for discovery replies."""
    if not course_use:
        return [], ""

    try:
        course_data = get_course_data_by_nos_bridge(course_use) or []
    except Exception as e:
        print("[AI_CUSTOM] LOAD COURSE DATA ERROR", repr(e), flush=True)
        return [], ""

    course_name_context = build_course_name_context(course_data)
    return course_data, course_name_context


def build_discovery_message_with_course_context(user_message: str, requirements, missing, course_name_context: str) -> str:
    return f"""
ข้อความผู้เรียน:
{user_message}

Requirement ปัจจุบัน:
{json.dumps(requirements or {}, ensure_ascii=False)}

Requirement ที่ยังขาด:
{json.dumps(missing or [], ensure_ascii=False)}

หลักสูตร/หัวข้อที่ระบบอนุญาตให้ใช้เป็นความรู้:
{course_name_context or ""}

คำสั่ง:
- ผู้เรียนยังไม่ได้ระบุ content/topic ที่ชัดเจน หรือยังมี requirement ที่ต้องเก็บเพิ่ม
- ห้ามตอบว่าระบบไม่มีข้อมูลหลักสูตร ถ้ามีรายชื่อหลักสูตรด้านบน
- ให้ทักทายหรือรับข้อความของผู้เรียนอย่างเป็นธรรมชาติ
- บอกแบบสั้น ๆ ว่าคุณช่วยเรื่องอะไรได้บ้างจากรายชื่อหลักสูตรด้านบน โดยยกตัวอย่างไม่เกิน 3-5 รายการ
- อย่าลิสต์ยาวเกินไป
- ปิดท้ายด้วยคำถาม 1 คำถาม เพื่อถามว่าอยากเรียนหรือพัฒนาเรื่องไหนเป็นพิเศษ
- ตอบภาษาไทย สุภาพ กระชับ
""".strip()


async def process_chat_aicustom_stream(req, state):
    if state is None:
        state = ChatState_aicustom()

    user_message = (req.user_message or "").strip()

    state.web_no = int(req.web_no) if req.web_no not in [None, ""] else None
    state.member_no = int(req.member_no) if req.member_no not in [None, ""] else None

    if req.course_use:
        state.course_use = [int(x) for x in req.course_use if str(x).strip()]

    course_use = state.course_use or []

    if not course_use:
        reply = "ขออภัยครับ ยังไม่พบรายการหลักสูตรที่อนุญาตให้ใช้งาน"
        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "no_course",
            "reason": "empty_course_use",
            "state": state,
            "source": "ai_custom_no_course",
            "active_video": None,
        }
        return

    # =========================
    # 0) โหลดรายชื่อ course ที่อนุญาตจาก course_no
    #    ใช้เป็น context ตอนผู้เรียนยังไม่ได้บอก content/topic
    #    ไม่ใช่การค้น RAG และไม่ควรใช้แทน RAG เมื่อต้องตอบเนื้อหาเชิงลึก
    # =========================
    course_data, course_name_context = load_allowed_course_context(course_use)
    state.allowed_course_data = course_data
    state.allowed_course_name_context = course_name_context

    # =========================
    # 1) เก็บ conversation history
    # =========================
    if not hasattr(state, "conversation_history") or state.conversation_history is None:
        state.conversation_history = []

    state.last_user_message = user_message

    state.conversation_history.append({
        "role": "user",
        "content": user_message
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    # =========================
    # 2) Extract requirement ก่อน
    # =========================
    old_requirements = dict(getattr(state, "requirements", {}) or {})

    new_requirements = await extract_requirements(
        user_message=user_message,
        current_requirements=old_requirements,
        conversation_history=state.conversation_history
    )

    old_req = " ".join(
    str(v).strip()
    for k, v in sorted((old_requirements or {}).items())
    if k not in ["matched_course"] and v
    ).strip()

    new_req = " ".join(
        str(v).strip()
        for k, v in sorted((new_requirements or {}).items())
        if k not in ["matched_course"] and v
    ).strip()

    old_content = str(old_requirements.get("content") or "").strip()
    new_content = str((new_requirements or {}).get("content") or "").strip()
    print(f"req : {new_requirements}", flush=True)

    print(
        f"REQ CHECK | old_req={old_req} | new_req={new_req}",
        flush=True
    )

    req_changed = (
        old_req
        and new_req
        and old_req.lower() != new_req.lower()
    )

    req_same = (
        old_req
        and new_req
        and old_req.lower() == new_req.lower()
    )

    has_cached_rag = bool(getattr(state, "matched_rag_results", None))

    if req_changed:
        print(f"[AI_CUSTOM] REQUIREMENT CHANGED: {old_req} -> {new_req}", flush=True)

        state.requirements = new_requirements or {}
        state.matched_rag_results = []
        state.active_course_no = None
        state.topic = new_content or getattr(state, "topic", "unknown")
        has_cached_rag = False
    else:
        state.requirements = new_requirements or {}

    print(
            f"REQ CHECK | old_content={old_content} | new_content={new_content}",
            flush=True
        )

    topic_changed = (
        old_content
        and new_content
        and old_content.lower() != new_content.lower()
    )

    topic_same = (
        old_content
        and new_content
        and old_content.lower() == new_content.lower()
    )

    has_cached_rag = bool(getattr(state, "matched_rag_results", None))

    if topic_changed:
        print(f"[AI_CUSTOM] TOPIC CHANGED: {old_content} -> {new_content}", flush=True)

        state.requirements = {
            "content": new_content,
            "goal": "",
            "event": "",
        }
        state.matched_rag_results = []
        state.active_course_no = None
        state.topic = new_content
        has_cached_rag = False
    else:
        state.requirements = new_requirements or {}

    missing = calc_missing_requirements(state.requirements)
    state.missing_requirements = missing
    state.requirement_ready = len(missing) == 0

    req_text = " ".join(
        str(v).strip()
        for v in (state.requirements or {}).values()
        if v
    ).strip()

    # =========================
    # 3) ตัดสินใจก่อนว่าจะ search RAG ไหม
    #    - content เดิม + มี cache => ใช้ข้อมูลเดิม ไม่ search
    #    - ยังไม่มี content / requirement ยังขาด + ไม่มี cache => ถาม requirement ต่อ ไม่ search
    #    - content ใหม่ / ยังไม่เคยมี cache => ค่อย search
    # =========================
    should_search_rag = False

    if req_changed:
        should_search_rag = True
    elif new_req and not has_cached_rag:
        should_search_rag = True
    elif req_same and has_cached_rag:
        should_search_rag = False
    elif not new_req:
        should_search_rag = False

    print(
        f"RAG DECISION | req_changed={req_changed} | req_same={req_same} | "
        f"has_cached_rag={has_cached_rag} | missing={missing} | should_search_rag={should_search_rag}",
        flush=True
    )

    rag_results = list(getattr(state, "matched_rag_results", []) or [])

    if should_search_rag:
        search_text = req_text

        try:
            rag_results = search_rag(
                user_message=search_text,
                course_nos=course_use,
                limit=5,
                score_threshold=0.20,
            )
        except Exception as e:
            print("[STREAM RAG ERROR TYPE]", type(e), flush=True)
            print("[STREAM RAG ERROR REPR]", repr(e), flush=True)

            reply = "ระบบค้นหาเนื้อหาขัดข้องชั่วคราวครับ"

            yield {"type": "chunk", "text": reply}
            yield {
                "type": "done",
                "reply": reply,
                "status": "error",
                "reason": "rag_exception",
                "state": state,
                "source": "qdrant_rag",
                "active_video": None,
            }
            return

        # state.matched_rag_results = rag_results or []
    else:
        rag_results = list(getattr(state, "matched_rag_results", []) or [])

    best = rag_results[0] if rag_results else None
    best_score = best.get("score") if best else 0

    if missing and not best:
        reply = ""

        # ถ้ามีรายชื่อ course ที่อนุญาต ให้ใช้เป็น hint ตอนถามหา content
        # เช่น user พิมพ์ "สวัสดี" แล้ว AI ควรบอกได้เล็กน้อยว่ามีความรู้เรื่องอะไรบ้าง
        if course_name_context:
            discovery_message = build_discovery_message_with_course_context(
                user_message=user_message,
                requirements=state.requirements,
                missing=missing,
                course_name_context=course_name_context,
            )

            async for item in reply_ask_concept_with_topic_stream(
                discovery_message,
                "หัวข้อที่เปิดให้เรียน",
                course_name_context,
            ):
                if item.get("type") == "chunk":
                    text = item.get("text", "")
                    reply += text
                    yield {"type": "chunk", "text": text}

                elif item.get("type") == "done":
                    reply = item.get("content") or reply

            answer_type = "ask_requirement_with_allowed_courses"
            source = "ai_custom_discovery_course_context"
        else:
            async for item in build_next_question(
                state.requirements,
                missing,
                state.conversation_history
            ):
                if item.get("type") == "chunk":
                    text = item.get("text", "")
                    reply += text
                    yield {"type": "chunk", "text": text}

                elif item.get("type") == "done":
                    reply = item.get("content") or reply

            answer_type = "ask_requirement_without_rag"
            source = "ai_custom_discovery"

        state.mode = "discovery"
        state.last_answer = reply
        state.last_intent = "collect_requirement"
        state.last_answer_type = answer_type

        state.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        if len(state.conversation_history) > 10:
            state.conversation_history = state.conversation_history[-10:]

        yield {
            "type": "done",
            "reply": reply,
            "status": "collecting_requirement",
            "reason": "missing_requirement_no_rag_search",
            "state": state,
            "source": source,
            "active_video": None,
        }
        return

    if should_search_rag and (not best or best_score < 0.35):

        print("[AI_CUSTOM] RAG NOT RELATED - ROLLBACK REQUIREMENTS", flush=True)

        state.requirements = old_requirements or {}
        state.missing_requirements = calc_missing_requirements(state.requirements)
        state.requirement_ready = len(state.missing_requirements) == 0
        state.matched_rag_results = []
        state.active_course_no = None
        state.topic = str((old_requirements or {}).get("content") or "unknown")
        reply = ""

        async for item in build_irrelevant_content_reply(
            user_message=user_message,
            requirements=state.requirements,
            conversation_history=state.conversation_history,
            course_name_context=course_name_context,
        ):
            if item.get("type") == "chunk":
                text = item.get("text", "")
                reply += text
                yield {"type": "chunk", "text": text}

            elif item.get("type") == "done":
                reply = item.get("content") or reply

        state.mode = "discovery"
        state.last_answer = reply
        state.last_intent = "irrelevant_content"
        state.last_answer_type = "irrelevant_content"

        state.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        if len(state.conversation_history) > 10:
            state.conversation_history = state.conversation_history[-10:]

        yield {
            "type": "done",
            "reply": reply,
            "status": "irrelevant",
            "reason": "rag_not_related",
            "state": state,
            "source": "ai_custom_irrelevant",
            "active_video": None,
        }
        return
        # ✅ ผ่านด่าน relevance แล้ว ค่อยเก็บ RAG cache
    if should_search_rag:
        state.matched_rag_results = rag_results or []

    # ตอบจากข้อมูลพร้อมถามต่อสิ่งที่ขาดอยู่
    if missing:
        # active_video = build_active_video_from_rag(best) if best else None

        matched_topic = (
            best.get("vdo_name")
            or best.get("course")
            or best.get("course_name")
            or state.requirements.get("content")
            or "เนื้อหาที่เกี่ยวข้อง"
        )

        rag_context = build_rag_context(rag_results)

        reply = ""

        brief_message = f"""
ข้อความผู้เรียน:
{user_message}

Requirement ปัจจุบัน:
{json.dumps(state.requirements, ensure_ascii=False)}

Requirement ที่ยังขาด:
{json.dumps(missing, ensure_ascii=False)}

RAG_CONTEXT:
{rag_context}

คำสั่ง:
- ตอบจาก RAG_CONTEXT เท่านั้น
- ถ้าเนื้อหาใน RAG_CONTEXT ตรงกับสิ่งที่ผู้เรียนถาม ให้สรุปคร่าว ๆ 2-4 ประโยค
- อย่าตอบจัดเต็ม
- ไม่ต้องบอกว่าตอบจากเนื้อหาอะไร
- ห้ามดึง topic เก่าที่ไม่เกี่ยวข้องมาตอบ
- หลังจากสรุปคร่าว ๆ แล้ว ให้ถามต่อ 1 คำถามเท่านั้น เพื่อเก็บ requirement ที่ยังขาด
""".strip()

        async for item in reply_ask_concept_with_topic_stream(
            brief_message,
            matched_topic,
            rag_context,
        ):
            if item.get("type") == "chunk":
                text = item.get("text", "")
                reply += text
                yield {"type": "chunk", "text": text}

            elif item.get("type") == "done":
                reply = item.get("content") or reply

        state.mode = "discovery"
        state.topic = state.requirements.get("content") or matched_topic
        state.active_course_no = best.get("course_no")
        state.last_answer = reply
        state.last_intent = "collect_requirement"
        state.last_answer_type = "rag_brief_then_ask_requirement"

        state.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        if len(state.conversation_history) > 10:
            state.conversation_history = state.conversation_history[-10:]

        yield {
            "type": "done",
            "reply": reply,
            "status": "collecting_requirement",
            "reason": "rag_brief_answer_then_missing_requirement",
            "state": state,
            "source": "ai_custom_requirement_rag_cached" if not should_search_rag else "ai_custom_requirement_rag",
            # "active_video": active_video,
        }
        return

    # =========================
    # 5) Requirement ครบแล้ว
    #    ค้นแบบจริงจังเฉพาะตอนจำเป็น
    #    - ถ้า content เดิมและมี matched_rag_results แล้ว ใช้ cache เดิม
    #    - ถ้า content ใหม่หรือยังไม่มี cache ค่อย search
    # =========================
    if req_same and getattr(state, "matched_rag_results", None):
        rag_results = list(state.matched_rag_results or [])
        state.search_query = getattr(state, "search_query", None)
        print("[AI_CUSTOM] USE CACHED RAG FOR COMPLETE REQUIREMENT", flush=True)
    else:
        search_query = await build_search_query(state.requirements)
        state.search_query = search_query

        try:
            rag_results = search_rag(
                user_message=search_query,
                course_nos=course_use,
                limit=5,
                score_threshold=0.20,
            )
        except Exception as e:
            print("[STREAM RAG ERROR TYPE]", type(e), flush=True)
            print("[STREAM RAG ERROR REPR]", repr(e), flush=True)

            reply = "ระบบค้นหาเนื้อหาขัดข้องชั่วคราวครับ"

            yield {"type": "chunk", "text": reply}
            yield {
                "type": "done",
                "reply": reply,
                "status": "error",
                "reason": "rag_exception_after_requirement_complete",
                "state": state,
                "source": "qdrant_rag",
                "active_video": None,
            }
            return

        state.matched_rag_results = rag_results or []

    if not rag_results:
        reply = "ตอนนี้ยังไม่พบเนื้อหาที่ตรงกับสิ่งที่ต้องการในหลักสูตรที่เปิดให้ใช้งานครับ ลองบอกหัวข้อหรือเป้าหมายให้เจาะจงขึ้นอีกนิดได้ไหมครับ"

        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "discover",
            "reason": "rag_not_found_after_requirement_complete",
            "state": state,
            "source": "qdrant_rag",
            "active_video": None,
        }
        return

    best = rag_results[0]
    best_score = best.get("score") or 0

    if best_score < 0.35:
        reply = "ผมเจอข้อมูลที่ใกล้เคียงอยู่บ้าง แต่ยังไม่มั่นใจว่าตรงกับสิ่งที่ต้องการครับ ช่วยระบุหัวข้อหรือเป้าหมายให้ชัดขึ้นอีกนิดได้ไหมครับ"

        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "discover",
            "reason": "rag_low_score_after_requirement_complete",
            "state": state,
            "source": "qdrant_rag_low_score",
            "active_video": None,
        }
        return

    # =========================
    # 6) ตอบจัดเต็มจาก RAG
    # =========================
    topic = best.get("vdo_name") or best.get("course") or "เนื้อหาที่เกี่ยวข้อง"
    rag_context = build_rag_context(rag_results)
    active_video = build_active_video_from_rag(best)

    rag_message = f"""
Requirement ผู้เรียน:
{json.dumps(state.requirements, ensure_ascii=False)}

คำถามล่าสุด:
{user_message}

คำสั่ง:
- ตอบจาก RAG_CONTEXT เท่านั้น
- ตอนนี้ requirement ครบแล้ว ให้ตอบแบบจัดเต็ม
- เชื่อมคำตอบกับ content / goal / event ของผู้เรียน
- ตอบภาษาไทย สุภาพ เข้าใจง่าย
- ถ้าไม่มีข้อมูลพอ ให้บอกว่าเนื้อหาไม่เพียงพอ
- ห้ามแต่งข้อมูลนอกเหนือจาก context

RAG_CONTEXT:
{rag_context}
""".strip()

    final_reply = ""

    async for item in reply_ask_concept_with_topic_stream(
        rag_message,
        topic,
        rag_context,
    ):
        if item["type"] == "chunk":
            text = item.get("text", "")
            final_reply += text
            yield {"type": "chunk", "text": text}

        elif item["type"] == "done":
            final_reply = item.get("content", final_reply)

    state.mode = "learning"
    state.topic = topic
    state.active_course_no = best.get("course_no")
    state.last_answer = final_reply
    state.last_intent = "rag_answer_full"
    state.last_answer_type = "requirement_complete_answer"

    state.conversation_history.append({
        "role": "assistant",
        "content": final_reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": final_reply,
        "status": "learning",
        "reason": "requirement_complete_rag_answer",
        "state": state,
        "source": "qdrant_rag_requirement",
        "active_video": active_video,
    }
    return