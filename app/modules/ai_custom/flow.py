from app.modules.ai_custom.course_service import get_course_data_by_nos_bridge
import json

from app.modules.ai_custom.service import (
    reply_discovery_with_course_context_stream,
    extract_requirements,
    calc_missing_requirements,
    build_next_question,
    build_search_query,
    reply_ask_concept_with_topic_stream,
    build_irrelevant_content_reply,
    build_rag_query_with_llm,
    filter_rag_results_by_relevance,
    build_next_question_after_no_rag,
    reply_ask_concept_with_topic_stream_new,

)

from app.modules.ai_custom.flow_learning_feedback import handle_learning_feedback_flow


from app.modules.ai_custom.schema import ChatState_aicustom
from app.modules.ai_custom.rag_service import search_rag, build_rag_context, build_active_video_from_rag

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


def build_learning_journey_name(requirements: dict, topic: str = "", existing_name: str | None = None) -> str:
    """
    สร้างชื่อ learning journey สำหรับใช้เป็นชื่อ chat history / room

    priority:
    1. existing_name ถ้ามีอยู่แล้ว
    2. goal เพราะมักเป็นชื่อ journey ที่ดีที่สุด
    3. content + event
    4. content
    5. topic
    """

    existing_name = str(existing_name or "").strip()
    if existing_name:
        return existing_name

    requirements = requirements or {}

    content = str(requirements.get("content") or "").strip()
    goal = str(requirements.get("goal") or "").strip()
    event = str(requirements.get("event") or "").strip()
    topic = str(topic or "").strip()

    if goal:
        return goal

    if content and event:
        return f"{content} - {event}"

    if content:
        return f"เรียนรู้เรื่อง{content}"

    if topic:
        return f"เรียนรู้เรื่อง{topic}"

    return "Learning Journey"


async def process_chat_aicustom_stream(req, state):
    
    # ถ้าไม่มี state ส่งมาจะสร้าง state ใหม่
    if state is None:
        state = ChatState_aicustom()
    
    print("[FLOW START]", {
    "mode": getattr(state, "mode", None),
    "topic": getattr(state, "topic", None),
    "has_rag": bool(getattr(state, "matched_rag_results", None)),
    "has_learning_phase": bool(getattr(state, "learning_phase", None)),
    }, flush=True)

    # แกะข้อมูลจาก req 
    user_message = (req.user_message or "").strip()
    state.web_no = int(req.web_no) if req.web_no not in [None, ""] else None
    state.member_no = int(req.member_no) if req.member_no not in [None, ""] else None
    if req.course_use:
        state.course_use = [int(x) for x in req.course_use if str(x).strip()]

    # เช็คว่ามี course_use ไหม
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
    # mode learning / feedback
    # หลังจากเก็บข้อมูลครบและแนะนำสิ่งที่ต้องทำไปแล้ว
    # =========================
    if state.mode in ["learning", "feedback"]:
        async for item in handle_learning_feedback_flow(user_message, state):
            yield item
        return

    # =========================
    # mode Discovery เก็บ requirements ตามที่ตั้งค่าใว้ จัดการเรื่อง new old requirements เพื่อจัดการ state
    # 2) Extract requirement ก่อน
    # =========================
    old_requirements = dict(getattr(state, "requirements", {}) or {})

    new_requirements = await extract_requirements(
        user_message=user_message,
        current_requirements=old_requirements,
        conversation_history=state.conversation_history,
    )

    old_content = str(old_requirements.get("content") or "").strip()
    new_content = str((new_requirements or {}).get("content") or "").strip()

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

    req_changed = bool(
        old_req
        and new_req
        and old_req.lower() != new_req.lower()
    )

    req_same = bool(
        old_req
        and new_req
        and old_req.lower() == new_req.lower()
    )

    content_changed = bool(
        old_content
        and new_content
        and old_content.lower() != new_content.lower()
    )

    print(f"req : {new_requirements}", flush=True)
    print(
        f"REQ CHECK | old_req={old_req} | new_req={new_req} | "
        f"content_changed={content_changed}",
        flush=True,
    )

    # update requirements ครั้งเดียวพอ
    state.requirements = new_requirements or {}

    # sync topic ให้เท่ากับ content เพราะคุณใช้เป็นตัวเดียวกัน
    if new_content:
        state.topic = new_content

    # ล้าง RAG cache เฉพาะตอน content/topic หลักเปลี่ยน
    if content_changed:
        print(f"[AI_CUSTOM] CONTENT CHANGED: {old_content} -> {new_content}", flush=True)

        state.matched_rag_results = []
        state.active_course_no = None

    has_cached_rag = bool(getattr(state, "matched_rag_results", None))

    # หาว่า requirements ขาดอะไรบ้าง
    missing = calc_missing_requirements(state.requirements)
    state.missing_requirements = missing
    state.requirement_ready = len(missing) == 0

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

    # search จาก rag
    if should_search_rag:
        search_text = await build_rag_query_with_llm(
            requirements=state.requirements or {},
            user_message=user_message,
            conversation_history=getattr(state, "conversation_history", None),
        )

        print(f"RAG SEARCH QUERY | {search_text}", flush=True)

        try:
            rag_results = search_rag(
                user_message=search_text,
                course_nos=course_use,
                limit=8,
                score_threshold=0.15,
            )

            rag_results = await filter_rag_results_by_relevance(
                user_message=user_message,
                requirements=state.requirements,
                rag_results=rag_results,
                limit=5,
            )

            print(
                f"RAG RELEVANCE FILTER | kept={len(rag_results or [])}",
                flush=True
            )

            # ไม่มีหลักสูตรที่เกี่ยวข้อง
            if not rag_results:
                # กรณี requirement ยังไม่ครบ แต่ user พูดเรื่อง learning ถูกทาง
                # ไม่ต้อง reject ให้ถาม requirement ถัดไปแทน
                if missing:
                    reply = ""

                    async for item in build_next_question_after_no_rag(
                        requirements=state.requirements,
                        missing=missing,
                        conversation_history=state.conversation_history,
                    ):
                        if isinstance(item, dict):
                            if item.get("type") == "chunk":
                                text = item.get("text", "")
                                reply += text
                                yield {"type": "chunk", "text": text}
                            elif item.get("type") == "done":
                                reply = item.get("content") or reply
                        else:
                            text = str(item)
                            reply += text
                            yield {"type": "chunk", "text": text}

                    state.mode = "discovery"
                    state.last_answer = reply
                    state.last_intent = "collect_requirement"
                    state.last_answer_type = "ask_requirement_after_no_rag"

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
                        "reason": "rag_empty_but_requirement_missing",
                        "state": state,
                        "source": "ai_custom_discovery",
                        "active_video": None,
                    }
                    return

                # กรณี requirement ครบแล้ว แต่ RAG ไม่เจอจริง ๆ
                # แปลว่า requirement ชุดนี้ไม่ตรงกับหลักสูตรที่เปิดให้ใช้
                # ต้อง reset requirement เพื่อไม่ให้วน search ด้วย req เดิมซ้ำ

                state.mode = "discovery"
                state.intent = "unknown"
                state.topic = "unknown"
                state.active_course_no = None

                state.requirements = {}
                state.missing_requirements = calc_missing_requirements({})
                state.requirement_ready = False

                state.search_query = None
                state.matched_rag_results = []

                reply = (
                    "ขออภัยครับ ตอนนี้ยังไม่พบเนื้อหาในระบบ Self-Learning "
                    "ที่ตรงกับความต้องการนี้ในหลักสูตรที่เปิดให้ใช้งานครับ "
                )

                if course_name_context:
                    reply += (
                        f"หัวข้อที่ระบบมีตอนนี้ เช่น {course_name_context} "
                        "คุณอยากเรียนหรือพัฒนาเรื่องไหนจากหัวข้อเหล่านี้ครับ?"
                    )
                else:
                    reply += "คุณอยากลองบอกหัวข้อใหม่ที่ต้องการเรียนหรือพัฒนาไหมครับ?"

                state.last_answer = reply
                state.last_intent = "course_not_matched"
                state.last_answer_type = "reset_requirement_after_relevance_empty"

                state.conversation_history.append({
                    "role": "assistant",
                    "content": reply
                })

                if len(state.conversation_history) > 10:
                    state.conversation_history = state.conversation_history[-10:]

                yield {"type": "chunk", "text": reply}
                yield {
                    "type": "done",
                    "reply": reply,
                    "status": "collecting_requirement",
                    "reason": "rag_relevance_filter_empty_requirement_reset",
                    "state": state,
                    "source": "ai_custom_rag_relevance_filter",
                    "active_video": None,
                }
                return
            
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

            async for item in reply_discovery_with_course_context_stream(
                user_message=user_message,
                requirements=state.requirements,
                missing=missing,
                course_name_context=course_name_context,
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

        # rag_context = build_rag_context(rag_results)
        rag_context = build_rag_context((rag_results or [])[:2])

        reply = ""

        async for item in reply_ask_concept_with_topic_stream_new(
            user_message=user_message,
            topic=matched_topic,
            rag_context=rag_context,
            requirements=state.requirements,
            missing=missing,
            mode="brief_then_ask_requirement",
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
        search_query = await build_rag_query_with_llm(
            requirements=state.requirements or {},
            user_message=user_message,
            conversation_history=getattr(state, "conversation_history", None),
        )

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

    best = best or {}

    topic = best.get("vdo_name") or best.get("course") or "เนื้อหาที่เกี่ยวข้อง"
    rag_context = build_rag_context(rag_results)
    active_video = build_active_video_from_rag(best)

    if not rag_context.strip():
        final_reply = "ตอนนี้เนื้อหาไม่เพียงพอสำหรับตอบคำถามนี้ครับ"

        state.mode = "discovery"
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
            "reason": "insufficient_rag_context",
            "state": state,
            "source": "qdrant_rag_requirement",
            "active_video": active_video,
        }
        return


    final_reply = ""

    async for item in reply_ask_concept_with_topic_stream(
        user_message=user_message,
        topic=topic,
        rag_context=rag_context,
        requirements=state.requirements,
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

    journey_name = build_learning_journey_name(
        requirements=state.requirements,
        topic=topic,
        existing_name=getattr(state, "journey_name", None),
    )

    state.journey_name = journey_name
    state.learning_phase = {
        "phase": "learning",
        "journey_name": journey_name,
        "status": "active_learning",

        "topic": topic,
        "course_no": best.get("course_no"),
        "active_video": active_video,

        "requirements": dict(state.requirements or {}),

        "ai_recommendation_text": final_reply,

        "feedback_status": "not_started",
        "checkin_question": "ถ้าคุณได้นำแนวทางนี้ไปลองใช้แล้ว กลับมาเล่าได้เลยครับว่าทำอะไรไปบ้าง ผลเป็นอย่างไร และติดจุดไหน",

        "recommended_actions": [],
        "feedback_history": []
    }

    state.feedback_status = "not_started"

    state.feedback_status = "waiting_checkin"


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