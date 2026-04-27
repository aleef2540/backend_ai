from app.modules.ai_sale.schema import AISaleState
from app.modules.ai_sale.service import (
    extract_requirements,
    calc_missing_requirements,
    build_next_question,
    build_search_query,
    build_recommendation_reply,
)
from app.modules.ai_sale.qdrant_service import search_courses_from_qdrant


async def process_ai_sale_stream(req, state):
    if state is None:
        state = AISaleState()

    user_message = (req.user_message or "").strip()

    state.web_no = int(req.web_no) if req.web_no not in [None, ""] else None
    state.member_no = int(req.member_no) if req.member_no not in [None, ""] else None
    state.last_user_message = user_message

    # ✅ เก็บข้อความ user เข้า history ก่อน
    state.conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # ✅ กัน history ยาวเกิน
    if len(state.conversation_history) > 20:
        state.conversation_history = state.conversation_history[-20:]

    state.requirements = await extract_requirements(
        user_message=user_message,
        current_requirements=state.requirements or {},
        conversation_history=state.conversation_history
    )

    missing = calc_missing_requirements(state.requirements)

    state.missing_requirements = missing
    state.requirement_ready = len(missing) == 0

    if missing:
        reply = await build_next_question(
            state.requirements,
            missing,
            state.conversation_history
        )

        state.mode = "discovery"
        state.last_answer = reply
        state.last_step = "ask_requirement"

        # ✅ เก็บคำตอบ AI เข้า history
        state.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        if len(state.conversation_history) > 20:
            state.conversation_history = state.conversation_history[-20:]

        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "collecting_requirement",
            "reason": "missing_requirement",
            "state": state,
            "source": "ai_sale_discovery",
        }
        return

    search_query = await build_search_query(state.requirements)
    state.search_query = search_query

    courses = await search_courses_from_qdrant(search_query, limit=5)
    state.recommended_courses = courses

    reply = await build_recommendation_reply(
        requirements=state.requirements,
        courses=courses
    )

    state.mode = "recommend"
    state.last_answer = reply
    state.last_step = "recommend_course"

    # ✅ เก็บคำตอบ AI เข้า history
    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 20:
        state.conversation_history = state.conversation_history[-20:]

    yield {"type": "chunk", "text": reply}
    yield {
        "type": "done",
        "reply": reply,
        "status": "recommended",
        "reason": "requirement_complete",
        "state": state,
        "source": "ai_sale_qdrant",
    }