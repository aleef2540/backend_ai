from app.modules.ai_assis.schema import AIAssisState
from app.modules.ai_assis.service import (
    detect_intent,
    handle_company_profile,
    handle_credibility,
    handle_contact,
    handle_irrelevant,
    handle_general_qa,

)
from app.modules.ai_assis.handler.course_handler import (
handle_course_search
)
from app.modules.ai_assis.handler.instructor_handler import handle_instructor_search
from app.modules.ai_assis.handler.quotation_handler import handle_quotation

from app.modules.ai_assis.qdrant_service import search_courses_from_qdrant, check_topic_exists_in_qdrant
import json

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

async def process_ai_assistant_stream(req, state):

    if state is None or not isinstance(state, AIAssisState):
        state = AIAssisState()

    user_message = (req.user_message or "").strip()
    from_web = (req.from_web or "").strip()

    state.last_user_message = user_message
    state.from_web = from_web

    state.conversation_history.append({
        "role": "user",
        "content": user_message
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    print("detect_intent", flush=True)

    try:

        intent_result = await detect_intent(
            user_message=user_message,
            state=state,
            conversation_history=state.conversation_history
        )

        print("AI ASSISTANT INTENT =", intent_result, flush=True)

        if not state.course_context:
            state.course_context = {}
        
        if not state.instructor_context:
            state.instructor_context = {}

        state.current_intent = intent_result.get(
            "intent",
            "general_qa"
        )

        state.course_context["course_type"] = intent_result.get(
            "course_type",
            "unknown"
        )

        state.course_context["course_action"] = intent_result.get(
            "course_action",
            "unknown"
        )

        state.instructor_context["instructor_action"] = intent_result.get(
            "instructor_action",
            "unknown"
        )

        state.course_context["topic"] = intent_result.get(
            "topic",
            ""
        )

        intent = intent_result.get("intent", "general_qa")

        course_type = intent_result.get("course_type", "unknown")

        if intent in ["course_search", "quotation"]:
            if not state.course_context:
                state.course_context = {}

            if course_type and course_type != "unknown":
                state.course_context["course_type"] = course_type


        state.previous_intent = state.current_intent
        state.current_intent = intent
        state.last_step = intent

        # =========================
        # COURSE SEARCH
        # =========================
        if intent == "course_search":

            async for item in handle_course_search(
                req=req,
                state=state
            ):

                yield item

            return

        # =========================
        # INSTRUCTOR SEARCH
        # =========================
        elif intent == "instructor_search":

            async for item in handle_instructor_search(
                req=req,
                state=state
            ):

                yield item

            return

        # =========================
        # COMPANY PROFILE
        # =========================
        elif intent == "company_profile":

            async for item in handle_company_profile(
                req=req,
                state=state
            ):

                yield item

            return

        # =========================
        # CREDIBILITY
        # =========================
        elif intent == "credibility":

            async for item in handle_credibility(
                req=req,
                state=state
            ):

                yield item

            return

        # =========================
        # QUOTATION
        # =========================
        elif intent == "quotation":

            async for item in handle_quotation(
                req=req,
                state=state
            ):

                yield item

            return

        # =========================
        # CONTACT
        # =========================
        elif intent == "contact":

            async for item in handle_contact(
                req=req,
                state=state
            ):

                yield item

            return
        
        elif intent == "irrelevant":

            async for item in handle_irrelevant(
                req=req,
                state=state
            ):

                yield item

            return

        # =========================
        # GENERAL QA
        # =========================
        else:
            async for item in handle_general_qa(
                req=req,
                state=state
            ):

                yield item

            return

    except Exception as e:

        print("AI ASSISTANT ERROR =", str(e), flush=True)

        reply = (
            "ขออภัยครับ ระบบ AI Assistant เกิดข้อผิดพลาดชั่วคราว "
            "กรุณาลองใหม่อีกครั้ง"
        )

        state.last_answer = reply
        state.last_step = "error"

        state.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        if len(state.conversation_history) > 10:
            state.conversation_history = state.conversation_history[-10:]

        yield {
            "type": "done",
            "reply": reply,
            "status": "error",
            "reason": "exception",
            "state": state,
            "source": "ai_assistant",
        }

        return