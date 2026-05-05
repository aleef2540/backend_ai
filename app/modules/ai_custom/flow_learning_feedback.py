from app.modules.ai_custom.service import (
    detect_feedback_intent_ai,
    reply_learning_feedback_stream,
)

from app.modules.ai_custom.rag_service import build_rag_context

from app.modules.ai_custom.flow_helpers_learning_feedback import (
    append_history,
    reset_to_discovery,
    update_learning_feedback_state,
    build_unrelated_feedback_reply,
)

FEEDBACK_INTENTS = {
    "report_done",
    "report_partial",
    "report_not_done",
    "blocked",
    "review_request",
}


async def handle_learning_feedback_flow(user_message: str, state):
    print(f"mode : {state.mode}", flush=True)

    intent_result = await detect_feedback_intent_ai(user_message, state)
    feedback_intent = intent_result.get("intent", "general_feedback")

    print("[LEARNING/FEEDBACK INTENT]", intent_result, flush=True)

    # -------------------------
    # 1) ผู้ใช้ขอเริ่มหัวข้อใหม่
    # -------------------------
    if feedback_intent == "restart_learning":
        print(f"mode : {state.mode} | new topic", flush=True)

        reset_to_discovery(state, clear_learning=True)

        state.last_intent = "restart_learning"
        state.last_answer_type = "restart_learning"

        reply = "ได้ครับ เรามาเริ่มหัวข้อใหม่กัน คุณอยากเรียนหรือพัฒนาเรื่องอะไรเป็นพิเศษครับ?"

        state.last_answer = reply
        append_history(state, "assistant", reply)

        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "state": state,
            "source": "ai_custom_learning_feedback",
            "status": "collecting_requirement",
            "reason": "restart_learning_requested",
            "active_video": None,
            "intent_result": intent_result,
        }
        return

    # -------------------------
    # 2) คำถามไม่เกี่ยวกับ learning phase เดิม
    # -------------------------
    if feedback_intent == "unrelated":
        print(f"mode : {state.mode} | out of topic", flush=True)

        reply = build_unrelated_feedback_reply(state)

        state.last_answer = reply
        state.last_intent = "learning_unrelated"
        state.last_answer_type = "learning_unrelated_redirect"

        append_history(state, "assistant", reply)

        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "state": state,
            "source": "ai_custom_learning_feedback",
            "status": state.mode,
            "reason": "unrelated_to_learning_phase",
            "active_video": None,
            "intent_result": intent_result,
        }
        return

    # -------------------------
    # 3) learning / feedback ต่อใน journey เดิม
    # -------------------------
    if feedback_intent in FEEDBACK_INTENTS:
        next_mode = "feedback"
        print(f"mode : {state.mode} | feedback flow | intent={feedback_intent}", flush=True)
    else:
        next_mode = "learning"
        print(f"mode : {state.mode} | learning flow | intent={feedback_intent}", flush=True)

    rag_results = getattr(state, "matched_rag_results", []) or []
    rag_context = build_rag_context(rag_results) if rag_results else ""

    final_reply = ""

    async for item in reply_learning_feedback_stream(
        user_message=user_message,
        state=state,
        feedback_intent=feedback_intent,
        rag_context=rag_context,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            final_reply += text
            yield {"type": "chunk", "text": text}

        elif item.get("type") == "done":
            final_reply = item.get("content") or final_reply

    state = update_learning_feedback_state(
        state=state,
        user_message=user_message,
        reply=final_reply,
        feedback_intent=feedback_intent,
    )

    state.mode = next_mode
    state.last_answer = final_reply
    state.last_intent = feedback_intent
    state.last_answer_type = f"{next_mode}_{feedback_intent}"

    append_history(state, "assistant", final_reply)

    yield {
        "type": "done",
        "reply": final_reply,
        "state": state,
        "source": f"ai_custom_{next_mode}",
        "status": next_mode,
        "reason": feedback_intent,
        "active_video": None,
        "intent_result": intent_result,
    }

    return