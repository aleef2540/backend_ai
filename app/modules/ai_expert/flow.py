from app.modules.ai_expert.schema import ChatState_aiexpert
from app.modules.ai_expert.service import (
    reply_ai_expert_direct_stream,
    evaluate_coaching_state_before_reply,
    merge_extracted_context,
)


VALID_STATES = {"OPENING", "GOAL", "REALITY", "OPTIONS", "WILL", "SUMMARY", "COMPLETED"}
STATE_ALIASES = {
    "INTRO": "OPENING",
    "TOPIC": "GOAL",
}


def normalize_coaching_state(value: str | None) -> str:
    state = (value or "OPENING").upper()
    state = STATE_ALIASES.get(state, state)
    return state if state in VALID_STATES else "OPENING"


def build_conversation_context(state: ChatState_aiexpert, limit: int = 10) -> str:
    lines = []
    history = getattr(state, "conversation_history", []) or []

    for item in history[-limit:]:
        role = item.get("role", "")
        content = str(item.get("content") or "").strip()

        if role and content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def get_last_bot_message(state: ChatState_aiexpert) -> str:
    history = getattr(state, "conversation_history", []) or []
    for item in reversed(history):
        if item.get("role") == "assistant":
            return item.get("content", "") or ""
    return ""


def update_repeated_question_state(state: ChatState_aiexpert, new_question_type: str | None):
    new_question_type = (new_question_type or "other").strip().lower()
    old_question_type = (getattr(state, "last_question_type", None) or "").strip().lower()

    if old_question_type and old_question_type == new_question_type:
        state.repeated_question_count = int(getattr(state, "repeated_question_count", 0) or 0) + 1
    else:
        state.repeated_question_count = 0

    state.last_question_type = new_question_type


async def process_chat_aiexpert_stream(req, state):
    if state is None:
        state = ChatState_aiexpert()

    state.current_coaching_state = normalize_coaching_state(
        getattr(state, "current_coaching_state", None)
    )

    user_message = (req.user_message or "").strip()

    state.web_no = int(req.web_no) if req.web_no not in [None, ""] else None
    state.member_no = int(req.member_no) if req.member_no not in [None, ""] else None

    state.course_use = [
        str(x).strip()
        for x in (req.course_use or [])
        if str(x).strip()
    ]

    if not user_message:
        reply = "กรุณาเล่าประเด็นที่อยากปรึกษาเกี่ยวกับการพัฒนาคน ทีม หรือการทำงานครับ"
        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "empty_message",
            "reason": "user_message_empty",
            "state": state,
            "source": "ai_expert",
            "active_video": None,
        }
        return

    if not hasattr(state, "conversation_history") or state.conversation_history is None:
        state.conversation_history = []

    if not hasattr(state, "extracted_context") or state.extracted_context is None:
        state.extracted_context = ChatState_aiexpert().extracted_context

    state.turn_count = int(getattr(state, "turn_count", 0) or 0) + 1

    last_bot_message = get_last_bot_message(state)

    # PRE-EVALUATION: ประเมิน state/clarity ก่อนตอบ เพื่อกันถามวนและข้าม phase ที่ชัดแล้ว
    eval_result = None
    if state.current_coaching_state not in ["COMPLETED"]:
        try:
            eval_result = await evaluate_coaching_state_before_reply(
                current_state=state.current_coaching_state,
                last_bot_message=last_bot_message,
                user_message=user_message,
                conversation_context=build_conversation_context(state, limit=8),
                extracted_context=state.extracted_context,
                turn_count=state.turn_count,
            )

            state.goal_clarity = int(eval_result.get("goal_clarity", state.goal_clarity) or 0)
            state.reality_clarity = int(eval_result.get("reality_clarity", state.reality_clarity) or 0)
            state.options_count = int(eval_result.get("options_count", state.options_count) or 0)
            state.commitment_clarity = int(eval_result.get("commitment_clarity", state.commitment_clarity) or 0)

            next_state = normalize_coaching_state(
                eval_result.get("recommended_next_state", state.current_coaching_state)
            )
            state.current_coaching_state = next_state

            state.coach_move = eval_result.get("recommended_coach_move") or state.coach_move
            update_repeated_question_state(state, eval_result.get("last_question_type"))

            extracted_patch = eval_result.get("extracted_patch") or {}
            state.extracted_context = merge_extracted_context(
                state.extracted_context,
                extracted_patch,
            )

        except Exception as eval_err:
            print(f"[PRE-EVAL ERROR] Skipping transition check: {eval_err}")
            eval_result = None

    # บันทึกข้อความผู้เรียนหลังประเมิน เพื่อให้ context ตอบรอบนี้มีข้อความล่าสุดด้วย
    state.last_user_message = user_message
    state.conversation_history.append({
        "role": "user",
        "content": user_message,
    })

    if len(state.conversation_history) > 24:
        state.conversation_history = state.conversation_history[-24:]

    conversation_context = build_conversation_context(state)
    final_reply = ""

    try:
        async for item in reply_ai_expert_direct_stream(
            user_message=user_message,
            conversation_context=conversation_context,
            current_state=state.current_coaching_state,
            extracted_context=state.extracted_context,
            coach_move=state.coach_move,
            goal_clarity=state.goal_clarity,
            reality_clarity=state.reality_clarity,
            options_count=state.options_count,
            commitment_clarity=state.commitment_clarity,
            needs_scaffold=bool((eval_result or {}).get("needs_scaffold", False)),
            repeated_question_count=int(getattr(state, "repeated_question_count", 0) or 0),
        ):
            if item.get("type") == "chunk":
                text = item.get("text", "")
                if text:
                    final_reply += text
                    yield {
                        "type": "chunk",
                        "text": text,
                    }

            elif item.get("type") == "done":
                final_reply = item.get("content") or final_reply
                break

    except Exception as e:
        reply = "ขออภัยครับ ระบบ AI Expert ขัดข้องชั่วคราวครับ"
        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "error",
            "reason": str(e),
            "state": state,
            "source": "ai_expert_error",
            "active_video": None,
        }
        return

    # ถ้าสรุปแล้ว และ commitment ชัดพอ รอบถัดไปให้ถือว่า completed
    if state.current_coaching_state == "SUMMARY" and state.commitment_clarity >= 90:
        state.current_coaching_state = "COMPLETED"

    state.mode = "expert"
    state.intent = "consulting"
    state.last_intent = "expert_consulting"
    state.last_answer_type = "openai_direct"
    state.last_answer = final_reply

    state.conversation_history.append({
        "role": "assistant",
        "content": final_reply,
    })

    if len(state.conversation_history) > 24:
        state.conversation_history = state.conversation_history[-24:]

    yield {
        "type": "done",
        "reply": final_reply,
        "status": "success",
        "reason": "openai_direct_answer",
        "state": state,
        "source": "ai_expert_openai",
        "active_video": None,
    }
