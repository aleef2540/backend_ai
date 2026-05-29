from app.modules.ai_expert.schema import ChatState_aiexpert
from app.modules.ai_expert.service import reply_ai_expert_direct_stream


def build_conversation_context(state: ChatState_aiexpert, limit: int = 10) -> str:
    lines = []

    history = getattr(state, "conversation_history", []) or []

    for item in history[-limit:]:
        role = item.get("role", "")
        content = str(item.get("content") or "").strip()

        if role and content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


async def process_chat_aiexpert_stream(req, state):
    if state is None:
        state = ChatState_aiexpert()

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

    state.last_user_message = user_message

    state.conversation_history.append({
        "role": "user",
        "content": user_message,
    })

    if len(state.conversation_history) > 20:
        state.conversation_history = state.conversation_history[-20:]

    conversation_context = build_conversation_context(state)

    final_reply = ""

    try:
        async for item in reply_ai_expert_direct_stream(
            user_message=user_message,
            conversation_context=conversation_context,
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

    state.mode = "expert"
    state.intent = "consulting"
    state.last_intent = "expert_consulting"
    state.last_answer_type = "openai_direct"
    state.last_answer = final_reply

    state.conversation_history.append({
        "role": "assistant",
        "content": final_reply,
    })

    if len(state.conversation_history) > 20:
        state.conversation_history = state.conversation_history[-20:]

    yield {
        "type": "done",
        "reply": final_reply,
        "status": "success",
        "reason": "openai_direct_answer",
        "state": state,
        "source": "ai_expert_openai",
        "active_video": None,
    }