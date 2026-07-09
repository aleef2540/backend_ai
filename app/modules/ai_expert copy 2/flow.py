from app.modules.ai_expert.schema import ChatState_aiexpert
from app.modules.ai_expert.service import (
    reply_ai_expert_direct_stream, 
    evaluate_coaching_state_before_reply
)


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
    print(f"state : {state.current_coaching_state}")
    if state is None:
        state = ChatState_aiexpert()
    print(f"state : {state.current_coaching_state}")
    # ตั้งต้นสเตจเริ่มต้นหากเป็นแชตใหม่แกะกล่อง
    if not hasattr(state, "current_coaching_state") or not state.current_coaching_state:
        state.current_coaching_state = "INTRO"
    
    print(f"state : {state.current_coaching_state}")

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

    # ดึงคำตอบ/คำถามครั้งล่าสุดของบอตตลับก่อนหน้า มาประกบตรวจคู่กับข้อความผู้เรียนรอบนี้
    last_bot_message = ""
    if state.conversation_history:
        for item in reversed(state.conversation_history):
            if item.get("role") == "assistant":
                last_bot_message = item.get("content", "")
                break

    # --- 🚨 ย้ายจุดที่ 3 ขึ้นมาทำตรงนี้: PRE-EVALUATION 🚨 ---
    # จะทำการตรวจวิเคราะห์ข้อความขาเข้าทันที ก่อนส่งไปหาบอตคู่ใจ
    if state.current_coaching_state not in ["COMPLETED"]:
        try:
            eval_result = await evaluate_coaching_state_before_reply(
                current_state=state.current_coaching_state,
                last_bot_message=last_bot_message,
                user_message=user_message,
                conversation_context=build_conversation_context(state, limit=6)
            )
            
            # หากประเมินแล้วคำตอบของผู้เรียนสอบผ่านเกณฑ์ของสเตจเดิม
            if eval_result.get("is_state_achieved") is True:
                # 1. ขยับสถานะในโมเดลไปสเตจถัดไปทันที
                next_state = eval_result.get("suggested_next_state", state.current_coaching_state)
                state.current_coaching_state = next_state.upper()
                
                # 2. ป้อนข้อมูลสำคัญที่สกัดได้ลงขวด Context Pool
                extracted_data = eval_result.get("extracted_data")
                if extracted_data:
                    state_key_map = {"INTRO": "intro","TOPIC": "topic","GOAL": "goal",  "REALITY": "reality", "OPTIONS": "options", "WILL": "will_action"}
                    mapped_key = state_key_map.get(eval_result.get("evaluated_state", "").upper())
                    if mapped_key and hasattr(state, "extracted_context"):
                        state.extracted_context[mapped_key] = extracted_data
                        
        except Exception as eval_err:
            print(f"[PRE-EVAL ERROR] Skipping transition check: {eval_err}")

    # บันทึกประวัติข้อความของฝั่งผู้เรียนลงระบบตามปกติ
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
        # --- จุดที่ 2: ส่ง current_coaching_state ที่อัปเดตเรียบร้อยสด ๆ ร้อน ๆ ไปเจนคำตอบ ---
        async for item in reply_ai_expert_direct_stream(
            user_message=user_message,
            conversation_context=conversation_context,
            current_state=state.current_coaching_state,
            extracted_context=state.extracted_context
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

    # ล็อกแชตหน้าบ้านทันทีหากไต่สเตจจนถึงความสำเร็จสูงสุดของการโค้ช
    if state.current_coaching_state == "COMPLETED":
        final_reply += "\n\n🎉 ยอดเยี่ยมมากครับ! แผนงานนี้ถูกบันทึกเรียบร้อย ขอให้สนุกกับการทดลองลุยจริงหน้างาน 70% นะครับ!"

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