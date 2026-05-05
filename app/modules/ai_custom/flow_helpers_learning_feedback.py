from datetime import datetime

def append_history(state, role: str, content: str, limit: int = 10):
    if not hasattr(state, "conversation_history") or state.conversation_history is None:
        state.conversation_history = []

    state.conversation_history.append({
        "role": role,
        "content": content,
    })

    if len(state.conversation_history) > limit:
        state.conversation_history = state.conversation_history[-limit:]

    return state

def reset_to_discovery(state, clear_learning: bool = True):
    state.mode = "discovery"
    state.intent = "unknown"
    state.topic = "unknown"
    state.active_course_no = None

    state.requirements = {}
    state.missing_requirements = []
    state.requirement_ready = False

    state.search_query = None
    state.matched_rag_results = []

    if clear_learning:
        state.learning_phase = {}
        state.feedback_status = None

    return state

def update_learning_feedback_state(state, user_message: str, reply: str, feedback_intent: str):
    now = datetime.utcnow().isoformat()

    if not getattr(state, "learning_phase", None):
        state.learning_phase = {}

    learning_phase = state.learning_phase or {}

    feedback_history = learning_phase.get("feedback_history") or []

    feedback_record = {
        "intent": feedback_intent,
        "user_message": user_message,
        "assistant_reply": reply,
        "created_at": now,
    }

    if feedback_intent == "report_done":
        feedback_record["completion_status"] = "done"
        learning_phase["status"] = "user_reported_done"
        state.feedback_status = "user_reported_done"

    elif feedback_intent == "report_partial":
        feedback_record["completion_status"] = "partial"
        learning_phase["status"] = "user_reported_partial"
        state.feedback_status = "user_reported_partial"

    elif feedback_intent == "report_not_done":
        feedback_record["completion_status"] = "not_done"
        learning_phase["status"] = "user_not_started"
        state.feedback_status = "user_not_started"

    elif feedback_intent == "blocked":
        feedback_record["completion_status"] = "blocked"
        learning_phase["status"] = "user_blocked"
        state.feedback_status = "user_blocked"

    elif feedback_intent == "review_request":
        learning_phase["status"] = "review_requested"
        state.feedback_status = "review_requested"

    else:
        learning_phase["status"] = "active_learning"
        state.feedback_status = learning_phase.get("feedback_status") or "not_started"

    feedback_history.append(feedback_record)

    # กัน history ยาวเกินไป
    learning_phase["feedback_history"] = feedback_history[-20:]
    learning_phase["last_feedback"] = feedback_record
    learning_phase["updated_at"] = now

    state.learning_phase = learning_phase

    return state

def build_unrelated_feedback_reply(state) -> str:
    learning_phase = getattr(state, "learning_phase", {}) or {}
    requirements = learning_phase.get("requirements") or getattr(state, "requirements", {}) or {}

    topic = learning_phase.get("topic") or getattr(state, "topic", "หัวข้อที่กำลังเรียน")
    content = requirements.get("content") or topic

    return f"""
ตอนนี้ห้องนี้อยู่ในโหมด learning feedback เรื่อง “{content}” ครับ

คำถามล่าสุดดูเหมือนยังไม่เกี่ยวกับหัวข้อที่เรากำลังเรียนอยู่โดยตรง ผมจึงยังไม่เปลี่ยนหัวข้อให้อัตโนมัติ เพื่อไม่ให้แผนการเรียนรู้เดิมหลุดบริบทครับ

คุณสามารถคุยต่อในเรื่องนี้ได้ เช่น:
- ขอให้ผมอธิบายเพิ่ม
- ขอตัวอย่างการนำไปใช้
- เล่าว่าคุณได้ลองทำแล้วหรือยัง
- บอกจุดที่ติดปัญหา

ถ้าต้องการเริ่มเรื่องใหม่ ให้พิมพ์ว่า “เริ่มหัวข้อใหม่” ได้ครับ
""".strip()