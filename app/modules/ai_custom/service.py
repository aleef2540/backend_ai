import json
import re
from app.shared.ai.openai_client import call_openai_chat_full, call_openai_chat_stream_full


REQUIRED_FIELDS = [
    "content",
    "goal",
    "event",
   
]

OPTIONAL_FIELDS = [
  
]

FIELD_LABELS = {
    "content": "อยากพัฒนาเรื่องหรือหัวข้ออะไร",
    "goal": "ปัญหาหรือสถานการณ์ที่กำลังเจอ",
    "event": "ผลลัพธ์ที่อยากให้เกิดหลังอบรม",
    
}

FEEDBACK_INTENTS = {
    "ask_more",
    "not_understand",
    "ask_example",
    "ask_summary",
    "ask_how_to_apply",
    "scenario_question",
    "report_done",
    "report_partial",
    "report_not_done",
    "blocked",
    "review_request",
    "unrelated",
    "restart_learning",
    "general_feedback",
}


def clean_json(text: str) -> str:
    text = re.sub(r"```json|```", "", text)
    return text.strip()


import json

def build_user_only_conversation_context(conversation_history):
    lines = []

    for item in conversation_history or []:
        if item.get("role") != "user":
            continue

        content = str(item.get("content") or "").strip()
        if content:
            lines.append(f"ผู้ใช้: {content}")

    return "\n".join(lines[-6:])

async def _stream_text_response(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
):
    """
    helper กลางสำหรับ stream ข้อความธรรมดา
    คืน event รูปแบบ:
    - {"type":"chunk","text":"..."}
    - {"type":"done","content":"...","usage":...,"cost":...}
    """
    final_content = ""

    async for item in call_openai_chat_stream_full(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            if text:
                final_content += text
                yield {
                    "type": "chunk",
                    "text": text,
                }

        elif item.get("type") == "done":
            content = (item.get("content") or final_content).strip()
            yield {
                "type": "done",
                "content": content,
                "usage": item.get("usage"),
                "cost": item.get("cost"),
            }
            return

def build_conversation_context(history: list | None, limit: int = 8) -> str:
    if not history:
        return ""

    recent = history[-limit:]
    lines = []

    for item in recent:
        role = item.get("role", "user")
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)

def calc_missing_requirements(requirements: dict) -> list:
    missing = []

    for field in REQUIRED_FIELDS:
        value = str(requirements.get(field) or "").strip()
        if not value or value == "unknown":
            missing.append(field)

    return missing

async def build_search_query(requirements: dict) -> str:
    content = str(requirements.get("content") or "").strip()
    goal = str(requirements.get("goal") or "").strip()
    event = str(requirements.get("event") or "").strip()

    parts = [content, goal, event]
    return " ".join([p for p in parts if p]).strip()


async def extract_requirements(
    user_message: str,
    current_requirements: dict, 
    conversation_history: list | None = None
) -> dict:
    conversation_context = build_user_only_conversation_context(conversation_history)

    system_prompt = f"""
คุณคือ AI Learning Consultant สำหรับระบบ Self-Learning ของสถาบัน

หน้าที่ของคุณ:
- อ่านบทสนทนาแล้วสกัด Requirement สำหรับแนะนำคอร์ส Self-Learning
- รวมข้อมูลใหม่กับ Requirement เดิม
- เก็บข้อมูลให้เป็นธรรมชาติ ไม่ใช่แบบฟอร์ม
- ห้ามเดา ถ้าไม่ชัดให้เว้นว่าง

Requirement ที่ต้องเก็บ:
- content = เนื้อหา/หัวข้อ/ทักษะที่ผู้เรียนสนใจ เช่น ภาวะผู้นำ, การขาย, การสื่อสาร, การบริการ, การบริหารทีม, การคิดเชิงกลยุทธ์
- goal = สิ่งที่ผู้เรียนตั้งใจจะเปลี่ยนหรือลงมือทำ (เน้นการปฏิบัติใหม่).
- event = สถานการณ์หรือบริบทที่ทำให้ต้องเรียน เช่น กำลังจะเลื่อนตำแหน่ง, ต้องดูแลทีมใหม่, ยอดขายตก, ลูกค้าร้องเรียน, ต้องเตรียมอบรมพนักงาน

Requirement เดิม:
{json.dumps(current_requirements or {}, ensure_ascii=False)}

บทสนทนาก่อนหน้า:
{conversation_context}

กฎสำคัญ:
- ถ้าข้อมูลเดิมมีอยู่แล้ว และข้อความใหม่ไม่ได้แก้ไข ให้คงค่าเดิม
- ถ้าข้อความใหม่ให้ข้อมูลชัดกว่าเดิม ให้ปรับให้ดีขึ้น
- ถ้าผู้ใช้ตอบสั้น ๆ เช่น "หัวหน้างาน", "การขาย", "บริการลูกค้า" ให้ตีความจากคำถามล่าสุดในบทสนทนา
- ห้ามสร้างข้อมูลเอง
- ถ้าไม่พบข้อมูล ให้ใช้ ""
- ตอบ JSON เท่านั้น
- content ต้องเป็นหัวข้อหรือทักษะที่เกี่ยวกับการเรียน Self-Learning
- event ต้องเป็นบริบท/เหตุการณ์/สถานการณ์ ไม่ใช่หัวข้อเรียน
- ถ้าผู้ใช้พูดแค่ "อยากเรียน" แต่ไม่บอกเรื่องอะไร ให้ content เป็น ""
- ถ้าผู้ใช้พูดแค่ "มีปัญหา" แต่ไม่บอกปัญหาอะไร ให้ event เป็น ""
- ถ้าผู้ใช้บอกปัญหาชัด เช่น "ปิดการขายไม่ได้" ให้ content เป็น "การขาย" และ goal เป็น "ปิดการขายได้ดีขึ้น"
- ถ้าผู้ใช้บอกว่า "กำลังจะเป็นหัวหน้างาน" ให้ event เป็น "กำลังจะเป็นหัวหน้างาน" และ content อาจเป็น "ภาวะผู้นำ" ได้เฉพาะเมื่อบริบทชัดเจน
- อย่าใส่ข้อมูลซ้ำกันทุก field ถ้าข้อมูลเดียวกันเหมาะกับ field เดียวมากกว่า

รูปแบบ JSON:
{{
  "content": "",
  "goal": "",
  "event": ""
}}
""".strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_message,
        temperature=0.1,
    )

    text = (result.get("content") or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)

        allowed_keys = [
            "content",
            "goal",
            "event",
        ]

        merged = dict(current_requirements or {})

        for key in allowed_keys:
            value = data.get(key, "")
            if value not in [None, "", "unknown"]:
                merged[key] = value
            elif key not in merged:
                merged[key] = ""

        return merged

    except Exception:
        return current_requirements or {}
    
async def build_next_question(
    requirements: dict,
    missing: list,
    conversation_history: list | None = None
):
    next_field = missing[0] if missing else None
    label = FIELD_LABELS.get(next_field, next_field or "")
    conversation_context = build_conversation_context(conversation_history)

    system_prompt = f"""
คุณคือ AI Learning Consultant สำหรับระบบ Self-Learning

หน้าที่:
- ตอบรับสิ่งที่ผู้เรียนบอกแบบสั้น ๆ
- ถามต่อเพียง 1 คำถาม เพื่อเก็บ Requirement ที่ยังขาด
- ห้ามถามหลายข้อพร้อมกัน
- ไม่ต้องแนะนำคอร์สเต็มจนกว่า Requirement จะครบ

Requirement ปัจจุบัน:
{json.dumps(requirements or {}, ensure_ascii=False)}

Requirement ที่ยังขาด:
{json.dumps(missing or [], ensure_ascii=False)}

บทสนทนาก่อนหน้า:
{conversation_context}

field ที่ต้องถามตอนนี้:
{next_field} = {label}

แนวทาง:
- ถ้าขาด content ให้ถามว่าอยากเรียน/พัฒนาเรื่องอะไร
- ถ้าขาด goal ให้ถามว่าอยากได้ผลลัพธ์อะไรหลังเรียน
- ถ้าขาด event ให้ถามว่ามีสถานการณ์หรือเหตุผลอะไรที่ทำให้สนใจเรื่องนี้ตอนนี้

ข้อกำหนด:
- ตอบภาษาไทย
- สุภาพ กระชับ เป็นธรรมชาติ
- ถามคำถามเดียวเท่านั้น
""".strip()

    user_prompt = "ช่วยถามคำถามถัดไปจากข้อมูลที่มี"

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
    ):
        yield item

async def reply_ask_concept_with_topic_stream(
    user_message: str,
    topic: str,
    rag_context: str,
    requirements: dict | None = None,
):
    requirement_text = json.dumps(requirements or {}, ensure_ascii=False, indent=2)

    system_prompt = f"""
คุณคือ AI Learning Consultant สำหรับระบบ Self-Learning

หน้าที่:
- ตอบคำถามผู้เรียนจาก RAG_CONTEXT เท่านั้น
- ห้ามแต่งข้อมูลนอกเหนือจาก RAG_CONTEXT
- ตอบภาษาไทย สุภาพ เข้าใจง่าย
- เชื่อมคำตอบกับ content / goal / event ของผู้เรียน ถ้าข้อมูล requirement มีเพียงพอ
- ไม่ต้องบอกว่าตอบจากเนื้อหาอะไร
- ถ้า RAG_CONTEXT ไม่พอ ให้บอกว่า "เนื้อหาไม่เพียงพอสำหรับตอบคำถามนี้"

หัวข้อ:
{topic}

RAG_CONTEXT:
{rag_context}
""".strip()

    user_prompt = f"""
Requirement ผู้เรียน:
{requirement_text}

คำถามล่าสุด:
{user_message}
""".strip()

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
    ):
        yield item

async def build_irrelevant_content_reply(
    user_message: str,
    requirements: dict,
    conversation_history: list | None = None,
    course_name_context: str = ""
):
    conversation_context = build_conversation_context(conversation_history)

    system_prompt = f"""
คุณคือ AI Learning Consultant สำหรับระบบ Self-Learning

สถานะ:
ระบบค้นหาในคลังบทเรียนแล้ว ยังไม่พบเนื้อหาที่เกี่ยวข้องกับสิ่งที่ผู้เรียนถาม

Requirement ปัจจุบัน:
{json.dumps(requirements or {}, ensure_ascii=False)}

หลักสูตร/หัวข้อที่ระบบมีความรู้และอนุญาตให้ใช้:
{course_name_context or "ยังไม่มีข้อมูลรายชื่อหลักสูตร"}

บทสนทนาก่อนหน้า:
{conversation_context}

ข้อกำหนด:
- ตอบภาษาไทย สุภาพ กระชับ
- บอกอย่างเป็นธรรมชาติว่าเรื่องที่ผู้เรียนถามอาจยังไม่อยู่ในหลักสูตร/บทเรียนที่เปิดให้เรียน
- ห้ามชวนให้ผู้เรียนอธิบายหัวข้อเดิมเพิ่ม ถ้าหัวข้อนั้นไม่อยู่ในคลัง เช่น ขับรถ ทำอาหาร เล่นหุ้น ฯลฯ
- ถ้ามีรายชื่อหลักสูตรด้านบน ให้บอกสั้น ๆ ว่าตอนนี้ช่วยเรื่องอะไรได้บ้าง โดยยกตัวอย่างไม่เกิน 3-5 รายการ
- อย่าแต่งชื่อหลักสูตรเอง ให้ใช้เฉพาะรายชื่อด้านบน
- ปิดท้ายด้วยคำถามเดียว เพื่อให้ผู้เรียนเลือกหัวข้อใหม่จากสิ่งที่ระบบมี
""".strip()

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_message,
        temperature=0.3,
    ):
        yield item

def detect_feedback_intent(user_message: str, state) -> str:
    text = str(user_message or "").strip().lower()

    if not text:
        return "general_feedback"

    restart_keywords = [
        "เริ่มใหม่",
        "เริ่มหัวข้อใหม่",
        "เปลี่ยนหัวข้อ",
        "เรียนเรื่องใหม่",
        "ขอเรื่องใหม่",
        "ไม่เอาเรื่องนี้แล้ว",
        "เปลี่ยนเรื่อง",
    ]

    if any(k in text for k in restart_keywords):
        return "restart_learning"

    done_keywords = [
        "ทำแล้ว",
        "ลองแล้ว",
        "ได้ลองแล้ว",
        "เอาไปใช้แล้ว",
        "ทำตามแล้ว",
        "ลองนำไปใช้แล้ว",
        "เริ่มทำแล้ว",
    ]

    if any(k in text for k in done_keywords):
        return "report_done"

    partial_keywords = [
        "ทำบางส่วน",
        "ลองไปนิดหน่อย",
        "ทำไปบ้าง",
        "ยังไม่ครบ",
        "ทำได้บางข้อ",
        "เริ่มไปนิดนึง",
    ]

    if any(k in text for k in partial_keywords):
        return "report_partial"

    not_done_keywords = [
        "ยังไม่ได้ทำ",
        "ยังไม่ได้ลอง",
        "ยังไม่เริ่ม",
        "ไม่มีเวลา",
        "ยังไม่ได้เอาไปใช้",
        "ยังไม่ได้ใช้",
    ]

    if any(k in text for k in not_done_keywords):
        return "report_not_done"

    blocked_keywords = [
        "ติดปัญหา",
        "ทำไม่ได้",
        "ไม่รู้จะเริ่ม",
        "ไม่รู้เริ่มยังไง",
        "ไม่เวิร์ค",
        "ไม่ได้ผล",
        "เจอปัญหา",
        "เขาไม่ตอบ",
        "พนักงานไม่ตอบ",
        "ไม่ให้ความร่วมมือ",
        "ไม่เข้าใจว่าจะทำยังไง",
    ]

    if any(k in text for k in blocked_keywords):
        return "blocked"

    review_keywords = [
        "ช่วยดู",
        "ช่วยตรวจ",
        "ตรวจให้หน่อย",
        "ประเมินให้หน่อย",
        "ผมเขียนมา",
        "นี่คือสิ่งที่ทำ",
        "แผนที่ผมทำ",
        "ช่วย feedback",
    ]

    if any(k in text for k in review_keywords):
        return "review_request"

    not_understand_keywords = [
        "ไม่เข้าใจ",
        "ยังงง",
        "งง",
        "อธิบายใหม่",
        "ไม่เคลียร์",
        "ยังไม่เก็ต",
        "ไม่แน่ใจ",
    ]

    if any(k in text for k in not_understand_keywords):
        return "not_understand"

    summary_keywords = [
        "สรุป",
        "ย่อ",
        "สั้นๆ",
        "สั้น ๆ",
        "เอาแบบสั้น",
        "สรุปอีกที",
    ]

    if any(k in text for k in summary_keywords):
        return "ask_summary"

    example_keywords = [
        "ตัวอย่าง",
        "ยกตัวอย่าง",
        "มีตัวอย่างไหม",
        "ขอเคส",
        "เคสตัวอย่าง",
    ]

    if any(k in text for k in example_keywords):
        return "ask_example"

    apply_keywords = [
        "ใช้ยังไง",
        "นำไปใช้",
        "เอาไปใช้",
        "ปรับใช้",
        "ประยุกต์ใช้",
        "เริ่มยังไง",
        "ขั้นตอน",
        "ต้องทำยังไง",
        "ลงมือยังไง",
    ]

    if any(k in text for k in apply_keywords):
        return "ask_how_to_apply"

    ask_more_keywords = [
        "ขยาย",
        "เพิ่ม",
        "เพิ่มเติม",
        "อธิบายเพิ่ม",
        "เล่าเพิ่ม",
        "ขอรายละเอียด",
        "อยากรู้เพิ่ม",
    ]

    if any(k in text for k in ask_more_keywords):
        return "ask_more"

    # ตรวจแบบง่ายว่าเกี่ยวกับ topic เดิมไหม
    learning_phase = getattr(state, "learning_phase", {}) or {}
    requirements = learning_phase.get("requirements") or getattr(state, "requirements", {}) or {}

    topic = str(learning_phase.get("topic") or getattr(state, "topic", "") or "").lower()
    content = str(requirements.get("content") or "").lower()
    goal = str(requirements.get("goal") or "").lower()
    event = str(requirements.get("event") or "").lower()

    context_text = f"{topic} {content} {goal} {event}".strip()

    # ถ้ามีคำสำคัญจาก context เดิมอยู่ในข้อความ user ให้ถือว่าเกี่ยวข้อง
    for word in context_text.split():
        if len(word) >= 4 and word in text:
            return "ask_more"

    # ถ้าข้อความสั้นมากใน feedback mode มักเป็น follow-up มากกว่า unrelated
    if len(text) <= 30:
        return "ask_more"

    return "unrelated"

def build_feedback_user_prompt(
    user_message: str,
    state,
    feedback_intent: str,
    rag_context: str,
) -> str:
    learning_phase = getattr(state, "learning_phase", {}) or {}

    ai_recommendation_text = (
        learning_phase.get("ai_recommendation_text")
        or getattr(state, "last_answer", "")
        or ""
    )

    requirements = (
        learning_phase.get("requirements")
        or getattr(state, "requirements", {})
        or {}
    )

    topic = (
        learning_phase.get("topic")
        or getattr(state, "topic", "")
        or "หัวข้อที่กำลังเรียน"
    )

    feedback_history = learning_phase.get("feedback_history") or []

    return f"""
โหมดปัจจุบัน: learning_feedback

หัวข้อการเรียนรู้:
{topic}

Requirement เดิมของผู้เรียน:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

คำแนะนำเดิมที่ AI เคยให้ผู้เรียน:
{ai_recommendation_text}

Feedback history:
{json.dumps(feedback_history[-5:], ensure_ascii=False, indent=2)}

Intent ล่าสุดของผู้เรียน:
{feedback_intent}

ข้อความล่าสุดจากผู้เรียน:
{user_message}

RAG_CONTEXT เดิมที่ใช้กับ learning phase นี้:
{rag_context}

คำสั่ง:
- ห้ามเก็บ requirement ใหม่
- ห้ามเปลี่ยนหัวข้อเอง ยกเว้น intent เป็น restart_learning
- ให้ตอบต่อจาก learning phase เดิม
- ถ้าผู้เรียนถามเพิ่ม ให้ตอบจากคำแนะนำเดิมและ RAG_CONTEXT เดิม
- ถ้าผู้เรียนบอกว่ายังไม่เข้าใจ ให้อธิบายใหม่ให้ง่ายขึ้น
- ถ้าผู้เรียนขอตัวอย่าง ให้ยกตัวอย่างที่เข้ากับ requirement เดิม
- ถ้าผู้เรียนถามวิธีนำไปใช้ ให้แปลงเป็นขั้นตอนปฏิบัติ
- ถ้าผู้เรียนรายงานว่าทำแล้ว ให้ถามผลลัพธ์/สิ่งที่เกิดขึ้น/สิ่งที่ติด
- ถ้าผู้เรียนบอกว่ายังไม่ได้ทำ ให้ลด task ให้เล็กลงและช่วยเริ่มทีละขั้น
- ถ้าผู้เรียนติดปัญหา ให้วิเคราะห์ blocker และเสนอวิธีแก้
- ถ้าคำถามไม่เกี่ยวข้อง ให้ดึงกลับมาที่หัวข้อเดิมอย่างสุภาพ
- ตอบภาษาไทย สุภาพ เข้าใจง่าย
""".strip()

async def reply_learning_feedback_stream(
    user_message: str,
    state,
    feedback_intent: str,
    rag_context: str,
):
    learning_phase = getattr(state, "learning_phase", {}) or {}
    topic = (
        learning_phase.get("topic")
        or getattr(state, "topic", "")
        or "หัวข้อที่กำลังเรียน"
    )

    system_prompt = f"""
คุณคือ AI Learning Coach สำหรับระบบ Self-Learning

หน้าที่ในโหมด learning_feedback:
- ช่วยผู้เรียนเรียนต่อจากคำแนะนำเดิม
- อธิบายเพิ่ม สรุป ยกตัวอย่าง หรือแปลงเป็นขั้นตอนปฏิบัติได้
- เก็บ feedback จากผู้เรียนว่าได้ลองทำหรือยัง ทำอะไรไปแล้ว ติดปัญหาอะไร
- วิเคราะห์ปัญหาและปรับ next step ให้เหมาะสม
- ใช้ RAG_CONTEXT เดิมเป็นฐานความรู้
- ห้ามเปลี่ยนหัวข้อเอง
- ห้ามแต่งข้อมูลที่ขัดกับ RAG_CONTEXT
- ตอบภาษาไทย สุภาพ กระชับแต่ช่วยได้จริง

หัวข้อปัจจุบัน:
{topic}
""".strip()

    user_prompt = build_feedback_user_prompt(
        user_message=user_message,
        state=state,
        feedback_intent=feedback_intent,
        rag_context=rag_context,
    )

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
    ):
        yield item

async def detect_feedback_intent_ai(user_message: str, state) -> dict:
    learning_phase = getattr(state, "learning_phase", {}) or {}

    requirements = (
        learning_phase.get("requirements")
        or getattr(state, "requirements", {})
        or {}
    )

    topic = (
        learning_phase.get("topic")
        or getattr(state, "topic", "")
        or "unknown"
    )

    ai_recommendation_text = (
        learning_phase.get("ai_recommendation_text")
        or getattr(state, "last_answer", "")
        or ""
    )

    feedback_history = learning_phase.get("feedback_history") or []

    system_prompt = """
คุณคือ intent classifier สำหรับระบบ AI Learning Feedback

บริบท:
- ห้องนี้เข้าสู่โหมด learning_feedback แล้ว
- ผู้เรียนได้รับคำแนะนำจาก AI ไปแล้ว
- ข้อความล่าสุดของผู้เรียนอาจเป็นการถามเพิ่ม รายงานผล บอกว่าติดปัญหา หรือขอเริ่มหัวข้อใหม่
- โดยปกติให้ถือว่าผู้เรียนกำลังถามต่อจาก learning phase เดิมก่อน
- ห้ามจัดเป็น unrelated ง่ายเกินไป

ให้เลือก intent ได้เพียง 1 ค่า จากรายการนี้:
- ask_more: ผู้เรียนถามเพิ่มหรือขอขยายความทั่วไป
- not_understand: ผู้เรียนบอกว่ายังไม่เข้าใจ งง ไม่เคลียร์
- ask_example: ผู้เรียนขอตัวอย่างหรือเคสตัวอย่าง
- ask_summary: ผู้เรียนขอสรุปหรือย่อ
- ask_how_to_apply: ผู้เรียนถามวิธีนำไปใช้ ขั้นตอน หรือเริ่มทำอย่างไร
- scenario_question: ผู้เรียนถามกรณีเฉพาะ เช่น "ถ้า...", "แล้วถ้า...", "กรณี...", "สมมติว่า..."
- report_done: ผู้เรียนรายงานว่าลองทำแล้ว หรือเอาไปใช้แล้ว
- report_partial: ผู้เรียนรายงานว่าทำบางส่วน ยังไม่ครบ
- report_not_done: ผู้เรียนบอกว่ายังไม่ได้ทำ ยังไม่ได้ลอง ไม่มีเวลา
- blocked: ผู้เรียนบอกว่าติดปัญหา ทำไม่ได้ ไม่ได้ผล หรือเจออุปสรรค
- review_request: ผู้เรียนส่งสิ่งที่ทำมาให้ช่วยดู ช่วยตรวจ หรือขอ feedback
- unrelated: ข้อความไม่เกี่ยวกับ learning phase เดิมจริง ๆ
- restart_learning: ผู้เรียนขอเริ่มหัวข้อใหม่ เปลี่ยนหัวข้อ หรือไม่เอาเรื่องเดิมแล้ว
- general_feedback: feedback ทั่วไปที่จัดเข้าหมวดอื่นไม่ได้

กติกาสำคัญ:
- ถ้าผู้เรียนถามว่า "แล้วถ้า X ล่ะ" และ X ยังเกี่ยวกับหัวข้อเดิม ให้เลือก scenario_question
- ถ้าข้อความสามารถโยงกับ topic, requirement, คำแนะนำเดิม หรือผู้เรียน/งาน/สถานการณ์เดิมได้ ให้ถือว่า related
- unrelated ใช้เฉพาะเมื่อชัดเจนว่าออกนอกเรื่อง เช่น ร้านอาหาร ท่องเที่ยว ข่าว เรื่องที่ไม่เกี่ยวกับหัวข้อเรียนเดิม
- restart_learning ใช้เฉพาะเมื่อผู้เรียนบอกชัดว่าอยากเริ่มใหม่หรือเปลี่ยนหัวข้อ
- ตอบ JSON เท่านั้น ห้ามใช้ markdown
""".strip()

    user_prompt = f"""
Learning topic:
{topic}

Requirement เดิม:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

คำแนะนำเดิมของ AI:
{ai_recommendation_text[:2500]}

Feedback history ล่าสุด:
{json.dumps(feedback_history[-5:], ensure_ascii=False, indent=2)}

ข้อความล่าสุดของผู้เรียน:
{user_message}

ตอบ JSON format นี้เท่านั้น:
{{
  "intent": "หนึ่งใน intent ที่กำหนด",
  "confidence": 0.0,
  "is_related_to_learning_phase": true,
  "reason": "เหตุผลสั้น ๆ ภาษาไทย"
}}
""".strip()

    try:
        result = await call_openai_chat_full(
            model="gpt-4.1-mini",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0,
        )

        raw = (result.get("content") or "").strip()
        raw = clean_json(raw)

        data = json.loads(raw)

        intent = str(data.get("intent") or "general_feedback").strip()

        if intent not in FEEDBACK_INTENTS:
            intent = "general_feedback"

        confidence = data.get("confidence", 0)

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        return {
            "intent": intent,
            "confidence": confidence,
            "is_related_to_learning_phase": bool(
                data.get("is_related_to_learning_phase", True)
            ),
            "reason": str(data.get("reason") or "").strip(),
        }

    except Exception as e:
        print("[FEEDBACK INTENT AI ERROR]", repr(e), flush=True)

        # fallback ใช้ rule-based เดิม ถ้ายังเก็บ function detect_feedback_intent ไว้
        fallback_intent = detect_feedback_intent(user_message, state)

        return {
            "intent": fallback_intent,
            "confidence": 0.0,
            "is_related_to_learning_phase": fallback_intent != "unrelated",
            "reason": "fallback_to_rule_based",
        }