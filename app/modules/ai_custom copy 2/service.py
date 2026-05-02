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
- goal = เป้าหมายการเรียนรู้หรือผลลัพธ์ที่อยากได้หลังเรียน เช่น อยากปิดการขายได้ดีขึ้น, อยากสื่อสารกับทีมดีขึ้น, อยากเป็นหัวหน้างานที่ดีขึ้น
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
- goal ต้องเป็นผลลัพธ์ที่ผู้เรียนอยากพัฒนา
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
):
    system_prompt = f"""
คุณคือ AI Learning Consultant สำหรับระบบ Self-Learning

หน้าที่:
- ตอบคำถามผู้เรียนจาก RAG_CONTEXT เท่านั้น
- ห้ามแต่งข้อมูลนอกเหนือจาก context
- ตอบภาษาไทย สุภาพ เข้าใจง่าย
- ไม่ต้องบอกว่าตอบจากเนื้อหาอะไร
- ถ้า context ไม่พอ ให้บอกว่าเนื้อหาไม่เพียงพอ

หัวข้อ:
{topic}

RAG_CONTEXT:
{rag_context}
""".strip()

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_message,
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
