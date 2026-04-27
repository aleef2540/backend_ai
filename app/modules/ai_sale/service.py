import json
from app.shared.ai.openai_client import call_openai_chat_full


REQUIRED_FIELDS = [
    "target_group",
    "pain_point",
    "development_goal",
]

OPTIONAL_FIELDS = [
    "training_format",
    "duration",
    "industry",
    "budget",
    "urgency",
    "special_condition",
]

FIELD_LABELS = {
    "target_group": "กลุ่มผู้เรียนหลักเป็นใคร เช่น หัวหน้างาน ผู้จัดการ ทีมขาย หรือพนักงานทั่วไป",
    "pain_point": "ปัญหาหรือสถานการณ์ที่องค์กรกำลังเจอ",
    "development_goal": "ผลลัพธ์ที่อยากให้ผู้เรียนเปลี่ยนแปลงหลังเรียน",
    "training_format": "รูปแบบที่สะดวก เช่น In-house, Workshop, Coaching, Self-learning",
    "duration": "กรอบเวลาอบรมที่ต้องการ เช่น 1 วัน 2 วัน หรือหลายรุ่น",
}


def calc_missing_requirements(requirements: dict) -> list:
    missing = []

    for field in REQUIRED_FIELDS:
        value = str(requirements.get(field) or "").strip()
        if not value or value == "unknown":
            missing.append(field)

    return missing


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


async def extract_requirements(
    user_message: str,
    current_requirements: dict,
    conversation_history: list | None = None
) -> dict:
    conversation_context = build_conversation_context(conversation_history)

    system_prompt = f"""
คุณคือ AI Sales Consultant สำหรับคุยกับลูกค้าเพื่อแนะนำหลักสูตรฝึกอบรม

หน้าที่ของคุณ:
- อ่านบทสนทนาแล้วสกัด Requirement ที่ลูกค้าบอกไว้
- รวมข้อมูลใหม่กับ Requirement เดิม
- เก็บข้อมูลให้เป็นธรรมชาติ ไม่ใช่แบบฟอร์ม
- ห้ามเดา ถ้าไม่ชัดให้เว้นว่าง

Requirement ที่ต้องเก็บ:
- target_group = กลุ่มผู้เรียน
- pain_point = ปัญหา / สถานการณ์ / ความท้าทาย
- development_goal = ผลลัพธ์ที่อยากพัฒนา
- training_format = รูปแบบการเรียนที่ต้องการ
- duration = ระยะเวลาอบรม
- industry = ประเภทธุรกิจ/อุตสาหกรรม
- budget = งบประมาณ
- urgency = ความเร่งด่วน / ช่วงเวลาที่ต้องใช้
- special_condition = เงื่อนไขพิเศษ เช่น หลายรุ่น ออนไลน์ ผสม onsite ต้องการ proposal

Requirement เดิม:
{json.dumps(current_requirements or {}, ensure_ascii=False)}

บทสนทนาก่อนหน้า:
{conversation_context}

กฎสำคัญ:
- ถ้าข้อมูลเดิมมีอยู่แล้ว และข้อความใหม่ไม่ได้แก้ไข ให้คงค่าเดิม
- ถ้าข้อความใหม่ให้ข้อมูลชัดกว่าเดิม ให้ปรับให้ดีขึ้น
- ถ้าผู้ใช้ตอบสั้น ๆ เช่น "หัวหน้างาน" ให้ตีความว่าเป็นคำตอบของคำถามล่าสุดจากบริบทได้
- ห้ามสร้างข้อมูลเอง
- ถ้าไม่พบข้อมูล ให้ใช้ ""
- ตอบ JSON เท่านั้น

รูปแบบ JSON:
{{
  "target_group": "",
  "pain_point": "",
  "development_goal": "",
  "training_format": "",
  "duration": "",
  "industry": "",
  "budget": "",
  "urgency": "",
  "special_condition": ""
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
        merged = dict(current_requirements or {})
        for key, value in data.items():
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
) -> str:
    next_field = missing[0]
    label = FIELD_LABELS.get(next_field, next_field)
    conversation_context = build_conversation_context(conversation_history)

    system_prompt = """
คุณคือ AI Sales Consultant ของบริษัทฝึกอบรม

บุคลิกการคุย:
- เป็นธรรมชาติ อบอุ่น สุภาพ เหมือนฝ่ายขายมืออาชีพ
- ไม่ถามเหมือนกรอกแบบฟอร์ม
- ไม่ถามหลายข้อพร้อมกัน
- คุยต่อจากสิ่งที่ลูกค้าเล่า
- ถ้าลูกค้าเล่าปัญหามา ให้สะท้อนความเข้าใจก่อน แล้วค่อยถามต่อ
- ถ้าลูกค้ายังพูดกว้าง ให้ช่วยจัดกรอบให้เลือกง่ายขึ้น

ข้อกำหนด:
- ตอบ 2-3 ประโยค
- ถามคำถามเดียวเท่านั้น
- ห้ามลิสต์ Requirement ทั้งหมด
- ห้ามพูดว่า “กรุณาระบุ Requirement”
- ห้ามพูดเหมือนบอทหรือแบบสอบถาม
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

Requirement ปัจจุบัน:
{json.dumps(requirements or {}, ensure_ascii=False)}

ข้อมูลที่ยังอยากเข้าใจเพิ่ม:
{label}

ช่วยตอบกลับลูกค้าแบบธรรมชาติ แล้วถามต่อ 1 คำถาม
""".strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.55,
    )

    return (result.get("content") or "").strip()


async def build_search_query(requirements: dict) -> str:
    system_prompt = """
คุณคือ AI Sales Consultant
ให้แปลง Requirement ลูกค้าเป็น search query สำหรับค้นหาหลักสูตรใน Vector Database

หลักการเขียน query:
- เขียนเหมือนโจทย์ความต้องการของลูกค้า
- รวมกลุ่มผู้เรียน ปัญหา เป้าหมาย และบริบทสำคัญ
- ไม่ต้องยาวเกินไป
- ใช้ภาษาไทยธรรมชาติ
- ห้ามแต่งข้อมูลที่ไม่มีใน Requirement
- ตอบเป็น query อย่างเดียว
""".strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=json.dumps(requirements or {}, ensure_ascii=False),
        temperature=0.2,
    )

    return (result.get("content") or "").strip()


async def build_recommendation_reply(requirements: dict, courses: list) -> str:
    system_prompt = """
คุณคือ AI Sales Consultant สำหรับแนะนำหลักสูตรฝึกอบรม

รูปแบบการตอบ:
- ตอบแบบฝ่ายขายใช้คุยกับลูกค้าได้จริง
- เริ่มจากสรุปความเข้าใจ Requirement ของลูกค้าสั้น ๆ
- แนะนำหลักสูตรที่เหมาะ 1-3 หลักสูตร
- อธิบายเหตุผลว่าเหมาะกับ pain point / target group / development goal อย่างไร
- ถ้ามี score ให้ใช้เป็นข้อมูลภายในได้ แต่ไม่ต้องโชว์คะแนนละเอียด
- ปิดท้ายด้วยคำถามชวนไปขั้นตอนถัดไป เช่น ต้องการให้จัดเป็นแนว proposal หรือเทียบหลักสูตรไหม

ข้อกำหนด:
- ตอบเป็น HTML snippet เท่านั้น
- ใช้ได้เฉพาะ <div>, <p>, <strong>, <ul>, <li>, <br>
- ห้ามใช้ markdown
- ห้ามแต่งชื่อหลักสูตรเอง
- ถ้าผลค้นหาไม่มี ให้บอกว่ายังไม่พบหลักสูตรที่ตรงชัดเจน และถามเพื่อเก็บข้อมูลเพิ่ม
""".strip()

    user_prompt = f"""
Requirement ลูกค้า:
{json.dumps(requirements or {}, ensure_ascii=False)}

ผลค้นหาหลักสูตรจาก Qdrant:
{json.dumps(courses or [], ensure_ascii=False)}
""".strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.45,
    )

    return (result.get("content") or "").strip()