import json
from app.shared.ai.openai_client import call_openai_chat_full, call_openai_chat_stream_full
from app.modules.ai_assis.company_service import fetch_company_profile_context
from app.core.database import run_query_bridge


REQUIRED_FIELDS = [
    "topic",
    "pain_point",
   
]

OPTIONAL_FIELDS = [
    "competency",
    "budget",
        "development_goal",
         "target_group",
]

FIELD_LABELS = {
    "topic": "อยากพัฒนาเรื่องหรือหัวข้ออะไร",
    "pain_point": "ปัญหาหรือสถานการณ์ที่กำลังเจอ",
    "development_goal": "ผลลัพธ์ที่อยากให้เกิดหลังอบรม",
    "competency": "ทักษะหรือสมรรถนะที่อยากพัฒนาเพิ่มเติม",
    "target_group": "กลุ่มผู้เรียน เช่น พนักงานทั่วไป หัวหน้างาน ผู้จัดการ หรือผู้บริหาร",
    "budget": "งบประมาณโดยประมาณสำหรับการจัดอบรม",
}


async def _stream_text_response(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
):
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
คุณคือ AI Sales Consultant สำหรับคุยกับลูกค้าเพื่อแนะนำหลักสูตรฝึกอบรมของสถาบัน

หน้าที่ของคุณ:
- อ่านบทสนทนาแล้วสกัด Requirement ที่ลูกค้าบอกไว้
- รวมข้อมูลใหม่กับ Requirement เดิม
- เก็บข้อมูลให้เป็นธรรมชาติ ไม่ใช่แบบฟอร์ม
- ห้ามเดา ถ้าไม่ชัดให้เว้นว่าง

Requirement ที่ต้องเก็บ:
- topic = ระบุหัวข้อหรือสิ่งที่ผู้ใช้พูดถึง ถ้าลูกค้าพูดถึง topic ใหม่ให้เปลี่ยน topic ด้วย
- pain_point = ปัญหา / สถานการณ์ / ความท้าทายที่กำลังเจอแบบเฉพาะเจาะจง
- development_goal = ผลลัพธ์ที่อยากให้ผู้เรียนเปลี่ยนแปลงหลังเรียนเกี่ยวกับเรื่องหลักสูตรขิงสถาบันเท่านั้น
- competency = ทักษะหรือสมรรถนะที่ต้องการพัฒนา เช่น การปิดการขาย การสื่อสาร การบริหารทีม การคิดเชิงกลยุทธ์
- target_group = กลุ่มผู้เรียน เช่น พนักงานทั่วไป หัวหน้างาน ผู้จัดการ ผู้บริหาร ทีมขาย
- budget = งบประมาณโดยประมาณ

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
- ถ้าผู้ใช้บอกเพียงว่า "มีปัญหา" แต่ไม่บอกปัญหาอะไร ให้ pain_point เป็น ""
- ถ้า pain_point ชัด เช่น "ปิดการขายไม่ได้", "หัวหน้างานสื่อสารไม่ดี", "ทีมบริการโดนลูกค้าร้องเรียน" ให้เก็บได้ทันที
- competency สามารถสรุปจาก pain_point หรือ development_goal ได้ ถ้าชัดเจนมากพอ แต่ห้ามเดาเกินข้อมูลที่มี

รูปแบบ JSON:
{{
  "topic": "",
  "pain_point": "",
  "development_goal": "",
  "competency": "",
  "target_group": "",
  "budget": ""
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
            "topic",
            "pain_point",
            "development_goal",
            "competency",
            "target_group",
            "budget",
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
    next_field = missing[0]
    label = FIELD_LABELS.get(next_field, next_field)
    conversation_context = build_conversation_context(conversation_history)

    system_prompt = """
คุณคือ AI Sales Consultant สำหรับคุยกับลูกค้าเพื่อแนะนำหลักสูตรฝึกอบรมของสถาบันห้ามชวนคุยหรือแนะนำเรื่องที่ไม่เกี่ยวข้อง

บุคลิกการคุย:
- เป็นธรรมชาติ อบอุ่น สุภาพ เหมือนฝ่ายขายมืออาชีพ
- ไม่ถามเหมือนกรอกแบบฟอร์ม
- ไม่ถามหลายข้อพร้อมกัน
- คุยต่อจากสิ่งที่ลูกค้าเล่า
- ถ้าลูกค้าเล่าปัญหามา ให้สะท้อนความเข้าใจก่อน แล้วค่อยถามต่อ
- ถ้าลูกค้ายังพูดกว้าง ให้ช่วยจัดกรอบให้เลือกง่ายขึ้น
- **ห้ามพูดเรื่องที่ไม่เกี่ยวข้องกับการฝึกอบรมหรือทักษะที่เกี่ยวข้องกับหลักสูตรฝึกอบรม**

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

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.55,
    ):
        yield item

async def build_search_query(requirements: dict) -> str:
    system_prompt = """
คุณคือ AI Course Name Query Builder

หน้าที่:
- สร้าง search query สำหรับค้นจาก "ชื่อหลักสูตร"
- ถ้าผู้ใช้พิมพ์ชื่อหลักสูตรมาชัดเจน ให้ใช้ชื่อนั้นหรือส่วนสำคัญของชื่อนั้น
- ถ้าผู้ใช้พิมพ์เป็นความต้องการ ให้แปลงเป็นชื่อหลักสูตรที่น่าจะมี
- Query ต้องคล้ายชื่อหลักสูตร ไม่ใช่วัตถุประสงค์ยาว ๆ
- ใช้ได้ทั้งไทยและอังกฤษ
- ห้ามใส่คำฟุ่มเฟือย เช่น หลักสูตร อบรม inhouse public training สัมมนา
- ตอบสั้น ไม่เกิน 6 คำ

ตัวอย่าง:
ผู้ใช้ถามเรื่อง AI
=> AI Development

ผู้ใช้ถามเรื่องการใช้งาน AI
=> AI Development

ผู้ใช้ถามเรื่อง ChatGPT
=> ChatGPT

ผู้ใช้ถามเรื่องหัวหน้างาน
=> หัวหน้างาน

ผู้ใช้ถามเรื่อง leadership
=> Leadership

ผู้ใช้ถามเรื่องทีมขาย
=> Sales

ผู้ใช้ถามเรื่องการโค้ช
=> Coaching

ผู้ใช้พิมพ์ "AI Development Path"
=> AI Development Path

ผู้ใช้พิมพ์ "Strategic AI for Leaders"
=> Strategic AI for Leaders

ตอบเป็น search query อย่างเดียว
""".strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=json.dumps(requirements or {}, ensure_ascii=False),
    )

    query = (result.get("content") or "").strip()

    banned_words = [
        "หลักสูตร",
        "อบรม",
        "training",
        "inhouse",
        "public",
        "สัมมนา",
        "คอร์ส",
        "เกี่ยวกับ",
    ]

    for word in banned_words:
        query = query.replace(word, "")

    query = " ".join(query.split()).strip()

    print("FINAL SEARCH QUERY =", query, flush=True)

    return query

async def build_recommendation_reply(
    requirements: dict,
    courses: list
):
    system_prompt = """
คุณคือ AI Sales Consultant สำหรับวิเคราะห์ความต้องการลูกค้า (Training Needs Analysis)

บุคลิกการตอบ:
- เป็นธรรมชาติ เหมือนฝ่ายขายคุยกับลูกค้า
- ภาษาลื่น อ่านแล้วไม่รู้สึกว่าเป็นการสรุปเป็นข้อ
- เชื่อมประโยคให้เป็นเรื่องเดียวกัน

แนวทางการตอบ:
- เริ่มจากสะท้อนความเข้าใจลูกค้าแบบสั้น ๆ
- อธิบายให้เห็นภาพว่า ลูกค้าเป็นกลุ่มไหน กำลังเจอปัญหาอะไร และอยากได้ผลลัพธ์อะไร
- เขียนให้เป็นย่อหน้าเดียวหรือ 2-3 ประโยคที่ต่อเนื่องกัน
- ไม่ต้องแยกข้อ ไม่ต้องใช้ตัวเลข

ข้อกำหนด:
- ห้ามแนะนำหลักสูตร
- ห้ามพูดถึง course / score / ranking
- ห้ามสร้างข้อมูลที่ไม่มีจาก requirement
- ตอบเป็น text เท่านั้น
- ห้ามใช้ markdown
- ปิดท้ายด้วยประโยคว่า “เราเจอหลักสูตรที่เหมาะสมดังนี้”
""".strip()

    user_prompt = f"""
Requirement ลูกค้า:
{json.dumps(requirements or {}, ensure_ascii=False)}

ข้อมูลหลักสูตร (ใช้เป็น context เท่านั้น ไม่ต้องแสดงผล):
{json.dumps(courses or [], ensure_ascii=False)}
""".strip()

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
    ):
        yield item

async def detect_post_recommend_intent(
    user_message: str,
    requirements: dict,
    conversation_history: list | None = None
) -> dict:
    conversation_context = build_conversation_context(conversation_history)

    system_prompt = """
คุณคือ AI Sales Consultant

หน้าที่คือวิเคราะห์ว่า หลังจากระบบแนะนำหลักสูตรแล้ว
ข้อความล่าสุดของผู้ใช้ต้องการอะไรต่อ

เลือก intent เดียวเท่านั้น:

1. ask_more_courses
= ต้องการตัวเลือกเพิ่ม แต่ยังเป็นหัวข้อเดิม/โจทย์เดิม
เช่น:
- มีอีกไหม
- ขอเพิ่มอีก 3 หลักสูตร
- มีตัวเลือกอื่นไหม

2. refine_requirement
= หัวข้อเดิมยังเหมือนเดิม แต่ต้องการปรับรายละเอียดเงื่อนไข
เช่น เปลี่ยน:
- กลุ่มผู้เรียน
- ระดับผู้เข้าอบรม
- งบประมาณ
- ระยะเวลา
- รูปแบบอบรม onsite / online
- เน้น workshop มากขึ้น
- เน้นหัวหน้างานมากขึ้น

สำคัญ:
หาก "หัวข้อหลักสูตรหลัก" ยังเหมือนเดิม จึงใช้ refine_requirement

3. new_requirement
= เปลี่ยนหัวข้อใหม่ เปลี่ยน competency ใหม่ เปลี่ยนคนละเรื่องกับโจทย์เดิม

เช่น:
- จาก communication -> coaching
- จาก leadership -> sales
- จาก team building -> KPI
- ถ้าเป็นเรื่องของการสร้างโค้ชในองค์กรล่ะ
- ขอหลักสูตร succession plan

ถ้าหัวข้อใหม่คนละเรื่อง ให้เลือก new_requirement ทันที

4. ask_detail
= ถามรายละเอียดคอร์สที่แนะนำแล้ว
เช่น:
- คอร์สนี้กี่ชั่วโมง
- ราคาเท่าไร
- outline มีอะไรบ้าง
- จัด onsite ได้ไหม

5. unclear
= ยังไม่ชัดเจน

ตอบ JSON เท่านั้น:
{
  "intent": "ask_more_courses|refine_requirement|new_requirement|ask_detail|unclear",
  "reason": ""
}
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

Requirement ปัจจุบัน:
{json.dumps(requirements or {}, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}
""".strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
    )

    text = (result.get("content") or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
        intent = data.get("intent", "unclear")
        if intent not in [
            "ask_more_courses",
            "refine_requirement",
            "new_requirement",
            "ask_detail",
            "unclear",
        ]:
            intent = "unclear"

        return {
            "intent": intent,
            "reason": data.get("reason", "")
        }
    except Exception:
        return {
            "intent": "unclear",
            "reason": "parse_failed"
        }

async def build_more_courses_reply(
    requirements: dict,
    courses: list
):
    system_prompt = """
คุณคือ AI Sales Consultant

สถานการณ์:
ลูกค้าได้เห็นคำแนะนำหลักสูตรไปแล้ว และตอนนี้ต้องการ "ตัวเลือกเพิ่มเติม"

แนวทางการตอบ:
- ห้ามสรุปความเข้าใจลูกค้าซ้ำ
- ห้ามขึ้นต้นว่า "จากที่เข้าใจ..."
- ห้ามเล่า pain point ใหม่
- ให้พูดเหมือนเสนอทางเลือกเพิ่ม
- กระชับ เป็นธรรมชาติ

ตัวอย่าง:
- "ผมมีอีกตัวเลือกหนึ่งที่น่าสนใจ ลองดูเพิ่มเติมนะครับ"
- "อีกตัวเลือกที่ใกล้เคียงกับโจทย์ของคุณมีดังนี้ครับ"

ข้อกำหนด:
- ตอบเป็น text เท่านั้น
- ห้ามใช้ markdown
""".strip()

    user_prompt = f"""
Requirement:
{json.dumps(requirements or {}, ensure_ascii=False)}

Courses:
{json.dumps(courses or [], ensure_ascii=False)}
""".strip()

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
    ):
        yield item

async def build_irrelevant_topic_reply(
    user_message: str,
    old_topic: str | None,
    conversation_history: list | None = None
):
    conversation_context = build_conversation_context(conversation_history)

    system_prompt = """
คุณคือ AI Sales Consultant สำหรับแนะนำหลักสูตรฝึกอบรมในองค์กร

เป้าหมาย:
- ถ้าลูกค้าพูดเรื่องที่ไม่เกี่ยวกับหลักสูตรหรืออยู่นอก scope
- ให้ตอบแบบสุภาพ ไม่ปฏิเสธแข็ง
- พยายามดึงบทสนทนากลับมาที่เรื่องการพัฒนา/อบรม/ทักษะ
- ถ้ามี topic เดิม ให้โยงกลับไป

สไตล์:
- สุภาพ เป็นธรรมชาติ
- 2-3 ประโยค
- ไม่แข็ง ไม่ robotic
- ปิดท้ายด้วยคำถาม 1 คำถามเพื่อพากลับเข้า flow
""".strip()

    user_prompt = f"""
ข้อความล่าสุดของลูกค้า:
{user_message}

topic ก่อนหน้า:
{old_topic}

บทสนทนาก่อนหน้า:
{conversation_context}

ช่วยตอบลูกค้า:
- บอกแบบสุภาพว่าสิ่งที่พูดอาจไม่ตรงกับหลักสูตร
- แล้วพากลับไปคุยเรื่อง training / skill / development
- ถ้ามี topic เดิม ให้โยงกลับ
- ปิดด้วยคำถาม 1 คำถาม
""".strip()

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.6,
    ):
        yield item
    
def get_course_payload(course):
    if isinstance(course, dict):
        payload = course.get("payload") or course
        return payload if isinstance(payload, dict) else {}

    return {}

def compact_course_for_prompt(course: dict) -> dict:
    if not isinstance(course, dict):
        return {
            "course_id": None,
            "course_name": str(course or ""),
            "detail_link": None,
            "outline_link": None,
            "detail_url": None,
            "outline_url": None,
            "description": "",
            "course_outline": "",
            "instructor": "",
        }

    payload = course.get("payload") or course

    return {
        "course_id": (
            course.get("course_id")
            or payload.get("course_id")
            or payload.get("ICourse_no")
        ),
        "course_name": (
            course.get("course_name")
            or payload.get("course_name")
            or payload.get("ICourse_nameEN")
            or payload.get("ICourse_nameTH")
            or ""
        ),
        "detail_link": course.get("detail_link") or payload.get("detail_link"),
        "outline_link": course.get("outline_link") or payload.get("outline_link"),
        "detail_url": course.get("detail_url") or payload.get("detail_url"),
        "outline_url": course.get("outline_url") or payload.get("outline_url"),
        "description": (
            course.get("description")
            or payload.get("ICourse_description")
            or payload.get("description")
            or ""
        ),
        "course_outline": (
            course.get("course_outline")
            or payload.get("ICourse_outline")
            or payload.get("course_outline")
            or ""
        ),
        "instructor": (
            course.get("instructor")
            or payload.get("ICourse_instructor")
            or payload.get("instructor")
            or ""
        ),
    }


async def build_next_question_topic(
    requirements: dict,
    missing: list,
    conversation_history: list | None = None,
    matched_course=None,
):

    clean_requirements = {
        k: v for k, v in (requirements or {}).items()
        if k not in [
            "matched_course",
            "primary_matched_course",
            "last_inhouse_course",
            "last_inhouse_course_detail",
            "recommended_courses",
            "recommended_course_cta",
        ]
    }

    conversation_context = build_conversation_context(conversation_history)

    matched_courses = []

    if isinstance(matched_course, list):
        matched_courses = matched_course
    elif matched_course:
        matched_courses = [matched_course]

    compact_courses = [
        compact_course_for_prompt(course)
        for course in matched_courses
    ]

    compact_courses = [
        c for c in compact_courses
        if c.get("course_name")
    ]

    best_course = compact_courses[0] if compact_courses else {}

    ai_courses = []

    for c in compact_courses:
        ai_courses.append({
            "course_id": c.get("course_id"),
            "course_name": c.get("course_name"),

            # IMPORTANT
            "detail_link": c.get("detail_link"),
            "outline_link": c.get("outline_link"),
            "detail_url": c.get("detail_url"),
            "outline_url": c.get("outline_url"),

            "description": c.get("description", ""),
            "course_outline": c.get("course_outline", ""),
            "instructor": c.get("instructor", ""),
        })
    
    print("ai_courses =", ai_courses, flush=True)

    if not missing:
        print("not missing", flush=True)
        system_prompt = """
คุณคือ AI Sales Consultant สำหรับแนะนำหลักสูตรฝึกอบรม

หน้าที่:
- ตอบจาก matched_courses เท่านั้น
- ถ้ามีหลายหลักสูตร ให้เลือกตัวที่เกี่ยวข้องที่สุด
- ห้ามบอกว่าไม่มีหลักสูตร ถ้า matched_courses มีข้อมูล
- ถ้าพูดถึงชื่อหลักสูตร ต้องใช้ detail_link แทนชื่อธรรมดาเสมอ
- ห้ามพิมพ์ชื่อหลักสูตรแบบ plain text
- ห้ามสร้าง URL เอง
- ถ้าพูดถึง Course Outline ให้ใช้ outline_link
- ห้ามใช้ markdown
- ห้ามใช้ bullet
- ตอบแบบธรรมชาติ เหมือนเจ้าหน้าที่แนะนำหลักสูตร
- ตอบ 2-4 ประโยค
""".strip()

        user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

Requirement ปัจจุบัน:
{json.dumps(clean_requirements or {}, ensure_ascii=False)}

matched_courses:
{json.dumps(ai_courses, ensure_ascii=False)}

คำสั่งสำคัญ:
- ถ้าพูดชื่อหลักสูตร ต้องใช้ detail_link
- ถ้าพูดถึงเอกสารหลักสูตร ให้ใช้ outline_link
- ห้ามพิมพ์ชื่อหลักสูตรแบบปกติ
- ห้ามตอบว่าไม่มีหลักสูตร ถ้ามี matched_courses
""".strip()

        reply = ""

        async for item in _stream_text_response(
            model="gpt-4.1-mini",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ):
            if item.get("type") == "chunk":
                text = item.get("text", "")
                reply += text
                yield item

            elif item.get("type") == "done":
                reply = item.get("content") or reply

        yield {
            "type": "done",
            "content": reply,
        }

        return

    next_field = missing[0]
    label = FIELD_LABELS.get(next_field, next_field)

    system_prompt = """
คุณคือ AI Sales Consultant สำหรับแนะนำหลักสูตรฝึกอบรม

หน้าที่:
- สะท้อนความต้องการของลูกค้าแบบสั้น ๆ
- ถ้ามี matched_courses ให้แนะนำหลักสูตรที่เหมาะสมจาก matched_courses เท่านั้น
- ถามข้อมูลที่ขาดอยู่ 1 ข้ออย่างเป็นธรรมชาติ
- ห้ามสร้างลิงก์ HTML, URL, form หรือ CTA เอง
- ห้ามใช้ bullet
- ห้ามถามหลายคำถาม
- ตอบ 2-3 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

Requirement ปัจจุบัน:
{json.dumps(clean_requirements or {}, ensure_ascii=False)}

matched_courses:
{json.dumps(ai_courses, ensure_ascii=False)}

ข้อมูลที่ต้องถามเพิ่มตอนนี้คือ:
{label}

ให้ตอบแบบธรรมชาติ และถามต่อเรื่อง {label} เพียง 1 คำถาม
ห้ามสร้างลิงก์หรือ CTA เอง
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text
            yield item

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    if best_course:
        html_parts = []

        if best_course.get("detail_link"):
            html_parts.append(best_course["detail_link"])

        if best_course.get("outline_link"):
            html_parts.append(best_course["outline_link"])

        if html_parts:
            extra_html = "\n\n" + "\n\n".join(html_parts)

            yield {
                "type": "chunk",
                "text": extra_html,
            }

            reply += extra_html

    yield {
        "type": "done",
        "content": reply,
    }

# Ai assis
async def detect_intent(
    user_message: str,
    state=None,
    conversation_history: list | None = None
) -> dict:
    conversation_context = build_conversation_context(conversation_history)

    previous_intent = None
    previous_course_type = "unknown"

    if state is not None:
        previous_intent = getattr(state, "current_intent", None)

        course_context = getattr(state, "course_context", {}) or {}
        previous_course_type = course_context.get("course_type") or "unknown"

    system_prompt = """
คุณคือ Intent Router สำหรับ AI Assistant ของเว็บไซต์บริษัทฝึกอบรม

หน้าที่ของคุณ:
- อ่านข้อความล่าสุดของผู้ใช้
- อ่านบทสนทนาก่อนหน้า
- เลือก intent ที่เหมาะสมที่สุดเพียง 1 intent
- ถ้าเป็นเรื่องหลักสูตร ให้ระบุ course_type ด้วย
- ห้ามตอบนอกเหนือจาก JSON

Intent ที่อนุญาต:

1. course_search
ใช้เมื่อผู้ใช้ถามหา:
- หลักสูตร
- คอร์ส
- การอบรม
- training
- workshop
- inhouse training
- public training
- หัวข้อที่อยากเรียนหรืออยากพัฒนา

2. instructor_search
ใช้เมื่อผู้ใช้ถามหา:
- วิทยากร
- อาจารย์
- ผู้สอน
- trainer
- speaker
- ประวัติวิทยากร
- ใครเป็นคนสอน

3. company_profile
ใช้เมื่อผู้ใช้ถามเกี่ยวกับ:
- บริษัทคือใคร
- บริษัททำอะไร
- ประวัติบริษัท
- ความเป็นมา
- ข้อมูลบริษัท
- เกี่ยวกับบริษัท

4. credibility
ใช้เมื่อผู้ใช้ถามเกี่ยวกับ:
- ความน่าเชื่อถือ
- ผลงานที่ผ่านมา
- ลูกค้า
- รีวิว
- case study
- ประสบการณ์
- เคยอบรมให้ใครบ้าง

5. quotation
ใช้เมื่อผู้ใช้ต้องการ:
- ราคา
- ค่าอบรม
- ใบเสนอราคา
- quotation
- quote
- ขอราคา
- สนใจจัดอบรม
- จอง / ซื้อ / ให้ติดต่อกลับเพื่อเสนอราคา

6. contact
ใช้เมื่อผู้ใช้ถาม:
- ช่องทางติดต่อ
- เบอร์โทร
- line
- email
- แอดมิน
- ติดต่อเจ้าหน้าที่

7. irrelevant
ใช้เมื่อข้อความอยู่นอกขอบเขตเว็บไซต์บริษัทฝึกอบรมอย่างชัดเจน

8. general_qa
ใช้เมื่อ:
- เป็นคำทักทาย
- คำถามยังไม่ชัดเจน
- ถามกว้าง ๆ
- ยังจัดเข้า intent อื่นไม่ได้

course_type:
- public = ผู้ใช้ถามหารอบอบรม วันที่อบรม สมัครเรียน อบรมแบบเปิดรับสมัคร public training ราคาต่อคน
- inhouse = ผู้ใช้ถามหาอบรมในองค์กร จัดอบรมภายในบริษัท ทีมงานของเรา ปรับหลักสูตร ขอใบเสนอราคา อบรมพนักงานในบริษัท
- unknown = ผู้ใช้ถามหาหลักสูตรทั่วไป แต่ยังไม่ชัดว่าเป็น public หรือ inhouse

course_action:
- overview = ผู้ใช้ถามภาพรวม รายชื่อหลักสูตร ตารางอบรม มีคอร์สอะไรบ้าง รอบอบรมเดือนนี้ เดือนหน้า
- detail = ใช้เฉพาะเมื่อผู้ใช้ถามรายละเอียดของ "หลักสูตรที่ระบุชัดเจน" หรือ follow-up ถึงหลักสูตรเดิม เช่น "ขอรายละเอียดคอร์สนี้", "หลักสูตรนี้เรียนอะไร", "outline คอร์สนี้มีอะไร"
- unknown = ยังไม่ชัดเจน

กฎสำคัญสำหรับ inhouse:
- ถ้า course_type=inhouse และผู้ใช้พูดว่า "อยากได้ที่...", "มีที่สอน...", "หาแบบที่...", "เน้น...", "เกี่ยวกับ..." ให้ตั้ง course_action=overview ไม่ใช่ detail
- สำหรับ inhouse ให้ใช้ course_action=detail เฉพาะเมื่อผู้ใช้ถามรายละเอียดของหลักสูตรเดิมหรือชื่อหลักสูตรชัด


instructor_action:
- overview = ถามรายชื่อ/แนะนำวิทยากร
- detail = ถามประวัติ รายละเอียด ความเชี่ยวชาญ ผลงาน ของวิทยากรคนใดคนหนึ่ง
- unknown = ยังไม่ชัด

ตัวอย่าง:
- "public มีหลักสูตรอะไรบ้าง" 
  => course_action=overview

- "รายละเอียดคอร์สหัวหน้างานยุคใหม่"
  => course_action=detail

- "คอร์สนี้เรียนอะไรบ้าง"
  => course_action=detail, is_followup=true

- "สมัครคอร์สนี้"
  => course_action=register, is_followup=true

- "ขอโบรชัวร์"
  => course_action=brochure, is_followup=true

- "ราคาเท่าไหร่"
  => course_action=price, is_followup=true

ตัวอย่าง:
- "มีหลักสูตร leadership ไหม" 
  => intent=course_search, course_type=unknown

- "มีรอบอบรม leadership เดือนไหน"
  => intent=course_search, course_type=public

- "อยากสมัครคอร์ส AI"
  => intent=course_search, course_type=public

- "อยากจัดอบรม leadership ให้หัวหน้างานในบริษัท"
  => intent=course_search, course_type=inhouse

- "มี inhouse training ด้าน sales ไหม"
  => intent=course_search, course_type=inhouse

- "ขอใบเสนอราคาหลักสูตร sales"
  => intent=quotation, course_type=inhouse

กฎสำคัญ:
- ถ้าผู้ใช้ถามต่อสั้น ๆ เช่น "มีอีกไหม", "รายละเอียดเพิ่ม", "เอาอันนี้", "ใช่", "แบบนี้แหละ" ให้ดู previous_intent, previous_course_type และบทสนทนาก่อนหน้า
- ถ้า previous_intent เป็น course_search และข้อความล่าสุดเป็นคำตอบต่อเนื่องเกี่ยวกับ public/inhouse ให้คง intent=course_search
- ถ้าผู้ใช้พูดว่า public, รอบอบรม, สมัคร, วันที่อบรม ให้ course_type=public
- ถ้าผู้ใช้พูดว่า inhouse, อบรมภายใน, จัดอบรมในบริษัท, ทีมของเรา, ขอใบเสนอราคา ให้ course_type=inhouse
- ถ้าถามราคา หรือใบเสนอราคา ให้เลือก quotation แม้ก่อนหน้านี้จะเป็น course_search
- ถ้าถามคนสอนหรือวิทยากร ให้เลือก instructor_search
- ถ้าไม่แน่ใจ ให้เลือก general_qa
- ตอบ JSON เท่านั้น
- ห้ามใช้ markdown
- confidence เป็นเลข 0 ถึง 1

รูปแบบ JSON:
{
  "intent": "course_search",
  "course_type": "unknown",
  "course_action": "unknown",
  "instructor_action": "unknown",
  "confidence": 0.0,
  "reason": "",
  "topic": "",
  "is_followup": false
}
""".strip()

    user_prompt = f"""
ข้อความล่าสุดของผู้ใช้:
{user_message}

previous_intent:
{previous_intent}

previous_course_type:
{previous_course_type}

บทสนทนาก่อนหน้า:
{conversation_context}

วิเคราะห์ intent และตอบเป็น JSON เท่านั้น
""".strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
    )

    text = (result.get("content") or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    allowed_intents = [
        "course_search",
        "instructor_search",
        "company_profile",
        "credibility",
        "quotation",
        "contact",
        "irrelevant",
        "general_qa",
    ]

    allowed_course_types = [
        "public",
        "inhouse",
        "unknown",
    ]

    allowed_course_actions = [
    "overview",
    "detail",
    "unknown",
    ]

    allowed_instructor_actions = [
    "overview",
    "detail",
    "unknown",
]
    
    try:
        data = json.loads(text)

        intent = data.get("intent", "general_qa")

        if intent not in allowed_intents:
            intent = "general_qa"

        course_type = data.get("course_type", "unknown")

        if course_type not in allowed_course_types:
            course_type = "unknown"

        if intent != "course_search" and intent != "quotation":
            course_type = "unknown"

        course_action = data.get("course_action", "unknown")

        if course_action not in allowed_course_actions:
            course_action = "unknown"

        if intent not in ["course_search", "quotation"]:
            course_action = "unknown"

        instructor_action = data.get("instructor_action", "unknown")

        if instructor_action not in allowed_instructor_actions:
            instructor_action = "unknown"

        if intent != "instructor_search":
            instructor_action = "unknown"

        return {
            "intent": intent,
            "course_type": course_type,
            "course_action": course_action,
            "instructor_action": instructor_action,
            "confidence": float(data.get("confidence", 0.5)),
            "reason": data.get("reason", ""),
            "topic": data.get("topic", ""),
            "is_followup": bool(data.get("is_followup", False)),
            "previous_intent": previous_intent,
            "previous_course_type": previous_course_type,
        }

    except Exception as e:
        print("DETECT INTENT ERROR =", str(e), flush=True)
        print("DETECT INTENT RAW =", text, flush=True)

        return {
            "intent": "general_qa",
            "course_type": "unknown",
            "course_action": "unknown",
            "instructor_action": "unknown",
            "confidence": 0.3,
            "reason": "parse_failed",
            "topic": "",
            "is_followup": False,
            "previous_intent": previous_intent,
            "previous_course_type": previous_course_type,
        }
    
async def handle_general_qa(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์บริษัทฝึกอบรม

หน้าที่:
- ตอบคำถามทั่วไปของผู้ใช้
- ช่วยพาผู้ใช้ไปยังเรื่องที่เกี่ยวข้อง เช่น หลักสูตร วิทยากร ข้อมูลบริษัท ความน่าเชื่อถือ ช่องทางติดต่อ และใบเสนอราคา
- ตอบแบบเป็นธรรมชาติ สุภาพ มืออาชีพ
- ไม่ตอบเหมือนข้อความสำเร็จรูป
- ถ้าผู้ใช้ยังถามกว้าง ให้ถามต่อ 1 คำถามเพื่อช่วยคัดกรองความต้องการ
- ห้ามใช้ markdown
- ตอบ 2-4 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบกลับในฐานะ AI Assistant ของเว็บไซต์
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text
            yield {
                "type": "chunk",
                "text": text
            }

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    state.mode = "normal"
    state.last_answer = reply
    state.last_step = "general_qa"

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": "answered",
        "reason": "general_qa",
        "state": state,
        "source": "ai_assistant",
    }

    return

async def fetch_inhouse_course_detail(
    course_id: str | int | None = None,
    course_name: str | None = None,
) -> dict:

    print("\n===== START fetch_inhouse_course_detail =====", flush=True)
    print("course_id =", course_id, flush=True)

    if not course_id:
        print("❌ no course_id", flush=True)
        return {}

    try:
        course_id = int(str(course_id).strip())
    except Exception:
        print("❌ invalid course_id", flush=True)
        return {}

    sql = """
        SELECT
            *
        FROM inhouse_course
        WHERE ICourse_no = ?
        LIMIT 1
    """

    rows = run_query_bridge(sql, [course_id])

    print("DETAIL ROWS =", rows, flush=True)

    if not rows:
        print("❌ no rows", flush=True)
        return {}

    if (
        isinstance(rows, list)
        and len(rows) > 0
        and isinstance(rows[0], dict)
        and "error" in rows[0]
    ):
        print("❌ SQL ERROR =", rows[0], flush=True)
        return {}

    row = rows[0]

    if not isinstance(row, dict):
        print("❌ invalid row type", type(row), flush=True)
        return {}

    detail = {
        "course_id": row.get("ICourse_no"),
        "course_name": (
            row.get("course_name")
            or row.get("ICourse_name")
            or ""
        ),
        "course_outline": (
            row.get("course_outline")
            or row.get("outline")
            or ""
        ),
        "objective": (
            row.get("objective")
            or ""
        ),
        "target_group": (
            row.get("target_group")
            or ""
        ),
        "detail_text": (
            row.get("detail_text")
            or row.get("description")
            or ""
        ),
        "raw": row,
    }

    print("✅ DETAIL =", detail, flush=True)
    print("===== END fetch_inhouse_course_detail =====\n", flush=True)

    return detail

async def fetch_inhouse_courses_by_ids(course_ids: list) -> list[dict]:
    clean_ids = []

    for x in course_ids or []:
        try:
            clean_ids.append(int(str(x).strip()))
        except Exception:
            continue

    if not clean_ids:
        return []

    placeholders = ",".join(["?"] * len(clean_ids))

    sql = f"""
        SELECT
            `ICourse_no`,`ICourse_category`,`ICourse_nameEN`,`ICourse_nameTH`,`ICourse_description`,`ICourse_rewrite`,`ICourse_outline`,`ICourse_instructor`
        FROM inhouse_course
        WHERE ICourse_no IN ({placeholders})
    """

    rows = run_query_bridge(sql, clean_ids)
    print("rows =", rows, flush=True)
    if not rows:
        return []

    if (
        isinstance(rows, list)
        and rows
        and isinstance(rows[0], dict)
        and "error" in rows[0]
    ):
        print("SQL ERROR fetch_inhouse_courses_by_ids =", rows[0], flush=True)
        return []

    row_map = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        try:
            course_id = int(row.get("ICourse_no"))
        except Exception:
            continue

        row_map[course_id] = row

    result = []

    for cid in clean_ids:
        if cid in row_map:
            result.append(row_map[cid])

    return result

async def handle_irrelevant(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์บริษัทฝึกอบรม

หน้าที่:
- ตอบเมื่อผู้ใช้ถามเรื่องที่อยู่นอกขอบเขตของเว็บไซต์ฝึกอบรม
- ไม่ปฏิเสธแข็ง
- ตอบสุภาพ เป็นธรรมชาติ
- พยายามพาผู้ใช้กลับมาที่เรื่องหลักสูตร วิทยากร การพัฒนาทักษะ ข้อมูลบริษัท หรือการขอใบเสนอราคา
- ห้ามใช้ markdown
- ตอบ 2-3 ประโยค
- ปิดท้ายด้วยคำถาม 1 คำถาม
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบกลับแบบสุภาพและพากลับเข้าสู่เรื่องที่เว็บไซต์ช่วยได้
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.55,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text
            yield {
                "type": "chunk",
                "text": text
            }

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    state.mode = "normal"
    state.last_answer = reply
    state.last_step = "irrelevant"

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": "answered",
        "reason": "irrelevant",
        "state": state,
        "source": "ai_assistant",
    }

    return

async def build_inhouse_discovery_reply(
    user_message: str,
    requirements: dict,
    conversation_history: list | None = None,
):
    conversation_context = build_conversation_context(conversation_history)

    system_prompt = """
คุณคือ AI Sales Consultant ของเว็บไซต์ En-Training

หน้าที่:
- ช่วยลูกค้าที่สนใจหลักสูตร In-house Training
- ตอบแบบที่ปรึกษา ไม่ใช่ chatbot แข็ง ๆ
- ถ้ายังไม่รู้หัวข้อชัด ให้ช่วยชวนคุยเพื่อค้นหาโจทย์ ปัญหา หรือเป้าหมายของทีม
- สามารถยกตัวอย่างหมวดหลักสูตรได้สั้น ๆ เช่น ภาวะผู้นำ การขาย การสื่อสาร การบริการ การทำงานเป็นทีม
- ถามต่อเพียง 1 คำถาม
- ห้ามใช้ markdown
- ตอบ 2-4 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

requirements:
{json.dumps(requirements or {}, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบลูกค้าแบบที่ปรึกษาและถามต่อ 1 คำถาม
""".strip()

    async for item in _stream_text_response(
        model="gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    ):
        yield item


async def build_inhouse_topic_not_found_reply(
    user_message: str,
    requirements: dict,
    conversation_history: list | None = None,
):
    conversation_context = build_conversation_context(conversation_history)

    system_prompt = """
คุณคือ AI Sales Consultant ของเว็บไซต์ En-Training

หน้าที่:
- ตอบเมื่อลูกค้าสนใจ In-house Training แต่ระบบยังไม่พบหลักสูตรที่ตรงชัดเจน
- ห้ามบอกแข็ง ๆ ว่าไม่มี
- ให้ตอบแบบที่ปรึกษา และชวนลูกค้าขยายโจทย์อีกนิด
- ถามต่อเพียง 1 คำถาม
- ห้ามใช้ markdown
- ตอบ 2-4 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

requirements:
{json.dumps(requirements or {}, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบแบบธรรมชาติและถามต่อเพื่อให้ค้นหาหลักสูตรได้ตรงขึ้น
""".strip()

    async for item in _stream_text_response(
        model="gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    ):
        yield 

async def classify_inhouse_need(
    user_message: str,
    course_context: dict,
    conversation_history: list | None = None,
) -> dict:
    conversation_context = build_conversation_context(conversation_history)

    system_prompt = """
คุณคือ In-house Training Need Classifier

หน้าที่:
- วิเคราะห์ว่าผู้ใช้ต้องการอะไรเกี่ยวกับ In-house Training
- ห้ามตอบเป็นข้อความคุยกับลูกค้า
- ตอบ JSON เท่านั้น

need_type:
- generic_interest = สนใจ inhouse กว้าง ๆ ยังไม่มีหัวข้อชัด เช่น แนะนำหลักสูตร inhouse ให้หน่อย
- topic_search = มีหัวข้อหรือทักษะชัด เช่น AI, Leadership, Sales, Communication, หัวหน้างาน
- pain_point_search = มีปัญหาหรือโจทย์ชัด เช่น ทีมขายปิดการขายไม่ได้ หัวหน้างานสื่อสารไม่ดี
- detail_request = ขอรายละเอียดหลักสูตร เช่น รายละเอียดคอร์สนี้ outline เรียนอะไร
- ask_more = ขอหลักสูตรเพิ่ม เช่น มีอีกไหม ขออีกตัวเลือก
- quotation = ขอราคา ใบเสนอราคา จัดอบรม
- unclear = ยังไม่ชัด

ให้สกัด:
- topic
- pain_point
- target_group
- search_query
- should_search

กฎ:
- ถ้าเป็นคำกว้าง เช่น "แนะนำ inhouse", "มีหลักสูตรอะไรบ้าง" โดยไม่มีหัวข้อ ให้ should_search=false
- ถ้ามีหัวข้อเฉพาะ เช่น "AI", "Leadership", "การขาย", "สื่อสาร" ให้ should_search=true
- ถ้าเป็น detail_request แต่ไม่มี matched_course เดิม ให้ should_search=false
- search_query ต้องเป็นข้อความธรรมชาติสำหรับค้นหาหลักสูตร

JSON:
{
  "need_type": "generic_interest",
  "should_search": false,
  "topic": "",
  "pain_point": "",
  "target_group": "",
  "search_query": "",
  "reason": ""
}
""".strip()

    user_prompt = f"""
ข้อความล่าสุด:
{user_message}

course_context:
{json.dumps(course_context or {}, ensure_ascii=False)}

บทสนทนาก่อนหน้า:
{conversation_context}

วิเคราะห์ความต้องการของผู้ใช้
""".strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    text = (result.get("content") or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
    except Exception:
        return {
            "need_type": "unclear",
            "should_search": False,
            "topic": "",
            "pain_point": "",
            "target_group": "",
            "search_query": "",
            "reason": "parse_failed",
        }

    allowed = [
        "generic_interest",
        "topic_search",
        "pain_point_search",
        "detail_request",
        "ask_more",
        "quotation",
        "unclear",
    ]

    need_type = data.get("need_type", "unclear")
    if need_type not in allowed:
        need_type = "unclear"

    return {
        "need_type": need_type,
        "should_search": bool(data.get("should_search", False)),
        "topic": data.get("topic", ""),
        "pain_point": data.get("pain_point", ""),
        "target_group": data.get("target_group", ""),
        "search_query": data.get("search_query", ""),
        "reason": data.get("reason", ""),
    }
# async def handle_instructor_search(req, state):

#     user_message = (req.user_message or "").strip()
#     conversation_context = build_conversation_context(state.conversation_history)

#     system_prompt = """
# คุณคือ AI Assistant ของเว็บไซต์บริษัทฝึกอบรม

# หน้าที่:
# - ตอบคำถามเกี่ยวกับวิทยากร ผู้สอน อาจารย์ trainer หรือ speaker
# - ถ้าผู้ใช้ถามต่อจากหลักสูตรเดิม เช่น "ใครสอน" ให้ตอบโดยอิงจากบริบทเดิม
# - ถ้ายังไม่มีข้อมูลวิทยากรจริงใน context ห้ามแต่งชื่อหรือประวัติเอง
# - ให้ถามต่ออย่างสุภาพว่าผู้ใช้สนใจวิทยากรด้านไหน หรือหลักสูตรไหน
# - ห้ามใช้ markdown
# - ตอบ 2-4 ประโยค
# """.strip()

#     user_prompt = f"""
# บทสนทนาก่อนหน้า:
# {conversation_context}

# course_context:
# {json.dumps(getattr(state, "course_context", {}) or {}, ensure_ascii=False)}

# instructor_context:
# {json.dumps(getattr(state, "instructor_context", {}) or {}, ensure_ascii=False)}

# ข้อความล่าสุดของผู้ใช้:
# {user_message}

# ช่วยตอบกลับในฐานะ AI Assistant ด้านวิทยากร
# """.strip()

#     reply = ""

#     async for item in _stream_text_response(
#         model="gpt-4.1-mini",
#         system_prompt=system_prompt,
#         user_prompt=user_prompt,
#         temperature=0.45,
#     ):
#         if item.get("type") == "chunk":
#             text = item.get("text", "")
#             reply += text
#             yield {
#                 "type": "chunk",
#                 "text": text
#             }

#         elif item.get("type") == "done":
#             reply = item.get("content") or reply

#     state.mode = "normal"
#     state.last_answer = reply
#     state.last_step = "instructor_search"

#     state.conversation_history.append({
#         "role": "assistant",
#         "content": reply
#     })

#     if len(state.conversation_history) > 10:
#         state.conversation_history = state.conversation_history[-10:]

#     yield {
#         "type": "done",
#         "reply": reply,
#         "status": "answered",
#         "reason": "instructor_search",
#         "state": state,
#         "source": "ai_assistant",
#     }

#     return

async def handle_company_profile(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    company_context = await fetch_company_profile_context()

    state.company_context = {
        "source_url": "https://entstaffs.entraining.net/api/about/about.php",
        "content": company_context,
    }

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์ En-Training

หน้าที่:
- ตอบคำถามเกี่ยวกับบริษัทโดยอ้างอิงข้อมูลจาก company_context
- สรุปและเรียบเรียงใหม่ด้วยภาษาธรรมชาติ เหมือนเจ้าหน้าที่บริษัทตอบเอง
- ห้ามคัดลอกข้อความจาก company_context มาตอบแบบตรงตัว
- ห้ามตอบเหมือนบทความหรือข้อความบนหน้าเว็บ
- ห้ามแต่งข้อมูลที่ไม่มีใน context
- ถ้าไม่มีข้อมูลใน context ให้บอกอย่างสุภาพว่าไม่มีข้อมูลส่วนนั้น
- ตอบให้กระชับ อบอุ่น มืออาชีพ และเป็นกันเอง
- ห้ามใช้ markdown
- ตอบ 2-4 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

company_context:
{company_context}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบคำถามจาก company_context เท่านั้น
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text

            yield {
                "type": "chunk",
                "text": text
            }

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    state.mode = "normal"
    state.last_answer = reply
    state.last_step = "company_profile"

    # กัน course card / CTA เก่าติดมาตอนตอบเรื่องบริษัท
    state.recommended_course_cta = []
    state.recommended_courses = []
    state.matched_course = None
    state.matched_course_id = None

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": "answered",
        "reason": "company_profile",
        "state": state,
        "source": "entraining_about_page",
    }

    return

async def handle_credibility(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์บริษัทฝึกอบรม

หน้าที่:
- ตอบคำถามเกี่ยวกับความน่าเชื่อถือ ผลงาน ประสบการณ์ ลูกค้า รีวิว หรือ case study ของบริษัท
- ถ้ามีข้อมูลใน company_context ให้ใช้ข้อมูลนั้นก่อน
- ห้ามแต่งรายชื่อลูกค้า จำนวนลูกค้า รีวิว หรือผลงานที่ไม่มีใน context
- ถ้าไม่มีข้อมูลจริง ให้ตอบอย่างโปร่งใส และชวนผู้ใช้ถามเรื่องหลักสูตรหรือขอข้อมูลเพิ่มเติม
- น้ำเสียงมั่นใจ สุภาพ ไม่โอ้อวดเกินจริง
- ห้ามใช้ markdown
- ตอบ 2-4 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

company_context:
{json.dumps(getattr(state, "company_context", {}) or {}, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบกลับเรื่องความน่าเชื่อถือของบริษัท
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.35,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text
            yield {
                "type": "chunk",
                "text": text
            }

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    state.mode = "normal"
    state.last_answer = reply
    state.last_step = "credibility"

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": "answered",
        "reason": "credibility",
        "state": state,
        "source": "ai_assistant",
    }

    return

async def handle_quotation(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์บริษัทฝึกอบรม

หน้าที่:
- ช่วยผู้ใช้ที่ต้องการราคา ใบเสนอราคา หรือสนใจจัดอบรม
- เก็บข้อมูลทีละน้อยอย่างเป็นธรรมชาติ
- ถ้ายังไม่รู้หัวข้ออบรม ให้ถามหัวข้อหรือหลักสูตรที่สนใจ
- ถ้ารู้หัวข้อแล้ว อาจถามจำนวนผู้เรียน กลุ่มผู้เรียน รูปแบบอบรม หรือช่วงเวลาที่ต้องการ
- ไม่ถามหลายข้อพร้อมกัน
- ห้ามบอกว่าจะติดต่อกลับภายในเวลาที่แน่นอน
- ห้ามแต่งราคาเองถ้าไม่มีข้อมูลจริง
- ห้ามใช้ markdown
- ตอบ 2-4 ประโยค และถามต่อ 1 คำถาม
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

quotation_context:
{json.dumps(getattr(state, "quotation_context", {}) or {}, ensure_ascii=False)}

course_context:
{json.dumps(getattr(state, "course_context", {}) or {}, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบกลับเพื่อพาผู้ใช้ไปสู่การขอใบเสนอราคา
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.45,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text
            yield {
                "type": "chunk",
                "text": text
            }

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    state.mode = "collecting_info"
    state.pending_action = "quotation"
    state.last_answer = reply
    state.last_step = "quotation"

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": "collecting_info",
        "reason": "quotation",
        "state": state,
        "source": "ai_assistant",
    }

    return

async def handle_contact(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์บริษัทฝึกอบรม
ติดต่อเรา
บริษัท เอ็นมาร์ก โซลูชั่น จำกัด (สถาบันฝึกอบรมเอ็นเทรนนิ่ง)
112 ซ.รามคำแหง 30/1 ถ.รามคำแหง แขวงหัวหมาก เขตบางกะปิ กรุงเทพฯ 10240
โทรศัพท์: 0-2374-8638
Hotline : 084-164-2057 / 093-649-3561
แฟกซ์ : 0-2375-2347
เจ้าหน้าที่ Public(ส่งคนออกมาร่วมอบรม) : 099-0764277 วา(ซัซวาย์) / 097-090-6448 (สุนิศา)
Website : www.entraining.net
E-mail : asst.entraining@gmail.com, cocoachentraining@gmail.com

หน้าที่:
- ตอบคำถามเกี่ยวกับช่องทางติดต่อของบริษัท
- ถ้ามีข้อมูลติดต่อใน company_context ให้ใช้ข้อมูลนั้นก่อน
- ถ้าไม่มีเบอร์โทร Line หรือ Email จริงใน context ห้ามแต่งข้อมูลเอง
- สามารถแนะนำให้ผู้ใช้ฝากรายละเอียด เช่น หัวข้ออบรม ชื่อบริษัท หรือจำนวนผู้เรียนได้
- ห้ามบอกว่าจะติดต่อกลับภายในเวลาที่แน่นอน ถ้าไม่มีข้อมูลจริง
- ห้ามใช้ markdown
- ตอบ 2-4 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

company_context:
{json.dumps(getattr(state, "company_context", {}) or {}, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบกลับเรื่องช่องทางติดต่อ
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.4,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            reply += text
            yield {
                "type": "chunk",
                "text": text
            }

        elif item.get("type") == "done":
            reply = item.get("content") or reply

    state.mode = "normal"
    state.last_answer = reply
    state.last_step = "contact"

    state.conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    yield {
        "type": "done",
        "reply": reply,
        "status": "answered",
        "reason": "contact",
        "state": state,
        "source": "ai_assistant",
    }

    return