import json
import re
from urllib.parse import quote

from app.modules.ai_assis.service import (
    build_conversation_context,
    _stream_text_response,
)

from app.modules.ai_assis.instructor_service import (
    fetch_instructor_context,
)


def build_instructor_url(instructor_name: str) -> str:
    instructor_name = (instructor_name or "").strip()
    print("instructor_name =", instructor_name, flush=True)
    if not instructor_name:
        return ""

    slug_name = instructor_name.strip()

    prefixes = [
        "อ.",
        "อ. ",
        "อ ",
        "อาจารย์",
        "ดร. ",
        "ดร ",
        "คุณ",
    ]

    for prefix in prefixes:
        if slug_name.startswith(prefix):
            slug_name = slug_name[len(prefix):].strip()
            break

    slug = re.sub(r"\s+", "-", slug_name)
    slug = re.sub(r"-+", "-", slug)

    return "https://www.entraining.net/expert/" + slug + "/"


async def handle_instructor_search(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    instructors = await fetch_instructor_context()

    instructor_context_with_links = []

    for item in instructors:
        if isinstance(item, dict):
            instructor_name = (item.get("instructor_name") or "").strip()
            instructor_url = (item.get("instructor_url") or "").strip() or build_instructor_url(instructor_name)
            style = (item.get("style") or "").strip()
            designation = (item.get("designation") or "").strip()
            image_url = (item.get("image_url") or "").strip()
        else:
            instructor_name = str(item or "").strip()
            instructor_url = build_instructor_url(instructor_name)
            style = ""
            designation = ""
            image_url = ""

        if not instructor_name:
            continue

        instructor_context_with_links.append({
            "instructor_name": instructor_name,
            "instructor_url": instructor_url,
            "style": style,
            "designation": designation,
            "image_url": image_url,
        })

    if not state.instructor_context:
        state.instructor_context = {}

    state.instructor_context["source_url"] = "https://www.entraining.net/expert/"
    state.instructor_context["matched_count"] = len(instructor_context_with_links)

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์ En-Training

หน้าที่:
- ตอบคำถามเกี่ยวกับวิทยากร ผู้สอน อาจารย์ trainer หรือ speaker จาก instructor_context เท่านั้น
- สรุปและแนะนำวิทยากรที่เกี่ยวข้องกับสิ่งที่ผู้ใช้ถาม
- ถ้าพูดถึงชื่อวิทยากร ให้ทำเป็นลิงก์ HTML จาก instructor_url ที่ให้มา
- รูปแบบลิงก์ต้องเป็น <a href="URL"
target="_blank"
style="
color:#004AAD;
font-weight:700;
text-decoration:none;
border-bottom:1px solid #287CED;
">
ชื่อวิทยากร
</a>

- ห้ามสร้าง URL เอง ถ้าไม่มี instructor_url
- ห้ามแต่งชื่อวิทยากร ประวัติ ตำแหน่ง หรือความเชี่ยวชาญที่ไม่มีใน context
- ถ้าพูดถึงรายชื่อวิทยากรรวมทั้งหมด หรือภาพรวมวิทยากรทั้งหมด ให้ใช้ลิงก์ https://www.entraining.net/expert/ 
- instructor_context อาจมี field style, designation, image_url ให้ใช้ข้อมูลเหล่านี้ประกอบการแนะนำ
- ถ้าผู้ใช้ถามแนวทางการสอน เช่น coaching, group coaching, hardskill, workshop ให้เทียบกับ field style
- ถ้าผู้ใช้ถามความเชี่ยวชาญหรือตำแหน่ง ให้ดูจาก designation
- ห้ามนำชื่อ "รวมวิทยากรณ, วิทยากรณ์ที่ดำเนอนการสอน" ไปสร้างเป็นลิงก์วิทยากรณ์
- ใช้ลิงก์ /expert/{instructor_name}/ เฉพาะเมื่อเป็นชื่อวิทยากรจริงเท่านั้น
- ถ้า context ไม่มีข้อมูลตรงกับสิ่งที่ถาม ให้บอกอย่างสุภาพว่าไม่พบข้อมูลวิทยากรที่ตรงชัดเจน และถามต่อว่าสนใจวิทยากรด้านไหนหรือหลักสูตรไหน
- ถ้ามีข้อมูลหลายคน ให้แนะนำเฉพาะคนที่เกี่ยวข้องที่สุด
- ตอบแบบธรรมชาติ เหมือนเจ้าหน้าที่ช่วยแนะนำ
- ห้ามคัดลอก context มาทั้งดุ้น
- ห้ามใช้ markdown
- ตอบ 2-5 ประโยค
- ปิดท้ายด้วยการแนะนำหน้ารวมวิทยากรณ์ทั้งหมด https://www.entraining.net/expert/ รูปแบบลิงก์ต้องเป็น <a href="URL"
target="_blank"
style="
color:#004AAD;
font-weight:700;
text-decoration:none;
border-bottom:1px solid #287CED;
">
รวมวิทยากรทั้งหมด
</a>
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

course_context:
{json.dumps(getattr(state, "course_context", {}) or {}, ensure_ascii=False)}

instructor_context:
{json.dumps(instructor_context_with_links, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบคำถามหรือแนะนำวิทยากรจาก instructor_context เท่านั้น
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

    state.mode = "instructor_search"
    state.last_answer = reply
    state.last_step = "instructor_search"

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
        "reason": "instructor_search",
        "state": state,
        "source": "entraining_expert_page",
    }

    return