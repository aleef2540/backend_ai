import json
import re
from urllib.parse import quote

from app.modules.ai_assis.service import (
    build_conversation_context,
    _stream_text_response,
)

from app.modules.ai_assis.instructor_service import (
    fetch_instructor_context,
    fetch_instructor_detail,
)

from app.shared.ai.openai_client import call_openai_chat_full

def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", "", text)
    return text


async def match_instructor_by_ai(
    user_message: str,
    instructors: list[dict],
    conversation_history: list | None = None,
    state=None
) -> dict | None:

    conversation_context = build_conversation_context(conversation_history)

    compact_instructors = []

    for i, instructor in enumerate(instructors):
        compact_instructors.append({
            "index": i,
            "instructor_name": instructor.get("instructor_name", ""),
            "style": instructor.get("style", ""),
            "designation": instructor.get("designation", ""),
        })

    last_instructor = {}

    if state is not None:
        instructor_context = getattr(state, "instructor_context", {}) or {}
        last_instructor = instructor_context.get("last_instructor") or {}

    system_prompt = """
คุณคือ Instructor Matcher

หน้าที่:
- เลือกวิทยากรที่ผู้ใช้หมายถึงจาก instructor_candidates เท่านั้น
- ถ้าผู้ใช้พูดว่า "คนนี้", "ท่านนี้", "อาจารย์คนนี้", "วิทยากรคนนี้" ให้ดูบทสนทนาก่อนหน้าและ last_instructor
- ถ้าผู้ใช้พูดถึงชื่อบางส่วน เช่น ชื่อจริง นามสกุล หรือคำนำหน้า ให้เลือกคนที่ตรงที่สุด
- ถ้าผู้ใช้พูดถึงแนวทาง เช่น coaching, group coaching, hardskill, workshop ให้เทียบกับ style และ designation
- ถ้าไม่มั่นใจ ให้ตอบ matched_index = null
- ห้ามแต่งชื่อวิทยากร
- ตอบ JSON เท่านั้น

รูปแบบ JSON:
{
  "matched_index": 0,
  "confidence": 0.0,
  "reason": ""
}
""".strip()

    user_prompt = f"""
ข้อความล่าสุดของผู้ใช้:
{user_message}

บทสนทนาก่อนหน้า:
{conversation_context}

last_instructor:
{json.dumps(last_instructor, ensure_ascii=False)}

instructor_candidates:
{json.dumps(compact_instructors, ensure_ascii=False)}

เลือกวิทยากรที่ตรงที่สุด
""".strip()

    result = await call_openai_chat_full(
        model="gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
    )

    text = (result.get("content") or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
        matched_index = data.get("matched_index", None)
        confidence = float(data.get("confidence", 0))

        if matched_index is None:
            return None

        matched_index = int(matched_index)

        if confidence < 0.55:
            return None

        if matched_index < 0 or matched_index >= len(instructors):
            return None

        return instructors[matched_index]

    except Exception:
        return None

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

async def handle_instructor_list(req, state):

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
คุณคือ AI Assistant ชื่อ En-Assistant ของเว็บไซต์บริษัทฝึกอบรม En-Training เพศชาย

หน้าที่:
- ตอบคำถามเกี่ยวกับวิทยากร ผู้สอน อาจารย์ trainer หรือ speaker จาก instructor_context เท่านั้น
- สรุปและแนะนำวิทยากรที่เกี่ยวข้องกับสิ่งที่ผู้ใช้ถาม
- ถ้าพูดถึงชื่อวิทยากร ให้ทำเป็นลิงก์ HTML จาก instructor_url ที่ให้มา
- ถ้าพูดถึงชื่อวิทยากร ต้องสร้าง HTML link ให้ครบทั้ง tag เท่านั้น
- รูปแบบต้องเป็น:
<a href="URL" target="_blank" style="color:#004AAD;font-weight:700;text-decoration:none;border-bottom:1px solid #287CED;">ชื่อวิทยากร</a>
- ห้ามตัด <a href=
- ห้ามตอบเฉพาะ style หรือ target

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
        model="gpt-4o-mini",
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

async def handle_instructor_detail(req, state):
    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    if not state.instructor_context:
        state.instructor_context = {}

    instructors = await fetch_instructor_context()

    instructor_context_with_links = []

    for item in instructors:
        if isinstance(item, dict):
            instructor_context_with_links.append(item)

    matched_instructor = await match_instructor_by_ai(
    user_message=user_message,
    instructors=instructor_context_with_links,
    conversation_history=state.conversation_history,
    state=state
    )

    # if not matched_instructor:
    #     matched_instructor = find_instructor_from_message(
    #         user_message=user_message,
    #         instructors=instructor_context_with_links,
    #         state=state
    #     )

    if not matched_instructor:
        reply = "อยากทราบรายละเอียดของวิทยากรท่านไหนครับ รบกวนพิมพ์ชื่อวิทยากร หรือบอกแนวทาง/หลักสูตรที่สนใจได้เลยครับ"

        state.mode = "instructor_detail"
        state.last_answer = reply
        state.last_step = "instructor_detail_need_name"

        yield {
            "type": "done",
            "reply": reply,
            "status": "collecting_info",
            "reason": "instructor_detail_need_name",
            "state": state,
            "source": "entraining_instructor_detail",
        }

        return

    instructor_url = (matched_instructor.get("instructor_url") or "").strip()

    if not instructor_url:
        reply = "ยังไม่พบลิงก์รายละเอียดของวิทยากรท่านนี้ครับ"

        yield {
            "type": "done",
            "reply": reply,
            "status": "not_found",
            "reason": "instructor_url_not_found",
            "state": state,
            "source": "entraining_instructor_detail",
        }

        return

    instructor_detail_context = await fetch_instructor_detail(instructor_url)

    merged_instructor_detail = {
        **matched_instructor,
        **instructor_detail_context,
    }

    state.instructor_context["instructor_action"] = "detail"
    state.instructor_context["last_instructor"] = matched_instructor
    state.instructor_context["last_instructor_detail"] = merged_instructor_detail

    system_prompt = """
คุณคือ AI Assistant ชื่อ En-Assistant ของเว็บไซต์บริษัทฝึกอบรม En-Training เพศชาย

หน้าที่:
- ตอบรายละเอียดวิทยากรจาก instructor_detail เท่านั้น
- ใช้ข้อมูลชื่อวิทยากร style designation image_url instructor_url และ profile_detail ประกอบการตอบ
- ห้ามแต่งประวัติ ความเชี่ยวชาญ ผลงาน หรือข้อมูลที่ไม่มีใน context
- ถ้าพูดถึงชื่อวิทยากร ให้ทำเป็นลิงก์ HTML จาก instructor_url ที่ให้มา
- รูปแบบลิงก์ต้องเป็น <a href="URL" target="_blank" style="color:#004AAD;font-weight:700;text-decoration:none;border-bottom:1px solid #287CED;">ชื่อวิทยากร</a>
- ห้ามใช้ markdown
- ตอบแบบธรรมชาติ เหมือนเจ้าหน้าที่ช่วยแนะนำ
- ตอบ 3-6 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

instructor_detail:
{json.dumps(merged_instructor_detail, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบรายละเอียดวิทยากรจาก instructor_detail เท่านั้น
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4o-mini",
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

    state.mode = "instructor_detail"
    state.last_answer = reply
    state.last_step = "instructor_detail"

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
        "reason": "instructor_detail",
        "state": state,
        "source": "entraining_instructor_detail",
    }

    return

async def handle_instructor_search(req, state):

    instructor_context = getattr(state, "instructor_context", {}) or {}

    instructor_action = (
        instructor_context.get("instructor_action")
        or "overview"
    )

    if instructor_action == "detail":

        async for item in handle_instructor_detail(
            req=req,
            state=state
        ):
            yield item

        return

    async for item in handle_instructor_list(
        req=req,
        state=state
    ):
        yield item

    return