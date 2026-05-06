import json
import re
from urllib.parse import quote

from app.modules.ai_assis.service import (
    extract_requirements,
    calc_missing_requirements,
    build_next_question,
    build_search_query,
    build_recommendation_reply,
    build_next_question_topic,
    build_more_courses_reply,
    detect_post_recommend_intent,
    build_conversation_context,
    _stream_text_response,
)

from app.modules.ai_assis.qdrant_service import (
    search_courses_from_qdrant,
    check_topic_exists_in_qdrant,
)

from app.modules.ai_assis.public_course_service import (
    fetch_public_course_context,
    fetch_public_course_detail,
)

def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", "", text)
    return text


def find_public_course_from_message(user_message: str, courses: list[dict], state=None):
    message_norm = normalize_text(user_message)

    for course in courses:
        course_name = course.get("course_name", "")
        course_name_en = course.get("course_name_en", "")

        if course_name and normalize_text(course_name) in message_norm:
            return course

        if course_name_en and normalize_text(course_name_en) in message_norm:
            return course

    if state is not None:
        course_context = getattr(state, "course_context", {}) or {}

        last_public_course = course_context.get("last_public_course")
        if isinstance(last_public_course, dict):
            return last_public_course

    return None

def get_course_payload(course):
    if isinstance(course, dict):
        payload = course.get("payload") or course
        return payload if isinstance(payload, dict) else {}

    return {}


def get_course_id(course):
    payload = get_course_payload(course)

    return (
        payload.get("course_no")
        or payload.get("OCourse_no")
        or payload.get("id")
    )


def build_public_course_url(course_name: str) -> str:
    course_name = (course_name or "").strip()

    if not course_name:
        return ""

    # หน้าแผนอบรมรวมทั้งหมด
    if (
        "ตารางอบรม" in course_name
        or "Public Course" in course_name
        or "ทั้งหมด" in course_name
        or "Entraining" in course_name
    ):
        return "https://www.entraining.net/public-course/plan/all/"

    slug = re.sub(r"\s+", "-", course_name)
    slug = re.sub(r"-+", "-", slug)

    return f"https://www.entraining.net/public-course/{slug}"

async def handle_public_course_detail(req, state):
    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    if not state.course_context:
        state.course_context = {}

    public_courses = state.course_context.get("public_context")

    if not public_courses:
        public_courses = await fetch_public_course_context()
        state.course_context["public_context"] = public_courses

    matched_course = find_public_course_from_message(
        user_message=user_message,
        courses=public_courses,
        state=state
    )

    print("matched_course =", matched_course, flush=True)

    if matched_course:
        state.course_context["last_public_course"] = matched_course

    if not matched_course:
        reply = "อยากทราบรายละเอียดของหลักสูตรไหนครับ รบกวนพิมพ์ชื่อหลักสูตร หรือบอกหัวข้อที่สนใจเพิ่มเติมได้เลยครับ"

        state.mode = "public_course_detail"
        state.last_answer = reply
        state.last_step = "public_course_detail_need_course"

        yield {
            "type": "done",
            "reply": reply,
            "status": "collecting_info",
            "reason": "public_course_detail_need_course",
            "state": state,
            "source": "entraining_public_course_detail",
        }

        return

    course_url = (matched_course.get("course_url") or "").strip()
    course_url = course_url.replace(
    "https://www.entraining.net/public-course/",
    "https://entstaffs.entraining.net/api/public-course/"
)

    if not course_url:
        reply = "ยังไม่พบลิงก์รายละเอียดของหลักสูตรนี้ครับ"

        yield {
            "type": "done",
            "reply": reply,
            "status": "not_found",
            "reason": "public_course_url_not_found",
            "state": state,
            "source": "entraining_public_course_detail",
        }

        return

    course_detail_context = await fetch_public_course_detail(course_url)

    original_course_url = course_url.replace(
    "https://entstaffs.entraining.net/api/public-course/",
    "https://www.entraining.net/public-course/"
    )
    course_detail_context["course_url"] = original_course_url

    merged_course_detail = {
        **matched_course,
        **course_detail_context,
    }


    state.course_context["course_type"] = "public"
    state.course_context["course_action"] = "detail"
    state.course_context["last_public_course"] = matched_course
    state.course_context["last_public_course_detail"] = merged_course_detail

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์ En-Training

หน้าที่:
- ตอบรายละเอียดหลักสูตร Public Training จาก public_course_detail เท่านั้น
- ใช้ข้อมูลชื่อหลักสูตร วันที่ ราคา ลิงก์สมัคร โบรชัวร์ และ course_outline ประกอบการตอบ
- ห้ามแต่งข้อมูลที่ไม่มีใน context
- ถ้าพูดถึงชื่อหลักสูตร ให้ทำเป็นลิงก์ HTML จาก course_url ที่ให้มา
- รูปแบบลิงก์หลักสูตรต้องเป็น:
<a href="URL" target="_blank" style="color:#004AAD;font-weight:700;text-decoration:none;border-bottom:1px solid #287CED;">ชื่อหลักสูตร</a>
- ห้ามสร้าง URL เอง ถ้าไม่มี course_url
- ห้ามนำชื่อ "ตารางอบรม Public Course เดือน ทั้งหมด 2569 | Entraining" ไปสร้างเป็นลิงก์หลักสูตร
- ใช้ลิงก์ /public-course/{course_name} เฉพาะเมื่อเป็นชื่อหลักสูตรจริงเท่านั้น
- ถ้ามี register_url ให้แนบลิงก์สมัครอบรม โดยใช้คำว่า:
<a href="URL" target="_blank" style="color:#004AAD;font-weight:700;text-decoration:none;border-bottom:1px solid #287CED;">สมัครอบรม</a>
- ถ้ามี brochure_url ให้แนบลิงก์ดาวน์โหลดโบรชัวร์ โดยใช้คำว่า:
<a href="URL" target="_blank" style="color:#004AAD;font-weight:700;text-decoration:none;border-bottom:1px solid #287CED;">ดาวน์โหลดโบรชัวร์</a>
- ถ้าพูดถึงตารางอบรมรวมทั้งหมด หรือภาพรวม Public Training ทั้งหมด ให้ใช้ลิงก์:
<a href="https://www.entraining.net/public-course/plan/all/" target="_blank" style="color:#004AAD;font-weight:700;text-decoration:none;border-bottom:1px solid #287CED;">ตารางอบรม Public Training ทั้งหมด</a>
- ห้ามใช้ markdown
- ตอบแบบธรรมชาติ เหมือนเจ้าหน้าที่ช่วยแนะนำ
- ตอบ 3-6 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

public_course_detail:
{json.dumps(merged_course_detail, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบรายละเอียดหลักสูตรนี้จาก public_course_detail เท่านั้น
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-nano",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
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

    state.mode = "public_course_detail"
    state.last_answer = reply
    state.last_step = "public_course_detail"

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
        "reason": "public_course_detail",
        "state": state,
        "source": "entraining_public_course_detail",
    }

    return

async def handle_inhouse_course_search(req, state):

    user_message = (req.user_message or "").strip()

    skip_extract = False

    # =========================
    # POST RECOMMEND FOLLOW-UP
    # =========================
    if state.mode == "course_post_recommend":

        post_intent = await detect_post_recommend_intent(
            user_message=user_message,
            requirements=state.course_context or {},
            conversation_history=state.conversation_history
        )

        print("COURSE POST RECOMMEND INTENT =", post_intent, flush=True)

        intent = post_intent.get("intent", "unclear")
        state.last_step = intent

        if intent == "new_requirement":

            state.course_context = {}
            state.recommended_courses = []
            state.recommended_course_cta = []
            state.matched_course = None
            state.matched_course_id = None
            state.mode = "course_discovery"

            state.conversation_history = [{
                "role": "user",
                "content": user_message
            }]

        elif intent == "ask_more_courses":

            skip_extract = True
            state.mode = "course_discovery"
            state.last_step = "ask_more_courses"

        elif intent == "refine_requirement":

            state.mode = "course_discovery"

        else:

            state.mode = "course_discovery"

    # =========================
    # NORMAL COURSE DISCOVERY
    # =========================
    if not skip_extract:

        old_context = dict(state.course_context or {})

        old_req = " ".join(
            str(v).strip()
            for k, v in sorted(old_context.items())
            if k not in [
                "matched_course",
                "missing_requirements",
                "requirement_ready",
                "search_query",
                "course_type",
            ] and v
        ).strip()

        new_context = await extract_requirements(
            user_message=user_message,
            current_requirements=state.course_context or {},
            conversation_history=state.conversation_history
        )

        new_req = " ".join(
            str(v).strip()
            for k, v in sorted((new_context or {}).items())
            if k not in [
                "matched_course",
                "missing_requirements",
                "requirement_ready",
                "search_query",
                "course_type",
            ] and v
        ).strip()

        if "course_type" not in new_context:
            new_context["course_type"] = "inhouse"

        state.course_context = new_context

        new_topic = new_context.get("topic")
        old_topic = old_context.get("topic")

        print(
            f"COURSE REQ CHECK | old_req={old_req} | new_req={new_req}",
            flush=True
        )

        topic_check_courses = None
        course_id = None

        missing = calc_missing_requirements(state.course_context)

        state.course_context["missing_requirements"] = missing
        state.course_context["requirement_ready"] = len(missing) == 0

        # =========================
        # CASE: NO REQUIREMENT YET
        # =========================
        if not new_req:

            if missing:

                reply = ""

                async for item in build_next_question(
                    state.course_context,
                    missing,
                    state.conversation_history
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

                state.mode = "course_discovery"
                state.last_answer = reply
                state.last_step = "course_ask_requirement"

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
                    "reason": "missing_course_requirement",
                    "state": state,
                    "source": "ai_assistant_course_discovery",
                }

                return

            state.mode = "course_post_recommend"

        # =========================
        # IMPORTANT: CHECK TOPIC FIRST
        # =========================
        if new_req and new_req != old_req:

            topic_check_courses, course_id = await check_topic_exists_in_qdrant(
                new_req,
                limit=1
            )

        elif new_req and old_req and new_req == old_req:

            topic_check_courses = state.course_context.get("matched_course")
            course_id = getattr(state, "matched_course_id", None)

        # =========================
        # TOPIC NOT FOUND
        # =========================
        if new_req and new_topic and not topic_check_courses and not old_topic:

            reply = (
                "ตอนนี้ยังไม่พบหลักสูตรที่ตรงกับหัวข้อนี้โดยตรงครับ "
                "ลองบอกหัวข้อหรือทักษะที่อยากพัฒนาเพิ่มเติมอีกนิดได้ไหมครับ"
            )

            state.last_answer = reply
            state.last_step = "course_topic_not_found"
            state.mode = "course_post_recommend"

            state.conversation_history.append({
                "role": "assistant",
                "content": reply
            })

            if len(state.conversation_history) > 10:
                state.conversation_history = state.conversation_history[-10:]

            yield {
                "type": "done",
                "reply": reply,
                "status": "irrelevant",
                "reason": "course_topic_not_found",
                "state": state,
                "source": "ai_assistant_course",
            }

            return

        # =========================
        # TOPIC FOUND: RECOMMEND FIRST, THEN ASK
        # =========================
        if topic_check_courses:

            state.course_context = new_context
            state.course_context["topic"] = new_topic
            state.course_context["matched_course"] = topic_check_courses

            missing = calc_missing_requirements(state.course_context)

            state.course_context["missing_requirements"] = missing
            state.course_context["requirement_ready"] = len(missing) == 0

            reply = ""

            async for item in build_next_question_topic(
                requirements=state.course_context,
                missing=missing,
                conversation_history=state.conversation_history,
                matched_course=state.course_context.get("matched_course")
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

            state.last_answer = reply
            state.matched_course = state.course_context["matched_course"]
            state.matched_course_id = course_id
            state.last_step = "course_ask_requirement_topic"

            old_courses = state.recommended_courses or []

            exists_ids = set()

            for course in old_courses:
                cid = get_course_id(course)
                if cid:
                    exists_ids.add(str(cid))

            matched_payload = get_course_payload(topic_check_courses)

            matched_course_id = (
                matched_payload.get("course_no")
                or matched_payload.get("OCourse_no")
                or matched_payload.get("id")
                or course_id
            )

            if matched_course_id and str(matched_course_id) not in exists_ids:
                state.recommended_courses = old_courses + [topic_check_courses]
            else:
                state.recommended_courses = old_courses

            if not missing:
                state.mode = "course_post_recommend"
                status = "ready_to_recommend"
                reason = "course_requirement_complete"
            else:
                state.mode = "course_discovery"
                status = "collecting_info"
                reason = "matched_course_need_more_info"

            state.conversation_history.append({
                "role": "assistant",
                "content": reply
            })

            if len(state.conversation_history) > 10:
                state.conversation_history = state.conversation_history[-10:]

            yield {
                "type": "done",
                "reply": reply,
                "status": status,
                "reason": reason,
                "state": state,
                "source": "ai_assistant_course_discovery",
            }

            return

        if old_topic:
            new_context["topic"] = old_topic
            state.course_context = new_context
            print(f"COURSE KEEP OLD TOPIC: {old_topic}", flush=True)

    # =========================
    # ASK MORE COURSES
    # =========================
    else:

        search_query = await build_search_query(state.course_context)

        state.course_context["search_query"] = search_query

        excluded_courses = []

        for course in state.recommended_courses or []:
            course_no = get_course_id(course)
            if course_no:
                excluded_courses.append(str(course_no))

        courses = await search_courses_from_qdrant(
            search_query,
            limit=1,
            excluded_courses=excluded_courses
        )

        if courses:

            first_course = courses[0]
            payload = get_course_payload(first_course)

            if payload:

                new_course_id = (
                    payload.get("course_no")
                    or payload.get("OCourse_no")
                    or payload.get("id")
                )

                new_course_name = (
                    payload.get("course_name")
                    or payload.get("vdo_name")
                    or payload.get("Course_name")
                    or payload.get("title")
                )

                state.matched_course = new_course_name
                state.matched_course_id = new_course_id

                if state.course_context is None:
                    state.course_context = {}

                state.course_context["matched_course"] = new_course_name

        old_courses = state.recommended_courses or []
        merged_courses = []
        seen_course_ids = set()

        for course in old_courses + (courses or []):
            course_no = get_course_id(course)

            if course_no:
                key = str(course_no)
                if key in seen_course_ids:
                    continue

                seen_course_ids.add(key)

            merged_courses.append(course)

        state.recommended_courses = merged_courses

        course_cta = []

        for c in courses or []:

            payload = get_course_payload(c)

            if not payload:
                continue

            course_no = (
                payload.get("course_no")
                or payload.get("OCourse_no")
                or payload.get("id")
            )

            course_name = (
                payload.get("course_name")
                or payload.get("vdo_name")
                or payload.get("Course_name")
                or payload.get("title")
            )

            if course_no and course_name:
                course_cta.append({
                    "course_no": course_no,
                    "course_name": course_name,
                })

        old_cta = state.recommended_course_cta or []
        merged_cta = []
        seen = set()

        for c in old_cta + course_cta:

            course_no = c.get("course_no")

            if not course_no:
                continue

            course_no_key = str(course_no)

            if course_no_key in seen:
                continue

            seen.add(course_no_key)

            merged_cta.append({
                "course_no": course_no,
                "course_name": c.get("course_name", "")
            })

        state.recommended_course_cta = merged_cta

        reply = ""

        async for item in build_more_courses_reply(
            requirements=state.course_context,
            courses=courses
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

        state.mode = "course_post_recommend"
        state.last_answer = reply
        state.last_step = "course_ask_more_courses"

        state.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        if len(state.conversation_history) > 10:
            state.conversation_history = state.conversation_history[-10:]

        yield {
            "type": "done",
            "reply": reply,
            "status": "recommended",
            "reason": "ask_more_courses",
            "state": state,
            "source": "ai_assistant_course_qdrant",
        }

        return

async def handle_public_course_search(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    public_course_context = await fetch_public_course_context()

    public_course_context_with_links = []

    for item in public_course_context:
        if isinstance(item, dict):
            course_name = (item.get("course_name") or "").strip()
            course_url = (item.get("course_url") or "").strip()
            course_name_en = (item.get("course_name_en") or "").strip()
            course_date = (item.get("course_date") or "").strip()
            month = (item.get("month") or "").strip()
            price = (item.get("price") or "").strip()
            badge = (item.get("badge") or "").strip()
            image_url = (item.get("image_url") or "").strip()
            register_url = (item.get("register_url") or "").strip()
            brochure_url = (item.get("brochure_url") or "").strip()
        else:
            course_name = str(item or "").strip()
            course_url = build_public_course_url(course_name)
            course_name_en = ""
            course_date = ""
            month = ""
            price = ""
            badge = ""
            image_url = ""
            register_url = ""
            brochure_url = ""

        if not course_name:
            continue

        public_course_context_with_links.append({
            "course_name": course_name,
            "course_name_en": course_name_en,
            "course_url": course_url,
            "course_date": course_date,
            "month": month,
            "price": price,
            "badge": badge,
            "image_url": image_url,
            "register_url": register_url,
            "brochure_url": brochure_url,
        })

    if not state.course_context:
        state.course_context = {}

    state.course_context["course_type"] = "public"
    state.course_context["public_source_url"] = "https://www.entraining.net/public-course/plan/all/"
    state.course_context["public_context"] = public_course_context_with_links
    state.course_context["last_public_courses"] = public_course_context_with_links[:5]

    if public_course_context_with_links:
        state.course_context["last_public_course"] = public_course_context_with_links[0]

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์ En-Training

หน้าที่:
- ตอบคำถามเกี่ยวกับหลักสูตรแบบ Public Training จาก public_course_context เท่านั้น
- สรุปและแนะนำหลักสูตรที่เกี่ยวข้องกับสิ่งที่ผู้ใช้ถาม
- ถ้าพูดถึงชื่อหลักสูตร ให้ทำเป็นลิงก์ HTML จาก course_url ที่ให้มา
- รูปแบบลิงก์ต้องเป็น <a href="URL"
target="_blank"
style="
color:#004AAD;
font-weight:700;
text-decoration:none;
border-bottom:1px solid #287CED;
">
ชื่อหลักสูตร
</a>

- ห้ามสร้าง URL เอง ถ้าไม่มี course_url
- ห้ามแต่งชื่อหลักสูตร วันที่ ราคา หรือรายละเอียดที่ไม่มีใน context
- public_course_context อาจมี field course_name_en, course_date, month, price, badge, register_url, brochure_url ให้ใช้ข้อมูลเหล่านี้ประกอบการตอบ
- ถ้าผู้ใช้ถามวันที่อบรม ให้ดูจาก course_date
- ถ้าผู้ใช้ถามเดือนที่เปิดอบรม ให้ดูจาก month
- ถ้าผู้ใช้ต้องการสมัคร ให้ทำลิงก์จาก register_url โดยใช้คำว่า สมัครอบรม
- ถ้าผู้ใช้ต้องการโบรชัวร์ ให้ทำลิงก์จาก brochure_url โดยใช้คำว่า ดาวน์โหลดโบรชัวร์
- ถ้าผู้ใช้ถามราคา ให้ดูจาก price
- ถ้าผู้ใช้ต้องการสมัคร ให้ใช้ register_url ถ้ามี
- ถ้าผู้ใช้ต้องการโบรชัวร์ ให้ใช้ brochure_url ถ้ามี
- ถ้าพูดถึงตารางอบรมรวมทั้งหมด หรือภาพรวม Public Training ทั้งหมด ให้ใช้ลิงก์ https://www.entraining.net/public-course/plan/all/
- ห้ามนำชื่อ "ตารางอบรม Public Course เดือน ทั้งหมด 2569 | Entraining" ไปสร้างเป็นลิงก์หลักสูตร
- ใช้ลิงก์ /public-course/{course_name} เฉพาะเมื่อเป็นชื่อหลักสูตรจริงเท่านั้น
- ถ้า context ไม่มีข้อมูลตรงกับสิ่งที่ถาม ให้บอกอย่างสุภาพว่าไม่พบข้อมูลชัดเจนในรอบ public ที่มีอยู่
- ถ้ามีข้อมูลหลายหลักสูตร ให้แนะนำเฉพาะตัวที่เกี่ยวข้องที่สุด
- ตอบแบบธรรมชาติ เหมือนเจ้าหน้าที่ช่วยแนะนำ
- ห้ามคัดลอก context มาทั้งดุ้น
- ห้ามใช้ markdown
- ตอบ 2-5 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

public_course_context:
{json.dumps(public_course_context_with_links, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบคำถามหรือแนะนำหลักสูตร Public Training จาก public_course_context เท่านั้น
""".strip()

    reply = ""

    async for item in _stream_text_response(
        model="gpt-4.1-nano",
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

    state.mode = "public_course_search"
    state.last_answer = reply
    state.last_step = "public_course_search"

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
        "reason": "public_course_search",
        "state": state,
        "source": "entraining_public_course_page",
    }

    return

async def handle_course_search(req, state):

    course_context = getattr(state, "course_context", {}) or {}
    course_type = course_context.get("course_type") or "unknown"
    course_action = getattr(state, "course_context", {}).get("course_action", "unknown")

    if course_type == "public":
        if course_action == "detail":
            async for item in handle_public_course_detail(req=req, state=state):
                yield item
            return

        async for item in handle_public_course_search(
            req=req,
            state=state
        ):
            yield item

        return

    if course_type == "inhouse":

        async for item in handle_inhouse_course_search(
            req=req,
            state=state
        ):
            yield item

        return

    async for item in ask_course_type(
        req=req,
        state=state
    ):
        yield item

    return

async def ask_course_type(req, state):

    user_message = (req.user_message or "").strip()
    conversation_context = build_conversation_context(state.conversation_history)

    course_context = getattr(state, "course_context", {}) or {}

    system_prompt = """
คุณคือ AI Assistant ของเว็บไซต์บริษัทฝึกอบรม

หน้าที่:
- ช่วยผู้ใช้ที่สนใจหลักสูตร แต่ยังไม่ชัดว่าอยากดูแบบ Public Training หรือ In-house Training
- ห้ามถามแข็ง ๆ เหมือนเลือกเมนู
- ให้สะท้อนหัวข้อที่ผู้ใช้สนใจก่อน ถ้ามี
- อธิบายความต่างแบบสั้น ๆ:
  Public Training = รอบอบรมทั่วไปที่เปิดให้สมัคร
  In-house Training = จัดอบรมภายในองค์กรและปรับให้เหมาะกับทีม
- ถามต่อ 1 คำถามอย่างเป็นธรรมชาติ
- ห้ามใช้ markdown
- ตอบ 2-4 ประโยค
""".strip()

    user_prompt = f"""
บทสนทนาก่อนหน้า:
{conversation_context}

course_context:
{json.dumps(course_context or {}, ensure_ascii=False)}

ข้อความล่าสุดของผู้ใช้:
{user_message}

ช่วยตอบกลับเพื่อถามต่ออย่างเป็นธรรมชาติว่าอยากดูแบบ Public หรือ In-house
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

    state.mode = "course_select_type"
    state.last_answer = reply
    state.last_step = "ask_course_type"

    if not state.course_context:
        state.course_context = {}

    state.course_context["course_type"] = "unknown"

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
        "reason": "ask_course_type",
        "state": state,
        "source": "ai_assistant_course_type",
    }

    return