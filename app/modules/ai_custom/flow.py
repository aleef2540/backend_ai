from app.modules.ai_custom.course_service import get_course_data_by_nos
from app.modules.ai_custom.service import (
    detect_intent,
    reply_greeting,
    reply_general,
    reply_learning,
    reply_with_topic,
    reply_out_of_scope,
    reply_ask_recommend_course,
    reply_ask_concept_with_topic,
    reply_ask_concept_no_topic,

    reply_greeting_stream,
    reply_general_stream,
    reply_out_of_scope_stream,
    reply_ask_recommend_course_stream,
    reply_ask_concept_with_topic_stream,
    reply_ask_concept_no_topic_stream,
)

from app.modules.ai_custom.schema import ChatState_aicustom
import random



def build_video_payload(video):
    if not video:
        return None

    video_url = str(video.get("video_url") or "").strip()

    return {
        "video_part": video.get("video_part"),
        "video_name": video.get("video_name"),
        "video_id": video_url,
        "embed_url": f"https://www.youtube.com/embed/{video_url}" if video_url else None
    }

def build_course_name_context(course_data) -> str:
    names = []
    seen = set()

    for row in course_data:
        course_name = str(row.get("course_name") or "").strip()

        if not course_name:
            continue

        if course_name in seen:
            continue

        seen.add(course_name)
        names.append(course_name)

    return ", ".join(names)


def find_script_by_topic(course_data, topic: str) -> str:
    topic_clean = str(topic or "").strip().lower()

    if not topic_clean or topic_clean == "unknown":
        return ""

    for row in course_data:
        course_name = str(row.get("course_name") or "").strip()
        script = str(row.get("script") or "").strip()

        if course_name.lower() == topic_clean:
            return script

    for row in course_data:
        course_name = str(row.get("course_name") or "").strip().lower()
        script = str(row.get("script") or "").strip()

        if topic_clean in course_name or course_name in topic_clean:
            return script

    return ""

def find_course_by_topic(course_data, topic: str):
    topic_clean = str(topic or "").strip().lower()

    if not topic_clean or topic_clean == "unknown":
        return None

    for row in course_data:
        course_name = str(row.get("course_name") or "").strip()
        if course_name.lower() == topic_clean:
            return row

    for row in course_data:
        course_name = str(row.get("course_name") or "").strip().lower()
        if topic_clean in course_name or course_name in topic_clean:
            return row

    return None

def find_course_by_no(course_data, course_no):
    if not course_no:
        return None

    try:
        target_no = int(course_no)
    except Exception:
        return None

    for row in course_data:
        try:
            row_no = int(row.get("course_no"))
        except Exception:
            row_no = None

        if row_no == target_no:
            return row

    return None


def detect_followup_type(message: str) -> str:
    text = str(message or "").strip().lower()

    summary_keywords = [
        "สรุป", "ย่อ", "เอาแบบสั้น", "แบบสั้น", "สั้นๆ", "สั้น ๆ", "short summary"
    ]
    example_keywords = [
        "ตัวอย่าง", "ยกตัวอย่าง", "example", "มีเคสไหม", "มีกรณีไหม"
    ]
    application_keywords = [
        "เอาไปใช้", "นำไปใช้", "ใช้ยังไง", "ทำยังไง", "ควรทำยังไง",
        "ปรับใช้", "ใช้กับงาน", "ใช้กับลูกทีม", "ประยุกต์ใช้"
    ]
    comparison_keywords = [
        "ต่างจาก", "แตกต่าง", "เปรียบเทียบ", "เทียบกับ", "ดีกว่า", "เหมือนกันไหม"
    ]
    expand_keywords = [
        "ขยาย", "เพิ่ม", "เพิ่มอีก", "อีกหน่อย", "เพิ่มเติม", "เล่าเพิ่ม", "อธิบายเพิ่ม"
    ]

    if any(k in text for k in summary_keywords):
        return "summary"

    if any(k in text for k in example_keywords):
        return "example"

    if any(k in text for k in application_keywords):
        return "application"

    if any(k in text for k in comparison_keywords):
        return "comparison"

    if any(k in text for k in expand_keywords):
        return "expand"

    return "expand"


def map_followup_answer_type(followup_type: str) -> str:
    mapping = {
        "summary": "summary_given",
        "example": "example_given",
        "application": "application_given",
        "comparison": "comparison_given",
        "expand": "followup_expanded",
    }
    return mapping.get(followup_type, "followup_expanded")

def build_application_message(user_message: str, state) -> str:
    return (
        f"{user_message}\n\n"
        f"[ANSWER_MODE]: application\n"
        f"[PREVIOUS_ANSWER]: {state.last_answer or ''}\n"
        f"[PREVIOUS_INTENT]: {state.last_intent or ''}\n"
        f"[PREVIOUS_ANSWER_TYPE]: {state.last_answer_type or ''}\n"
        f"[INSTRUCTION]: ช่วยตอบในเชิงการนำไปใช้จริง วิธีเริ่มต้น ขั้นตอน หรือแนวทางปฏิบัติที่นำไปใช้กับงานได้"
    )

def build_summary_message(user_message: str, state) -> str:
    return (
        f"{user_message}\n\n"
        f"[ANSWER_MODE]: summary\n"
        f"[PREVIOUS_ANSWER]: {state.last_answer or ''}\n"
        f"[PREVIOUS_INTENT]: {state.last_intent or ''}\n"
        f"[PREVIOUS_ANSWER_TYPE]: {state.last_answer_type or ''}\n"
        f"[INSTRUCTION]: ช่วยสรุปให้กระชับ เข้าใจง่าย เน้นใจความสำคัญ ไม่ต้องยาว"
    )


async def process_chat_aicustom(req, state, conn):
    if state is None:
        state = ChatState_aicustom()

    user_message = (req.user_message or "").strip()

    state.web_no = int(req.web_no) if req.web_no not in [None, ""] else None
    state.member_no = int(req.member_no) if req.member_no not in [None, ""] else None

    if req.course_use:
        state.course_use = [int(x) for x in req.course_use if str(x).strip()]

    course_use = state.course_use or []

    if not course_use:
        reply = "ขออภัยครับ ยังไม่พบรายการหลักสูตรที่อนุญาตให้ใช้งาน"
        state.last_user_message = user_message
        state.last_answer = reply
        state.last_intent = "unknown"
        state.last_answer_type = "no_course"

        return type("Obj", (), {
            "reply": reply,
            "status": "no_course",
            "reason": "empty_course_use",
            "state": state,
            "source": "ai_custom_no_course",
            "active_video": None,
        })()

    course_data = get_course_data_by_nos(conn, course_use)
    course_context = build_course_name_context(course_data)

    state.last_user_message = user_message

    # ==================================================
    # 1) ถ้ามี topic ค้างอยู่แล้ว ใช้ topic เดิมก่อนเลย
    # ==================================================
    # current_topic = str(getattr(state, "topic", "") or "").strip()

    # if current_topic and current_topic != "unknown":

    #     # 🔥 เช็คก่อนว่าผู้ใช้เปลี่ยน topic ไหม
    #     intent_data = await detect_intent(user_message, course_context)
    #     new_topic = intent_data.get("topic", "unknown")

    #     # ถ้ามี topic ใหม่ และไม่เหมือนเดิม -> เปลี่ยน topic
    #     if new_topic and new_topic != "unknown" and new_topic != current_topic:
    #         state.topic = new_topic
    #         current_topic = new_topic

    #     script = find_script_by_topic(course_data, current_topic)

    #     if script:
    #         reply = await reply_with_topic(user_message, current_topic, script)
    #     else:
    #         reply = await reply_learning(user_message, course_context)

    #     state.intent = "learning"
    #     state.mode = "learning"
    #     state.last_answer = reply

    #     return type("Obj", (), {
    #         "reply": reply,
    #         "status": "learning",
    #         "reason": "use_existing_topic",
    #         "state": state,
    #         "source": "ai_custom_topic_continue",
    #     })()

    # ==================================================
    # 2) ถ้าอยู่ใน learning อยู่แล้ว แต่ยังไม่มี topic
    #    ก็ไม่ต้อง detect ใหม่
    # ==================================================
    # if getattr(state, "intent", "") == "learning" or getattr(state, "mode", "") == "learning":
    #     intent_data = await detect_intent(user_message, course_context)
    #     topic = intent_data.get("topic", "unknown")

    #     state.intent = "learning"
    #     state.topic = topic

    #     if topic and topic != "unknown":
    #         script = find_script_by_topic(course_data, topic)

    #         if script:
    #             reply = await reply_with_topic(user_message, topic, script)
    #         else:
    #             reply = await reply_learning(user_message, course_context)
    #     else:
    #         reply = await reply_learning(user_message, course_context)

    #     state.mode = "learning"
    #     state.last_answer = reply

    #     return type("Obj", (), {
    #         "reply": reply,
    #         "status": "learning",
    #         "reason": "continue_learning_mode_detect_topic",
    #         "state": state,
    #         "source": "ai_custom_learning_continue",
    #     })()

    # ==================================================
    # 3) ค่อย detect intent เฉพาะตอนยังไม่มี context เดิม
    # ==================================================
    intent_data = await detect_intent(user_message, course_context)
    intent = intent_data["intent"]
    detected_topic = intent_data.get("topic", "unknown")

    reply = f"intent: {intent} | topic: {detected_topic}"
    state.intent = intent

    if detected_topic and detected_topic != "unknown":
        state.topic = detected_topic

    topic = state.topic or "unknown"
    active_video = None

    if intent == "greeting":
        reply = await reply_greeting(user_message, course_context)
        state.mode = "idle"
        state.last_answer = reply
        state.last_intent = "greeting"
        state.last_answer_type = "greeting_given"

        return type("Obj", (), {
            "reply": reply,
            "status": "greeting",
            "reason": "intent_greeting",
            "state": state,
            "source": "ai_custom_greeting",
            "active_video": None,
        })()

    elif intent == "general":
        reply = await reply_general(user_message, course_context)
        state.mode = "idle"
        state.last_answer = reply
        state.last_intent = "general"
        state.last_answer_type = "general_replied"

        return type("Obj", (), {
            "reply": reply,
            "status": "general",
            "reason": "intent_general",
            "state": state,
            "source": "ai_custom_general",
            "active_video": None,
        })()
    
    elif intent == "out_of_scope":
        reply = await reply_out_of_scope(user_message, course_context)
        state.mode = "idle"
        state.last_answer = reply
        state.last_intent = "out_of_scope"
        state.last_answer_type = "out_of_scope_replied"

        return type("Obj", (), {
            "reply": reply,
            "status": "general",
            "reason": "intent_general",
            "state": state,
            "source": "ai_custom_general",
            "active_video": None,
        })()
    
    elif intent == "ask_recommend_course":
        reply = await reply_ask_recommend_course(user_message, course_context)
        state.mode = "recommend"
        state.last_answer = reply
        state.last_intent = "ask_recommend_course"
        state.last_answer_type = "recommendation_given"
        state.active_course_no = None

        return type("Obj", (), {
            "reply": reply,
            "status": "general",
            "reason": "intent_general",
            "state": state,
            "source": "ai_custom_general",
            "active_video": None,
        })()
    
    elif intent == "ask_concept":

        active_video = None
        resolved_topic = topic if topic != "unknown" else state.topic

        if resolved_topic and resolved_topic != "unknown":

            course_match = None

            if state.active_course_no:
                course_match = find_course_by_no(course_data, state.active_course_no)

            if not course_match:
                course_match = find_course_by_topic(course_data, resolved_topic)

            print("DEBUG topic =", topic)
            print("DEBUG state.topic =", state.topic)
            print("DEBUG resolved_topic =", resolved_topic)
            print("DEBUG course_match =", course_match)
            print("DEBUG course_match type =", type(course_match))

            if course_match:
                state.active_course_no = course_match.get("course_no")

                script = str(course_match.get("script") or "").strip()
                videos = course_match.get("videos") or []

                print("DEBUG script =", script)
                print("DEBUG script type =", type(script))
                print("DEBUG videos =", videos)

                if script:
                    reply = await reply_ask_concept_with_topic(user_message, resolved_topic, script)
                    state.mode = "learning"
                    state.topic = resolved_topic
                    state.active_course_no = course_match.get("course_no")
                    state.last_intent = "ask_concept"
                    state.last_answer_type = "concept_explained"

                    active_video = build_video_payload(videos[0]) if videos else None

                else:
                    reply = await reply_ask_concept_no_topic(user_message, course_context)
                    state.mode = "discover"
                    state.active_course_no = None
                    state.last_intent = "ask_concept"
                    state.last_answer_type = "concept_not_found"

            else:
                reply = await reply_ask_concept_no_topic(user_message, course_context)
                state.mode = "discover"
                state.active_course_no = None
                state.last_intent = "ask_concept"
                state.last_answer_type = "concept_not_found"

        else:
            reply = await reply_ask_concept_no_topic(user_message, course_context)
            state.mode = "discover"
            state.active_course_no = None
            state.last_intent = "ask_concept"
            state.last_answer_type = "concept_not_found"

        state.last_answer = reply

    elif intent == "ask_followup":

        followup_type = detect_followup_type(user_message)

        if not state.topic or state.topic == "unknown" or not state.last_answer:
            reply = "ต้องการให้ขยายหรือสรุปเรื่องอะไรครับ 😊"
            state.mode = "discover"
            state.active_course_no = None
            state.last_intent = "ask_followup"
            state.last_answer_type = "followup_needs_context"
            state.last_answer = reply

            return type("Obj", (), {
                "reply": reply,
                "status": "discover",
                "reason": "followup_needs_context",
                "state": state,
                "source": "ai_custom_followup",
                "active_video": None,
            })()

        course_match = None

        if state.active_course_no:
            course_match = find_course_by_no(course_data, state.active_course_no)

        if not course_match and state.topic and state.topic != "unknown":
            course_match = find_course_by_topic(course_data, state.topic)

        print("DEBUG followup topic =", state.topic)
        print("DEBUG followup active_course_no =", state.active_course_no)
        print("DEBUG followup course_match =", course_match)
        print("DEBUG followup_type =", followup_type)

        if course_match:
            script = str(course_match.get("script") or "").strip()
            videos = course_match.get("videos") or []

            if script:
                followup_message = (
                    f"{user_message}\n\n"
                    f"[FOLLOWUP_TYPE]: {followup_type}\n"
                    f"[PREVIOUS_ANSWER]: {state.last_answer or ''}\n"
                    f"[PREVIOUS_INTENT]: {state.last_intent or ''}\n"
                    f"[PREVIOUS_ANSWER_TYPE]: {state.last_answer_type or ''}"
                )

                reply = await reply_ask_concept_with_topic(
                    followup_message,
                    state.topic,
                    script
                )

                state.mode = "learning"
                state.active_course_no = course_match.get("course_no")
                state.last_intent = "ask_followup"
                state.last_answer_type = map_followup_answer_type(followup_type)
                state.last_answer = reply

                if len(videos) > 1:
                    active_video = build_video_payload(random.choice(videos[1:]))
                elif len(videos) == 1:
                    active_video = build_video_payload(videos[0])
                else:
                    active_video = None

            else:
                reply = "ผมหาหัวข้อที่ต่อเนื่องได้แล้ว แต่ยังไม่พบรายละเอียดเพียงพอสำหรับขยายคำตอบครับ"
                state.mode = "discover"
                state.active_course_no = None
                state.last_intent = "ask_followup"
                state.last_answer_type = "followup_no_script"
                state.last_answer = reply

        else:
            reply = "ผมยังจับหัวข้อเดิมได้ไม่ชัด ช่วยพิมพ์ชื่อเรื่องที่ต้องการให้ขยายอีกนิดได้ไหมครับ 😊"
            state.mode = "discover"
            state.active_course_no = None
            state.last_intent = "ask_followup"
            state.last_answer_type = "followup_no_course"
            state.last_answer = reply

    elif intent == "ask_application":

        course_match = None

        if state.active_course_no:
            course_match = find_course_by_no(course_data, state.active_course_no)

        if not course_match and topic and topic != "unknown":
            course_match = find_course_by_topic(course_data, topic)

        if not course_match and state.topic and state.topic != "unknown":
            course_match = find_course_by_topic(course_data, state.topic)

        print("DEBUG ask_application topic =", topic)
        print("DEBUG ask_application state.topic =", state.topic)
        print("DEBUG ask_application active_course_no =", state.active_course_no)
        print("DEBUG ask_application course_match =", course_match)

        if course_match:
            script = str(course_match.get("script") or "").strip()
            videos = course_match.get("videos") or []

            if script:
                resolved_topic = topic if topic and topic != "unknown" else state.topic
                application_message = build_application_message(user_message, state)

                reply = await reply_ask_concept_with_topic(
                    application_message,
                    resolved_topic,
                    script
                )

                state.mode = "learning"
                state.topic = resolved_topic
                state.active_course_no = course_match.get("course_no")
                state.last_intent = "ask_application"
                state.last_answer_type = "application_given"
                state.last_answer = reply

                if len(videos) > 1:
                    active_video = build_video_payload(random.choice(videos[1:]))
                elif len(videos) == 1:
                    active_video = build_video_payload(videos[0])
                else:
                    active_video = None

            else:
                reply = "ผมหัวข้อที่เกี่ยวข้องได้แล้ว แต่ยังไม่พบรายละเอียดเพียงพอสำหรับอธิบายการนำไปใช้ครับ"
                state.mode = "discover"
                state.active_course_no = None
                state.last_intent = "ask_application"
                state.last_answer_type = "application_not_found"
                state.last_answer = reply

        else:
            reply = "ต้องการให้นำเรื่องอะไรไปใช้ครับ 😊"
            state.mode = "discover"
            state.active_course_no = None
            state.last_intent = "ask_application"
            state.last_answer_type = "application_needs_topic"
            state.last_answer = reply

        return type("Obj", (), {
            "reply": reply,
            "status": "learning" if state.mode == "learning" else "discover",
            "reason": "intent_ask_application",
            "state": state,
            "source": "ai_custom_application",
            "active_video": active_video,
        })()
    
    elif intent == "ask_summary":

        course_match = None

        if state.active_course_no:
            course_match = find_course_by_no(course_data, state.active_course_no)

        if not course_match and topic and topic != "unknown":
            course_match = find_course_by_topic(course_data, topic)

        if not course_match and state.topic and state.topic != "unknown":
            course_match = find_course_by_topic(course_data, state.topic)

        print("DEBUG ask_summary topic =", topic)
        print("DEBUG ask_summary state.topic =", state.topic)
        print("DEBUG ask_summary active_course_no =", state.active_course_no)
        print("DEBUG ask_summary course_match =", course_match)

        if course_match:
            script = str(course_match.get("script") or "").strip()
            videos = course_match.get("videos") or []

            if script:
                resolved_topic = topic if topic and topic != "unknown" else state.topic
                summary_message = build_summary_message(user_message, state)

                reply = await reply_ask_concept_with_topic(
                    summary_message,
                    resolved_topic,
                    script
                )

                state.mode = "learning"
                state.topic = resolved_topic
                state.active_course_no = course_match.get("course_no")
                state.last_intent = "ask_summary"
                state.last_answer_type = "summary_given"
                state.last_answer = reply

                active_video = build_video_payload(videos[-1]) if videos else None

            else:
                reply = "ผมหาหัวข้อที่เกี่ยวข้องได้แล้ว แต่ยังไม่พบรายละเอียดเพียงพอสำหรับสรุปให้ครับ"
                state.mode = "discover"
                state.active_course_no = None
                state.last_intent = "ask_summary"
                state.last_answer_type = "summary_not_found"
                state.last_answer = reply

        else:
            reply = "ต้องการให้สรุปเรื่องอะไรครับ 😊"
            state.mode = "discover"
            state.active_course_no = None
            state.last_intent = "ask_summary"
            state.last_answer_type = "summary_needs_topic"
            state.last_answer = reply

        return type("Obj", (), {
            "reply": reply,
            "status": "learning" if state.mode == "learning" else "discover",
            "reason": "intent_ask_summary",
            "state": state,
            "source": "ai_custom_summary",
            "active_video": active_video,
        })()
    

    # ==================================================
    # 4) learning
    # ==================================================
    # if topic and topic != "unknown":
    #     script = find_script_by_topic(course_data, topic)

    #     if script:
    #         reply = await reply_with_topic(user_message, topic, script)
    #     else:
    #         reply = await reply_learning(user_message, course_context)
    # else:
    #     reply = await reply_learning(user_message, course_context)

    # state.mode = "learning"
    # state.last_answer = reply

    return type("Obj", (), {
        "reply": reply,
        "status": "learning",
        "reason": "intent_learning",
        "state": state,
        "source": "ai_custom_learning",
        "active_video": active_video,
    })()

async def process_chat_aicustom_stream(req, state, conn):
    if state is None:
        state = ChatState_aicustom()

    user_message = (req.user_message or "").strip()

    state.web_no = int(req.web_no) if req.web_no not in [None, ""] else None
    state.member_no = int(req.member_no) if req.member_no not in [None, ""] else None

    if req.course_use:
        state.course_use = [int(x) for x in req.course_use if str(x).strip()]

    course_use = state.course_use or []

    if not course_use:
        reply = "ขออภัยครับ ยังไม่พบรายการหลักสูตรที่อนุญาตให้ใช้งาน"
        state.last_user_message = user_message
        state.last_answer = reply
        state.last_intent = "unknown"
        state.last_answer_type = "no_course"

        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "no_course",
            "reason": "empty_course_use",
            "state": state,
            "source": "ai_custom_no_course",
            "active_video": None,
        }
        return

    course_data = get_course_data_by_nos(conn, course_use)
    course_context = build_course_name_context(course_data)

    state.last_user_message = user_message

    intent_data = await detect_intent(user_message, course_context)
    intent = intent_data["intent"]
    detected_topic = intent_data.get("topic", "unknown")

    state.intent = intent

    if detected_topic and detected_topic != "unknown":
        state.topic = detected_topic

    topic = state.topic or "unknown"
    active_video = None

    if intent == "greeting":
        final_reply = ""
        async for item in reply_greeting_stream(user_message, course_context):
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}
            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        state.mode = "idle"
        state.last_answer = final_reply
        state.last_intent = "greeting"
        state.last_answer_type = "greeting_given"

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "greeting",
            "reason": "intent_greeting",
            "state": state,
            "source": "ai_custom_greeting",
            "active_video": None,
        }
        return

    elif intent == "general":
        final_reply = ""
        async for item in reply_general_stream(user_message, course_context):
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}
            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        state.mode = "idle"
        state.last_answer = final_reply
        state.last_intent = "general"
        state.last_answer_type = "general_replied"

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "general",
            "reason": "intent_general",
            "state": state,
            "source": "ai_custom_general",
            "active_video": None,
        }
        return

    elif intent == "out_of_scope":
        final_reply = ""
        async for item in reply_out_of_scope_stream(user_message, course_context):
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}
            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        state.mode = "idle"
        state.last_answer = final_reply
        state.last_intent = "out_of_scope"
        state.last_answer_type = "out_of_scope_replied"

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "general",
            "reason": "intent_general",
            "state": state,
            "source": "ai_custom_general",
            "active_video": None,
        }
        return

    elif intent == "ask_recommend_course":
        final_reply = ""
        async for item in reply_ask_recommend_course_stream(user_message, course_context):
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}
            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        state.mode = "recommend"
        state.last_answer = final_reply
        state.last_intent = "ask_recommend_course"
        state.last_answer_type = "recommendation_given"
        state.active_course_no = None

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "general",
            "reason": "intent_general",
            "state": state,
            "source": "ai_custom_general",
            "active_video": None,
        }
        return

    elif intent == "ask_concept":
        active_video = None
        resolved_topic = topic if topic != "unknown" else state.topic

        if resolved_topic and resolved_topic != "unknown":
            course_match = None

            if state.active_course_no:
                course_match = find_course_by_no(course_data, state.active_course_no)

            if not course_match:
                course_match = find_course_by_topic(course_data, resolved_topic)

            print("DEBUG topic =", topic)
            print("DEBUG state.topic =", state.topic)
            print("DEBUG resolved_topic =", resolved_topic)
            print("DEBUG course_match =", course_match)
            print("DEBUG course_match type =", type(course_match))

            if course_match:
                state.active_course_no = course_match.get("course_no")

                script = str(course_match.get("script") or "").strip()
                videos = course_match.get("videos") or []

                print("DEBUG script =", script)
                print("DEBUG script type =", type(script))
                print("DEBUG videos =", videos)

                if script:
                    final_reply = ""
                    async for item in reply_ask_concept_with_topic_stream(user_message, resolved_topic, script):
                        if item["type"] == "chunk":
                            text = item.get("text", "")
                            final_reply += text
                            yield {"type": "chunk", "text": text}
                        elif item["type"] == "done":
                            final_reply = item.get("content", final_reply)

                    state.mode = "learning"
                    state.topic = resolved_topic
                    state.active_course_no = course_match.get("course_no")
                    state.last_intent = "ask_concept"
                    state.last_answer_type = "concept_explained"
                    state.last_answer = final_reply

                    active_video = build_video_payload(videos[0]) if videos else None

                    yield {
                        "type": "done",
                        "reply": final_reply,
                        "status": "learning",
                        "reason": "intent_learning",
                        "state": state,
                        "source": "ai_custom_learning",
                        "active_video": active_video,
                    }
                    return

        final_reply = ""
        async for item in reply_ask_concept_no_topic_stream(user_message, course_context):
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}
            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        state.mode = "discover"
        state.active_course_no = None
        state.last_intent = "ask_concept"
        state.last_answer_type = "concept_not_found"
        state.last_answer = final_reply

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "discover",
            "reason": "intent_ask_concept",
            "state": state,
            "source": "ai_custom_concept",
            "active_video": None,
        }
        return

    elif intent == "ask_followup":
        followup_type = detect_followup_type(user_message)

        if not state.topic or state.topic == "unknown" or not state.last_answer:
            reply = "ต้องการให้ขยายหรือสรุปเรื่องอะไรครับ 😊"
            state.mode = "discover"
            state.active_course_no = None
            state.last_intent = "ask_followup"
            state.last_answer_type = "followup_needs_context"
            state.last_answer = reply

            yield {"type": "chunk", "text": reply}
            yield {
                "type": "done",
                "reply": reply,
                "status": "discover",
                "reason": "followup_needs_context",
                "state": state,
                "source": "ai_custom_followup",
                "active_video": None,
            }
            return

        course_match = None

        if state.active_course_no:
            course_match = find_course_by_no(course_data, state.active_course_no)

        if not course_match and state.topic and state.topic != "unknown":
            course_match = find_course_by_topic(course_data, state.topic)

        print("DEBUG followup topic =", state.topic)
        print("DEBUG followup active_course_no =", state.active_course_no)
        print("DEBUG followup course_match =", course_match)
        print("DEBUG followup_type =", followup_type)

        if course_match:
            script = str(course_match.get("script") or "").strip()
            videos = course_match.get("videos") or []

            if script:
                followup_message = (
                    f"{user_message}\n\n"
                    f"[FOLLOWUP_TYPE]: {followup_type}\n"
                    f"[PREVIOUS_ANSWER]: {state.last_answer or ''}\n"
                    f"[PREVIOUS_INTENT]: {state.last_intent or ''}\n"
                    f"[PREVIOUS_ANSWER_TYPE]: {state.last_answer_type or ''}"
                )

                final_reply = ""
                async for item in reply_ask_concept_with_topic_stream(
                    followup_message,
                    state.topic,
                    script
                ):
                    if item["type"] == "chunk":
                        text = item.get("text", "")
                        final_reply += text
                        yield {"type": "chunk", "text": text}
                    elif item["type"] == "done":
                        final_reply = item.get("content", final_reply)

                state.mode = "learning"
                state.active_course_no = course_match.get("course_no")
                state.last_intent = "ask_followup"
                state.last_answer_type = map_followup_answer_type(followup_type)
                state.last_answer = final_reply

                if len(videos) > 1:
                    active_video = build_video_payload(random.choice(videos[1:]))
                elif len(videos) == 1:
                    active_video = build_video_payload(videos[0])
                else:
                    active_video = None

                yield {
                    "type": "done",
                    "reply": final_reply,
                    "status": "learning",
                    "reason": "intent_ask_followup",
                    "state": state,
                    "source": "ai_custom_followup",
                    "active_video": active_video,
                }
                return

            reply = "ผมหาหัวข้อที่ต่อเนื่องได้แล้ว แต่ยังไม่พบรายละเอียดเพียงพอสำหรับขยายคำตอบครับ"
            state.mode = "discover"
            state.active_course_no = None
            state.last_intent = "ask_followup"
            state.last_answer_type = "followup_no_script"
            state.last_answer = reply

            yield {"type": "chunk", "text": reply}
            yield {
                "type": "done",
                "reply": reply,
                "status": "discover",
                "reason": "followup_no_script",
                "state": state,
                "source": "ai_custom_followup",
                "active_video": None,
            }
            return

        reply = "ผมยังจับหัวข้อเดิมได้ไม่ชัด ช่วยพิมพ์ชื่อเรื่องที่ต้องการให้ขยายอีกนิดได้ไหมครับ 😊"
        state.mode = "discover"
        state.active_course_no = None
        state.last_intent = "ask_followup"
        state.last_answer_type = "followup_no_course"
        state.last_answer = reply

        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "discover",
            "reason": "followup_no_course",
            "state": state,
            "source": "ai_custom_followup",
            "active_video": None,
        }
        return

    elif intent == "ask_application":
        course_match = None

        if state.active_course_no:
            course_match = find_course_by_no(course_data, state.active_course_no)

        if not course_match and topic and topic != "unknown":
            course_match = find_course_by_topic(course_data, topic)

        if not course_match and state.topic and state.topic != "unknown":
            course_match = find_course_by_topic(course_data, state.topic)

        print("DEBUG ask_application topic =", topic)
        print("DEBUG ask_application state.topic =", state.topic)
        print("DEBUG ask_application active_course_no =", state.active_course_no)
        print("DEBUG ask_application course_match =", course_match)

        if course_match:
            script = str(course_match.get("script") or "").strip()
            videos = course_match.get("videos") or []

            if script:
                resolved_topic = topic if topic and topic != "unknown" else state.topic
                application_message = build_application_message(user_message, state)

                final_reply = ""
                async for item in reply_ask_concept_with_topic_stream(
                    application_message,
                    resolved_topic,
                    script
                ):
                    if item["type"] == "chunk":
                        text = item.get("text", "")
                        final_reply += text
                        yield {"type": "chunk", "text": text}
                    elif item["type"] == "done":
                        final_reply = item.get("content", final_reply)

                state.mode = "learning"
                state.topic = resolved_topic
                state.active_course_no = course_match.get("course_no")
                state.last_intent = "ask_application"
                state.last_answer_type = "application_given"
                state.last_answer = final_reply

                if len(videos) > 1:
                    active_video = build_video_payload(random.choice(videos[1:]))
                elif len(videos) == 1:
                    active_video = build_video_payload(videos[0])
                else:
                    active_video = None

                yield {
                    "type": "done",
                    "reply": final_reply,
                    "status": "learning",
                    "reason": "intent_ask_application",
                    "state": state,
                    "source": "ai_custom_application",
                    "active_video": active_video,
                }
                return

            reply = "ผมหัวข้อที่เกี่ยวข้องได้แล้ว แต่ยังไม่พบรายละเอียดเพียงพอสำหรับอธิบายการนำไปใช้ครับ"
            state.mode = "discover"
            state.active_course_no = None
            state.last_intent = "ask_application"
            state.last_answer_type = "application_not_found"
            state.last_answer = reply

            yield {"type": "chunk", "text": reply}
            yield {
                "type": "done",
                "reply": reply,
                "status": "discover",
                "reason": "application_not_found",
                "state": state,
                "source": "ai_custom_application",
                "active_video": None,
            }
            return

        reply = "ต้องการให้นำเรื่องอะไรไปใช้ครับ 😊"
        state.mode = "discover"
        state.active_course_no = None
        state.last_intent = "ask_application"
        state.last_answer_type = "application_needs_topic"
        state.last_answer = reply

        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "discover",
            "reason": "application_needs_topic",
            "state": state,
            "source": "ai_custom_application",
            "active_video": None,
        }
        return

    elif intent == "ask_summary":
        course_match = None

        if state.active_course_no:
            course_match = find_course_by_no(course_data, state.active_course_no)

        if not course_match and topic and topic != "unknown":
            course_match = find_course_by_topic(course_data, topic)

        if not course_match and state.topic and state.topic != "unknown":
            course_match = find_course_by_topic(course_data, state.topic)

        print("DEBUG ask_summary topic =", topic)
        print("DEBUG ask_summary state.topic =", state.topic)
        print("DEBUG ask_summary active_course_no =", state.active_course_no)
        print("DEBUG ask_summary course_match =", course_match)

        if course_match:
            script = str(course_match.get("script") or "").strip()
            videos = course_match.get("videos") or []

            if script:
                resolved_topic = topic if topic and topic != "unknown" else state.topic
                summary_message = build_summary_message(user_message, state)

                final_reply = ""
                async for item in reply_ask_concept_with_topic_stream(
                    summary_message,
                    resolved_topic,
                    script
                ):
                    if item["type"] == "chunk":
                        text = item.get("text", "")
                        final_reply += text
                        yield {"type": "chunk", "text": text}
                    elif item["type"] == "done":
                        final_reply = item.get("content", final_reply)

                state.mode = "learning"
                state.topic = resolved_topic
                state.active_course_no = course_match.get("course_no")
                state.last_intent = "ask_summary"
                state.last_answer_type = "summary_given"
                state.last_answer = final_reply

                active_video = build_video_payload(videos[-1]) if videos else None

                yield {
                    "type": "done",
                    "reply": final_reply,
                    "status": "learning",
                    "reason": "intent_ask_summary",
                    "state": state,
                    "source": "ai_custom_summary",
                    "active_video": active_video,
                }
                return

            reply = "ผมหาหัวข้อที่เกี่ยวข้องได้แล้ว แต่ยังไม่พบรายละเอียดเพียงพอสำหรับสรุปให้ครับ"
            state.mode = "discover"
            state.active_course_no = None
            state.last_intent = "ask_summary"
            state.last_answer_type = "summary_not_found"
            state.last_answer = reply

            yield {"type": "chunk", "text": reply}
            yield {
                "type": "done",
                "reply": reply,
                "status": "discover",
                "reason": "summary_not_found",
                "state": state,
                "source": "ai_custom_summary",
                "active_video": None,
            }
            return

        reply = "ต้องการให้สรุปเรื่องอะไรครับ 😊"
        state.mode = "discover"
        state.active_course_no = None
        state.last_intent = "ask_summary"
        state.last_answer_type = "summary_needs_topic"
        state.last_answer = reply

        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "discover",
            "reason": "summary_needs_topic",
            "state": state,
            "source": "ai_custom_summary",
            "active_video": None,
        }
        return

    final_reply = ""
    async for item in reply_learning_stream(user_message, course_context):
        if item["type"] == "chunk":
            text = item.get("text", "")
            final_reply += text
            yield {"type": "chunk", "text": text}
        elif item["type"] == "done":
            final_reply = item.get("content", final_reply)

    state.mode = "learning"
    state.last_answer = final_reply
    state.last_intent = intent
    state.last_answer_type = "learning_replied"

    yield {
        "type": "done",
        "reply": final_reply,
        "status": "learning",
        "reason": "intent_learning",
        "state": state,
        "source": "ai_custom_learning",
        "active_video": active_video,
    }