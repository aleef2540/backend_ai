from app.modules.ai_sale.schema import AISaleState
from app.modules.ai_sale.service import (
    extract_requirements,
    calc_missing_requirements,
    build_next_question,
    build_search_query,
    build_recommendation_reply,
    detect_post_recommend_intent,
    build_more_courses_reply,
    build_irrelevant_topic_reply,
    build_next_question_topic
)
from app.modules.ai_sale.qdrant_service import search_courses_from_qdrant, check_topic_exists_in_qdrant
import json

async def process_ai_sale_stream(req, state):
    if state is None:
        state = AISaleState()

    user_message = (req.user_message or "").strip()
    from_web = (req.from_web or "").strip()
    state.last_user_message = user_message
    state.from_web = from_web

    state.conversation_history.append({
        "role": "user",
        "content": user_message
    })

    if len(state.conversation_history) > 10:
        state.conversation_history = state.conversation_history[-10:]

    # ✅ ใส่ตรงนี้
    skip_extract = False
    if state.mode == "post_recommend":
        post_intent = await detect_post_recommend_intent(
            user_message=user_message,
            requirements=state.requirements or {},
            conversation_history=state.conversation_history
        )

        print("POST RECOMMEND INTENT =", post_intent, flush=True)
        print("STATE MODE =", state.mode, flush=True)

        state.last_step = post_intent.get("intent", "unclear")

        if post_intent["intent"] == "new_requirement":
            state.requirements = {}
            state.missing_requirements = []
            state.requirement_ready = False
            state.search_query = None
            state.recommended_courses = []
            state.recommended_course_cta = []
            state.mode = "discovery"

            # ✅ สำคัญมาก: ล้าง history เก่า เหลือเฉพาะข้อความล่าสุด
            state.conversation_history = [{
                "role": "user",
                "content": user_message
            }]

        elif post_intent["intent"] == "ask_more_courses":
            skip_extract = True
            state.mode = "discovery"
            state.last_step = "ask_more_courses"

        elif post_intent["intent"] == "refine_requirement":
            state.mode = "discovery"

        else:
            state.mode = "discovery"

    # 👇 แล้วค่อย extract
    if not skip_extract:

        old_requirements = dict(state.requirements or {})
        old_topic = old_requirements.get("topic")

        new_requirements = await extract_requirements(
            user_message=user_message,
            current_requirements=state.requirements or {},
            conversation_history=state.conversation_history
        )
        state.requirements = new_requirements
        req = json.dumps(new_requirements or {}, ensure_ascii=False)

        req = " ".join(
            str(v)
            for k, v in (new_requirements or {}).items()
            if k != "matched_course"
        ).strip()
        
        new_topic = new_requirements.get("topic")
        new_topic_norm = (new_topic or "").strip()
        old_topic_norm = (old_topic or "").strip()

        print(f"TOPIC CHECK | new topic={new_topic_norm} | old topic={old_topic_norm} | req={req}", flush=True)

        # if new_topic and new_topic_norm != old_topic_norm:
        if True:
            
            #ยังไม่มี req ให้่นำไปค้นหาใน rag
            if not req:
                    
                    missing = calc_missing_requirements(state.requirements)
                    state.missing_requirements = missing
                    state.requirement_ready = len(missing) == 0

                    if missing:
                        reply = ""
                        print("CALL build_next_question", flush=True)
                        async for item in build_next_question(
                            state.requirements,
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

                        state.mode = "discovery"
                        state.last_answer = reply
                        state.last_step = "ask_requirement"

                        state.conversation_history.append({
                            "role": "assistant",
                            "content": reply
                        })

                        if len(state.conversation_history) > 10:
                            state.conversation_history = state.conversation_history[-10:]

                        yield {
                            "type": "done",
                            "reply": reply,
                            "status": "collecting_requirement",
                            "reason": "missing_requirement",
                            "state": state,
                            "source": "ai_sale_discovery",
                        }
                        return
            # มี req แล้วเอาเอาไปค้นหา        
            else:
                 # 1. เรียกใช้งานฟังก์ชัน (จะได้ค่าเป็น "ชื่อหลักสูตร" หรือ "")
                topic_check_courses, course_id = await check_topic_exists_in_qdrant(
                    req,
                    limit=1
                )

            # มี req แต่หา course ไม่เจอในระบบ ใช้ topic เก่า
            if new_topic and not topic_check_courses and not old_topic:
                reply = ""

                async for item in build_irrelevant_topic_reply(
                    user_message=user_message,
                    old_topic=old_topic,
                    conversation_history=state.conversation_history
                ):
                    if item.get("type") == "chunk":
                        text = item.get("text", "")
                        reply += text
                        yield {"type": "chunk", "text": text}

                    elif item.get("type") == "done":
                        reply = item.get("content") or reply

                state.last_answer = reply
                state.last_step = "irrelevant_topic"

                state.conversation_history.append({
                    "role": "assistant",
                    "content": reply
                })
                state.mode = "post_recommend"
                yield {
                    "type": "done",
                    "reply": reply,
                    "status": "irrelevant",
                    "reason": "topic_not_found",
                    "state": state,
                }
                return

            # 3. กรณีที่เจอหลักสูตรที่เกี่ยวข้อง (Score ผ่านเกณฑ์)
            if topic_check_courses:
                state.requirements = new_requirements
                # ✅ เก็บ topic เดิมของ user
                state.requirements["topic"] = new_topic

                # ✅ แยกเก็บ course ที่ match
                state.requirements["matched_course"] = topic_check_courses

                reply = ""
                print("CALL build_next_question_topic", flush=True)
                async for item in build_next_question_topic(
                    requirements=state.requirements,
                    missing=calc_missing_requirements(state.requirements),
                    conversation_history=state.conversation_history,
                    matched_course=state.requirements.get("matched_course")
                ):
                    if item.get("type") == "chunk":
                        text = item.get("text", "")
                        reply += text
                        yield {"type": "chunk", "text": text}
                    elif item.get("type") == "done":
                        reply = item.get("content") or reply

                state.last_answer = reply
                state.matched_course = state.requirements["matched_course"]
                state.matched_course_id = course_id
                state.last_step = "ask_requirement_topic"

                missing = calc_missing_requirements(state.requirements)
                state.missing_requirements = missing
                state.requirement_ready = len(missing) == 0
                if not missing:
                    state.mode = "post_recommend"
                else:
                    state.mode = "discovery"

                state.conversation_history.append({
                    "role": "assistant",
                    "content": reply
                })

                yield {
                    "type": "done",
                    "reply": reply,
                    "status": "collecting_requirement",
                    "reason": "matched_topic_need_more_info",
                    "state": state,
                    "source": "ai_sale_discovery",
                }
                return

            # 4. กรณีที่ไม่เจอหลักสูตรใหม่ แต่มีหัวข้อเก่า (Old Topic) ให้ประคองไว้
            else:
                if old_topic:
                    new_requirements["topic"] = old_topic

                state.requirements = new_requirements
                print(f"KEEP OLD TOPIC: {old_topic}", flush=True)

        else:
            state.requirements = new_requirements
    

    # missing = calc_missing_requirements(state.requirements)

    # state.missing_requirements = missing
    # state.requirement_ready = len(missing) == 0

    # if missing:
    #     reply = ""
    #     print("CALL build_next_question", flush=True)
    #     async for item in build_next_question(
    #         state.requirements,
    #         missing,
    #         state.conversation_history
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

    #     state.mode = "discovery"
    #     state.last_answer = reply
    #     state.last_step = "ask_requirement"

    #     state.conversation_history.append({
    #         "role": "assistant",
    #         "content": reply
    #     })

    #     if len(state.conversation_history) > 10:
    #         state.conversation_history = state.conversation_history[-10:]

    #     yield {
    #         "type": "done",
    #         "reply": reply,
    #         "status": "collecting_requirement",
    #         "reason": "missing_requirement",
    #         "state": state,
    #         "source": "ai_sale_discovery",
    #     }
    #     return

    # search_query = await build_search_query(state.requirements)
    # state.search_query = search_query

    # # สมมติว่า state.recommended_courses เก็บ course_no ที่แนะนำไปแล้ว
    # excluded_courses = [course['course_no'] for course in state.recommended_courses]

    # # ค้นหาหลักสูตรที่ไม่ซ้ำกับที่แนะนำไปแล้ว
    # courses = await search_courses_from_qdrant(search_query, limit=1, excluded_courses=excluded_courses)

    # # courses = await search_courses_from_qdrant(search_query, limit=3)
    # state.recommended_courses = courses

    # course_cta = []

    # for c in courses:
    #     payload = c.get("payload") or c

    #     course_no = (
    #         payload.get("course_no")
    #         or payload.get("OCourse_no")
    #         or payload.get("id")
    #     )

    #     course_name = (
    #         payload.get("course_name")
    #         or payload.get("vdo_name")
    #         or payload.get("Course_name")
    #         or payload.get("title")
    #     )

    #     if course_no and course_name:
    #         course_cta.append({
    #             "course_no": course_no,
    #             "course_name": course_name,
    #         })

    # reply = ""

    # # ✅ เลือก function ตาม intent
    # if state.last_step == "ask_more_courses":
    #     reply_builder = build_more_courses_reply
    # else:
    #     reply_builder = build_recommendation_reply

    # reply = ""

    # async for item in reply_builder(
    #     requirements=state.requirements,
    #     courses=courses
    # ):
        
    #     if item.get("type") == "chunk":
    #         text = item.get("text", "")
    #         reply += text
    #         yield {
    #             "type": "chunk",
    #             "text": text
    #         }

    #     elif item.get("type") == "done":
    #         reply = item.get("content") or reply

    # state.mode = "post_recommend"
    # state.last_answer = reply
    # state.last_step = "recommend_course"
    # state.recommended_course_cta = course_cta

    # state.conversation_history.append({
    #     "role": "assistant",
    #     "content": reply
    # })

    # if len(state.conversation_history) > 10:
    #     state.conversation_history = state.conversation_history[-10:]

    # yield {
    #     "type": "done",
    #     "reply": reply,
    #     "status": "recommended",
    #     "reason": "requirement_complete",
    #     "state": state,
    #     "source": "ai_sale_qdrant",
    # }