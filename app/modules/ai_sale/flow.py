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

    skip_extract = False

    if state.mode == "post_recommend":
        post_intent = await detect_post_recommend_intent(
            user_message=user_message,
            requirements=state.requirements or {},
            conversation_history=state.conversation_history
        )

        print("POST RECOMMEND INTENT =", post_intent, flush=True)
        print("STATE MODE =", state.mode, flush=True)

        intent = post_intent.get("intent", "unclear")
        state.last_step = intent

        if intent == "new_requirement":
            state.requirements = {}
            state.missing_requirements = []
            state.requirement_ready = False
            state.search_query = None
            state.recommended_courses = []
            state.recommended_course_cta = []
            state.mode = "discovery"

            state.conversation_history = [{
                "role": "user",
                "content": user_message
            }]

        elif intent == "ask_more_courses":
            skip_extract = True
            state.mode = "discovery"
            state.last_step = "ask_more_courses"

        elif intent == "refine_requirement":
            state.mode = "discovery"

        else:
            state.mode = "discovery"

    if not skip_extract:
        old_requirements = dict(state.requirements or {})

        old_req = " ".join(
            str(v).strip()
            for k, v in sorted(old_requirements.items())
            if k != "matched_course" and v
        ).strip()

        new_requirements = await extract_requirements(
            user_message=user_message,
            current_requirements=state.requirements or {},
            conversation_history=state.conversation_history
        )

        new_req = " ".join(
            str(v).strip()
            for k, v in sorted((new_requirements or {}).items())
            if k != "matched_course" and v
        ).strip()

        state.requirements = new_requirements

        new_topic = new_requirements.get("topic")
        old_topic = old_requirements.get("topic")

        print(
            f"REQ CHECK | old_req={old_req} | new_req={new_req}",
            flush=True
        )

        topic_check_courses = None
        course_id = None

        missing = calc_missing_requirements(state.requirements)
        state.missing_requirements = missing
        state.requirement_ready = len(missing) == 0

        if not new_req:
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

            state.mode = "post_recommend"

        if new_req and new_req != old_req:
            topic_check_courses, course_id = await check_topic_exists_in_qdrant(
                new_req,
                limit=1
            )

        elif new_req and old_req and new_req == old_req:
            topic_check_courses = state.requirements.get("matched_course")
            course_id = getattr(state, "matched_course_id", None)

        if new_req and new_topic and not topic_check_courses and not old_topic:
            reply = ""

            async for item in build_irrelevant_topic_reply(
                user_message=user_message,
                old_topic=old_topic,
                conversation_history=state.conversation_history
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

        if topic_check_courses:
            state.requirements = new_requirements
            state.requirements["topic"] = new_topic
            state.requirements["matched_course"] = topic_check_courses

            missing = calc_missing_requirements(state.requirements)
            state.missing_requirements = missing
            state.requirement_ready = len(missing) == 0

            reply = ""

            print("CALL build_next_question_topic", flush=True)

            async for item in build_next_question_topic(
                requirements=state.requirements,
                missing=missing,
                conversation_history=state.conversation_history,
                matched_course=state.requirements.get("matched_course")
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
            state.matched_course = state.requirements["matched_course"]
            state.matched_course_id = course_id
            state.last_step = "ask_requirement_topic"

            # ✅ เก็บ course ที่ match แล้ว กันซ้ำตอน user ขอ course เพิ่ม
            old_courses = state.recommended_courses or []

            exists_ids = set()
            for course in old_courses:
                payload = get_course_payload(course)

                if not payload:
                    continue

                cid = (
                    payload.get("course_no")
                    or payload.get("OCourse_no")
                    or payload.get("id")
                )

                if cid:
                    exists_ids.add(str(cid))

            # ✅ normalize payload ให้เป็น dict เสมอ
            if isinstance(topic_check_courses, dict):
                matched_payload = topic_check_courses.get("payload") or topic_check_courses
            else:
                matched_payload = {}

            matched_course_id = None

            if isinstance(matched_payload, dict):
                matched_course_id = (
                    matched_payload.get("course_no")
                    or matched_payload.get("OCourse_no")
                    or matched_payload.get("id")
                    or course_id
                )
            else:
                matched_course_id = course_id

            if matched_payload is None:
                matched_payload = topic_check_courses

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
                state.mode = "post_recommend"
                status = "ready_to_recommend"
                reason = "requirement_complete"
            else:
                state.mode = "discovery"
                status = "collecting_requirement"
                reason = "matched_topic_need_more_info"

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
                "source": "ai_sale_discovery",
            }
            return

        if old_topic:
            new_requirements["topic"] = old_topic
            state.requirements = new_requirements
            print(f"KEEP OLD TOPIC: {old_topic}", flush=True)

    else:
        # ✅ กรณีลูกค้าขอ course เพิ่ม / course ใกล้เคียงของเดิม
        search_query = await build_search_query(state.requirements)
        state.search_query = search_query

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

        # ✅ อัปเดต matched_course เป็น course ใหม่
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

                if state.requirements is None:
                    state.requirements = {}

                state.requirements["matched_course"] = new_course_name

        # ✅ เก็บ courses แบบไม่ซ้ำ
        old_courses = state.recommended_courses or []
        merged_courses = []
        seen_course_ids = set()

        for course in old_courses + courses:
            course_no = get_course_id(course)

            if course_no:
                key = str(course_no)
                if key in seen_course_ids:
                    continue
                seen_course_ids.add(key)

            merged_courses.append(course)

        state.recommended_courses = merged_courses

        course_cta = []

        for c in courses:
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

        reply = ""

        async for item in build_more_courses_reply(
            requirements=state.requirements,
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

        state.mode = "post_recommend"
        state.last_answer = reply
        state.last_step = "ask_more_courses"

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
            "source": "ai_sale_qdrant",
        }
        return
    

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