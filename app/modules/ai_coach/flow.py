from app.modules.ai_coach.schema import ChatRequest_aicoach, ChatResponse_aicoach, ChatState

from app.modules.ai_coach.service import (
   generate_opening_ai_coach_question,
   evaluate_user_answer,
   generate_retry_same_step_question,
   generate_probe_same_step_question,
   generate_next_step_question,
   generate_opening_ai_coach_question_stream,
   evaluate_answer,
   ask,
   ask_followup,
   ask_phase_transition,
   generate_tgrow_final_summary_stream,
   generate_next_step_question_stream,
   generate_retry_same_step_question_stream,
   generate_probe_same_step_question_stream,
   
   
)
from app.modules.ai_coach.constants import FIXED_QUESTIONS, TOPIC, PHASE1_RULES, PHASES


async def process_chat_aicoach(req: ChatRequest_aicoach, state: ChatState) -> ChatResponse_aicoach:
    user_message = req.user_message.strip()

    if not user_message:
        return ChatResponse_aicoach(
            reply="กรุณาพิมพ์สิ่งที่ต้องการพัฒนา / ปัญหาที่อยากแก้",
            state=state,
            source="empty_message",
        )

    if state.step == 0:
        # 1) ดึง fixed question
        fixed_q = state.fixed_question

        # 2) ให้ AI rewrite ให้เป็นโค้ช
        q1 = await generate_opening_ai_coach_question(
            fixed_question=fixed_q
        )

        # 3) update state
        new_state = ChatState(
            step=1,
            fixed_question=fixed_q,
            last_question=q1,
            answers_by_step=state.answers_by_step,
            history=state.history,
        )

        # เก็บ history: AI ถามเปิดข้อแรก
        new_state.history.append({
            "step": 1,
            "event": "ai_question",
            "fixed_question": fixed_q,
            "asked_question": q1,
        })

        reply = q1

    else:
        fixed_q = state.fixed_question
        current_step = state.step

        # ตั้งต้นไว้ก่อนกัน UnboundLocalError
        new_state = state

        # เก็บ history: user answer
        state.history.append({
            "step": current_step,
            "event": "user_answer",
            "fixed_question": fixed_q,
            "asked_question": state.last_question,
            "user_answer": user_message,
        })

        check = await evaluate_user_answer(
            question=fixed_q,
            user_answer=user_message,
        )

        status = check.get("status", "off_topic")
        reason = check.get("reason", "")
        confidence = check.get("confidence", 0.0)

        # เก็บ history: evaluation result
        state.history.append({
            "step": current_step,
            "event": "evaluation",
            "fixed_question": fixed_q,
            "asked_question": state.last_question,
            "user_answer": user_message,
            "status": status,
            "reason": reason,
            "confidence": confidence,
            "raw": check.get("raw", ""),
        })

        # สร้างที่เก็บของข้อปัจจุบันถ้ายังไม่มี
        if current_step not in state.answers_by_step:
            state.answers_by_step[current_step] = {
                "fixed_question": fixed_q,
                "asked_question": state.last_question,
                "user_answer": "",
                "status": "",
                "reason": "",
                "confidence": 0.0,
                "is_completed": False,
            }

        # update ข้อมูลของข้อปัจจุบัน
        state.answers_by_step[current_step]["fixed_question"] = fixed_q
        state.answers_by_step[current_step]["asked_question"] = state.last_question
        state.answers_by_step[current_step]["user_answer"] = user_message
        state.answers_by_step[current_step]["status"] = status
        state.answers_by_step[current_step]["reason"] = reason
        state.answers_by_step[current_step]["confidence"] = confidence

        new_state = state

        if status in {"off_topic", "too_short"}:
            reply = await generate_retry_same_step_question(
                fixed_question=fixed_q,
                user_answer=user_message,
                status=status,
            )
            new_state.last_question = reply

            # เก็บ history: AI retry same step
            state.history.append({
                "step": current_step,
                "event": "ai_retry_question",
                "fixed_question": fixed_q,
                "asked_question": reply,
                "based_on_status": status,
            })

        elif status in {"partial", "reflecting", "clear_but_needs_guidance"}:
            # ถ้าข้อนี้เริ่มมีคำตอบที่ใช้ได้แล้ว ค่อย mark completed
            if status == "clear_but_needs_guidance":
                state.answers_by_step[current_step]["is_completed"] = True

            reply = await generate_probe_same_step_question(
                fixed_question=fixed_q,
                user_answer=user_message,
                status=status,
            )
            new_state.last_question = reply

            # เก็บ history: AI probe same step
            state.history.append({
                "step": current_step,
                "event": "ai_probe_question",
                "fixed_question": fixed_q,
                "asked_question": reply,
                "based_on_status": status,
            })

        elif status == "clear_complete":
            state.answers_by_step[current_step]["is_completed"] = True

            next_step = current_step + 1
            next_fixed_q = FIXED_QUESTIONS.get(next_step)

            if next_fixed_q:
                next_question = await generate_next_step_question(
                    fixed_question=next_fixed_q,
                    previous_answer=user_message,
                )
                new_state.step = next_step
                new_state.fixed_question = next_fixed_q
                new_state.last_question = next_question
                reply = next_question

                # เก็บ history: move to next step
                state.history.append({
                    "step": next_step,
                    "event": "ai_next_question",
                    "previous_step": current_step,
                    "fixed_question": next_fixed_q,
                    "asked_question": next_question,
                    "previous_answer": user_message,
                })
            else:
                all_answers = state.answers_by_step

                reply_lines = [
                    "ขอบคุณมากครับ ตอนนี้เราได้สำรวจประเด็นสำคัญครบแล้ว",
                    "",
                    "สรุปคำตอบของคุณ:"
                ]

                for step_no, item in sorted(all_answers.items()):
                    fixed_question_item = item.get("fixed_question", "")
                    user_answer_item = item.get("user_answer", "")
                    status_item = item.get("status", "")

                    reply_lines.append(
                        f"\nข้อ {step_no}\n"
                        f"คำถาม: {fixed_question_item}\n"
                        f"คำตอบ: {user_answer_item}\n"
                        f"สถานะ: {status_item}"
                    )

                reply = "\n".join(reply_lines)

                # เก็บ history: conversation completed
                state.history.append({
                    "step": current_step,
                    "event": "conversation_completed",
                    "summary_count": len(all_answers),
                    "final_reply": reply,
                })

        else:
            reply = "ขอชวนเล่าเพิ่มเติมอีกนิดนะครับ"

            # เก็บ history: fallback
            state.history.append({
                "step": current_step,
                "event": "fallback_reply",
                "fixed_question": fixed_q,
                "asked_question": state.last_question,
                "user_answer": user_message,
            })

    return ChatResponse_aicoach(
        reply=reply,
        state=new_state,
        source="learning_entry",
    )

async def process_chat_aicoach_stream(req: ChatRequest_aicoach, state: ChatState):
    if state is None:
        state = ChatState()

    user_message = (req.user_message or "").strip()

    # map web/member แบบเดียวกับ aicustom ถ้ามี field นี้ใน req/state

    if not user_message:
        reply = "กรุณาพิมพ์สิ่งที่ต้องการพัฒนา / ปัญหาที่อยากแก้"

        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "empty_message",
            "reason": "user_message_empty",
            "state": state,
            "source": "empty_message",
        }
        return

    # --------------------------------------------------
    # phase 1 = ถาม topic ข้อมูล user ที่จำเป้นก่อน
    # --------------------------------------------------

    if state.phase == 1 and state.step == 0:
        final_reply =""
        next_step = 1

        phase_data = PHASES[1]
        rules = phase_data["rules"]
        rule = rules[next_step]

        fixed_q = rule["question"]

        async for item in generate_opening_ai_coach_question_stream(
            fixed_question=fixed_q,
            goal=rule["goal"]
        ):
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}

            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        new_state = ChatState(
            phase=1,
            step=1,
            last_question=final_reply,
            answers_by_step=state.answers_by_step,
            answers=state.answers,
            history=state.history,
            retry_count=0,
        )

        new_state.history.append({
            "phase": 1,
            "step": 1,
            "role": "assistant",
            "event": "ask_question",
            "rule_key": rule["key"],
            "text": final_reply,
        })

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "ask_first_question",
            "reason": "phase1_step0_to_step1",
            "state": new_state,
            "source": "phase1_rules",
        }
        return
    
    elif state.phase in PHASES:

        phase_data = PHASES[state.phase]
        rules = phase_data["rules"]
        rule = rules[state.step]

        result = await evaluate_answer(
            rule=rule,
            user_answer=user_message,
            state=state
        )

        # ----------------------
        # PASS CASE
        # ----------------------
        if result["pass"]:

            state.answers[rule["key"]] = user_message

            max_step = max(rules.keys())

            # next step in same phase
            if state.step < max_step:

                state.step += 1
                next_rule = rules[state.step]

                final_reply = ""

                async for item in ask(
                    state=state,
                    next_rule=next_rule
                ):
                    if item["type"] == "chunk":
                        yield item

                    elif item["type"] == "done":
                        final_reply = item.get("content", final_reply)

                yield {
                    "type": "done",
                    "reply": final_reply,
                    "status": "next_step",
                    "reason": "accepted",
                    "state": state,
                    "source": "engine"
                }
                return

            # ----------------------
            # NEXT PHASE
            # ----------------------
            else:
                state.phase += 1
                state.step = 1

                # ถ้ามี phase ต่อไป
                if state.phase in PHASES:

                    next_phase_data = PHASES[state.phase]
                    next_rule = next_phase_data["rules"][1]

                    final_reply = ""

                    async for item in ask_phase_transition(
                        state=state,
                        next_rule=next_rule,
                        from_phase=PHASES[state.phase - 1]["title"],
                        to_phase=PHASES[state.phase]["title"]
                    ):
                        if item["type"] == "chunk":
                            text = item.get("text", "")
                            final_reply += text
                            yield {
                                "type": "chunk",
                                "text": text
                            }

                        elif item["type"] == "done":
                            final_reply = item.get("content", final_reply)

                    yield {
                        "type": "done",
                        "reply": final_reply,
                        "status": "next_phase",
                        "reason": "phase advanced",
                        "state": state,
                        "source": "engine"
                    }
                    return

                # ไม่มี phase ต่อแล้ว
                # ไม่มี phase ต่อแล้ว = ครบ TGROW แล้ว ให้ AI สรุปผล
                else:
                    final_reply = ""

                    async for item in generate_tgrow_final_summary_stream(state):
                        if item["type"] == "chunk":
                            text = item.get("text", "")
                            final_reply += text
                            yield {
                                "type": "chunk",
                                "text": text
                            }

                        elif item["type"] == "done":
                            final_reply = item.get("content", final_reply)

                    state.history.append({
                        "phase": state.phase,
                        "step": state.step,
                        "role": "assistant",
                        "event": "coaching_complete_summary",
                        "summary": final_reply,
                    })

                    yield {
                        "type": "done",
                        "reply": final_reply,
                        "status": "coaching_complete",
                        "reason": "all phases completed",
                        "state": state,
                        "source": "engine"
                    }
                    return

        # ----------------------
        # FAIL CASE
        # ----------------------
        state.retry_count += 1

        final_reply = ""

        async for item in ask_followup(
            state=state,
            rule=rule,
            eval_result=result,
            user_answer=user_message
        ):
            if item["type"] == "chunk":
                yield item

            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "followup",
            "reason": result["reason"],
            "state": state,
            "source": "engine"
        }
        return

        # if result["pass"]:

        #     state.answers[rule["key"]] = user_message

        #     if state.step < 6:
        #         current_step = state.step
        #         state.step += 1
        #         next_rule = PHASE1_RULES[state.step]

        #         yield {
        #             "type": "done",
        #             "reply": f"[DEBUG] step {current_step} passed -> next step {state.step}\nnext_question: {next_rule['question']}",
        #             "status": "next_step",
        #             "reason": "answer accepted",
        #             "state": state,
        #             "source": "phase1_debug",
        #             "confidence": result.get("confidence"),
        #         }
        #         return

        #     else:
        #         state.phase = 2
        #         state.step = 1

        #         yield {
        #             "type": "done",
        #             "reply": "[DEBUG] phase1 completed -> move to phase2",
        #             "status": "phase_complete",
        #             "reason": "phase1 completed",
        #             "state": state,
        #             "source": "phase1_debug",
        #             "confidence": result.get("confidence"),
        #         }
        #         return

        # else:
        #     state.retry_count += 1

        #     yield {
        #         "type": "done",
        #         "reply": f"[DEBUG] followup needed\nstatus={result['status']}\nreason={result['reason']}\nretry={state.retry_count}",
        #         "status": "followup_needed",
        #         "reason": "answer not sufficient",
        #         "state": state,
        #         "source": "phase1_debug",
        #         "confidence": result.get("confidence"),
        #     }
        #     return


        
        # --------------------------------------------------
        # STEP > 0 = รับคำตอบผู้ใช้ + evaluate
        # --------------------------------------------------
        # fixed_q = state.fixed_question
        # current_step = state.step
        # new_state = state

        # # เก็บ history: user answer
        # state.history.append({
        #     "step": current_step,
        #     "event": "user_answer",
        #     "fixed_question": fixed_q,
        #     "asked_question": state.last_question,
        #     "user_answer": user_message,
        # })

        # check = await evaluate_user_answer(
        #     question=fixed_q,
        #     user_answer=user_message,
        # )

        # status = check.get("status", "off_topic")
        # reason = check.get("reason", "")
        # confidence = check.get("confidence", 0.0)

        # # เก็บ history: evaluation result
        # state.history.append({
        #     "step": current_step,
        #     "event": "evaluation",
        #     "fixed_question": fixed_q,
        #     "asked_question": state.last_question,
        #     "user_answer": user_message,
        #     "status": status,
        #     "reason": reason,
        #     "confidence": confidence,
        #     "raw": check.get("raw", ""),
        # })

        # if current_step not in state.answers_by_step:
        #     state.answers_by_step[current_step] = {
        #         "fixed_question": fixed_q,
        #         "asked_question": state.last_question,
        #         "user_answer": "",
        #         "status": "",
        #         "reason": "",
        #         "confidence": 0.0,
        #         "is_completed": False,
        #     }

        # state.answers_by_step[current_step]["fixed_question"] = fixed_q
        # state.answers_by_step[current_step]["asked_question"] = state.last_question
        # state.answers_by_step[current_step]["user_answer"] = user_message
        # state.answers_by_step[current_step]["status"] = status
        # state.answers_by_step[current_step]["reason"] = reason
        # state.answers_by_step[current_step]["confidence"] = confidence

        # new_state = state

        # # --------------------------------------------------
        # # 1) off_topic / too_short -> retry same step
        # # --------------------------------------------------
        # if status in {"off_topic", "too_short"}:
        #     final_reply = ""
        #     async for item in generate_retry_same_step_question_stream(
        #         fixed_question=fixed_q,
        #         user_answer=user_message,
        #         status=status,
        #     ):
        #         if item["type"] == "chunk":
        #             text = item.get("text", "")
        #             final_reply += text
        #             yield {"type": "chunk", "text": text}
        #         elif item["type"] == "done":
        #             final_reply = item.get("content", final_reply)

        #     new_state.last_question = final_reply

        #     state.history.append({
        #         "step": current_step,
        #         "event": "ai_retry_question",
        #         "fixed_question": fixed_q,
        #         "asked_question": final_reply,
        #         "based_on_status": status,
        #     })

        #     yield {
        #         "type": "done",
        #         "reply": final_reply,
        #         "status": status,
        #         "reason": reason or "retry_same_step",
        #         "confidence": confidence,
        #         "state": new_state,
        #         "source": "learning_entry",
        #     }
        #     return

        # # --------------------------------------------------
        # # 2) partial / reflecting / clear_but_needs_guidance -> probe
        # # --------------------------------------------------
        # elif status in {"partial", "reflecting", "clear_but_needs_guidance"}:
        #     if status == "clear_but_needs_guidance":
        #         state.answers_by_step[current_step]["is_completed"] = True

        #     final_reply = ""
        #     async for item in generate_probe_same_step_question_stream(
        #         fixed_question=fixed_q,
        #         user_answer=user_message,
        #         status=status,
        #     ):
        #         if item["type"] == "chunk":
        #             text = item.get("text", "")
        #             final_reply += text
        #             yield {"type": "chunk", "text": text}
        #         elif item["type"] == "done":
        #             final_reply = item.get("content", final_reply)

        #     new_state.last_question = final_reply

        #     state.history.append({
        #         "step": current_step,
        #         "event": "ai_probe_question",
        #         "fixed_question": fixed_q,
        #         "asked_question": final_reply,
        #         "based_on_status": status,
        #     })

        #     yield {
        #         "type": "done",
        #         "reply": final_reply,
        #         "status": status,
        #         "reason": reason or "probe_same_step",
        #         "confidence": confidence,
        #         "state": new_state,
        #         "source": "learning_entry",
        #     }
        #     return

        # # --------------------------------------------------
        # # 3) clear_complete -> next step or final summary
        # # --------------------------------------------------
        # elif status == "clear_complete":
        #     state.answers_by_step[current_step]["is_completed"] = True

        #     next_step = current_step + 1
        #     next_fixed_q = FIXED_QUESTIONS.get(next_step)

        #     if next_fixed_q:
        #         final_reply = ""
        #         async for item in generate_next_step_question_stream(
        #             fixed_question=next_fixed_q,
        #             previous_answer=user_message,
        #         ):
        #             if item["type"] == "chunk":
        #                 text = item.get("text", "")
        #                 final_reply += text
        #                 yield {"type": "chunk", "text": text}
        #             elif item["type"] == "done":
        #                 final_reply = item.get("content", final_reply)

        #         new_state.step = next_step
        #         new_state.fixed_question = next_fixed_q
        #         new_state.last_question = final_reply

        #         state.history.append({
        #             "step": next_step,
        #             "event": "ai_next_question",
        #             "previous_step": current_step,
        #             "fixed_question": next_fixed_q,
        #             "asked_question": final_reply,
        #             "previous_answer": user_message,
        #         })

        #         yield {
        #             "type": "done",
        #             "reply": final_reply,
        #             "status": "next_step",
        #             "reason": "clear_complete",
        #             "confidence": confidence,
        #             "state": new_state,
        #             "source": "learning_entry",
        #         }
        #         return

        #     # ครบทุกข้อแล้ว
        #     all_answers = state.answers_by_step

        #     reply_lines = [
        #         "ขอบคุณมากครับ ตอนนี้เราได้สำรวจประเด็นสำคัญครบแล้ว",
        #         "",
        #         "สรุปคำตอบของคุณ:"
        #     ]

        #     for step_no, item in sorted(all_answers.items()):
        #         fixed_question_item = item.get("fixed_question", "")
        #         user_answer_item = item.get("user_answer", "")
        #         status_item = item.get("status", "")

        #         reply_lines.append(
        #             f"\nข้อ {step_no}\n"
        #             f"คำถาม: {fixed_question_item}\n"
        #             f"คำตอบ: {user_answer_item}\n"
        #             f"สถานะ: {status_item}"
        #         )

        #     final_reply = "\n".join(reply_lines)

        #     state.history.append({
        #         "step": current_step,
        #         "event": "conversation_completed",
        #         "summary_count": len(all_answers),
        #         "final_reply": final_reply,
        #     })

        #     yield {"type": "chunk", "text": final_reply}
        #     yield {
        #         "type": "done",
        #         "reply": final_reply,
        #         "status": "completed",
        #         "reason": "all_steps_completed",
        #         "confidence": confidence,
        #         "state": new_state,
        #         "source": "learning_entry",
        #     }
        #     return

        # # --------------------------------------------------
        # # 4) fallback
        # # --------------------------------------------------
        # final_reply = "ขอชวนเล่าเพิ่มเติมอีกนิดนะครับ"

        # state.history.append({
        #     "step": current_step,
        #     "event": "fallback_reply",
        #     "fixed_question": fixed_q,
        #     "asked_question": state.last_question,
        #     "user_answer": user_message,
        # })

        # yield {"type": "chunk", "text": final_reply}
        # yield {
        #     "type": "done",
        #     "reply": final_reply,
        #     "status": "fallback",
        #     "reason": reason or "unknown_status",
        #     "confidence": confidence,
        #     "state": new_state,
        #     "source": "learning_entry",
        # }