from app.modules.ai_coach.schema import ChatRequest_aicoach, ChatState, StepAnswer
from app.modules.ai_coach.constants import PHASES
from app.modules.ai_coach.service import (
    classify_coaching_turn,
    decide_dialogue_policy,
    update_coaching_memory,
    step_key,
    ask_opening,
    ask_scope_redirect,
    ask_reframe_scope,
    ask_clarify_same_step,
    ask_deeper_coaching_question,
    ask_next_question,
    ask_phase_transition,
    ask_final_summary,
)


def get_current_rule(state: ChatState) -> dict:
    phase_data = PHASES[state.phase]
    return phase_data["rules"][state.step]


def get_next_position(state: ChatState) -> tuple[int, int] | None:
    phase_data = PHASES[state.phase]
    rules = phase_data["rules"]

    if state.step < max(rules.keys()):
        return state.phase, state.step + 1

    next_phase = state.phase + 1
    if next_phase in PHASES:
        return next_phase, 1

    return None


def save_current_answer(state: ChatState, rule: dict, user_message: str, eval_result: dict):
    key = rule["key"]
    state.answers[key] = eval_result.get("extracted", {}).get(key, user_message)

    item_key = step_key(state.phase, state.step)
    state.answers_by_step[item_key] = StepAnswer(
        phase=state.phase,
        step=state.step,
        rule_key=key,
        question=rule.get("question", ""),
        user_answer=user_message,
        status=eval_result.get("status", ""),
        is_completed=bool(eval_result.get("pass")),
        extracted=eval_result.get("extracted", {}),
    )


def advance_state(state: ChatState):
    next_pos = get_next_position(state)
    if next_pos is None:
        state.is_completed = True
        return None

    next_phase, next_step = next_pos
    old_phase = state.phase

    state.phase = next_phase
    state.step = next_step
    state.retry_count = 0

    return {
        "old_phase": old_phase,
        "new_phase": next_phase,
        "new_step": next_step,
        "is_phase_transition": old_phase != next_phase,
    }


async def process_chat_aicoach_stream(req: ChatRequest_aicoach, state: ChatState):
    if state is None:
        state = ChatState()

    user_message = (req.user_message or "").strip()

    if not user_message:
        reply = "กรุณาพิมพ์เรื่องที่อยากพัฒนา หรือประเด็นที่อยากปรึกษาโค้ชครับ"
        yield {"type": "chunk", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "status": "empty_message",
            "reason": "user_message_empty",
            "state": state,
            "source": "coach_engine_v2",
        }
        return

    if state.is_completed:
        state = ChatState()

    # start conversation
    if state.phase == 1 and state.step == 0:
        state.step = 1
        rule = get_current_rule(state)
        final_reply = ""

        async for item in ask_opening(rule):
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}
            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        state.last_question = final_reply
        state.history.append({
            "phase": state.phase,
            "step": state.step,
            "role": "assistant",
            "event": "ask_opening",
            "rule_key": rule["key"],
            "text": final_reply,
        })

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "ask_first_question",
            "reason": "start_coaching",
            "state": state,
            "source": "coach_engine_v2",
        }
        return

    rule = get_current_rule(state)

    state.history.append({
        "phase": state.phase,
        "step": state.step,
        "role": "user",
        "event": "user_answer",
        "rule_key": rule["key"],
        "question": state.last_question,
        "text": user_message,
    })

    classification = await classify_coaching_turn(
        rule=rule,
        user_answer=user_message,
        state=state,
    )
    eval_result = classification["eval_result"]
    analysis = classification["analysis"]

    # print(f"eval_result", eval_result,flush=True)
    # print(f"analysis", analysis,flush=True)

    policy = decide_dialogue_policy(
        eval_result=eval_result,
        analysis=analysis,
        state=state,
        rule=rule,
        phase_rules=PHASES[state.phase]["rules"],
    )
    print(f"policy", policy,flush=True)
    update_coaching_memory(state, analysis, policy)

    state.history.append({
        "phase": state.phase,
        "step": state.step,
        "role": "system",
        "event": "coach_decision",
        "rule_key": rule["key"],
        "eval_result": eval_result,
        "analysis": analysis,
        "policy": policy,
    })

    if policy.get("save_answer"):
        save_current_answer(state, rule, user_message, eval_result)

    action = policy["action"]
    final_reply = ""

    if action == "redirect_scope":
        state.retry_count += 1
        async for item in ask_scope_redirect(user_message):
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}
            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        state.last_question = final_reply

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "redirect_scope",
            "reason": policy.get("reason"),
            "confidence": eval_result.get("confidence"),
            "state": state,
            "source": "coach_engine_v2",
        }
        return

    if action == "reframe_scope":
        state.retry_count += 1
        async for item in ask_reframe_scope(user_message):
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}
            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        state.last_question = final_reply

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "reframe_scope",
            "reason": policy.get("reason"),
            "confidence": eval_result.get("confidence"),
            "state": state,
            "source": "coach_engine_v2",
        }
        return

    if action == "clarify_same_step":
        state.retry_count += 1
        async for item in ask_clarify_same_step(state, rule, user_message, eval_result):
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}
            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        state.last_question = final_reply

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "clarify_same_step",
            "reason": policy.get("reason"),
            "confidence": eval_result.get("confidence"),
            "state": state,
            "source": "coach_engine_v2",
        }
        return

    if action in {"probe_deeper", "reflect_and_probe"}:
        state.retry_count += 1
        async for item in ask_deeper_coaching_question(state, rule, user_message, analysis):
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}
            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        state.last_question = final_reply

        yield {
            "type": "done",
            "reply": final_reply,
            "status": action,
            "reason": policy.get("reason"),
            "confidence": eval_result.get("confidence"),
            "state": state,
            "source": "coach_engine_v2",
        }
        return

    if action in {"ask_next", "summarize_then_next"}:
        advance_info = advance_state(state)

        if state.is_completed:
            async for item in ask_final_summary(state):
                if item["type"] == "chunk":
                    text = item.get("text", "")
                    final_reply += text
                    yield {"type": "chunk", "text": text}
                elif item["type"] == "done":
                    final_reply = item.get("content", final_reply)

            state.last_question = final_reply
            state.history.append({
                "role": "assistant",
                "event": "final_summary",
                "text": final_reply,
            })

            yield {
                "type": "done",
                "reply": final_reply,
                "status": "coaching_complete",
                "reason": "all_phases_completed",
                "confidence": eval_result.get("confidence"),
                "state": state,
                "source": "coach_engine_v2",
            }
            return

        next_rule = get_current_rule(state)

        if advance_info and advance_info["is_phase_transition"]:
            from_phase = PHASES[advance_info["old_phase"]]["title"]
            to_phase = PHASES[state.phase]["title"]
            stream = ask_phase_transition(state, from_phase, to_phase, next_rule)
            status = "next_phase"
        else:
            print(f"next_rule = ", next_rule,flush=True)
            stream = ask_next_question(state, next_rule)
            status = "next_step"

        async for item in stream:
            if item["type"] == "chunk":
                text = item.get("text", "")
                final_reply += text
                yield {"type": "chunk", "text": text}
            elif item["type"] == "done":
                final_reply = item.get("content", final_reply)

        state.last_question = final_reply
        state.history.append({
            "phase": state.phase,
            "step": state.step,
            "role": "assistant",
            "event": status,
            "rule_key": next_rule["key"],
            "text": final_reply,
        })

        yield {
            "type": "done",
            "reply": final_reply,
            "status": status,
            "reason": policy.get("reason"),
            "confidence": eval_result.get("confidence"),
            "state": state,
            "source": "coach_engine_v2",
        }
        return

    fallback = "ผมขอชวนคุณเล่าเพิ่มอีกนิดนะครับ เพื่อให้เราเห็นประเด็นนี้ชัดขึ้น"
    state.last_question = fallback
    yield {"type": "chunk", "text": fallback}
    yield {
        "type": "done",
        "reply": fallback,
        "status": "fallback",
        "reason": "unknown_policy_action",
        "state": state,
        "source": "coach_engine_v2",
    }