from dotenv import load_dotenv
load_dotenv()

import json
import re
from typing import Any, Dict, AsyncGenerator

from app.modules.ai_coach.schema import ChatState
from app.shared.ai.openai_client import call_openai_chat_full, call_openai_chat_stream_full


MODEL_DEFAULT = "gpt-4.1-mini"


def clean_json(text: str) -> str:
    return re.sub(r"```json|```", "", text or "").strip()


def safe_json_loads(text: str, fallback: Dict[str, Any] | None = None) -> Dict[str, Any]:
    fallback = fallback or {}
    try:
        return json.loads(clean_json(text))
    except Exception:
        return fallback


def unique_extend(target: list, values: list, limit: int = 12):
    for value in values or []:
        value = str(value).strip()
        if value and value not in target:
            target.append(value)
    del target[limit:]


async def _stream_text_response(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = MODEL_DEFAULT,
    temperature: float = 0.5,
) -> AsyncGenerator[Dict[str, Any], None]:
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
                yield {"type": "chunk", "text": text}

        elif item.get("type") == "done":
            content = (item.get("content") or final_content).strip()
            yield {
                "type": "done",
                "content": content,
                "usage": item.get("usage"),
                "cost": item.get("cost"),
            }
            return


async def evaluate_answer(rule: dict, user_answer: str, state: ChatState, model: str = MODEL_DEFAULT) -> dict:
    text = (user_answer or "").strip()

    if not text:
        return {
            "ok": True,
            "pass": False,
            "status": "too_short",
            "reason": "ผู้ใช้ยังไม่ได้ตอบคำถาม",
            "confidence": 1.0,
            "extracted": {},
            "raw": "",
        }

    short_words = {"ครับ", "ค่ะ", "คับ", "จ้า", "อือ", "อืม", "โอเค", "ok", "yes", "ไม่รู้", "ไม่แน่ใจ", "เฉยๆ"}
    if text.lower() in short_words or len(text) <= 2:
        return {
            "ok": True,
            "pass": False,
            "status": "too_short",
            "reason": "คำตอบสั้นเกินไป",
            "confidence": 0.98,
            "extracted": {},
            "raw": "",
        }

    previous_context = json.dumps(state.answers, ensure_ascii=False, indent=2)

    system_prompt = """
คุณคือระบบประเมินคำตอบของผู้ใช้สำหรับ AI Coach ของสถาบันฝึกอบรม

หน้าที่:
- ประเมินว่าคำตอบตรงกับเป้าหมายของคำถามหรือไม่
- ไม่ใช่การจับผิด แต่ดูว่ามีข้อมูลพอให้โค้ชไปต่อได้ไหม
- ใช้บริบทคำตอบก่อนหน้าประกอบ

สถานะที่อนุญาต:
- accepted = ตอบตรงและพอใช้ไปต่อ
- partial = ตอบถูกทางแต่ยังไม่พอ
- off_topic = ไม่ตรงคำถาม
- unclear = กำกวม ยังตีความไม่ได้
- too_short = สั้นเกินไป

กฎ:
- ถ้า answer_type เป็น emotion แล้วผู้ใช้บอกความรู้สึกชัด ให้ accepted ได้
- ถ้า answer_type เป็น topic ต้องเป็นหัวข้อที่ชัดเจน
- ถ้า answer_type เป็น goal ต้องมีเป้าหมายหรือทิศทางที่อยากเปลี่ยน
- ถ้า answer_type เป็น score ต้องมีคะแนนหรือตัวชี้วัดใกล้เคียง
- ถ้าคำตอบมีสาระพอ แม้ไม่ยาว ให้ accepted

ตอบ JSON เท่านั้น:
{
  "pass": true,
  "status": "accepted",
  "reason": "ตอบตรงประเด็น",
  "confidence": 0.9,
  "extracted_value": "สาระสำคัญที่ดึงได้"
}
""".strip()

    user_prompt = f"""
ข้อมูลก่อนหน้า:
{previous_context}

คำถามปัจจุบัน:
{rule.get("question", "")}

เป้าหมายคำถาม:
{rule.get("goal", "")}

ประเภทคำตอบ:
{rule.get("answer_type", "")}

สิ่งที่ต้องการ:
{", ".join(rule.get("required", []))}

คำตอบล่าสุด:
{text}
""".strip()

    try:
        result = await call_openai_chat_full(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        raw = clean_json(result.get("content", ""))
        data = safe_json_loads(raw)

        allowed = {"accepted", "partial", "off_topic", "unclear", "too_short"}
        status = str(data.get("status", "unclear")).strip()
        if status not in allowed:
            status = "unclear"

        passed = bool(data.get("pass", False)) and status == "accepted"
        extracted_value = data.get("extracted_value") or text

        return {
            "ok": True,
            "pass": passed,
            "status": status,
            "reason": str(data.get("reason", "")).strip(),
            "confidence": float(data.get("confidence", 0.0)),
            "extracted": {rule.get("key", "answer"): extracted_value},
            "raw": raw,
        }
    except Exception as e:
        return {
            "ok": False,
            "pass": False,
            "status": "unclear",
            "reason": f"parse_error: {str(e)}",
            "confidence": 0.0,
            "extracted": {},
            "raw": "",
        }


async def analyze_coaching_context(rule: dict, user_answer: str, eval_result: dict, state: ChatState, model: str = MODEL_DEFAULT) -> dict:
    previous_context = json.dumps(state.answers, ensure_ascii=False, indent=2)
    memory_context = state.coaching_memory.model_dump()

    system_prompt = """
คุณคือ Coaching Analyzer สำหรับ AI Coach ของสถาบันฝึกอบรม

หน้าที่:
วิเคราะห์คำตอบเชิง coaching ไม่ใช่แค่ pass/fail โดยดูว่าในคำตอบมีสัญญาณอะไรที่ควรพาโค้ชต่อ

ขอบเขตของ AI Coach นี้:
valid:
- การเรียนรู้ การพัฒนาทักษะ การทำงาน competency performance leadership career growth communication teamwork problem solving mindset เพื่อการพัฒนา

invalid:
- เรื่องสัตว์เลี้ยงทั่วไป งานอดิเรก สุขภาพทั่วไป ความสัมพันธ์ส่วนตัว หรือเรื่องส่วนตัวที่ไม่เกี่ยวกับการเรียนรู้/การทำงาน/การพัฒนา

needs_reframe:
- เรื่องทั่วไปที่อาจโยงกับการพัฒนาได้ ถ้าผู้ใช้ตั้งใจพัฒนาทักษะหรือพฤติกรรมผ่านเรื่องนั้นจริง

ให้วิเคราะห์:
- main_signal: clarity | fear | confusion | motivation | resistance | action_ready | neutral
- scope_status: valid | invalid | needs_reframe
- readiness: low | medium | high | unknown
- depth_needed: shallow | medium | deep
- coaching_opportunity: clarify_goal | surface_blocker | build_confidence | move_to_action | set_boundary | continue_flow

ตอบ JSON เท่านั้น:
{
  "main_signal": "clarity",
  "scope_status": "valid",
  "emotion": "",
  "themes": [],
  "blockers": [],
  "strengths": [],
  "risks": [],
  "readiness": "medium",
  "depth_needed": "medium",
  "coaching_opportunity": "continue_flow",
  "reason": "เหตุผลสั้น ๆ"
}
""".strip()

    user_prompt = f"""
คำตอบก่อนหน้า:
{previous_context}

coaching memory ปัจจุบัน:
{json.dumps(memory_context, ensure_ascii=False, indent=2)}

คำถามปัจจุบัน:
{rule.get("question", "")}

เป้าหมายคำถาม:
{rule.get("goal", "")}

ผล evaluate:
{json.dumps(eval_result, ensure_ascii=False, indent=2)}

คำตอบล่าสุดของผู้ใช้:
{user_answer}
""".strip()

    try:
        result = await call_openai_chat_full(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        raw = clean_json(result.get("content", ""))
        data = safe_json_loads(raw)

        return {
            "main_signal": data.get("main_signal", "neutral"),
            "scope_status": data.get("scope_status", "valid"),
            "emotion": data.get("emotion", ""),
            "themes": data.get("themes", []) or [],
            "blockers": data.get("blockers", []) or [],
            "strengths": data.get("strengths", []) or [],
            "risks": data.get("risks", []) or [],
            "readiness": data.get("readiness", "unknown"),
            "depth_needed": data.get("depth_needed", "medium"),
            "coaching_opportunity": data.get("coaching_opportunity", "continue_flow"),
            "reason": data.get("reason", ""),
            "raw": raw,
        }
    except Exception as e:
        return {
            "main_signal": "neutral",
            "scope_status": "valid",
            "emotion": "",
            "themes": [],
            "blockers": [],
            "strengths": [],
            "risks": [],
            "readiness": "unknown",
            "depth_needed": "medium",
            "coaching_opportunity": "continue_flow",
            "reason": f"parse_error: {str(e)}",
            "raw": "",
        }


def decide_dialogue_policy(eval_result: dict, analysis: dict, state: ChatState, rule: dict, phase_rules: dict) -> dict:
    scope_status = analysis.get("scope_status", "valid")
    status = eval_result.get("status", "unclear")
    main_signal = analysis.get("main_signal", "neutral")
    depth_needed = analysis.get("depth_needed", "medium")
    readiness = analysis.get("readiness", "unknown")

    if rule.get("allow_scope_redirect") and scope_status == "invalid":
        return {
            "action": "redirect_scope",
            "advance_step": False,
            "save_answer": False,
            "coaching_intent": "set_boundary",
            "reason": "topic_outside_training_scope",
        }

    if rule.get("allow_scope_redirect") and scope_status == "needs_reframe":
        return {
            "action": "reframe_scope",
            "advance_step": False,
            "save_answer": False,
            "coaching_intent": "reframe_to_development",
            "reason": "topic_needs_reframe",
        }

    if status in {"too_short", "off_topic", "unclear"}:
        return {
            "action": "clarify_same_step",
            "advance_step": False,
            "save_answer": False,
            "coaching_intent": "make_question_easier",
            "reason": status,
        }

    if eval_result.get("pass") and main_signal in {"fear", "confusion", "resistance"} and state.retry_count < 1:
        return {
            "action": "probe_deeper",
            "advance_step": False,
            "save_answer": True,
            "coaching_intent": "surface_blocker",
            "reason": f"signal:{main_signal}",
        }

    if eval_result.get("pass") and depth_needed == "deep" and state.retry_count < 1:
        return {
            "action": "reflect_and_probe",
            "advance_step": False,
            "save_answer": True,
            "coaching_intent": "create_insight",
            "reason": "depth_needed",
        }

    if eval_result.get("pass") and readiness == "high" and state.phase < 5 and state.phase >= 3:
        return {
            "action": "summarize_then_next",
            "advance_step": True,
            "save_answer": True,
            "coaching_intent": "continue_with_momentum",
            "reason": "high_readiness",
        }

    if eval_result.get("pass"):
        return {
            "action": "ask_next",
            "advance_step": True,
            "save_answer": True,
            "coaching_intent": "continue_flow",
            "reason": "accepted",
        }

    return {
        "action": "clarify_same_step",
        "advance_step": False,
        "save_answer": False,
        "coaching_intent": "clarify",
        "reason": "fallback",
    }


def update_coaching_memory(state: ChatState, analysis: dict, policy: dict):
    memory = state.coaching_memory

    unique_extend(memory.themes, analysis.get("themes", []))
    unique_extend(memory.blockers, analysis.get("blockers", []))
    unique_extend(memory.strengths, analysis.get("strengths", []))
    unique_extend(memory.risks, analysis.get("risks", []))

    emotion = str(analysis.get("emotion", "")).strip()
    if emotion:
        unique_extend(memory.emotions, [emotion])

    memory.readiness = analysis.get("readiness", memory.readiness)
    memory.scope_status = analysis.get("scope_status", memory.scope_status)
    memory.last_signal = analysis.get("main_signal", "")
    memory.last_policy_action = policy.get("action", "")


def step_key(phase: int, step: int) -> str:
    return f"p{phase}_s{step}"


async def ask_opening(rule: dict):
    system_prompt = """
คุณคือ AI Coach ของสถาบันฝึกอบรม

หน้าที่:
- เปิดบทสนทนาอย่างอบอุ่นและมืออาชีพ
- บอกกรอบสั้น ๆ ว่าโค้ชนี้เน้นการเรียนรู้ การพัฒนาทักษะ การทำงาน และเป้าหมายการเติบโต
- ค่อย ๆ เชื่อมเข้าสู่คำถามหลัก

กฎ:
- ไม่ถามหลายคำถาม
- ไม่เหมือนแบบฟอร์ม
- 2-4 ประโยค
- ประโยคสุดท้ายเป็นคำถามเดียว
- ภาษาไทย
""".strip()

    user_prompt = f"""
คำถามหลัก:
{rule["question"]}

เป้าหมาย:
{rule["goal"]}
""".strip()

    async for item in _stream_text_response(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.5):
        yield item


async def ask_scope_redirect(user_answer: str):
    system_prompt = """
คุณคือ AI Coach ของสถาบันฝึกอบรม

ผู้ใช้เสนอหัวข้อที่อยู่นอกขอบเขตของโค้ชนี้
ให้ตอบแบบ:
1. ยอมรับว่าสิ่งที่ผู้ใช้พูดมีความหมายได้
2. อธิบายขอบเขตอย่างสุภาพว่าโค้ชนี้เน้นการเรียนรู้ การพัฒนาทักษะ การทำงาน และเป้าหมายด้านการเติบโต
3. ชวนกลับมาที่หัวข้อในขอบเขต

ห้าม:
- ห้ามทำเหมือนเรื่องของผู้ใช้ไม่สำคัญ
- ห้าม force mapping หัวข้อนั้นให้เป็น skill แบบฝืน ๆ
- ห้ามถามหลายคำถาม
- 2-3 ประโยค
""".strip()

    user_prompt = f"หัวข้อที่ผู้ใช้เสนอ: {user_answer}"

    async for item in _stream_text_response(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.4):
        yield item


async def ask_reframe_scope(user_answer: str):
    system_prompt = """
คุณคือ AI Coach ของสถาบันฝึกอบรม

ผู้ใช้เสนอหัวข้อที่อาจเป็นเรื่องทั่วไป แต่สามารถโยงกับการพัฒนาได้ถ้าผู้ใช้ตั้งใจพัฒนาทักษะหรือพฤติกรรมบางอย่าง
ให้ช่วย reframe อย่างนุ่มนวล โดยถามว่าผู้ใช้อยากพัฒนาอะไรจากเรื่องนี้ในมุมทักษะ การเรียนรู้ การทำงาน หรือการเติบโต

ห้ามถามหลายคำถาม
ห้ามยาว
""".strip()

    user_prompt = f"หัวข้อที่ผู้ใช้เสนอ: {user_answer}"

    async for item in _stream_text_response(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.5):
        yield item


async def ask_clarify_same_step(state: ChatState, rule: dict, user_answer: str, eval_result: dict):
    system_prompt = """
คุณคือ AI Coach ของสถาบันฝึกอบรม

หน้าที่:
- ผู้ใช้ยังตอบไม่ชัดหรือตอบสั้นไป
- สะท้อนสิ่งที่ผู้ใช้พูดแบบไม่ตัดสิน
- ถามคำถามเดิมให้ง่ายขึ้นและเฉพาะขึ้น

กฎ:
- ถามเพียง 1 คำถาม
- ไม่เปิดหัวข้อใหม่
- ไม่เหมือน validator
- 1-3 ประโยค
""".strip()

    user_prompt = f"""
คำถามเดิม:
{rule["question"]}

เป้าหมายคำถาม:
{rule["goal"]}

คำตอบผู้ใช้:
{user_answer}

ผลประเมิน:
{json.dumps(eval_result, ensure_ascii=False)}
""".strip()

    async for item in _stream_text_response(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.6):
        yield item


async def ask_deeper_coaching_question(state: ChatState, rule: dict, user_answer: str, analysis: dict):
    system_prompt = """
คุณคือ AI Coach มืออาชีพของสถาบันฝึกอบรม

หน้าที่:
- ผู้ใช้ตอบพอใช้แล้ว แต่มีสัญญาณสำคัญเชิง coaching เช่น ความกลัว ความสับสน ความลังเล หรือ blocker
- ให้สะท้อน insight จากคำตอบผู้ใช้
- ถามคำถามเดียวที่ช่วยให้ผู้ใช้เห็นตัวเองชัดขึ้น

กฎ:
- ห้ามให้คำแนะนำยาว
- ห้ามถามหลายคำถาม
- ห้ามถามเหมือนแบบสอบถาม
- คำถามต้องเชื่อมกับสิ่งที่ผู้ใช้พูดจริง
- 2-3 ประโยค
""".strip()

    user_prompt = f"""
คำถามเดิม:
{rule["question"]}

คำตอบผู้ใช้:
{user_answer}

coaching analysis:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

memory:
{json.dumps(state.coaching_memory.model_dump(), ensure_ascii=False, indent=2)}
""".strip()

    async for item in _stream_text_response(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.65):
        yield item


async def ask_next_question(state: ChatState, next_rule: dict):
    memory = state.coaching_memory.model_dump()
    answers = state.answers

    system_prompt = """
คุณคือ AI Coach ของสถาบันฝึกอบรมที่สนทนาแบบธรรมชาติ

หน้าที่:
- พาบทสนทนาไปคำถามถัดไปอย่างลื่นไหล
- อ้างอิงสิ่งที่ผู้ใช้เล่ามาเล็กน้อย
- ถามคำถามเดียวตาม next_rule

กฎ:
- ไม่เหมือน checklist
- 2-4 ประโยค
- ประโยคสุดท้ายเป็นคำถามเดียว
- ไม่สรุปยาว
""".strip()

    user_prompt = f"""
คำตอบที่เก็บแล้ว:
{json.dumps(answers, ensure_ascii=False, indent=2)}

coaching memory:
{json.dumps(memory, ensure_ascii=False, indent=2)}

คำถามถัดไป:
{next_rule["question"]}

เป้าหมายคำถาม:
{next_rule["goal"]}
""".strip()

    async for item in _stream_text_response(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.6):
        yield item


async def ask_phase_transition(state: ChatState, from_phase_title: str, to_phase_title: str, next_rule: dict):
    system_prompt = """
คุณคือ AI Coach ของสถาบันฝึกอบรม

หน้าที่:
- เชื่อมจาก phase เดิมไป phase ใหม่อย่างเป็นธรรมชาติ
- สรุปสั้นมากว่าได้เห็นอะไรจากผู้ใช้
- ถามคำถามแรกของ phase ใหม่

กฎ:
- ไม่เกิน 4 ประโยค
- ถามเพียง 1 คำถาม
- ไม่ใช้ bullet
""".strip()

    user_prompt = f"""
จาก phase:
{from_phase_title}

ไป phase:
{to_phase_title}

answers:
{json.dumps(state.answers, ensure_ascii=False, indent=2)}

memory:
{json.dumps(state.coaching_memory.model_dump(), ensure_ascii=False, indent=2)}

คำถามแรกของ phase ใหม่:
{next_rule["question"]}
""".strip()

    async for item in _stream_text_response(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.55):
        yield item


async def ask_final_summary(state: ChatState):
    system_prompt = """
คุณคือ AI Coach ของสถาบันฝึกอบรม

หน้าที่:
- สรุปผลการโค้ชตาม TGROW
- ใช้ข้อมูลจากผู้ใช้เท่านั้น ห้ามแต่งข้อมูลใหม่
- ทำให้เป็นแผนปฏิบัติที่ชัดเจนและให้กำลังใจ

รูปแบบ:
1. ประเด็นที่ต้องการพัฒนา
2. เป้าหมาย
3. สถานการณ์ปัจจุบัน / อุปสรรค
4. ทางเลือกที่เลือก
5. สิ่งที่จะลงมือทำ
6. ความมั่นใจ / ข้อชวนคิดสั้น ๆ
""".strip()

    user_prompt = f"""
answers:
{json.dumps(state.answers, ensure_ascii=False, indent=2)}

coaching memory:
{json.dumps(state.coaching_memory.model_dump(), ensure_ascii=False, indent=2)}
""".strip()

    async for item in _stream_text_response(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.4):
        yield item