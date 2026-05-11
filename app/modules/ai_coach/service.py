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


async def classify_coaching_turn(
    rule: dict,
    user_answer: str,
    state: ChatState,
    model: str = MODEL_DEFAULT,
) -> dict:
    text = (user_answer or "").strip()
    previous_context = json.dumps(state.answers, ensure_ascii=False, indent=2)
    memory_context = state.coaching_memory.model_dump()

    system_prompt = """
คุณคือ Coaching Turn Classifier สำหรับ AI Coach ของสถาบันฝึกอบรม

หน้าที่:
วิเคราะห์คำตอบล่าสุดของผู้ใช้ใน 2 มิติพร้อมกัน:
1. ประเมินว่าคำตอบตอบโจทย์คำถามปัจจุบันพอให้ไปต่อไหม
2. วิเคราะห์สัญญาณเชิง coaching เพื่อช่วยเลือกวิธีตอบถัดไป

ขอบเขตของ AI Coach:
valid:
- การเรียนรู้ การพัฒนาทักษะ การทำงาน competency performance leadership career growth communication teamwork problem solving mindset เพื่อการพัฒนา

invalid:
- เรื่องสัตว์เลี้ยงทั่วไป งานอดิเรก สุขภาพทั่วไป ความสัมพันธ์ส่วนตัว หรือเรื่องส่วนตัวที่ไม่เกี่ยวกับการเรียนรู้/การทำงาน/การพัฒนา

needs_reframe:
- เรื่องทั่วไปที่อาจโยงกับการพัฒนาได้ ถ้าผู้ใช้ตั้งใจพัฒนาทักษะหรือพฤติกรรมผ่านเรื่องนั้นจริง

eval.status ที่อนุญาต:
- accepted = ตอบตรงและพอใช้ไปต่อ
- partial = ตอบถูกทางแต่ยังไม่พอ
- off_topic = ไม่ตรงคำถาม
- unclear = กำกวม ยังตีความไม่ได้
- too_short = สั้นเกินไป
- unsure = ผู้ใช้บอกว่ายังไม่แน่ใจ / ไม่รู้ / คิดไม่ออก

analysis.main_signal:
- clarity | fear | confusion | motivation | resistance | action_ready | neutral

analysis.scope_status:
- valid | invalid | needs_reframe

analysis.readiness:
- low | medium | high | unknown

analysis.depth_needed:
- shallow | medium | deep

analysis.coaching_opportunity:
- clarify_goal | surface_blocker | build_confidence | move_to_action | set_boundary | continue_flow | reduce_cognitive_load

กฎสำคัญ:
- ถ้าผู้ใช้ตอบว่าไม่แน่ใจ / ไม่รู้ / คิดไม่ออก ให้ eval.status = unsure
- อย่าตัดสินว่า too_short เพียงเพราะคำตอบสั้น ถ้ามีสาระพอให้ไปต่อ
- ถ้า answer_type เป็น topic และผู้ใช้ให้หัวข้อชัด ให้ accepted
- ถ้า answer_type เป็น emotion และผู้ใช้บอกความรู้สึกชัด ให้ accepted
- ถ้า answer_type เป็น goal ต้องมีเป้าหมายหรือทิศทางที่อยากเปลี่ยน
- ถ้า answer_type เป็น score ต้องมีคะแนนหรือตัวชี้วัดใกล้เคียง
- ใช้บริบทก่อนหน้าและ coaching memory ประกอบ

ตอบ JSON เท่านั้น:
{
  "eval": {
    "pass": true,
    "status": "accepted",
    "reason": "ตอบตรงประเด็น",
    "confidence": 0.9,
    "extracted_value": "สาระสำคัญที่ดึงได้"
  },
  "analysis": {
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

ประเภทคำตอบ:
{rule.get("answer_type", "")}

สิ่งที่ต้องการ:
{", ".join(rule.get("required", []))}

คำตอบล่าสุด:
{text}
""".strip()

    fallback = {
        "eval": {
            "ok": False,
            "pass": False,
            "status": "unclear",
            "reason": "classification_fallback",
            "confidence": 0.0,
            "extracted": {},
            "raw": "",
        },
        "analysis": {
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
            "reason": "classification_fallback",
            "raw": "",
        },
    }

    if not text:
        fallback["eval"].update({
            "ok": True,
            "status": "too_short",
            "reason": "ผู้ใช้ยังไม่ได้ตอบคำถาม",
            "confidence": 1.0,
        })
        return fallback

    try:
        result = await call_openai_chat_full(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        raw = clean_json(result.get("content", ""))
        data = safe_json_loads(raw, fallback)

        eval_data = data.get("eval", {})
        analysis_data = data.get("analysis", {})

        allowed_status = {
            "accepted",
            "partial",
            "off_topic",
            "unclear",
            "too_short",
            "unsure",
        }

        status = str(eval_data.get("status", "unclear")).strip()
        if status not in allowed_status:
            status = "unclear"

        passed = bool(eval_data.get("pass", False)) and status == "accepted"
        extracted_value = eval_data.get("extracted_value") or text

        eval_result = {
            "ok": True,
            "pass": passed,
            "status": status,
            "reason": str(eval_data.get("reason", "")).strip(),
            "confidence": float(eval_data.get("confidence", 0.0)),
            "extracted": {rule.get("key", "answer"): extracted_value},
            "raw": raw,
        }

        analysis = {
            "main_signal": analysis_data.get("main_signal", "neutral"),
            "scope_status": analysis_data.get("scope_status", "valid"),
            "emotion": analysis_data.get("emotion", ""),
            "themes": analysis_data.get("themes", []) or [],
            "blockers": analysis_data.get("blockers", []) or [],
            "strengths": analysis_data.get("strengths", []) or [],
            "risks": analysis_data.get("risks", []) or [],
            "readiness": analysis_data.get("readiness", "unknown"),
            "depth_needed": analysis_data.get("depth_needed", "medium"),
            "coaching_opportunity": analysis_data.get("coaching_opportunity", "continue_flow"),
            "reason": analysis_data.get("reason", ""),
            "raw": raw,
        }

        return {
            "eval_result": eval_result,
            "analysis": analysis,
            "raw": raw,
        }

    except Exception as e:
        fallback["eval"]["reason"] = f"classification_error: {str(e)}"
        fallback["analysis"]["reason"] = f"classification_error: {str(e)}"
        return {
            "eval_result": fallback["eval"],
            "analysis": fallback["analysis"],
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


async def ask_clarify_same_step(
    state: ChatState,
    rule: dict,
    user_answer: str,
    eval_result: dict,
):
    memory = state.coaching_memory.model_dump()
    answers = state.answers

    system_prompt = """
คุณคือ AI Coach ที่กำลังสนทนาแบบธรรมชาติ ไม่ใช่แบบสอบถาม

หน้าที่:
- ผู้ใช้ยังตอบคำถามเดิมไม่ชัดพอ หรือตอบสั้นเกินไป
- ช่วยพาผู้ใช้กลับมาที่คำถามเดิมอย่างนุ่มนวล
- ทำให้ผู้ใช้รู้สึกว่าถูกเข้าใจ ไม่ใช่ถูกตรวจคำตอบ
- ถามคำถามเดิมให้ง่ายขึ้น เฉพาะขึ้น และตอบได้ง่ายขึ้น

โครงสร้างคำตอบ:
1. สะท้อนสิ่งที่ผู้ใช้พูดล่าสุดเล็กน้อย
2. เชื่อมกลับมาที่คำถามเดิมอย่างเป็นธรรมชาติ
3. ถามคำถามเดียวตามคำถามหลักเดิม

หลักการ:
- ใช้คำถามหลักเดิมเป็นแกน
- รักษาเป้าหมายของ step เดิม
- ห้ามเปลี่ยนเจตนาของคำถาม
- ห้ามเปิดหัวข้อใหม่จากคำตอบของผู้ใช้
- ฟังดูเป็นคนคุยกันจริง
- 2-4 ประโยค
- ประโยคสุดท้ายต้องเป็นคำถามเดียว

Reflection:
- สะท้อนแบบ coaching ไม่ใช่สรุปยาว
- สะท้อนเฉพาะคำตอบล่าสุดของผู้ใช้
- ห้ามดึงข้อมูลเก่ามาสรุปซ้ำ ถ้าไม่จำเป็น
- ห้ามสรุปแทนผู้ใช้เกินกว่าที่เขาพูด

ห้าม:
- ห้ามถามหลายคำถาม
- ห้ามเหมือน validator
- ห้ามเหมือน checklist
- ห้ามข้ามไปถาม step อื่น
- ห้ามให้คำแนะนำยาว
""".strip()

    user_prompt = f"""
คำตอบที่เก็บแล้ว:
{json.dumps(answers, ensure_ascii=False, indent=2)}

coaching memory:
{json.dumps(memory, ensure_ascii=False, indent=2)}

key:
{rule["key"]}

คำถามหลักเดิม:
{rule["question"]}

goal ที่ต้องการได้:
{rule["goal"]}

ข้อมูลที่ต้องการ:
{", ".join(rule.get("required", []))}

ประเภทคำตอบ:
{rule.get("answer_type", "")}

คำตอบล่าสุดของผู้ใช้:
{user_answer}

ผลประเมิน:
{json.dumps(eval_result, ensure_ascii=False)}

คำสั่ง:
ช่วยตอบแบบโค้ชที่ต่อบทสนทนาจากคำตอบล่าสุดของผู้ใช้
แต่ต้องพากลับมาที่คำถามหลักเดิม ไม่ใช่ถามคำถามใหม่
""".strip()

    async for item in _stream_text_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.55,
    ):
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
คุณคือ AI Coach ที่กำลังสนทนาแบบธรรมชาติ ไม่ใช่แบบสอบถาม

หน้าที่:
- สร้างบทสนทนาที่ต่อเนื่องและเป็นมนุษย์
- ต้องมีการตอบรับสิ่งที่ผู้ใช้พูดก่อนถามคำถาม
- ทำให้ผู้ใช้รู้สึกว่าถูกเข้าใจ ไม่ใช่ถูกสอบ

โครงสร้างคำตอบ:
1. สะท้อนสิ่งที่ผู้ใช้เล็กน้อย
2. เชื่อมเข้าบริบทของคำถามถัดไป
3. ถามคำถามเดียวตามคำถามหลัก

หลักการ:
- ใช้คำถามหลักเป็นแกน
- รักษาเป้าหมายของ step เดิม
- ห้ามเปลี่ยนเจตนาของคำถาม
- ฟังดูเป็นคนคุยกันจริง
- 2-4 ประโยค
- ประโยคสุดท้ายต้องเป็นคำถามเดียว

ห้าม:
- ห้ามถามหลายคำถาม
- ห้ามเหมือน checklist
- ห้ามข้ามไปถาม step อื่น
- ห้ามสรุปแทนผู้ใช้เกินกว่าที่เขาพูด
""".strip()

    user_prompt = f"""
คำตอบที่เก็บแล้ว:
{json.dumps(answers, ensure_ascii=False, indent=2)}

coaching memory:
{json.dumps(memory, ensure_ascii=False, indent=2)}

key:
{next_rule["key"]}

คำถามหลัก:
{next_rule["question"]}

goal ที่ต้องการได้:
{next_rule["goal"]}

แนวทางเฉพาะของ step นี้:
{next_rule.get("step_prompt_hint", "")}

ข้อมูลที่ต้องการ:
{", ".join(next_rule.get("required", []))}

ประเภทคำตอบ:
{next_rule.get("answer_type", "")}


คำสั่ง:
ช่วยตอบแบบโค้ชที่ต่อบทสนทนา ไม่ใช่ถามคำถามใหม่ทันที
""".strip()

    async for item in _stream_text_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.45,
    ):
        yield item


async def ask_phase_transition(
    state: ChatState,
    from_phase_title: str,
    to_phase_title: str,
    next_rule: dict,
):
    system_prompt = """
คุณคือ AI Coach ที่กำลังสนทนาแบบธรรมชาติ ไม่ใช่แบบสอบถาม

หน้าที่:
- เชื่อมจาก phase เดิมไป phase ใหม่อย่างลื่นไหล
- สะท้อนสิ่งที่ผู้ใช้เล่ามาสั้น ๆ โดยไม่สรุปเกินจริง
- ทำให้ผู้ใช้รู้สึกว่าบทสนทนายังต่อเนื่อง
- ถามคำถามแรกของ phase ใหม่ตามคำถามหลัก

โครงสร้างคำตอบ:
1. สะท้อนภาพรวมจาก phase ก่อนหน้าเล็กน้อย
2. เชื่อมเข้าสู่ phase ใหม่อย่างเป็นธรรมชาติ
3. ถามคำถามเดียวตามคำถามหลักของ next_rule

หลักการ:
- ใช้คำถามหลักเป็นแกน
- รักษาเป้าหมายของ step ใหม่
- ห้ามเปลี่ยนเจตนาของคำถาม
- ห้ามข้ามไปถาม step อื่น
- ฟังดูเป็นคนคุยกันจริง
- 2-4 ประโยค
- ประโยคสุดท้ายต้องเป็นคำถามเดียว

ห้าม:
- ห้ามถามหลายคำถาม
- ห้ามเหมือน checklist
- ห้ามใช้ bullet
- ห้ามสรุปแทนผู้ใช้เกินกว่าข้อมูลที่มี
- ห้ามใช้ภาษาทางการเกินไป เช่น "เข้าสู่กระบวนการถัดไป", "ดำเนินการใน phase ต่อไป"
""".strip()

    user_prompt = f"""
จาก phase:
{from_phase_title}

ไป phase:
{to_phase_title}

คำตอบที่เก็บแล้ว:
{json.dumps(state.answers, ensure_ascii=False, indent=2)}

coaching memory:
{json.dumps(state.coaching_memory.model_dump(), ensure_ascii=False, indent=2)}

key:
{next_rule["key"]}

คำถามหลักของ step ใหม่:
{next_rule["question"]}

goal ที่ต้องการได้:
{next_rule["goal"]}

ข้อมูลที่ต้องการ:
{", ".join(next_rule.get("required", []))}

ประเภทคำตอบ:
{next_rule.get("answer_type", "")}

คำสั่ง:
ช่วยตอบแบบโค้ชที่เชื่อมบทสนทนาจาก phase ก่อนหน้าไป phase ใหม่อย่างเป็นธรรมชาติ
ต้องลงท้ายด้วยคำถามเดียวตามคำถามหลักของ step ใหม่
""".strip()

    async for item in _stream_text_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.45,
    ):
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