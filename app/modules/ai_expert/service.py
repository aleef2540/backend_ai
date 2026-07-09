import json
from app.shared.ai.openai_client import call_openai_chat_stream_full, call_openai_chat


AI_EXPERT_SYSTEM_PROMPT = """
คุณคือ อาจารย์ ปกรณ์ วงศ์รัตนพิบูลย์ ผู้เชี่ยวชาญด้านการพัฒนาบุคลากร และ Executive Coach มืออาชีพ

บทบาทของคุณ (AI Coach - Social Learning 20%):
- ทำหน้าที่เป็น Coach และเพื่อนคู่คิด (Companion) ที่ช่วยดึงศักยภาพของผู้ใช้ออกมา ไม่ใช่ผู้บรรยายทฤษฎีฝั่ง 10%
- ผู้เรียนกลุ่มนี้ผ่านการทำระบบ Self-learning คอร์สเรียนออนไลน์ฝั่ง 10% ครบถ้วน 100% มาเรียบร้อยแล้ว
- เน้นกระบวนการ 2-Way Learning ชวนคิด สะท้อนมุมมองตนเอง เพื่อเปลี่ยนทฤษฎีในหัวให้กลายเป็นแผนลงมือทำงานจริง 70%
- ใช้กรอบการโค้ชแบบ GROW Model (Goal, Reality, Options, Will) อย่างยืดหยุ่น ไม่แข็งเป็นสคริปต์

หลักคิดสำคัญ:
- ห้ามถาม Goal ซ้ำ หากผู้เรียนระบุปัญหา เป้าหมาย หรือผลลัพธ์ที่ต้องการชัดเจนแล้ว
- ถ้าผู้เรียนให้ข้อมูลของหลายช่วงพร้อมกัน เช่น Goal และ Reality อยู่ในคำตอบเดียว ให้ยอมรับข้อมูลนั้นและเดินไปจุดที่ยังไม่ชัด
- ถ้าผู้เรียนตอบว่า “ไม่รู้”, “ไม่แน่ใจ”, “คิดไม่ออก” ห้ามถามคำถามเดิมซ้ำ ให้ช่วยเสนอกรอบคิดสั้นๆ 2-3 ทางเลือก แล้วถามให้เขาเลือกสิ่งที่ใกล้เคียงที่สุด
- อย่ารีบสรุป Action Plan ให้ผู้เรียนทันที ต้องช่วยให้ผู้เรียนคิดและเลือกด้วยตนเอง

Coaching Moves ที่ใช้ได้:
1. clarify_goal = ช่วยทำเป้าหมายให้ชัด
2. explore_reality = สำรวจสถานการณ์จริง อุปสรรค สาเหตุ คนที่เกี่ยวข้อง และผลกระทบ
3. challenge_assumption = ท้าทายสมมติฐานอย่างสุภาพ เช่น สิ่งที่คิดว่าเป็นสาเหตุ มีหลักฐานอะไรสนับสนุน
4. expand_options = ชวนคิดทางเลือกหลายวิธี โดยให้กรอบสั้นๆ ได้ไม่เกิน 20%
5. commit_action = แปลงทางเลือกเป็น Action ที่ชัด มีเวลาเริ่ม และวิธีวัดผล
6. summarize_commitment = สรุปแผนและยืนยัน commitment

กฎเรื่องน้ำเสียงให้เหมือน Coach จริง:
- ห้ามเริ่มคำตอบด้วยแพตเทิร์นเดิมซ้ำๆ เช่น “ผมเข้าใจว่า...”, “ผมเห็นว่า...”, “ผมเข้าใจครับว่าคุณ...” เกิน 1 ครั้งในบทสนทนาติดกัน
- ห้ามทวนประโยคผู้เรียนแบบ copy ความหมายเดิมทั้งประโยคโดยไม่เพิ่มคุณค่า
- ก่อนถามคำถาม ให้เลือกใช้ได้หลายรูปแบบ เช่น empathy, observation, reframe, summary สั้นๆ, challenge เบาๆ หรือถามตรงๆ ได้เลย
- ภาษาต้องเหมือนคนคุยกับคน ไม่ใช่แบบฟอร์มประเมินหรือ script สำเร็จรูป
- ใช้คำขึ้นต้นให้หลากหลาย เช่น “ประเด็นนี้น่าสนใจครับ”, “อันนี้เป็นจุดสำคัญเลยครับ”, “ถ้ามองจากที่เล่ามา...”, “งั้นเราลองแยกดูทีละชั้นครับ”, “ไม่เป็นไรครับ ลองเดาแบบคร่าวๆ ก็ได้ครับ”
- ไม่ต้องชมทุกครั้ง และไม่ต้องสะท้อนทุกครั้ง หากผู้เรียนตอบชัดแล้วให้ต่อยอดทันที

กฎเหล็กในการตอบ:
1. รักษาแนวทาง "ถามชวนคิด 80% / ให้ข้อมูลไกด์ไลน์สั้นๆ 20%"
2. ความยาวรวมไม่เกิน 3-5 ประโยค
3. เปิดคำตอบแบบเป็นธรรมชาติ อาจเป็น empathy, observation, reframe, summary สั้นๆ หรือถามตรงๆ ได้ ไม่บังคับต้องขึ้นต้นด้วย “ผมเข้าใจว่า/ผมเห็นว่า”
4. ปิดท้ายด้วยคำถามปลายเปิดเพียง 1 คำถามเสมอ ยกเว้น state SUMMARY/COMPLETED ที่สามารถสรุปและให้กำลังใจได้
5. ห้ามแต่งข้อมูลหลักสูตรหรือข้อมูลภายในองค์กรเอง
6. ใช้ภาษาไทย สุภาพ เป็นธรรมชาติ และใช้คำว่า “ครับ” อย่างเหมาะสม
"""


STATE_ORDER = ["OPENING", "GOAL", "REALITY", "OPTIONS", "WILL", "SUMMARY", "COMPLETED"]


def _safe_json_loads(raw_content: str) -> dict:
    raw_content = (raw_content or "").strip()

    if raw_content.startswith("```json"):
        raw_content = raw_content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif raw_content.startswith("```"):
        raw_content = raw_content.split("```", 1)[1].split("```", 1)[0].strip()

    return json.loads(raw_content)


def _normalize_score(value, default=0):
    try:
        value = int(value)
        return max(0, min(100, value))
    except Exception:
        return default


def _normalize_options_count(value, default=0):
    try:
        return max(0, int(value))
    except Exception:
        return default


async def evaluate_coaching_state_before_reply(
    current_state: str,
    last_bot_message: str,
    user_message: str,
    conversation_context: str = "",
    extracted_context: dict | None = None,
    turn_count: int = 0,
) -> dict:
    """
    วิเคราะห์ข้อความล่าสุดแบบ Coach จริงขึ้น:
    - ไม่ดูแค่ผ่าน/ไม่ผ่านของ state ปัจจุบัน
    - ประเมิน clarity ของ Goal / Reality / Options / Will พร้อมกัน
    - เลือก recommended_next_state และ coach_move ที่เหมาะสม
    """
    current_state = (current_state or "OPENING").upper()
    if current_state == "INTRO":
        current_state = "OPENING"
    if current_state == "TOPIC":
        current_state = "GOAL"

    context_pool_str = json.dumps(extracted_context or {}, ensure_ascii=False)

    evaluator_system_prompt = """
คุณคือระบบประเมินขั้นตอนการโค้ช GROW หลังบ้าน
หน้าที่ของคุณคืออ่านบทสนทนาและข้อความล่าสุดของผู้เรียน เพื่อประเมินว่าข้อมูลตอนนี้ชัดพอในแต่ละมิติหรือยัง

คุณต้องประเมินแบบ Coach จริง ไม่ใช่บังคับเดิน state แบบเส้นตรง
หากผู้เรียนให้ข้อมูล Goal และ Reality มาพร้อมกัน ให้ให้คะแนนทั้งสองส่วน และแนะนำให้ไปถามส่วนที่ยังไม่ชัด

เกณฑ์คะแนนโดยประมาณ:
- goal_clarity 0-100:
  0 = ยังไม่รู้เรื่องที่จะคุย
  40 = มีประเด็นกว้างๆ
  70 = มีเป้าหมายหรือปัญหาหลักชัด
  90 = มีผลลัพธ์ที่อยากเห็นหรือ success indicator

- reality_clarity 0-100:
  0 = ยังไม่รู้สถานการณ์จริง
  40 = รู้ปัญหาเบื้องต้น
  70 = รู้สถานการณ์/สาเหตุ/บทบาทผู้ใช้บางส่วน
  90 = รู้สถานการณ์ คนเกี่ยวข้อง สาเหตุ ผลกระทบ และข้อจำกัดชัด

- options_count:
  จำนวนทางเลือกที่ผู้เรียนเป็นคนคิดเองหรือยอมรับว่าน่าลอง

- commitment_clarity 0-100:
  0 = ยังไม่มี action
  40 = มี action กว้างๆ
  70 = มี action ชัด
  90 = มี action + deadline/เวลาเริ่ม + วิธีวัดผล

การเลือก recommended_next_state:
- ถ้ายังทักทาย/ยังไม่บอกประเด็น ให้ OPENING หรือ GOAL
- ถ้า goal_clarity < 70 ให้ GOAL
- ถ้า reality_clarity < 70 ให้ REALITY
- ถ้า options_count < 2 ให้ OPTIONS ยกเว้นผู้เรียนเลือกทางเดียวชัดมากแล้วให้ไป WILL ได้
- ถ้า commitment_clarity < 90 ให้ WILL
- ถ้ามี action + deadline + success measure ชัด ให้ SUMMARY

การเลือก recommended_coach_move:
- GOAL ใช้ clarify_goal
- REALITY ใช้ explore_reality หรือ challenge_assumption
- OPTIONS ใช้ expand_options
- WILL ใช้ commit_action
- SUMMARY ใช้ summarize_commitment

ถ้าผู้เรียนตอบว่า ไม่รู้ / ไม่แน่ใจ / คิดไม่ออก:
- อย่าบอกให้ถามคำถามเดิมซ้ำ
- ให้ recommended_coach_move เป็น expand_options หรือ explore_reality ตาม state
- ใส่ needs_scaffold = true

ตอบกลับเป็น JSON เท่านั้น ห้ามมี Markdown
"""

    evaluator_user_prompt = f"""
[CURRENT_STATE]
{current_state}

[TURN_COUNT]
{turn_count}

[CONTEXT_POOL]
{context_pool_str}

[CONVERSATION_CONTEXT]
{conversation_context}

[LAST_BOT_MESSAGE]
{last_bot_message}

[LATEST_USER_MESSAGE]
{user_message}

กรุณาวิเคราะห์และส่ง JSON ตามโครงสร้างนี้เท่านั้น:
{{
  "goal_clarity": 0,
  "reality_clarity": 0,
  "options_count": 0,
  "commitment_clarity": 0,
  "recommended_next_state": "GOAL",
  "recommended_coach_move": "clarify_goal",
  "last_question_type": "goal|reality|options|will|summary|other",
  "needs_scaffold": false,
  "reason": "เหตุผลสั้นๆ",
  "extracted_patch": {{
    "topic": null,
    "goal": {{
      "raw": null,
      "desired_outcome": null,
      "success_indicator": null
    }},
    "reality": {{
      "current_situation": null,
      "root_causes": [],
      "people_involved": [],
      "constraints": [],
      "user_assumptions": [],
      "impact": null
    }},
    "options": {{
      "ideas": [],
      "selected_option": null
    }},
    "will": {{
      "action": null,
      "deadline": null,
      "first_step": null,
      "success_measure": null,
      "risk": null
    }},
    "summary": null
  }}
}}
"""

    fallback = {
        "goal_clarity": 0,
        "reality_clarity": 0,
        "options_count": 0,
        "commitment_clarity": 0,
        "recommended_next_state": current_state,
        "recommended_coach_move": "explore_reality" if current_state == "REALITY" else "clarify_goal",
        "last_question_type": "other",
        "needs_scaffold": False,
        "reason": "Fallback triggered",
        "extracted_patch": {},
    }

    try:
        eval_string_response = await call_openai_chat(
            model="gpt-4.1-mini",
            system_prompt=evaluator_system_prompt,
            user_prompt=evaluator_user_prompt,
            temperature=0.0,
        )

        if eval_string_response:
            parsed = _safe_json_loads(eval_string_response)

            next_state = str(parsed.get("recommended_next_state") or current_state).upper()
            if next_state == "INTRO":
                next_state = "OPENING"
            if next_state == "TOPIC":
                next_state = "GOAL"
            if next_state not in STATE_ORDER:
                next_state = current_state

            parsed["goal_clarity"] = _normalize_score(parsed.get("goal_clarity"), 0)
            parsed["reality_clarity"] = _normalize_score(parsed.get("reality_clarity"), 0)
            parsed["options_count"] = _normalize_options_count(parsed.get("options_count"), 0)
            parsed["commitment_clarity"] = _normalize_score(parsed.get("commitment_clarity"), 0)
            parsed["recommended_next_state"] = next_state
            parsed["recommended_coach_move"] = str(parsed.get("recommended_coach_move") or "clarify_goal")
            parsed["last_question_type"] = str(parsed.get("last_question_type") or "other")
            parsed["needs_scaffold"] = bool(parsed.get("needs_scaffold", False))
            parsed["extracted_patch"] = parsed.get("extracted_patch") or {}
            return parsed

    except Exception as e:
        print(f"[ERROR] Coaching Evaluation Failure: {e}")

    return fallback


def merge_extracted_context(base: dict, patch: dict) -> dict:
    """
    Merge แบบไม่ลบข้อมูลเดิม:
    - string ใหม่ที่ไม่ว่างจะทับของเดิม
    - list จะ append แบบไม่ซ้ำ
    - dict จะ merge recursive
    """
    if not isinstance(base, dict):
        base = {}
    if not isinstance(patch, dict):
        return base

    for key, value in patch.items():
        if value is None or value == "":
            continue

        if isinstance(value, dict):
            old = base.get(key)
            if not isinstance(old, dict):
                old = {}
            base[key] = merge_extracted_context(old, value)

        elif isinstance(value, list):
            old = base.get(key)
            if not isinstance(old, list):
                old = []
            for item in value:
                if item not in [None, ""] and item not in old:
                    old.append(item)
            base[key] = old

        else:
            base[key] = value

    return base


async def reply_ai_expert_direct_stream(
    user_message: str,
    conversation_context: str = "",
    current_state: str = "OPENING",
    extracted_context: dict | None = None,
    coach_move: str | None = None,
    goal_clarity: int = 0,
    reality_clarity: int = 0,
    options_count: int = 0,
    commitment_clarity: int = 0,
    needs_scaffold: bool = False,
    repeated_question_count: int = 0,
    model: str = "gpt-4.1-mini",
):
    current_state = (current_state or "OPENING").upper()
    if current_state == "INTRO":
        current_state = "OPENING"
    if current_state == "TOPIC":
        current_state = "GOAL"

    state_instructions = {
        "OPENING": "ต้อนรับอย่างอบอุ่น แล้วชวนผู้เรียนเล่าเรื่องหน้างานที่อยากปรึกษาแบบไม่กดดัน",
        "GOAL": "ช่วยจับเป้าหมายให้ชัด ถ้าผู้เรียนบอกปัญหาชัดแล้ว ให้ถามผลลัพธ์ที่อยากเห็นหรือภาพความสำเร็จ ไม่ถาม Goal ซ้ำแบบเดิม",
        "REALITY": "สำรวจสถานการณ์จริงให้ลึกขึ้น เช่น เกิดกับใคร งานแบบไหน สาเหตุที่เป็นไปได้ บทบาทของผู้เรียน ผลกระทบ และข้อจำกัด",
        "OPTIONS": "ชวนคิดทางเลือกหลายวิธี ก่อนเลือก action ห้ามรีบสรุปให้ หากผู้เรียนคิดไม่ออก ให้เสนอกรอบคิดสั้นๆ 2-3 ทางเลือกแล้วให้เขาเลือก",
        "WILL": "ช่วยแปลงทางเลือกเป็น Action ที่ชัดเจน ต้องมีสิ่งที่จะทำ เวลาเริ่ม/กำหนดส่ง และวิธีวัดผล",
        "SUMMARY": "สรุปสิ่งที่ผู้เรียนเลือกทำ เป้าหมาย เหตุผล Action เวลาเริ่ม และวิธีวัดผล แบบกระชับ พร้อมยืนยัน commitment",
        "COMPLETED": "ปิดบทสนทนาด้วยการให้กำลังใจสั้นๆ และสะท้อนว่าผู้เรียนมีแผนไปทดลองใช้จริงแล้ว",
    }

    coach_move_instructions = {
        "clarify_goal": "ใช้คำถามเพื่อทำเป้าหมายให้ชัดขึ้น โดยไม่ถามซ้ำถ้าเป้าหมายชัดแล้ว",
        "explore_reality": "ใช้คำถามเพื่อสำรวจข้อเท็จจริง สาเหตุ คนที่เกี่ยวข้อง และผลกระทบ",
        "challenge_assumption": "ท้าทายสมมติฐานอย่างสุภาพ เช่น ชวนดูหลักฐานหรือมุมมองอื่น",
        "expand_options": "ช่วยให้ผู้เรียนเห็นทางเลือกมากกว่า 1 ทาง โดยให้กรอบคิดสั้นๆ ได้",
        "commit_action": "ถามให้ได้ Action ที่ทำจริง เวลาเริ่ม และวิธีวัดผล",
        "summarize_commitment": "สรุป commitment และตรวจความพร้อมในการลงมือทำ",
    }

    state_guide = state_instructions.get(current_state, state_instructions["GOAL"])
    move_guide = coach_move_instructions.get(coach_move or "", "เลือกวิธีโค้ชที่เหมาะกับข้อมูลที่ยังไม่ชัด")
    context_pool_str = json.dumps(extracted_context or {}, ensure_ascii=False)

    scaffold_instruction = ""
    if needs_scaffold:
        scaffold_instruction = """
ผู้เรียนมีแนวโน้มยังคิดไม่ออกหรือไม่แน่ใจ รอบนี้ห้ามถามคำถามเดิมซ้ำ
ให้ช่วยเสนอกรอบคิดสั้นๆ 2-3 ทางเลือก แล้วถามให้เขาเลือกข้อที่ใกล้เคียงที่สุด
"""

    repeated_instruction = ""
    if repeated_question_count >= 1:
        repeated_instruction = """
ตรวจพบความเสี่ยงว่าคำถามอาจซ้ำกับรอบก่อน รอบนี้ต้องเปลี่ยนมุมถาม หรือสรุปสิ่งที่รู้แล้วแล้วถามเฉพาะช่องว่างที่ยังไม่ชัด
"""

    user_prompt = f"""
[CURRENT_COACHING_STATE]
{current_state}

[RECOMMENDED_COACH_MOVE]
{coach_move}

[STATE_GUIDE]
{state_guide}

[COACH_MOVE_GUIDE]
{move_guide}

[CLARITY_SCORES]
Goal clarity: {goal_clarity}/100
Reality clarity: {reality_clarity}/100
Options count: {options_count}
Commitment clarity: {commitment_clarity}/100

[CONTEXT_POOL]
{context_pool_str}

[CONVERSATION_CONTEXT]
{conversation_context}

[LATEST_USER_MESSAGE]
{user_message}

[SPECIAL_INSTRUCTION]
{scaffold_instruction}
{repeated_instruction}

คำสั่งย้ำท้าย:
ตอบเป็นภาษาไทยแบบอาจารย์ปกรณ์ สุภาพ เป็นธรรมชาติ กระชับ ไม่เกิน 3-5 ประโยค
ห้ามขึ้นต้นด้วย “ผมเข้าใจว่า”, “ผมเห็นว่า”, “ผมเข้าใจครับว่าคุณ” หากบทสนทนาก่อนหน้าใช้ไปแล้วหรือเสี่ยงซ้ำ
เปิดคำตอบให้หลากหลาย: อาจใช้ empathy, observation, reframe, summary สั้นๆ หรือถามตรงๆ ได้เลย
อย่าทวนสิ่งที่ผู้เรียนเพิ่งพูดแบบซ้ำความเดิม ให้ต่อยอดจากข้อมูลนั้นทันที
ปิดท้ายด้วยคำถามปลายเปิด 1 คำถามที่ตรงกับ state และ coach_move ปัจจุบัน
ถ้า state เป็น SUMMARY ให้สรุปแผนก่อน แล้วถามยืนยันความพร้อมหรือสิ่งที่ต้องระวังเพียง 1 คำถาม
"""

    print(f"Executing Stream Generation with State: [{current_state}] Move: [{coach_move}]")

    async for item in call_openai_chat_stream_full(
        model=model,
        system_prompt=AI_EXPERT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.4,
    ):
        if item.get("type") == "chunk":
            text = item.get("text", "")
            if text:
                yield {
                    "type": "chunk",
                    "text": text,
                }
        elif item.get("type") == "done":
            yield {
                "type": "done",
                "content": item.get("content")
            }
            return
