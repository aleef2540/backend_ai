import json
from pydantic import BaseModel
from app.shared.ai.openai_client import call_openai_chat_stream_full, call_openai_chat 

AI_EXPERT_SYSTEM_PROMPT = """
คุณคือ อาจารย์ ปกรณ์ วงศ์รัตนพิบูลย์ ผู้เชี่ยวชาญด้านการพัฒนาบุคลากร และ Executive Coach มืออาชีพ

บทบาทของคุณ (AI Coach - Social Learning 20%):
- ทำหน้าที่เป็น Coach และเพื่อนคู่คิด (Companion) ที่ช่วยดึงศักยภาพของผู้ใช้ออกมา ไม่ใช่ผู้บรรยายทฤษฎี (10%)
- ผู้เรียนกลุ่มนี้ผ่านการทำระบบ Self-learning คอร์สเรียนออนไลน์ฝั่ง 10% ครบถ้วน 100% เต็มมาเรียบร้อยแล้ว
- เน้นกระบวนการ 2-Way Learning ชวนคิด สะท้อนมุมมองตนเอง เพื่อเปลี่ยนทฤษฎีในหัวให้กลายเป็นแผนลงมือทำงานจริง (70%)
- ใช้กรอบการโค้ชแบบ GROW Model (Goal, Reality, Options, Will) ในการดำเนินบทสนทนาอย่างเหนียวแน่นทีละขั้นตอน

กฎเหล็กในการตอบ (STRICT RULES):
1. ห้ามแจกคำตอบสำเร็จรูป หรือคิด Action Plan สรุปส่งให้ผู้ใช้ทันที หน้าที่ของคุณคือถามให้เขาพูดออกมาเอง
2. รักษาแนวทาง "ถามชวนคิด 80% / ให้ข้อมูลไกด์ไลน์ความรู้สั้นๆ 20%" 
3. ความยาวรวมห้ามเกิน 3-4 ประโยค: ให้เริ่มต้นด้วยการทวนความเข้าใจ ชื่นชม หรือสะท้อนกรอบคิดสั้นๆ 1-2 ประโยค และ *ต้องปิดท้ายด้วยคำถามปลายเปิดที่สอดคล้องกับพฤติกรรมใน State ปัจจุบันเพียง 1 คำถามเสมอ*
4. ห้ามแต่งข้อมูลหลักสูตรหรือข้อมูลภายในองค์กรเอง
5. ใช้ภาษาไทย สุภาพ เป็นธรรมชาติ และใช้คำว่า “ครับ” อย่างเหมาะสม
"""


async def evaluate_coaching_state_before_reply(
    current_state: str,
    last_bot_message: str,
    user_message: str,
    conversation_context: str = ""
) -> dict:
    """
    ฟังก์ชันวิเคราะห์ข้อความขาเข้า (Pre-Evaluation) 
    เพื่อตรวจสอบว่าข้อความที่พิมพ์เข้ามารอบนี้ สอบผ่านเกณฑ์ของสเตจปัจจุบันหรือยัง
    """
    state_order = ["INTRO", "TOPIC", "GOAL", "REALITY", "OPTIONS", "WILL", "COMPLETED"]
    
    try:
        current_index = state_order.index(current_state.upper())
        fallback_next_state = state_order[current_index + 1] if current_index < len(state_order) - 1 else "COMPLETED"
    except ValueError:
        fallback_next_state = "INTRO"

    evaluator_system_prompt = """
คุณคือระบบตรวจสอบขั้นตอนการโค้ช (Coaching State Evaluator) หลังบ้าน
หน้าที่ของคุณคืออ่านบทสนทนาและวิเคราะห์ข้อความล่าสุดของผู้ใช้ เทียบกับคำถามรอบล่าสุดของบอต เพื่อประเมินว่าผู้เรียนให้ข้อมูลเข้าเกณฑ์ขยับสเตจต่อไปแล้วหรือยัง

เกณฑ์การให้ผ่าน (is_state_achieved = true):
- INTRO: ผู้เรียนพิมพ์ตอบรับ ทักทาย หรือแสดงความพร้อมที่จะพูดคุยปรึกษาแผน
- GOAL: ผู้เรียนแชร์ปักหมุดเป้าหมายหน้างานจริง 1 เรื่อง หรือระบุปัญหาสำคัญในทีมที่อยากเคลียร์ในเซสชันนี้ได้แล้ว
- REALITY: ผู้เรียนยอมเปิดอกเล่าถึงสถานการณ์จริง อุปสรรค หรือจุดติด (Pain Points) ที่ทำให้เป้าหมายยังไม่สำเร็จ
- OPTIONS: ผู้เรียนระบุไอเดีย แผนคิด หรือทางเลือกของตนเองที่จะเอามาปรับแก้ปัญหาได้แล้วอย่างน้อย 1-2 แนวทาง
- WILL: ผู้เรียนยอมระบุเลือก Action Item ชัดเจน พร้อมรับปากลงเวลาสิ้นสุด (Deadline) ที่แน่นอนที่จะเริ่มทำ

กฎเหล็ก: อย่าตั้งเกณฑ์สูงเกินไป หากผู้เรียนให้ความร่วมมือดีตอบตรงประเด็นของสเตจปัจจุบัน ให้ปรับผ่าน (true) ทันที
นอกจากนี้ จงทำหน้าที่สกัดข้อมูลสาระสำคัญ (Extracted Data) ที่ผู้เรียนคุยในสเตจนั้นออกมาสั้นๆ 1 ประโยคด้วย
"""

    evaluator_user_prompt = f"""
[สเตจปัจจุบันที่ต้องตรวจสอบ: {current_state.upper()}]

ประวัติการคุยย่อย:
{conversation_context}

คำถามรอบที่แล้วของคุณ (โค้ช):
{last_bot_message}

ข้อความล่าสุดที่ผู้เรียนเพิ่งพิมพ์ตอบกลับมา:
{user_message}

กรุณาวิเคราะห์และส่งข้อความรูปแบบ JSON กลับมาตามโครงสร้างนี้เท่านั้น (ห้ามพ่น Markdown block หรือข้อความอื่น):
{{
  "evaluated_state": "{current_state.upper()}",
  "is_state_achieved": true หรือ false,
  "reason": "สรุปวิเคราะห์การผ่านไม่ผ่านสั้นๆ 1 ประโยค",
  "suggested_next_state": "ถ้าผ่านให้ระบุ '{fallback_next_state}' ถ้าไม่ผ่านให้ระบุ '{current_state.upper()}'",
  "extracted_data": "ข้อความข้อมูลที่ผู้เรียนระบุในรอบนี้สั้นๆ เช่น ชื่อเป้าหมาย หรือชื่อแผนงาน (ถ้าไม่มีข้อมูลใส่ null)"
}}
"""
    
    default_fallback = {
        "evaluated_state": current_state.upper(),
        "is_state_achieved": False,
        "reason": "Fallback triggered",
        "suggested_next_state": current_state.upper(),
        "extracted_data": None
    }

    try:
        eval_string_response = await call_openai_chat(
            model="gpt-4.1-mini", 
            system_prompt=evaluator_system_prompt,
            user_prompt=evaluator_user_prompt,
            temperature=0.0
        )
        
        if eval_string_response:
            raw_content = eval_string_response.strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content.split("```")[1].split("```")[0].strip()
                
            return json.loads(raw_content)
    except Exception as e:
        print(f"[ERROR] Pre-Evaluation Critical Failure: {e}")
        
    return default_fallback


async def reply_ai_expert_direct_stream(
    user_message: str,
    conversation_context: str = "",
    current_state: str = "INTRO",
    extracted_context: dict = None,
    model: str = "gpt-4.1-mini",
):
    state_instructions = {
        "INTRO": "ทักทายต้อนรับผู้เรียนอย่างอบอุ่น ชื่นชมที่เรียนฝั่งออนไลน์จบ 100% แล้วชวนเปิดใจนำทฤษฎีมาวางแผนสั้นๆ ด้วยความผ่อนคลาย",
        "TOPIC": "อยู่ในช่วงเริ่มต้นของการโค้ช ชวนผู้เรียนคุยอย่างเป็นธรรมชาติ พร้อมทั้งนำไปสู่การกำหนด ระบุหัวข้อที่เกี่ยวข้องกับการเรียนรู้",
        "GOAL": "ผู้เรียนอยู่ในสเตจตั้งเป้าหมาย หน้าที่ของคุณคือชวนคิดกระตุ้นให้เขาดึงเรื่องท้าทายหน้างานจริงขึ้นมาปักหมุด 1 เรื่องถ้วน",
        "REALITY": "ทวนเป้าหมายที่ผู้เรียนบอกในตอนแรกให้ชื่นใจ จากนั้นสับสวิตช์เข้าสู่การสำรวจความจริง ชวนคุยจี้เรื่องอุปสรรคหรือสภาวะที่เจอหน้างานปัจจุบัน",
        "OPTIONS": "ผู้เรียนเล่าจุดติดเรียบร้อย ตอนนี้หน้าที่ของคุณคือชวนเขาดึงเนื้อหาที่จำได้จากคอร์สเรียนฝั่ง 10% มาช่วยคิดแนวทางแก้ปัญหา (คุณสามารถแจกใบ้แนวคิดสั้นๆ ได้ไม่เกิน 20% เพื่อกระตุ้นสมองเขา)",
        "WILL": "ผู้เรียนได้ไอเดียเด็ดแล้ว หน้าที่สำคัญของคุณในรอบนี้คือ ปลุกยักษ์ให้เกิด Commitment ชวนสรุปแปลงไอเดียเป็น 1 Action Item สั้นๆ พร้อมระบุกำหนดเวลาเริ่มต้นลงมือทำที่ชัดเจน"
    }

    state_guide = state_instructions.get(current_state.upper(), state_instructions["GOAL"])
    context_pool_str = json.dumps(extracted_context or {}, ensure_ascii=False)

    user_prompt = f"""
[CURRENT COACHING STATE FOR THIS TURN: {current_state.upper()}]
แนวทางปฏิบัติของโค้ชรอบนี้: {state_guide}

คลังข้อมูลสรุปสะสมที่สกัดได้จากสเตจก่อนๆ (Context Pool):
{context_pool_str}

ประวัติพฤติกรรมการคุยที่ผ่านมา:
{conversation_context}

ข้อความล่าสุดจากผู้เรียน:
{user_message}

คำสั่งย้ำท้าย: 
สวมบทบาทเป็นอาจารย์ปกรณ์ พ่นประโยคทวนความต้องการสั้นๆ 1-2 ประโยค (ยึดหลักถาม 80/ให้ความรู้ 20) และขมวดปิดท้ายไอเดียด้วย "1 คำถามปลายเปิด" ให้ตรงสเตจปัจจุบันอย่างคมคาย
"""

    print(f"Executing Stream Generation with State: [{current_state.upper()}]")

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