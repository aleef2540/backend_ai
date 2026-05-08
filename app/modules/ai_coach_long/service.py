from dotenv import load_dotenv
load_dotenv()

import json
import re
import requests
from openai import OpenAI
from qdrant_client import QdrantClient
from app.modules.ai_coach.schema import IntentResult

client = OpenAI()

from app.shared.ai.openai_client import call_openai_chat_full, call_openai_chat_stream_full

STATUS_ACTION_MAP = {
    "off_topic": "retry_same_step",
    "too_short": "retry_same_step",
    "partial": "probe_same_step",
    "reflecting": "probe_same_step",
    "clear_but_needs_guidance": "probe_same_step",
    "clear_complete": "next_step",
}

def clean_json(text: str) -> str:
    text = re.sub(r"```json|```", "", text)
    return text.strip()


def safe_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None
    

async def _stream_text_response(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
):
    """
    helper กลางสำหรับ stream ข้อความธรรมดา
    คืน event รูปแบบ:
    - {"type":"chunk","text":"..."}
    - {"type":"done","content":"...","usage":...,"cost":...}
    """
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
                yield {
                    "type": "chunk",
                    "text": text,
                }

        elif item.get("type") == "done":
            content = (item.get("content") or final_content).strip()
            yield {
                "type": "done",
                "content": content,
                "usage": item.get("usage"),
                "cost": item.get("cost"),
            }
            return


# ---------------------------------------------------
# NON-STREAM VERSION (เผื่อยังมีจุดอื่นเรียกใช้อยู่)
# ---------------------------------------------------
async def detect_intent(user_message: str) -> IntentResult:

    system_prompt = """คุณคือ AI ที่มีหน้าที่จำแนก intent ของข้อความผู้ใช้

        ให้เลือกเพียง 1 ค่าเท่านั้น:
        - greeting = ข้อความทักทายสั้น ๆ เช่น สวัสดี หวัดดี hello hi
        - general = พูดคุยทั่วไป ระบายความรู้สึก ยังไม่มีเป้าหมายพัฒนา
        - learning = ผู้ใช้มีปัญหา อยากพัฒนา อยากเรียนรู้ หรือขอคำปรึกษา

        กฎ:
        - ถ้ามีคำว่า "อยากพัฒนา", "มีปัญหา", "ไม่รู้จะทำยังไง", "ควรทำอย่างไร" → learning
        - ถ้าเป็นแค่ทักทาย → greeting
        - ที่เหลือ → general

        ตอบเป็น JSON เท่านั้น:

        {
        "intent": "greeting|general|learning"
        }
        """

    result = await call_openai_chat_full(
    model="gpt-4.1-mini",
    system_prompt=system_prompt,
    user_prompt=user_message,
    temperature=0.1,
    )

    text = result["content"] or ""
    text = clean_json(text)

    try:
        data = json.loads(text)
        intent = data.get("intent", "general")
    except:
        intent = "general"

    if intent not in ["greeting", "general", "learning"]:
        intent = "general"

    return IntentResult(intent=intent)

async def generate_opening_ai_coach_question(
    fixed_question: str,
    model: str = "gpt-4.1-mini",
) -> str:
    system_prompt = """คุณคือ AI Coach ที่กำลังเริ่มต้นบทสนทนากับผู้ใช้

บทบาท:
- คุณคือโค้ชที่ช่วยให้ผู้ใช้ “มองเห็นตัวเองได้ชัดขึ้น”
- การสื่อสารต้องให้ความรู้สึกเป็นมิตร อบอุ่น และเป็นธรรมชาติ
- ไม่ใช่การซักถาม แต่เป็นการชวนคุย

หน้าที่:
- เริ่มต้นด้วยการทักทายและแนะนำตัวสั้น ๆ
- บอก purpose ของการคุย เช่น ช่วยให้มองเห็นตัวเอง/เป้าหมายชัดขึ้น
- จากนั้น “ค่อย ๆ เชื่อม” ไปสู่คำถามหลักที่ระบบกำหนด

หลักการสำคัญ:
- ต้องทำให้เหมือนบทสนทนา ไม่ใช่แบบสอบถาม
- ห้ามยิงคำถามทันทีโดยไม่มีบริบท
- ต้องมีการเกริ่นก่อนถาม
- ต้องใช้คำถามเพียง 1 คำถามเท่านั้น
- ต้องคงความหมายของคำถามหลักไว้

ลักษณะภาษา:
- เป็นกันเอง ไม่ทางการเกินไป
- ไม่แข็ง ไม่เป็นหุ่นยนต์
- อ่านแล้วรู้สึกเหมือนมีคนคุยด้วยจริง
- ใช้ภาษาไทย

รูปแบบ:
- ตอบเป็นย่อหน้าเดียว (Single paragraph)
- ไม่ขึ้นบรรทัดใหม่
- ความยาวประมาณ 2-4 ประโยค
- ประโยคสุดท้ายต้องเป็นคำถาม

ข้อห้าม:
- ห้ามถามหลายคำถาม
- ห้ามเปลี่ยนประเด็นของคำถามหลัก
- ห้ามอธิบายยืดยาว
- ห้ามใช้ bullet หรือ markdown

รูปแบบคำตอบ:
- ตอบเฉพาะข้อความที่ใช้แสดงกับผู้ใช้"""

    user_prompt = f"""คำถามหลัก:
{fixed_question}

ช่วยปรับคำถามนี้ให้เป็นคำถามเปิดบทสนทนาแบบโค้ช"""

    result = await call_openai_chat_full(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.4,
    )

    content = (result.get("content") or "").strip()

    if not content:
        content = fixed_question.strip()

    return content


async def evaluate_user_answer(
    question: str,
    user_answer: str,
    model: str = "gpt-4.1-mini",
) -> dict:
    question = (question or "").strip()
    user_answer = (user_answer or "").strip()

    if not user_answer:
        return {
            "ok": True,
            "status": "too_short",
            "is_on_topic": False,
            "is_sufficient": False,
            "needs_guidance": False,
            "reason": "ผู้ใช้ยังไม่ได้ตอบคำถาม",
            "confidence": 1.0,
            "raw": "",
        }

    short_direct = {
        "ครับ", "ค่ะ", "คับ", "จ้า", "อือ", "อืม", "อืมม", "อืมมม",
        "โอเค", "ok", "yes", "ใช่", "ไม่", "ไม่รู้", "ไม่แน่ใจ"
    }
    if user_answer.lower() in short_direct:
        return {
            "ok": True,
            "status": "too_short",
            "is_on_topic": False,
            "is_sufficient": False,
            "needs_guidance": False,
            "reason": "คำตอบสั้นเกินไปและยังไม่มีสาระพอใช้ต่อ",
            "confidence": 0.98,
            "raw": "",
        }

    reflecting_markers = [
        "บอกยาก", "ตอบยาก", "คิดยาก", "ขอคิดก่อน", "ยังไม่เคยคิด",
        "ไม่เคยมองแบบนี้", "น่าสนใจ", "ยากเหมือนกัน", "ยังตอบไม่ถูก",
        "อธิบายยาก", "พูดไม่ถูก", "ไม่แน่ใจว่ารู้สึกยังไง"
    ]
    if any(marker in user_answer for marker in reflecting_markers):
        return {
            "ok": True,
            "status": "reflecting",
            "is_on_topic": True,
            "is_sufficient": False,
            "needs_guidance": True,
            "reason": "ผู้ใช้ไม่ได้หลุดประเด็น แต่กำลังคิดหาคำตอบหรือยังตอบได้ไม่ชัด",
            "confidence": 0.9,
            "raw": "",
        }

    system_prompt = f"""
คุณคือระบบ AI ตรวจสอบความสอดคล้องเชิงความหมาย (Semantic Consistency) ระหว่าง "คำถามของ AI Coach" และ "คำตอบของผู้ใช้"

หน้าที่หลัก:
- ประเมินว่าคำตอบของผู้ใช้ "ตอบตรงคำถาม" หรือไม่
- พิจารณาตาม "ความหมายของคำถาม" เป็นหลัก
- เป้าหมายไม่ใช่จับผิด แต่ดูว่า "คำตอบนี้เพียงพอให้โค้ชไปต่อได้หรือยัง"

บริบท:
- AI Coach อาจถามได้หลายแบบ เช่น ถามความรู้สึก, ถามเป้าหมาย, ถามปัญหา, ถามการสะท้อนคิด
- ต้องตัดสินให้สอดคล้องกับ "สิ่งที่คำถามนั้นต้องการจริง ๆ"
- ห้ามใช้เกณฑ์เดียวกันกับทุกคำถาม

สถานะที่อนุญาตมี 6 แบบเท่านั้น:
1. off_topic
2. too_short
3. partial
4. reflecting
5. clear_but_needs_guidance
6. clear_complete

คำอธิบายสถานะ:
- off_topic = ไม่ตอบสิ่งที่ถาม, เปลี่ยนเรื่อง, หรือพูดสิ่งที่ไม่เกี่ยวข้อง
- too_short = สั้นเกินไปจนไม่สื่อสารสาระ เช่น "ครับ", "ใช่", "โอเค", "ไม่รู้", "อะไรก็ได้"
- partial = ตอบถูกทาง แต่ยังขาดสาระสำคัญที่คำถามนั้นต้องการ ทำให้โค้ชยังไปต่อได้ไม่เต็มที่
- reflecting = ผู้ใช้ยังไม่ตอบเนื้อหาเต็ม แต่กำลังคิด, ลังเล, บอกไม่แน่ใจ, หรือกำลังพยายามหาคำตอบในประเด็นที่ถาม
- clear_but_needs_guidance = ตอบตรงคำถามและมีสาระพอแล้ว แต่เนื้อหาสะท้อนว่าผู้ใช้ยังต้องการการช่วยคิดต่อ การคลี่ประเด็น หรือแนวทางเพิ่มเติม
- clear_complete = ตอบตรงคำถาม ชัดเจน และเพียงพอสำหรับไปขั้นถัดไปโดยไม่ต้องถามซ้ำในประเด็นเดิม

หลักการสำคัญ:
1. ให้ดูว่า "คำถามต้องการอะไร" ก่อน แล้วค่อยตัดสินว่าคำตอบพอหรือยัง
2. ถ้าคำถามถามเรื่อง "ความรู้สึก" และคำตอบระบุความรู้สึกได้ชัดเจน ให้ถือว่าเพียงพอได้ แม้ไม่มีรายละเอียดอื่นเพิ่ม
3. ถ้าคำถามถามเรื่อง "เป้าหมาย / ปัญหา / สถานการณ์ / เหตุผล" คำตอบต้องมีสาระตามนั้น จึงจะถือว่าเพียงพอ
4. ห้ามตัดสินว่า partial เพียงเพราะคำตอบสั้น ถ้าคำตอบนั้นตอบตรงสิ่งที่คำถามต้องการแล้ว
5. ถ้าคำตอบมีเพียงท่าทีลอย ๆ ที่ยังไม่ตอบสาระจริง เช่น "ยากจัง", "ไม่แน่ใจ", "ขอคิดก่อน" ให้เป็น reflecting
6. ถ้าคำตอบตรงคำถามและมีสาระพอแล้ว แต่ยังสะท้อนความติดขัด ความกังวล ความสับสน หรือความต้องการให้โค้ชช่วยต่อ ให้เป็น clear_but_needs_guidance
7. ถ้าคำตอบตรงคำถามและเพียงพอโดยตัวมันเอง ให้เป็น clear_complete

แนวทางตีความตามชนิดคำถาม:
- ถามความรู้สึก: ต้องการ "อารมณ์ / ความรู้สึก / ท่าที" เป็นหลัก
  ตัวอย่าง:
  ถาม: "คุณรู้สึกยังไงกับเป้าหมายในตอนนี้"
  ตอบ: "เหนื่อยมากๆเลย"
  => ถือว่าตอบตรงและเพียงพอ สามารถเป็น clear_complete หรือ clear_but_needs_guidance ได้

- ถามเป้าหมาย:
  ต้องการรู้ว่าเป้าหมายคืออะไร
  ตอบแค่ "กังวล" ยังไม่พอ => partial

- ถามปัญหา / อุปสรรค:
  ต้องการรู้ว่าติดอะไร เจออะไร
  ตอบแค่ "เหนื่อย" ยังไม่พอ => partial หรือ clear_but_needs_guidance ตามบริบท

- ถามสะท้อนคิด:
  ถ้าผู้ใช้กำลังพยายามทบทวนตนเอง เช่น "น่าจะเพราะ...", "อาจเป็นเพราะ..." อาจถือว่า reflecting หรือ partial ได้ตามความชัดเจน

กฎการตัดสิน:
- is_on_topic = true เมื่อคำตอบยังอยู่ในประเด็นที่คำถามถาม
- is_sufficient = true เมื่อคำตอบเพียงพอให้โค้ชไปต่อได้โดยไม่ต้องถามย้ำในประเด็นเดิม
- needs_guidance = true เมื่อคำตอบสะท้อนว่าผู้ใช้ยังต้องการการช่วยคิดต่อ คลี่ประเด็น หรือแนวทางเพิ่มเติม

การตีความตามประเภทคำถาม (สำคัญมาก):

- question_type = emotional
  คำถามที่ต้องการ "ความรู้สึก"
  → ถ้าคำตอบมีความรู้สึกชัด ถือว่าเพียงพอได้
  → ถ้าเป็น state เบา เช่น ง่วง เฉยๆ ให้เป็น clear_but_needs_guidance

- question_type = goal
  ต้องมี "เป้าหมาย" ชัดเจน
  → ถ้าไม่มี ถือว่า partial

- question_type = problem
  ต้องมี "ปัญหา/อุปสรรค"
  → ถ้าไม่มี ถือว่า partial

- question_type = reflection
  ถ้าผู้ใช้กำลังคิด/ลังเล → reflecting

ข้อบังคับเด็ดขาด:
1. ตอบเป็น JSON object เท่านั้น
2. ห้ามมีข้อความอื่นนอกเหนือจาก JSON
3. status ต้องเป็นหนึ่งใน 6 ค่านี้เท่านั้น:
   off_topic, too_short, partial, reflecting, clear_but_needs_guidance, clear_complete
4. confidence เป็นเลข 0.0 ถึง 1.0
5. reason ต้องสั้น กระชับ และอธิบายเหตุผลตามความหมายของคำถาม
6. ห้ามตัดสินจากความยาวของคำตอบเพียงอย่างเดียว
7. ต้องตีความโดยอิง "สิ่งที่คำถามถาม" เป็นหลัก

รูปแบบผลลัพธ์:
{{
  "status": "off_topic|too_short|partial|reflecting|clear_but_needs_guidance|clear_complete",
  "is_on_topic": true,
  "is_sufficient": true,
  "needs_guidance": false,
  "reason": "อธิบายสั้น ๆ",
  "confidence": 0.0
}}
"""

    user_prompt = f"""
คำถามล่าสุดของ AI Coach: {question}
คำตอบของผู้ใช้: {user_answer}

ช่วยประเมินว่าคำตอบนี้สอดคล้องกับคำถามหรือไม่ โดยใช้ความหมายของคำถามเป็นบริบทหลัก
"""

    try:
        result = await call_openai_chat_full(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        text = (result.get("content") or "").strip()
        text = re.sub(r"```json|```", "", text).strip()

        decoded = json.loads(text) if text else {}

        allowed_status = {
            "off_topic",
            "too_short",
            "partial",
            "reflecting",
            "clear_but_needs_guidance",
            "clear_complete",
        }

        status = str(decoded.get("status", "off_topic")).strip()
        if status not in allowed_status:
            status = "off_topic"

        return {
            "ok": True,
            "status": status,
            "is_on_topic": bool(decoded.get("is_on_topic", False)),
            "is_sufficient": bool(decoded.get("is_sufficient", False)),
            "needs_guidance": bool(decoded.get("needs_guidance", False)),
            "reason": str(decoded.get("reason", "")).strip(),
            "confidence": float(decoded.get("confidence", 0.0)),
            "raw": text,
        }

    except Exception as e:
        print("DEBUG ERROR (evaluate_user_answer):", repr(e))
        return {
            "ok": False,
            "status": "off_topic",
            "is_on_topic": False,
            "is_sufficient": False,
            "needs_guidance": False,
            "reason": f"parse_error: {str(e)}",
            "confidence": 0.0,
            "raw": "",
        }


async def generate_retry_same_step_question(
    fixed_question: str,
    user_answer: str,
    status: str,
    model: str = "gpt-4.1-mini",
) -> str:
    system_prompt = """คุณคือ AI Coach

หน้าที่:
- ผู้ใช้ยังตอบคำถามเดิมไม่ตรงพอ หรือสั้นเกินไป
- ช่วยพาผู้ใช้กลับมาที่คำถามเดิมอย่างนุ่มนวลและเป็นธรรมชาติ
- ก่อนถามใหม่ ให้สะท้อนหรือรับรู้สิ่งที่ผู้ใช้พูดสั้น ๆ ก่อน 1 ช่วง
- จากนั้นค่อยเชื่อมกลับไปยังคำถามหลักที่ต้องถาม

หลักการ:
- ต้องยังยึดคำถามหลักเดิมเป็นแกน
- ต้องสะท้อนคำตอบของผู้ใช้เล็กน้อย เช่น รับรู้ความรู้สึก สภาวะ หรือท่าทีที่ผู้ใช้พูด
- แล้วค่อยถามกลับไปยังคำถามหลักอย่างเป็นธรรมชาติ
- ถามเพียง 1 คำถามเท่านั้น
- ภาษาไทย เป็นกันเอง อบอุ่น
- ความยาว 1-2 ประโยคสั้น

แนวทางสำคัญ:
- ถ้าผู้ใช้ตอบสั้นหรือหลุดประเด็น ห้ามพาไปเปิดประเด็นใหม่ตามคำตอบนั้น
- ให้ใช้คำตอบของผู้ใช้เพียงเพื่อ "สะท้อน" ไม่ใช่เปลี่ยนหัวข้อสนทนา
- หลังสะท้อนแล้วต้องกลับมาที่คำถามหลักเดิมเสมอ

ตัวอย่าง:
คำถามหลัก: "คุณรู้สึกกับเป้าหมายอย่างไร?"
ผู้ใช้ตอบ: "ง่วงมากๆ"

คำตอบที่ดี:
"ฟังดูเหมือนตอนนี้คุณอาจจะล้าอยู่พอสมควรเลยนะครับ ถ้าลองกลับมาที่เป้าหมายนี้ คุณรู้สึกกับมันอย่างไรบ้างครับ"

คำตอบที่ไม่ดี:
"มีอะไรที่ทำให้คุณรู้สึกเหนื่อยหรือง่วงในช่วงนี้ไหม?"
เพราะเป็นการเปลี่ยนประเด็นจากคำถามหลัก

ข้อห้าม:
- ห้ามถามหลายคำถาม
- ห้ามเปลี่ยนประเด็น
- ห้ามเปิดหัวข้อใหม่จากคำตอบของผู้ใช้
- ห้ามกดดันหรือฟังดูเหมือนจับผิด

รูปแบบ:
- ตอบเป็นข้อความที่พร้อมส่งให้ผู้ใช้ได้ทันที
- ไม่ใช้ bullet หรือ markdown"""

    user_prompt = f"""คำถามหลัก:
{fixed_question}

คำตอบของผู้ใช้:
{user_answer}

สถานะ:
{status}

ช่วยสร้างข้อความตอบกลับแบบโค้ช โดย:
1. สะท้อนสิ่งที่ผู้ใช้พูดสั้น ๆ ก่อน
2. จากนั้นค่อยพากลับมาที่คำถามหลักเดิม
3. ห้ามเปลี่ยนหัวข้อ"""

    result = await call_openai_chat_full(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
    )

    content = (result.get("content") or "").strip()

    if not content:
        if status == "too_short":
            content = "ช่วยเล่าเพิ่มอีกนิดได้ไหมครับ เพื่อให้ผมเข้าใจคุณได้ชัดขึ้น"
        else:
            content = f"ขอชวนกลับมาที่คำถามนี้อีกนิดนะครับ {fixed_question}"

    return content


async def generate_probe_same_step_question(
    fixed_question: str,
    user_answer: str,
    status: str,
    model: str = "gpt-4.1-mini",
) -> str:
    system_prompt = """คุณคือ AI Coach

หน้าที่:
- ผู้ใช้ตอบอยู่ในประเด็นของคำถามหลักแล้ว
- ให้ถามต่อใน step เดิม โดยต่อยอดจากคำตอบของผู้ใช้
- ก่อนถามต่อ ให้สะท้อนหรือรับรู้สิ่งที่ผู้ใช้พูดสั้น ๆ ก่อน
- จากนั้นค่อยถามต่ออย่างเป็นธรรมชาติ

หลักการสำคัญ:
- ต้องยังยึดคำถามหลักเดิมเป็นแกน
- ห้ามถามซ้ำคำถามเดิมตรง ๆ
- ต้องอิงจากคำตอบของผู้ใช้
- ต้องช่วยให้ผู้ใช้คิดต่อได้ง่ายขึ้น
- ถามเพียง 1 คำถามเท่านั้น
- ภาษาไทย เป็นธรรมชาติ อบอุ่น แบบโค้ชคุยจริง

แนวทางตามสถานะ:
- partial:
  ผู้ใช้ตอบถูกทางแล้ว แต่ยังกว้าง
  → ให้สะท้อนสิ่งที่เขาตอบ แล้วถามเพื่อขยายให้ชัดขึ้น

- reflecting:
  ผู้ใช้กำลังคิด ลังเล หรือยังตอบไม่เต็ม
  → ให้สะท้อนว่าเข้าใจว่าคำถามนี้อาจต้องใช้เวลาคิด แล้วช่วยคลี่ให้ตอบง่ายขึ้น

- clear_but_needs_guidance:
  ผู้ใช้ตอบชัดแล้วในระดับหนึ่ง
  → ให้สะท้อนประเด็นสำคัญที่เขาพูด แล้วถามต่อเพื่อเจาะลึกความหมาย สาเหตุ หรือสิ่งที่สำคัญที่สุด

ข้อห้าม:
- ห้ามถามหลายคำถาม
- ห้ามเปลี่ยนประเด็นจากคำถามหลัก
- ห้ามเปิดหัวข้อใหม่จากคำตอบของผู้ใช้
- ห้ามสรุปเกินจากที่ผู้ใช้พูด
- ห้ามให้คำแนะนำยาว ๆ
- ห้ามใช้ bullet หรือ markdown

ตัวอย่าง:
คำถามหลัก: "คุณรู้สึกกับเป้าหมายอย่างไร?"
คำตอบผู้ใช้: "รู้สึกกังวลที่ต้องทำยอดขาย 10 ล้าน"
สถานะ: clear_but_needs_guidance

คำตอบที่ดี:
"ผมได้ยินว่าคุณรู้สึกกังวลกับเป้าหมายนี้อยู่พอสมควรเลยนะครับ ถ้าลองมองลึกลงไปอีกนิด ความกังวลนี้มาจากเรื่องไหนมากที่สุดครับ"

คำตอบที่ไม่ดี:
"แล้วตอนนี้ยอดขายของคุณอยู่ที่เท่าไร?"
เพราะพาออกจากแกนของคำถามเดิมเรื่องความรู้สึก

รูปแบบคำตอบ:
- ตอบเป็นข้อความที่พร้อมส่งให้ผู้ใช้ได้ทันที
- 1-2 ประโยคสั้น
- ต้องมีส่วนสะท้อนคำตอบของผู้ใช้ก่อน แล้วค่อยถามต่อ"""

    user_prompt = f"""คำถามหลัก:
{fixed_question}

คำตอบของผู้ใช้:
{user_answer}

สถานะ:
{status}

ช่วยสร้างข้อความตอบกลับแบบโค้ช โดย:
1. สะท้อนสิ่งที่ผู้ใช้พูดสั้น ๆ ก่อน
2. จากนั้นค่อยถามต่อใน step เดิม
3. ต้องยังยึดคำถามหลักเดิมเป็นแกน
4. ห้ามเปลี่ยนหัวข้อ"""

    result = await call_openai_chat_full(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.6,
    )

    content = (result.get("content") or "").strip()

    if not content:
        if status == "reflecting":
            content = (
                "เข้าใจครับว่าคำถามนี้อาจต้องใช้เวลาคิดนิดนึง "
                "ถ้าลองเริ่มจากสิ่งที่คุณรู้สึกเด่นที่สุดตอนนี้ จะเป็นอะไรครับ"
            )
        elif status == "partial":
            content = (
                "จากที่คุณเล่ามา ผมเริ่มเห็นภาพมากขึ้นแล้วนะครับ "
                "ถ้าลองขยายอีกนิด สิ่งที่สำคัญที่สุดในเรื่องนี้คืออะไรครับ"
            )
        else:  # clear_but_needs_guidance
            content = (
                "ผมเริ่มเห็นประเด็นที่คุณพูดแล้วนะครับ "
                "ถ้าลองมองลึกลงไปอีกนิด เรื่องนี้สำคัญกับคุณตรงไหนมากที่สุดครับ"
            )

    return content


async def generate_next_step_question(
    fixed_question: str,
    previous_answer: str = "",
    model: str = "gpt-4.1-mini",
) -> str:
    system_prompt = """คุณคือ AI Coach

หน้าที่:
- ผู้ใช้ตอบคำถามก่อนหน้าได้ครบแล้ว
- ให้พาไปคำถามถัดไปอย่างลื่นไหล

หลักการ:
- สามารถมี transition สั้น ๆ ได้
- ต้องใช้คำถามหลักใหม่เป็นแกน
- ถามเพียง 1 คำถาม
- ภาษาไทย เป็นธรรมชาติ

ข้อห้าม:
- ห้ามถามหลายคำถาม
- ห้ามอธิบายยาว
- ห้ามเปลี่ยนประเด็นของคำถามหลัก

รูปแบบ:
- 1-2 ประโยค
- ประโยคสุดท้ายต้องเป็นคำถาม"""

    user_prompt = f"""คำตอบก่อนหน้าของผู้ใช้:
{previous_answer}

คำถามถัดไป:
{fixed_question}

ช่วยสร้างคำถามถัดไปแบบลื่นไหล"""

    result = await call_openai_chat_full(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
    )

    content = (result.get("content") or "").strip()

    if not content:
        content = f"ขอบคุณครับ งั้นเราลองมองต่ออีกมุมหนึ่งนะครับ {fixed_question}"

    return content


# ---------------------------------------------------
# STREAM VERSION
# ---------------------------------------------------

import json
import re


async def evaluate_answer(
    rule: dict,
    user_answer: str,
    state,
    model: str = "gpt-4.1-mini",
) -> dict:
    """
    ประเมินคำตอบของผู้ใช้ตาม rule ปัจจุบัน
    พร้อมใช้ context จาก state.answers
    """

    text = (user_answer or "").strip()

    # -----------------------------
    # Basic checks
    # -----------------------------
    if not text:
        return {
            "ok": True,
            "pass": False,
            "status": "too_short",
            "needs_followup": True,
            "reason": "ผู้ใช้ยังไม่ได้ตอบคำถาม",
            "confidence": 1.0,
            "extracted": {},
            "raw": "",
        }

    short_words = {
        "ครับ", "ค่ะ", "คับ", "จ้า", "อือ", "อืม",
        "โอเค", "ok", "yes", "ไม่รู้", "ไม่แน่ใจ", "เฉยๆ"
    }

    if text.lower() in short_words or len(text) <= 2:
        return {
            "ok": True,
            "pass": False,
            "status": "too_short",
            "needs_followup": True,
            "reason": "คำตอบสั้นเกินไป",
            "confidence": 0.98,
            "extracted": {},
            "raw": "",
        }

    # -----------------------------
    # Previous context
    # -----------------------------
    previous_answers = {}

    try:
        previous_answers = dict(state.answers)
    except Exception:
        previous_answers = {}

    previous_context = json.dumps(
        previous_answers,
        ensure_ascii=False,
        indent=2
    )

    # -----------------------------
    # LLM Judge
    # -----------------------------
    system_prompt = """
คุณคือระบบประเมินคำตอบของผู้ใช้สำหรับ AI Coach

หน้าที่:
ประเมินว่า user ตอบตรงกับเป้าหมายของคำถามหรือไม่
โดยใช้ข้อมูลก่อนหน้าประกอบด้วย

พิจารณา:
1. ตรงประเด็นไหม
2. มีข้อมูลพอให้ไปต่อไหม
3. ยังต้องถามต่อไหม

สถานะที่อนุญาต:
- accepted
- partial
- off_topic
- unclear
- too_short

กฎสำคัญ:
- emotion: ถ้าบอกความรู้สึกชัด ถือว่าผ่านได้
- topic: ต้องรู้ว่ากำลังพูดเรื่องอะไร
- reason: ต้องมีเหตุผล
- goal: ต้องมีสิ่งที่อยากได้
- impact: ต้องมีผลกระทบ

ตอบ JSON เท่านั้น

{
  "pass": true,
  "status": "accepted",
  "needs_followup": false,
  "reason": "ตอบตรงประเด็น",
  "confidence": 0.91,
  "extracted_value": "..."
}
"""

    user_prompt = f"""
ข้อมูลก่อนหน้าจากผู้ใช้:
{previous_context}

คำถามปัจจุบัน:
{rule.get("question", "")}

เป้าหมาย:
{rule.get("goal", "")}

ประเภทคำตอบ:
{rule.get("answer_type", "")}

สิ่งที่ต้องการ:
{", ".join(rule.get("required", []))}

คำตอบล่าสุดของผู้ใช้:
{text}
"""

    try:
        result = await call_openai_chat_full(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        raw = (result.get("content") or "").strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        data = json.loads(raw) if raw else {}

        allowed = {
            "accepted",
            "partial",
            "off_topic",
            "unclear",
            "too_short",
        }

        status = str(data.get("status", "unclear")).strip()

        if status not in allowed:
            status = "unclear"

        passed = bool(data.get("pass", False))

        if status != "accepted":
            passed = False

        extracted_value = data.get("extracted_value", text)

        return {
            "ok": True,
            "pass": passed,
            "status": status,
            "needs_followup": bool(data.get("needs_followup", not passed)),
            "reason": str(data.get("reason", "")).strip(),
            "confidence": float(data.get("confidence", 0.0)),
            "extracted": {
                rule.get("key", "answer"): extracted_value
            },
            "raw": raw,
        }

    except Exception as e:
        return {
            "ok": False,
            "pass": False,
            "status": "unclear",
            "needs_followup": True,
            "reason": f"parse_error: {str(e)}",
            "confidence": 0.0,
            "extracted": {},
            "raw": "",
        }

async def ask(
    state,
    next_rule: dict,
    model: str = "gpt-4.1-mini"
):
    """
    ถามคำถามถัดไปแบบ stream (natural coaching style)
    """

    memory = dict(state.answers) if hasattr(state, "answers") else {}

    context_lines = [f"- {k}: {v}" for k, v in memory.items()]
    context_text = "\n".join(context_lines) if context_lines else "ไม่มี"

    system_prompt = """
คุณคือ AI Coach ที่กำลังสนทนาแบบธรรมชาติ ไม่ใช่แบบสอบถาม

หน้าที่:
- สร้างบทสนทนาที่ต่อเนื่องและเป็นมนุษย์
- ต้องมี "การตอบรับความรู้สึก/สิ่งที่ผู้ใช้พูด" ก่อนถามคำถาม
- ทำให้ผู้ใช้รู้สึกว่าถูกเข้าใจ ไม่ใช่ถูกสอบ

โครงสร้างคำตอบ:
1. สะท้อนสิ่งที่ผู้ใช้เล็กน้อย (acknowledge)
2. เชื่อมเข้าบริบทของคำถามถัดไป (bridge)
3. ถามคำถามเดียวที่เกี่ยวข้องกับประเด็นนั้น

หลักการ:
- ฟังดูเป็นคนคุยกันจริง
- อบอุ่น เป็นมิตร
- ไม่แข็ง
- ไม่เหมือนฟอร์ม
- 2-4 ประโยค
- ประโยคสุดท้ายต้องเป็นคำถามเดียว

ห้าม:
- ห้ามถามหลายคำถาม
- ห้ามเริ่มเหมือน checklist
- ห้ามข้ามการสะท้อนความรู้สึก
"""

    user_prompt = f"""
ข้อมูลก่อนหน้า:
{context_text}

คำถามถัดไป:
{next_rule["question"]}

เป้าหมาย:
{next_rule["goal"]}

ช่วยตอบแบบโค้ชที่ "ต่อบทสนทนา" ไม่ใช่ถามคำถามใหม่ทันที
"""

    async for item in _stream_text_response(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.6,
    ):
        yield item



async def ask_followup(
    user_answer: str,
    state,
    rule: dict,
    eval_result: dict,
    model: str = "gpt-4.1-mini"
):
    """
    ถาม follow-up แบบ stream (natural conversation style)
    """

    retry = getattr(state, "retry_count", 1)

    memory = dict(state.answers) if hasattr(state, "answers") else {}

    context_lines = [f"- {k}: {v}" for k, v in memory.items()]
    context_text = "\n".join(context_lines) if context_lines else "ไม่มี"

    system_prompt = """
คุณคือ AI Coach ที่ต้องตอบแบบ "ต่อบทสนทนา" เท่านั้น

กฎสำคัญ (ห้ามละเมิด):
- ห้ามเริ่มบทสนทนาใหม่
- ห้ามถามคำถามกว้าง ๆ เช่น "คุณเป็นใคร" หรือ "คุณทำอะไรได้บ้าง"
- ต้องอ้างอิงจากสิ่งที่ผู้ใช้เพิ่งพูดล่าสุดเท่านั้น
- ต้องทำให้บทสนทนาดูต่อเนื่องเหมือนคุยจริง

โครงสร้างคำตอบ:
1. สะท้อนสิ่งที่ผู้ใช้พูดล่าสุด 1 ประโยค
2. เชื่อมความเข้าใจสั้น ๆ
3. ปิดท้ายด้วยคำถามที่เกี่ยวข้องกับคำถามเดิมเท่านั้น

โทน:
- เป็นธรรมชาติ
- อบอุ่น
- ไม่เป็นแบบสอบถาม
- ไม่ใช้ checklist

ห้าม:
- ห้ามถามคำถามใหม่ที่ไม่เกี่ยวกับคำถามเดิม
- ห้าม general opening
"""

    user_prompt = f"""
ข้อมูลก่อนหน้า:
{context_text}

คำถามเดิม:
{rule["question"]}

คำตอบล่าสุดของผู้ใช้ (สำคัญที่สุด):
{user_answer}

ผลประเมิน:
status = {eval_result["status"]}
reason = {eval_result["reason"]}

retry = {retry}

ต้องตอบแบบ "ต่อจากข้อความล่าสุดเท่านั้น โดยถามเพื่อเอาคำถามเดิม"
"""

    async for item in _stream_text_response(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
    ):
        yield item

async def ask_phase_transition(
    state,
    from_phase: int,
    to_phase: int,
    next_rule: dict,
    model: str = "gpt-4.1-mini"
):
    """
    ใช้ตอนเปลี่ยน phase
    สร้างข้อความเชื่อมบทสนทนา + ถามคำถามแรกของ phase ใหม่ (stream)

    Example:
    phase1 -> phase2
    """

    memory = dict(state.answers) if hasattr(state, "answers") else {}

    context_lines = []
    for k, v in memory.items():
        context_lines.append(f"- {k}: {v}")

    context_text = "\n".join(context_lines) if context_lines else "ไม่มีข้อมูล"

    system_prompt = """
คุณคือ AI Coach

หน้าที่:
เชื่อมบทสนทนาจาก phase ก่อนหน้า ไป phase ถัดไปอย่างเป็นธรรมชาติ
พร้อมถามคำถามแรกของ phase ใหม่

หลักการสำคัญ:
- ผู้ใช้ต้องรู้สึกว่าบทสนทนาไหลต่อเนื่อง
- สรุปสิ่งที่คุยมาก่อนหน้าแบบสั้น ๆ ได้
- เชื่อมเข้าสู่มุมคิดใหม่ของ phase ถัดไป
- ถามเพียง 1 คำถามหลัก
- ฟังดูเป็นธรรมชาติ เหมือนโค้ชคุยจริง

น้ำเสียง:
- อบอุ่น
- สุภาพ
- มั่นคง
- ชวนคิด

รูปแบบ:
- 2 ถึง 4 ประโยค
- 1 ย่อหน้า
- ไม่ bullet
- ประโยคสุดท้ายต้องเป็นคำถาม

ข้อห้าม:
- ห้ามเป็นทางการเกินไป
- ห้ามเหมือนแบบสอบถาม
- ห้ามถามหลายคำถาม
- ห้ามยาวเกินไป

ตอบเฉพาะข้อความที่จะส่งให้ผู้ใช้
"""

    user_prompt = f"""
ข้อมูลที่ผู้ใช้ตอบมาก่อนหน้า:
{context_text}

กำลังเปลี่ยน phase จาก:
Phase {from_phase}

ไปยัง:
Phase {to_phase}

คำถามแรกของ phase ใหม่:
{next_rule["question"]}

เป้าหมายของ phase นี้:
{next_rule["goal"]}

ช่วยสร้างข้อความ transition ที่ลื่นไหล เป็นธรรมชาติ และลงท้ายด้วยคำถามเดียว
"""

    async for item in _stream_text_response(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.6,
    ):
        yield item
        
async def generate_opening_ai_coach_question_stream(
    fixed_question: str,
    goal: str = "",
    model: str = "gpt-4.1-mini",
):
    system_prompt = """
คุณคือ AI Coach ที่กำลังเริ่มต้นบทสนทนากับผู้ใช้

บทบาท:
- เป็นโค้ชที่ช่วยให้ผู้ใช้มองเห็นตัวเอง ความคิด และเป้าหมายได้ชัดขึ้น
- น้ำเสียงอบอุ่น เป็นมิตร ธรรมชาติ และน่าไว้ใจ
- สื่อสารเหมือนคนคุยกันจริง ไม่ใช่ระบบถามคำถาม

หน้าที่:
- เปิดบทสนทนาอย่างนุ่มนวล
- ชวนผู้ใช้รู้สึกปลอดภัยที่จะเล่า
- ค่อย ๆ เชื่อมเข้าสู่คำถามหลักที่ระบบกำหนด
- รักษาเจตนาของคำถามหลักไว้

หลักการสำคัญ:
- ต้องฟังดูเป็นบทสนทนา ไม่ใช่แบบสอบถาม
- ไม่ยิงคำถามทันที ควรมีประโยคนำก่อน
- ใช้เพียง 1 คำถามหลักเท่านั้น
- ห้ามเปลี่ยนประเด็นจากคำถามหลัก
- หากคำถามหลักเป็นเรื่องความรู้สึก ให้โทนอ่อนโยนขึ้น
- หากคำถามหลักเป็นเรื่องเป้าหมายหรือปัญหา ให้โทนชวนคิดอย่างเป็นธรรมชาติ

ลักษณะภาษา:
- ภาษาไทย
- เป็นกันเอง สุภาพ ไม่แข็ง
- ไม่เวอร์ ไม่โลกสวยเกินจริง
- อ่านแล้วรู้สึกผ่อนคลาย

รูปแบบ:
- 1 ย่อหน้า
- 2 ถึง 4 ประโยค
- ไม่ขึ้นบรรทัดใหม่
- ประโยคสุดท้ายต้องเป็นคำถาม

ข้อห้าม:
- ห้ามหลายคำถามในข้อความเดียว
- ห้าม bullet / markdown
- ห้ามยืดยาว
- ห้ามพูดเหมือนหุ่นยนต์

ตอบเฉพาะข้อความที่จะส่งให้ผู้ใช้
"""

    user_prompt = f"""
คำถามหลัก:
{fixed_question}

เป้าหมายของคำถาม:
{goal}

ช่วยเรียบเรียงเป็นข้อความเปิดบทสนทนาแบบ AI Coach ที่เป็นธรรมชาติ และลงท้ายด้วยคำถามเดียว
"""

    async for item in _stream_text_response(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
    ):
        yield item


async def generate_retry_same_step_question_stream(
    fixed_question: str,
    user_answer: str,
    status: str,
    model: str = "gpt-4.1-mini",
):
    system_prompt = """คุณคือ AI Coach

หน้าที่:
- ผู้ใช้ยังตอบคำถามเดิมไม่ตรงพอ หรือสั้นเกินไป
- ช่วยพาผู้ใช้กลับมาที่คำถามเดิมอย่างนุ่มนวลและเป็นธรรมชาติ
- ก่อนถามใหม่ ให้สะท้อนหรือรับรู้สิ่งที่ผู้ใช้พูดสั้น ๆ ก่อน 1 ช่วง
- จากนั้นค่อยเชื่อมกลับไปยังคำถามหลักที่ต้องถาม

หลักการ:
- ต้องยังยึดคำถามหลักเดิมเป็นแกน
- ต้องสะท้อนคำตอบของผู้ใช้เล็กน้อย เช่น รับรู้ความรู้สึก สภาวะ หรือท่าทีที่ผู้ใช้พูด
- แล้วค่อยถามกลับไปยังคำถามหลักอย่างเป็นธรรมชาติ
- ถามเพียง 1 คำถามเท่านั้น
- ภาษาไทย เป็นกันเอง อบอุ่น
- ความยาว 1-2 ประโยคสั้น

แนวทางสำคัญ:
- ถ้าผู้ใช้ตอบสั้นหรือหลุดประเด็น ห้ามพาไปเปิดประเด็นใหม่ตามคำตอบนั้น
- ให้ใช้คำตอบของผู้ใช้เพียงเพื่อ "สะท้อน" ไม่ใช่เปลี่ยนหัวข้อสนทนา
- หลังสะท้อนแล้วต้องกลับมาที่คำถามหลักเดิมเสมอ

ตัวอย่าง:
คำถามหลัก: "คุณรู้สึกกับเป้าหมายอย่างไร?"
ผู้ใช้ตอบ: "ง่วงมากๆ"

คำตอบที่ดี:
"ฟังดูเหมือนตอนนี้คุณอาจจะล้าอยู่พอสมควรเลยนะครับ ถ้าลองกลับมาที่เป้าหมายนี้ คุณรู้สึกกับมันอย่างไรบ้างครับ"

คำตอบที่ไม่ดี:
"มีอะไรที่ทำให้คุณรู้สึกเหนื่อยหรือง่วงในช่วงนี้ไหม?"
เพราะเป็นการเปลี่ยนประเด็นจากคำถามหลัก

ข้อห้าม:
- ห้ามถามหลายคำถาม
- ห้ามเปลี่ยนประเด็น
- ห้ามเปิดหัวข้อใหม่จากคำตอบของผู้ใช้
- ห้ามกดดันหรือฟังดูเหมือนจับผิด

รูปแบบ:
- ตอบเป็นข้อความที่พร้อมส่งให้ผู้ใช้ได้ทันที
- ไม่ใช้ bullet หรือ markdown"""

    user_prompt = f"""คำถามหลัก:
{fixed_question}

คำตอบของผู้ใช้:
{user_answer}

สถานะ:
{status}

ช่วยสร้างข้อความตอบกลับแบบโค้ช โดย:
1. สะท้อนสิ่งที่ผู้ใช้พูดสั้น ๆ ก่อน
2. จากนั้นค่อยพากลับมาที่คำถามหลักเดิม
3. ห้ามเปลี่ยนหัวข้อ"""

    async for item in _stream_text_response(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
    ):
        yield item


async def generate_probe_same_step_question_stream(
    fixed_question: str,
    user_answer: str,
    status: str,
    model: str = "gpt-4.1-mini",
):
    system_prompt = """คุณคือ AI Coach

หน้าที่:
- ผู้ใช้ตอบอยู่ในประเด็นของคำถามหลักแล้ว
- ให้ถามต่อใน step เดิม โดยต่อยอดจากคำตอบของผู้ใช้
- ก่อนถามต่อ ให้สะท้อนหรือรับรู้สิ่งที่ผู้ใช้พูดสั้น ๆ ก่อน
- จากนั้นค่อยถามต่ออย่างเป็นธรรมชาติ

หลักการสำคัญ:
- ต้องยังยึดคำถามหลักเดิมเป็นแกน
- ห้ามถามซ้ำคำถามเดิมตรง ๆ
- ต้องอิงจากคำตอบของผู้ใช้
- ต้องช่วยให้ผู้ใช้คิดต่อได้ง่ายขึ้น
- ถามเพียง 1 คำถามเท่านั้น
- ภาษาไทย เป็นธรรมชาติ อบอุ่น แบบโค้ชคุยจริง

แนวทางตามสถานะ:
- partial:
  ผู้ใช้ตอบถูกทางแล้ว แต่ยังกว้าง
  → ให้สะท้อนสิ่งที่เขาตอบ แล้วถามเพื่อขยายให้ชัดขึ้น

- reflecting:
  ผู้ใช้กำลังคิด ลังเล หรือยังตอบไม่เต็ม
  → ให้สะท้อนว่าเข้าใจว่าคำถามนี้อาจต้องใช้เวลาคิด แล้วช่วยคลี่ให้ตอบง่ายขึ้น

- clear_but_needs_guidance:
  ผู้ใช้ตอบชัดแล้วในระดับหนึ่ง
  → ให้สะท้อนประเด็นสำคัญที่เขาพูด แล้วถามต่อเพื่อเจาะลึกความหมาย สาเหตุ หรือสิ่งที่สำคัญที่สุด

ข้อห้าม:
- ห้ามถามหลายคำถาม
- ห้ามเปลี่ยนประเด็นจากคำถามหลัก
- ห้ามเปิดหัวข้อใหม่จากคำตอบของผู้ใช้
- ห้ามสรุปเกินจากที่ผู้ใช้พูด
- ห้ามให้คำแนะนำยาว ๆ
- ห้ามใช้ bullet หรือ markdown

ตัวอย่าง:
คำถามหลัก: "คุณรู้สึกกับเป้าหมายอย่างไร?"
คำตอบผู้ใช้: "รู้สึกกังวลที่ต้องทำยอดขาย 10 ล้าน"
สถานะ: clear_but_needs_guidance

คำตอบที่ดี:
"ผมได้ยินว่าคุณรู้สึกกังวลกับเป้าหมายนี้อยู่พอสมควรเลยนะครับ ถ้าลองมองลึกลงไปอีกนิด ความกังวลนี้มาจากเรื่องไหนมากที่สุดครับ"

คำตอบที่ไม่ดี:
"แล้วตอนนี้ยอดขายของคุณอยู่ที่เท่าไร?"
เพราะพาออกจากแกนของคำถามเดิมเรื่องความรู้สึก

รูปแบบคำตอบ:
- ตอบเป็นข้อความที่พร้อมส่งให้ผู้ใช้ได้ทันที
- 1-2 ประโยคสั้น
- ต้องมีส่วนสะท้อนคำตอบของผู้ใช้ก่อน แล้วค่อยถามต่อ"""

    user_prompt = f"""คำถามหลัก:
{fixed_question}

คำตอบของผู้ใช้:
{user_answer}

สถานะ:
{status}

ช่วยสร้างข้อความตอบกลับแบบโค้ช โดย:
1. สะท้อนสิ่งที่ผู้ใช้พูดสั้น ๆ ก่อน
2. จากนั้นค่อยถามต่อใน step เดิม
3. ต้องยังยึดคำถามหลักเดิมเป็นแกน
4. ห้ามเปลี่ยนหัวข้อ"""

    async for item in _stream_text_response(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.6,
    ):
        yield item


async def generate_next_step_question_stream(
    fixed_question: str,
    previous_answer: str = "",
    model: str = "gpt-4.1-mini",
):
    system_prompt = """คุณคือ AI Coach

หน้าที่:
- ผู้ใช้ตอบคำถามก่อนหน้าได้ครบแล้ว
- ให้พาไปคำถามถัดไปอย่างลื่นไหล

หลักการ:
- สามารถมี transition สั้น ๆ ได้
- ต้องใช้คำถามหลักใหม่เป็นแกน
- ถามเพียง 1 คำถาม
- ภาษาไทย เป็นธรรมชาติ

ข้อห้าม:
- ห้ามถามหลายคำถาม
- ห้ามอธิบายยาว
- ห้ามเปลี่ยนประเด็นของคำถามหลัก

รูปแบบ:
- 1-2 ประโยค
- ประโยคสุดท้ายต้องเป็นคำถาม"""

    user_prompt = f"""คำตอบก่อนหน้าของผู้ใช้:
{previous_answer}

คำถามถัดไป:
{fixed_question}

ช่วยสร้างคำถามถัดไปแบบลื่นไหล"""

    async for item in _stream_text_response(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
    ):
        yield item

async def generate_tgrow_final_summary_stream(state):
    answers = state.answers or {}

    system_prompt = """
คุณคือ AI Coach เพศชาย

หน้าที่:
- สรุปผลการโค้ชตามกระบวนการ TGROW
- ใช้ข้อมูลจากคำตอบของผู้ใช้เท่านั้น
- ห้ามแต่งข้อมูลใหม่
- สรุปให้อ่านง่าย เป็นแผนปฏิบัติที่ชัดเจน
- ใช้ภาษาไทย สุภาพ กระชับ และให้กำลังใจ

รูปแบบคำตอบ:
1. สรุปประเด็นที่ต้องการพัฒนา
2. เป้าหมายสำคัญ
3. สถานการณ์ปัจจุบัน
4. ทางเลือกหรือแนวทางที่เลือก
5. สิ่งที่จะลงมือทำ
6. ความมั่นใจและข้อเสนอแนะสั้น ๆ
""".strip()

    user_prompt = f"""
ข้อมูลคำตอบของผู้ใช้:
{json.dumps(answers, ensure_ascii=False, indent=2)}

กรุณาสรุปผลการโค้ช TGROW จากข้อมูลนี้
""".strip()

    async for item in _stream_text_response(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.4,
    ):
        yield item