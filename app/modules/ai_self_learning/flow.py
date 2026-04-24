from app.modules.ai_self_learning.course_service import  get_course_data_by_no_bridge
# get_course_data_by_no,

# shared layer (ตัวนี้ใช้ร่วม)
from app.shared.ai.openai_client import (
    call_openai_chat_full,
    call_openai_chat_stream_full,
)

from app.modules.ai_self_learning.schema import ChatState_aiselflearning


import json


async def process_chat_aiselflearning(req, state, conn):
    if state is None:
        state = ChatState_aiselflearning()

    user_message = req.user_message.strip()
    course_no = req.OCourse_no

    course_data = get_course_data_by_no_bridge(conn, course_no)

    if not course_data:
        reply = "ขออภัยครับ ไม่พบข้อมูลหลักสูตรนี้"
        state.chat_id = req.chat_id
        state.OCourse_no = course_no
        state.last_user_message = user_message
        state.last_answer = reply

        return type("Obj", (), {
            "reply": reply,
            "state": state,
            "source": "ai_self_learning_no_data",
        })()

    scripts = [row[0] for row in course_data if row[0]]
    course_name = [row[1] for row in course_data if row[1]]
    context = "\n\n".join(scripts[:3])

    system_prompt = f"""
คุณคือผู้ช่วยบนหน้า Self Learning ที่คุยกับผู้เรียนอย่างเป็นธรรมชาติ
น้ำเสียงที่ต้องการ:
- สุภาพ อบอุ่น เป็นกันเอง
- ฟังเหมือนโค้ชหรือผู้ช่วยสอน ไม่ใช่ระบบอัตโนมัติ
- อธิบายแบบเข้าใจง่าย กระชับ และช่วยให้ผู้เรียนรู้สึกว่าได้รับคำแนะนำจริง
- หลีกเลี่ยงภาษาทางการแข็ง ๆ หรือภาษาที่ฟังเหมือนรายงาน
- ไม่ต้องเกริ่นว่า "จากข้อมูลที่มี" หรือ "ตามข้อมูลหลักสูตร" บ่อยเกินจำเป็น
- ถ้าตอบได้ ให้ตอบแบบลื่นและเป็นธรรมชาติ
- ถ้าตอบไม่ได้หรือข้อมูลไม่พอ ให้บอกอย่างนุ่มนวลและตรงไปตรงมา

ขอบเขตการตอบ:
- ใช้ข้อมูลจากเนื้อหาหลักสูตรที่ให้ไว้เป็นหลัก
- ห้ามแต่งข้อมูลเกินจากเนื้อหาที่มี
- ถ้าคำถามไม่เกี่ยวกับหลักสูตร ให้จัดเป็น out_of_scope
- ถ้าคำถามสั้นเกินไปหรือไม่ชัด ให้จัดเป็น unclear

ให้ตอบกลับมาเป็น JSON เท่านั้น และห้ามมีข้อความอื่นนอกจาก JSON
รูปแบบต้องเป็นแบบนี้เท่านั้น:
{{
  "reply": "ข้อความตอบผู้ใช้",
  "status": "answered | out_of_scope | unclear",
  "reason": "เหตุผลสั้น ๆ สำหรับใช้ภายในระบบ"
}}

แนวทางการเขียน reply:
- ถ้า status = answered → ตอบให้เป็นธรรมชาติ เหมือนกำลังอธิบายให้ผู้เรียน
- ถ้า status = out_of_scope → ปฏิเสธอย่างนุ่มนวล และชวนกลับมาถามในประเด็นที่เกี่ยวกับหลักสูตร
- ถ้า status = unclear → ขอให้ผู้ใช้เล่าเพิ่มหรือถามให้ชัดขึ้นแบบเป็นกันเอง

ชื่อหลักสูตร: {course_name}
ข้อมูลหลักสูตร:
{context}
""".strip()

    result = await call_openai_chat_full(
        model="gpt-4.1-nano",
        system_prompt=system_prompt,
        user_prompt=user_message,
        temperature=0.3,
    )

    print("DEBUG result =", result)
    print("DEBUG ok =", result.get("ok"))
    print("DEBUG content =", result.get("content"))
    print("DEBUG error =", result.get("error"))

    content = (result.get("content") or "").strip()

    try:
        ai_json = json.loads(content)

        reply = ai_json.get("reply", "")
        status = ai_json.get("status", "unknown")
        reason = ai_json.get("reason", "")

    except Exception as e:
        print("JSON PARSE ERROR =", e)
        reply = "ขออภัยครับ ระบบไม่สามารถแปลผลคำตอบได้"
        status = "error"
        reason = "invalid_json"

    state.chat_id = req.chat_id
    state.OCourse_no = course_no
    state.last_user_message = user_message
    state.last_answer = reply

    return type("Obj", (), {
    "reply": reply,
    "status": status,
    "reason": reason,
    "state": state,
    "source": "ai_self_learning",
})()

async def process_chat_aiselflearning_stream(req, state, conn):
    if state is None:
        state = ChatState_aiselflearning()

    user_message = req.user_message.strip()
    course_no = req.OCourse_no

    course_data = get_course_data_by_no_bridge(conn, course_no)

    if not course_data:
        reply = "ขออภัยครับ ไม่พบข้อมูลหลักสูตรนี้"
        state.chat_id = req.chat_id
        state.OCourse_no = course_no
        state.last_user_message = user_message
        state.last_answer = reply

        yield {
            "type": "chunk",
            "text": reply,
        }

        yield {
            "type": "done",
            "reply": reply,
            "status": "error",
            "reason": "course_not_found",
            "state": state,
            "source": "ai_self_learning_no_data",
        }
        return

    scripts = [row[0] for row in course_data if row[0]]
    course_name = [row[1] for row in course_data if row[1]]
    context = "\n\n".join(scripts[:3])

    system_prompt = f"""
คุณคือผู้ช่วยบนหน้า Self Learning ที่คุยกับผู้เรียนอย่างเป็นธรรมชาติ
น้ำเสียงที่ต้องการ:
- สุภาพ อบอุ่น เป็นกันเอง
- ฟังเหมือนโค้ชหรือผู้ช่วยสอน ไม่ใช่ระบบอัตโนมัติ
- อธิบายแบบเข้าใจง่าย กระชับ และช่วยให้ผู้เรียนรู้สึกว่าได้รับคำแนะนำจริง
- หลีกเลี่ยงภาษาทางการแข็ง ๆ หรือภาษาที่ฟังเหมือนรายงาน
- ถ้าตอบได้ ให้ตอบแบบลื่นและเป็นธรรมชาติ
- ถ้าตอบไม่ได้หรือข้อมูลไม่พอ ให้บอกอย่างนุ่มนวลและตรงไปตรงมา

ขอบเขตการตอบ:
- ใช้ข้อมูลจากเนื้อหาหลักสูตรที่ให้ไว้เป็นหลัก
- ห้ามแต่งข้อมูลเกินจากเนื้อหาที่มี
- ถ้าคำถามไม่เกี่ยวกับหลักสูตร ให้ปฏิเสธอย่างนุ่มนวล และชวนกลับมาถามในประเด็นที่เกี่ยวกับหลักสูตร
- ถ้าคำถามสั้นเกินไปหรือไม่ชัด ให้ขอให้ผู้ใช้เล่าเพิ่มหรือถามให้ชัดขึ้นแบบเป็นกันเอง

กฎสำคัญที่สุด:
- ถ้าคำถามของผู้ใช้ "ไม่เกี่ยวกับหลักสูตรนี้เลย" ห้ามตอบเนื้อหาอื่นโดยเด็ดขาด
- ให้ตอบปฏิเสธอย่างสุภาพเท่านั้น และชวนให้ถามใหม่ในเรื่องที่เกี่ยวข้องกับหลักสูตร
- ห้ามให้คำแนะนำทั่วไปนอกเหนือจากเนื้อหาหลักสูตร

ตอบเป็นข้อความธรรมดาเท่านั้น ห้ามตอบเป็น JSON

ชื่อหลักสูตร: {course_name}
ข้อมูลหลักสูตร:
{context}
""".strip()

    final_reply = ""
    final_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    final_cost = None

    try:
        async for item in call_openai_chat_stream_full(
            model="gpt-4.1-nano",
            system_prompt=system_prompt,
            user_prompt=user_message,
            temperature=0.3,
        ):
            if item["type"] == "chunk":
                text = item.get("text", "")
                if text:
                    final_reply += text
                    yield {
                        "type": "chunk",
                        "text": text,
                    }

            elif item["type"] == "done":
                final_reply = item.get("content", final_reply).strip()
                final_usage = item.get("usage", final_usage)
                final_cost = item.get("cost", None)

    except Exception as e:
        final_reply = "ขออภัยครับ ระบบไม่สามารถตอบได้ชั่วคราว"
        yield {
            "type": "chunk",
            "text": final_reply,
        }

        state.chat_id = req.chat_id
        state.OCourse_no = course_no
        state.last_user_message = user_message
        state.last_answer = final_reply

        yield {
            "type": "done",
            "reply": final_reply,
            "status": "error",
            "reason": f"stream_error: {str(e)}",
            "state": state,
            "source": "ai_self_learning",
            "usage": final_usage,
            "cost": final_cost,
        }
        return

    state.chat_id = req.chat_id
    state.OCourse_no = course_no
    state.last_user_message = user_message
    state.last_answer = final_reply

    yield {
        "type": "done",
        "reply": final_reply,
        "status": "answered" if final_reply else "error",
        "reason": "stream_completed" if final_reply else "empty_reply",
        "state": state,
        "source": "ai_self_learning",
        "usage": final_usage,
        "cost": final_cost,
    }