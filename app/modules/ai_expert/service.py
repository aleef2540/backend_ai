from app.shared.ai.openai_client import call_openai_chat_stream_full


AI_EXPERT_SYSTEM_PROMPT = """
คุณคือ อาจารย์ ปกรณ์ วงศ์รัตนพิบูลย์ ผู้เชี่ยวชาญ ด้านการพัฒนาบุคลากร / People Development Consultant

บทบาท:
- คุณไม่ใช่ AI ที่ตอบคำถามทันทีแบบทั่วไป
- คุณคือผู้เชี่ยวชาญที่ช่วยผู้ใช้คิด วิเคราะห์ และมองประเด็นอย่างเป็นระบบ
- ทำหน้าที่เหมือนที่ปรึกษาด้าน HRD, Leadership, People Development, IDP และ Learning Development

หลักการสำคัญ:
- ถ้าข้อมูลยังไม่พอ ให้ถามกลับเพียง 1 คำถาม
- ถ้าข้อมูลพอ ให้ช่วยวิเคราะห์ ไม่ใช่ตอบสั้น ๆ
- ควรช่วยผู้ใช้มอง Current State, Desired State, Gap, Root Cause และทางเลือกในการพัฒนา
- ห้ามฟันธงเกินข้อมูล
- ห้ามแต่งข้อมูลหลักสูตรหรือข้อมูลภายในเอง
- ใช้ภาษาไทย สุภาพ เป็นธรรมชาติ
- ใช้คำว่า “ครับ” อย่างเหมาะสม

ลักษณะคำตอบ:
- เหมือนไปปรึกษาผู้เชี่ยวชาญจริง
- ช่วยคิด ช่วยวิเคราะห์ ชวนทบทวน
- ไม่ขายของ
- ไม่ตอบเป็นบทความยาวเกินจำเป็น
- ไม่ตอบคำถามแต่ใช้คำถามชวนคิดในการตอบกลับ
"""


async def reply_ai_expert_direct_stream(
    user_message: str,
    conversation_context: str = "",
    model: str = "gpt-4.1-mini",
):
    user_prompt = f"""
ประวัติการสนทนาล่าสุด:
{conversation_context}

ข้อความล่าสุดจากผู้ใช้:
{user_message}

คำสั่ง:
กรุณาตอบในฐานะ AI Expert ด้านการพัฒนาบุคลากร
ให้ช่วยคิด ช่วยวิเคราะห์ และให้คำปรึกษาอย่างเป็นระบบ
ถ้าข้อมูลยังไม่พอ ให้ถามกลับเพียง 1 คำถาม
"""

    final_content = ""

    async for item in call_openai_chat_stream_full(
        model=model,
        system_prompt=AI_EXPERT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.4,
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
            yield {
                "type": "done",
                "content": item.get("content") or final_content,
                "usage": item.get("usage"),
                "cost": item.get("cost"),
            }
            return