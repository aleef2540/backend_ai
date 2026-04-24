from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import time

# from app.core.database import get_mysql_connection
from app.utils.debug_state import print_debug, print_state

from app.modules.ai_self_learning.schema import ChatRequest_aiselflearning
from app.modules.ai_self_learning.state_store import chat_state_store_aiselflearning
from app.modules.ai_self_learning.flow import process_chat_aiselflearning_stream

# ⚠️ ตัวนี้เดี๋ยวต้องย้าย/สร้างทีหลัง
from app.modules.ai_self_learning.service import insert_chat_history_aiselflearning

router = APIRouter(tags=["AI Self Learning"])

@router.post("/chat/ai-self-learning")
async def chat_ai_self_learning_stream(req: ChatRequest_aiselflearning):
    req_start = time.time()
    print(f"[ROUTE] /chat/ai-self-learning/stream START at {req_start:.3f}", flush=True)

    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()
    req.user_message = user_message
    # conn_mysql = get_mysql_connection()

    if req.state:
        state = req.state
    else:
        state = chat_state_store_aiselflearning.get_state(req.chat_id)

    print_debug("req.user_message", user_message)
    print_debug("before state", state)
    print_state("BEFORE STATE", state)

    async def event_generator():
        final_reply = ""
        final_status = "error"
        final_reason = ""
        final_state = state
        final_source = "ai_self_learning"

        try:
            async for item in process_chat_aiselflearning_stream(req, state):
                item_type = item.get("type")

                if item_type == "chunk":
                    text = item.get("text", "")
                    final_reply += text

                    payload = json.dumps({
                        "type": "chunk",
                        "text": text
                    }, ensure_ascii=False)

                    yield f"data: {payload}\n\n"

                elif item_type == "done":
                    final_reply = item.get("reply", final_reply)
                    final_status = item.get("status", "answered")
                    final_reason = item.get("reason", "")
                    final_state = item.get("state", final_state)

                    # insert_chat_history_aiselflearning(
                    #     conn=conn_mysql,
                    #     chat_id=req.chat_id,
                    #     course_no=req.OCourse_no,
                    #     user_message=req.user_message,
                    #     ai_reply=final_reply,
                    #     ai_status=final_status,
                    #     ai_reason=final_reason,
                    # )

                    chat_state_store_aiselflearning.set_state(req.chat_id, final_state)

                    payload = json.dumps({
                        "type": "done",
                        "reply": final_reply,
                        "status": final_status,
                        "reason": final_reason,
                        "state": final_state.model_dump() if hasattr(final_state, "model_dump") else None,
                        "source": final_source,
                        "chat_id": req.chat_id,
                    }, ensure_ascii=False)

                    yield f"data: {payload}\n\n"
                    return

        except Exception as e:
            payload = json.dumps({
                "type": "error",
                "message": str(e),
                "chat_id": req.chat_id,
            }, ensure_ascii=False)

            yield f"data: {payload}\n\n"

        # finally:
        #     try:
        #         conn_mysql.close()
        #     except Exception:
        #         pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )