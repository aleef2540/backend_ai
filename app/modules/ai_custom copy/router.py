from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import time

# from app.core.database import get_mysql_connection
from app.utils.debug_state import print_debug, print_state

from app.modules.ai_custom.schema import ChatRequest_aicustom, ResetRequest_aicustom
from app.modules.ai_custom.state_store import chat_state_store_aicustom
from app.modules.ai_custom.flow import process_chat_aicustom_stream

router = APIRouter(tags=["AI Custom"])

@router.post("/chat/ai-custom")
async def chat_ai_custom_stream(req: ChatRequest_aicustom):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    user_message = req.user_message.strip()
    req.user_message = user_message
    # conn_mysql = get_mysql_connection()

    if req.state:
        state = req.state
    else:
        state = chat_state_store_aicustom.get_state(req.web_no, req.member_no)

    state.web_no = int(req.web_no) if req.web_no not in [None, ""] else None
    state.member_no = int(req.member_no) if req.member_no not in [None, ""] else None

    if req.course_use:
        state.course_use = [str(x).strip() for x in req.course_use if str(x).strip()]

    print(f"[ROUTE] /chat/ai-custom/stream START {time.time():.3f}", flush=True)
    print_debug("req.user_message", user_message)
    print_debug("before state", state)
    print_state("BEFORE STATE", state)

    async def event_generator():
        stream_start = time.time()
        final_reply = ""
        final_state = state
        final_source = "ai_custom"
        final_active_video = None
        chunk_count = 0

        print(f"[STREAM] generator START {stream_start:.3f}", flush=True)

        try:
            async for item in process_chat_aicustom_stream(req, state):
                item_type = item.get("type")

                if item_type == "chunk":
                    text = item.get("text", "")
                    chunk_count += 1
                    final_reply += text

                    payload = json.dumps({
                        "type": "chunk",
                        "text": text
                    }, ensure_ascii=False)

                    yield f"data: {payload}\n\n"

                elif item_type == "done":
                    final_reply = item.get("reply", final_reply)
                    final_state = item.get("state", final_state)
                    final_source = item.get("source", final_source)
                    final_active_video = item.get("active_video", final_active_video)

                    chat_state_store_aicustom.set_state(
                        req.web_no,
                        req.member_no,
                        final_state
                    )

                    payload = json.dumps({
                        "type": "done",
                        "reply": final_reply,
                        "state": final_state.model_dump() if hasattr(final_state, "model_dump") else None,
                        "source": final_source,
                        "active_video": final_active_video
                    }, ensure_ascii=False)

                    yield f"data: {payload}\n\n"
                    return

        except Exception as e:
            print(f"[STREAM] EXCEPTION {repr(e)}", flush=True)

            payload = json.dumps({
                "type": "error",
                "message": str(e)
            }, ensure_ascii=False)

            yield f"data: {payload}\n\n"

        finally:
            print(
                f"[STREAM] FINALLY total_chunks={chunk_count} "
                f"total_reply_len={len(final_reply)} "
                f"total_time={time.time() - stream_start:.3f}s",
                flush=True
            )

            # try:
            #     conn_mysql.close()
            # except Exception:
            #     pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/chat/reset/ai-custom")
async def reset_chat_ai_custom(payload: ResetRequest_aicustom):
    state = chat_state_store_aicustom.reset_state(
        payload.web_no,
        payload.member_no
    )

    return {
        "status": "ok",
        "state": state.model_dump(),
        "web_no": payload.web_no,
        "member_no": payload.member_no,
    }