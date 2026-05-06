from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import time
import traceback

from app.modules.ai_assis.schema import AISaleRequest, AISaleResetRequest
from app.modules.ai_assis.state_store import ai_sale_state_store
from app.modules.ai_assis.flow import process_ai_assistant_stream
from app.modules.ai_assis.log_bridge import insert_ai_sale_chat_log_bridge

router = APIRouter(tags=["AI Assis"])


@router.post("/chat/ai-assis")
async def chat_ai_sale_stream(req: AISaleRequest):
    print("🔥 HIT /chat/ai-assis", flush=True)
    # print("REQ =", req.model_dump(), flush=True)

    if not req.user_message or not req.user_message.strip():
        print("❌ user_message is empty", flush=True)
        raise HTTPException(status_code=400, detail="user_message is required")

    req.user_message = req.user_message.strip()

    state = ai_sale_state_store.get_state(req.chat_id)
    print("✅ USE STATE FROM STORE", flush=True)

    # print("STATE BEFORE =", state.model_dump(), flush=True)

    async def event_generator():
        print("🔥 STREAM START", flush=True)

        stream_start = time.time()
        final_reply = ""
        final_state = state
        final_source = "ai_sale"
        chunk_count = 0

        try:
            async for item in process_ai_assistant_stream(req, state):
                print("STREAM ITEM =", item.get("type"), flush=True)

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

                    print("✅ STREAM DONE", flush=True)
                    print("FINAL STATE =", final_state.model_dump(), flush=True)

                    ai_sale_state_store.set_state(
                        req.chat_id,
                        final_state
                    )

                    # ✅ บันทึก log ลง DB ผ่าน PHP bridge
                    # log_result = insert_ai_sale_chat_log_bridge(
                    #     chat_id=req.chat_id,
                    #     user_message=req.user_message,
                    #     ai_reply=final_reply,
                    #     state=final_state,
                    #     status=item.get("status", ""),
                    #     reason=item.get("reason", ""),
                    #     source=final_source,
                    # )

                    # print("✅ AI SALE LOG RESULT =", log_result, flush=True)

                    courses = item.get("courses", [])
                    payload = json.dumps({
                        "type": "done",
                        "reply": final_reply,
                        "courses": courses,
                        "state": final_state.model_dump() if hasattr(final_state, "model_dump") else None,
                        "source": final_source,
                    }, ensure_ascii=False)

                    yield f"data: {payload}\n\n"
                    return

        except Exception as e:
            print("🔥 AI SALE STREAM ERROR:", repr(e), flush=True)
            traceback.print_exc()

            payload = json.dumps({
                "type": "error",
                "message": str(e)
            }, ensure_ascii=False)

            yield f"data: {payload}\n\n"

        finally:
            print(
                f"[AI SALE STREAM FINALLY] chunks={chunk_count} "
                f"reply_len={len(final_reply)} "
                f"time={time.time() - stream_start:.3f}s",
                flush=True
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/reset/ai-assis")
async def reset_ai_sale(payload: AISaleResetRequest):
    print("🔥 HIT /chat/reset/ai-assis", flush=True)
    print("RESET PAYLOAD =", payload.model_dump(), flush=True)

    state = ai_sale_state_store.reset_state(
    payload.chat_id
    )

    return {
        "status": "ok",
        "state": state.model_dump(),
        "chat_id": payload.chat_id,
    }