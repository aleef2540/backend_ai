from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import time
import traceback

from app.modules.ai_sale.schema import AISaleRequest, AISaleResetRequest
from app.modules.ai_sale.state_store import ai_sale_state_store
from app.modules.ai_sale.flow import process_ai_sale_stream

router = APIRouter(tags=["AI Sale"])


@router.post("/chat/ai-sale")
async def chat_ai_sale_stream(req: AISaleRequest):
    print("🔥 HIT /chat/ai-sale", flush=True)
    # print("REQ =", req.model_dump(), flush=True)

    if not req.user_message or not req.user_message.strip():
        print("❌ user_message is empty", flush=True)
        raise HTTPException(status_code=400, detail="user_message is required")

    req.user_message = req.user_message.strip()

    if req.state:
        state = req.state
        print("✅ USE STATE FROM REQUEST", flush=True)
    else:
        state = ai_sale_state_store.get_state(req.web_no, req.member_no)
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
            async for item in process_ai_sale_stream(req, state):
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
                        req.web_no,
                        req.member_no,
                        final_state
                    )

                    payload = json.dumps({
                        "type": "done",
                        "reply": final_reply,
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


@router.post("/chat/reset/ai-sale")
async def reset_ai_sale(payload: AISaleResetRequest):
    print("🔥 HIT /chat/reset/ai-sale", flush=True)
    print("RESET PAYLOAD =", payload.model_dump(), flush=True)

    state = ai_sale_state_store.reset_state(
        payload.web_no,
        payload.member_no
    )

    return {
        "status": "ok",
        "state": state.model_dump(),
        "web_no": payload.web_no,
        "member_no": payload.member_no,
    }