from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import time

from app.utils.debug_state import print_state

from app.modules.ai_coach.schema import ChatRequest_aicoach, ResetRequest_aicaoch
from app.modules.ai_coach.state_store import chat_state_store_aicoach
from app.modules.ai_coach.flow import process_chat_aicoach_stream
from app.modules.ai_coach.constants import FIXED_QUESTIONS
from app.modules.ai_coach.schema import ChatState

router = APIRouter(tags=["AI Coach"])

@router.post("/start/ai-coach")
async def start_ai_coach_stream(req: ChatRequest_aicoach):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    step = 1
    fixed_question = FIXED_QUESTIONS[step]

    state = ChatState(
        step=0,
        fixed_question=fixed_question,
    )

    chat_state_store_aicoach.set_state(req.web_no, req.member_no, state)

    print(f"[ROUTE] /start/ai-coach/stream START {time.time():.3f}", flush=True)
    print_state("BEFORE STATE", state)

    async def event_generator():
        stream_start = time.time()
        final_reply = ""
        final_state = state
        final_source = "debug_chat"
        chunk_count = 0

        print(f"[STREAM] generator START {stream_start:.3f}", flush=True)

        try:
            async for item in process_chat_aicoach_stream(req, state):
                item_type = item.get("type")

                if item_type == "chunk":
                    text = item.get("text", "")
                    if text:
                        final_reply += text
                        chunk_count += 1

                    payload = json.dumps({
                        "type": "chunk",
                        "text": text,
                    }, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

                elif item_type == "done":
                    final_reply = item.get("reply", final_reply) or final_reply
                    final_state = item.get("state", final_state) or final_state
                    final_source = item.get("source", final_source) or final_source

                    chat_state_store_aicoach.set_state(req.web_no, req.member_no, final_state)

                    payload = json.dumps({
                        "type": "done",
                        "reply": final_reply,
                        "status": item.get("status"),
                        "reason": item.get("reason"),
                        "confidence": item.get("confidence"),
                        "state": final_state.model_dump() if hasattr(final_state, "model_dump") else final_state.dict() if hasattr(final_state, "dict") else None,
                        "source": final_source,
                    }, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    return

                elif item_type == "error":
                    payload = json.dumps({
                        "type": "error",
                        "message": item.get("message", "Unknown error")
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
                f"[STREAM] FINALLY total_chunks={chunk_count} total_reply_len={len(final_reply)} total_time={time.time() - stream_start:.3f}s",
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

@router.post("/chat/ai-coach")
async def chat_ai_coach_stream(req: ChatRequest_aicoach):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    if req.state:
        state = req.state
    else:
        state = chat_state_store_aicoach.get_state(req.web_no, req.member_no)

    print(f"[ROUTE] /chat/ai-coach/stream START {time.time():.3f}", flush=True)
    print_state("BEFORE STATE", state)

    async def event_generator():
        stream_start = time.time()
        final_reply = ""
        final_state = state
        final_source = "debug_chat"
        chunk_count = 0

        print(f"[STREAM] generator START {stream_start:.3f}", flush=True)

        try:
            async for item in process_chat_aicoach_stream(req, state):
                item_type = item.get("type")

                if item_type == "chunk":
                    text = item.get("text", "")
                    if text:
                        final_reply += text
                        chunk_count += 1

                    payload = json.dumps({
                        "type": "chunk",
                        "text": text,
                    }, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

                elif item_type == "done":
                    final_reply = item.get("reply", final_reply) or final_reply
                    final_state = item.get("state", final_state) or final_state
                    final_source = item.get("source", final_source) or final_source

                    chat_state_store_aicoach.set_state(req.web_no, req.member_no, final_state)

                    payload = json.dumps({
                        "type": "done",
                        "reply": final_reply,
                        "status": item.get("status"),
                        "reason": item.get("reason"),
                        "confidence": item.get("confidence"),
                        "state": final_state.model_dump() if hasattr(final_state, "model_dump") else final_state.dict() if hasattr(final_state, "dict") else None,
                        "source": final_source,
                    }, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    return

                elif item_type == "error":
                    payload = json.dumps({
                        "type": "error",
                        "message": item.get("message", "Unknown error")
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
                f"[STREAM] FINALLY total_chunks={chunk_count} total_reply_len={len(final_reply)} total_time={time.time() - stream_start:.3f}s",
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

@router.post("/chat/reset/ai-coach")
async def reset_chat(payload: ResetRequest_aicaoch):
    state = chat_state_store_aicoach.reset_state(payload.web_no, payload.member_no)
    return {"status": "ok", "state": state.model_dump()}
