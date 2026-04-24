from __future__ import annotations

import os
from typing import Any, Optional
import httpx
import json

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from decimal import Decimal, ROUND_HALF_UP
from openai import AsyncOpenAI

MODEL_PRICING = {
    "gpt-4.1-mini": {
        "input_per_1m": Decimal("0.40"),
        "output_per_1m": Decimal("1.60"),
    },
    "gpt-4.1": {
        "input_per_1m": Decimal("2.00"),
        "output_per_1m": Decimal("8.00"),
    },
    "gpt-4.1-nano": {
        "input_per_1m": Decimal("0.10"),
        "output_per_1m": Decimal("0.40"),
    },
    "text-embedding-3-large": {
        "input_per_1m": Decimal("0.13"),
        "output_per_1m": Decimal("0.00"),
    },
}

def q4(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    usd_to_thb: Decimal = Decimal("36.0"),
):
    pricing = MODEL_PRICING.get(model)

    if not pricing:
        return {
            "total_cost_usd": 0.0,
            "total_cost_thb": 0.0,
        }

    input_cost = (Decimal(prompt_tokens) / Decimal("1000000")) * pricing["input_per_1m"]
    output_cost = (Decimal(completion_tokens) / Decimal("1000000")) * pricing["output_per_1m"]

    total_cost = input_cost + output_cost
    total_cost_thb = total_cost * usd_to_thb

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "input_cost_usd": float(q4(input_cost)),
        "output_cost_usd": float(q4(output_cost)),
        "total_cost_usd": float(q4(total_cost)),
        "total_cost_thb": float(q4(total_cost_thb)),
    }



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)
clients = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def call_openai_chat(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    extra_payload: Optional[dict[str, Any]] = None,
) -> str:
    """
    เรียก OpenAI Chat Completion และคืนเฉพาะ content กลับไปใช้งาน

    ใช้สำหรับ:
    - reply_greeting
    - reply_general
    - detect_intent
    - generate_question
    - generate_final_reply
    ฯลฯ

    ตัวอย่าง:
        content = await call_openai_chat(
            model="gpt-4.1-mini",
            system_prompt="คุณคือผู้ช่วย",
            user_prompt="สวัสดี",
            temperature=0.4,
        )
    """
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if extra_payload:
        payload.update(extra_payload)

    response = client.chat.completions.create(**payload)
    content = response.choices[0].message.content or ""

    return content.strip()


async def call_openai_chat_full(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or ""

    usage = getattr(response, "usage", None)

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    cost = calculate_cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    
    # 🔥 print แค่ cost
    print(f"[TOKEN] total={cost['total_tokens']}")
    print(f"         prompt={cost['prompt_tokens']} | completion={cost['completion_tokens']}")
    print(f"[COST]  ${cost['total_cost_usd']} (~{cost['total_cost_thb']} บาท)")


    return {
        "content": content.strip(),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cost": cost,
        "raw": response.model_dump(),
    }


async def call_openai_embedding(
    *,
    model: str,
    input_text: str,
    extra_payload: Optional[dict[str, Any]] = None,
) -> list[float]:
    """
    เรียก OpenAI Embedding และคืน vector กลับไป

    ใช้สำหรับ:
    - get_embedding_python
    - vector search
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": input_text,
    }

    if extra_payload:
        payload.update(extra_payload)

    response = client.embeddings.create(**payload)
    embedding = response.data[0].embedding

    return embedding


async def call_openai_embedding_full(
    *,
    model: str,
    input_text: str,
    extra_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:

    payload: dict[str, Any] = {
        "model": model,
        "input": input_text,
    }

    if extra_payload:
        payload.update(extra_payload)

    response = client.embeddings.create(**payload)
    embedding = response.data[0].embedding

    # =========================
    # usage
    # =========================
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0

    # embedding ไม่มี completion
    completion_tokens = 0

    # =========================
    # cost
    # =========================
    cost = calculate_cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    # =========================
    # debug (เหมือน chat)
    # =========================
    print(
        f"[EMBED] tokens={cost['total_tokens']} "
        f"(prompt={cost['prompt_tokens']}) | "
        f"${cost['total_cost_usd']} (~{cost['total_cost_thb']} บาท)"
    )

    return {
        "embedding": embedding,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        },
        "cost": cost,
        "raw": response.model_dump(),
    }

async def call_openai_chat_stream_full(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    extra_payload: Optional[dict[str, Any]] = None,
):
    """
    stream ข้อความทีละ chunk
    และเมื่อจบแล้วจะ yield event ปิดท้ายเป็น dict
    """

    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if extra_payload:
        payload.update(extra_payload)

    full_content = ""
    prompt_tokens = 0
    completion_tokens = 0

    stream = await clients.chat.completions.create(**payload)

    async for chunk in stream:
        try:
            if getattr(chunk, "usage", None):
                prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta
            text = delta.content or ""

            if text:
                full_content += text
                yield {
                    "type": "chunk",
                    "text": text,
                }
        except Exception as e:
            print("OPENAI STREAM ERROR =", repr(e), flush=True)
            continue

    cost = calculate_cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    print(f"[TOKEN] total={cost['total_tokens']}")
    print(f"         prompt={cost['prompt_tokens']} | completion={cost['completion_tokens']}")
    print(f"[COST]  ${cost['total_cost_usd']} (~{cost['total_cost_thb']} บาท)")

    yield {
        "type": "done",
        "content": full_content.strip(),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cost": cost,
    }
    