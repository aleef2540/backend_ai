import json
from app.core.database import run_query_bridge


def insert_ai_sale_chat_log_bridge(
    *,
    chat_id: str,
    user_message: str,
    ai_reply: str,
    state,
    status: str = "",
    reason: str = "",
    source: str = "",
):
    sql = """
    INSERT INTO ai_sale_chat_log
    (
        chat_id,
        web_no,
        member_no,
        user_message,
        ai_reply,
        mode,
        status,
        reason,
        requirements,
        recommended_courses,
        search_query,
        source,
        created_at
    )
    VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
    """

    params = [
        chat_id,
        getattr(state, "web_no", None),
        getattr(state, "member_no", None),
        user_message,
        ai_reply,
        getattr(state, "mode", ""),
        status or "",
        reason or "",
        json.dumps(getattr(state, "requirements", {}) or {}, ensure_ascii=False),
        json.dumps(getattr(state, "recommended_course_cta", []) or [], ensure_ascii=False),
        getattr(state, "search_query", "") or "",
        source or "",
    ]

    return run_query_bridge(sql, params)