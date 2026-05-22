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
        from_web,
        user_message,
        ai_reply,
        matched_course_id,
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
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
    """

    params = [
        chat_id,
        "test",
        "test",
        "test",
        user_message,
        ai_reply,
        "test",
        "test",
        status or "",
        reason or "",
        "test",
        "test",
        "test",
        source or "",
    ]

    return run_query_bridge(sql, params)