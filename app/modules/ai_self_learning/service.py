def insert_chat_history_aiselflearning(
    conn,
    chat_id: str,
    course_no: int,
    user_message: str,
    ai_reply: str,
    ai_status: str,
    ai_reason: str,
):
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO ai_self_learning_chat_history (
            chat_id,
            OCourse_no,
            user_message,
            ai_reply,
            ai_status,
            ai_reason
        ) VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        chat_id,
        course_no,
        user_message,
        ai_reply,
        ai_status,
        ai_reason,
    ))

    conn.commit()
    cur.close()