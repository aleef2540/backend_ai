def get_course_data_by_no(conn, course_no: int):
    cur = conn.cursor()

    cur.execute("""
        SELECT
            script,
            course
        FROM ai_data_sl
        WHERE OCourse_no = %s
    """, (course_no,))

    rows = cur.fetchall()
    cur.close()

    return rows