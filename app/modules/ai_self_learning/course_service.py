from app.core.database import run_query_bridge # import ฟังก์ชันใหม่มาใช้

# def get_course_data_by_no(conn, course_no: int):
#     cur = conn.cursor()

#     cur.execute("""
#         SELECT  
#             script,
#             course
#         FROM ai_data_sl 
#         WHERE OCourse_no = %s
#     """, (course_no,))

#     rows = cur.fetchall()   
#     cur.close()

#     return rows

def get_course_data_by_no_bridge(course_no: int):
    # เปลี่ยน %s เป็น ?
    sql = """
        SELECT script, course
        FROM ai_data_sl
        WHERE OCourse_no = ?
    """
    rows = run_query_bridge(sql, [course_no])
    
    # แปลงข้อมูลกลับเป็น list of tuples เพื่อให้กระทบ code ส่วนอื่นน้อยที่สุด
    if rows and isinstance(rows, list):
        if len(rows) > 0 and "error" in rows[0]:
            print(f"SQL Error: {rows[0]['error']}")
            return []
        return [(r['script'], r['course']) for r in rows]
    return []