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
    # ใช้ SQL เดิม แต่เปลี่ยน %s เป็น ? (สำหรับ PHP mysqli prepare)
    sql = """
        SELECT
            script,
            course
        FROM ai_data_sl
        WHERE OCourse_no = ?
    """
    # เรียกผ่าน Bridge แทนการใช้ cursor
    rows = run_query_bridge(sql, [course_no])
    
    # แปลงรูปแบบผลลัพธ์ให้เหมือน fetchall() ของเดิม (ที่เป็น list of tuples) 
    # เพื่อให้โค้ดส่วนอื่นไม่ต้องแก้ไข
    if rows and isinstance(rows, list):
        return [(r['script'], r['course']) for r in rows]
    return []