from app.core.database import run_query_bridge # import ฟังก์ชันใหม่มาใช้

def get_course_data_by_nos(conn, course_nos: list[str]):
    if not course_nos:
        return []

    clean_course_nos = []
    for x in course_nos:
        x = str(x).strip()
        if x:
            clean_course_nos.append(x)

    if not clean_course_nos:
        return []

    placeholders = ",".join(["%s"] * len(clean_course_nos))

    cur = conn.cursor(dictionary=True)

    # ---------------------------------
    # 1) ดึงข้อมูล course ก่อน
    # ---------------------------------
    sql_course = f"""
        SELECT
            OCourse_no,
            course,
            script
        FROM ai_data_sl
        WHERE OCourse_no IN ({placeholders})
    """
    cur.execute(sql_course, tuple(clean_course_nos))
    course_rows = cur.fetchall()

    if not course_rows:
        cur.close()
        return []

    # ทำ map เก็บ course ไว้ก่อน
    course_map = {}
    for row in course_rows:
        course_no = int(row["OCourse_no"])
        course_map[course_no] = {
            "course_no": course_no,
            "course_name": row.get("course") or "",
            "script": row.get("script") or "",
            "videos": []
        }

    # ---------------------------------
    # 2) ดึงวิดีโอของทุก course อีกรอบ
    # ---------------------------------
    sql_vdo = f"""
        SELECT
            Video_OCourse_no,
            Video_part,
            Video_name,
            Embed_youtube
        FROM course_online_vdo
        WHERE Video_OCourse_no IN ({placeholders})
          AND Video_type = 'VDO'
          AND Video_name != ''
        ORDER BY Video_OCourse_no ASC, Video_part ASC
    """
    cur.execute(sql_vdo, tuple(clean_course_nos))
    video_rows = cur.fetchall()
    cur.close()

    # เอา videos ไปยัดกลับเข้า course_map
    for v in video_rows:
        course_no = int(v["Video_OCourse_no"])

        if course_no in course_map:
            course_map[course_no]["videos"].append({
                "video_part": v.get("Video_part"),
                "video_name": v.get("Video_name") or "",
                "video_url": v.get("Embed_youtube") or "",
            })

    # ---------------------------------
    # 3) คืนค่าตามลำดับ course_nos เดิม
    # ---------------------------------
    result = []
    for cno in clean_course_nos:
        cno_int = int(cno)
        if cno_int in course_map:
            result.append(course_map[cno_int])

    return result


def get_course_data_by_nos_bridge(course_nos: list):
    if not course_nos:
        return []

    # clean input
    clean_course_nos = []
    for x in course_nos:
        try:
            clean_course_nos.append(int(str(x).strip()))
        except (ValueError, TypeError):
            continue

    if not clean_course_nos:
        return []

    placeholders = ",".join(["?"] * len(clean_course_nos))

    # ---------------------------
    # 1) ดึง course
    # ---------------------------
    sql_course = f"""
        SELECT OCourse_no, course, script
        FROM ai_data_sl
        WHERE OCourse_no IN ({placeholders})
    """
    course_rows = run_query_bridge(sql_course, clean_course_nos)

    if not course_rows or (isinstance(course_rows, list) and "error" in course_rows[0]):
        return []

    # 🔥 ใช้ dict เป็นหลัก (เลิก tuple)
    course_map = {}
    for r in course_rows:
        if not isinstance(r, dict):
            continue

        c_no = int(r["OCourse_no"])
        course_map[c_no] = {
            "course_no": c_no,
            "course": r.get("course"),
            "script": r.get("script"),
            "videos": []
        }

    # ---------------------------
    # 2) ดึง video
    # ---------------------------
    sql_vdo = f"""
        SELECT Video_OCourse_no, Video_part, Video_name, Embed_youtube
        FROM course_online_vdo
        WHERE Video_OCourse_no IN ({placeholders})
          AND Video_type = 'VDO'
          AND Video_name != ''
        ORDER BY Video_OCourse_no ASC, Video_part ASC
    """
    video_rows = run_query_bridge(sql_vdo, clean_course_nos)

    if video_rows and not (isinstance(video_rows, list) and "error" in video_rows[0]):

        # 🔥 กัน tuple ตรงนี้
        columns = ["Video_OCourse_no", "Video_part", "Video_name", "Embed_youtube"]

        for v in video_rows:
            if isinstance(v, tuple):
                v = dict(zip(columns, v))
            elif not isinstance(v, dict):
                continue

            try:
                v_c_no = int(v.get("Video_OCourse_no"))
            except:
                continue

            if v_c_no not in course_map:
                continue

            course_map[v_c_no]["videos"].append({
                "video_part": v.get("Video_part"),
                "video_name": v.get("Video_name") or "",
                "video_url": v.get("Embed_youtube") or "",
            })

    # ---------------------------
    # 3) คืนค่า (เรียงตาม input เดิม)
    # ---------------------------
    result = []
    for c_no in clean_course_nos:
        if c_no in course_map:
            result.append(course_map[c_no])

    return result