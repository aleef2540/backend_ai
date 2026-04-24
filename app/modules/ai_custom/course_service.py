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