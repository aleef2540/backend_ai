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
    print("\n===== START get_course_data_by_nos_bridge =====")
    print("INPUT course_nos:", course_nos, type(course_nos))

    if not course_nos:
        print("❌ course_nos ว่าง")
        return []

    # ---------------------------
    # clean input
    # ---------------------------
    clean_course_nos = []
    for x in course_nos:
        try:
            val = int(str(x).strip())
            clean_course_nos.append(val)
        except (ValueError, TypeError):
            print("⚠️ แปลงค่าไม่ได้:", x)

    print("CLEAN course_nos:", clean_course_nos)

    if not clean_course_nos:
        print("❌ ไม่มี course_no ที่ใช้ได้")
        return []

    placeholders = ",".join(["?"] * len(clean_course_nos))
    print("PLACEHOLDERS:", placeholders)

    # ---------------------------
    # 1) ดึง course
    # ---------------------------
    sql_course = f"""
        SELECT OCourse_no, course, script
        FROM ai_data_sl
        WHERE OCourse_no IN ({placeholders})
    """
    print("\n--- QUERY COURSE ---")
    print("SQL:", sql_course)
    print("PARAMS:", clean_course_nos)

    course_rows = run_query_bridge(sql_course, clean_course_nos)

    print("COURSE_ROWS RAW:", course_rows)
    print("COURSE_ROWS TYPE:", type(course_rows))

    if not course_rows:
        print("❌ course_rows ว่าง (query ไม่ได้ผล)")
        return []

    if isinstance(course_rows, list) and len(course_rows) > 0:
        print("COURSE_ROWS[0]:", course_rows[0])
        print("COURSE_ROWS[0] TYPE:", type(course_rows[0]))

    if isinstance(course_rows, list) and "error" in course_rows[0]:
        print("❌ SQL ERROR:", course_rows[0])
        return []

    # ---------------------------
    # build course_map
    # ---------------------------
    course_map = {}
    for r in course_rows:
        if not isinstance(r, dict):
            print("🔥 FOUND NON-DICT COURSE ROW:", r)
            continue

        try:
            c_no = int(r["OCourse_no"])
        except Exception as e:
            print("❌ parse OCourse_no error:", r, e)
            continue

        course_map[c_no] = {
            "course_no": c_no,
            "course_name": r.get("course"),
            "script": r.get("script"),
            "videos": []
        }

    print("COURSE_MAP:", course_map)

    if not course_map:
        print("❌ course_map ว่าง (mapping พัง)")
        return []

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

    print("\n--- QUERY VIDEO ---")
    print("SQL:", sql_vdo)
    print("PARAMS:", clean_course_nos)

    video_rows = run_query_bridge(sql_vdo, clean_course_nos)

    print("VIDEO_ROWS RAW:", video_rows)
    print("VIDEO_ROWS TYPE:", type(video_rows))

    if video_rows and isinstance(video_rows, list):
        print("VIDEO_ROWS[0]:", video_rows[0])
        print("VIDEO_ROWS[0] TYPE:", type(video_rows[0]))

    if video_rows and isinstance(video_rows, list) and "error" in video_rows[0]:
        print("❌ VIDEO SQL ERROR:", video_rows[0])
        return []

    # ---------------------------
    # process video
    # ---------------------------
    if video_rows:
        columns = ["Video_OCourse_no", "Video_part", "Video_name", "Embed_youtube"]

        for v in video_rows:
            print("\nPROCESS VIDEO ROW:", v)

            if isinstance(v, tuple):
                print("⚠️ tuple detected → converting")
                v = dict(zip(columns, v))

            if not isinstance(v, dict):
                print("❌ invalid video row:", v)
                continue

            try:
                v_c_no = int(v.get("Video_OCourse_no"))
            except Exception as e:
                print("❌ parse Video_OCourse_no error:", v, e)
                continue

            print("MATCH course_no:", v_c_no)

            if v_c_no not in course_map:
                print("⚠️ video ไม่มี course_map:", v_c_no)
                continue

            video_item = {
                "video_part": v.get("Video_part"),
                "video_name": v.get("Video_name") or "",
                "video_url": v.get("Embed_youtube") or "",
            }

            course_map[v_c_no]["videos"].append(video_item)
            print("✅ ADD VIDEO:", video_item)

    # ---------------------------
    # 3) build result
    # ---------------------------
    result = []
    for c_no in clean_course_nos:
        if c_no in course_map:
            result.append(course_map[c_no])
        else:
            print("⚠️ course_no ไม่อยู่ใน course_map:", c_no)

    print("\nFINAL RESULT:", result)
    print("===== END =====\n")

    return result