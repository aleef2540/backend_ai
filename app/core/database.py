import json
import cloudscraper
import base64 # เพิ่ม import base64
import mysql.connector
from typing import Optional

# from app.schemas_aiweb import ChatState_aiweb
# URL ของ PHP Bridge (เปลี่ยนเป็น IP ของเครื่อง XAMPP หรือ URL ของ ngrok)
PHP_BRIDGE_URL = "https://www.entraining.net/2018/api_bridge.php"
BRIDGE_KEY = "1234"

def get_mysql_connection():
    return mysql.connector.connect(
        host="entstaffs.entraining.net",
        user="entraini1_entrain",
        password="Ent.Pw78x.@a27df!z88",
        database="entraini1_entrainingdb",
        charset="utf8mb4",
    )

def run_query_bridge(sql: str, params: list = None):
    sql_encoded = base64.b64encode(sql.encode('utf-8')).decode('utf-8')
    payload = {
        'key': BRIDGE_KEY,
        'query': sql_encoded,
        'params': json.dumps(params) if params else json.dumps([])
    }
    
    try:
        # ใช้ cloudscraper สร้าง session แทน requests ปกติ
        scraper = cloudscraper.create_scraper() 
        response = scraper.post(PHP_BRIDGE_URL, data=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Bridge Connection Error: {e}")
        return None


# def ensure_chat_session(
#     conn,
#     chat_id: str,
#     ip_address: Optional[str] = None,
#     user_agent: Optional[str] = None,
# ):
#     sql = """
#         INSERT INTO chat_sessions (chat_id, state_json, ip_address, user_agent, created_at, updated_at)
#         VALUES (%s, NULL, %s, %s, NOW(), NOW())
#         ON DUPLICATE KEY UPDATE
#             ip_address = VALUES(ip_address),
#             user_agent = VALUES(user_agent),
#             updated_at = NOW()
#     """
#     cur = conn.cursor()
#     cur.execute(sql, (chat_id, ip_address, user_agent))
#     conn.commit()
#     cur.close()

# def load_chat_state(conn, chat_id: str) -> ChatState_aiweb:
#     sql = "SELECT state_json FROM chat_sessions WHERE chat_id = %s LIMIT 1"
#     cur = conn.cursor(dictionary=True)
#     cur.execute(sql, (chat_id,))
#     row = cur.fetchone()
#     cur.close()

#     if not row or not row.get("state_json"):
#         return ChatState_aiweb()

#     raw = row["state_json"]

#     if isinstance(raw, str):
#         try:
#             raw = json.loads(raw)
#         except Exception:
#             return ChatState_aiweb()

#     try:
#         return ChatState_aiweb(**raw)
#     except Exception:
#         return ChatState_aiweb()

# def save_chat_state(conn, chat_id: str, state: ChatState_aiweb):
#     sql = """
#         UPDATE chat_sessions
#         SET state_json = %s,
#             updated_at = NOW()
#         WHERE chat_id = %s
#     """
#     cur = conn.cursor()
#     cur.execute(
#         sql,
#         (json.dumps(state.model_dump(), ensure_ascii=False), chat_id),
#     )
#     conn.commit()
#     cur.close()

# def insert_chat_message(conn, chat_id: str, sender_type: str, message_text: str):
#     sql = """
#         INSERT INTO chat_messages (chat_id, sender_type, message_text, created_at)
#         VALUES (%s, %s, %s, NOW())
#     """
#     cur = conn.cursor()
#     cur.execute(sql, (chat_id, sender_type, message_text))
#     conn.commit()
#     cur.close()

# def insert_request_log(conn, chat_id: str, ip_address: Optional[str] = None):
#     sql = """
#         INSERT INTO chat_request_logs (chat_id, ip_address, created_at)
#         VALUES (%s, %s, NOW())
#     """
#     cur = conn.cursor()
#     cur.execute(sql, (chat_id, ip_address))
#     conn.commit()
#     cur.close()

# def reset_chat_state(conn, chat_id: str):
    # sql1 = """
    #     UPDATE chat_sessions
    #     SET state_json = %s,
    #         updated_at = NOW()
    #     WHERE chat_id = %s
    # """
    # sql2 = "DELETE FROM chat_messages WHERE chat_id = %s"

    # cur = conn.cursor()
    # cur.execute(
    #     sql1,
    #     (json.dumps(ChatState_aiweb().model_dump(), ensure_ascii=False), chat_id),
    # )
    # cur.execute(sql2, (chat_id,))
    # conn.commit()
    # cur.close()