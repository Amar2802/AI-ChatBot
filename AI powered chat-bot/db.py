import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
from config import DB_PATH, BASE_DIR
SCHEMA_PATH = BASE_DIR / "schema.sql"
class DB:
    def __init__(self, path: Path = DB_PATH):
        self.path = str(path)
        self._bootstrap()

    def _connect(self):
        return sqlite3.connect(self.path, check_same_thread=False)

    def _bootstrap(self):
        conn = self._connect()
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
        conn.close()

    # Session ops
    def ensure_session(self, session_id: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)",
                    (session_id,))
        conn.commit()
        conn.close()

    def add_message(self, session_id: str, role: str, text: str) -> int:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages(session_id, role, text) VALUES (?,?,?)",
            (session_id, role, text),
        )
        conn.commit()
        mid = cur.lastrowid
        conn.close()
        return mid

    def get_recent_messages(self, session_id: str, limit: int) -> List[Tuple[str, str]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT role, text FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit * 2),
        )
        rows = cur.fetchall()
        conn.close()
        rows.reverse()
        return rows

    def log_feedback(self, session_id: str, message_id: int, rating: int, note: Optional[str]):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO feedback(session_id, message_id, rating, note) VALUES (?,?,?,?)",
            (session_id, message_id, rating, note),
        )
        conn.commit()
        conn.close()

    # FAQ ops
    def insert_faqs(self, pairs: Iterable[Tuple[str, str]]):
        conn = self._connect()
        cur = conn.cursor()
        cur.executemany("INSERT INTO faqs(question, answer) VALUES (?,?)",
                        list(pairs))
        conn.commit()
        conn.close()

    def fetch_faqs(self) -> List[Tuple[int, str, str]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT id, question, answer FROM faqs")
        rows = cur.fetchall()
        conn.close()
        return rows

    def all_messages(self) -> List[Tuple]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM messages ORDER BY created_at DESC")
        rows = cur.fetchall()
        conn.close()
        return rows


