import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import bcrypt


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("SMARTSUM_DB_PATH", BASE_DIR / "database" / "smartsum.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                is_admin INTEGER NOT NULL DEFAULT 0,
                default_method TEXT DEFAULT 'TextRank (Extractive)',
                default_language TEXT DEFAULT 'en',
                default_style TEXT DEFAULT 'Paragraph',
                lite_mode INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                method TEXT NOT NULL,
                style TEXT NOT NULL,
                language TEXT NOT NULL,
                input_type TEXT NOT NULL,
                original_words INTEGER NOT NULL DEFAULT 0,
                summary_words INTEGER NOT NULL DEFAULT 0,
                compression_ratio REAL NOT NULL DEFAULT 0,
                time_saved_min REAL NOT NULL DEFAULT 0,
                summary_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                favorite INTEGER NOT NULL DEFAULT 0,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                summary_id INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK(rating IN (-1, 1)),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (summary_id) REFERENCES summaries(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                summary_id INTEGER,
                source_snippets TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (summary_id) REFERENCES summaries(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_summaries_user_created ON summaries(user_id, created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_summary ON feedback(summary_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_messages(user_id, created_at DESC)"
        )
        conn.commit()
    finally:
        conn.close()


def validate_email(email: str) -> bool:
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        return False
    return True


def validate_password(password: str) -> bool:
    if not password or len(password) < 8:
        return False
    return True


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def get_first_registered_user():
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users ORDER BY id ASC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_user(name: str, email: str, password: str):
    if not name or not name.strip():
        raise ValueError("Name is required.")
    normalized_name = name.strip()
    normalized_email = email.strip().lower()
    if not validate_email(normalized_email):
        raise ValueError("Please enter a valid email address.")
    if not validate_password(password):
        raise ValueError("Password must be at least 8 characters long.")

    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?",
            (normalized_email,),
        ).fetchone()
        if existing:
            raise ValueError("An account with that email already exists.")
        password_hash = hash_password(password)
        is_admin = 0
        first_user = conn.execute("SELECT id FROM users LIMIT 1").fetchone()
        if not first_user:
            is_admin = 1
        cursor = conn.execute(
            """
            INSERT INTO users (name, email, password_hash, is_admin, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (normalized_name, normalized_email, password_hash, is_admin, datetime.now(timezone.utc).isoformat()),
        )
        user_id = cursor.lastrowid
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(user)
    finally:
        conn.close()


def login_user(email: str, password: str):
    normalized_email = email.strip().lower()
    conn = get_connection()
    try:
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (normalized_email,),
        ).fetchone()
        if not user:
            return None
        if not verify_password(password, user["password_hash"]):
            return None
        return dict(user)
    finally:
        conn.close()


def get_user_by_id(user_id: int):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_user_settings(user_id: int, **kwargs):
    if not user_id:
        return None
    updates = []
    values = []
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in {"lite_mode"}:
            updates.append(f"{key} = ?")
            values.append(int(bool(value)))
        else:
            updates.append(f"{key} = ?")
            values.append(value)
    if not updates:
        return get_user_by_id(user_id)
    conn = get_connection()
    try:
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
        conn.execute(query, values + [user_id])
        conn.commit()
        return get_user_by_id(user_id)
    finally:
        conn.close()


def save_summary(user_id: int, *, title, method, style, language, input_type, original_words, summary_words, compression_ratio, time_saved_min, summary_text):
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO summaries (
                user_id, title, method, style, language, input_type,
                original_words, summary_words, compression_ratio, time_saved_min,
                summary_text, created_at, favorite, is_deleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 0, 0)
            """,
            (
                user_id,
                title,
                method,
                style,
                language,
                input_type,
                int(original_words),
                int(summary_words),
                float(compression_ratio),
                float(time_saved_min),
                summary_text,
            ),
        )
        conn.commit()
        return conn.execute("SELECT * FROM summaries WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
    finally:
        conn.close()


def get_user_summaries(user_id: int, *, search=None, method=None, date_filter=None, favorites_only=False):
    conn = get_connection()
    try:
        query = "SELECT * FROM summaries WHERE user_id = ? AND is_deleted = 0"
        params = [user_id]
        if search:
            query += " AND (title LIKE ? OR summary_text LIKE ?)"
            q = f"%{search}%"
            params.extend([q, q])
        if method:
            query += " AND method = ?"
            params.append(method)
        if date_filter:
            query += " AND date(created_at) = date(?)"
            params.append(date_filter)
        if favorites_only:
            query += " AND favorite = 1"
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_summary_by_id(summary_id: int):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM summaries WHERE id = ? AND is_deleted = 0", (summary_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def toggle_favorite(summary_id: int, favorite: bool):
    conn = get_connection()
    try:
        conn.execute("UPDATE summaries SET favorite = ? WHERE id = ?", (1 if favorite else 0, summary_id))
        conn.commit()
        return get_summary_by_id(summary_id)
    finally:
        conn.close()


def delete_summary(summary_id: int):
    conn = get_connection()
    try:
        conn.execute("UPDATE summaries SET is_deleted = 1 WHERE id = ?", (summary_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def save_feedback(user_id: int, summary_id: int, rating: int):
    if rating not in (-1, 1):
        raise ValueError("Feedback rating must be -1 or 1.")
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM feedback WHERE user_id = ? AND summary_id = ?",
            (user_id, summary_id),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE feedback SET rating = ?, created_at = CURRENT_TIMESTAMP WHERE id = ?",
                (rating, existing["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO feedback (user_id, summary_id, rating, created_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (user_id, summary_id, rating),
            )
        conn.commit()
        return True
    finally:
        conn.close()


def get_summary_feedback(summary_id: int):
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT rating FROM feedback WHERE summary_id = ?",
            (summary_id,),
        ).fetchall()
        ratings = [r["rating"] for r in rows]
        positive = sum(1 for r in ratings if r > 0)
        negative = sum(1 for r in ratings if r < 0)
        return {"positive": positive, "negative": negative, "total": len(ratings)}
    finally:
        conn.close()


def save_chat_message(user_id: int, role: str, content: str, summary_id: int = None, source_snippets=None):
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO chat_messages (user_id, role, content, summary_id, source_snippets, created_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (user_id, role, content, summary_id, json.dumps(source_snippets) if source_snippets else None),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def get_chat_messages(user_id: int):
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE user_id = ? ORDER BY id ASC",
            (user_id,),
        ).fetchall()
        msgs = []
        for row in rows:
            item = dict(row)
            if item.get("source_snippets"):
                item["source_snippets"] = json.loads(item["source_snippets"])
            msgs.append(item)
        return msgs
    finally:
        conn.close()


def clear_chat(user_id: int):
    conn = get_connection()
    try:
        conn.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def get_user_stats(user_id: int):
    conn = get_connection()
    try:
        stats = conn.execute(
            """
            SELECT
                COUNT(*) AS total_summaries,
                COALESCE(SUM(original_words), 0) AS total_read_words,
                COALESCE(SUM(summary_words), 0) AS total_summary_words,
                COALESCE(SUM(time_saved_min), 0) AS total_time_saved,
                COALESCE(SUM(CASE WHEN favorite = 1 THEN 1 ELSE 0 END), 0) AS favorite_count
            FROM summaries
            WHERE user_id = ? AND is_deleted = 0
            """,
            (user_id,),
        ).fetchone()
        return dict(stats)
    finally:
        conn.close()


def get_user_summary_dates(user_id: int):
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT date(created_at) AS d, COUNT(*) AS c FROM summaries WHERE user_id = ? AND is_deleted = 0 GROUP BY date(created_at) ORDER BY d ASC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_user_method_usage(user_id: int):
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT method, COUNT(*) AS c FROM summaries WHERE user_id = ? AND is_deleted = 0 GROUP BY method ORDER BY c DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_user_input_usage(user_id: int):
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT input_type, COUNT(*) AS c FROM summaries WHERE user_id = ? AND is_deleted = 0 GROUP BY input_type ORDER BY c DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_favorite_method(user_id: int):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT method, COUNT(*) AS c FROM summaries WHERE user_id = ? AND favorite = 1 AND is_deleted = 0 GROUP BY method ORDER BY c DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return dict(row) if row else {"method": "N/A", "c": 0}
    finally:
        conn.close()


def get_admin_overview():
    conn = get_connection()
    try:
        user_count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        summary_count = conn.execute("SELECT COUNT(*) AS c FROM summaries WHERE is_deleted = 0").fetchone()["c"]
        trend_rows = conn.execute(
            "SELECT date(created_at) AS d, COUNT(*) AS c FROM summaries WHERE is_deleted = 0 GROUP BY date(created_at) ORDER BY d ASC"
        ).fetchall()
        return {
            "total_users": user_count,
            "total_summaries": summary_count,
            "trend": [dict(r) for r in trend_rows],
        }
    finally:
        conn.close()


init_db()
