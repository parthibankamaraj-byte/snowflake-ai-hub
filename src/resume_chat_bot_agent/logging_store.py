from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Tuple


class QuestionLogger:
    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure()

    def _ensure(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    q TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def record_question(self, q: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("INSERT INTO questions(q) VALUES (?)", (q.strip(),))
            conn.commit()

    def get_top_questions(self, limit: int = 10) -> List[Tuple[str, int]]:
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "SELECT q, COUNT(*) as c FROM questions GROUP BY q ORDER BY c DESC, MAX(created_at) DESC LIMIT ?",
                (limit,),
            )
            return [(row[0], int(row[1])) for row in cur.fetchall()]


