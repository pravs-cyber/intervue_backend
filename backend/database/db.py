import sqlite3
from contextlib import contextmanager

import config
from database.question_bank import HR, TECHNICAL
from services.constants import LEVELS

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    target_role TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS interviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    type TEXT NOT NULL,                 -- technical | hr | resume
    score REAL,                         -- overall score, set when finished
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    -- session state (kept on the interview row to stay simple)
    status TEXT DEFAULT 'in_progress',  -- in_progress | completed
    total_questions INTEGER DEFAULT 10,
    topics TEXT DEFAULT '[]',           -- JSON list of skills to rotate through
    current_question_id INTEGER,
    question_queue TEXT DEFAULT '[]',   -- JSON list of question ids (resume interviews)
    resume_text TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    skill TEXT NOT NULL,
    difficulty TEXT NOT NULL,           -- Easy | Medium | Hard
    type TEXT NOT NULL,                 -- technical | hr | resume
    source TEXT DEFAULT 'bank',         -- bank | llm | template
    reference TEXT                      -- optional reference answer (enables the reference-aware model)
);
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id INTEGER NOT NULL REFERENCES interviews(id),
    question_id INTEGER NOT NULL REFERENCES questions(id),
    answer_text TEXT NOT NULL,
    ml_prediction TEXT NOT NULL,        -- Poor | Average | Good | Excellent
    score REAL NOT NULL,
    feedback TEXT                       -- JSON string
);
CREATE TABLE IF NOT EXISTS skill_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    skill TEXT NOT NULL,
    score REAL NOT NULL,                -- running average
    attempts INTEGER NOT NULL DEFAULT 0,
    UNIQUE(user_id, skill)
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)  # read at call time so tests can point it elsewhere
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(questions)")}
        if "reference" not in cols:                       # migrate databases created before v2
            conn.execute("ALTER TABLE questions ADD COLUMN reference TEXT")
        if conn.execute("SELECT COUNT(*) FROM questions WHERE source='bank'").fetchone()[0] == 0:
            for qtype, bank in (("technical", TECHNICAL), ("hr", HR)):
                conn.executemany(
                    "INSERT INTO questions (question, skill, difficulty, type, source) VALUES (?,?,?,?, 'bank')",
                    [(q, skill, LEVELS[d], qtype) for skill, d, q in bank],
                )
        _apply_reviewed_references(conn)


def _apply_reviewed_references(conn):
    """Approved, human-reviewed references for bank questions (data/sources/bank_references.csv), if present."""
    try:
        from ml.fill_bank_references import CSV, apply
        if CSV.exists():
            apply(conn)
    except Exception as e:  # a missing or malformed review file must never stop the app
        print(f"note: bank references not applied ({e})")
