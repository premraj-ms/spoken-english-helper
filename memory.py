import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_FILE = "tutor_memory.db"

class TutorMemory:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # User profile (level, interests, goals)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profile (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP
                )
            """)
            # Conversation logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    feedback TEXT,
                    created_at TIMESTAMP
                )
            """)
            # Vocabulary memory (words learned / reviewed)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vocabulary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT UNIQUE,
                    definition TEXT,
                    example_sentence TEXT,
                    times_practiced INTEGER DEFAULT 1,
                    mastered BOOLEAN DEFAULT 0,
                    last_practiced TIMESTAMP
                )
            """)
            # Grammar & speaking mistakes memory
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mistakes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_phrase TEXT,
                    corrected_phrase TEXT,
                    explanation TEXT,
                    category TEXT,
                    occurrence_count INTEGER DEFAULT 1,
                    resolved BOOLEAN DEFAULT 0,
                    last_occurred TIMESTAMP
                )
            """)
            conn.commit()

            # Set default profile if empty
            if not self.get_profile_field("proficiency_level"):
                self.set_profile_field("proficiency_level", "Intermediate (B1-B2)")
                self.set_profile_field("learning_goal", "Fluency, natural conversational idioms, and business communication")
                self.set_profile_field("interests", "Technology, travel, everyday conversations, movies")

    def get_profile_field(self, key: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM profile WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def set_profile_field(self, key: str, value: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO profile (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """, (key, value, datetime.now()))
            conn.commit()

    def get_all_profile(self) -> Dict[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM profile")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def add_message(self, role: str, content: str, feedback: Optional[Dict[str, Any]] = None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            fb_json = json.dumps(feedback) if feedback else None
            cursor.execute("""
                INSERT INTO conversations (role, content, feedback, created_at)
                VALUES (?, ?, ?, ?)
            """, (role, content, fb_json, datetime.now()))
            conn.commit()

    def get_recent_history(self, limit: int = 8) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content, feedback, created_at
                FROM conversations
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            history = []
            for r in reversed(rows):
                history.append({
                    "role": r[0],
                    "content": r[1],
                    "feedback": json.loads(r[2]) if r[2] else None,
                    "created_at": r[3]
                })
            return history

    def record_mistake(self, original: str, corrected: str, explanation: str, category: str = "grammar"):
        if not original or not corrected or original.strip().lower() == corrected.strip().lower():
            return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO mistakes (original_phrase, corrected_phrase, explanation, category, last_occurred)
                VALUES (?, ?, ?, ?, ?)
            """, (original, corrected, explanation, category, datetime.now()))
            conn.commit()

    def get_common_mistakes(self, limit: int = 5) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT original_phrase, corrected_phrase, explanation, category, occurrence_count
                FROM mistakes
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [
                {
                    "original": r[0],
                    "corrected": r[1],
                    "explanation": r[2],
                    "category": r[3],
                    "count": r[4]
                }
                for r in rows
            ]

    def add_vocabulary(self, word: str, definition: str, example: str):
        if not word:
            return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO vocabulary (word, definition, example_sentence, last_practiced)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(word) DO UPDATE SET
                    times_practiced = times_practiced + 1,
                    last_practiced = excluded.last_practiced
            """, (word.strip().lower(), definition, example, datetime.now()))
            conn.commit()

    def get_vocabulary_list(self, limit: int = 20) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT word, definition, example_sentence, times_practiced, mastered
                FROM vocabulary
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            return [
                {
                    "word": r[0],
                    "definition": r[1],
                    "example": r[2],
                    "practiced": r[3],
                    "mastered": bool(r[4])
                }
                for r in cursor.fetchall()
            ]

    def get_memory_summary_for_prompt(self) -> str:
        profile = self.get_all_profile()
        mistakes = self.get_common_mistakes(limit=3)
        vocab = self.get_vocabulary_list(limit=5)

        summary_parts = []
        if profile:
            summary_parts.append(f"Student Profile: Level={profile.get('proficiency_level', 'Intermediate')}, Goals={profile.get('learning_goal', 'Conversational Fluency')}, Interests={profile.get('interests', 'Various')}")
        
        if mistakes:
            mistakes_str = "; ".join([f"'{m['original']}' -> '{m['corrected']}' ({m['explanation']})" for m in mistakes])
            summary_parts.append(f"Past Mistakes to Watch Out For: {mistakes_str}")

        if vocab:
            vocab_str = ", ".join([v['word'] for v in vocab])
            summary_parts.append(f"Recent Vocabulary Practiced: {vocab_str}")

        return "\n".join(summary_parts)
