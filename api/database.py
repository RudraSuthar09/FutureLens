import sqlite3
import json
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = 'futurelens.db'

def get_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Creates all required tables if they do not exist."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Table: uploads
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    filename TEXT,
                    row_count INTEGER,
                    column_names TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table: forecasts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    forecast_json TEXT,
                    truth_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table: anomalies
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    anomaly_date TEXT,
                    anomaly_value REAL,
                    expected_value REAL,
                    severity TEXT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table: chat_history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_message TEXT,
                    agent_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Table: system_prompts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    system_prompt TEXT,
                    intelligence_card TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing DB: {e}")

def save_upload(session_id: str, filename: str, row_count: int, columns: List[str]) -> bool:
    """Saves upload file metadata to the db."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO uploads (session_id, filename, row_count, column_names) VALUES (?, ?, ?, ?)',
                (session_id, filename, row_count, json.dumps(columns))
            )
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving upload: {e}")
        return False

def save_forecast(session_id: str, forecast_dict: dict, truth_score: float) -> bool:
    """Saves the output of a forecast to the db."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            # remove some un-json-serializable objects (like models) before saving
            clean_forecast = {k: v for k, v in forecast_dict.items() if k not in ['model', 'X']}
            cursor.execute(
                'INSERT INTO forecasts (session_id, forecast_json, truth_score) VALUES (?, ?, ?)',
                (session_id, json.dumps(clean_forecast), truth_score)
            )
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving forecast: {e}")
        return False

def save_anomalies(session_id: str, anomalies_list: List[Dict[str, Any]]) -> bool:
    """Saves a batch of anomalies to the db."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            for anomaly in anomalies_list:
                cursor.execute(
                    '''INSERT INTO anomalies 
                       (session_id, anomaly_date, anomaly_value, expected_value, severity) 
                       VALUES (?, ?, ?, ?, ?)''',
                    (session_id, anomaly['date'], anomaly['actual'], anomaly['expected'], anomaly['severity'])
                )
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving anomalies: {e}")
        return False

def save_chat(session_id: str, message: str, response: str) -> bool:
    """Saves a chat exchange to the db."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO chat_history (session_id, user_message, agent_response) VALUES (?, ?, ?)',
                (session_id, message, response[:400])
            )
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving chat: {e}")
        return False

def get_recent_chat(session_id: str, limit: int = 4):
    """Returns recent chat history formatted for Gemini context."""
    try:
        with get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT user_message, agent_response
                FROM chat_history
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit)
            )

            rows = cursor.fetchall()

        rows = list(reversed(rows))

        history = []
        for row in rows:
            if row["user_message"]:
                history.append({
                    "role": "user",
                    "content": row["user_message"]
                })

            if row["agent_response"]:
                history.append({
                    "role": "assistant",
                    "content": row["agent_response"]
                })

        return history

    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []

def get_forecast(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves forecast results for a session."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT forecast_json FROM forecasts WHERE session_id = ? ORDER BY id DESC LIMIT 1', (session_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row['forecast_json'])
        return None
    except Exception as e:
        logger.error(f"Error getting forecast: {e}")
        return None

def get_anomalies(session_id: str) -> List[Dict[str, Any]]:
    """Retrieves list of anomalies for a session."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT anomaly_date as date, anomaly_value as actual, expected_value as expected, severity FROM anomalies WHERE session_id = ?', (session_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}")
        return []

def get_chat_history(session_id: str, limit: int = None) -> List[Dict[str, Any]]:
    """Retrieves chat history for a session."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            if limit:
                cursor.execute(
                    """SELECT user_message, agent_response
                       FROM chat_history
                       WHERE session_id = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (session_id, limit)
                )
                rows = cursor.fetchall()
                rows = list(reversed(rows))  # oldest first
            else:
                cursor.execute(
                    """SELECT user_message, agent_response
                       FROM chat_history
                       WHERE session_id = ?
                       ORDER BY created_at ASC""",
                    (session_id,)
                )
                rows = cursor.fetchall()
        return [{"user_message": r[0], "agent_response": r[1]} for r in rows]
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return []

def get_recent_uploads(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieves the N most recent uploads."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT session_id, filename, uploaded_at FROM uploads ORDER BY id DESC LIMIT ?', (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error getting recent uploads: {e}")
        return []

def save_system_prompt(session_id: str,
                       system_prompt: str,
                       intelligence_card: dict) -> None:
    """Save pre-built system prompt and intelligence card for a session. Called once per upload."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO system_prompts
                   (session_id, system_prompt, intelligence_card)
                   VALUES (?, ?, ?)""",
                (session_id, system_prompt, json.dumps(intelligence_card))
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving system prompt: {e}")

def get_system_prompt(session_id: str) -> dict | None:
    """Load pre-built system prompt from DB. Returns None if session not found."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT system_prompt, intelligence_card
                   FROM system_prompts
                   WHERE session_id = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (session_id,)
            )
            row = cursor.fetchone()
        if row:
            return {
                "system_prompt": row[0],
                "intelligence_card": json.loads(row[1])
            }
        return None
    except Exception as e:
        logger.error(f"Error getting system prompt: {e}")
        return None
