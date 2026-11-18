"""patient_db.py - Persistent SQLite clinical record keeping for patient profiles, arrhythmia episodes, and alert dispatch history."""
import os
import sqlite3
import time
from typing import List, Dict, Optional, Any

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'metadata', 'ecg_clinical_records.db')

def get_connection(db_path: str = DB_FILE) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DB_FILE) -> None:
    """Initialize database tables and seed default patient profile if empty."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                medical_notes TEXT,
                created_at TEXT,
                last_monitored_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arrhythmia_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                timestamp TEXT,
                primary_diagnosis TEXT,
                triggered_rules TEXT,
                hr_bpm REAL,
                burden_v REAL,
                burden_s REAL,
                burden_q REAL,
                rmssd_ms REAL,
                stress_state TEXT,
                telemetry_source TEXT,
                duration_s REAL DEFAULT 30.0,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                timestamp TEXT,
                diagnosis TEXT,
                channels TEXT,
                recipient TEXT,
                status TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)
        conn.commit()

        # Seed initial patient if database is fresh
        cursor.execute("SELECT COUNT(*) as cnt FROM patients")
        if cursor.fetchone()['cnt'] == 0:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            default_patients = [
                ("PT-LIVE", "Live Telemetry Subject", 28, "Other", "AD8232 Hardware telemetry stream", now, now),
                ("PT-101", "John Doe (Frequent PVCs)", 62, "Male", "History of ventricular ectopy and palpitations", now, None),
                ("PT-104", "Jane Smith (SVT)", 45, "Female", "Paroxysmal supraventricular tachycardia monitoring", now, None),
                ("PT-106", "Robert Taylor (Bradycardia)", 71, "Male", "Sinus node dysfunction with resting bradycardia", now, None)
            ]
            cursor.executemany("""
                INSERT INTO patients (patient_id, name, age, gender, medical_notes, created_at, last_monitored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, default_patients)
            conn.commit()

def add_patient(patient_id: str, name: str, age: Optional[int] = None, gender: Optional[str] = None, medical_notes: str = "", db_path: str = DB_FILE) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO patients (patient_id, name, age, gender, medical_notes, created_at, last_monitored_at)
            VALUES (?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM patients WHERE patient_id=?), ?), ?)
        """, (patient_id, name, age, gender, medical_notes, patient_id, now, now))
        conn.commit()
    return get_patient(patient_id, db_path)

def update_patient_active_timestamp(patient_id: str, db_path: str = DB_FILE) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE patients SET last_monitored_at=? WHERE patient_id=?", (now, patient_id))
        conn.commit()

def get_patients(db_path: str = DB_FILE) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.*,
                   COUNT(DISTINCT e.id) as episode_count,
                   COUNT(DISTINCT a.id) as alert_count,
                   MAX(e.timestamp) as last_episode_at
            FROM patients p
            LEFT JOIN arrhythmia_episodes e ON p.patient_id = e.patient_id
            LEFT JOIN alerts_history a ON p.patient_id = a.patient_id
            GROUP BY p.patient_id
            ORDER BY p.last_monitored_at DESC, p.created_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

def get_patient(patient_id: str, db_path: str = DB_FILE) -> Optional[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.*,
                   COUNT(DISTINCT e.id) as episode_count,
                   COUNT(DISTINCT a.id) as alert_count,
                   MAX(e.timestamp) as last_episode_at
            FROM patients p
            LEFT JOIN arrhythmia_episodes e ON p.patient_id = e.patient_id
            LEFT JOIN alerts_history a ON p.patient_id = a.patient_id
            WHERE p.patient_id = ?
            GROUP BY p.patient_id
        """, (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def log_episode(patient_id: str, primary_diagnosis: str, triggered_rules: List[str],
                hr_bpm: float, burden_v: float, burden_s: float, burden_q: float,
                rmssd_ms: float, stress_state: str, telemetry_source: str,
                duration_s: float = 30.0, db_path: str = DB_FILE) -> int:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rules_str = ", ".join(triggered_rules) if isinstance(triggered_rules, list) else str(triggered_rules)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO arrhythmia_episodes 
            (patient_id, timestamp, primary_diagnosis, triggered_rules, hr_bpm, burden_v, burden_s, burden_q, rmssd_ms, stress_state, telemetry_source, duration_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (patient_id, now, primary_diagnosis, rules_str, hr_bpm, burden_v, burden_s, burden_q, rmssd_ms, stress_state, telemetry_source, duration_s))
        ep_id = cursor.lastrowid
        cursor.execute("UPDATE patients SET last_monitored_at=? WHERE patient_id=?", (now, patient_id))
        conn.commit()
        return ep_id

def get_episodes(patient_id: Optional[str] = None, limit: int = 50, db_path: str = DB_FILE) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if patient_id:
            cursor.execute("""
                SELECT * FROM arrhythmia_episodes 
                WHERE patient_id = ? 
                ORDER BY id DESC LIMIT ?
            """, (patient_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM arrhythmia_episodes 
                ORDER BY id DESC LIMIT ?
            """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

def log_alert(patient_id: str, diagnosis: str, channels: str, recipient: str, status: str = "DISPATCHED", db_path: str = DB_FILE) -> int:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts_history (patient_id, timestamp, diagnosis, channels, recipient, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (patient_id, now, diagnosis, channels, recipient, status))
        conn.commit()
        return cursor.lastrowid

def get_alerts(patient_id: Optional[str] = None, limit: int = 50, db_path: str = DB_FILE) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if patient_id:
            cursor.execute("""
                SELECT * FROM alerts_history 
                WHERE patient_id = ? 
                ORDER BY id DESC LIMIT ?
            """, (patient_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM alerts_history 
                ORDER BY id DESC LIMIT ?
            """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
