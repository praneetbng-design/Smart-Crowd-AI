"""
Smart Crowd AI: Database Management Module
Handles persistent SQLite storage for detections, alerts, and venue zone configurations.
"""

import sqlite3
import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "crowd_data.db")


def get_connection() -> sqlite3.Connection:
    """Returns a thread-safe connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes database tables and default venue zone data if not present."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        source_type TEXT NOT NULL,
        source_name TEXT,
        person_count INTEGER NOT NULL,
        density REAL NOT NULL,
        risk_level TEXT NOT NULL,
        risk_score REAL NOT NULL,
        is_simulated INTEGER NOT NULL DEFAULT 0,
        venue_zone TEXT,
        processing_time_ms REAL DEFAULT 0,
        details_json TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        zone_name TEXT NOT NULL,
        alert_level TEXT NOT NULL,
        trigger_count INTEGER NOT NULL,
        density REAL NOT NULL,
        message TEXT NOT NULL,
        advisory TEXT NOT NULL,
        acknowledged INTEGER NOT NULL DEFAULT 0
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS venue_zones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        venue_name TEXT NOT NULL,
        zone_name TEXT NOT NULL,
        area_sqm REAL NOT NULL,
        max_capacity INTEGER NOT NULL,
        warning_threshold REAL NOT NULL,
        critical_threshold REAL NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        current_count INTEGER DEFAULT 0,
        current_risk TEXT DEFAULT 'SAFE'
    )
    """)

    conn.commit()

    # Seed default zones if empty
    cursor.execute("SELECT COUNT(*) FROM venue_zones")
    count = cursor.fetchone()[0]
    if count == 0:
        seed_zones = [
            ("Central Mega Arena", "Gate 1 - North Plaza", 450.0, 900, 1.6, 2.5, 40.7128, -74.0060, 240, "SAFE"),
            ("Central Mega Arena", "Gate 2 - East Concourse", 300.0, 600, 1.5, 2.4, 40.7135, -74.0045, 410, "WARNING"),
            ("Central Mega Arena", "Section A - Lower Bowl", 600.0, 1500, 1.8, 2.8, 40.7120, -74.0055, 1180, "CRITICAL"),
            ("Central Mega Arena", "Exit Corridor South", 180.0, 350, 1.4, 2.2, 40.7112, -74.0068, 95, "SAFE"),
            ("Grand Transit Terminal", "Main Concourse Hall", 1200.0, 2500, 1.5, 2.6, 40.7527, -73.9772, 850, "SAFE"),
            ("Grand Transit Terminal", "Subway Turnstile B", 220.0, 500, 1.8, 2.7, 40.7532, -73.9765, 460, "CRITICAL"),
            ("Grand Transit Terminal", "Platform 4 Express", 350.0, 700, 1.5, 2.3, 40.7521, -73.9780, 520, "WARNING"),
            ("Pilgrimage Sacred Square", "Main Sanctuary Steps", 800.0, 2000, 1.6, 2.7, 21.4225, 39.8262, 1420, "WARNING"),
            ("Pilgrimage Sacred Square", "Bottleneck Bridge East", 250.0, 500, 1.4, 2.2, 21.4235, 39.8280, 480, "CRITICAL"),
            ("Festival Parklands", "Main Stage Lawn", 2000.0, 5000, 1.8, 2.8, 51.5074, -0.1278, 2200, "SAFE"),
            ("Festival Parklands", "Food Court Bottleneck", 300.0, 600, 1.5, 2.4, 51.5085, -0.1290, 510, "WARNING"),
        ]
        cursor.executemany("""
        INSERT INTO venue_zones (
            venue_name, zone_name, area_sqm, max_capacity, warning_threshold, critical_threshold, lat, lon, current_count, current_risk
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seed_zones)
        conn.commit()

    conn.close()


def insert_detection(
    source_type: str,
    person_count: int,
    density: float,
    risk_level: str,
    risk_score: float,
    source_name: Optional[str] = "Uploaded Media",
    venue_zone: Optional[str] = "General Zone",
    is_simulated: bool = False,
    processing_time_ms: float = 0.0,
    details: Optional[Dict[str, Any]] = None
) -> int:
    """Records an AI or simulated detection entry."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    details_str = json.dumps(details) if details else "{}"

    cursor.execute("""
    INSERT INTO detections (
        timestamp, source_type, source_name, person_count, density, risk_level, risk_score,
        is_simulated, venue_zone, processing_time_ms, details_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str, source_type, source_name, person_count, density, risk_level, risk_score,
        1 if is_simulated else 0, venue_zone, processing_time_ms, details_str
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def insert_alert(
    zone_name: str,
    alert_level: str,
    trigger_count: int,
    density: float,
    message: str,
    advisory: str
) -> int:
    """Logs an automatic risk alert."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO alerts (
        timestamp, zone_name, alert_level, trigger_count, density, message, advisory, acknowledged
    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
    """, (now_str, zone_name, alert_level, trigger_count, density, message, advisory))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_recent_detections(limit: int = 100, filter_source: Optional[str] = None, filter_risk: Optional[str] = None) -> pd.DataFrame:
    """Fetches recent detections as a pandas DataFrame."""
    conn = get_connection()
    query = "SELECT * FROM detections WHERE 1=1"
    params = []

    if filter_source and filter_source != "All":
        query += " AND source_type = ?"
        params.append(filter_source)

    if filter_risk and filter_risk != "All":
        query += " AND risk_level = ?"
        params.append(filter_risk)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_recent_alerts(limit: int = 50, unacknowledged_only: bool = False) -> pd.DataFrame:
    """Fetches logged alerts."""
    conn = get_connection()
    query = "SELECT * FROM alerts"
    params = []

    if unacknowledged_only:
        query += " WHERE acknowledged = 0"

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def acknowledge_alert(alert_id: int):
    """Marks a specific alert as acknowledged."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def acknowledge_all_alerts():
    """Marks all alerts as acknowledged."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE alerts SET acknowledged = 1 WHERE acknowledged = 0")
    conn.commit()
    conn.close()


def get_venue_zones(venue_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns list of venue zones."""
    conn = get_connection()
    cursor = conn.cursor()
    if venue_name and venue_name != "All Venues":
        cursor.execute("SELECT * FROM venue_zones WHERE venue_name = ? ORDER BY id ASC", (venue_name,))
    else:
        cursor.execute("SELECT * FROM venue_zones ORDER BY id ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_all_venue_names() -> List[str]:
    """Returns distinct venue names."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT venue_name FROM venue_zones ORDER BY venue_name")
    venues = [row[0] for row in cursor.fetchall()]
    conn.close()
    return venues


def update_zone_metrics(zone_id: int, current_count: int, current_risk: str):
    """Updates the live metrics for a specific zone."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE venue_zones SET current_count = ?, current_risk = ? WHERE id = ?
    """, (current_count, current_risk, zone_id))
    conn.commit()
    conn.close()


def add_custom_zone(
    venue_name: str,
    zone_name: str,
    area_sqm: float,
    max_capacity: int,
    warning_threshold: float,
    critical_threshold: float,
    lat: float,
    lon: float
) -> int:
    """Adds a new custom monitored zone."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO venue_zones (
        venue_name, zone_name, area_sqm, max_capacity, warning_threshold, critical_threshold, lat, lon, current_count, current_risk
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 'SAFE')
    """, (venue_name, zone_name, area_sqm, max_capacity, warning_threshold, critical_threshold, lat, lon))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_summary_stats() -> Dict[str, Any]:
    """Calculates overall aggregate stats."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*), AVG(person_count), MAX(person_count), AVG(density) FROM detections")
    det_stats = cursor.fetchone()

    cursor.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0")
    unack_alerts = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM alerts")
    total_alerts = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*), SUM(current_count), SUM(max_capacity) FROM venue_zones")
    zone_stats = cursor.fetchone()

    conn.close()
    return {
        "total_scans": det_stats[0] or 0,
        "avg_count": round(det_stats[1] or 0, 1),
        "peak_count": det_stats[2] or 0,
        "avg_density": round(det_stats[3] or 0, 2),
        "active_alerts": unack_alerts,
        "total_alerts": total_alerts,
        "total_zones": zone_stats[0] or 0,
        "total_venue_population": zone_stats[1] or 0,
        "total_venue_capacity": zone_stats[2] or 0
    }


def clear_history():
    """Clears detection history and alerts for resetting."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    cursor.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()


def seed_sample_audit_logs():
    """Populates realistic baseline detection and alert telemetry for demonstration."""
    conn = get_connection()
    cursor = conn.cursor()

    sample_detections = [
        ("2026-10-22 00:40:15", "Image Upload", "sample_safe_crowd.jpg", 22, 0.55, "SAFE", 18.5, 0, "Gate 1 - North Plaza", 14.2, '{"model": "YOLOv8-Nano"}'),
        ("2026-10-22 00:46:22", "Video Stream", "CCTV Gate 2 Feed", 58, 1.45, "SAFE", 28.0, 0, "Gate 2 - East Concourse", 18.5, '{"frames": 45}'),
        ("2026-10-22 00:52:10", "Image Upload", "sample_warning_crowd.jpg", 74, 2.15, "WARNING", 52.4, 0, "Subway Turnstile B", 12.8, '{"model": "YOLOv8-Nano"}'),
        ("2026-10-22 00:58:45", "Simulation", "Simulator: Metro Evening Rush Hour", 92, 2.62, "WARNING", 61.0, 1, "Platform 4 Express", 9.4, '{"scenario": "Metro Rush"}'),
        ("2026-10-22 01:04:30", "Video Stream", "sample_crowd_surge.mp4", 118, 3.37, "CRITICAL", 74.5, 0, "Food Court Bottleneck", 22.1, '{"peak": 118}'),
        ("2026-10-22 01:10:18", "Image Upload", "sample_critical_surge.jpg", 155, 3.87, "CRITICAL", 81.2, 0, "Section A - Lower Bowl", 15.6, '{"model": "YOLOv8-Nano"}'),
        ("2026-10-22 01:15:02", "Simulation", "Simulator: Concert Gate Ingress Bottleneck", 184, 4.60, "CRITICAL", 84.8, 1, "Main Sanctuary Steps", 11.2, '{"scenario": "Concert Surge"}'),
        ("2026-10-22 01:19:40", "Image Upload", "sample_stampede_risk.jpg", 240, 5.33, "STAMPEDE RISK", 94.2, 0, "Bottleneck Bridge East", 17.3, '{"model": "YOLOv8-Nano"}'),
        ("2026-10-22 01:22:15", "Simulation", "Simulator: Severe Stampede & Chokepoint Crush", 265, 5.88, "STAMPEDE RISK", 97.5, 1, "Demo Sector A", 8.9, '{"scenario": "Severe Crush"}'),
    ]

    cursor.executemany("""
    INSERT INTO detections (
        timestamp, source_type, source_name, person_count, density, risk_level, risk_score,
        is_simulated, venue_zone, processing_time_ms, details_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_detections)

    sample_alerts = [
        ("2026-10-22 00:52:12", "Subway Turnstile B", "WARNING", 74, 2.15, "Queue buildup exceeding warning threshold (2.0 p/m²).", "Deploy crowd marshals to manage ticket turnstile queues.", 1),
        ("2026-10-22 01:04:32", "Food Court Bottleneck", "CRITICAL", 118, 3.37, "High density bottleneck observed during video scan.", "Meter entry gates to 40% capacity and activate one-way exit signs.", 1),
        ("2026-10-22 01:19:42", "Bottleneck Bridge East", "STAMPEDE RISK", 240, 5.33, "CRITICAL: Surface density 5.33 p/m² exceeds crush limit.", "Drop all turnstile barriers immediately, open emergency relief exits 2A/2B, and halt incoming flow.", 0),
    ]

    cursor.executemany("""
    INSERT INTO alerts (
        timestamp, zone_name, alert_level, trigger_count, density, message, advisory, acknowledged
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_alerts)

    conn.commit()
    conn.close()
