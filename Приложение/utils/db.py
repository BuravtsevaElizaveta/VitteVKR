# utils/db.py — утилиты для SQLite detections.db (с миграцией схемы и расширенными полями)

from __future__ import annotations
from pathlib import Path
from typing import Optional, Set
import sqlite3
import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    kind TEXT,
    source TEXT,
    filename TEXT,
    model TEXT,
    label_en TEXT,
    label_ru TEXT,
    score REAL,
    json_top5 TEXT,
    price_min REAL,
    price_max REAL,
    currency TEXT,
    image_path TEXT,
    -- новые поля для «Номер + ИИ-ассистент»
    plate_number TEXT,
    region TEXT,
    year_issued TEXT,
    brand TEXT,
    car_type TEXT,
    color TEXT,
    conf_avg REAL,
    extra_json TEXT
);
"""

REQUIRED_COLUMNS = [
    ("ts", "TEXT"), ("kind", "TEXT"), ("source", "TEXT"), ("filename", "TEXT"),
    ("model", "TEXT"), ("label_en", "TEXT"), ("label_ru", "TEXT"), ("score", "REAL"),
    ("json_top5", "TEXT"), ("price_min", "REAL"), ("price_max", "REAL"), ("currency", "TEXT"),
    ("image_path", "TEXT"),
    ("plate_number", "TEXT"), ("region", "TEXT"), ("year_issued", "TEXT"),
    ("brand", "TEXT"), ("car_type", "TEXT"), ("color", "TEXT"),
    ("conf_avg", "REAL"), ("extra_json", "TEXT"),
]

def ensure_schema(db_path: Path):
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        cur = conn.execute("PRAGMA table_info(detections)")
        existing: Set[str] = {row[1] for row in cur.fetchall()}
        for col, coltype in REQUIRED_COLUMNS:
            if col not in existing:
                conn.execute(f"ALTER TABLE detections ADD COLUMN {col} {coltype}")
        conn.commit()
    finally:
        conn.close()

def insert_detection(
    db_path: Path,
    ts: Optional[str],
    kind: Optional[str],
    source: Optional[str],
    filename: Optional[str],
    model: Optional[str],
    label_en: Optional[str],
    label_ru: Optional[str],
    score: Optional[float],
    json_top5: Optional[str],
    price_min: Optional[float],
    price_max: Optional[float],
    currency: Optional[str],
    image_path: Optional[str] = None,
    # новые опциональные поля
    plate_number: Optional[str] = None,
    region: Optional[str] = None,
    year_issued: Optional[str] = None,
    brand: Optional[str] = None,
    car_type: Optional[str] = None,
    color: Optional[str] = None,
    conf_avg: Optional[float] = None,
    extra_json: Optional[str] = None,
):
    ensure_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """INSERT INTO detections
               (ts, kind, source, filename, model, label_en, label_ru, score,
                json_top5, price_min, price_max, currency, image_path,
                plate_number, region, year_issued, brand, car_type, color, conf_avg, extra_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, kind, source, filename, model, label_en, label_ru, score,
             json_top5, price_min, price_max, currency, image_path,
             plate_number, region, year_issued, brand, car_type, color, conf_avg, extra_json)
        )
        conn.commit()
    finally:
        conn.close()

def fetch_detections_df(db_path: Path, limit: int = 1000) -> pd.DataFrame:
    ensure_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM detections ORDER BY ts DESC, id DESC LIMIT {int(limit)}",
            conn
        )
        return df
    finally:
        conn.close()
