"""
database.py  Handles SQLite database initialization and data insertion
for the OCR pipeline. Stores run metadata, per-document accuracy results,
and extracted FUNSD elements for downstream querying and QA review.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/processed/pipeline_results.db")


def init_db():
    """Creates the database and tables if they don't already exist."""
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            seed      INTEGER,
            run_date  TEXT
        );

        CREATE TABLE IF NOT EXISTS documents (
            doc_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id    INTEGER REFERENCES runs(run_id),
            filename  TEXT,
            cer       REAL,
            wer       REAL,
            flagged   INTEGER   -- 1 if CER > 0.20, else 0
        );

        CREATE TABLE IF NOT EXISTS elements (
            elem_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id    INTEGER REFERENCES documents(doc_id),
            label     TEXT,     -- question / answer / header / other
            text      TEXT,
            box_x1    REAL,
            box_y1    REAL,
            box_x2    REAL,
            box_y2    REAL
        );
    """)
    con.commit()
    return con


def insert_run(con, seed, run_date):
    """Logs a pipeline run and returns its auto-generated run_id."""
    cur = con.cursor()
    cur.execute(
        "INSERT INTO runs (seed, run_date) VALUES (?, ?)",
        (seed, run_date)
    )
    con.commit()
    return cur.lastrowid


def insert_document(con, run_id, filename, cer, wer):
    """Inserts a document's accuracy results and returns its doc_id."""
    cur = con.cursor()
    cur.execute(
        """INSERT INTO documents (run_id, filename, cer, wer, flagged)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, filename, cer, wer, 1 if cer > 0.20 else 0)
    )
    con.commit()
    return cur.lastrowid


def insert_elements(con, doc_id, funsd_json):
    """Inserts all FUNSD form elements for a document."""
    cur = con.cursor()
    for elem in funsd_json.get("form", []):
        box = elem.get("box", [None, None, None, None])
        cur.execute(
            """INSERT INTO elements
               (doc_id, label, text, box_x1, box_y1, box_x2, box_y2)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                elem.get("label"),
                elem.get("text", ""),
                box[0], box[1], box[2], box[3]
            )
        )
    con.commit()