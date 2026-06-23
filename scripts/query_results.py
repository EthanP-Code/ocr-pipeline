"""
query_results.py  Runs three analytical SQL queries against the pipeline
results database and prints formatted output. Demonstrates relational
querying of pipeline outputs for QA and reporting purposes.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/processed/pipeline_results.db")


def run_queries():
    if not DB_PATH.exists():
        print("No database found. Run pipeline.py first.")
        return

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # Query 1: Documents flagged for manual review (CER > 20%)
    # ------------------------------------------------------------------
    print("=" * 55)
    print("QUERY 1 — Documents Flagged for Manual Review (CER > 20%)")
    print("=" * 55)
    rows = con.execute("""
        SELECT filename,
               ROUND(cer * 100, 2) AS cer_pct,
               ROUND(wer * 100, 2) AS wer_pct
        FROM   documents
        WHERE  flagged = 1
        ORDER  BY cer DESC
    """).fetchall()
    print(f"{'Filename':<25} {'CER %':>7} {'WER %':>7}")
    print("-" * 42)
    for r in rows:
        print(f"{r['filename']:<25} {r['cer_pct']:>7} {r['wer_pct']:>7}")
    print(f"\n{len(rows)} document(s) flagged for review.\n")

    # ------------------------------------------------------------------
    # Query 2: Extracted element count by label type
    # ------------------------------------------------------------------
    print("=" * 55)
    print("QUERY 2 — Extracted Elements by Label Type")
    print("=" * 55)
    rows = con.execute("""
        SELECT label,
               COUNT(*)            AS total,
               COUNT(DISTINCT doc_id) AS docs_present
        FROM   elements
        GROUP  BY label
        ORDER  BY total DESC
    """).fetchall()
    print(f"{'Label':<12} {'Total Elements':>16} {'Docs Containing':>16}")
    print("-" * 46)
    for r in rows:
        print(f"{r['label']:<12} {r['total']:>16} {r['docs_present']:>16}")
    print()

    # ------------------------------------------------------------------
    # Query 3: Per-run accuracy summary
    # ------------------------------------------------------------------
    print("=" * 55)
    print("QUERY 3 — Per-Run Accuracy Summary")
    print("=" * 55)
    rows = con.execute("""
        SELECT r.run_date,
               r.seed,
               COUNT(d.doc_id)        AS docs_processed,
               ROUND(AVG(d.cer) * 100, 2) AS avg_cer_pct,
               ROUND(AVG(d.wer) * 100, 2) AS avg_wer_pct,
               SUM(d.flagged)         AS flagged_count
        FROM   runs r
        JOIN   documents d ON r.run_id = d.run_id
        GROUP  BY r.run_id
        ORDER  BY r.run_id DESC
    """).fetchall()
    print(f"{'Run Date':<25} {'Seed':>6} {'Docs':>5} "
          f"{'Avg CER%':>9} {'Avg WER%':>9} {'Flagged':>8}")
    print("-" * 67)
    for r in rows:
        print(f"{r['run_date']:<25} {r['seed']:>6} {r['docs_processed']:>5} "
              f"{r['avg_cer_pct']:>9} {r['avg_wer_pct']:>9} {r['flagged_count']:>8}")

    con.close()


if __name__ == "__main__":
    run_queries()