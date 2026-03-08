#!/usr/bin/env python3
"""Validate that all gold SQLs in training/test data execute and return non-empty results.

Usage:
    SYNSQL_DB_DIR=databases python scripts/validate_gold_sqls.py

Uses the same preprocessing pipeline as eval_exec_match (postprocess, remove_distinct,
replace_cur_year) so results reflect what the reward function actually sees at training time.
"""

import ast
import os
import re
import sys
import sqlite3
import sqlparse
import pandas as pd


# ---------- inlined from exec_eval.py / parse.py to avoid heavy verl imports ----------

def postprocess(query: str) -> str:
    return query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')


def remove_distinct(s: str) -> str:
    toks = [t.value for t in list(sqlparse.parse(s)[0].flatten())]
    return ''.join([t for t in toks if t.lower() != 'distinct'])


def replace_cur_year(query: str) -> str:
    return re.sub(
        r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE
    )

# -----------------------------------------------------------------------------------


def execute_gold_sql(db_path: str, gold_sql: str) -> dict:
    """Execute a gold SQL and return status info."""
    sql = postprocess(gold_sql)
    sql = remove_distinct(sql)
    sql = replace_cur_year(sql)

    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"status": "ok", "num_rows": len(rows), "rows": rows}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def validate_parquet(parquet_path: str, db_base: str):
    """Validate all gold SQLs in a parquet file."""
    if not os.path.exists(parquet_path):
        print(f"  SKIP: {parquet_path} not found")
        return

    df = pd.read_parquet(parquet_path)
    print(f"\n{'='*70}")
    print(f"Validating: {parquet_path} ({len(df)} rows)")
    print(f"DB base:    {db_base}")
    print(f"{'='*70}")

    total = 0
    non_empty = 0
    empty = 0
    errors = 0
    missing_dbs = 0
    failures = []

    for idx, row in df.iterrows():
        # Extract ground truth
        gt = row.get("reward_model", row.get("ground_truth", None))
        if gt is None:
            continue
        if isinstance(gt, str):
            try:
                gt = ast.literal_eval(gt)
            except (ValueError, SyntaxError):
                failures.append((idx, "N/A", "N/A", "Failed to parse ground_truth"))
                errors += 1
                total += 1
                continue

        inner = gt.get("ground_truth", gt)
        if isinstance(inner, dict):
            db_id = inner.get("db_id")
            gold_sql = inner.get("sql")
        else:
            db_id = gt.get("db_id")
            gold_sql = gt.get("sql")

        if not db_id or not gold_sql:
            failures.append((idx, db_id or "N/A", "N/A", "Missing db_id or sql"))
            errors += 1
            total += 1
            continue

        db_path = os.path.join(db_base, db_id, f"{db_id}.sqlite")
        total += 1

        if not os.path.exists(db_path):
            failures.append((idx, db_id, gold_sql[:80], f"DB not found: {db_path}"))
            missing_dbs += 1
            continue

        result = execute_gold_sql(db_path, gold_sql)

        if result["status"] == "error":
            failures.append((idx, db_id, gold_sql[:80], f"Error: {result['error']}"))
            errors += 1
        elif result["num_rows"] == 0:
            failures.append((idx, db_id, gold_sql[:80], "Empty result (0 rows)"))
            empty += 1
        else:
            non_empty += 1

    # Summary
    print(f"\nResults:")
    print(f"  Total:      {total}")
    print(f"  Non-empty:  {non_empty}")
    print(f"  Empty:      {empty}")
    print(f"  Errors:     {errors}")
    print(f"  Missing DB: {missing_dbs}")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        print(f"  {'Row':<8} {'DB ID':<30} {'Issue'}")
        print(f"  {'-'*8} {'-'*30} {'-'*40}")
        for row_idx, db_id, sql, issue in failures:
            print(f"  {row_idx:<8} {db_id:<30} {issue}")
            if sql != "N/A":
                print(f"           SQL: {sql}")
    else:
        print("\nAll gold SQLs executed successfully with non-empty results.")


def main():
    db_base = os.environ.get("SYNSQL_DB_DIR", "dataset/databases")
    db_base = os.path.abspath(db_base)

    if not os.path.isdir(db_base):
        print(f"ERROR: Database directory not found: {db_base}")
        print(f"Set SYNSQL_DB_DIR environment variable to the correct path.")
        sys.exit(1)

    print(f"Database directory: {db_base}")

    # Count databases available
    db_count = sum(1 for d in os.listdir(db_base)
                   if os.path.isdir(os.path.join(db_base, d)))
    print(f"Databases found: {db_count}")

    for parquet in ["data/train.parquet", "data/test.parquet"]:
        validate_parquet(parquet, db_base)


if __name__ == "__main__":
    main()
