#!/usr/bin/env python3
"""Local diagnostic: verify reward function can access databases and score correctly.

Usage: python scripts/test_reward.py
No GPU needed — runs on CPU only.
"""
import os
import sys
import ast

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

# Set SYNSQL_DB_DIR before importing reward module
db_dir = os.environ.get("SYNSQL_DB_DIR", "databases")
os.environ["SYNSQL_DB_DIR"] = db_dir

from verl.utils.reward_score.synsql import compute_score


def get_sample(data_path="data/train.parquet"):
    """Load one sample with a valid db_id and gold_sql."""
    df = pd.read_parquet(data_path)
    for _, row in df.iterrows():
        gt = row.get("reward_model", {})
        if isinstance(gt, str):
            try:
                gt = ast.literal_eval(gt)
            except (ValueError, SyntaxError):
                continue
        inner = gt.get("ground_truth", gt) if isinstance(gt, dict) else {}
        if not isinstance(inner, dict):
            continue
        db_id = inner.get("db_id")
        gold_sql = inner.get("sql")
        if db_id and gold_sql:
            return gt, db_id, gold_sql
    return None, None, None


def make_response(sql_text, well_formatted=True):
    """Fabricate a model response string."""
    if well_formatted:
        return f"<think>\nLet me analyze this query.\n</think>\n<answer>\n```sql\n{sql_text}\n```\n</answer>"
    else:
        return f"Here is the SQL: {sql_text}"


def main():
    # Check databases
    db_count = 0
    if os.path.isdir(db_dir):
        for root, dirs, files in os.walk(db_dir):
            db_count += sum(1 for f in files if f.endswith(".sqlite"))

    if db_count == 0:
        print(f"FAIL: No .sqlite files found in {db_dir}/")
        print(f"  -> Run: sh sh/setup_data.sh")
        sys.exit(1)

    print(f"DB check: found {db_count} databases in {db_dir}/")

    # Get a sample
    gt, db_id, gold_sql = get_sample()
    if gt is None:
        print("FAIL: Could not find a valid sample in data/train.parquet")
        sys.exit(1)

    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    print(f"Sample: db_id={db_id}, db_exists={os.path.exists(db_path)}")
    print(f"Gold SQL: {gold_sql[:120]}...")
    print()

    # Case 1: correct format + gold SQL (should get max reward)
    resp1 = make_response(gold_sql, well_formatted=True)
    acc1, eff1 = compute_score(resp1, gt)
    print(f"Case 1 (gold SQL, correct format): accuracy={acc1}, efficiency={eff1}")

    # Case 2: correct format + bad SQL
    resp2 = make_response("SELECT 1", well_formatted=True)
    acc2, eff2 = compute_score(resp2, gt)
    print(f"Case 2 (bad SQL, correct format):  accuracy={acc2}, efficiency={eff2}")

    # Case 3: bad format
    resp3 = make_response(gold_sql, well_formatted=False)
    acc3, eff3 = compute_score(resp3, gt)
    print(f"Case 3 (bad format):               accuracy={acc3}, efficiency={eff3}")

    # Verdict
    print()
    if acc1 >= 5.0:
        print("PASS: Reward function is working — databases are accessible.")
    elif acc1 == 1.0 or acc1 == 0.0:
        print("WARN: Got format reward only — SQL execution likely failing.")
        print(f"  Check db path: {db_path}")
    else:
        print(f"UNEXPECTED: accuracy={acc1} for gold SQL. Investigate manually.")


if __name__ == "__main__":
    main()
