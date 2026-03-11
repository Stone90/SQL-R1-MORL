#!/usr/bin/env python3
"""Filter zero-row gold SQLs from the dataset and backfill from SynSQL-2.5M.

Gold SQLs that return 0 rows provide no useful efficiency signal. This script:
1. Audits train/test parquets, tagging zero-row samples
2. Removes them
3. Downloads 1 raw SynSQL-2.5M shard for backfill candidates
4. Validates candidates (DB exists, SQL executes, >=1 row, not duplicate)
5. Backfills per quartile to restore 6400 train / 1600 test
6. Overwrites dataset/ and copies to data/

Usage: python scripts/filter_zero_rows.py
"""

import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse from create_dataset.py
from create_dataset import (
    SEED,
    _find_db,
    compute_complexity,
    format_prompt,
    format_reward_model,
)

PROJ_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJ_DIR / "dataset"
DATA_DIR = PROJ_DIR / "data"
DB_DIR = DATASET_DIR / "databases"

TARGET_TRAIN = 6400
TARGET_TEST = 1600
TARGET_TOTAL = TARGET_TRAIN + TARGET_TEST
SAMPLES_PER_QUARTILE = TARGET_TOTAL // 4  # 2000
TRAIN_PER_QUARTILE = int(SAMPLES_PER_QUARTILE * 0.8)  # 1600
TEST_PER_QUARTILE = SAMPLES_PER_QUARTILE - TRAIN_PER_QUARTILE  # 400

FINAL_COLS = [
    "db_id", "sql_complexity", "question_style", "question",
    "external_knowledge", "cot", "sql", "data_source",
    "prompt", "ability", "reward_model", "extra_info",
]


def execute_gold_sql(db_id: str, sql: str) -> int | None:
    """Execute gold SQL and return row count, or None on failure."""
    db_path = _find_db(DB_DIR, db_id)
    if db_path is None:
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda b: b.decode(errors="ignore")
        rows = conn.execute(sql).fetchall()
        conn.close()
        return len(rows)
    except Exception:
        return None


def audit_split(df: pd.DataFrame, name: str) -> pd.Series:
    """Tag each row with its gold SQL row count. Returns boolean mask of zero-row samples."""
    print(f"\n  Auditing {name}: {len(df)} rows")
    row_counts = []
    for i, (_, row) in enumerate(df.iterrows()):
        rm = row["reward_model"]
        gt = rm["ground_truth"]
        count = execute_gold_sql(gt["db_id"], gt["sql"])
        row_counts.append(count if count is not None else 0)
        if (i + 1) % 1000 == 0:
            zero_so_far = sum(1 for c in row_counts if c == 0)
            print(f"    ... {i + 1}/{len(df)} checked ({zero_so_far} zero-row so far)")

    row_counts = pd.Series(row_counts, index=df.index)
    zero_mask = row_counts == 0
    n_zero = zero_mask.sum()
    print(f"  {name}: {n_zero}/{len(df)} zero-row samples ({100 * n_zero / len(df):.1f}%)")
    return zero_mask


def download_raw_shard() -> pd.DataFrame:
    """Download 1 raw SynSQL-2.5M shard from HuggingFace."""
    from huggingface_hub import hf_hub_download

    repo_id = "iNeil77/SynSQL-2.5M"
    print("\n  Downloading 1 shard from iNeil77/SynSQL-2.5M...")
    tmp = tempfile.mkdtemp(prefix="synsql_backfill_")
    shard_file = "data/train-00000-of-00026.parquet"
    path = hf_hub_download(repo_id, shard_file, repo_type="dataset", local_dir=tmp)
    df = pd.read_parquet(path)
    print(f"  Downloaded shard 0: {len(df)} rows")
    shutil.rmtree(tmp, ignore_errors=True)
    return df


def validate_candidates(df_raw: pd.DataFrame, existing_keys: set) -> pd.DataFrame:
    """Filter raw shard to valid backfill candidates."""
    # Available bundled DB ids
    bundled_dbs = set(p.name for p in DB_DIR.iterdir() if p.is_dir())
    print(f"\n  Bundled databases: {len(bundled_dbs)}")

    # Filter to rows with bundled DBs
    has_db = df_raw["db_id"].isin(bundled_dbs)
    df_candidates = df_raw[has_db].copy()
    print(f"  Rows with bundled DB: {len(df_candidates)}/{len(df_raw)}")

    # Deduplicate against existing dataset
    candidate_keys = df_candidates["db_id"] + "||" + df_candidates["sql"]
    is_new = ~candidate_keys.isin(existing_keys)
    df_candidates = df_candidates[is_new].copy()
    print(f"  After deduplication: {len(df_candidates)}")

    # Validate: SQL executes and returns >= 1 row
    valid_mask = pd.Series(False, index=df_candidates.index)
    checked = 0
    for idx, row in df_candidates.iterrows():
        checked += 1
        if checked % 5000 == 0:
            print(f"    ... validated {checked}/{len(df_candidates)} ({valid_mask.sum()} valid)")
        count = execute_gold_sql(row["db_id"], row["sql"])
        if count is not None and count >= 1:
            valid_mask[idx] = True

    df_valid = df_candidates[valid_mask].copy()
    print(f"  Valid candidates (executes, >=1 row): {len(df_valid)}")
    return df_valid


def main():
    print("=" * 60)
    print("Filter zero-row SQLs and backfill dataset")
    print("=" * 60)

    # ── Step 1: Load and audit current dataset ──
    print("\n── Step 1: Audit current dataset ──")
    df_train = pd.read_parquet(DATASET_DIR / "train.parquet")
    df_test = pd.read_parquet(DATASET_DIR / "test.parquet")

    zero_train = audit_split(df_train, "train")
    zero_test = audit_split(df_test, "test")

    n_remove_train = zero_train.sum()
    n_remove_test = zero_test.sum()
    n_remove_total = n_remove_train + n_remove_test
    print(f"\n  Total to remove: {n_remove_total} ({n_remove_train} train + {n_remove_test} test)")

    if n_remove_total == 0:
        print("  No zero-row samples found. Nothing to do.")
        return

    # ── Step 2: Filter out zero-row samples ──
    print("\n── Step 2: Filter zero-row samples ──")
    df_train_clean = df_train[~zero_train].copy()
    df_test_clean = df_test[~zero_test].copy()
    print(f"  Train: {len(df_train)} -> {len(df_train_clean)}")
    print(f"  Test:  {len(df_test)} -> {len(df_test_clean)}")

    # Combine clean data, compute complexity, assign quartiles
    df_clean = pd.concat([df_train_clean, df_test_clean], ignore_index=True)
    df_clean["_complexity"] = compute_complexity(df_clean)
    df_clean["_quartile"] = pd.qcut(
        df_clean["_complexity"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )

    # Track per-quartile deficit
    deficit = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        have = (df_clean["_quartile"] == q).sum()
        need = SAMPLES_PER_QUARTILE
        deficit[q] = max(0, need - have)
        print(f"  {q}: have {have}, need {need}, deficit {deficit[q]}")

    total_deficit = sum(deficit.values())
    print(f"  Total backfill needed: {total_deficit}")

    if total_deficit == 0:
        print("  No backfill needed (enough samples remain).")
        # Just re-split and write
        df_backfilled = df_clean
    else:
        # ── Step 3: Download raw shard for backfill ──
        print("\n── Step 3: Download raw shard for backfill ──")
        df_raw = download_raw_shard()

        # ── Step 4: Validate candidates ──
        print("\n── Step 4: Validate backfill candidates ──")
        existing_keys = set(df_clean["db_id"] + "||" + df_clean["sql"])
        df_valid = validate_candidates(df_raw, existing_keys)

        if len(df_valid) < total_deficit:
            print(f"  WARNING: Only {len(df_valid)} valid candidates for {total_deficit} deficit")

        # Compute complexity for candidates
        df_valid["_complexity"] = compute_complexity(df_valid)

        # Assign quartile boundaries from the clean data
        quartile_bounds = df_clean["_complexity"].quantile([0.25, 0.5, 0.75]).values
        conditions = [
            df_valid["_complexity"] <= quartile_bounds[0],
            (df_valid["_complexity"] > quartile_bounds[0]) & (df_valid["_complexity"] <= quartile_bounds[1]),
            (df_valid["_complexity"] > quartile_bounds[1]) & (df_valid["_complexity"] <= quartile_bounds[2]),
            df_valid["_complexity"] > quartile_bounds[2],
        ]
        df_valid["_quartile"] = np.select(conditions, ["Q1", "Q2", "Q3", "Q4"], default="Q4")

        # ── Step 5: Backfill per quartile ──
        print("\n── Step 5: Backfill per quartile ──")
        rng = np.random.RandomState(SEED)
        backfill_parts = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            need = deficit[q]
            if need == 0:
                continue
            q_candidates = df_valid[df_valid["_quartile"] == q]
            if len(q_candidates) < need:
                print(f"  WARNING: {q} has only {len(q_candidates)} candidates for deficit {need}")
                sample = q_candidates
            else:
                sample = q_candidates.sample(n=need, random_state=rng)
            print(f"  {q}: backfilling {len(sample)} samples")

            # Format new samples
            prompts = []
            reward_models = []
            for _, row in sample.iterrows():
                prompts.append(format_prompt(row))
                reward_models.append(format_reward_model(row))
            sample = sample.copy()
            sample["prompt"] = prompts
            sample["reward_model"] = reward_models
            sample["data_source"] = "synsql"
            sample["ability"] = "nl2sql"
            sample["extra_info"] = [{"index": 0, "split": "backfill"} for _ in range(len(sample))]
            backfill_parts.append(sample)

        if backfill_parts:
            df_backfill = pd.concat(backfill_parts, ignore_index=True)
            print(f"  Total backfilled: {len(df_backfill)}")

            # Merge
            df_backfilled = pd.concat([df_clean, df_backfill], ignore_index=True)
        else:
            df_backfilled = df_clean

    # ── Step 6: Re-split into train/test ──
    print("\n── Step 6: Re-split into train/test ──")

    # Recompute complexity on the merged set
    df_backfilled["_complexity"] = compute_complexity(df_backfilled)

    # Use rank-based quartile assignment to guarantee exactly N per group
    # (pd.qcut can produce uneven groups due to ties at boundaries)
    df_backfilled = df_backfilled.sort_values("_complexity").reset_index(drop=True)
    n = len(df_backfilled)
    q_size = n // 4
    quartile_labels = (
        ["Q1"] * q_size + ["Q2"] * q_size + ["Q3"] * q_size + ["Q4"] * (n - 3 * q_size)
    )
    df_backfilled["_quartile"] = quartile_labels

    rng = np.random.RandomState(SEED)
    train_parts = []
    test_parts = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        q_df = df_backfilled[df_backfilled["_quartile"] == q]
        n_avail = len(q_df)
        n_test = min(TEST_PER_QUARTILE, n_avail)
        n_train = min(TRAIN_PER_QUARTILE, n_avail - n_test)
        q_shuffled = q_df.sample(frac=1.0, random_state=rng)
        test_parts.append(q_shuffled.iloc[:n_test])
        train_parts.append(q_shuffled.iloc[n_test:n_test + n_train])
        print(f"  {q}: {n_train} train + {n_test} test = {n_train + n_test}")

    df_train_final = pd.concat(train_parts).sort_values("_complexity").reset_index(drop=True)
    df_test_final = pd.concat(test_parts).sort_values("_complexity").reset_index(drop=True)

    # Update extra_info with correct indices and split
    df_train_final["extra_info"] = [{"index": i, "split": "train"} for i in range(len(df_train_final))]
    df_test_final["extra_info"] = [{"index": i, "split": "test"} for i in range(len(df_test_final))]

    print(f"  Final train: {len(df_train_final)}, Final test: {len(df_test_final)}")

    # ── Step 7: Write output ──
    print("\n── Step 7: Write output ──")
    df_train_out = df_train_final[FINAL_COLS]
    df_test_out = df_test_final[FINAL_COLS]

    train_path = DATASET_DIR / "train.parquet"
    test_path = DATASET_DIR / "test.parquet"
    df_train_out.to_parquet(train_path, index=False)
    df_test_out.to_parquet(test_path, index=False)
    print(f"  {train_path}: {len(df_train_out)} rows")
    print(f"  {test_path}: {len(df_test_out)} rows")

    # Copy to data/
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(train_path), str(DATA_DIR / "train.parquet"))
    shutil.copy2(str(test_path), str(DATA_DIR / "test.parquet"))
    print(f"  Copied to {DATA_DIR}/")

    # ── Step 8: Final audit ──
    print("\n── Step 8: Final audit ──")
    for name, path in [("train", train_path), ("test", test_path)]:
        df_check = pd.read_parquet(path)
        zero_count = 0
        for _, row in df_check.iterrows():
            gt = row["reward_model"]["ground_truth"]
            count = execute_gold_sql(gt["db_id"], gt["sql"])
            if count is not None and count == 0:
                zero_count += 1
        print(f"  {name}: {len(df_check)} rows, {zero_count} zero-row gold SQLs")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
