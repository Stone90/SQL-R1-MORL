#!/usr/bin/env python3
"""Create a self-contained 1K dataset with bundled SQLite databases.

Downloads the full SynSQL databases from HuggingFace, samples 1000 rows
(stratified by proxy complexity quartile) from data/train.parquet, splits
into 800 train / 200 test, copies only the referenced SQLite databases,
and writes everything to dataset/.

Usage: python scripts/create_dataset.py
"""

import ast
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_DIR / "data"
DATASET_DIR = PROJ_DIR / "dataset"
SEED = 42

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_ground_truth(row):
    """Extract (db_id, sql) from reward_model column."""
    rm = row["reward_model"]
    if isinstance(rm, str):
        rm = ast.literal_eval(rm)
    gt = rm["ground_truth"]
    return gt["db_id"], gt["sql"]


def compute_complexity(df):
    """Compute proxy complexity: joins + subqueries + normalized sql length."""
    sql_upper = df["sql"].str.upper()
    joins = sql_upper.str.count(r" JOIN ")
    subqueries = sql_upper.str.count("SELECT") - 1
    sql_len = df["sql"].str.len()
    max_len = sql_len.max()
    return joins + subqueries + sql_len / max_len


# ── Step 1: Ensure databases are available ───────────────────────────────────

def download_databases(tmp_dir: Path) -> Path:
    """Download and extract SynSQL databases to tmp_dir, return databases path."""
    db_dir = PROJ_DIR / "databases"

    # Check if databases already exist locally
    existing_dbs = list(db_dir.glob("*/*.sqlite"))
    if existing_dbs:
        print(f"  Found {len(existing_dbs)} existing databases in {db_dir}")
        return db_dir

    # Check if there's a symlinked/extracted source
    src = db_dir / "databases_src"
    if src.exists():
        real = Path(os.path.realpath(src))
        if real.exists():
            print(f"  Found existing database source at {real}")
            return real

    # Download from HuggingFace using wget (most reliable for large files)
    print("  Downloading data.zip from seeklhy/OmniSQL-datasets (this is large)...")
    zip_path = tmp_dir / "data.zip"
    url = "https://huggingface.co/datasets/seeklhy/OmniSQL-datasets/resolve/main/data.zip"
    subprocess.run(
        ["wget", "-c", "-q", "--show-progress", "-O", str(zip_path), url],
        check=True,
    )
    print(f"  Downloaded: {zip_path} ({zip_path.stat().st_size / 1e9:.1f} GB)")

    print("  Extracting data.zip (this may take a while)...")
    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    import zipfile
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(extract_dir))
    # Delete zip to free disk space
    zip_path.unlink()
    print("  Extraction complete, deleted zip to free space")

    # Find the databases directory
    candidate = extract_dir / "data" / "SynSQL-2.5M" / "databases"
    if candidate.is_dir():
        return candidate

    # Fallback: search
    for p in extract_dir.rglob("databases"):
        if p.is_dir():
            return p

    raise FileNotFoundError("Could not find databases directory in extracted archive")


# ── Step 2-8: Sample, validate, split ────────────────────────────────────────

def create_dataset():
    print("=" * 60)
    print("Creating self-contained 1K dataset")
    print("=" * 60)

    # Load source data
    train_path = DATA_DIR / "train.parquet"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run setup_data.sh first.")
        sys.exit(1)

    df = pd.read_parquet(train_path)
    print(f"\nLoaded {len(df)} samples from {train_path}")

    # Extract db_ids
    db_ids = []
    gold_sqls = []
    for _, row in df.iterrows():
        db_id, sql = get_ground_truth(row)
        db_ids.append(db_id)
        gold_sqls.append(sql)
    df["_db_id"] = db_ids
    df["_gold_sql"] = gold_sqls

    # Download/locate databases
    print("\n── Locating databases ──")
    tmp_dir = Path(tempfile.mkdtemp(prefix="sql_r1_"))
    try:
        db_source = download_databases(tmp_dir)
        print(f"  Database source: {db_source}")

        # Compute complexity and assign quartiles
        print("\n── Computing complexity quartiles ──")
        df["_complexity"] = compute_complexity(df)
        df["_quartile"] = pd.qcut(df["_complexity"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

        for q in ["Q1", "Q2", "Q3", "Q4"]:
            count = (df["_quartile"] == q).sum()
            print(f"  {q}: {count} samples")

        # Validate: check which samples have a working database
        print("\n── Validating database availability ──")
        valid_mask = pd.Series(False, index=df.index)
        for idx, row in df.iterrows():
            db_id = row["_db_id"]
            # Check multiple possible paths
            db_path = _find_db(db_source, db_id)
            if db_path is None:
                continue
            # Test gold SQL executes
            try:
                conn = sqlite3.connect(str(db_path))
                conn.text_factory = lambda b: b.decode(errors="ignore")
                conn.execute(row["_gold_sql"]).fetchall()
                conn.close()
                valid_mask[idx] = True
            except Exception:
                continue

        valid_count = valid_mask.sum()
        print(f"  {valid_count}/{len(df)} samples have valid databases")

        if valid_count < 1000:
            print(f"ERROR: Need 1000 valid samples but only found {valid_count}")
            sys.exit(1)

        df_valid = df[valid_mask].copy()

        # Recompute quartiles on valid samples only
        df_valid["_quartile"] = pd.qcut(df_valid["_complexity"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

        # Stratified sample: 250 per quartile
        print("\n── Stratified sampling: 250 per quartile ──")
        rng = np.random.RandomState(SEED)
        sampled_parts = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            q_df = df_valid[df_valid["_quartile"] == q]
            if len(q_df) < 250:
                print(f"  WARNING: {q} has only {len(q_df)} valid samples, using all")
                sampled_parts.append(q_df)
            else:
                sampled_parts.append(q_df.sample(n=250, random_state=rng))
        df_sampled = pd.concat(sampled_parts)
        print(f"  Sampled {len(df_sampled)} total")

        # Split: 200 train + 50 test per quartile, then sort by complexity
        print("\n── Splitting: 800 train / 200 test ──")
        train_parts = []
        test_parts = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            q_df = df_sampled[df_sampled["_quartile"] == q]
            q_shuffled = q_df.sample(frac=1.0, random_state=rng)
            test_parts.append(q_shuffled.iloc[:50])
            train_parts.append(q_shuffled.iloc[50:])

        df_train = pd.concat(train_parts).sort_values("_complexity").reset_index(drop=True)
        df_test = pd.concat(test_parts).sort_values("_complexity").reset_index(drop=True)
        print(f"  Train: {len(df_train)}, Test: {len(df_test)}")

        # Collect unique db_ids
        all_db_ids = set(df_train["_db_id"].tolist() + df_test["_db_id"].tolist())
        print(f"  Unique databases: {len(all_db_ids)}")

        # Drop helper columns
        drop_cols = ["_db_id", "_gold_sql", "_complexity", "_quartile"]
        df_train = df_train.drop(columns=drop_cols)
        df_test = df_test.drop(columns=drop_cols)

        # Write parquets
        print("\n── Writing dataset ──")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        train_out = DATASET_DIR / "train.parquet"
        test_out = DATASET_DIR / "test.parquet"
        df_train.to_parquet(train_out, index=False)
        df_test.to_parquet(test_out, index=False)
        print(f"  {train_out}: {len(df_train)} rows ({train_out.stat().st_size / 1e6:.1f} MB)")
        print(f"  {test_out}: {len(df_test)} rows ({test_out.stat().st_size / 1e6:.1f} MB)")

        # Copy databases
        print("\n── Copying referenced databases ──")
        db_out = DATASET_DIR / "databases"
        if db_out.exists():
            shutil.rmtree(db_out)
        db_out.mkdir(parents=True)

        copied = 0
        total_size = 0
        missing = []
        for db_id in sorted(all_db_ids):
            src_path = _find_db(db_source, db_id)
            if src_path is None:
                missing.append(db_id)
                continue
            dest_dir = db_out / db_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{db_id}.sqlite"
            shutil.copy2(str(src_path), str(dest))
            total_size += dest.stat().st_size
            copied += 1

        print(f"  Copied {copied} databases ({total_size / 1e6:.1f} MB)")
        if missing:
            print(f"  WARNING: {len(missing)} databases not found: {missing[:5]}...")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Train samples: {len(df_train)}")
        print(f"  Test samples:  {len(df_test)}")
        print(f"  Unique DBs:    {len(all_db_ids)}")
        print(f"  DB size:       {total_size / 1e6:.1f} MB")
        print(f"  Output dir:    {DATASET_DIR}")

        # Complexity distribution
        for split_name, split_df in [("train", df_train), ("test", df_test)]:
            sql_lens = split_df["sql"].str.len()
            joins = split_df["sql"].str.upper().str.count(r" JOIN ")
            print(f"  {split_name}: avg SQL length={sql_lens.mean():.0f}, avg JOINs={joins.mean():.1f}")

    finally:
        # Clean up temp download (but not if databases came from there and we still need them)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\n  Cleaned up temp dir: {tmp_dir}")


def _find_db(db_source: Path, db_id: str):
    """Find a .sqlite file for db_id under db_source."""
    # Direct path: db_source/db_id/db_id.sqlite
    candidate = db_source / db_id / f"{db_id}.sqlite"
    if candidate.exists():
        return candidate
    # Maybe db_source IS the parent with direct sqlite files
    candidate = db_source / f"{db_id}.sqlite"
    if candidate.exists():
        return candidate
    # Resolve symlinks
    if (db_source / db_id).is_symlink():
        real = Path(os.path.realpath(db_source / db_id))
        candidate = real / f"{db_id}.sqlite"
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    create_dataset()
