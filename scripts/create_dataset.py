#!/usr/bin/env python3
"""Create a self-contained 8K dataset with bundled SQLite databases.

Downloads raw SynSQL-2.5M data from HuggingFace (parquet shards), formats
each entry into the SQL-R1 prompt template, samples 8000 rows (stratified
by proxy complexity quartile), splits into 6400 train / 1600 test, copies
only the referenced SQLite databases, and writes everything to dataset/.

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

TOTAL_SAMPLES = 8000
SAMPLES_PER_QUARTILE = TOTAL_SAMPLES // 4  # 2000
TRAIN_PER_QUARTILE = int(SAMPLES_PER_QUARTILE * 0.8)  # 1600
TEST_PER_QUARTILE = SAMPLES_PER_QUARTILE - TRAIN_PER_QUARTILE  # 400

# Fixed one-shot example embedded in every prompt (matches SQL-R1 format)
ONE_SHOT_EXAMPLE = (
    "For example:\n"
    "<think>\n"
    "To translate the given natural language question into an executable SQLite query, "
    "we need to follow these detailed steps:\n"
    "1. **Identify Key Elements**: The question queries for code snippets that are both "
    "complicated (complexity score > 5) and public (`is_public` = 1). We need to retrieve "
    "their descriptions and complexity scores.\n"
    "2. **Focus on Relevant Tables**: The `code_snippets` table contains the necessary "
    "fields (`description`, `complexity`, `is_public`).\n"
    "3. **Construct the Query**: We should select the required fields (`description` and "
    "`complexity`) from the `code_snippets` table. We also apply the conditions specified "
    "in the question to filter the results.\n"
    "4. **Ordering**: The reference solution includes an `ORDER BY` clause to sort results "
    "by complexity in descending order, which is a reasonable way to present the data to "
    "highlight the most complex snippets first.\n"
    "5. **Final Query Construction**: Putting all this together into a SQL query.\n"
    "</think>\n"
    "<answer>\n"
    "Here's how the query can be written:\n"
    "```sql\n"
    "SELECT description, complexity FROM code_snippets WHERE complexity > 5 AND is_public = 1 "
    "ORDER BY complexity DESC;\n"
    "```\n"
    "This query retrieves the descriptions and complexity scores of code snippets that are "
    "both complicated (complexity > 5) and publicly available (`is_public` = 1), sorted by "
    "complexity in descending order.\n"
    "This solution is straightforward and precisely matches the requirements of the question. "
    "It avoids unnecessary complexities, such as joining or selecting columns not relevant "
    "to the query itself.\n"
    "</answer>\n"
)

SYSTEM_INSTRUCTIONS = (
    "You are a helpful SQL expert assistant.\n"
    "The assistant first thinks about how to write the SQL query by analyzing the question, "
    "database schema and external knowledge, then provides the final SQL query.\n"
    "The reasoning process and SQL query are enclosed within <think> </think> and "
    "<answer> </answer> tags respectively.\n"
    "The answer must contain the SQL query within ```sql ``` tags."
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def compute_complexity(df):
    """Compute proxy complexity: joins + subqueries + normalized sql length."""
    sql_upper = df["sql"].str.upper()
    joins = sql_upper.str.count(r" JOIN ")
    subqueries = sql_upper.str.count("SELECT") - 1
    sql_len = df["sql"].str.len()
    max_len = sql_len.max()
    return joins + subqueries + sql_len / max_len


def format_prompt(row):
    """Format a raw SynSQL-2.5M row into the SQL-R1 chat template."""
    schema = row["schema"]
    question = row["question"]
    external_knowledge = row.get("external_knowledge", "") or ""
    sql_complexity = row.get("sql_complexity", "") or ""

    # Build system section
    parts = [SYSTEM_INSTRUCTIONS, "", f"Database Schema:\n{schema}"]

    parts.append(f"\nExternal Knowledge:\n{external_knowledge}")
    parts.append(f"\nSQL Complexity Level: {sql_complexity}")
    parts.append(f"\n{ONE_SHOT_EXAMPLE}")

    system_content = "\n".join(parts)

    # Build full Qwen chat template as a single user message
    content = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n<think>"
    )

    return [{"role": "user", "content": content}]


def format_reward_model(row):
    """Create reward_model dict from raw row."""
    return {
        "ground_truth": {"db_id": row["db_id"], "sql": row["sql"]},
        "style": "rule",
    }


# ── Step 1: Download raw SynSQL-2.5M data ────────────────────────────────────

def download_raw_data() -> pd.DataFrame:
    """Download raw SynSQL-2.5M parquet shards from HuggingFace."""
    from huggingface_hub import hf_hub_download

    repo_id = "iNeil77/SynSQL-2.5M"
    num_shards = 26

    # We need ~8K valid samples; 1 shard has ~97K rows, plenty of headroom
    # Download 1 shard to minimize download time
    print("  Downloading 1 shard from iNeil77/SynSQL-2.5M...")
    tmp = tempfile.mkdtemp(prefix="synsql_raw_")
    shard_file = f"data/train-00000-of-000{num_shards}.parquet"
    path = hf_hub_download(repo_id, shard_file, repo_type="dataset", local_dir=tmp)
    df = pd.read_parquet(path)
    print(f"  Downloaded shard 0: {len(df)} rows")

    # Clean up temp download
    shutil.rmtree(tmp, ignore_errors=True)
    return df


# ── Step 2: Ensure databases are available ───────────────────────────────────

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

    # Download from HuggingFace
    print("  Downloading databases.zip from seeklhy/SynSQL-2.5M...")
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        "seeklhy/SynSQL-2.5M", "databases.zip", repo_type="dataset", local_dir=str(tmp_dir)
    )
    zip_path = Path(zip_path)
    print(f"  Downloaded: {zip_path} ({zip_path.stat().st_size / 1e9:.1f} GB)")

    print("  Extracting databases.zip (this may take a while)...")
    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    import zipfile
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(extract_dir))
    zip_path.unlink(missing_ok=True)
    print("  Extraction complete")

    # Find the databases directory
    for candidate in [
        extract_dir / "databases",
        extract_dir / "data" / "SynSQL-2.5M" / "databases",
    ]:
        if candidate.is_dir():
            return candidate

    # Fallback: search
    for p in extract_dir.rglob("databases"):
        if p.is_dir():
            return p

    raise FileNotFoundError("Could not find databases directory in extracted archive")


# ── Main ─────────────────────────────────────────────────────────────────────

def create_dataset():
    print("=" * 60)
    print(f"Creating self-contained {TOTAL_SAMPLES // 1000}K dataset")
    print("=" * 60)

    # Step 1: Get raw data
    print("\n── Downloading raw SynSQL-2.5M data ──")
    df_raw = download_raw_data()

    # Step 2: Download/locate databases
    print("\n── Locating databases ──")
    tmp_dir = Path(tempfile.mkdtemp(prefix="sql_r1_"))
    try:
        db_source = download_databases(tmp_dir)
        print(f"  Database source: {db_source}")

        # Compute complexity and assign quartiles
        print("\n── Computing complexity quartiles ──")
        df_raw["_complexity"] = compute_complexity(df_raw)
        df_raw["_quartile"] = pd.qcut(
            df_raw["_complexity"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )

        for q in ["Q1", "Q2", "Q3", "Q4"]:
            count = (df_raw["_quartile"] == q).sum()
            print(f"  {q}: {count} samples")

        # Validate: check which samples have a working database
        print("\n── Validating database availability ──")
        valid_mask = pd.Series(False, index=df_raw.index)
        checked = 0
        for idx, row in df_raw.iterrows():
            checked += 1
            if checked % 10000 == 0:
                print(f"  ... checked {checked}/{len(df_raw)} ({valid_mask.sum()} valid so far)")

            db_id = row["db_id"]
            db_path = _find_db(db_source, db_id)
            if db_path is None:
                continue
            # Test gold SQL executes
            try:
                conn = sqlite3.connect(str(db_path))
                conn.text_factory = lambda b: b.decode(errors="ignore")
                conn.execute(row["sql"]).fetchall()
                conn.close()
                valid_mask[idx] = True
            except Exception:
                continue

        valid_count = valid_mask.sum()
        print(f"  {valid_count}/{len(df_raw)} samples have valid databases")

        if valid_count < TOTAL_SAMPLES:
            print(f"ERROR: Need {TOTAL_SAMPLES} valid samples but only found {valid_count}")
            sys.exit(1)

        df_valid = df_raw[valid_mask].copy()

        # Recompute quartiles on valid samples only
        df_valid["_quartile"] = pd.qcut(
            df_valid["_complexity"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )

        # Stratified sample
        print(f"\n── Stratified sampling: {SAMPLES_PER_QUARTILE} per quartile ──")
        rng = np.random.RandomState(SEED)
        sampled_parts = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            q_df = df_valid[df_valid["_quartile"] == q]
            if len(q_df) < SAMPLES_PER_QUARTILE:
                print(f"  WARNING: {q} has only {len(q_df)} valid samples, using all")
                sampled_parts.append(q_df)
            else:
                sampled_parts.append(q_df.sample(n=SAMPLES_PER_QUARTILE, random_state=rng))
        df_sampled = pd.concat(sampled_parts)
        print(f"  Sampled {len(df_sampled)} total")

        # Split: train + test per quartile, then sort by complexity
        total_train = TRAIN_PER_QUARTILE * 4
        total_test = TEST_PER_QUARTILE * 4
        print(f"\n── Splitting: {total_train} train / {total_test} test ──")
        train_parts = []
        test_parts = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            q_df = df_sampled[df_sampled["_quartile"] == q]
            q_shuffled = q_df.sample(frac=1.0, random_state=rng)
            test_parts.append(q_shuffled.iloc[:TEST_PER_QUARTILE])
            train_parts.append(q_shuffled.iloc[TEST_PER_QUARTILE:])

        df_train = pd.concat(train_parts).sort_values("_complexity").reset_index(drop=True)
        df_test = pd.concat(test_parts).sort_values("_complexity").reset_index(drop=True)
        print(f"  Train: {len(df_train)}, Test: {len(df_test)}")

        # Format into SQL-R1 prompt template
        print("\n── Formatting prompts ──")
        for split_name, split_df in [("train", df_train), ("test", df_test)]:
            prompts = []
            reward_models = []
            for _, row in split_df.iterrows():
                prompts.append(format_prompt(row))
                reward_models.append(format_reward_model(row))
            split_df["prompt"] = prompts
            split_df["reward_model"] = reward_models
            split_df["data_source"] = "synsql"
            split_df["ability"] = "nl2sql"
            split_df["extra_info"] = [{"index": i, "split": split_name} for i in range(len(split_df))]
            print(f"  Formatted {len(split_df)} {split_name} prompts")

        # Assign back
        df_train = df_train
        df_test = df_test

        # Collect unique db_ids
        all_db_ids = set(df_train["db_id"].tolist() + df_test["db_id"].tolist())
        print(f"  Unique databases: {len(all_db_ids)}")

        # Select final columns (match SynSQL-Complex-5K schema)
        final_cols = [
            "db_id", "sql_complexity", "question_style", "question",
            "external_knowledge", "cot", "sql", "data_source",
            "prompt", "ability", "reward_model", "extra_info",
        ]
        df_train = df_train[final_cols]
        df_test = df_test[final_cols]

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
