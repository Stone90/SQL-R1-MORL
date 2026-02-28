#!/bin/bash
set -e

# Download and set up model, training data, and databases for SQL-R1-MORL
# Then validate data integrity (gold SQL queries run against DBs)
# Usage: sh sh/setup_data.sh

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$PROJ_DIR/models"
DATA_DIR="$PROJ_DIR/data"
DB_DIR="$PROJ_DIR/databases"

echo "=== SQL-R1-MORL Setup ==="
echo "Project dir: $PROJ_DIR"

# ── 1. Model: Qwen2.5-Coder-7B-Instruct ──
MODEL_NAME="Qwen2.5-Coder-7B-Instruct"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

if [ -d "$MODEL_PATH" ] && [ "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
    echo ">>> Model already exists at $MODEL_PATH, skipping."
else
    echo ">>> Downloading $MODEL_NAME from HuggingFace..."
    mkdir -p "$MODEL_DIR"
    if ! command -v huggingface-cli &>/dev/null; then
        echo "huggingface-cli not found, installing huggingface_hub..."
        pip install -q huggingface_hub
    fi
    huggingface-cli download "Qwen/$MODEL_NAME" --local-dir "$MODEL_PATH"
    echo ">>> Model downloaded to $MODEL_PATH"
fi

# ── 2. Training data: SynSQL-2.5M (parquet) ──
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/test.parquet"

if [ -f "$TRAIN_FILE" ] && [ -f "$TEST_FILE" ]; then
    echo ">>> Training data already exists at $DATA_DIR, skipping."
else
    echo ">>> Downloading SynSQL training data..."
    mkdir -p "$DATA_DIR"
    if ! command -v huggingface-cli &>/dev/null; then
        pip install -q huggingface_hub
    fi
    huggingface-cli download "StoneLin/SQL-R1-Data" --repo-type dataset --local-dir "$DATA_DIR"
    echo ">>> Training data downloaded to $DATA_DIR"
fi

# ── 3. Databases: SynSQL SQLite files (for reward computation) ──
if [ -d "$DB_DIR" ] && [ "$(ls -A "$DB_DIR" 2>/dev/null)" ]; then
    echo ">>> Databases already exist at $DB_DIR, skipping."
else
    echo ">>> Downloading SynSQL databases..."
    mkdir -p "$DB_DIR"
    if ! command -v huggingface-cli &>/dev/null; then
        pip install -q huggingface_hub
    fi
    huggingface-cli download "StoneLin/SQL-R1-Databases" --repo-type dataset --local-dir "$DB_DIR"
    echo ">>> Databases downloaded to $DB_DIR"
fi

# ── 4. Data cleaning: validate gold SQL runs against databases ──
echo ""
echo "=== Validating Data Integrity ==="
python3 - "$DATA_DIR" "$DB_DIR" <<'PYEOF'
import sys, os, ast, sqlite3
import pandas as pd

data_dir = sys.argv[1]
db_dir = sys.argv[2]

for split in ['train', 'test']:
    path = os.path.join(data_dir, f'{split}.parquet')
    if not os.path.exists(path):
        print(f"  [{split}] File not found, skipping.")
        continue

    df = pd.read_parquet(path)
    orig_len = len(df)
    drop_indices = []
    errors = {'no_db_id': 0, 'no_gold_sql': 0, 'db_missing': 0, 'sql_error': 0}

    print(f"  [{split}] Validating {orig_len} samples...")

    for idx, row in df.iterrows():
        gt = row.get('reward_model', {})
        if isinstance(gt, str):
            try:
                gt = ast.literal_eval(gt)
            except:
                drop_indices.append(idx)
                errors['no_gold_sql'] += 1
                continue

        inner = gt.get('ground_truth', gt) if isinstance(gt, dict) else {}
        if not isinstance(inner, dict):
            drop_indices.append(idx)
            errors['no_gold_sql'] += 1
            continue

        db_id = inner.get('db_id')
        gold_sql = inner.get('sql')

        if not db_id:
            drop_indices.append(idx)
            errors['no_db_id'] += 1
            continue
        if not gold_sql:
            drop_indices.append(idx)
            errors['no_gold_sql'] += 1
            continue

        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            drop_indices.append(idx)
            errors['db_missing'] += 1
            continue

        # Test that gold SQL actually executes
        try:
            conn = sqlite3.connect(db_path)
            conn.text_factory = lambda b: b.decode(errors="ignore")
            cursor = conn.cursor()
            cursor.execute(gold_sql)
            cursor.fetchall()
            cursor.close()
            conn.close()
        except Exception as e:
            drop_indices.append(idx)
            errors['sql_error'] += 1
            continue

    if drop_indices:
        df_clean = df.drop(index=drop_indices).reset_index(drop=True)
        df_clean.to_parquet(path, index=False)
        print(f"  [{split}] Dropped {len(drop_indices)}/{orig_len} bad samples -> {len(df_clean)} remaining")
        for reason, count in errors.items():
            if count > 0:
                print(f"           {reason}: {count}")
    else:
        print(f"  [{split}] All {orig_len} samples valid.")
PYEOF

echo ""
echo "=== Setup Complete ==="
echo "Model:     $MODEL_PATH"
echo "Data:      $DATA_DIR"
echo "Databases: $DB_DIR"
echo ""
echo "Set SYNSQL_DB_DIR before training if databases are not at default path:"
echo "  export SYNSQL_DB_DIR=$DB_DIR"
