#!/bin/bash
set -e

# Download and set up model, training data, and databases for SQL-R1-MORL
# Then validate data integrity (gold SQL queries run against DBs)
# Usage: sh sh/setup_data.sh

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$PROJ_DIR/models"
DATA_DIR="$PROJ_DIR/data"
DB_DIR="$PROJ_DIR/databases"

# ── Logging helpers ──
BOLD='\033[1m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
RESET='\033[0m'

banner() { echo ""; echo "${CYAN}╔══════════════════════════════════════════════════════╗${RESET}"; echo "${CYAN}║${RESET}  ${BOLD}$1${RESET}"; echo "${CYAN}╚══════════════════════════════════════════════════════╝${RESET}"; }
step()   { echo "${GREEN}>>>${RESET} ${BOLD}$1${RESET}"; }
info()   { echo "    ${YELLOW}→${RESET} $1"; }
ok()     { echo "    ${GREEN}✓${RESET} $1"; }
fail()   { echo "    ${RED}✗${RESET} $1"; }

banner "SQL-R1-MORL Setup"
info "Project dir: $PROJ_DIR"
info "Model dir:   $MODEL_DIR"
info "Data dir:    $DATA_DIR"
info "DB dir:      $DB_DIR"

# ── 1. Model: Qwen2.5-Coder-7B-Instruct ──
MODEL_NAME="Qwen2.5-Coder-7B-Instruct"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

banner "Step 1/4: Download Model ($MODEL_NAME)"

if [ -d "$MODEL_PATH" ] && [ "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
    ok "Model already exists at $MODEL_PATH — skipping download"
    info "Files: $(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l) safetensor shards"
else
    step "Downloading $MODEL_NAME from HuggingFace (~15 GB)..."
    info "This may take 5-30 minutes depending on connection speed"
    mkdir -p "$MODEL_DIR"
    if ! command -v huggingface-cli &>/dev/null; then
        info "Installing huggingface_hub CLI..."
        pip install -q huggingface_hub
    fi
    huggingface-cli download "Qwen/$MODEL_NAME" --local-dir "$MODEL_PATH"
    ok "Model downloaded to $MODEL_PATH"
    info "Files: $(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l) safetensor shards"
fi

# ── 2. Training data: SynSQL-2.5M (parquet) ──
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/test.parquet"

banner "Step 2/4: Download Training Data"

if [ -f "$TRAIN_FILE" ] && [ -f "$TEST_FILE" ]; then
    ok "Training data already exists at $DATA_DIR — skipping download"
    info "train.parquet: $(du -h "$TRAIN_FILE" | cut -f1)"
    info "test.parquet:  $(du -h "$TEST_FILE" | cut -f1)"
else
    step "Downloading SynSQL-Complex-5K training data from HuggingFace..."
    mkdir -p "$DATA_DIR"
    if ! command -v huggingface-cli &>/dev/null; then
        pip install -q huggingface_hub
    fi
    HF_DATA_TMP="$DATA_DIR/.hf_download"
    huggingface-cli download "MPX0222forHF/SynSQL-Complex-5K" --repo-type dataset --local-dir "$HF_DATA_TMP"
    # HF datasets store parquets in subdirs — find and copy them
    TRAIN_PQ=$(find "$HF_DATA_TMP" -name "*train*.parquet" | head -1)
    TEST_PQ=$(find "$HF_DATA_TMP" -name "*test*.parquet" | head -1)
    if [ -z "$TRAIN_PQ" ] || [ -z "$TEST_PQ" ]; then
        fail "Could not find train/test parquet files in downloaded dataset"
        info "Contents of $HF_DATA_TMP:"
        find "$HF_DATA_TMP" -type f
        exit 1
    fi
    cp "$TRAIN_PQ" "$TRAIN_FILE"
    cp "$TEST_PQ" "$TEST_FILE"
    rm -rf "$HF_DATA_TMP"
    ok "Training data downloaded to $DATA_DIR"
    info "train.parquet: $(du -h "$TRAIN_FILE" | cut -f1)"
    info "test.parquet:  $(du -h "$TEST_FILE" | cut -f1)"
fi

# ── 3. Databases: SynSQL SQLite files (for reward computation) ──
banner "Step 3/4: Download SQLite Databases"

if [ -d "$DB_DIR" ] && [ "$(ls -A "$DB_DIR" 2>/dev/null)" ]; then
    ok "Databases already exist at $DB_DIR — skipping download"
    info "Database count: $(ls -d "$DB_DIR"/*/ 2>/dev/null | wc -l) databases"
else
    step "Downloading SynSQL SQLite databases from HuggingFace (OmniSQL-datasets)..."
    info "These are needed for execution-based reward (EXPLAIN QUERY PLAN)"
    info "This download is large — 16,583 databases in data.zip"
    mkdir -p "$DB_DIR"
    if ! command -v huggingface-cli &>/dev/null; then
        pip install -q huggingface_hub
    fi
    DB_TMP="$DB_DIR/.hf_download"
    huggingface-cli download "seeklhy/OmniSQL-datasets" data.zip --repo-type dataset --local-dir "$DB_TMP"
    step "Extracting databases from data.zip..."
    unzip -q "$DB_TMP/data.zip" -d "$DB_TMP/extracted"
    # Find the SynSQL database directory (contains {db_id}/{db_id}.sqlite)
    SYNSQL_DB=$(find "$DB_TMP/extracted" -type d -name "SynSQL*" | head -1)
    if [ -z "$SYNSQL_DB" ]; then
        # Fall back: look for any directory containing .sqlite files
        SYNSQL_DB=$(dirname "$(find "$DB_TMP/extracted" -name "*.sqlite" -print -quit)")
        SYNSQL_DB=$(dirname "$SYNSQL_DB")  # go up one level to parent of {db_id}/
    fi
    if [ -z "$SYNSQL_DB" ] || [ ! -d "$SYNSQL_DB" ]; then
        fail "Could not find SynSQL databases in extracted archive"
        info "Contents of extracted archive:"
        find "$DB_TMP/extracted" -maxdepth 3 -type d
        exit 1
    fi
    # Move database subdirectories into DB_DIR
    cp -r "$SYNSQL_DB"/*/ "$DB_DIR/" 2>/dev/null || mv "$SYNSQL_DB"/* "$DB_DIR/"
    rm -rf "$DB_TMP"
    ok "Databases extracted to $DB_DIR"
    info "Database count: $(ls -d "$DB_DIR"/*/ 2>/dev/null | wc -l) databases"
fi

# ── 4. Data cleaning: validate gold SQL runs against databases ──
banner "Step 4/4: Validate Data Integrity"
step "Checking that every gold SQL query executes against its database..."
info "This ensures no bad samples corrupt training rewards"

python3 - "$DATA_DIR" "$DB_DIR" <<'PYEOF'
import sys, os, ast, sqlite3, time
import pandas as pd

data_dir = sys.argv[1]
db_dir = sys.argv[2]

BOLD = '\033[1m'
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
RESET = '\033[0m'

for split in ['train', 'test']:
    path = os.path.join(data_dir, f'{split}.parquet')
    if not os.path.exists(path):
        print(f"    {RED}✗{RESET} [{split}] File not found at {path}")
        continue

    df = pd.read_parquet(path)
    orig_len = len(df)
    drop_indices = []
    errors = {'no_db_id': 0, 'no_gold_sql': 0, 'db_missing': 0, 'sql_error': 0}

    print(f"    {YELLOW}→{RESET} [{BOLD}{split}{RESET}] Validating {orig_len:,} samples...")
    t0 = time.time()

    for idx, row in df.iterrows():
        if idx % 5000 == 0 and idx > 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            eta = (orig_len - idx) / rate if rate > 0 else 0
            print(f"      ... {idx:,}/{orig_len:,} checked ({len(drop_indices)} bad so far, ETA {eta:.0f}s)")

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

    elapsed = time.time() - t0

    if drop_indices:
        df_clean = df.drop(index=drop_indices).reset_index(drop=True)
        df_clean.to_parquet(path, index=False)
        print(f"    {RED}✗{RESET} [{BOLD}{split}{RESET}] Dropped {len(drop_indices):,}/{orig_len:,} bad samples → {GREEN}{len(df_clean):,}{RESET} remaining ({elapsed:.1f}s)")
        for reason, count in errors.items():
            if count > 0:
                print(f"           {reason}: {count:,}")
    else:
        print(f"    {GREEN}✓{RESET} [{BOLD}{split}{RESET}] All {orig_len:,} samples valid ({elapsed:.1f}s)")
PYEOF

# ── Summary ──
banner "Setup Complete"
echo ""
info "Model:     $MODEL_PATH"
info "Data:      $DATA_DIR"
info "Databases: $DB_DIR"
echo ""
step "Next steps:"
info "1. Set database path:  export SYNSQL_DB_DIR=$DB_DIR"
info "2. Run baseline:       sh sh/train_baseline.sh"
info "3. Run MORL (PC-Grad): sh sh/train.sh"
echo ""
