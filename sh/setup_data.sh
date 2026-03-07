#!/bin/bash
set -e

# Download and set up model, training data, and databases for SQL-R1-MORL
# Then validate data integrity (gold SQL queries run against DBs)
# Usage: sh sh/setup_data.sh

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Auto-activate venv if present
VENV_PATH="${VENV_PATH:-$PROJ_DIR/.venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    . "$VENV_PATH/bin/activate"
fi
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

BUNDLED_DIR="$PROJ_DIR/dataset"
USING_BUNDLED=false

# Detect bundled dataset (self-contained 1K dataset with SQLite DBs)
if [ -f "$BUNDLED_DIR/train.parquet" ] && [ -f "$BUNDLED_DIR/test.parquet" ] && \
   [ -n "$(find "$BUNDLED_DIR/databases" -name '*.sqlite' 2>/dev/null | head -1)" ]; then
    USING_BUNDLED=true
fi

banner "SQL-R1-MORL Setup"
info "Project dir: $PROJ_DIR"
info "Model dir:   $MODEL_DIR"
info "Data dir:    $DATA_DIR"
info "DB dir:      $DB_DIR"
if [ "$USING_BUNDLED" = true ]; then
    info "Mode:        ${GREEN}BUNDLED${RESET} (using dataset/ — no HF downloads needed)"
fi

# ── 1. Model: SQL-R1-3B (cold-start with SQL-tuned model) ──
MODEL_NAME="SQL-R1-3B"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

banner "Step 1/5: Download Model ($MODEL_NAME)"

if [ -d "$MODEL_PATH" ] && [ "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
    ok "Model already exists at $MODEL_PATH — skipping download"
    info "Files: $(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l) safetensor shards"
else
    step "Downloading $MODEL_NAME from HuggingFace (~6 GB)..."
    info "This may take 5-15 minutes depending on connection speed"
    mkdir -p "$MODEL_DIR"
    # Use hf_transfer for speed if available, otherwise fall back to default
    python -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1 || export HF_HUB_ENABLE_HF_TRANSFER=0
    MAX_RETRIES=3
    for attempt in $(seq 1 $MAX_RETRIES); do
        if hf download "MPX0222forHF/$MODEL_NAME" --local-dir "$MODEL_PATH" \
            --exclude "*.pt" --exclude "*.bin"; then
            break
        fi
        if [ "$attempt" -eq "$MAX_RETRIES" ]; then
            fail "Model download failed after $MAX_RETRIES attempts"
            exit 1
        fi
        info "Download interrupted — retrying ($attempt/$MAX_RETRIES)..."
    done
    # Verify at least one safetensor shard was actually downloaded
    if [ -z "$(ls "$MODEL_PATH"/*.safetensors 2>/dev/null)" ]; then
        fail "Model download incomplete — no .safetensors files found in $MODEL_PATH"
        exit 1
    fi
    if [ ! -f "$MODEL_PATH/tokenizer_config.json" ]; then
        fail "Model download incomplete — tokenizer_config.json missing from $MODEL_PATH"
        exit 1
    fi
    ok "Model downloaded to $MODEL_PATH"
    info "Files: $(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l) safetensor shards"
fi

# ── 2. Training data: SynSQL-2.5M (parquet) ──
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/test.parquet"

banner "Step 2/5: Download Training Data"

TRAIN_SIZE=$(stat -c%s "$TRAIN_FILE" 2>/dev/null || echo 0)
TEST_SIZE=$(stat -c%s "$TEST_FILE" 2>/dev/null || echo 0)
# Real files are ~15M and ~3.7M; empty parquets with just schema are a few KB
if [ "$TRAIN_SIZE" -gt 100000 ] && [ "$TEST_SIZE" -gt 100000 ]; then
    ok "Training data already exists at $DATA_DIR — skipping download"
    info "train.parquet: $(du -h "$TRAIN_FILE" | cut -f1)"
    info "test.parquet:  $(du -h "$TEST_FILE" | cut -f1)"
elif [ "$USING_BUNDLED" = true ]; then
    step "Copying bundled training data from dataset/..."
    mkdir -p "$DATA_DIR"
    cp "$BUNDLED_DIR/train.parquet" "$TRAIN_FILE"
    cp "$BUNDLED_DIR/test.parquet" "$TEST_FILE"
    ok "Copied bundled data to $DATA_DIR"
    info "train.parquet: $(du -h "$TRAIN_FILE" | cut -f1)"
    info "test.parquet:  $(du -h "$TEST_FILE" | cut -f1)"
else
    step "Downloading SynSQL-Complex-5K training data from HuggingFace..."
    mkdir -p "$DATA_DIR"
    HF_DATA_TMP="$DATA_DIR/.hf_download"
    hf download "MPX0222forHF/SynSQL-Complex-5K" --repo-type dataset --local-dir "$HF_DATA_TMP"
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
banner "Step 3/5: Download SQLite Databases"

DB_COUNT=$(find "$DB_DIR" -name "*.sqlite" 2>/dev/null | head -1)
if [ -n "$DB_COUNT" ]; then
    ok "Databases already exist at $DB_DIR — skipping download"
    info "Database count: $(find "$DB_DIR" -name "*.sqlite" | wc -l) databases"
elif [ "$USING_BUNDLED" = true ]; then
    step "Symlinking bundled databases from dataset/databases/..."
    mkdir -p "$DB_DIR"
    for db in "$BUNDLED_DIR/databases"/*/; do
        db_name=$(basename "$db")
        ln -sfn "$(cd "$db" && pwd)" "$DB_DIR/$db_name"
    done
    ok "Symlinked bundled databases into $DB_DIR"
    info "Database count: $(find -L "$DB_DIR" -name "*.sqlite" | wc -l) databases"
else
    step "Downloading SynSQL SQLite databases from HuggingFace (OmniSQL-datasets)..."
    info "These are needed for execution-based reward (EXPLAIN QUERY PLAN)"
    info "This download is large — 16,583 databases in data.zip"
    mkdir -p "$DB_DIR"
    DB_TMP="$PROJ_DIR/.db_download"
    hf download "seeklhy/OmniSQL-datasets" data.zip --repo-type dataset --local-dir "$DB_TMP"
    step "Extracting databases from data.zip (this may take a while)..."
    unzip -q "$DB_TMP/data.zip" -d "$DB_TMP/extracted"
    # Zip contains data/SynSQL-2.5M/databases/{db_id}/{db_id}.sqlite
    SYNSQL_DB="$DB_TMP/extracted/data/SynSQL-2.5M/databases"
    if [ ! -d "$SYNSQL_DB" ]; then
        # Fall back: search for the databases directory
        SYNSQL_DB=$(find "$DB_TMP/extracted" -type d -name "databases" | head -1)
    fi
    if [ -z "$SYNSQL_DB" ] || [ ! -d "$SYNSQL_DB" ]; then
        fail "Could not find SynSQL databases in extracted archive"
        info "Contents of extracted archive:"
        find "$DB_TMP/extracted" -maxdepth 4 -type d
        exit 1
    fi
    # Symlink instead of copying (saves disk space)
    ln -sfn "$SYNSQL_DB" "$DB_DIR/databases_src"
    # Move database subdirectories into DB_DIR
    for db in "$SYNSQL_DB"/*/; do
        db_name=$(basename "$db")
        ln -sfn "$db" "$DB_DIR/$db_name"
    done
    rm -f "$DB_TMP/data.zip"
    info "Kept extracted databases at $SYNSQL_DB (symlinked into $DB_DIR)"
    ok "Databases extracted to $DB_DIR"
    info "Database count: $(ls -d "$DB_DIR"/*/ 2>/dev/null | wc -l) databases"
fi

export SYNSQL_DB_DIR="$DB_DIR"
# Persist to .env so train scripts pick it up
if ! grep -q 'SYNSQL_DB_DIR' "$PROJ_DIR/.env" 2>/dev/null; then
    echo "SYNSQL_DB_DIR=$DB_DIR" >> "$PROJ_DIR/.env"
fi
ok "SYNSQL_DB_DIR=$DB_DIR"

# ── 4. Data cleaning: validate gold SQL runs against databases ──
banner "Step 4/5: Validate Data Integrity"

if [ "$USING_BUNDLED" = true ]; then
    ok "Skipped — bundled dataset is pre-validated"
else
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
        if len(drop_indices) == orig_len:
            print(f"    {RED}✗{RESET} [{BOLD}{split}{RESET}] ALL {orig_len:,} samples failed — skipping overwrite (likely a setup issue, not bad data)")
            for reason, count in errors.items():
                if count > 0:
                    print(f"           {reason}: {count:,}")
            print(f"    {YELLOW}→{RESET} Check that databases are downloaded and extracted to {db_dir}")
            sys.exit(1)
        df_clean = df.drop(index=drop_indices).reset_index(drop=True)
        df_clean.to_parquet(path, index=False)
        print(f"    {RED}✗{RESET} [{BOLD}{split}{RESET}] Dropped {len(drop_indices):,}/{orig_len:,} bad samples → {GREEN}{len(df_clean):,}{RESET} remaining ({elapsed:.1f}s)")
        for reason, count in errors.items():
            if count > 0:
                print(f"           {reason}: {count:,}")
    else:
        print(f"    {GREEN}✓{RESET} [{BOLD}{split}{RESET}] All {orig_len:,} samples valid ({elapsed:.1f}s)")
PYEOF
fi  # end USING_BUNDLED guard for step 4

# ── 5. Curriculum sorting: order training data by proxy complexity ──
banner "Step 5/5: Curriculum Sort (Easy → Hard)"

if [ "$USING_BUNDLED" = true ]; then
    ok "Skipped — bundled dataset is pre-sorted by complexity"
else
step "Sorting training data by proxy complexity (JOINs + subqueries + SQL length)..."

python3 - "$DATA_DIR" <<'PYEOF'
import sys, os
import pandas as pd

data_dir = sys.argv[1]
path = os.path.join(data_dir, 'train.parquet')

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
BOLD = '\033[1m'
RESET = '\033[0m'

df = pd.read_parquet(path)
df['_joins'] = df['sql'].str.upper().str.count(' JOIN ')
df['_subq'] = df['sql'].str.upper().str.count('SELECT') - 1
df['_len'] = df['sql'].str.len()
max_len = df['_len'].max()
df['_complexity'] = df['_joins'] + df['_subq'] + df['_len'] / max_len

df = df.sort_values('_complexity').drop(columns=['_joins', '_subq', '_len', '_complexity']).reset_index(drop=True)
df.to_parquet(path, index=False)

# Show distribution
print(f"    {GREEN}✓{RESET} Sorted {len(df):,} samples by complexity (easy → hard)")
# Show first/last examples
for label, idx in [('Easiest', 0), ('Hardest', len(df)-1)]:
    sql = df.iloc[idx]['sql']
    joins = sql.upper().count(' JOIN ')
    subq = sql.upper().count('SELECT') - 1
    print(f"    {YELLOW}→{RESET} {label}: {len(sql)} chars, {joins} JOINs, {subq} subqueries")
PYEOF
fi  # end USING_BUNDLED guard for step 5

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
