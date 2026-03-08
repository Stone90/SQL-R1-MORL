#!/usr/bin/env bash
# Inference on the bundled SynSQL 8k test set (parquet format)
# Usage: sh sh/inference_synsql.sh
#   MODE=baseline sh sh/inference_synsql.sh   # use baseline checkpoint
#   MODEL=/path/to/model sh sh/inference_synsql.sh  # explicit model path
source "$(dirname "$0")/_common.sh"

banner "SynSQL Test-Set Inference"

# ── Configuration ───────────────────────────────────────────────────
MODE="${MODE:-morl}"
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-0.8}"

# Model: explicit path > auto-detect from checkpoints > base model
if [ -n "${MODEL:-}" ]; then
    CKPT="$MODEL"
else
    CKPT=$(auto_detect_checkpoint "$MODE")
fi

NUM_GPUS="${NUM_GPUS:-$(auto_detect_gpus)}"
INPUT_FILE="data/test.parquet"
OUTPUT_FILE="results/synsql-${MODE}-generated_sql.json"

# ── Pre-flight checks ──────────────────────────────────────────────
step "Configuration"
info "Mode:        $MODE"
info "Model:       $CKPT"
info "GPUs:        $NUM_GPUS"
info "N:           $N"
info "Temperature: $TEMPERATURE"
info "Input:       $INPUT_FILE"
info "Output:      $OUTPUT_FILE"

[ -d "$CKPT" ] || fail "Model not found: $CKPT"
[ -f "$INPUT_FILE" ] || fail "Test data not found: $INPUT_FILE"

# ── Run inference ───────────────────────────────────────────────────
step "Running inference on $(python3 -c "import pandas as pd; print(len(pd.read_parquet('$INPUT_FILE')))" 2>/dev/null || echo '?') samples..."

python src/inference.py \
    --nl2sql_ckpt_path "$CKPT" \
    --dataset_name synsql \
    --input_file "$INPUT_FILE" \
    --input_format parquet \
    --output_file "$OUTPUT_FILE" \
    --database_path "${SYNSQL_DB_DIR:-databases}" \
    --tensor_parallel_size "$NUM_GPUS" \
    --n "$N" \
    --temperature "$TEMPERATURE" \
    --output_format json

ok "Inference complete → $OUTPUT_FILE"

# ── Inline evaluation (exec accuracy) ──────────────────────────────
step "Evaluating execution accuracy..."

python3 - "$OUTPUT_FILE" "${SYNSQL_DB_DIR:-databases}" <<'EVAL_SCRIPT'
import json, sys, sqlite3, os

results_path, db_dir = sys.argv[1], sys.argv[2]
results = json.load(open(results_path))

correct = 0
total = 0
errors = 0

for row in results:
    db_id = row.get("db_id", "")
    gold_sql = row.get("gold_sql", row.get("sql", ""))
    pred_sqls = row.get("pred_sqls", [])
    if not gold_sql or not pred_sqls:
        continue

    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        errors += 1
        continue

    # Check if ANY of the n predictions matches gold (pass@n)
    any_match = False
    # Also track majority / first-correct
    first_match = False

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()
        cursor.execute(gold_sql)
        gold_result = set(map(tuple, cursor.fetchall()))
    except Exception:
        errors += 1
        continue

    # Check first prediction (pass@1 proxy)
    pred_sql = pred_sqls[0] if pred_sqls else ""
    if pred_sql:
        try:
            cursor.execute(pred_sql)
            pred_result = set(map(tuple, cursor.fetchall()))
            first_match = (pred_result == gold_result)
        except Exception:
            pass

    # Check any prediction (pass@n)
    for sql in pred_sqls:
        if not sql:
            continue
        try:
            cursor.execute(sql)
            pred_result = set(map(tuple, cursor.fetchall()))
            if pred_result == gold_result:
                any_match = True
                break
        except Exception:
            continue

    conn.close()
    total += 1
    if first_match:
        correct += 1

print(f"\n{'='*50}")
print(f"  Execution Accuracy (pass@1):  {correct}/{total} = {correct/total*100:.1f}%")
print(f"  Samples with DB errors:       {errors}")
print(f"  Total samples:                {len(results)}")
print(f"{'='*50}\n")
EVAL_SCRIPT

ok "Done"
