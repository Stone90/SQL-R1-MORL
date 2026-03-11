#!/usr/bin/env bash
# Inference on Spider/BIRD benchmarks
# Usage: sh sh/inference.sh
#   DATASET=bird sh sh/inference.sh
#   MODE=baseline DATASET=spider sh sh/inference.sh
#   MODEL=/path/to/model sh sh/inference.sh
source "$(dirname "$0")/_common.sh"

banner "Spider/BIRD Inference"

# ── Configuration ───────────────────────────────────────────────────
DATASET="${DATASET:-spider}"
EVAL_MODE="${EVAL_MODE:-dev}"
MODE="${MODE:-morl}"
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-0.8}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-json}"

# Model: explicit path > auto-detect from checkpoints > base model
if [ -n "${MODEL:-}" ]; then
    CKPT="$MODEL"
else
    CKPT=$(auto_detect_checkpoint "$MODE")
fi

NUM_GPUS="${NUM_GPUS:-$(auto_detect_gpus)}"

# ── Dataset paths ───────────────────────────────────────────────────
if [ "$DATASET" = "spider" ]; then
    download_spider_data
    if [ "$EVAL_MODE" = "test" ]; then
        INPUT_FILE=data/NL2SQL/Spider/test.json
        DATABASE_PATH=data/NL2SQL/Spider/test_database
        OUTPUT_FILE="results/spidertest-${MODE}-generated_sql.${OUTPUT_FORMAT}"
        TABLE_VALUE_CACHE_PATH=data/NL2SQL/Spider/spidertest_db_id2sampled_db_values.json
        TABLE_INFO_CACHE_PATH=data/NL2SQL/Spider/spidertest_db_id2db_info.json
    else
        INPUT_FILE=data/NL2SQL/Spider/dev.json
        DATABASE_PATH=data/NL2SQL/Spider/database
        OUTPUT_FILE="results/spiderdev-${MODE}-generated_sql.${OUTPUT_FORMAT}"
        TABLE_VALUE_CACHE_PATH=data/NL2SQL/Spider/spiderdev_db_id2sampled_db_values.json
        TABLE_INFO_CACHE_PATH=data/NL2SQL/Spider/spiderdev_db_id2db_info.json
    fi
elif [ "$DATASET" = "bird" ]; then
    download_bird_data
    if [ "$EVAL_MODE" = "dev" ]; then
        INPUT_FILE=data/NL2SQL/BIRD/dev/dev.json
        DATABASE_PATH=data/NL2SQL/BIRD/dev/dev_databases
        OUTPUT_FILE="results/birddev-${MODE}-generated_sql.${OUTPUT_FORMAT}"
        TABLE_VALUE_CACHE_PATH=data/NL2SQL/BIRD/dev/bird_db_id2sampled_db_values.json
        TABLE_INFO_CACHE_PATH=data/NL2SQL/BIRD/dev/bird_db_id2db_info.json
    else
        fail "BIRD only supports dev mode"
    fi
elif [ "$DATASET" = "spider-dk" ]; then
    download_spider_data  # uses Spider databases
    INPUT_FILE=data/NL2SQL/Spider-DK/spiderdk_dev.json
    DATABASE_PATH=data/NL2SQL/Spider-DK/database
    OUTPUT_FILE="results/spiderdkdev-${MODE}-generated_sql.${OUTPUT_FORMAT}"
    TABLE_VALUE_CACHE_PATH=data/NL2SQL/Spider-DK/spiderdkdev_db_id2sampled_db_values.json
    TABLE_INFO_CACHE_PATH=data/NL2SQL/Spider-DK/spiderdkdev_db_id2db_info.json
elif [ "$DATASET" = "spider-syn" ]; then
    download_spider_data
    INPUT_FILE=data/NL2SQL/Spider-Syn/spider_syn.json
    DATABASE_PATH=data/NL2SQL/Spider/database
    OUTPUT_FILE="results/spidersyn-${MODE}-generated_sql.${OUTPUT_FORMAT}"
    TABLE_VALUE_CACHE_PATH=data/NL2SQL/Spider/spiderdev_db_id2sampled_db_values.json
    TABLE_INFO_CACHE_PATH=data/NL2SQL/Spider/spiderdev_db_id2db_info.json
elif [ "$DATASET" = "spider-realistic" ]; then
    download_spider_data
    INPUT_FILE=data/NL2SQL/Spider-Realistic/spider-realistic.json
    DATABASE_PATH=data/NL2SQL/Spider/database
    OUTPUT_FILE="results/spiderrealdev-${MODE}-generated_sql.${OUTPUT_FORMAT}"
    TABLE_VALUE_CACHE_PATH=data/NL2SQL/Spider/spiderdev_db_id2sampled_db_values.json
    TABLE_INFO_CACHE_PATH=data/NL2SQL/Spider/spiderdev_db_id2db_info.json
else
    fail "Unsupported dataset: $DATASET (use spider, bird, spider-dk, spider-syn, spider-realistic)"
fi

# ── Pre-flight checks ──────────────────────────────────────────────
step "Configuration"
info "Dataset:     $DATASET ($EVAL_MODE)"
info "Mode:        $MODE"
info "Model:       $CKPT"
info "GPUs:        $NUM_GPUS"
info "N:           $N"
info "Temperature: $TEMPERATURE"
info "Input:       $INPUT_FILE"
info "Output:      $OUTPUT_FILE"

[ -d "$CKPT" ] || fail "Model not found: $CKPT"
[ -f "$INPUT_FILE" ] || fail "Input file not found: $INPUT_FILE"
[ -d "$DATABASE_PATH" ] || fail "Database path not found: $DATABASE_PATH"

# ── Run inference ───────────────────────────────────────────────────
step "Running inference..."

python src/inference.py \
    --nl2sql_ckpt_path "$CKPT" \
    --dataset_name "$DATASET" \
    --input_file "$INPUT_FILE" \
    --input_format json \
    --output_file "$OUTPUT_FILE" \
    --database_path "$DATABASE_PATH" \
    --tensor_parallel_size "$NUM_GPUS" \
    --n "$N" \
    --temperature "$TEMPERATURE" \
    --output_format "$OUTPUT_FORMAT" \
    --table_value_cache_path "$TABLE_VALUE_CACHE_PATH" \
    --table_info_cache_path "$TABLE_INFO_CACHE_PATH"

ok "Inference complete → $OUTPUT_FILE"
