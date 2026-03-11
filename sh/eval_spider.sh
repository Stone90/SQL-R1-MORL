#!/usr/bin/env bash
# Evaluate Spider predictions (post-process + exec accuracy)
# Usage: sh sh/eval_spider.sh
#   PRED_SQL=results/spiderdev-morl-generated_sql.json sh sh/eval_spider.sh
#   MODE=test sh sh/eval_spider.sh
source "$(dirname "$0")/_common.sh"

banner "Spider Evaluation"

# ── Configuration ───────────────────────────────────────────────────
EVAL_MODE="${EVAL_MODE:-dev}"
POST_PROCESS_MODE="${POST_PROCESS_MODE:-Maj}"
SAVE_DIR="${SAVE_DIR:-results/eval/spider}"
mkdir -p "$SAVE_DIR"

# Auto-detect prediction file if not set
if [ -z "${PRED_SQL:-}" ]; then
    PRED_SQL=$(ls -t results/spider${EVAL_MODE}-*generated_sql.json 2>/dev/null | head -1)
    [ -n "$PRED_SQL" ] || fail "No prediction file found. Run inference.sh first or set PRED_SQL="
    info "Auto-detected: $PRED_SQL"
fi

# Download Spider data if needed
download_spider_data

if [ "$EVAL_MODE" = "dev" ]; then
    GOLD_SQL=data/NL2SQL/Spider/dev_gold.sql
    DB=data/NL2SQL/Spider/database
    TABLE=data/NL2SQL/Spider/tables.json
elif [ "$EVAL_MODE" = "test" ]; then
    GOLD_SQL=data/NL2SQL/Spider/test_gold.sql
    DB=data/NL2SQL/Spider/test_database
    TABLE=data/NL2SQL/Spider/test_tables.json
else
    fail "Only dev or test mode supported for Spider"
fi

ETYPE=all
PLUG_VALUE=false
KEEP_DISTINCT=false
PROGRESS_BAR_FOR_EACH_DATAPOINT=false

# ── Pre-flight checks ──────────────────────────────────────────────
step "Configuration"
info "Predictions:    $PRED_SQL"
info "Gold SQL:       $GOLD_SQL"
info "Database:       $DB"
info "Post-process:   $POST_PROCESS_MODE"

[ -f "$PRED_SQL" ] || fail "Prediction file not found: $PRED_SQL"
[ -f "$GOLD_SQL" ] || fail "Gold SQL not found: $GOLD_SQL"
[ -d "$DB" ] || fail "Database directory not found: $DB"

# ── Post-process + evaluate ─────────────────────────────────────────
if [ "$POST_PROCESS_MODE" = "Maj" ]; then
    step "Post-processing (major voting)..."
    python src/evaluation_spider_post.py \
        --pred "$PRED_SQL" \
        --gold "$GOLD_SQL" \
        --db_path "$DB/" \
        --table "$TABLE" \
        --mode major_voting \
        --save_pred_sqls False \
        --save_dir "$SAVE_DIR"

    PRED_SQL_TXT="${PRED_SQL%.*}_pred_major_voting_sqls.txt"
    step "Evaluating execution accuracy..."
    python src/evaluation_spider.py \
        --gold_sql "$GOLD_SQL" \
        --pred_sql "$PRED_SQL_TXT" \
        --db "$DB" \
        --table "$TABLE" \
        --etype "$ETYPE" \
        --plug_value "$PLUG_VALUE" \
        --keep_distinct "$KEEP_DISTINCT" \
        --progress_bar_for_each_datapoint "$PROGRESS_BAR_FOR_EACH_DATAPOINT" \
        --save_dir "$SAVE_DIR"

elif [ "$POST_PROCESS_MODE" = "Gre" ]; then
    step "Post-processing (greedy search)..."
    python src/evaluation_spider_post.py \
        --pred "$PRED_SQL" \
        --gold "$GOLD_SQL" \
        --db_path "$DB/" \
        --table "$TABLE" \
        --mode greedy_search \
        --save_pred_sqls False \
        --save_dir "$SAVE_DIR"

    PRED_SQL_TXT="${PRED_SQL%.*}_pred_greedy_search_sqls.txt"
    step "Evaluating execution accuracy..."
    python src/evaluation_spider.py \
        --gold_sql "$GOLD_SQL" \
        --pred_sql "$PRED_SQL_TXT" \
        --db "$DB" \
        --table "$TABLE"
else
    fail "Unknown POST_PROCESS_MODE: $POST_PROCESS_MODE (use Maj or Gre)"
fi

ok "Spider evaluation complete. Results in $SAVE_DIR"
