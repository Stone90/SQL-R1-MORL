#!/usr/bin/env bash
# Evaluate BIRD predictions (post-process + exec accuracy)
# Usage: sh sh/eval_bird.sh
#   PRED_SQL=results/birddev-morl-generated_sql.json sh sh/eval_bird.sh
source "$(dirname "$0")/_common.sh"

banner "BIRD Evaluation"

# ── Configuration ───────────────────────────────────────────────────
POST_PROCESS_MODE="${POST_PROCESS_MODE:-Maj}"
SAVE_DIR="${SAVE_DIR:-results/eval/bird}"
mkdir -p "$SAVE_DIR"

# Auto-detect prediction file if not set
if [ -z "${PRED_SQL:-}" ]; then
    PRED_SQL=$(ls -t results/birddev-*generated_sql.json 2>/dev/null | head -1)
    [ -n "$PRED_SQL" ] || fail "No prediction file found. Run inference.sh first or set PRED_SQL="
    info "Auto-detected: $PRED_SQL"
fi

# Download BIRD data if needed
download_bird_data

GROUND_TRUTH_JSON_PATH=data/NL2SQL/BIRD/dev/dev.json
DB_ROOT_PATH=data/NL2SQL/BIRD/dev/dev_databases/

# ── Pre-flight checks ──────────────────────────────────────────────
step "Configuration"
info "Predictions:    $PRED_SQL"
info "Gold JSON:      $GROUND_TRUTH_JSON_PATH"
info "Database:       $DB_ROOT_PATH"
info "Post-process:   $POST_PROCESS_MODE"

[ -f "$PRED_SQL" ] || fail "Prediction file not found: $PRED_SQL"
[ -f "$GROUND_TRUTH_JSON_PATH" ] || fail "Gold JSON not found: $GROUND_TRUTH_JSON_PATH"
[ -d "$DB_ROOT_PATH" ] || fail "Database directory not found: $DB_ROOT_PATH"

# ── Post-process + evaluate ─────────────────────────────────────────
if [ "$POST_PROCESS_MODE" = "Maj" ]; then
    step "Post-processing (major voting)..."
    python src/evaluation_bird_post.py \
        --pred "$PRED_SQL" \
        --gold "$GROUND_TRUTH_JSON_PATH" \
        --db_path "$DB_ROOT_PATH" \
        --mode major_voting

elif [ "$POST_PROCESS_MODE" = "Gre" ]; then
    step "Post-processing (greedy search)..."
    python src/evaluation_bird_post.py \
        --pred "$PRED_SQL" \
        --gold "$GROUND_TRUTH_JSON_PATH" \
        --db_path "$DB_ROOT_PATH" \
        --mode greedy_search
else
    fail "Unknown POST_PROCESS_MODE: $POST_PROCESS_MODE (use Maj or Gre)"
fi

ok "BIRD evaluation complete"
