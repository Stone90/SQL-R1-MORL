#!/usr/bin/env bash
# Shared helpers for inference & evaluation scripts
set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

banner()  { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}\n"; }
step()    { echo -e "${BOLD}▸ $*${NC}"; }
info()    { echo -e "${CYAN}  $*${NC}"; }
ok()      { echo -e "${GREEN}✓ $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠ $*${NC}"; }
fail()    { echo -e "${RED}✗ $*${NC}" >&2; exit 1; }

# ── Environment ─────────────────────────────────────────────────────
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"

# Load .env
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Activate venv
VENV_PATH="${VENV_PATH:-$PROJ_ROOT/.venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

# ── GPU detection ───────────────────────────────────────────────────
auto_detect_gpus() {
    local n
    n=$(nvidia-smi -L 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo "$n" || echo 1
}

# ── Checkpoint detection ────────────────────────────────────────────
auto_detect_checkpoint() {
    local mode="${1:-morl}"
    local run_id="3B"
    [ "$mode" = "baseline" ] && run_id="3B-baseline"

    local ckpt_base="logs/SQL-R1-MORL"
    local latest
    latest=$(ls -d "$ckpt_base"/*"$run_id"*/actor/global_step_* 2>/dev/null \
        | sort -t_ -k3 -n | tail -1)

    if [ -n "${latest:-}" ]; then
        echo "$latest"
    else
        echo "models/SQL-R1-3B"
    fi
}

# ── Data download helpers ───────────────────────────────────────────
download_spider_data() {
    local dest="data/NL2SQL/Spider"
    if [ -d "$dest/database" ] && [ -f "$dest/dev.json" ]; then
        ok "Spider data already present at $dest"
        return
    fi
    step "Downloading Spider dataset..."
    mkdir -p "$dest"
    huggingface-cli download xlangai/spider --repo-type dataset --local-dir "$dest"
    ok "Spider data downloaded to $dest"

    # Copy db_info caches if not present
    for f in spiderdev_db_id2db_info.json spiderdev_db_id2sampled_db_values.json \
             spidertest_db_id2db_info.json spidertest_db_id2sampled_db_values.json; do
        if [ -f "db_info/$f" ] && [ ! -f "$dest/$f" ]; then
            cp "db_info/$f" "$dest/$f"
            info "Copied $f → $dest/"
        fi
    done
}

download_bird_data() {
    local dest="data/NL2SQL/BIRD"
    if [ -d "$dest/dev/dev_databases" ] && [ -f "$dest/dev/dev.json" ]; then
        ok "BIRD data already present at $dest"
        return
    fi
    step "Downloading BIRD dataset..."
    mkdir -p "$dest"
    huggingface-cli download DAMO-NLP-SG/bird --repo-type dataset --local-dir "$dest"
    ok "BIRD data downloaded to $dest"

    # Copy db_info caches if not present
    for f in bird_db_id2db_info.json bird_db_id2sampled_db_values.json; do
        if [ -f "db_info/$f" ] && [ ! -f "$dest/dev/$f" ]; then
            cp "db_info/$f" "$dest/dev/$f"
            info "Copied $f → $dest/dev/"
        fi
    done
}

# Ensure results directory exists
mkdir -p results
