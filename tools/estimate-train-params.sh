#!/usr/bin/env bash
# estimate-train-params.sh
# Usage: ./estimate-train-params.sh <config.py> [reserved_gib]

set -euo pipefail

CONFIG="$1"
RESERVED_GIB="${2:-1}"

# temp file to hold measurements
TMP=$(mktemp)
trap "rm -f $TMP" EXIT

# 1) before: memory used (MiB)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 > $TMP

# 2) load model + one batch of size=1
#    - we pass max_epochs=0 so it loads weights and one iteration, then exits
python3 tools/train.py "$CONFIG" \
  --cfg-options \
    train_dataloader.batch_size=1 \
    train_dataloader.num_workers=0 \
    train_cfg.max_epochs=0 \
  > /dev/null 2>&1 || true

# 3) after: memory used
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 >> $TMP

read BEFORE AFTER < <(cat $TMP)

PER_SAMPLE=$(( AFTER - BEFORE ))         # MiB
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
RESERVED=$(( RESERVED_GIB * 1024 ))      # MiB
AVAILABLE=$(( TOTAL - RESERVED ))
MAX_BATCH=$(( AVAILABLE / PER_SAMPLE ))

# 4) CPU cores → num_workers heuristic
CPU_CORES=$(nproc)
# don't starve the OS, leave 1 core; cap at 4 workers per GPU
NUM_WORKERS=$(( CPU_CORES > 1 ? CPU_CORES - 1 : 1 ))
if [ "$NUM_WORKERS" -gt 4 ]; then
  NUM_WORKERS=4
fi

echo ""
echo "GPU total memory:    ${TOTAL} MiB"
echo "Per-sample footprint:${PER_SAMPLE} MiB"
echo "Reserved overhead:   ${RESERVED} MiB"
echo "→ Available for data:${AVAILABLE} MiB"
echo ""
echo "→ Recommended batch_size: ${MAX_BATCH}"
echo "CPU cores:           ${CPU_CORES}"
echo "→ Recommended num_workers: ${NUM_WORKERS}"
