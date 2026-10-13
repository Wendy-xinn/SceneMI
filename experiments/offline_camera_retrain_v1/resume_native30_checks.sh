#!/usr/bin/env bash
# Recheck formal data and model inputs after WSL Windows-drive recovery.
# This script never constructs an optimizer or starts training.
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$TASK_DIR/../.." && pwd)"
PY_NATIVE30=/home/wenxin/miniconda3/envs/scenemi/bin/python
CACHE_NATIVE30=/mnt/e/Projects/SceneMI/cache/native30_official_ego_exo_v3
REVIEW_NATIVE30="$TASK_DIR/runs/exo_occlusion_review_oct12"
PREPARED_NATIVE30="$TASK_DIR/runs/ego_exo_official_offline_memory_v4"
cd "$REPO_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY_NATIVE30" - "$CACHE_NATIVE30" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
for name in ('manifest.jsonl','source_index.jsonl','export_progress.json'):
    (p/name).stat()
status=json.loads((p/'export_progress.json').read_text())
assert status['status']=='complete' and status['completed_sources']==1280
print('E-drive accessible; rechecking full data, optimizer steps 0',flush=True)
PY
"$PY_NATIVE30" -u -m experiments.offline_camera_retrain_v1.audit_native30_full_cache \
    --repair-corrupt --root "$CACHE_NATIVE30" --output "$REVIEW_NATIVE30/FULL_NATIVE30_CACHE_AUDIT.json" \
    > "$REVIEW_NATIVE30/FULL_NATIVE30_CACHE_AUDIT.log" 2>&1
"$PY_NATIVE30" -u -m experiments.offline_camera_retrain_v1.prepare_native30 --full-data \
    > "$PREPARED_NATIVE30/PREPARE.log" 2>&1
exec "$PY_NATIVE30" -u -m experiments.offline_camera_retrain_v1.view_native30_cache \
    --port 8784 --cases-file "$REVIEW_NATIVE30/FORMAL_EXO_VISER_CASES.json"
