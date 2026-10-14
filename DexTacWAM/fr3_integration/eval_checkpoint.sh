#!/usr/bin/env bash
# Serve one fr3_bar_pick stage-3 checkpoint on the host GPU and run closed-loop
# rollouts in the FR3Py devcontainer. Videos land in
# FR3Py/examples/dextacwam/rollouts/step_<STEP>/.
#
#   STEP=2000 SEEDS="100000 100001 100002" bash fr3_integration/eval_checkpoint.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEP="${STEP:?set STEP=<n>}"
SEEDS="${SEEDS:-100000 100001 100002 100003 100004}"
GPU="${GPU:-0}"
PORT="${PORT:-5008}"
CONTAINER="${CONTAINER:-fr3py-sharpa-container}"
WEIGHT="$(ls -d "$REPO_ROOT"/outputs/stage3_action_only_fr3_sharpa_bar_pick/*/step_"$STEP" | tail -1)"
LOG="${LOG_DIR:-/tmp}/dextacwam_server_step${STEP}.log"

TASK=fr3_bar_pick WEIGHT="$WEIGHT" GPU="$GPU" MIN_FREE_MB=16000 HOST=127.0.0.1 PORT="$PORT" \
    bash "$REPO_ROOT/web_infer_scripts/run_server_tactile_relative.sh" > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; pkill -f "tactile_server_sharpa_dexmate.py.*--port $PORT" 2>/dev/null' EXIT

until grep -qE "ready|FATAL|Traceback" "$LOG"; do
    kill -0 $SERVER_PID 2>/dev/null || break
    sleep 3
done
grep -q ready "$LOG" || { echo "server failed to start; see $LOG"; tail -20 "$LOG"; exit 1; }

docker exec -e MUJOCO_GL=egl "$CONTAINER" bash -c \
    "cd /home/FR3Py/examples/dextacwam && python3 fr3_sharpa_rollout.py --port $PORT --seeds $SEEDS \
     --out-dir rollouts/step_${STEP}${OUT_SUFFIX:-} --label 'checkpoint: step $STEP'" 2>&1 | grep "^\[rollout\]"
