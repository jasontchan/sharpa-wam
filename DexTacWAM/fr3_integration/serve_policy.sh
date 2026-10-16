#!/usr/bin/env bash
# Serve the fr3_bar_pick policy on this GPU server for the real robot client
# (FR3Py/examples/dextacwam/real_robot/). Runs in the foreground; Ctrl-C stops it.
#
# The server binds 127.0.0.1 only: the robot PC reaches it through an SSH tunnel
# (real_robot/ssh_tunnel.sh), so the unauthenticated pickle protocol is never
# exposed on the network. The port only opens once the model is loaded.
#
#   STEP=4000 bash fr3_integration/serve_policy.sh
#   STEP=4000 GPU=1 PORT=5009 bash fr3_integration/serve_policy.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEP="${STEP:-4000}"
PORT="${PORT:-5008}"
RUN_DIR="${RUN_DIR:-$(ls -d "$REPO_ROOT"/outputs/stage3_action_only_fr3_sharpa_bar_pick/*/ | tail -1)}"
WEIGHT="${RUN_DIR%/}/step_${STEP}"
[[ -f "$WEIGHT/diffusion_pytorch_model.safetensors" ]] || {
    echo "[serve] FATAL: no checkpoint at $WEIGHT" >&2
    echo "        available: $(ls -d "${RUN_DIR%/}"/step_* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')" >&2
    exit 2
}

echo "[serve] fr3_bar_pick  step=$STEP  listening on 127.0.0.1:$PORT (tunnel only)"
echo "[serve] robot PC:  bash real_robot/ssh_tunnel.sh $(whoami)@$(hostname) $PORT"
exec env TASK=fr3_bar_pick WEIGHT="$WEIGHT" HOST=127.0.0.1 PORT="$PORT" \
    MIN_FREE_MB="${MIN_FREE_MB:-16000}" ${GPU:+GPU=$GPU} \
    bash "$REPO_ROOT/web_infer_scripts/run_server_tactile_relative.sh"
