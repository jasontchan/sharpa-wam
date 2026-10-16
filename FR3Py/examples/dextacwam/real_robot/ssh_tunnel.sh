#!/usr/bin/env bash
# Forward the robot PC's 127.0.0.1:<port> to the policy server on the L40 machine.
# Run on the robot PC host (the devcontainer uses host networking, so the client
# inside it reaches the tunnel at 127.0.0.1:<port>). Ctrl-C closes the tunnel.
#
#   bash ssh_tunnel.sh user@l40-server [port]
set -euo pipefail
TARGET="${1:?usage: ssh_tunnel.sh user@l40-server [port]}"
PORT="${2:-5008}"
echo "[tunnel] 127.0.0.1:$PORT -> $TARGET:127.0.0.1:$PORT"
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
    -L "127.0.0.1:$PORT:127.0.0.1:$PORT" "$TARGET"
