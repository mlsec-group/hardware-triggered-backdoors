#!/bin/bash
set -euo pipefail

GENERATOR_BACKEND="${1:-generator}"
BLIS_BACKEND="${2:-blis}"
OPENBLAS_BACKEND="${3:-openblas}"
DATASET_PATH="${4:-data/cifar10}"
SERVER_HOSTNAME="${CHIMERA_SERVER_HOSTNAME:-$(hostname)}"
PORT="${CHIMERA_PORT:-13370}"
GENERATOR_IMAGE="${CHIMERA_GENERATOR_IMAGE:-apptainer/chimera-generator.sif}"
BLIS_IMAGE="${CHIMERA_BLIS_IMAGE:-apptainer/chimera-blis.sif}"
OPENBLAS_IMAGE="${CHIMERA_OPENBLAS_IMAGE:-apptainer/chimera-openblas.sif}"

SERVER_PID=""
CLIENT_PIDS=()
CLEANED_UP=0

cleanup() {
    if [[ "$CLEANED_UP" == "1" ]]; then
        return
    fi
    CLEANED_UP=1

    trap - EXIT INT TERM

    for pid in "${CLIENT_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        fi
    done

    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    fi

    if command -v docker >/dev/null 2>&1; then
        docker ps --format '{{.ID}} {{.Image}} {{.Ports}}' \
            | while read -r container_id image ports; do
                if [[ "$image" == "diffmath-server" ]] && [[ "$ports" == *":${PORT}->"* ]]; then
                    docker stop --time 2 "$container_id" >/dev/null 2>&1 || true
                fi
            done
    fi

    sleep 2
    if command -v docker >/dev/null 2>&1; then
        docker ps --format '{{.ID}} {{.Image}} {{.Ports}}' \
            | while read -r container_id image ports; do
                if [[ "$image" == "diffmath-server" ]] && [[ "$ports" == *":${PORT}->"* ]]; then
                    docker kill "$container_id" >/dev/null 2>&1 || true
                fi
            done
    fi

    for pid in "${CLIENT_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
}

interrupt() {
    echo "Stopping Chimera run..."
    cleanup
    exit 130
}

ensure_port_available() {
    python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
for family, host in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError as exc:
        print(f"Port {port} is already in use ({host}): {exc}", file=sys.stderr)
        print("Stop the previous Chimera run or choose another CHIMERA_PORT/CHIMERA_HTTP_PORT.", file=sys.stderr)
        sys.exit(1)
PY
}

wait_for_server() {
    python3 - "$SERVER_HOSTNAME" "$PORT" "$SERVER_PID" <<'PY'
import os
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
server_pid = int(sys.argv[3])
deadline = time.time() + 180

while time.time() < deadline:
    try:
        os.kill(server_pid, 0)
    except OSError:
        print(f"Chimera server process {server_pid} exited before opening {host}:{port}", file=sys.stderr)
        sys.exit(1)

    try:
        with socket.create_connection((host, port), timeout=2):
            sys.exit(0)
    except OSError:
        time.sleep(1)

print(f"Timed out waiting for {host}:{port}", file=sys.stderr)
sys.exit(1)
PY
}

check_backend_image() {
    if [[ "${CHIMERA_SKIP_IMAGE_CHECK:-}" == "1" || "${CHIMERA_SKIP_IMAGE_CHECK:-}" == "true" ]]; then
        return
    fi

    if ! command -v apptainer >/dev/null 2>&1; then
        echo "Missing apptainer command; cannot check or run Chimera clients." >&2
        exit 1
    fi

    for image in "$GENERATOR_IMAGE" "$BLIS_IMAGE" "$OPENBLAS_IMAGE"; do
        if [[ ! -f "$image" ]]; then
            echo "Missing client image: $image" >&2
            echo "Build them with: make -j3 apptainer-build-chimera-client" >&2
            exit 1
        fi
    done

    for spec in "blis:$BLIS_IMAGE:blis" "openblas:$OPENBLAS_IMAGE:open"; do
        backend_id="${spec%%:*}"
        rest="${spec#*:}"
        image="${rest%%:*}"
        expected_torch_blas_info="${rest#*:}"
        echo "Checking ${backend_id} PyTorch backend in ${image}"
        apptainer exec "$image" python3 - "$backend_id" "$expected_torch_blas_info" <<'PY'
import os
import sys
import torch

backend_id = sys.argv[1].lower()
expected_torch_blas_info = sys.argv[2].lower()
config = torch.__config__.show()
config_lower = config.lower()
torch_file = os.path.realpath(torch.__file__)
executable = os.path.realpath(sys.executable)

if f"blas_info={expected_torch_blas_info}" not in config_lower:
    print(
        f"Bad Chimera image for {backend_id}: imports the wrong PyTorch build.\n"
        f"expected BLAS_INFO={expected_torch_blas_info}\n"
        f"sys.executable={executable}\n"
        f"torch.__file__={torch_file}\n"
        f"torch config:\n{config}\n"
        "Rebuild with: make -j3 apptainer-build-chimera-client",
        file=sys.stderr,
    )
    sys.exit(1)
PY
    done
}

trap cleanup EXIT
trap interrupt INT TERM

check_backend_image
ensure_port_available "$PORT"
ensure_port_available "${CHIMERA_HTTP_PORT:-9696}"

echo "Starting Chimera server on ${SERVER_HOSTNAME}:${PORT}"
setsid env CHIMERA_NO_TTY=1 bash chimera_server.sh \
    "$GENERATOR_BACKEND" \
    "$BLIS_BACKEND" \
    "$OPENBLAS_BACKEND" \
    "$DATASET_PATH" &
SERVER_PID="$!"

wait_for_server

echo "Starting Chimera clients"
setsid env CHIMERA_CLIENT_IMAGE="$GENERATOR_IMAGE" bash chimera_client.sh \
    "$SERVER_HOSTNAME" \
    "$GENERATOR_BACKEND" &
CLIENT_PIDS+=("$!")

setsid env CHIMERA_CLIENT_IMAGE="$BLIS_IMAGE" bash chimera_client.sh \
    "$SERVER_HOSTNAME" \
    "$BLIS_BACKEND" &
CLIENT_PIDS+=("$!")

setsid env CHIMERA_CLIENT_IMAGE="$OPENBLAS_IMAGE" bash chimera_client.sh \
    "$SERVER_HOSTNAME" \
    "$OPENBLAS_BACKEND" &
CLIENT_PIDS+=("$!")

wait "$SERVER_PID"
