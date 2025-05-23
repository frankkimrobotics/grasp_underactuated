#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# 1) derive repo root (one level up from this script) ----------
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=foundationpose:latest
NAME=foundationpose

# ------------------------------------------------------------
# 2) (re)start container in the background --------------------
echo "[run_container] (re)starting ${NAME}…"
docker rm -f "${NAME}" 2>/dev/null || true

# allow GUI apps from the container to use your X server
xhost +SI:localuser:root 2>/dev/null || true

docker run --rm -d \
  --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --network host \
  --name "${NAME}" \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "${DIR}:${DIR}" \
  -v /home:/home \
  -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  --ipc host \
  -e DISPLAY="${DISPLAY}" \
  -e GIT_INDEX_FILE \
  "${IMAGE}" \
  bash -lc "cd ${DIR} && sleep infinity"

echo "[run_container] ${NAME} is up (detached)"
