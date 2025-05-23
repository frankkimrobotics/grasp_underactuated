#!/usr/bin/env bash
##############################################################################
# self-healing header for launch_foundationpose.sh
##############################################################################
set -euo pipefail

## 0) absolute path to run_container.sh (adjust if you move things)
RUN_CONTAINER="/home/frank/Desktop/full-sim-manip-pipeline/FoundationPose-main/docker/run_container.sh"

## 1) make sure run_container.sh is executable
chmod +x "$RUN_CONTAINER"

## 2) check if docker is working
_docker_ok() { docker info >/dev/null 2>&1; }
if ! _docker_ok; then  # Fixed this line
  echo "[host] Docker is not installed or running. Please start Docker."
  exit 1
fi

##############################################################################
# Everything below is your original script — replace plain `docker`
# with the alias `d` (or leave as-is; `docker` still works after the sudo re-exec)
##############################################################################

CONTAINER=foundationpose
CONDA_ENV=my
IN_CONTAINER_SCRIPT=/home/frank/Desktop/full-sim-manip-pipeline/FoundationPose-main/rollout_estimation.py

# ---------- ensure container is running ------------------------------------
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "[host] starting container ${CONTAINER}…"
  "$RUN_CONTAINER"
fi

# ---------- run demo inside the container ----------------------------------
echo "[host] executing inside ${CONTAINER}…"
docker exec "${CONTAINER}" \
  bash -lc "conda run -n ${CONDA_ENV} python ${IN_CONTAINER_SCRIPT} $*" #add $* for args

echo "[host] pipeline finished ✔"
