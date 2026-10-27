#!/bin/bash
# Run one nightly PPU suite partition inside the worker pod.
#
# The board arrives as a pod submitted by flytiger-eco/ppu-distributed-action,
# not as `docker run` on a self-hosted runner, so the container setup that used
# to sit in the workflow now happens here. Per-partition values come from the
# environment:
#   NIGHTLY_SUITE            registered suite name for run_suite.py
#   NIGHTLY_DEVICE_COUNT     PPUs to expose; must equal nproc_per_node
#   NIGHTLY_TIMEOUT_PER_FILE seconds run_suite.py gives each registered file
#   NIGHTLY_WARM_MODELS      '|'-separated checkpoints to page-warm first
#   NIGHTLY_PARTITION_ID     LPT partition to run; omit to run the whole suite
#   NIGHTLY_PARTITION_SIZE   number of partitions the suite is cut into
#   NIGHTLY_MARLIN_PRELOAD   "true" to compile the Marlin JIT before the suite
#   NIGHTLY_RESULTS_DIR      pod-side path to copy the suite log to; the runner
#                            reads the same bytes through the /mnt/wl_nas view
#
# Lists are '|'-separated because the action's `extra_env` is a comma-separated
# KEY=VALUE list.
set -e

# The board cannot reach pypi.org and the image's Artifactory index answers 401
# to the CI credentials; every other K8s lane on these boards uses this mirror.
mkdir -p ~/.pip && cat > ~/.pip/pip.conf << 'PIPEOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
PIPEOF

: "${NIGHTLY_SUITE:?the registered suite name is required}"
: "${NIGHTLY_DEVICE_COUNT:?the device count is required}"
: "${NIGHTLY_TIMEOUT_PER_FILE:?the per-file timeout is required}"

# The action carries no files out of the pod, only its logs, so the copy travels
# over the NAS both sides mount. A missing destination costs the archive, not
# the verdict: the suite still runs and still decides the job.
results_dir="${NIGHTLY_RESULTS_DIR:-}"
if [ -n "$results_dir" ] && ! mkdir -p "$results_dir"; then
  echo "WARN: cannot create $results_dir; the results will not be archived"
  results_dir=""
fi

# Constant for every partition, so they are not repeated in each job.
# run_answer_suite_board.sh records the measurements behind the PPU/SGLANG_*
# values.
export SGLANG_IS_IN_CI=true
export CPLUS_INCLUDE_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/include
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub
export PPU_SUPPORTS_FP8=0
export SGL_KERNEL_WHEEL_CACHE_DIR=/mnt/wl_nas/cache/sgl-kernel-wheels
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_SAIL_PLA_CUDA=0

# nproc_per_node is a resource request, not device isolation: a pod that asks
# for one PPU still sees all eight, so narrow the visible set here or
# _wait_for_gpu_idle_in_ci gates on a neighbour's memory. The PPU shim parses
# comma lists only ("0-3" is read as a single device), and `seq -s` leaves a
# trailing comma to trim.
visible_devices="$(seq -s, 0 "$((NIGHTLY_DEVICE_COUNT - 1))")"
export CUDA_VISIBLE_DEVICES="${visible_devices%,}"

cd /workspace/source
git config --global --add safe.directory /workspace/source

bash scripts/ci/ppu/ppu_install_dependency.sh

# The JIT cache is keyed on source mtime and a fresh checkout resets it, so pin
# to HEAD to keep reusing the shared cache (a local copy is not worth it across
# sixty-odd files).
JIT_MTIME=$(git log -1 --format=%ct 2>/dev/null || echo 0)
if [ "$JIT_MTIME" != 0 ] && [ -d python/sglang/kernels/jit ]; then
  find python/sglang/kernels/jit -type f -exec touch -d @"$JIT_MTIME" {} +
  echo "Pinned JIT source mtimes to HEAD @$JIT_MTIME; ninja reuses the shared JIT cache"
else
  echo 'Skipped JIT mtime pinning (git unavailable); JIT will recompile'
fi

# Warm on the node that will load: the page cache is that node's and is
# reclaimed within thirty minutes, so this cannot move to the orchestration job.
if [ -n "${NIGHTLY_WARM_MODELS:-}" ]; then
  read -r -a warm_models <<< "${NIGHTLY_WARM_MODELS//|/ }"
  bash scripts/ci/warm_page_cache.sh "${warm_models[@]}"
fi

if [ "${NIGHTLY_MARLIN_PRELOAD:-false}" = true ]; then
  echo 'Preloading Marlin JIT outside the per-test watchdog (instant on a cache hit, ~16min on a two-dtype miss)...'
  timeout -k 30 1500 python3 -c 'import torch; from sglang.kernels.ops.moe.moe_wna16_marlin import _jit_moe_wna16_marlin_module; _jit_moe_wna16_marlin_module(torch.float16, False, False); _jit_moe_wna16_marlin_module(torch.bfloat16, False, False); print("Marlin JIT ready")' ||
    echo 'Marlin JIT preload did not finish; test_marlin_moe may hit its own 1200s timeout'
fi

cd test
args=(
  --hw ppu
  --suite "${NIGHTLY_SUITE}"
  --nightly
  --continue-on-error
  --timeout-per-file "${NIGHTLY_TIMEOUT_PER_FILE}"
)
if [ -n "${NIGHTLY_PARTITION_ID:-}" ]; then
  : "${NIGHTLY_PARTITION_SIZE:?the partition size is required with a partition id}"
  args+=(--auto-partition-id "${NIGHTLY_PARTITION_ID}" --auto-partition-size "${NIGHTLY_PARTITION_SIZE}")
fi

if [ -z "$results_dir" ]; then
  python3 run_suite.py "${args[@]}"
  exit
fi

# What ran, so the artifact answers "which set was this" without the job log.
{
  echo "suite=${NIGHTLY_SUITE}"
  echo "visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "device_count=${NIGHTLY_DEVICE_COUNT}"
  echo "partition=${NIGHTLY_PARTITION_ID:-none}/${NIGHTLY_PARTITION_SIZE:-none}"
  echo "timeout_per_file=${NIGHTLY_TIMEOUT_PER_FILE}"
  echo "marlin_preload=${NIGHTLY_MARLIN_PRELOAD:-false}"
  echo "source_revision=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
} > "${results_dir}/suite.txt" ||
  echo "WARN: cannot write ${results_dir}/suite.txt; the archive will lack its manifest"

# tee keeps the live log on the run page; pipefail keeps the suite's verdict.
set -o pipefail
python3 run_suite.py "${args[@]}" 2>&1 | tee "${results_dir}/run.log"
