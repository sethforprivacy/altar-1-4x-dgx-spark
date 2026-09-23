#!/usr/bin/env bash
# MIT: original Altar orchestration. Upstream image retains its own licenses.
set -euo pipefail
rank=${1:?usage: run-rank.sh RANK ENV_FILE [probe|ring-probe]}
env_file=${2:?supply a host-local configuration file}
mode=${3:-serve}
[[ "$rank" =~ ^[0-3]$ ]] || exit 2
[[ "$mode" == serve || "$mode" == probe || "$mode" == ring-probe ]] || exit 2
# The environment file is operator-owned shell configuration, never model input.
source "$env_file"
: "${IMAGE_ID:?pin the built image by sha256 ID}"
: "${MODEL_PATH:?absolute local model directory}"
: "${CACHE_PATH:?absolute local compilation-cache directory}"
: "${HOST_IP:?management address for this rank}"
: "${MASTER_ADDR:?rank-zero management address}"
: "${NCCL_IB_HCA:?exact connected HCA port list, starting with =}"
: "${SOCKET_IFNAME:?management network interface}"
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
[[ "$MODEL_PATH" == /* && "$CACHE_PATH" == /* ]] || exit 2
[[ "${KV_CACHE_DTYPE:-fp8}" == fp8 || "${KV_CACHE_DTYPE:-fp8}" == nvfp4_ds_mla ]] || exit 2
[[ "${NCCL_CHANNELS:-4}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "${EXPERIMENTAL_DSA_CKV_GATHER:-0}" =~ ^[01]$ ]] || exit 2
[[ "${EXPERIMENTAL_DSA_CKV_MIXED:-0}" =~ ^[01]$ ]] || exit 2
if [[ "${EXPERIMENTAL_DSA_CKV_MIXED:-0}" == 1 ]]; then
  [[ "${EXPERIMENTAL_DSA_CKV_GATHER:-0}" == 1 ]] || exit 2
  [[ "$(docker image inspect --format '{{index .Config.Labels "io.altar.experimental.dsa-ckv-mixed"}}' "$IMAGE_ID")" == 1 ]] || {
    echo 'Build the mixed-CKV image before enabling mixed batches.' >&2
    exit 2
  }
fi
if [[ "${EXPERIMENTAL_DSA_CKV_GATHER:-0}" == 1 ]]; then
  [[ "${KV_CACHE_DTYPE:-fp8}" == fp8 && "${DCP_SIZE:-1}" == 4 &&
     "${MAX_MODEL_LEN:-131072}" == 327680 && "${MAX_NUM_SEQS:-4}" == 4 &&
     "${MAX_BATCHED_TOKENS:-2048}" == 8192 && "${CUDA_GRAPH_MODE:-none}" == none ]] || {
    echo 'Experimental CKV requires the documented FP8/DCP4/c4/327680/b8192/eager profile.' >&2
    exit 2
  }
  [[ "$(docker image inspect --format '{{index .Config.Labels "io.altar.experimental.dsa-ckv"}}' "$IMAGE_ID")" == 1 ]] || {
    echo 'Build the experimental dsa-ckv image before enabling its launcher profile.' >&2
    exit 2
  }
fi
name="${CONTAINER_PREFIX:-altar}-r${rank}"
docker image inspect "$IMAGE_ID" >/dev/null
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "Container $name already exists; inspect and remove it explicitly before relaunch." >&2
  exit 1
fi
mkdir -p "$CACHE_PATH"
args=(create --name "$name" --gpus all --network host --ipc host
  --device /dev/infiniband --ulimit memlock=-1 --ulimit stack=67108864
  --entrypoint /bin/sh
  -v "$MODEL_PATH:/models/altar:ro" -v "$CACHE_PATH:/cache"
  -e "VLLM_HOST_IP=$HOST_IP" -e "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME"
  -e "GLOO_SOCKET_IFNAME=$SOCKET_IFNAME" -e "NCCL_IB_HCA=$NCCL_IB_HCA"
  -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}"
  -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_MERGE_NICS=0
  -e NCCL_IB_EXTENDED_IPV4_GIDS=1 -e NCCL_IB_PRESERVE_PCI_DOMAIN=1
  -e NCCL_CROSS_NIC=1 -e NCCL_ALGO=Ring -e NCCL_SWITCHLESS_RING_ONLY=1
  -e NCCL_CUMEM_ENABLE=0 -e "NCCL_MIN_NCHANNELS=${NCCL_CHANNELS:-4}" -e "NCCL_MAX_NCHANNELS=${NCCL_CHANNELS:-4}"
  -e NCCL_DEBUG=INFO -e NCCL_P2P_LEVEL=SYS
  -e LD_PRELOAD=/opt/local-inference/nccl/lib/libnccl.so.2
  -e VLLM_NCCL_SO_PATH=/opt/local-inference/nccl/lib/libnccl.so.2
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1
  -e VLLM_PLUGINS=b12x_loader -e XDG_CACHE_HOME=/cache
  -e TORCH_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a
  -e FLASHINFER_CUDA_ARCH_LIST=12.1f -e OMP_NUM_THREADS=4
  -e PYTHONUNBUFFERED=1)
if [[ "${EXPERIMENTAL_DSA_CKV_GATHER:-0}" == 1 ]]; then
  args+=(-e ALTAR_ENABLE_DSA_CKV_GATHER=1 -e VLLM_B12X_MLA_CKV_GATHER=1
    -e VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS=1024
    -e VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=1310720)
fi
if [[ "${EXPERIMENTAL_DSA_CKV_MIXED:-0}" == 1 ]]; then
  args+=(-e ALTAR_ENABLE_DSA_CKV_MIXED=1)
fi
command=(-m vllm.entrypoints.cli.main serve /models/altar
  --served-model-name altar-1 --tensor-parallel-size 4
  --distributed-executor-backend mp --nnodes 4 --node-rank "$rank"
  --decode-context-parallel-size "${DCP_SIZE:-1}" --dcp-comm-backend ag_rs
  --master-addr "$MASTER_ADDR" --master-port "${MASTER_PORT:-29823}"
  --disable-custom-all-reduce --attention-backend B12X --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}"
  --max-model-len "${MAX_MODEL_LEN:-131072}"
  --kv-cache-memory-bytes "${KV_CACHE_BYTES:-10737418240}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}"
  --generation-config vllm
  --host "${API_BIND:-127.0.0.1}" --port "${API_PORT:-8025}")
# Full decode-only graphs keep prefill eager and bound capture to four seats.
case "${CUDA_GRAPH_MODE:-none}" in
  none) command+=(--enforce-eager) ;;
  decode) command+=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}') ;;
  *) echo 'CUDA_GRAPH_MODE must be none or decode' >&2; exit 2 ;;
esac
case "${FLASHINFER_AUTOTUNE:-1}" in
  0) command+=(--no-enable-flashinfer-autotune) ;;
  1) command+=(--enable-flashinfer-autotune) ;;
  *) echo 'FLASHINFER_AUTOTUNE must be 0 or 1' >&2; exit 2 ;;
esac
case "${PREFIX_CACHING:-0}" in
  0) command+=(--no-enable-prefix-caching) ;;
  1) command+=(--enable-prefix-caching) ;;
  *) echo 'PREFIX_CACHING must be 0 or 1' >&2; exit 2 ;;
esac
if [[ "$mode" == ring-probe ]]; then
  here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  args+=(-v "$here/tests/ring_probe.py:/tests/ring_probe.py:ro"
    -e "RANK=$rank" -e WORLD_SIZE=4 -e "MASTER_ADDR=$MASTER_ADDR"
    -e "MASTER_PORT=${MASTER_PORT:-29823}")
  command=(/tests/ring_probe.py)
elif [[ "$mode" == probe ]]; then
  command+=(--load-format dummy --skip-tokenizer-init)
else
  command+=(--load-format safetensors --reasoning-parser glm45
    --tool-call-parser glm47 --enable-auto-tool-choice --enable-prompt-tokens-details)
  : "${API_KEY_FILE:?private host-local API key file}"
  [[ -s "$API_KEY_FILE" ]] || exit 2
  # A single dedicated API key avoids exposing credentials in shell argv/logs.
  # Docker's environment metadata is visible to Docker administrators.
  export VLLM_API_KEY
  VLLM_API_KEY=$(head -n 1 "$API_KEY_FILE")
  [[ -n "$VLLM_API_KEY" ]] || exit 2
  args+=(-e VLLM_API_KEY)
fi
if [[ "$rank" != 0 && "$mode" != ring-probe ]]; then command+=(--headless); fi
# NVIDIA's sh wrapper runs its CUDA forward-compatibility check first.
# Calling Python directly skips that initialization on the pinned base image.
# Keep compiled kernels, but use a fresh autotuning directory per launch.
container_id=$(docker "${args[@]}" "$IMAGE_ID" -c '
  export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
  VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$(mktemp -d /cache/flashinfer-autotune-run.XXXXXX) || exit 1
  exec /opt/venv/bin/python "$@"
' altar "${command[@]}")
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 "$here/memory_guard.py" --container "$container_id" \
  --state "$CACHE_PATH/guard-$name.json" --daemon >/dev/null
docker start "$container_id"
