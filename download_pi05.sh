#!/usr/bin/env bash
# Download π0.5 RoboTwin weights from Hugging Face dataset loop-wam/loopwam
# into the layout expected by collect_pi05 / pi_model.py:
#   policy/pi05/checkpoints/pi05_robotwin2/demo_clean/2000/
#
# Uses curl (avoids huggingface_hub/httpx2 decompress bugs). Supports resume (-C -).
#
#   bash download_pi05.sh
#   HF_ENDPOINT=https://hf-mirror.com bash download_pi05.sh
#   FORCE=1 bash download_pi05.sh
#
# Private dataset: log in as loop-wam first if needed:
#   unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
#   hf auth login

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

REPO_ID="${REPO_ID:-loop-wam/loopwam}"
REMOTE_DIR="${REMOTE_DIR:-pi05_robotwin2/demo_clean/2000}"
OUT_DIR="${OUT_DIR:-${ROOT}/policy/pi05/checkpoints/pi05_robotwin2/demo_clean/2000}"
FORCE="${FORCE:-0}"
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
ENDPOINT="${ENDPOINT%/}"

MARKER="${OUT_DIR}/model.safetensors"
# Size from Hub API for loop-wam/loopwam .../model.safetensors
EXPECTED_MODEL_BYTES="${EXPECTED_MODEL_BYTES:-7473091464}"

# Files relative to REMOTE_DIR
FILES=(
  "model.safetensors"
  "metadata.pt"
  "assets/pi0.5_clean_randomize_joint_training/norm_stats.json"
)

mkdir -p "${OUT_DIR}"

need_any=0
if [[ "${FORCE}" == "1" ]]; then
  need_any=1
else
  for rel in "${FILES[@]}"; do
    if [[ ! -f "${OUT_DIR}/${rel}" ]]; then
      need_any=1
      break
    fi
  done
  if [[ -f "${MARKER}" ]]; then
    ACTUAL="$(stat -c%s "${MARKER}" 2>/dev/null || echo 0)"
    if [[ "${ACTUAL}" != "${EXPECTED_MODEL_BYTES}" ]]; then
      need_any=1
      echo "[download_pi05] incomplete model.safetensors (${ACTUAL} != ${EXPECTED_MODEL_BYTES}); will resume"
    fi
  fi
fi

if [[ "${need_any}" == "0" ]]; then
  echo "[download_pi05] already complete under ${OUT_DIR}"
  echo "[download_pi05] set FORCE=1 to re-download."
  exit 0
fi

# huggingface_hub/httpx treats ALL_PROXY=socks5h://... as SOCKS and needs socksio.
if [[ "${KEEP_ALL_PROXY:-0}" != "1" ]]; then
  if [[ -n "${ALL_PROXY:-}${all_proxy:-}" ]]; then
    echo "[download_pi05] unsetting ALL_PROXY/all_proxy (was SOCKS; use HTTP_PROXY instead)"
  fi
  unset ALL_PROXY all_proxy
fi

resolve_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "${HF_TOKEN}"
    return
  fi
  if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "${HUGGING_FACE_HUB_TOKEN}"
    return
  fi
  python - <<'PY' 2>/dev/null || true
try:
    from huggingface_hub import get_token
    t = get_token()
    if t:
        print(t)
except Exception:
    pass
PY
}

TOKEN="$(resolve_token)"
if [[ -z "${TOKEN}" ]]; then
  echo "[download_pi05] no HF token found; private dataset may 401." >&2
  echo "  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN && hf auth login" >&2
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[download_pi05] curl is required" >&2
  exit 1
fi

echo "[download_pi05] endpoint=${ENDPOINT}"
echo "[download_pi05] repo=${REPO_ID}  remote=${REMOTE_DIR}/  -> ${OUT_DIR}"
echo "[download_pi05] (~7.5GB model.safetensors; curl with resume)"

download_one() {
  local rel="$1"
  local url="${ENDPOINT}/datasets/${REPO_ID}/resolve/main/${REMOTE_DIR}/${rel}"
  local dest="${OUT_DIR}/${rel}"
  mkdir -p "$(dirname "${dest}")"

  if [[ "${FORCE}" != "1" && -f "${dest}" ]]; then
    if [[ "${rel}" == "model.safetensors" ]]; then
      local sz
      sz="$(stat -c%s "${dest}")"
      if [[ "${sz}" == "${EXPECTED_MODEL_BYTES}" ]]; then
        echo "[download_pi05] skip ${rel} (already ${sz} bytes)"
        return
      fi
    else
      echo "[download_pi05] skip ${rel} (already present)"
      return
    fi
  fi

  local args=(-L --fail --retry 5 --retry-delay 2 -C - -o "${dest}")
  # Prefer identity encoding to avoid proxy/httpx-style decode issues on some stacks.
  args+=(-H "Accept-Encoding: identity")
  if [[ -n "${TOKEN}" ]]; then
    args+=(-H "Authorization: Bearer ${TOKEN}")
  fi

  echo "[download_pi05] GET ${rel}"
  if ! curl "${args[@]}" "${url}"; then
    echo "[download_pi05] curl failed for ${rel}" >&2
    echo "[download_pi05] url=${url}" >&2
    exit 1
  fi
}

for rel in "${FILES[@]}"; do
  download_one "${rel}"
done

if [[ ! -f "${MARKER}" ]]; then
  echo "[download_pi05] expected file missing: ${MARKER}" >&2
  exit 1
fi

ACTUAL="$(stat -c%s "${MARKER}")"
if [[ "${ACTUAL}" != "${EXPECTED_MODEL_BYTES}" ]]; then
  echo "[download_pi05] size mismatch for model.safetensors: got ${ACTUAL}, expected ${EXPECTED_MODEL_BYTES}" >&2
  echo "[download_pi05] re-run with FORCE=1 or check the Hub file size." >&2
  exit 1
fi

SIZE="$(du -h "${MARKER}" | awk '{print $1}')"
echo "[download_pi05] ready: ${OUT_DIR}"
echo "[download_pi05]   model.safetensors (${SIZE}, ${ACTUAL} bytes)"
ls -la "${OUT_DIR}/assets/pi0.5_clean_randomize_joint_training/" 2>/dev/null | head -5 || true
echo "[download_pi05] collect with:"
echo "  export ROBOTWIN_ROOT=${ROOT}"
echo "  bash /path/to/LoopWAM/rollout_collect/collect_pi05.sh task_list.txt ./rollout_data/pi05"
