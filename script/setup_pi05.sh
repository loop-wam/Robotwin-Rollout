#!/usr/bin/env bash
# Install OpenPI / JAX for policy/pi05 into policy/pi05/.venv (Python 3.11 via uv).
#
# Do NOT sync into a conda env: uv rejects conda prefixes, and openpi pins 3.11
# (.python-version) while fastwam is often 3.10.
#
# After this script, LoopWAM collect_pi05 prefers:
#   $ROBOTWIN_ROOT/policy/pi05/.venv/bin/python
# Or manually:
#   source policy/pi05/.venv/bin/activate
#
# Also installs CuRobo + warp-lang==1.12.0 into that .venv (needed for RoboTwin eval).

set -euo pipefail

ROBOTWIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI05_DIR="${ROBOTWIN_ROOT}/policy/pi05"
VENV_DIR="${PI05_DIR}/.venv"
SKIP_CUROBO="${SKIP_CUROBO:-0}"

cd "${PI05_DIR}"

echo "[setup_pi05] root=${ROBOTWIN_ROOT}"
echo "[setup_pi05] host python=$(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo '?'))"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup_pi05] installing uv ..."
  python -m pip install uv
fi

# Never point UV_PROJECT_ENVIRONMENT at a conda prefix — uv cannot recreate it.
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  echo "[setup_pi05] unsetting UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT} (use project .venv)"
  unset UV_PROJECT_ENVIRONMENT
fi

# SOCKS proxies break some uv/httpx stacks; HTTP(S) proxy is fine.
if [[ -n "${ALL_PROXY:-}${all_proxy:-}" ]]; then
  echo "[setup_pi05] unsetting ALL_PROXY/all_proxy for uv sync"
  unset ALL_PROXY all_proxy
fi

echo "[setup_pi05] GIT_LFS_SKIP_SMUDGE=1 uv sync → ${VENV_DIR} ..."
GIT_LFS_SKIP_SMUDGE=1 uv sync

VENV_PY="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "[setup_pi05] ERROR: missing ${VENV_PY} after uv sync" >&2
  exit 1
fi
echo "[setup_pi05] venv python=${VENV_PY} ($(${VENV_PY} -c 'import sys; print(sys.version.split()[0])'))"

py_importable() {
  "${VENV_PY}" -c "import $1" >/dev/null 2>&1
}

export_curobo_scm_pretend() {
  local ver="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.7.8}"
  export SETUPTOOLS_SCM_PRETEND_VERSION="${ver}"
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO:-${ver}}"
  echo "[setup_pi05] SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}"
  echo "[setup_pi05] SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO}"
}

install_warp_112() {
  local wheel=""
  local cand
  for cand in \
    "${WARP_WHEEL:-}" \
    "${ROBOTWIN_ROOT}/../LoopWAM/third_party/wheels/warp_lang-1.12.0-py3-none-manylinux_2_28_x86_64.whl" \
    "${ROBOTWIN_ROOT}/third_party/wheels/warp_lang-1.12.0-py3-none-manylinux_2_28_x86_64.whl"
  do
    if [[ -n "${cand}" && -f "${cand}" ]]; then
      wheel="${cand}"
      break
    fi
  done
  echo "[setup_pi05] pinning warp-lang==1.12.0 into .venv ..."
  if [[ -n "${wheel}" ]]; then
    echo "[setup_pi05] using local wheel: ${wheel}"
    "${VENV_PY}" -m pip install "${wheel}"
  else
    "${VENV_PY}" -m pip install "warp-lang==1.12.0"
  fi
}

# CuRobo into the π0.5 venv (official RoboTwin π0.5 flow).
if [[ "${SKIP_CUROBO}" == "1" ]]; then
  echo "[setup_pi05] skip CuRobo (SKIP_CUROBO=1)"
elif py_importable curobo; then
  echo "[setup_pi05] CuRobo already importable in .venv"
  install_warp_112
else
  CUROBO_DIR="${ROBOTWIN_ROOT}/envs/curobo"
  if [[ ! -d "${CUROBO_DIR}" ]]; then
    echo "[setup_pi05] ERROR: ${CUROBO_DIR} missing. Run prepare.sh or script/_install.sh first." >&2
    exit 1
  fi
  # .git may exist on a soft-linked tree while the host has no `git` binary.
  if ! command -v git >/dev/null 2>&1 || [[ ! -d "${CUROBO_DIR}/.git" ]]; then
    export_curobo_scm_pretend
  fi
  echo "[setup_pi05] pip install -e envs/curobo into .venv ..."
  (
    cd "${CUROBO_DIR}"
    "${VENV_PY}" -m pip install -e . --no-build-isolation
  )
  install_warp_112
fi

echo "[setup_pi05] verifying imports in .venv ..."
"${VENV_PY}" - <<'PY'
import jax
import openpi
print(f"  jax={jax.__version__}")
print("  openpi ok")
try:
    import curobo
    print("  curobo ok")
except ImportError as e:
    print(f"  curobo MISSING: {e}")
    raise SystemExit(1)
PY

echo "[setup_pi05] done."
echo "[setup_pi05] Collect/eval must use this interpreter (LoopWAM collect_pi05 auto-detects it):"
echo "  ${VENV_PY}"
echo "  # or: source ${VENV_DIR}/bin/activate"
