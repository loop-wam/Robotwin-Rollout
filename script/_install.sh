#!/usr/bin/env bash
# RoboTwin simulator deps: requirements, optional pytorch3d, CuRobo, sapien/mplib patches.
#
# Safe to re-run. Skips pieces that already import.
# Soft-linked / git-less curobo trees (or hosts without a `git` binary) use
# SETUPTOOLS_SCM_PRETEND_VERSION*_ so editable install still works.
#
# Env knobs:
#   SKIP_REQUIREMENTS=1  — do not pip install script/requirements.txt
#   SKIP_PYTORCH3D=1     — skip pytorch3d (not needed for RGB-only π0.5)
#   SKIP_CUROBO=1        — skip CuRobo editable install
#   FORCE=1              — reinstall even if importable

set -euo pipefail

ROBOTWIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROBOTWIN_ROOT}"

SKIP_REQUIREMENTS="${SKIP_REQUIREMENTS:-0}"
SKIP_PYTORCH3D="${SKIP_PYTORCH3D:-0}"
SKIP_CUROBO="${SKIP_CUROBO:-0}"
FORCE="${FORCE:-0}"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

py_importable() {
  python -c "import $1" >/dev/null 2>&1
}

# setuptools-scm needs a pretend version when `git` is missing OR the tree has no usable SCM
# (softlink may still contain a .git dir from the source checkout).
export_curobo_scm_pretend() {
  local ver="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.7.8}"
  export SETUPTOOLS_SCM_PRETEND_VERSION="${ver}"
  # Normalized dist name for package `nvidia_curobo` (see envs/curobo/setup.cfg).
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO:-${ver}}"
  echo "[_install] SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}"
  echo "[_install] SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO}"
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
  echo "[_install] pinning warp-lang==1.12.0 ..."
  if [[ -n "${wheel}" ]]; then
    echo "[_install] using local wheel: ${wheel}"
    python -m pip install "${wheel}"
  else
    python -m pip install "warp-lang==1.12.0"
  fi
}

echo "[_install] root=${ROBOTWIN_ROOT}"
echo "[_install] python=$(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"

# ---- requirements (loose pins in requirements.txt avoid clobbering newer fastwam stacks) ----
if [[ "${SKIP_REQUIREMENTS}" == "1" ]]; then
  echo "[_install] skip requirements (SKIP_REQUIREMENTS=1)"
else
  echo "[_install] Installing script/requirements.txt ..."
  python -m pip install -r script/requirements.txt --upgrade-strategy only-if-needed
fi

# ---- pytorch3d (optional; needs git + network) ----
if [[ "${SKIP_PYTORCH3D}" == "1" ]]; then
  echo "[_install] skip pytorch3d (SKIP_PYTORCH3D=1)"
elif [[ "${FORCE}" != "1" ]] && py_importable pytorch3d; then
  echo "[_install] skip pytorch3d (already importable)"
elif ! have_cmd git; then
  echo "[_install] WARN: git not found — skip pytorch3d (RGB-only policies still work)."
  echo "[_install]        Install git and re-run, or set SKIP_PYTORCH3D=1 to silence."
else
  echo "[_install] Installing pytorch3d ..."
  python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
fi

# ---- sapien urdf_loader UTF-8 / .srdf patch ----
echo "[_install] Patching sapien urdf_loader.py ..."
SAPIEN_LOCATION="$(python -m pip show sapien 2>/dev/null | grep '^Location:' | awk '{print $2}')/sapien" || true
URDF_LOADER="${SAPIEN_LOCATION}/wrapper/urdf_loader.py"
if [[ -f "${URDF_LOADER}" ]]; then
  sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${URDF_LOADER}"
  echo "[_install] patched ${URDF_LOADER}"
else
  echo "[_install] WARN: sapien urdf_loader.py not found (is sapien installed?)"
fi

# ---- mplib planner: drop `or collide` in screw-plan failure ----
echo "[_install] Patching mplib planner.py ..."
MPLIB_LOCATION="$(python -m pip show mplib 2>/dev/null | grep '^Location:' | awk '{print $2}')/mplib" || true
PLANNER="${MPLIB_LOCATION}/planner.py"
if [[ -f "${PLANNER}" ]]; then
  sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${PLANNER}"
  echo "[_install] patched ${PLANNER}"
else
  echo "[_install] WARN: mplib planner.py not found (is mplib installed?)"
fi

# ---- CuRobo ----
install_curobo() {
  local curobo_dir="${ROBOTWIN_ROOT}/envs/curobo"
  mkdir -p "${ROBOTWIN_ROOT}/envs"

  if [[ ! -d "${curobo_dir}" ]]; then
    if ! have_cmd git; then
      echo "[_install] ERROR: envs/curobo missing and git not found." >&2
      echo "[_install]        Soft-link via: ROBOTWIN_PREBUILT=... bash prepare.sh" >&2
      echo "[_install]        Or install git and re-run to clone NVlabs/curobo@v0.7.8." >&2
      exit 1
    fi
    echo "[_install] Cloning CuRobo v0.7.8 into envs/curobo ..."
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "${curobo_dir}"
  else
    echo "[_install] using existing ${curobo_dir}"
  fi

  # Always pretend-version when git binary is missing, even if a .git directory exists
  # (setuptools-scm still shells out to `git` and fails with CommandNotFoundError).
  if ! have_cmd git || [[ ! -d "${curobo_dir}/.git" ]]; then
    export_curobo_scm_pretend
  fi

  echo "[_install] pip install -e envs/curobo ..."
  (
    cd "${curobo_dir}"
    python -m pip install -e . --no-build-isolation
  )

  install_warp_112
  python -m pip install "setuptools==69.5.1"
}

if [[ "${SKIP_CUROBO}" == "1" ]]; then
  echo "[_install] skip CuRobo (SKIP_CUROBO=1)"
elif [[ "${FORCE}" != "1" ]] && py_importable curobo; then
  echo "[_install] skip CuRobo (already importable)"
  install_warp_112 >/dev/null || true
else
  install_curobo
fi

echo "[_install] Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download or soft-link assets (script/_download_assets.sh or prepare.sh)."
echo -e "    2. For π0.5: bash script/setup_pi05.sh  (creates policy/pi05/.venv with JAX)."
echo "See README.md for more instructions."
