#!/usr/bin/env bash
# Local soft-link helper for Robotwin-Rollout.
# This file is gitignored — keep machine-specific paths out of the repo.
#
# Links large assets + cuRobo into this checkout from a prebuilt RoboTwin tree:
#   ROBOTWIN_PREBUILT=/path/to/RoboTwin bash prepare.sh
#
# Or set sources separately:
#   ASSETS_SRC=/path/to/assets CUROBO_SRC=/path/to/envs/curobo bash prepare.sh
#
# Expected under ASSETS_SRC: objects/  embodiments/  background_texture/

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

ROBOTWIN_PREBUILT="${ROBOTWIN_PREBUILT:-}"
ASSETS_SRC="${ASSETS_SRC:-${ROBOTWIN_PREBUILT:+${ROBOTWIN_PREBUILT}/assets}}"
CUROBO_SRC="${CUROBO_SRC:-${ROBOTWIN_PREBUILT:+${ROBOTWIN_PREBUILT}/envs/curobo}}"

ASSETS_DST="${ROOT}/assets"
CUROBO_DST="${ROOT}/envs/curobo"

# Subdirs linked under assets/ (keeps tracked assets/_download.py).
ASSET_NAMES=(objects embodiments background_texture)

link_path() {
  local src="$1"
  local dst="$2"
  local marker="${3:-}"

  if [[ ! -e "${src}" ]]; then
    echo "[prepare] missing source: ${src}" >&2
    exit 1
  fi

  local src_real
  src_real="$(realpath "${src}")"

  if [[ -e "${dst}" || -L "${dst}" ]]; then
    local dst_real
    dst_real="$(realpath "${dst}")"
    if [[ "${dst_real}" == "${src_real}" ]]; then
      echo "[prepare] skip ${dst} (already -> ${dst_real})"
      return
    fi
    if [[ -n "${marker}" && -e "${dst}/${marker}" && ! -L "${dst}" ]]; then
      echo "[prepare] skip ${dst} (local tree already present)"
      return
    fi
    if [[ ! -L "${dst}" && -d "${dst}" && -z "${marker}" ]]; then
      # Non-empty real dir that is not a symlink target we manage.
      if [[ -n "$(ls -A "${dst}" 2>/dev/null || true)" ]]; then
        echo "[prepare] skip ${dst} (local directory already present)"
        return
      fi
    fi
    rm -rf "${dst}"
  fi

  mkdir -p "$(dirname "${dst}")"
  ln -sfn "${src_real}" "${dst}"
  echo "[prepare] ${dst} -> ${src_real}"
}

echo "[prepare] root=${ROOT}"

if [[ -z "${ASSETS_SRC}" ]]; then
  echo "[prepare] set ROBOTWIN_PREBUILT or ASSETS_SRC" >&2
  exit 1
fi
if [[ -z "${CUROBO_SRC}" ]]; then
  echo "[prepare] set ROBOTWIN_PREBUILT or CUROBO_SRC" >&2
  exit 1
fi

mkdir -p "${ASSETS_DST}"

for name in "${ASSET_NAMES[@]}"; do
  link_path "${ASSETS_SRC}/${name}" "${ASSETS_DST}/${name}"
done

link_path "${CUROBO_SRC}" "${CUROBO_DST}"

echo "[prepare] verifying ..."
ls "${ASSETS_DST}/objects" "${ASSETS_DST}/embodiments" "${CUROBO_DST}" >/dev/null
echo "[prepare] done"
echo "[prepare] assets under ${ASSETS_DST}/"
echo "[prepare] curobo at ${CUROBO_DST}"
