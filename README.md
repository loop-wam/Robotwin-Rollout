# Robotwin-Rollout

RoboTwin 2.0 root used by [LoopWAM](https://github.com/loop-wam/LoopWAM) rollout collection.

Clone this repository and set `ROBOTWIN_ROOT` to the checkout. Collection scripts live in LoopWAM under `rollout_collect/`.

```bash
git clone https://github.com/loop-wam/Robotwin-Rollout.git
export ROBOTWIN_ROOT=$PWD/Robotwin-Rollout
```

The tree keeps the simulator layout (`envs/`, `description/`, `task_config/`, `policy/`, `code_gen/`, `script/`) and two collection entry points:

- `run_xvla.py` — X-VLA rollouts
- `script/eval_policy_gwx.py` — π0.5 and FastWAM rollouts. π0.5 loads `policy/pi05`. FastWAM loads its policy from the LoopWAM checkout, not from this tree.

Upstream simulator: [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin). Setup below follows the [RoboTwin 2.0 install guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html), using this tree's `script/` layout.

## Environment

Linux with an NVIDIA GPU. Python 3.10. CUDA 12.1 is the version RoboTwin 2.0 recommends. Rendering needs Vulkan, and `ffmpeg` must be on `PATH`.

Docker needs `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`. Without `graphics`, SAPIEN can segfault because Vulkan is missing.

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
vulkaninfo
ffmpeg -version

conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
cd Robotwin-Rollout
bash script/_install.sh
# π0.5 needs OpenPI / JAX (separate from the SAPIEN stack above):
bash script/setup_pi05.sh
```

`script/_install.sh` installs `script/requirements.txt` (SAPIEN, mplib, and the rest), optional pytorch3d, and [CuRobo v0.7.8](https://github.com/NVlabs/curobo) under `envs/curobo`. It then patches two installed files, the same way upstream does:

- `sapien/wrapper/urdf_loader.py`: open URDF and SRDF as UTF-8, and look for a `.srdf` suffix
- `mplib/planner.py`: drop the `or collide` check in the screw-plan failure condition

CuRobo then pins `warp-lang==1.12.0`. A newer Warp build breaks planner init (`'Robot' object has no attribute 'left_planner'`). pytorch3d is only required for point-cloud paths. If `git` is missing or that install fails and you do not use 3D observations, the rest of the simulator still runs (`SKIP_PYTORCH3D=1` to silence).

Soft-linked `envs/curobo` (via `prepare.sh`) may still contain a `.git` directory while the host has no `git` binary. `_install.sh` / `setup_pi05.sh` always set `SETUPTOOLS_SCM_PRETEND_VERSION` and `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO` in that case so editable install works. Re-runs skip packages that already import. Useful knobs: `SKIP_REQUIREMENTS=1`, `SKIP_PYTORCH3D=1`, `SKIP_CUROBO=1`, `FORCE=1`.

`script/setup_pi05.sh` runs `uv sync` into `policy/pi05/.venv` (Python 3.11; see the [RoboTwin π0.5 docs](https://robotwin-platform.github.io/doc/usage/Pi05.html)). It also installs CuRobo into that venv. Do not point `UV_PROJECT_ENVIRONMENT` at a conda prefix — uv cannot use it. LoopWAM `collect_pi05` auto-picks `$ROBOTWIN_ROOT/policy/pi05/.venv/bin/python` when present (override with `PI05_PYTHON`).

## Assets

Objects, textures, and embodiments are not in git.

**Option A — download** from [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0):

```bash
bash script/_download_assets.sh
```

**Option B — soft-link** a prebuilt RoboTwin tree (objects / embodiments / background_texture + cuRobo):

```bash
ROBOTWIN_PREBUILT=/path/to/prebuilt/RoboTwin bash prepare.sh
# or:
# ASSETS_SRC=/path/to/assets CUROBO_SRC=/path/to/envs/curobo bash prepare.sh
```

Expected layout after either option:

```text
assets
├── background_texture
├── embodiments
│   └── aloha-agilex
│       ├── config.yml
│       └── ...
└── objects
envs/curobo/   # install or soft-link
```

`envs/curobo/` and everything under `assets/` except `assets/_download.py` stay untracked.

## Example: π0.5 rollout collection

End-to-end flow using this repo as `ROBOTWIN_ROOT` and LoopWAM’s `rollout_collect/`.

Prerequisite: `bash script/_install.sh` (simulator) and `bash script/setup_pi05.sh` (OpenPI/JAX) in the same conda env you will use for collect.

### 1. Download π0.5 weights (Hugging Face)

Weights are on the dataset [loop-wam/loopwam](https://huggingface.co/datasets/loop-wam/loopwam) under `pi05_robotwin2/demo_clean/2000/` (`model.safetensors`, `metadata.pt`, `assets/.../norm_stats.json`). The dataset may be private — log in as the `loop-wam` user first if needed:

```bash
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
hf auth login

cd "$ROBOTWIN_ROOT"
# Optional HTTP proxy (script clears SOCKS ALL_PROXY so curl/httpx do not need socksio):
# export http_proxy=http://HOST:PORT https_proxy=http://HOST:PORT
bash download_pi05.sh
# HF_ENDPOINT=https://hf-mirror.com bash download_pi05.sh
# FORCE=1 bash download_pi05.sh
```

Installs into:

```text
policy/pi05/checkpoints/pi05_robotwin2/demo_clean/2000/
```

That path matches the defaults used by LoopWAM `collect_pi05.sh`:

| Variable | Default |
| --- | --- |
| `TRAIN_CONFIG_NAME` | `pi05_robotwin2` |
| `MODEL_NAME` / `TASK_CONFIG` | `demo_clean` |
| `CHECKPOINT_ID` | `2000` |

For `demo_randomized` scenes, keep this checkpoint tree and set `TASK_CONFIG=demo_randomized` (leave `MODEL_NAME=demo_clean` unless you trained a separate run name).

### 2. Task list

The repo ships an example `task_list.txt` (one RoboTwin task name per line). Edit it or point `collect_pi05.sh` at your own list.

### 3. Collect rollouts

```bash
export ROBOTWIN_ROOT=/path/to/Robotwin-Rollout
# git clone https://github.com/loop-wam/LoopWAM.git   # if needed

bash /path/to/LoopWAM/rollout_collect/collect_pi05.sh \
  task_list.txt \
  ./rollout_data/pi05
```

Useful overrides:

```bash
TASK_CONFIG=demo_clean \
EVAL_NUM_EPISODES=20 \
SEED=76 \
GPU=0 \
TRAIN_CONFIG_NAME=pi05_robotwin2 \
MODEL_NAME=demo_clean \
CHECKPOINT_ID=2000 \
  bash /path/to/LoopWAM/rollout_collect/collect_pi05.sh task_list.txt ./rollout_data/pi05
```

Dry-run (print commands only):

```bash
python /path/to/LoopWAM/rollout_collect/collect.py \
  --policy pi05 \
  --task-list task_list.txt \
  --save-root ./rollout_data/pi05 \
  --dry-run
```

Under the hood this runs `$ROBOTWIN_ROOT/script/eval_policy_gwx.py` with `policy/pi05/deploy_policy.yml`.

### 4. Output

```text
rollout_data/pi05/<task>/<task_config>/data/episode*.hdf5
rollout_data/pi05/<task>/<task_config>/scene_info.json
rollout_data/pi05/<task>/<task_config>/instructions/episode*.json
rollout_data/pi05/_logs/
```

Collection resumes from episodes already on disk.

### 5. Convert HDF5 → LeRobot (LoopWAM)

For mixed-quality training, convert the collect tree with LoopWAM’s `rollout_collect/convert_robotwin_to_lerobot.py` (run from the LoopWAM checkout so `src/fastwam` imports):

```bash
cd /path/to/LoopWAM

python rollout_collect/convert_robotwin_to_lerobot.py \
  --source-root /path/to/Robotwin-Rollout/rollout_data/pi05 \
  --output-dir ./data/robotwin2_0-ours/lerobot_format_new/pi05_processed_stitched \
  --task-configs demo_clean demo_randomized \
  --fps 50
```

Use `--resume` to append into an existing output dir, or `--merge-parts` to merge already-converted shards. Full flag list: `rollout_collect/README.md` in LoopWAM.
