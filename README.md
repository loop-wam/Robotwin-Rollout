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
```

`script/_install.sh` installs `script/requirements.txt` (SAPIEN, mplib, and the rest), pytorch3d, and [CuRobo v0.7.8](https://github.com/NVlabs/curobo) under `envs/curobo`. It then patches two installed files, the same way upstream does:

- `sapien/wrapper/urdf_loader.py`: open URDF and SRDF as UTF-8, and look for a `.srdf` suffix
- `mplib/planner.py`: drop the `or collide` check in the screw-plan failure condition

CuRobo then pins `warp-lang==1.12.0`. A newer Warp build breaks planner init (`'Robot' object has no attribute 'left_planner'`). pytorch3d is only required for point-cloud paths. If that install fails and you do not use 3D observations, the rest of the simulator still runs.

If `bash script/_install.sh` stops partway, install the same pieces by hand:

```bash
python -m pip install -r script/requirements.txt
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
cd envs
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo
python -m pip install -e . --no-build-isolation
python -m pip install warp-lang==1.12.0
python -m pip install setuptools==69.5.1
```

Apply the two patches above, then rerun `bash script/_install.sh` only if the `sed` steps have not run yet.

For π0.5, also install the OpenPI stack under `policy/pi05` (see the [RoboTwin π0.5 docs](https://robotwin-platform.github.io/doc/usage/Pi05.html)):

```bash
conda activate RoboTwin
cd policy/pi05
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

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

Collection resumes from episodes already on disk. Convert to LeRobot with LoopWAM `rollout_collect/convert_robotwin_to_lerobot.py` when you are ready for mixed-quality training.
