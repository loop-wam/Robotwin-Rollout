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

## Assets

Objects, textures, and embodiments are not in git. They come from the Hugging Face dataset [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0). If the download is rate-limited, run `huggingface-cli login` first.

```bash
bash script/_download_assets.sh
```

The script downloads `background_texture.zip`, `embodiments.zip`, and `objects.zip`, unpacks them into `assets/`, and rewrites embodiment configs so `${ASSETS_PATH}` is this checkout. After a move, run `python script/update_embodiment_config_path.py` again from the repository root.

```text
assets
├── background_texture
├── embodiments
│   └── aloha-agilex
│       ├── config.yml
│       └── ...
└── objects
```

`envs/curobo/` and everything under `assets/` except `assets/_download.py` stay untracked.
