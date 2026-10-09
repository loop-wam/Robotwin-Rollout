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

Upstream simulator: [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin).
