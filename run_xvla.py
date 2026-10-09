import os
import sys
from pathlib import Path
# RoboTwin root comes from ROBOTWIN_ROOT. The fallback keeps older in-repo launches working.
_robotwin_root = os.environ.get("ROBOTWIN_ROOT", "").strip()
if _robotwin_root:
    robowin_root = Path(os.path.expanduser(os.path.expandvars(_robotwin_root))).resolve()
else:
    robowin_root = Path("/share/liangyingping/RTOpenPI-main")
# Ensure RoboTwin root is the first search in order to use envs
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))
# Ensure CWD is RoboTwin in order to load assets
os.chdir(robowin_root)

import argparse
import collections
from collections import Counter, defaultdict, deque
import logging
import os
import importlib
import numpy as np
import torch
import yaml
import PIL.Image as Image 
from tqdm import tqdm
import traceback
import json
import random
import imageio
import cv2
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float32)

ALL_TASKS = [
   "adjust_bottle", 
    "beat_block_hammer", 
    "blocks_ranking_rgb", 
    "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", 
    "move_can_pot", 
    "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", 
    "pick_dual_bottles", 
    "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", 
    "place_dual_shoes", 
    "place_empty_cup",
    "place_fan", "place_mouse_pad", 
    "place_object_basket", 
    "place_object_scale", "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", 
    "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle_horizontally", "shake_bottle", 
    "stack_blocks_three", "stack_blocks_two", 
    "stack_bowls_three", 
    "stack_bowls_two", "stamp_seal", "turn_switch"
]

print("Number of tasks evaluating:", len(ALL_TASKS))

def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6D_to_quat(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_quat()

def decode_image_from_bytes(camera_rgb_image):
    if isinstance(camera_rgb_image, (bytes, bytearray)): camera_rgb_image = np.frombuffer(camera_rgb_image, dtype=np.uint8)
    rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
    if rgb is None: 
        rgb = np.frombuffer(camera_rgb_image, dtype=np.uint8) 
        if rgb.size == 2764800: 
            rgb = rgb.reshape(720, 1280, 3) 
        elif rgb.size == 921600: 
            rgb = rgb.reshape(480, 640, 3)
    return Image.fromarray(rgb)



class LocalXVLAModel:
    """Run XVLA in-process instead of deploying a FastAPI service.

    This wrapper keeps the same `step(obs) -> action` interface as the old
    HTTP ClientModel, so the rest of the RoboTwin eval + rollout saving code can
    stay unchanged.

    This version directly supports the XVLA repo whose inference API is
    `model.generate_actions(...)`. It does not deploy FastAPI and does not use
    HTTP requests.
    """

    def __init__(
        self,
        model_path,
        processor_path=None,
        LoRA_path=None,
        device="cuda",
        xvla_repo_root=None,
        local_action_method="generate_actions",
        torch_dtype="float32",
        inference_steps=10,
        success_delay_steps=0,
    ):
        if xvla_repo_root is not None and str(xvla_repo_root).strip():
            xvla_repo_root = str(Path(xvla_repo_root).expanduser().resolve())
            if xvla_repo_root not in sys.path:
                sys.path.insert(0, xvla_repo_root)

        from models.modeling_xvla import XVLA
        from models.processing_xvla import XVLAProcessor

        if str(device).lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map.get(str(torch_dtype).lower(), torch.float32)

        processor_root = processor_path if processor_path else model_path
        print(f"Loading XVLAProcessor from: {processor_root}")
        self.processor = XVLAProcessor.from_pretrained(processor_root)

        print(f"Loading XVLA model from: {model_path}")
        self.model = XVLA.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
        ).to(self.device).to(self.torch_dtype)

        if LoRA_path is not None and str(LoRA_path).strip():
            print(f"Applying LoRA weights from: {LoRA_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                LoRA_path,
                torch_dtype=self.torch_dtype,
            ).to(self.device)

        self.model.eval()
        self.local_action_method = local_action_method
        self.inference_steps = int(inference_steps)
        self.model_dtype = next(self.model.parameters()).dtype

        # Success-delay state, compatible with the pi0.5 eval logic.
        # success_delay_steps=0 keeps the original immediate-success behavior.
        self.success_delay_steps = int(success_delay_steps)
        self.success_delay_active = False
        self.success_delay_remaining = 0

        self.vision_record = []
        self.instruction = None

    def set_instruction(self, instruction):
        self.instruction = instruction

    def reset_episode_state(self):
        """Reset per-episode state before each rollout."""
        self.success_delay_active = False
        self.success_delay_remaining = 0

    def return_vision_record(self):
        video = np.stack(self.vision_record)
        self.vision_record = []
        return video

    def _build_payload(self, obs):
        head_view = obs['observation']['head_camera']['rgb']
        left_view = obs['observation']['left_camera']['rgb']
        right_view = obs['observation']['right_camera']['rgb']
        front_view = obs['observation']['front_camera']['rgb']

        # Align shapes for stacking (e.g. if front_camera resolution differs).
        views = [head_view, left_view, right_view, front_view]
        target_hw = views[0].shape[:2]
        aligned = []
        for view in views:
            view = np.asarray(view)
            if view.shape[:2] != target_hw:
                view = cv2.resize(view, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
            aligned.append(view)
        head_view, left_view, right_view, front_view = aligned

        image_obs = np.stack([head_view, left_view, right_view, front_view])[None, ]
        self.vision_record.append(image_obs)

        left_ee = np.expand_dims(np.array(obs["endpose"]["left_endpose"]), axis=0)
        right_ee = np.expand_dims(np.array(obs["endpose"]["right_endpose"]), axis=0)
        left_grip = np.expand_dims(np.array(obs["endpose"]["left_gripper"]), axis=0)
        right_grip = np.expand_dims(np.array(obs["endpose"]["right_gripper"]), axis=0)
        left_grip = 1 - left_grip * 2
        right_grip = 1 - right_grip * 2

        abs_eef = np.concatenate([
            left_ee[:, :3],
            quat_to_rotate6D(left_ee[:, 3:]),
            left_grip[:, None],
            right_ee[:, :3],
            quat_to_rotate6D(right_ee[:, 3:]),
            right_grip[:, None],
        ], axis=-1)

        return {
            "domain_id": 6,
            "proprio": abs_eef.squeeze(0).astype(np.float32),
            "language_instruction": self.instruction,
            "image0": head_view,
            "image1": left_view,
            "image2": right_view,
            "image3": front_view,
        }

    @staticmethod
    def _move_to_device(x, device):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, dict):
            return {k: LocalXVLAModel._move_to_device(v, device) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(LocalXVLAModel._move_to_device(v, device) for v in x)
        return x

    @staticmethod
    def _extract_action(output):
        """Normalize model output into a numpy action array."""
        if isinstance(output, dict):
            for key in ("action", "actions", "pred_action", "pred_actions"):
                if key in output:
                    return LocalXVLAModel._extract_action(output[key])
        if hasattr(output, "action"):
            return LocalXVLAModel._extract_action(output.action)
        if torch.is_tensor(output):
            output = output.detach().cpu().float().numpy()
        output = np.asarray(output)
        if output.ndim == 1:
            output = output[None, :]
        return output

    def _call_named_method(self, method_name, payload):
        fn = getattr(self.model, method_name)

        # Try signatures used by common VLA repos. The first successful call wins.
        call_patterns = [
            lambda: fn(processor=self.processor, **payload),
            lambda: fn(self.processor, **payload),
            lambda: fn(**payload),
            lambda: fn(payload, processor=self.processor),
            lambda: fn(payload),
        ]
        last_error = None
        with torch.no_grad():
            for call in call_patterns:
                try:
                    return call()
                except TypeError as e:
                    last_error = e
                    continue
        raise TypeError(
            f"Method `{method_name}` exists, but none of the supported call signatures matched. "
            f"Last TypeError: {last_error}"
        )

    def _call_via_processor_forward(self, payload):
        """Fallback for HF-style processor + model.forward APIs.

        This may need small adjustment if your processor uses custom argument
        names. Prefer passing `--local_action_method` if XVLA exposes a native
        action method.
        """
        images = [
            Image.fromarray(payload["image0"]),
            Image.fromarray(payload["image1"]),
            Image.fromarray(payload["image2"]),
        ]
        processor_call_patterns = [
            lambda: self.processor(
                images=images,
                text=payload["language_instruction"],
                proprio=payload["proprio"],
                domain_id=payload["domain_id"],
                return_tensors="pt",
            ),
            lambda: self.processor(
                images=images,
                instruction=payload["language_instruction"],
                state=payload["proprio"],
                return_tensors="pt",
            ),
            lambda: self.processor(
                images=images,
                text=payload["language_instruction"],
                return_tensors="pt",
            ),
        ]
        last_error = None
        inputs = None
        for call in processor_call_patterns:
            try:
                inputs = call()
                break
            except TypeError as e:
                last_error = e
                continue
        if inputs is None:
            raise TypeError(f"Processor fallback failed. Last TypeError: {last_error}")

        inputs = self._move_to_device(inputs, self.device)
        with torch.no_grad():
            if hasattr(self.model, "generate"):
                try:
                    return self.model.generate(**inputs)
                except TypeError:
                    pass
            return self.model(**inputs)

    def _to_model_tensor(self, value):
        """Move tensors to XVLA device. Cast only floating tensors to model dtype."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.is_floating_point():
            return value.to(device=self.device, dtype=self.model_dtype)
        return value.to(device=self.device)

    def _local_infer(self, payload):
        """Run XVLA.generate_actions locally.

        Expected payload keys are the same as the old /act HTTP payload, but raw
        numpy arrays are used instead of json_numpy strings:
          image0/image1/image2, language_instruction, proprio, domain_id.
        """
        method_name = self.local_action_method or "generate_actions"
        if method_name == "auto":
            method_name = "generate_actions"

        if method_name != "generate_actions":
            # Keep this escape hatch for custom repos, but the default path below
            # is the correct one for the XVLA implementation you showed.
            return self._call_named_method(method_name, payload)

        if not hasattr(self.model, "generate_actions"):
            raise RuntimeError("The loaded XVLA model does not have generate_actions(...).")

        images = [
            Image.fromarray(np.asarray(payload["image0"]).astype(np.uint8)),
            Image.fromarray(np.asarray(payload["image1"]).astype(np.uint8)),
            Image.fromarray(np.asarray(payload["image2"]).astype(np.uint8)),
        ]

        # XVLAProcessor.__call__ internally sets return_tensors="pt" for both
        # tokenizer and image_processor, so do not pass return_tensors here.
        inputs = self.processor(
            images=images,
            language_instruction=payload["language_instruction"],
        )

        required = {"input_ids", "image_input", "image_mask"}
        missing = required.difference(inputs.keys())
        if missing:
            raise RuntimeError(f"XVLAProcessor output missing keys: {sorted(missing)}")

        inputs = {k: self._to_model_tensor(v) for k, v in inputs.items()}

        proprio = torch.as_tensor(np.asarray(payload["proprio"]), dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        proprio = self._to_model_tensor(proprio)

        domain_id = torch.tensor(
            [int(payload.get("domain_id", 6))],
            dtype=torch.long,
            device=self.device,
        )

        steps = int(payload.get("steps", self.inference_steps))

        with torch.no_grad():
            action = self.model.generate_actions(
                input_ids=inputs["input_ids"],
                image_input=inputs["image_input"],
                image_mask=inputs["image_mask"],
                domain_id=domain_id,
                proprio=proprio,
                steps=steps,
            )

        return action.squeeze(0).float().cpu().numpy()

    def step(self, obs):
        payload = self._build_payload(obs)
        output = self._local_infer(payload)
        action = self._extract_action(output)
        return action

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args



def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _rollout_dir(rollout_root, task_name, task_config):
    return Path(str(rollout_root)) / str(task_name) / str(task_config)


def _count_rollout_episodes(rollout_dir):
    data_dir = rollout_dir / "data"
    if not data_dir.is_dir():
        return 0
    return len(list(data_dir.glob("episode*.hdf5")))


def _count_4d_episodes(rollout_dir):
    if not rollout_dir.is_dir():
        return 0
    return len([p for p in rollout_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")])


def _load_rollout_resume_state(rollout_root, task_name, task_config, save_4d=False):
    rollout_dir = _rollout_dir(rollout_root, task_name, task_config)
    if parse_bool(save_4d):
        episode_offset = _count_4d_episodes(rollout_dir)
    else:
        episode_offset = _count_rollout_episodes(rollout_dir)
    last_seed = None
    cursor_path = rollout_dir / "eval_seed_cursor.json"
    if cursor_path.exists():
        with open(cursor_path, "r", encoding="utf-8") as f:
            cursor = json.load(f)
        last_seed = int(cursor["last_seed"])
        episode_offset = int(cursor.get("episode_count", episode_offset))
    return episode_offset, last_seed


def _save_rollout_cursor(rollout_root, task_name, task_config, *, last_seed, episode_count):
    rollout_dir = _rollout_dir(rollout_root, task_name, task_config)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    cursor_path = rollout_dir / "eval_seed_cursor.json"
    with open(cursor_path, "w", encoding="utf-8") as f:
        json.dump(
            {"last_seed": int(last_seed), "episode_count": int(episode_count)},
            f,
            indent=2,
        )


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _ensure_4d_data_type(args):
    """Force modalities needed for 4D training export."""
    data_type = dict(args.get("data_type") or {})
    data_type["rgb"] = True
    data_type["depth"] = True
    data_type["endpose"] = True
    data_type["qpos"] = True
    if parse_bool(args.get("save_scene_flow", True)):
        data_type["scene_flow"] = True
        data_type["world_xyz"] = True
        data_type["actor_seg"] = True
    args["data_type"] = data_type
    return data_type


def _resolve_save_interval(args, TASK_ENV=None):
    if args.get("save_interval") is not None and str(args.get("save_interval")).strip() != "":
        interval = int(args.get("save_interval"))
    elif args.get("save_frame_interval") is not None and str(args.get("save_frame_interval")).strip() != "":
        interval = int(args.get("save_frame_interval"))
    elif TASK_ENV is not None and getattr(TASK_ENV, "save_freq", None) not in (None, -1):
        interval = int(TASK_ENV.save_freq)
    else:
        interval = 1
    if interval <= 0:
        raise ValueError(f"save_interval must be > 0, got {interval}")
    return interval


def _list_4d_frame_indices(episode_dir):
    episode_dir = Path(episode_dir)
    rgb_root = episode_dir / "rgb"
    if not rgb_root.is_dir():
        return []
    cam_dirs = [p for p in rgb_root.iterdir() if p.is_dir()]
    if not cam_dirs:
        return []
    indices = set()
    for cam_dir in cam_dirs:
        for p in cam_dir.glob("*.jpg"):
            indices.add(int(p.stem))
    return sorted(indices)


def _save_entity_poses_cache(path, entity_poses):
    if not entity_poses:
        np.savez_compressed(str(path))
        return
    np.savez_compressed(str(path), **{str(k): np.asarray(v) for k, v in entity_poses.items()})


def _load_entity_poses_cache(path):
    if not Path(path).is_file():
        return {}
    data = np.load(str(path))
    return {int(k): np.asarray(data[k]) for k in data.files}


def _prune_4d_episode(episode_dir, interval, *, save_scene_flow, camera_buffers, camera_names):
    """Keep every N-th action frame; delete the rest after the episode ends."""
    episode_dir = Path(episode_dir)
    indices = _list_4d_frame_indices(episode_dir)
    if not indices:
        return 0, []

    kept = [i for i in indices if i % int(interval) == 0]
    to_delete = [i for i in indices if i not in kept]

    for idx in to_delete:
        for cam_name in camera_names:
            rgb_path = episode_dir / "rgb" / cam_name / f"{idx:06d}.jpg"
            depth_path = episode_dir / "depth" / cam_name / f"{idx:06d}.png"
            if rgb_path.is_file():
                rgb_path.unlink()
            if depth_path.is_file():
                depth_path.unlink()
        flow_cache_pose = episode_dir / ".flow_cache" / "entity_poses" / f"{idx:06d}.npz"
        if flow_cache_pose.is_file():
            flow_cache_pose.unlink()
        for cam_name in camera_names:
            cache_path = episode_dir / ".flow_cache" / cam_name / f"{idx:06d}.npz"
            if cache_path.is_file():
                cache_path.unlink()

    pose_path = episode_dir / "pose" / "pose.jsonl"
    if pose_path.is_file():
        kept_set = set(kept)
        kept_lines = []
        with open(pose_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if int(entry.get("frame", -1)) in kept_set:
                    kept_lines.append(entry)
        with open(pose_path, "w", encoding="utf-8") as f:
            for entry in kept_lines:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    for cam_name in camera_names:
        cam_param_dir = episode_dir / "camera" / cam_name
        cam_param_dir.mkdir(parents=True, exist_ok=True)
        buffers = (camera_buffers or {}).get(cam_name, {})
        if buffers.get("intrinsic_cv") is not None:
            np.save(str(cam_param_dir / "intrinsic.npy"), buffers["intrinsic_cv"])
        if buffers.get("extrinsic_cv"):
            ext = np.stack(buffers["extrinsic_cv"], axis=0)
            np.save(str(cam_param_dir / "extrinsic_cv.npy"), ext[np.asarray(kept, dtype=int)])
        if buffers.get("cam2world_gl"):
            c2w = np.stack(buffers["cam2world_gl"], axis=0)
            np.save(str(cam_param_dir / "cam2world_gl.npy"), c2w[np.asarray(kept, dtype=int)])

    if save_scene_flow:
        for cam_name in camera_names:
            flow_dir = episode_dir / "scene_flow" / cam_name
            flow_dir.mkdir(parents=True, exist_ok=True)
            for p in flow_dir.glob("*.npy"):
                p.unlink()
            cache_dir = episode_dir / ".flow_cache" / cam_name
            entity_dir = episode_dir / ".flow_cache" / "entity_poses"
            for src_idx, dst_idx in zip(kept[:-1], kept[1:]):
                cache_path = cache_dir / f"{src_idx:06d}.npz"
                if not cache_path.is_file():
                    continue
                prev = np.load(str(cache_path))
                poses_t = _load_entity_poses_cache(entity_dir / f"{src_idx:06d}.npz")
                poses_t1 = _load_entity_poses_cache(entity_dir / f"{dst_idx:06d}.npz")
                flow = _compute_scene_flow_cam(
                    prev["world_xyz"],
                    prev["valid"],
                    prev["actor_seg"],
                    poses_t,
                    poses_t1,
                    prev["extrinsic_cv"],
                )
                np.save(str(flow_dir / f"{src_idx:06d}.npy"), flow.astype(np.float16))

    import shutil
    shutil.rmtree(episode_dir / ".flow_cache", ignore_errors=True)
    return len(kept), kept


def _init_4d_episode(TASK_ENV, args, *, episode_idx):
    """Prepare per-episode folder for 4D frame export.

    During rollout: save one frame per policy action (000000, 000001, ...).
    After the episode: keep every save_interval-th frame and delete the rest.
    """
    rollout_root = args.get("rollout_save_root")
    if not parse_bool(args.get("save_4d", False)) or rollout_root is None or str(rollout_root).strip() == "":
        return None

    data_type = _ensure_4d_data_type(args)
    if getattr(TASK_ENV, "data_type", None) is not None:
        TASK_ENV.data_type = data_type

    rollout_dir = _rollout_dir(rollout_root, args["task_name"], args["task_config"])
    episode_dir = rollout_dir / f"episode_{int(episode_idx):04d}"
    if episode_dir.exists():
        import shutil
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "rgb").mkdir(exist_ok=True)
    (episode_dir / "depth").mkdir(exist_ok=True)
    (episode_dir / "camera").mkdir(exist_ok=True)
    (episode_dir / "pose").mkdir(exist_ok=True)

    save_scene_flow = parse_bool(args.get("save_scene_flow", True))
    if save_scene_flow:
        (episode_dir / "scene_flow").mkdir(exist_ok=True)
        (episode_dir / ".flow_cache").mkdir(exist_ok=True)

    save_interval = _resolve_save_interval(args, TASK_ENV)

    TASK_ENV._4d_episode_dir = str(episode_dir)
    TASK_ENV._4d_frame_idx = 0
    TASK_ENV._4d_camera_buffers = {}
    TASK_ENV._4d_camera_names = []
    TASK_ENV._4d_save_scene_flow = save_scene_flow
    TASK_ENV._4d_save_interval = save_interval
    return episode_dir


def _save_4d_frame(TASK_ENV, obs):
    """Save one policy-action snapshot with sequential frame index."""
    episode_dir = getattr(TASK_ENV, "_4d_episode_dir", None)
    if episode_dir is None:
        return
    episode_dir = Path(episode_dir)
    frame_idx = int(getattr(TASK_ENV, "_4d_frame_idx", 0))
    observation = obs.get("observation") or {}
    entity_poses = obs.get("entity_poses") or {}
    save_scene_flow = bool(getattr(TASK_ENV, "_4d_save_scene_flow", False))

    camera_names = []
    for cam_name, cam_data in observation.items():
        if not isinstance(cam_data, dict):
            continue
        has_rgb = "rgb" in cam_data
        has_depth = "depth" in cam_data
        has_cam = any(k in cam_data for k in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl"))
        if not (has_rgb or has_depth or has_cam):
            continue
        camera_names.append(cam_name)

        cam_rgb_dir = episode_dir / "rgb" / cam_name
        cam_depth_dir = episode_dir / "depth" / cam_name
        cam_param_dir = episode_dir / "camera" / cam_name
        cam_rgb_dir.mkdir(parents=True, exist_ok=True)
        cam_depth_dir.mkdir(parents=True, exist_ok=True)
        cam_param_dir.mkdir(parents=True, exist_ok=True)

        if has_rgb:
            rgb = np.asarray(cam_data["rgb"])
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 and rgb.shape[-1] == 3 else rgb
            cv2.imwrite(str(cam_rgb_dir / f"{frame_idx:06d}.jpg"), bgr)

        if has_depth:
            depth_mm = np.asarray(cam_data["depth"], dtype=np.float64)
            depth_mm_u16 = np.clip(
                np.rint(np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)),
                0,
                65535,
            ).astype(np.uint16)
            depth_path = cam_depth_dir / f"{frame_idx:06d}.png"
            if not cv2.imwrite(str(depth_path), depth_mm_u16):
                raise RuntimeError(f"cv2.imwrite failed: {depth_path}")

        buffers = TASK_ENV._4d_camera_buffers.setdefault(
            cam_name,
            {"intrinsic_cv": None, "extrinsic_cv": [], "cam2world_gl": []},
        )
        if buffers["intrinsic_cv"] is None and "intrinsic_cv" in cam_data:
            buffers["intrinsic_cv"] = np.asarray(cam_data["intrinsic_cv"], dtype=np.float64)
        extrinsic_cv = None
        if "extrinsic_cv" in cam_data:
            extrinsic_cv = np.asarray(cam_data["extrinsic_cv"], dtype=np.float64)
            buffers["extrinsic_cv"].append(extrinsic_cv)
        if "cam2world_gl" in cam_data:
            buffers["cam2world_gl"].append(np.asarray(cam_data["cam2world_gl"], dtype=np.float64))

        if save_scene_flow and "world_xyz" in cam_data and "actor_seg" in cam_data and extrinsic_cv is not None:
            world_xyz = np.asarray(cam_data["world_xyz"], dtype=np.float32)
            valid = np.asarray(cam_data["valid"], dtype=bool) if "valid" in cam_data else (np.linalg.norm(world_xyz, axis=-1) > 0)
            actor_seg = np.asarray(cam_data["actor_seg"])
            cache_dir = episode_dir / ".flow_cache" / cam_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(cache_dir / f"{frame_idx:06d}.npz"),
                world_xyz=world_xyz,
                valid=valid,
                actor_seg=actor_seg,
                extrinsic_cv=extrinsic_cv,
            )

    if save_scene_flow:
        entity_dir = episode_dir / ".flow_cache" / "entity_poses"
        entity_dir.mkdir(parents=True, exist_ok=True)
        _save_entity_poses_cache(entity_dir / f"{frame_idx:06d}.npz", entity_poses)

    if camera_names and not getattr(TASK_ENV, "_4d_camera_names", None):
        TASK_ENV._4d_camera_names = camera_names

    pose_entry = {
        "frame": frame_idx,
        "endpose": _to_serializable(obs.get("endpose", {})),
        "joint_action": _to_serializable(obs.get("joint_action", {})),
    }
    with open(episode_dir / "pose" / "pose.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(pose_entry, ensure_ascii=False) + "\n")

    TASK_ENV._4d_frame_idx = frame_idx + 1


def _compute_scene_flow_cam(
    world_xyz_t,
    valid_t,
    actor_seg_t,
    poses_t,
    poses_t1,
    extrinsic_cv_t,
):
    """Rigid GT scene flow in camera-t coordinates (meters)."""
    H, W = actor_seg_t.shape
    flow_world = np.zeros((H, W, 3), dtype=np.float32)
    valid = np.asarray(valid_t, dtype=bool)
    seg = np.asarray(actor_seg_t)
    xyz = np.asarray(world_xyz_t, dtype=np.float64)

    unique_ids = np.unique(seg[valid]) if np.any(valid) else []
    for uid in unique_ids:
        uid_i = int(uid)
        if uid_i == 0:
            continue
        if uid_i not in poses_t or uid_i not in poses_t1:
            continue
        mask = valid & (seg == uid)
        if not np.any(mask):
            continue
        T0 = poses_t[uid_i]
        T1 = poses_t1[uid_i]
        G = T1 @ np.linalg.inv(T0)
        pts = xyz[mask]
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        pts1 = pts_h @ G.T
        flow_world[mask] = (pts1[:, :3] - pts).astype(np.float32)

    R_w2c = np.asarray(extrinsic_cv_t[:3, :3], dtype=np.float64)
    flow_cam = np.einsum("ij,hwj->hwi", R_w2c, flow_world.astype(np.float64)).astype(np.float32)
    flow_cam[~valid] = 0.0
    return flow_cam


def _finalize_4d_episode(
    TASK_ENV,
    args,
    *,
    episode_dir,
    episode_idx,
    instruction,
    episode_info,
    success,
    env_seed=None,
):
    if episode_dir is None:
        return
    episode_dir = Path(episode_dir)

    if parse_bool(args.get("rollout_save_only_success", False)) and not success:
        import shutil
        shutil.rmtree(episode_dir, ignore_errors=True)
        TASK_ENV._4d_episode_dir = None
        TASK_ENV._4d_camera_buffers = {}
        TASK_ENV._4d_frame_idx = 0
        return

    num_raw_frames = int(getattr(TASK_ENV, "_4d_frame_idx", 0))
    if num_raw_frames <= 0:
        import shutil
        shutil.rmtree(episode_dir, ignore_errors=True)
        TASK_ENV._4d_episode_dir = None
        return

    save_interval = int(getattr(TASK_ENV, "_4d_save_interval", args.get("save_interval", 1)))
    save_scene_flow = bool(getattr(TASK_ENV, "_4d_save_scene_flow", True))
    camera_buffers = dict(getattr(TASK_ENV, "_4d_camera_buffers", {}) or {})
    camera_names = list(getattr(TASK_ENV, "_4d_camera_names", []) or [])

    num_frames, kept_frames = _prune_4d_episode(
        episode_dir,
        save_interval,
        save_scene_flow=save_scene_flow,
        camera_buffers=camera_buffers,
        camera_names=camera_names,
    )
    if num_frames <= 0:
        import shutil
        shutil.rmtree(episode_dir, ignore_errors=True)
        TASK_ENV._4d_episode_dir = None
        return

    meta = {
        "episode_idx": int(episode_idx),
        "instruction": instruction,
        "success": bool(success),
        "num_frames": num_frames,
        "num_raw_action_frames": num_raw_frames,
        "kept_frame_indices": kept_frames,
        "camera_names": camera_names,
        "depth_format": "uint16_png",
        "depth_unit": "millimeter",
        "depth_note": (
            "Stored as 16-bit PNG in millimeters. "
            "Load: depth_m = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0"
        ),
        "depth_file_ext": ".png",
        "rgb_note": "Saved as JPEG via OpenCV (BGR). Use cv2.imread + COLOR_BGR2RGB for RGB float arrays.",
        "save_mode": "save_all_actions_then_prune",
        "save_interval": save_interval,
        "frame_index_is_action_index": True,
        "camera_convention": {
            "intrinsic_cv": "3x3 OpenCV K",
            "extrinsic_cv": "4x4 OpenCV world-to-camera (Sapien get_extrinsic_matrix)",
            "cam2world_gl": "4x4 OpenGL camera-to-world (Sapien get_model_matrix)",
        },
        "pose_fields": {
            "endpose": "left/right end-effector pose + gripper",
            "joint_action": "left/right arm qpos + gripper + vector",
        },
        "info": _to_serializable(episode_info),
    }
    if save_scene_flow:
        meta.update(
            {
                "scene_flow_format": "float16_npy",
                "scene_flow_unit": "meter",
                "scene_flow_frame": "camera_t",
                "scene_flow_file_ext": ".npy",
                "scene_flow_note": (
                    "scene_flow/{cam}/{t:06d}.npy is float16 (H,W,3) flow from kept action frame t "
                    "to the next kept action frame, in camera-t OpenCV coordinates (meters)."
                ),
                "scene_flow_num_files": max(0, num_frames - 1),
            }
        )
    if env_seed is not None:
        meta["env_seed"] = int(env_seed)

    with open(episode_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    rollout_dir = episode_dir.parent
    index_path = rollout_dir / "scene_info.json"
    info_db = {}
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            info_db = json.load(f)
    info_db[f"episode_{episode_idx}"] = {
        "path": episode_dir.name,
        "instruction": instruction,
        "success": bool(success),
        "num_frames": num_frames,
        "num_raw_action_frames": num_raw_frames,
        "env_seed": int(env_seed) if env_seed is not None else None,
        "save_interval": save_interval,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(info_db, f, ensure_ascii=False, indent=2)

    TASK_ENV._4d_episode_dir = None
    TASK_ENV._4d_camera_buffers = {}
    TASK_ENV._4d_camera_names = []
    TASK_ENV._4d_frame_idx = 0


def _init_rollout_recording(TASK_ENV, args, *, episode_idx):
    """Enable RoboTwin built-in data recording and/or 4D frame export.

    RoboTwin hdf5 layout (save_rollout):
      rollout_save_root/task_name/task_config/data/episode*.hdf5
      rollout_save_root/task_name/task_config/scene_info.json
      rollout_save_root/task_name/task_config/instructions/episode*.json

    4D layout (save_4d):
      rollout_save_root/task_name/task_config/episode_XXXX/{rgb,depth,camera,pose,meta.json}
    """
    rollout_root = args.get("rollout_save_root")
    save_rollout = parse_bool(args.get("save_rollout", False))
    save_4d = parse_bool(args.get("save_4d", False))
    if (not save_rollout and not save_4d) or rollout_root is None or str(rollout_root).strip() == "":
        return None

    rollout_dir = Path(str(rollout_root)) / str(args["task_name"]) / str(args["task_config"])
    rollout_dir.mkdir(parents=True, exist_ok=True)

    if save_4d:
        _init_4d_episode(TASK_ENV, args, episode_idx=episode_idx)

    if save_rollout:
        # Ensure depth is in pkl/hdf5 when collecting together with 4D.
        if save_4d:
            _ensure_4d_data_type(args)
            if getattr(TASK_ENV, "data_type", None) is not None:
                TASK_ENV.data_type = args["data_type"]
        TASK_ENV.save_data = True
        TASK_ENV.save_dir = str(rollout_dir)
        TASK_ENV.ep_num = int(episode_idx)
        TASK_ENV.FRAME_IDX = 0
    else:
        # 4D-only mode: do not write RoboTwin pkl/hdf5 cache.
        TASK_ENV.save_data = False

    return rollout_dir


def _record_rollout_observation(TASK_ENV, args, obs=None):
    """Save 4D data after each action; RoboTwin hdf5 uses _take_picture when enabled."""
    rollout_root = args.get("rollout_save_root")
    if rollout_root is None:
        return
    save_rollout = parse_bool(args.get("save_rollout", False))
    save_4d = parse_bool(args.get("save_4d", False))
    if not save_rollout and not save_4d:
        return

    if save_4d and getattr(TASK_ENV, "_4d_episode_dir", None) is not None:
        if obs is None:
            obs = TASK_ENV.get_obs()
        _save_4d_frame(TASK_ENV, obs)
        if save_rollout and getattr(TASK_ENV, "save_data", False):
            TASK_ENV._take_picture()
        return

    if save_rollout and getattr(TASK_ENV, "save_data", False):
        TASK_ENV._take_picture()


def _finalize_rollout_recording(
    TASK_ENV,
    args,
    *,
    rollout_dir,
    episode_idx,
    instruction,
    episode_info,
    success,
    env_seed=None,
):
    if rollout_dir is None:
        return

    save_4d = parse_bool(args.get("save_4d", False))
    save_rollout = parse_bool(args.get("save_rollout", False))

    if save_4d and getattr(TASK_ENV, "_4d_episode_dir", None) is not None:
        _finalize_4d_episode(
            TASK_ENV,
            args,
            episode_dir=TASK_ENV._4d_episode_dir,
            episode_idx=episode_idx,
            instruction=instruction,
            episode_info=episode_info,
            success=success,
            env_seed=env_seed,
        )

    if not save_rollout or not getattr(TASK_ENV, "save_data", False):
        return

    if parse_bool(args.get("rollout_save_only_success", False)) and not success:
        TASK_ENV.save_data = False
        TASK_ENV.FRAME_IDX = 0
        TASK_ENV.remove_data_cache()
        return

    if getattr(TASK_ENV, "FRAME_IDX", 0) <= 0:
        TASK_ENV.save_data = False
        return

    TASK_ENV.merge_pkl_to_hdf5_video()
    TASK_ENV.remove_data_cache()

    info_path = rollout_dir / "scene_info.json"
    info_db = {}
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info_db = json.load(f)

    entry = {
        "info": episode_info,
        "instruction": instruction,
        "success": bool(success),
    }
    if env_seed is not None:
        entry["env_seed"] = int(env_seed)
    info_db[f"episode_{episode_idx}"] = entry

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_db, f, ensure_ascii=False, indent=4)

    instructions_dir = rollout_dir / "instructions"
    instructions_dir.mkdir(parents=True, exist_ok=True)
    with open(instructions_dir / f"episode{episode_idx}.json", "w", encoding="utf-8") as f:
        json.dump({"seen": [instruction], "unseen": []}, f, indent=2, ensure_ascii=False)

    TASK_ENV.save_data = False
    TASK_ENV.FRAME_IDX = 0

def load_env(task_name, task_config):
    CONFIGS_PATH = "task_config"
    task = class_decorator(task_name)

    config_path = os.path.join(CONFIGS_PATH, f"{task_config}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("missing embodiment files")
        return robot_file

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    args["embodiment_name"] = "+".join(embodiment_type) if len(embodiment_type) > 1 else embodiment_type[0]
    args["task_config"] = task_config

    return task, args



def _mark_eval_success(TASK_ENV):
    """Mark eval success in the same spirit as the patched RoboTwin take_action."""
    TASK_ENV.eval_success = True
    try:
        TASK_ENV.get_obs()  # update now_obs
    except Exception:
        pass

    if getattr(TASK_ENV, "eval_video_path", None) is not None:
        try:
            TASK_ENV.eval_video_ffmpeg.stdin.write(
                TASK_ENV.now_obs["observation"]["head_camera"]["rgb"].tobytes()
            )
        except Exception:
            pass


def _take_action_delay_aware(TASK_ENV, action, *, action_type="ee", mark_success=True):
    """Call take_action with mark_success when the env supports it.

    Older RoboTwin envs may not have the `mark_success` argument. In that case,
    success_delay_steps cannot be implemented faithfully because take_action will
    terminate immediately on success. We fall back to the old behavior instead of
    crashing.
    """
    try:
        return TASK_ENV.take_action(action, action_type=action_type, mark_success=mark_success)
    except TypeError as e:
        if "mark_success" not in str(e):
            raise
        if mark_success is False:
            print(
                "warning: TASK_ENV.take_action(...) does not support mark_success=False; "
                "success_delay_steps will behave like 0 for this environment."
            )
        return TASK_ENV.take_action(action, action_type=action_type)


def _convert_model_actions_to_ee_actions(actions):
    """Convert XVLA action output to RoboTwin ee action format.

    Supported inputs:
      - (B, 20): [left xyz, left rot6d, left grip, right xyz, right rot6d, right grip]
                 converted to (B, 16) ee actions.
      - (B, 16): already RoboTwin ee action format, returned as-is.
      - (20,) or (16,): automatically batched.
    """
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]

    if actions.shape[-1] == 16:
        return actions

    if actions.shape[-1] < 20:
        raise ValueError(
            f"Expected model action dim 20 for rot6d format or 16 for ee format, got {actions.shape}."
        )

    left_xyz = actions[:, :3]
    left_rotate6d = actions[:, 3:9]
    left_gripper = actions[:, 9:10]
    left_quat = rotate6D_to_quat(left_rotate6d)
    left_grip = 1 - 2 * (left_gripper > 0.7)
    left_new = np.concatenate([left_xyz, left_quat, left_grip], axis=1)

    right_xyz = actions[:, 10:13].reshape(-1, 3)
    right_rotate6d = actions[:, 13:19].reshape(-1, 6)
    right_quat = rotate6D_to_quat(right_rotate6d)
    right_gripper = actions[:, 19:20].reshape(-1, 1)
    right_grip = 1 - 2 * (right_gripper > 0.7)
    right_new = np.concatenate([right_xyz, right_quat, right_grip], axis=1)

    return np.concatenate([left_new, right_new], axis=1)


def _rollout(env, policy, args=None):
    """Run one episode.

    success_delay_steps behavior:
      - 0: original behavior. take_action marks success immediately.
      - >0: take_action(mark_success=False) returns raw_success but does not end
            the episode. After the first raw_success, continue executing exactly
            `success_delay_steps` more policy actions, then mark eval_success.
    """
    success_flag = False
    error_flag = False
    env._update_render()
    if env.render_freq:
        env.viewer.render()
    env.actor_pose = True

    if hasattr(policy, "reset_episode_state"):
        policy.reset_episode_state()

    success_delay_steps = int(getattr(policy, "success_delay_steps", 0))

    images = []
    obs = env.get_obs()

    # Keep the original upper bound: at most 10 policy queries. Each query can
    # return a chunk of multiple actions from XVLA.generate_actions.
    for j in range(10):
        actions = policy.step(obs)
        rollout_action = _convert_model_actions_to_ee_actions(actions)

        for action in tqdm(rollout_action):
            # --------------------------------------------------------------
            # success_delay_steps == 0: original immediate-success behavior
            # --------------------------------------------------------------
            if success_delay_steps <= 0:
                _take_action_delay_aware(env, action, action_type='ee', mark_success=True)
                raw_success = bool(getattr(env, "eval_success", False) or env.check_success())
            else:
                # ----------------------------------------------------------
                # success_delay_steps > 0: do not let take_action terminate
                # immediately. It should only return raw_success.
                # ----------------------------------------------------------
                raw_success = bool(
                    _take_action_delay_aware(env, action, action_type='ee', mark_success=False)
                )

            obs = env.get_obs()
            if args is not None:
                _record_rollout_observation(env, args, obs=obs)

            # Keep obs endpose aligned with the executed ee action, matching the
            # original script behavior.
            try:
                obs['endpose']['left_endpose'] = list(action[:7].reshape(7,))
                obs['endpose']['right_endpose'] = list(action[8:-1].reshape(7,))
            except Exception:
                pass

            images.append(obs["observation"]["head_camera"]["rgb"])

            if getattr(env, "actor_pose", True) is False:
                print('false actor_pose')
                error_flag = True
                break

            if success_delay_steps <= 0:
                if raw_success:
                    success_flag = True
                    break
            else:
                # Already in delayed-success phase: this action counts as one
                # post-success policy action.
                if getattr(policy, "success_delay_active", False):
                    policy.success_delay_remaining -= 1
                    if policy.success_delay_remaining <= 0:
                        _mark_eval_success(env)
                        success_flag = True
                        break
                # First time success is detected. This action itself is NOT
                # counted as one of the delayed actions, matching your pi0.5 eval.
                elif raw_success:
                    policy.success_delay_active = True
                    policy.success_delay_remaining = success_delay_steps
                    print(
                        f"\nraw success detected; continue {success_delay_steps} delayed policy actions before marking success."
                    )

        if error_flag:
            print("\nfail due to false actor_pose!")
            return 0, images

        if success_flag:
            print("\nsuccess!")
            env.suc += 1
            return 1, images

        if getattr(env, "actor_pose", True) is False:
            print('false actor_pose2')
            break

        env._update_render()

    print("\nfail!")
    return 0, images

def eval_episodes(task_name, task_config, policy, test_num=10, seed=0, eval_log_dir=None, instruction_type=None,
                  save_rollout=False, rollout_save_root=None, rollout_save_only_success=False,
                  rollout_resume=False, rollout_episode_offset=0, eval_start_seed=None,
                  success_delay_steps=0, save_4d=False, save_scene_flow=True, save_interval=10):
    """
    rollout several episodes and log the mean episode return
    """
    if not os.path.exists(os.path.join(eval_log_dir, task_name)):
        print('save to', os.path.join(eval_log_dir, task_name))
        os.makedirs(os.path.join(eval_log_dir, task_name))
    
    TASK_ENV, args = load_env(task_name, task_config)
    args["save_rollout"] = parse_bool(save_rollout)
    args["save_4d"] = parse_bool(save_4d)
    args["save_scene_flow"] = parse_bool(save_scene_flow)
    args["save_interval"] = int(save_interval)
    args["rollout_save_root"] = rollout_save_root
    args["rollout_save_only_success"] = parse_bool(rollout_save_only_success)
    args["success_delay_steps"] = int(success_delay_steps)
    if hasattr(policy, "success_delay_steps"):
        policy.success_delay_steps = int(success_delay_steps)

    if (args["save_rollout"] or args["save_4d"]) and (rollout_save_root is None or str(rollout_save_root).strip() == ""):
        raise ValueError("`save_rollout=true` or `save_4d=true` requires non-empty `rollout_save_root`.")

    if args["save_4d"]:
        data_type = _ensure_4d_data_type(args)
        if getattr(TASK_ENV, "data_type", None) is not None:
            TASK_ENV.data_type = data_type
        interval = _resolve_save_interval(args, TASK_ENV)
        print(
            f"[4D] save every action, prune with save_interval={interval}, "
            f"scene_flow={args['save_scene_flow']}"
        )

    if (args["save_rollout"] or args["save_4d"]) and rollout_resume:
        auto_offset, auto_seed = _load_rollout_resume_state(
            rollout_save_root, task_name, task_config, save_4d=args["save_4d"]
        )
        if int(rollout_episode_offset) == 0:
            rollout_episode_offset = auto_offset
        if eval_start_seed is None and auto_seed is not None:
            eval_start_seed = auto_seed
    args["rollout_episode_offset"] = int(rollout_episode_offset)

    if eval_start_seed is not None:
        st_seed = int(eval_start_seed)
    else:
        st_seed = 2000 * (1 + seed)

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args['policy_name'] = 'V4'

    args["eval_mode"] = True
    args["render_freq"] = 0
    args['ckpt_setting'] = '60k'
    point = 0.0
    while succ_seed < test_num:
        render_freq = args["render_freq"]
        print('Running test', now_id)
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except Exception as e:
                # print(" -------------")
                print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        instruction = args["task_name"].replace('_', ' ') # use task name as instruction
        print('instruction:', instruction)
        policy.set_instruction(instruction=instruction)  # set language instruction

        episode_idx = TASK_ENV.test_num + int(args.get("rollout_episode_offset", 0))
        rollout_dir = _init_rollout_recording(TASK_ENV, args, episode_idx=episode_idx)

        try:
            status, images = _rollout(TASK_ENV, policy, args=args)
        except Exception as e:
            print(f"Rollout failed (seed={now_seed}): {e}")
            traceback.print_exc()
            if getattr(TASK_ENV, "save_data", False):
                TASK_ENV.save_data = False
                try:
                    TASK_ENV.remove_data_cache()
                except Exception:
                    pass
            if getattr(TASK_ENV, "_4d_episode_dir", None):
                import shutil
                shutil.rmtree(TASK_ENV._4d_episode_dir, ignore_errors=True)
                TASK_ENV._4d_episode_dir = None
            TASK_ENV.close_env()
            # Episode did not finish; undo seed accounting so we still collect test_num episodes.
            succ_seed = max(0, succ_seed - 1)
            if suc_test_seed_list and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            now_seed += 1
            args["render_freq"] = render_freq
            continue
        save_path = f'{eval_log_dir}/{task_name}/{now_id}_{status}.mp4'
        save_video(save_path, images)
        # log per episode results
        metrics = {f'sim/{task_name}': status}
        save_path = f'{eval_log_dir}/results.json'
        _log_results(metrics, save_path)
        _finalize_rollout_recording(
            TASK_ENV,
            args,
            rollout_dir=rollout_dir,
            episode_idx=episode_idx,
            instruction=instruction,
            episode_info=episode_info.get("info", episode_info) if isinstance(episode_info, dict) else episode_info,
            success=bool(status),
            env_seed=now_seed,
        )
        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(f"Success rate: {round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%")
        now_seed += 1
        try:
            point = round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)
            test_done = True
        except Exception as e:
            test_done = False
            print('redo tests due to:', e)
    metrics = {f'sim/{task_name}': point}
    _log_results(metrics, f'{eval_log_dir}/perc.json')

    if (parse_bool(save_rollout) or parse_bool(save_4d)) and rollout_save_root:
        _save_rollout_cursor(
            rollout_save_root,
            task_name,
            task_config,
            last_seed=now_seed,
            episode_count=int(args.get("rollout_episode_offset", 0)) + int(TASK_ENV.test_num),
        )

    return now_seed, TASK_ENV.suc


def save_video(output_path, frames, fps=30):
    print('saving video to ', output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if frames is None or len(frames) == 0:
        print('warning: no frames collected; skip video saving')
        return
    imageio.mimsave(output_path, frames, fps=fps) 


def _log_results(metrics, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a+') as f:
        line = json.dumps(metrics)
        f.write(line+'\n')


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--eval_log_dir", default='/home/dodo/fyc/HeteroDiffusionPolicy/AbsEEFFlowV4/runnings/RoboTwin/', type=str, help="Where to log the evaluation results.")
    parser.add_argument("--device", default="cuda", type=str, help="Device for XVLA model, e.g. cuda, cuda:0, cpu, auto")
    parser.add_argument("--num_episodes", default=1000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task (envs.<task>)")
    parser.add_argument("--task_config", type=str, required=True, help="Task config name (without .yml)")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the output video")
    parser.add_argument("--instruction_type", type=str, required=False, help="Where to save the output video")
    parser.add_argument("--save_rollout", default=False, type=parse_bool, help="Whether to save RoboTwin-format rollout data.")
    parser.add_argument("--save_4d", default=False, type=parse_bool, help="Save per-frame multi-view RGB, depth, camera params, and robot pose for 4D training.")
    parser.add_argument("--save_scene_flow", default=True, type=parse_bool, help="When save_4d, also save float16 scene-flow npy after pruning.")
    parser.add_argument("--save_interval", default=10, type=int, help="After each episode, keep every N-th action frame (0, N, 2N, ...).")
    parser.add_argument("--rollout_save_root", default=None, type=str, help="Root directory for saving RoboTwin-format rollout data.")
    parser.add_argument("--rollout_save_only_success", default=False, type=parse_bool, help="Only keep rollout data for successful episodes.")
    parser.add_argument("--rollout_resume", default=False, type=parse_bool, help="Resume episode index and seed cursor from rollout_save_root.")
    parser.add_argument("--rollout_episode_offset", default=0, type=int, help="Episode index offset when saving rollout data.")
    parser.add_argument("--eval_start_seed", default=None, type=int, help="Optional start seed for evaluation/resume.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to XVLA pretrained model directory.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to XVLA processor directory. Defaults to model_path.")
    parser.add_argument("--LoRA_path", type=str, default=None, help="Optional LoRA checkpoint path.")
    parser.add_argument("--xvla_repo_root", type=str, default=None, help="Optional root of the XVLA repo containing the `models` package.")
    parser.add_argument("--local_action_method", type=str, default="generate_actions", help="XVLA local inference method name. For your XVLA repo, use generate_actions.")
    parser.add_argument("--inference_steps", type=int, default=10, help="Denoising steps passed to XVLA.generate_actions(...).")
    parser.add_argument("--success_delay_steps", type=int, default=0, help="After first raw success, continue this many policy actions before marking success. 0 keeps original immediate-success behavior.")
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"], help="Torch dtype for XVLA model loading.")
    args = parser.parse_args()
    kwargs = vars(args)
    
    model = LocalXVLAModel(
        model_path=args.model_path,
        processor_path=args.processor_path,
        LoRA_path=args.LoRA_path,
        device=args.device,
        xvla_repo_root=args.xvla_repo_root,
        local_action_method=args.local_action_method,
        torch_dtype=args.torch_dtype,
        inference_steps=args.inference_steps,
        success_delay_steps=args.success_delay_steps,
    )
    if args.task_name == 'all':
        for task in tqdm(ALL_TASKS):
            print(f"Evaluating task {task} for {kwargs['num_episodes']} episodes...")
            rewards = eval_episodes(task_name=task, 
                                    task_config=args.task_config,
                                    policy=model, 
                                    seed=args.seed,
                                    test_num=kwargs['num_episodes'],
                                    eval_log_dir=kwargs['eval_log_dir'],
                                    instruction_type=args.instruction_type,
                                    save_rollout=args.save_rollout,
                                    save_4d=args.save_4d,
                                    save_scene_flow=args.save_scene_flow,
                                    save_interval=args.save_interval,
                                    rollout_save_root=args.rollout_save_root,
                                    rollout_save_only_success=args.rollout_save_only_success,
                                    rollout_resume=args.rollout_resume,
                                    rollout_episode_offset=args.rollout_episode_offset,
                                    eval_start_seed=args.eval_start_seed,
                                    success_delay_steps=args.success_delay_steps)
    else:
        print(f"Evaluating task {args.task_name} for {kwargs['num_episodes']} episodes...")
        rewards = eval_episodes(task_name=args.task_name, 
                                task_config=args.task_config,
                                policy=model, 
                                seed=args.seed,
                                test_num=kwargs['num_episodes'],
                                eval_log_dir=kwargs['eval_log_dir'],
                                instruction_type=args.instruction_type,
                                save_rollout=args.save_rollout,
                                save_4d=args.save_4d,
                                save_scene_flow=args.save_scene_flow,
                                save_interval=args.save_interval,
                                rollout_save_root=args.rollout_save_root,
                                rollout_save_only_success=args.rollout_save_only_success,
                                rollout_resume=args.rollout_resume,
                                rollout_episode_offset=args.rollout_episode_offset,
                                eval_start_seed=args.eval_start_seed,
                                success_delay_steps=args.success_delay_steps)
if __name__ == "__main__":
    main()
