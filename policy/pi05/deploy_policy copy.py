import numpy as np
from collections import deque
import os
import sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import PI0


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


class Pi05PolicyWrapper:
    def __init__(self, pi0_model, success_delay_steps=0):
        self.pi0 = pi0_model
        self.pending_actions = deque()

        # 第一次检测到 success 后，还要继续执行多少个 policy action
        self.success_delay_steps = int(success_delay_steps)

        # success 延迟状态
        self.success_delay_active = False
        self.success_delay_remaining = 0

    def should_request_observation(self):
        return not self.pending_actions

    def __getattr__(self, name):
        return getattr(self.pi0, name)


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (
        usr_args["train_config_name"],
        usr_args["model_name"],
        usr_args["checkpoint_id"],
        usr_args["pi0_step"],
    )

    success_delay_steps = int(usr_args.get("success_delay_steps", 0))

    pi0 = PI0(train_config_name, model_name, checkpoint_id, pi0_step)
    return Pi05PolicyWrapper(pi0, success_delay_steps=success_delay_steps)


def _mark_eval_success(TASK_ENV):
    """
    和原 take_action 里 success 后做的事情保持一致。
    """
    TASK_ENV.eval_success = True
    TASK_ENV.get_obs()  # update obs

    if TASK_ENV.eval_video_path is not None:
        TASK_ENV.eval_video_ffmpeg.stdin.write(
            TASK_ENV.now_obs["observation"]["head_camera"]["rgb"].tobytes()
        )


def eval(TASK_ENV, model, observation):
    """
    Execute a single policy action step from pi0.5.
    observation: None only allowed if pending_actions is not empty.
    """
    pi0 = model.pi0

    if observation is not None:
        if pi0.observation_window is None:
            instruction = TASK_ENV.get_instruction()
            pi0.set_language(instruction)

        input_rgb_arr, input_state = encode_obs(observation)
        pi0.update_observation_window(input_rgb_arr, input_state)

    if not model.pending_actions:
        actions = pi0.get_action()[:pi0.pi0_step]
        for action in actions:
            model.pending_actions.append(action)

    if not model.pending_actions:
        return

    action = model.pending_actions.popleft()
    TASK_ENV.last_policy_action = action

    # ==========================================================
    # success_delay_steps = 0 时：完全保持原逻辑
    # take_action 内部 check_success 后会直接设置 eval_success 并 return
    # ==========================================================
    if model.success_delay_steps <= 0:
        TASK_ENV.take_action(action)
        return

    # ==========================================================
    # success_delay_steps > 0 时：
    # take_action 只返回 raw_success，不直接终止 episode
    # ==========================================================
    raw_success = TASK_ENV.take_action(action, mark_success=False)

    # 如果已经进入 success 延迟阶段，说明当前 action 是 success 后的连续动作
    if model.success_delay_active:
        model.success_delay_remaining -= 1

        if model.success_delay_remaining <= 0:
            _mark_eval_success(TASK_ENV)

        return

    # 第一次检测到 raw success：启动延迟计数
    if raw_success:
        model.success_delay_active = True
        model.success_delay_remaining = model.success_delay_steps


def reset_model(model):
    model.pending_actions.clear()
    model.pi0.reset_obsrvationwindows()

    # reset success delay state
    model.success_delay_active = False
    model.success_delay_remaining = 0