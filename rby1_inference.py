import time
from collections import deque
from pathlib import Path

import av


import cv2
import numpy as np
import torch
import mujoco
import mujoco.viewer

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors





# =========================
# User config
# =========================
XML_PATH = "scene_rby1a_1.2.xml"

# 로컬 체크포인트 경로 또는 HF repo id
MODEL_PATH = "lerobot_tommoro/outputs/train/act_var_original_delta_0320/checkpoints/030000/pretrained_model"

# 실영상 head mp4
HEAD_MP4 = "/home/soyounglee/datasets/0206_Amore_osan_feeder_pink_toner/videos/observation.images.cam_head/chunk-000/file-000.mp4"
LEFT_WRIST_MP4 = "/home/soyounglee/datasets/0206_Amore_osan_feeder_pink_toner/videos/observation.images.cam_wrist_left/chunk-000/file-000.mp4"
RIGHT_WRIST_MP4 = "/home/soyounglee/datasets/0206_Amore_osan_feeder_pink_toner/videos/observation.images.cam_wrist_right/chunk-000/file-000.mp4"

USE_ZERO_WRIST = False
USE_MUJOCO_WRIST_CAMERA = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_STEPS = 1000

# 시뮬레이션 속도 조절
SLEEP_SEC = 0.02

# 초기 로봇 자세 (state 값 기반)
INITIAL_QPOS_REAL = np.array([
    0.09777229279279709,    # left_arm_0
    0.25477370619773865,    # left_arm_1
    -0.19718867540359497,   # left_arm_2
    -2.5167081356048584,    # left_arm_3
    -0.0542377270758152,    # left_arm_4
    0.6507606506347656,     # left_arm_5
    -1.3683338165283203,    # left_arm_6

    0.09078584611415863,    # right_arm_0
    -0.10642561316490173,   # right_arm_1
    -0.20940355956554413,   # right_arm_2
    -2.4643402099609375,    # right_arm_3
    -0.5427607297897339,    # right_arm_4
    0.7747868299484253,     # right_arm_5
    1.8727527856826782,     # right_arm_6

    0.002210884355008602,   # left_trigger
    0.00197508349083364,    # right_trigger
], dtype=np.float32)



# 한 step에서 너무 큰 delta가 나오면 clamp
MAX_ACTION_ABS = 3.2


# =========================
# Real-data order from your robot
# =========================
REAL_DATA_ORDER = [
    "left_arm_0",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "left_arm_4",
    "left_arm_5",
    "left_arm_6",
    "right_arm_0",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "right_arm_4",
    "right_arm_5",
    "right_arm_6",
    # "left_trigger",
    # "right_trigger",
]

MUJOCO_ACTUATOR_ORDER = [
    "left_wheel_act",
    "right_wheel_act",
    "link1_act",
    "link2_act",
    "link3_act",
    "link4_act",
    "link5_act",
    "link6_act",
    "right_arm_1_act",
    "right_arm_2_act",
    "right_arm_3_act",
    "right_arm_4_act",
    "right_arm_5_act",
    "right_arm_6_act",
    "right_arm_7_act",
    "left_arm_1_act",
    "left_arm_2_act",
    "left_arm_3_act",
    "left_arm_4_act",
    "left_arm_5_act",
    "left_arm_6_act",
    "left_arm_7_act",
    "head_0_act",
    "head_1_act",
    "right_finger_act",
    "left_finger_act",
]


# =========================
# Helpers
# =========================


def make_batched_obs(raw_obs, device: str):
    obs = {}

    for k, v in raw_obs.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
        else:
            t = torch.as_tensor(v)

        t = t.to(torch.float32)

        if "images" in k:
            if t.ndim != 3:
                raise ValueError(f"{k} expected CHW, got shape {tuple(t.shape)}")
            t = t.unsqueeze(0)   # CHW -> BCHW
        else:
            if t.ndim != 1:
                raise ValueError(f"{k} expected [D], got shape {tuple(t.shape)}")
            t = t.unsqueeze(0)   # [D] -> [1,D]

        obs[k] = t.to(device)

    return obs


def reorder_real_to_mujoco(action_real_order: np.ndarray) -> np.ndarray:
    action_real_order = np.asarray(action_real_order, dtype=np.float32).reshape(-1)
    if len(action_real_order) != len(REAL_DATA_ORDER):
        raise ValueError(f"Expected {len(REAL_DATA_ORDER)}-dim action, got {action_real_order.shape}")

    value_by_name = {
        name: action_real_order[i]
        for i, name in enumerate(REAL_DATA_ORDER)
    }

    ctrl = np.zeros(26, dtype=np.float32)

    # right arm: actuator 8~14
    ctrl[8]  = value_by_name["right_arm_0"]
    ctrl[9]  = value_by_name["right_arm_1"]
    ctrl[10] = value_by_name["right_arm_2"]
    ctrl[11] = value_by_name["right_arm_3"]
    ctrl[12] = value_by_name["right_arm_4"]
    ctrl[13] = value_by_name["right_arm_5"]
    ctrl[14] = value_by_name["right_arm_6"]

    # left arm: actuator 15~21
    ctrl[15] = value_by_name["left_arm_0"]
    ctrl[16] = value_by_name["left_arm_1"]
    ctrl[17] = value_by_name["left_arm_2"]
    ctrl[18] = value_by_name["left_arm_3"]
    ctrl[19] = value_by_name["left_arm_4"]
    ctrl[20] = value_by_name["left_arm_5"]
    ctrl[21] = value_by_name["left_arm_6"]

    # gripper
    # ctrl[24] = value_by_name["right_trigger"]   # right_finger_act
    # ctrl[25] = value_by_name["left_trigger"]    # left_finger_act

    return ctrl

def chw_from_bgr(frame_bgr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
    frame = frame.astype(np.float32)
    return frame


def center_square_crop_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    return frame_rgb[top:top+s, left:left+s]


class CachedVideoReader:
    def __init__(self, path: str):
        self.path = path
        self.frames = []

        container = av.open(path)

        max_frames = 1000  # 원하는 만큼 조절

        for i, frame in enumerate(container.decode(video=0)):
            img = frame.to_ndarray(format="bgr24")
            self.frames.append(img)

            if i + 1 >= max_frames:
                break

        container.close()

        if not self.frames:
            raise RuntimeError(f"No frames decoded from video: {path}")

        self.idx = 0
        print(f"[CachedVideoReader] Loaded {len(self.frames)} frames from {path}")

    def read(self) -> np.ndarray:
        frame = self.frames[self.idx]

        self.idx = (self.idx + 1) % len(self.frames)
        return frame

    def close(self):
        pass

class y1a2MujocoEnv:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.joint_name_to_qposadr = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is not None:
                self.joint_name_to_qposadr[name] = self.model.jnt_qposadr[i]

        self.renderer_left = mujoco.Renderer(self.model, height=240, width=424)
        self.renderer_right = mujoco.Renderer(self.model, height=240, width=424)

    def set_state_real(self, qpos_real_order: np.ndarray):
        qpos_real_order = np.asarray(qpos_real_order, dtype=np.float32).reshape(-1)
        if qpos_real_order.shape[0] != 16:
            raise ValueError(f"Expected 16-dim initial state, got {qpos_real_order.shape}")

        for i, joint_name in enumerate(REAL_DATA_ORDER):
            adr = self.joint_name_to_qposadr[joint_name]
            self.data.qpos[adr] = qpos_real_order[i]

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, initial_qpos_real: np.ndarray | None = None):
        mujoco.mj_resetData(self.model, self.data)
        if initial_qpos_real is not None:
            self.set_state_real(initial_qpos_real)
        else:
            mujoco.mj_forward(self.model, self.data)
        return self.get_state()


    def get_state(self) -> np.ndarray:
        vals = []
        for joint_name in REAL_DATA_ORDER:
            adr = self.joint_name_to_qposadr[joint_name]
            vals.append(self.data.qpos[adr])
        return np.asarray(vals, dtype=np.float32)

    def render_wrist_left_chw(self) -> np.ndarray:
        self.renderer_left.update_scene(self.data, camera="cam_wrist_left")
        img = self.renderer_left.render()  # HWC RGB uint8
        img = np.transpose(img, (2, 0, 1))  # CHW
        return img

    def render_wrist_right_chw(self) -> np.ndarray:
        self.renderer_right.update_scene(self.data, camera="cam_wrist_right")
        img = self.renderer_right.render()  # HWC RGB uint8
        img = np.transpose(img, (2, 0, 1))  # CHW
        return img


    def step(self, action_real_order: np.ndarray):
        action_real_order = np.asarray(action_real_order, dtype=np.float32).reshape(-1)
        if action_real_order.shape[0] != 14:
            raise ValueError(f"Expected action dim 14, got {action_real_order.shape}")

        # action_real_order = np.clip(action_real_order, -MAX_ACTION_ABS, MAX_ACTION_ABS)
        target_mujoco = reorder_real_to_mujoco(action_real_order)
        self.data.ctrl[:] = target_mujoco

        # 30Hz 비슷하게 hold
        n_substeps = 16
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = {"observation.state": self.get_state()}
        return obs, 0.0, False, {}



def main():
    print(f"Using device: {DEVICE}")

    # Policy
    # model = ACTPolicy.from_pretrained(MODEL_PATH).to(DEVICE)
    model = DiffusionPolicy.from_pretrained(MODEL_PATH).to(DEVICE)

    model.config.n_action_steps = 1
    model.eval()

    preprocess, postprocess = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=MODEL_PATH,
        dataset_stats=None,
    )

    # Env
    env = y1a2MujocoEnv(XML_PATH)
    env.model.opt.gravity[:] = 0 # 중력제거
    env.reset(initial_qpos_real=INITIAL_QPOS_REAL)

    # Videos
    head_reader = CachedVideoReader(HEAD_MP4)
    left_reader = CachedVideoReader(LEFT_WRIST_MP4)
    right_reader = CachedVideoReader(RIGHT_WRIST_MP4)

    # print("MODEL_PATH:", MODEL_PATH)
    # print("Loaded policy config:", model.config)
    # print("Pre/Post processors loaded from pretrained_model directory")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(N_STEPS):
            head_bgr = head_reader.read()
            left_bgr = left_reader.read()
            right_bgr = right_reader.read()

            # 카메라 입력 전처리
            head_chw = chw_from_bgr(head_bgr, out_h=376, out_w=672)
            left_chw = chw_from_bgr(left_bgr, out_h=240, out_w=424)
            right_chw = chw_from_bgr(right_bgr, out_h=240, out_w=424)

            raw_obs = {
                "observation.state": env.get_state(),
                "observation.images.cam_head": head_chw,
                "observation.images.cam_wrist_left": left_chw,
                "observation.images.cam_wrist_right": right_chw,
            }

            if step == 0:
                print("raw obs keys:", raw_obs.keys())
                print("state shape:", raw_obs["observation.state"].shape)
                print("head shape:", raw_obs["observation.images.cam_head"].shape)
                print("left shape:", raw_obs["observation.images.cam_wrist_left"].shape)
                print("right shape:", raw_obs["observation.images.cam_wrist_right"].shape)
                print("initial state:", raw_obs["observation.state"])

            obs_frame = make_batched_obs(raw_obs, DEVICE)
            

            # Policy 추론
            with torch.inference_mode():
                obs_pre = preprocess(obs_frame)
                action_out = model.select_action(obs_pre)
                action_out = postprocess(action_out)

                print("관측값[:8] =", env.get_state()[:8])
                print("전처리 후[:8] =", obs_pre["observation.state"][0, :8].detach().cpu().numpy())
                print("예측값[:8] =", action_out[0,:8].detach().cpu().numpy())

            action_vec = action_out.squeeze(0).detach().cpu().numpy()


            # step with action
            _, _, _, _ = env.step(action_vec)


            if step == 0:
                print("head raw:", head_chw.shape, head_chw.dtype, head_chw.min(), head_chw.max())
                print("left raw:", left_chw.shape, left_chw.dtype, left_chw.min(), left_chw.max())
                print("right raw:", right_chw.shape, right_chw.dtype, right_chw.min(), right_chw.max())

                obs_frame = make_batched_obs(raw_obs, DEVICE)
                obs_pre = preprocess(obs_frame)

                for k, v in obs_pre.items():
                    if isinstance(v, torch.Tensor) and "images" in k:
                        print(
                            k,
                            tuple(v.shape),
                            v.dtype,
                            v.min().item(),
                            v.max().item(),
                            v.mean().item(),
                            v.std().item(),
                        )

            if step % 20 == 0:
                print(
                    f"step={step:04d} "
                    f"action[min,max]=({action_vec.min():+.4f}, {action_vec.max():+.4f}) "
                    f"states={env.get_state()}"
                )

            viewer.sync()
            time.sleep(SLEEP_SEC)

        # for step in range(N_STEPS):
        #     curr_state = env.get_state().astype(np.float32)

        #     # 1) 현재 상태 유지 테스트
        #     action_vec = curr_state.copy()

        #     # 또는 2) 초기 자세 유지 테스트
        #     # action_vec = INITIAL_QPOS_REAL.copy()

        #     _, _, _, _ = env.step(action_vec)

        #     if step % 20 == 0:
        #         print("state =", env.get_state())       

        #     viewer.sync()
        #     time.sleep(SLEEP_SEC)


    head_reader.close()
    if left_reader:
        left_reader.close()
    if right_reader:
        right_reader.close()


if __name__ == "__main__":
    main()