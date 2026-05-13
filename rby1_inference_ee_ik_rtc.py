"""
rby1_inference_ee_ik.py

Sanity-test a LeRobot DiffusionPolicy trained with EE-pose actions in MuJoCo.

Policy interface expected by this file: 왼 + 오 순서!!!!!! 
  observation.state: concat([observation.left_arm, observation.right_arm]) -> 16D
    each arm: [x, y, z, qx, qy, qz, qw, trigger]
  action: concat([action.left_arm, action.right_arm]) -> 16D
    each arm: [x, y, z, qx, qy, qz, qw, trigger]

Control path:
  policy EE action in head/controller frame
    -> (oM_head * action_pose) * fixed tool offset
    -> Pinocchio IK
    -> MuJoCo position actuator targets

This intentionally keeps many debug prints and safety clamps for first bring-up.
"""

import multiprocessing as mp
import queue
from collections import deque

import time
from pathlib import Path

import av
import cv2
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin
import torch
from scipy.spatial.transform import Rotation as R

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


# =========================
# User config
# =========================
XML_PATH = "./scene_rby1a_1.2.xml"
URDF_PATH = "./urdf/model_v1.2_camera.urdf"

# Local checkpoint path
MODEL_PATH = "/home/soyounglee/project/simulation/rby1_sim/lerobot_tommoro/outputs/vietnam/dp2"

# Real videos to feed the policy. These must match training feature keys below.
HEAD_MP4 = "/home/soyounglee/datasets/data_foundry_test/videos/observation.images.head_cam/chunk-000/file-000.mp4"
LEFT_WRIST_MP4 = "/home/soyounglee/datasets/data_foundry_test/videos/observation.images.left_wrist_cam/chunk-000/file-000.mp4"
RIGHT_WRIST_MP4 = "/home/soyounglee/datasets/data_foundry_test/videos/observation.images.right_wrist_cam/chunk-000/file-000.mp4"

# Dataset metadata
IMAGE_H = 240
IMAGE_W = 320

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONTROL_HZ = 30  # dataset fps is 30
DT = 1.0 / CONTROL_HZ
N_STEPS = 1000

# Bring-up safety. Start conservative and loosen after things look sane.
MAX_EE_POS_DELTA = 0.04       # m per policy step, clamp target movement
MAX_JOINT_DELTA = 0.05        # rad per policy step, clamp q_des movement
MAX_IK_ITERS = 1
TORQUE_LIMIT = 200.0
USE_POSITION_CTRL = True
KP = 500.0
KD = 80.0

# Useful for first test. Gravity off avoids falling/settling artifacts.
ZERO_GRAVITY = True

# If video paths are unavailable, set True to feed black images just to test wiring.
USE_ZERO_IMAGES = False

DEBUG_PRINT_EVERY = 60


# =========================
# Robot / arm config
# =========================
ARM_CONFIG = {
    "right": {
        "ee_body_name": "EE_BODY_R",
        "pin_frame_name": "ee_right",
        "joint_names": [f"right_arm_{i}" for i in range(7)],
        "init_q": np.array([
            0.0,
            0.0,
            0.0,
            -2.26892803,  # -130 deg
            0.0,
            0.0,
            1.57079633,   # +90 deg
        ], dtype=float),
    },
    "left": {
        "ee_body_name": "EE_BODY_L",
        "pin_frame_name": "ee_left",
        "joint_names": [f"left_arm_{i}" for i in range(7)],
        "init_q": np.array([
            0.0,
            0.0,
            0.0,
            -2.26892803,  # -130 deg
            0.0,
            0.0,
            -1.57079633,  # -90 deg
        ], dtype=float),
    },
}

TORSO_INIT = {
    "torso_0": 0.0,
    "torso_1": np.deg2rad(-22.8),
    "torso_2": 0.0,
    "torso_3": np.deg2rad(60.3),
    "torso_4": 0.0,
    "torso_5": 0.0,
}

# Tool offset
R_OFF_XYZ = np.array([-0.0073, 0.0039, -0.0720], dtype=float)
R_OFF_RPY = np.array([98.99, 0.30, 163.89], dtype=float)

L_OFF_XYZ = np.array([-0.0073, -0.0051, -0.0720], dtype=float)
L_OFF_RPY = np.array([-97.58, 1.71, 15.36], dtype=float)

HEAD_FRAME_NAME = "zed2i_left_camera_optical_frame"




# =========================
# Async policy / action queue
# =========================
USE_ASYNC_POLICY = True

POLICY_NUM_INFERENCE_STEPS = 10
WORKER_DEVICE = DEVICE

RAW_CHUNK_SIZE = 16
DENSE_CHUNK_SIZE = 45
LOW_WATERMARK = 25
BLEND_STEPS = 7
MAX_ACTION_QUEUE_LEN = 135

# =========================
# Utilities / Video 
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
            t = t.unsqueeze(0)  # CHW -> BCHW
        else:
            if t.ndim != 1:
                raise ValueError(f"{k} expected [D], got shape {tuple(t.shape)}")
            t = t.unsqueeze(0)  # [D] -> [1,D]

        obs[k] = t.to(device)
    return obs



def chw_from_bgr(frame_bgr: np.ndarray, out_h: int = IMAGE_H, out_w: int = IMAGE_W) -> np.ndarray:
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
    return frame.astype(np.float32)


class CachedVideoReader:
    def __init__(self, path: str, max_frames: int = 2000):
        self.path = path
        self.frames = []

        container = av.open(path)
        for i, frame in enumerate(container.decode(video=0)):
            self.frames.append(frame.to_ndarray(format="bgr24"))
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


class ZeroImageReader:
    def __init__(self, h: int = IMAGE_H, w: int = IMAGE_W):
        self.frame = np.zeros((h, w, 3), dtype=np.uint8)

    def read(self) -> np.ndarray:
        return self.frame

    def close(self):
        pass


# =========================
# # Utilities / SE3, kinematics
# =========================
def ee_vec_to_se3(vec):
    """Convert [x,y,z,qx,qy,qz,qw,trigger] or [x,y,z,qx,qy,qz,qw] to pin.SE3."""
    vec = np.asarray(vec, dtype=float).reshape(-1)
    if vec.shape[0] < 7:
        raise ValueError(f"EE pose expected at least 7 dims, got {vec.shape}")

    pos = vec[:3]
    quat_xyzw = vec[3:7]
    qn = np.linalg.norm(quat_xyzw)
    if qn < 1e-8 or not np.isfinite(qn):
        print(f"[WARN] Bad quaternion {quat_xyzw}; using identity")
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    else:
        quat_xyzw = quat_xyzw / qn

    rot = R.from_quat(quat_xyzw).as_matrix()
    return pin.SE3(rot, pos)


def se3_to_ee_vec(T: pin.SE3, trigger: float = 0.0) -> np.ndarray:
    quat_xyzw = R.from_matrix(T.rotation).as_quat().astype(np.float32)
    return np.concatenate([
        T.translation.astype(np.float32),
        quat_xyzw,
        np.array([trigger], dtype=np.float32),
    ])


def get_offset_se3(xyz, rpy_deg):
    """Same offset convention as replay_with_real.py: SE3(R,0) * SE3(I,xyz)."""
    rot = R.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    return pin.SE3(rot, np.zeros(3)) * pin.SE3(np.eye(3), xyz)


def get_all_arm_joints():
    return ARM_CONFIG["right"]["joint_names"] + ARM_CONFIG["left"]["joint_names"]


def get_torso_joints():
    return [f"torso_{i}" for i in range(6)]


def make_pin_joint_index_map(pin_model, joint_names):
    result = {}
    for name in joint_names:
        if pin_model.existJointName(name):
            jid = pin_model.getJointId(name)
            result[name] = pin_model.joints[jid].idx_q
        else:
            print(f"[WARN] Pinocchio joint not found: {name}")
    return result


def make_mujoco_joint_index_map(mj_model, joint_names):
    result = {}
    for name in joint_names:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            result[name] = mj_model.jnt_qposadr[jid]
        else:
            print(f"[WARN] MuJoCo joint not found: {name}")
    return result


def make_actuator_index_map(model, joint_names):
    actuator_map = {}
    for joint_name in joint_names:
        actuator_name = None

        if joint_name.startswith("right_arm_"):
            idx = int(joint_name.split("_")[-1]) + 1
            actuator_name = f"right_arm_{idx}_act"
        elif joint_name.startswith("left_arm_"):
            idx = int(joint_name.split("_")[-1]) + 1
            actuator_name = f"left_arm_{idx}_act"
        elif joint_name.startswith("torso_"):
            idx = int(joint_name.split("_")[-1]) + 1
            actuator_name = f"link{idx}_act"

        if actuator_name is None:
            continue

        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id == -1:
            print(f"[WARN] Actuator not found: joint={joint_name}, expected={actuator_name}")
            continue
        actuator_map[joint_name] = actuator_id

    return actuator_map


def mujoco_q_to_pin_q(mj_data, pin_model, joint_names, mj_qpos_map, pin_qidx_map):
    q = pin.neutral(pin_model)
    for name in joint_names:
        if name not in mj_qpos_map or name not in pin_qidx_map:
            continue
        q[pin_qidx_map[name]] = mj_data.qpos[mj_qpos_map[name]]
    return q


def set_initial_pose(mj_data, q_pin, joint_names, mj_qpos_map, pin_qidx_map):
    for side, cfg in ARM_CONFIG.items():
        for name, val in zip(cfg["joint_names"], cfg["init_q"]):
            if name in pin_qidx_map:
                q_pin[pin_qidx_map[name]] = val
            if name in mj_qpos_map:
                mj_data.qpos[mj_qpos_map[name]] = val

    for name, val in TORSO_INIT.items():
        if name in pin_qidx_map:
            q_pin[pin_qidx_map[name]] = val
        if name in mj_qpos_map:
            mj_data.qpos[mj_qpos_map[name]] = val


def apply_q_to_mujoco_ctrl(mj_model, mj_data, q_des_pin, pin_qidx_map, mj_qpos_map, actuator_map, joint_names):
    for name in joint_names:
        if name not in pin_qidx_map or name not in actuator_map:
            continue

        aid = actuator_map[name]
        q_des = q_des_pin[pin_qidx_map[name]]

        if USE_POSITION_CTRL:
            mj_data.ctrl[aid] = q_des
        else:
            if name not in mj_qpos_map:
                continue
            q_now = mj_data.qpos[mj_qpos_map[name]]
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof_adr = mj_model.jnt_dofadr[jid]
            dq_now = mj_data.qvel[dof_adr]
            tau = KP * (q_des - q_now) - KD * dq_now
            mj_data.ctrl[aid] = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)


def get_locked_v_indices(pin_model):
    locked = []
    locked_keywords = ["torso", "head", "wheel"]
    for j in range(1, pin_model.njoints):
        name = pin_model.names[j]
        if any(k in name for k in locked_keywords):
            joint = pin_model.joints[j]
            for k in range(joint.nv):
                locked.append(joint.idx_v + k)
    return locked


def apply_fixed_torso_to_q(q, pin_qidx_map):
    q_out = q.copy()
    for name, val in TORSO_INIT.items():
        if name in pin_qidx_map:
            q_out[pin_qidx_map[name]] = val
    return q_out


# =========================
# IK solver by 완성맨~
# =========================
def solve_ik_single(
    pin_model,
    pin_data,
    q_current,
    q_nominal,
    frame_name,
    target_se3,
    locked_v_indices,
    other_arm_keyword=None,
    max_iters=MAX_IK_ITERS,
):
    frame_id = pin_model.getFrameId(frame_name)
    if frame_id >= len(pin_model.frames):
        print(f"[ERROR] Pinocchio frame not found: {frame_name}")
        return q_current, False

    q = q_current.copy()
    nv = pin_model.nv
    current_locked = list(locked_v_indices)

    if other_arm_keyword is not None:
        for j in range(1, pin_model.njoints):
            name = pin_model.names[j]
            if other_arm_keyword in name:
                joint = pin_model.joints[j]
                for k in range(joint.nv):
                    current_locked.append(joint.idx_v + k)

    ok = False
    for _ in range(max_iters):
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)

        current = pin_data.oMf[frame_id]
        err = pin.log6(current.actInv(target_se3)).vector

        if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
            ok = True
            break

        J = pin.computeFrameJacobian(
            pin_model,
            pin_data,
            q,
            frame_id,
            pin.ReferenceFrame.LOCAL,
        )

        for idx in current_locked:
            if 0 <= idx < J.shape[1]:
                J[:, idx] = 0.0

        damp = 1e-3 + np.linalg.norm(err[:3]) * 0.5
        damp = min(damp, 0.1)

        JJt = J @ J.T
        JJt += damp * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJt)

        v_task = J_pinv @ err

        # Keep nominal posture in null space.
        dq_nominal = pin.difference(pin_model, q, q_nominal)
        null = np.eye(nv) - J_pinv @ J
        v_null = null @ (0.05 * dq_nominal)
        v = v_task + v_null

        for idx in current_locked:
            if 0 <= idx < len(v):
                v[idx] = 0.0

        max_v = np.max(np.abs(v))
        if max_v > 0.1:
            v *= 0.1 / max_v

        q = pin.integrate(pin_model, q, v)

        # Clamp single-DoF joint limits.
        for j in range(1, pin_model.njoints):
            joint = pin_model.joints[j]
            if joint.nq == 1:
                idx_q = joint.idx_q
                q[idx_q] = np.clip(
                    q[idx_q],
                    pin_model.lowerPositionLimit[idx_q],
                    pin_model.upperPositionLimit[idx_q],
                )

    if np.any(np.isnan(q)):
        return q_current, False

    # Even if the strict tolerance did not trigger, return q; caller can inspect errors.
    return q, ok


def clamp_target_position(prev_target, new_target, max_pos_delta=MAX_EE_POS_DELTA):
    if prev_target is None:
        return new_target

    out = pin.SE3(new_target.rotation, new_target.translation.copy())
    d = out.translation - prev_target.translation
    n = np.linalg.norm(d)
    if n > max_pos_delta:
        out.translation = prev_target.translation + d / n * max_pos_delta
    return out


def limit_joint_step(q_current, q_des, pin_qidx_map, joint_names, max_delta=MAX_JOINT_DELTA):
    q_out = q_des.copy()
    for name in joint_names:
        if name not in pin_qidx_map:
            continue
        idx = pin_qidx_map[name]
        dq = np.clip(q_des[idx] - q_current[idx], -max_delta, max_delta)
        q_out[idx] = q_current[idx] + dq
    return q_out

# From joint to EE target 
def make_current_ee_obs_state(
    pin_model,
    pin_data,
    q_current,
    head_id,
    left_frame_name,
    right_frame_name,
    T_off_l_const,
    T_off_r_const,
    last_left_trigger=0.0,
    last_right_trigger=0.0,
):
    """
    Convert current robot FK back to dataset-like EE observation state.

    replay uses: world_target = (oM_head * dataset_pose) * T_offset
    Therefore inverse observation is approximately:
      dataset_pose = oM_head^-1 * oM_ee * T_offset^-1
    """
    pin.forwardKinematics(pin_model, pin_data, q_current)
    pin.updateFramePlacements(pin_model, pin_data)

    left_id = pin_model.getFrameId(left_frame_name)
    right_id = pin_model.getFrameId(right_frame_name)
    if left_id >= len(pin_model.frames) or right_id >= len(pin_model.frames):
        raise RuntimeError(f"EE frame not found: {left_frame_name} / {right_frame_name}")

    oM_head = pin_data.oMf[head_id]
    oM_left = pin_data.oMf[left_id]
    oM_right = pin_data.oMf[right_id]

    head_inv = oM_head.inverse()
    left_pose_like = head_inv * oM_left * T_off_l_const.inverse()
    right_pose_like = head_inv * oM_right * T_off_r_const.inverse()

    left_vec = se3_to_ee_vec(left_pose_like, trigger=last_left_trigger)
    right_vec = se3_to_ee_vec(right_pose_like, trigger=last_right_trigger)
    return np.concatenate([left_vec, right_vec]).astype(np.float32)


def extract_policy_action(action_out: torch.Tensor) -> np.ndarray:
    action = action_out.detach().cpu()
    # Expected select_action output is [B, 16]. Be permissive for [B, T, 16].
    if action.ndim == 3:
        action = action[:, 0, :]
    if action.ndim != 2:
        raise RuntimeError(f"Unexpected action tensor shape: {tuple(action.shape)}")
    action_vec = action[0].numpy().astype(np.float32).reshape(-1)
    if action_vec.shape[0] != 16:
        raise RuntimeError(f"Expected 16D EE action, got {action_vec.shape}")
    return action_vec


def print_target_debug(step, action_vec, left_target, right_target, ok_l, ok_r, q_current, q_des, joint_names, pin_qidx_map):
    q_delta = []
    for name in joint_names:
        if name in pin_qidx_map:
            idx = pin_qidx_map[name]
            q_delta.append(abs(q_des[idx] - q_current[idx]))
    q_delta_max = max(q_delta) if q_delta else 0.0

    lquat_norm = np.linalg.norm(action_vec[3:7])
    rquat_norm = np.linalg.norm(action_vec[11:15])

    print(f"\n[step {step:04d}] policy / IK debug")
    print(f"  left action pos   : {action_vec[:3]}")
    print(f"  right action pos  : {action_vec[8:11]}")
    print(f"  quat norm L/R     : {lquat_norm:.4f} / {rquat_norm:.4f}")
    print(f"  left target world : {left_target.translation}")
    print(f"  right target world: {right_target.translation}")
    print(f"  IK ok L/R         : {ok_l} / {ok_r}")
    print(f"  max joint delta   : {q_delta_max:.5f}")


def normalize_action_quats(actions: np.ndarray) -> np.ndarray:
    actions = actions.copy()

    for s, e in [(3, 7), (11, 15)]:
        q = actions[..., s:e]
        n = np.linalg.norm(q, axis=-1, keepdims=True)
        n = np.maximum(n, 1e-8)
        actions[..., s:e] = q / n

    return actions


def interpolate_action_chunk(actions: np.ndarray, dense_steps: int) -> np.ndarray:
    """
    actions: [K, 16]
    returns: [dense_steps, 16]
    """
    actions = np.asarray(actions, dtype=np.float32)
    k, d = actions.shape

    if k == dense_steps:
        return normalize_action_quats(actions)

    src_t = np.linspace(0.0, 1.0, k)
    dst_t = np.linspace(0.0, 1.0, dense_steps)

    dense = np.zeros((dense_steps, d), dtype=np.float32)
    for j in range(d):
        dense[:, j] = np.interp(dst_t, src_t, actions[:, j])

    return normalize_action_quats(dense)


def blend_new_chunk_with_queue_tail(
    action_queue: deque,
    new_actions: np.ndarray,
    blend_steps: int,
) -> np.ndarray:
    """
    Smooth the beginning of new_actions so transition from current queue tail is less abrupt.
    Does not modify already queued actions.
    """
    if len(action_queue) == 0 or blend_steps <= 0:
        return normalize_action_quats(new_actions)

    new_actions = new_actions.copy()
    anchor = np.asarray(action_queue[-1], dtype=np.float32)

    n = min(blend_steps, len(new_actions))
    for i in range(n):
        alpha = (i + 1) / (n + 1)
        new_actions[i] = (1.0 - alpha) * anchor + alpha * new_actions[i]

    return normalize_action_quats(new_actions)


def collect_policy_chunk(
    policy,
    preprocess,
    postprocess,
    obs_frame,
    raw_chunk_size: int,
    dense_chunk_size: int,
    device: str,
):
    """
    Calls select_action raw_chunk_size times.
    First call generates a diffusion chunk; following calls pop from policy's internal action queue.
    """
    raw_actions = []

    policy.reset()

    with torch.inference_mode():
        obs_pre = preprocess(obs_frame)

        for _ in range(raw_chunk_size):
            action_out = policy.select_action(obs_pre)
            action_out = postprocess(action_out)

            action_vec = extract_policy_action(action_out)
            raw_actions.append(action_vec)

    raw_actions = np.stack(raw_actions, axis=0)
    dense_actions = interpolate_action_chunk(raw_actions, dense_chunk_size)
    return dense_actions


def policy_process_loop(
    model_path,
    request_queue,
    result_queue,
    device,
    raw_chunk_size,
    dense_chunk_size,
    num_inference_steps,
):
    """
    Runs in a separate process.
    Owns the DiffusionPolicy and CUDA context.
    Main process must send CPU numpy raw_obs dicts, not CUDA tensors.
    """
    import time
    import torch

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print("[POLICY PROCESS] starting")
    print("[POLICY PROCESS] loading:", model_path)

    policy = DiffusionPolicy.from_pretrained(model_path).to(device)

    policy.config.n_action_steps = raw_chunk_size
    policy.config.num_inference_steps = num_inference_steps

    # Important: actual inference uses policy.diffusion.num_inference_steps.
    policy.diffusion.num_inference_steps = num_inference_steps

    policy.reset()
    policy.eval()

    preprocess, postprocess = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        dataset_stats=None,
    )

    print("[POLICY PROCESS] loaded")
    print("[POLICY PROCESS] input_features:", policy.config.input_features)
    print("[POLICY PROCESS] n_action_steps:", policy.config.n_action_steps)
    print("[POLICY PROCESS] num_inference_steps:", policy.diffusion.num_inference_steps)

    while True:
        raw_obs = request_queue.get()

        if raw_obs is None:
            print("[POLICY PROCESS] stopping")
            break

        try:
            t0 = time.perf_counter()

            # raw_obs is CPU numpy arrays. Convert to torch tensors inside worker process.
            obs_frame = make_batched_obs(raw_obs, device)

            dense_actions = collect_policy_chunk(
                policy=policy,
                preprocess=preprocess,
                postprocess=postprocess,
                obs_frame=obs_frame,
                raw_chunk_size=raw_chunk_size,
                dense_chunk_size=dense_chunk_size,
                device=device,
            )

            t1 = time.perf_counter()

            # Keep latest result only.
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break

            result_queue.put(("ok", dense_actions, t1 - t0))

        except Exception as e:
            result_queue.put(("error", repr(e)))


# =========================
# Main
# =========================
def main():
    print(f"Using device: {DEVICE}")
    print(f"MODEL_PATH: {MODEL_PATH}")

    # Main process does NOT load DiffusionPolicy.
    # Policy is loaded only inside the worker process.
    print("[INFO] Policy will be loaded in worker process")

    # =========================
    # Load MuJoCo / Pinocchio
    # =========================
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data = mujoco.MjData(mj_model)

    if ZERO_GRAVITY:
        mj_model.opt.gravity[:] = 0.0

    pin_model = pin.buildModelFromUrdf(URDF_PATH)
    pin_data = pin_model.createData()

    # Offsets.
    T_off_r_const = get_offset_se3(R_OFF_XYZ, R_OFF_RPY)
    T_off_l_const = get_offset_se3(L_OFF_XYZ, L_OFF_RPY)

    arm_joint_names = get_all_arm_joints()
    torso_joint_names = get_torso_joints()
    joint_names = arm_joint_names + torso_joint_names

    pin_qidx_map = make_pin_joint_index_map(pin_model, joint_names)
    mj_qpos_map = make_mujoco_joint_index_map(mj_model, joint_names)
    actuator_map = make_actuator_index_map(mj_model, joint_names)

    if len(actuator_map) == 0:
        raise RuntimeError("No actuators found. Check actuator names in XML.")

    locked_v_indices = get_locked_v_indices(pin_model)

    q_current = pin.neutral(pin_model)
    set_initial_pose(mj_data, q_current, joint_names, mj_qpos_map, pin_qidx_map)
    q_nominal = q_current.copy()

    mujoco.mj_forward(mj_model, mj_data)
    q_current = mujoco_q_to_pin_q(
        mj_data,
        pin_model,
        joint_names,
        mj_qpos_map,
        pin_qidx_map,
    )

    head_id = pin_model.getFrameId(HEAD_FRAME_NAME)
    if head_id >= len(pin_model.frames):
        raise RuntimeError(f"Frame not found: {HEAD_FRAME_NAME}")

    print(f"[INFO] Head frame found: {HEAD_FRAME_NAME}")

    # =========================
    # Videos
    # =========================
    if USE_ZERO_IMAGES:
        head_reader = ZeroImageReader()
        left_reader = ZeroImageReader()
        right_reader = ZeroImageReader()
    else:
        head_reader = CachedVideoReader(HEAD_MP4)
        left_reader = CachedVideoReader(LEFT_WRIST_MP4)
        right_reader = CachedVideoReader(RIGHT_WRIST_MP4)

    prev_left_target = None
    prev_right_target = None
    last_left_trigger = 0.0
    last_right_trigger = 0.0

    # =========================
    # Multiprocessing policy worker
    # =========================
    action_playback_queue = deque()

    ctx = mp.get_context("spawn")

    policy_request_queue = ctx.Queue(maxsize=1)
    policy_result_queue = ctx.Queue(maxsize=1)

    policy_request_pending = False
    policy_process = None

    if USE_ASYNC_POLICY:
        policy_process = ctx.Process(
            target=policy_process_loop,
            args=(
                MODEL_PATH,
                policy_request_queue,
                policy_result_queue,
                WORKER_DEVICE,
                RAW_CHUNK_SIZE,
                DENSE_CHUNK_SIZE,
                POLICY_NUM_INFERENCE_STEPS,
            ),
        )
        policy_process.start()

    print("\n========== EE DiffusionPolicy + IK simulation ==========")
    print("[INFO] observation.state is current EE pose 16D")
    print("[INFO] action is predicted EE pose 16D")
    print("[INFO] target = (oM_head * predicted_pose) * T_offset")
    print("[INFO] async policy:", USE_ASYNC_POLICY)
    print("[INFO] raw_chunk_size:", RAW_CHUNK_SIZE)
    print("[INFO] dense_chunk_size:", DENSE_CHUNK_SIZE)
    print("[INFO] low_watermark:", LOW_WATERMARK)
    print("[INFO] blend_steps:", BLEND_STEPS)
    print("[INFO] policy num_inference_steps:", POLICY_NUM_INFERENCE_STEPS)
    print("========================================================\n")

    # =========================
    # Prefill first action chunk before opening viewer
    # =========================
    if USE_ASYNC_POLICY:
        print("[ASYNC POLICY] prefill first chunk before viewer...")

        mujoco.mj_forward(mj_model, mj_data)
        q_current = mujoco_q_to_pin_q(
            mj_data,
            pin_model,
            joint_names,
            mj_qpos_map,
            pin_qidx_map,
        )

        current_ee_state = make_current_ee_obs_state(
            pin_model,
            pin_data,
            q_current,
            head_id,
            ARM_CONFIG["left"]["pin_frame_name"],
            ARM_CONFIG["right"]["pin_frame_name"],
            T_off_l_const,
            T_off_r_const,
            last_left_trigger=last_left_trigger,
            last_right_trigger=last_right_trigger,
        )

        head_chw = chw_from_bgr(head_reader.read())
        left_chw = chw_from_bgr(left_reader.read())
        right_chw = chw_from_bgr(right_reader.read())

        raw_obs = {
            "observation.state": current_ee_state.copy(),
            "observation.images.head_cam": head_chw.copy(),
            "observation.images.left_wrist_cam": left_chw.copy(),
            "observation.images.right_wrist_cam": right_chw.copy(),
        }

        policy_request_queue.put(raw_obs)
        status, *payload = policy_result_queue.get()

        if status == "ok":
            dense_actions, policy_dt = payload

            for a in dense_actions:
                action_playback_queue.append(a)

            print(
                f"[ASYNC POLICY] prefilled dense chunk {dense_actions.shape}, "
                f"policy_dt={policy_dt:.3f}s, "
                f"queue_len={len(action_playback_queue)}"
            )
        else:
            err = payload[0]
            raise RuntimeError(f"Policy worker failed during prefill: {err}")

    # =========================
    # Viewer / control loop
    # =========================
    last_time = time.time()
    last_loop_perf = time.perf_counter()

    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            for step in range(N_STEPS):
                if not viewer.is_running():
                    break

                loop_now = time.perf_counter()
                loop_dt = loop_now - last_loop_perf
                last_loop_perf = loop_now

                if loop_dt > 0.06:
                    print(
                        f"[LOOP SPIKE] step={step} "
                        f"dt={loop_dt:.4f}s "
                        f"queue_len={len(action_playback_queue)} "
                        f"pending={policy_request_pending}"
                    )

                now = time.time()
                elapsed = now - last_time
                if elapsed < DT:
                    time.sleep(max(0.0, DT - elapsed))
                last_time = time.time()

                # Current sim -> Pinocchio q.
                mujoco.mj_forward(mj_model, mj_data)
                q_current = mujoco_q_to_pin_q(
                    mj_data,
                    pin_model,
                    joint_names,
                    mj_qpos_map,
                    pin_qidx_map,
                )

                # Current EE obs in same 16D convention as training state.
                current_ee_state = make_current_ee_obs_state(
                    pin_model,
                    pin_data,
                    q_current,
                    head_id,
                    ARM_CONFIG["left"]["pin_frame_name"],
                    ARM_CONFIG["right"]["pin_frame_name"],
                    T_off_l_const,
                    T_off_r_const,
                    last_left_trigger=last_left_trigger,
                    last_right_trigger=last_right_trigger,
                )

                # Images.
                head_chw = chw_from_bgr(head_reader.read())
                left_chw = chw_from_bgr(left_reader.read())
                right_chw = chw_from_bgr(right_reader.read())

                raw_obs = {
                    "observation.state": current_ee_state.copy(),
                    "observation.images.head_cam": head_chw.copy(),
                    "observation.images.left_wrist_cam": left_chw.copy(),
                    "observation.images.right_wrist_cam": right_chw.copy(),
                }

                if step == 0:
                    print("[DEBUG] raw obs keys:", list(raw_obs.keys()))
                    print("[DEBUG] obs state shape:", raw_obs["observation.state"].shape)
                    print("[DEBUG] obs state first 8:", raw_obs["observation.state"][:8])
                    print(
                        "[DEBUG] head image:",
                        head_chw.shape,
                        head_chw.dtype,
                        head_chw.min(),
                        head_chw.max(),
                    )

                # =========================
                # Async policy result handling
                # =========================
                if USE_ASYNC_POLICY:
                    try:
                        status, *payload = policy_result_queue.get_nowait()

                        if status == "ok":
                            dense_actions, policy_dt = payload

                            dense_actions = blend_new_chunk_with_queue_tail(
                                action_playback_queue,
                                dense_actions,
                                BLEND_STEPS,
                            )

                            for a in dense_actions:
                                if len(action_playback_queue) < MAX_ACTION_QUEUE_LEN:
                                    action_playback_queue.append(a)

                            policy_request_pending = False

                            print(
                                f"[ASYNC POLICY] received dense chunk {dense_actions.shape}, "
                                f"policy_dt={policy_dt:.3f}s, "
                                f"queue_len={len(action_playback_queue)}"
                            )

                        else:
                            err = payload[0]
                            policy_request_pending = False
                            print("[ASYNC POLICY ERROR]", err)

                    except queue.Empty:
                        pass

                # Request next policy chunk when playback queue is getting low.
                if USE_ASYNC_POLICY:
                    if len(action_playback_queue) <= LOW_WATERMARK and not policy_request_pending:
                        try:
                            policy_request_queue.put_nowait(raw_obs)
                            policy_request_pending = True
                            print(
                                f"[ASYNC POLICY] requested new chunk, "
                                f"queue_len={len(action_playback_queue)}"
                            )
                        except queue.Full:
                            pass

                # Select action to execute.
                if USE_ASYNC_POLICY:
                    if len(action_playback_queue) > 0:
                        action_vec = action_playback_queue.popleft()
                    else:
                        action_vec = current_ee_state.copy()
                        if step % DEBUG_PRINT_EVERY == 0:
                            print("[ASYNC POLICY] queue empty; holding current EE pose")
                else:
                    # Blocking fallback path. Not recommended for demo.
                    obs_frame = make_batched_obs(raw_obs, DEVICE)

                    t_policy0 = time.perf_counter()

                    # This fallback path requires policy/preprocess/postprocess to be loaded in main,
                    # so do not use it unless you re-enable main-process policy loading.
                    raise RuntimeError(
                        "USE_ASYNC_POLICY=False is not supported in this multiprocessing main()."
                    )

                last_left_trigger = float(action_vec[7])
                last_right_trigger = float(action_vec[15])

                left_pose = ee_vec_to_se3(action_vec[:8])
                right_pose = ee_vec_to_se3(action_vec[8:16])

                # Head-frame action -> world EE target.
                pin.forwardKinematics(pin_model, pin_data, q_current)
                pin.updateFramePlacements(pin_model, pin_data)
                oM_head = pin_data.oMf[head_id]

                left_target = (oM_head * left_pose) * T_off_l_const
                right_target = (oM_head * right_pose) * T_off_r_const

                # Clamp target translation movement for safe first tests.
                left_target = clamp_target_position(
                    prev_left_target,
                    left_target,
                    MAX_EE_POS_DELTA,
                )
                right_target = clamp_target_position(
                    prev_right_target,
                    right_target,
                    MAX_EE_POS_DELTA,
                )

                prev_left_target = left_target
                prev_right_target = right_target

                # IK solve, one arm at a time with the other arm locked.
                q_des = q_current.copy()
                q_des = apply_fixed_torso_to_q(q_des, pin_qidx_map)

                q_des, ok_l = solve_ik_single(
                    pin_model,
                    pin_data,
                    q_des,
                    q_nominal,
                    ARM_CONFIG["left"]["pin_frame_name"],
                    left_target,
                    locked_v_indices,
                    other_arm_keyword="right_arm",
                )

                q_des, ok_r = solve_ik_single(
                    pin_model,
                    pin_data,
                    q_des,
                    q_nominal,
                    ARM_CONFIG["right"]["pin_frame_name"],
                    right_target,
                    locked_v_indices,
                    other_arm_keyword="left_arm",
                )

                # Clamp joint target movement.
                q_des = limit_joint_step(
                    q_current,
                    q_des,
                    pin_qidx_map,
                    joint_names,
                    MAX_JOINT_DELTA,
                )

                apply_q_to_mujoco_ctrl(
                    mj_model,
                    mj_data,
                    q_des,
                    pin_qidx_map,
                    mj_qpos_map,
                    actuator_map,
                    joint_names,
                )

                n_substeps = 16
                for _ in range(n_substeps):
                    mujoco.mj_step(mj_model, mj_data)

                viewer.sync()

                if step % DEBUG_PRINT_EVERY == 0:
                    print_target_debug(
                        step,
                        action_vec,
                        left_target,
                        right_target,
                        ok_l,
                        ok_r,
                        q_current,
                        q_des,
                        joint_names,
                        pin_qidx_map,
                    )

    finally:
        # =========================
        # Cleanup
        # =========================
        if USE_ASYNC_POLICY:
            try:
                # Clear pending request if any.
                while True:
                    try:
                        policy_request_queue.get_nowait()
                    except queue.Empty:
                        break

                policy_request_queue.put(None)
            except Exception:
                pass

            if policy_process is not None:
                try:
                    policy_process.join(timeout=2.0)
                except Exception:
                    pass

                if policy_process.is_alive():
                    policy_process.terminate()

        head_reader.close()
        left_reader.close()
        right_reader.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()