import time
import numpy as np
import mujoco
import mujoco.viewer
import pinocchio as pin

import pandas as pd
import pyarrow.parquet as pq

from scipy.spatial.transform import Rotation as R


# =========================
# Path config
# =========================
XML_PATH = "./scene_rby1a_1.2.xml"
URDF_PATH = "./urdf/model_v1.2_camera.urdf"
DATA_PATH = "/home/soyounglee/datasets/test_good/data/chunk-000/file-000.parquet"


# =========================
# Arm config
# =========================

ARM_CONFIG = {
    "right": {
        "ee_body_name": "EE_BODY_R",
        "pin_frame_name": "ee_right",
        "joint_names": [f"right_arm_{i}" for i in range(7)],
        "init_q": np.array([
            0.0,               # arm_0
            0.0,               # arm_1
            0.0,               # arm_2
            -2.26892803,       # arm_3 = -130 deg
            0.0,               # arm_4
            0.0,               # arm_5
            1.57079633,        # arm_6 = +90 deg
        ], dtype=float),
    },

    "left": {
        "ee_body_name": "EE_BODY_L",
        "pin_frame_name": "ee_left",
        "joint_names": [f"left_arm_{i}" for i in range(7)],
        "init_q": np.array([
            0.0,               # arm_0
            0.0,               # arm_1
            0.0,               # arm_2
            -2.26892803,       # arm_3 = -130 deg
            0.0,               # arm_4
            0.0,               # arm_5
            -1.57079633,       # arm_6 = -90 deg
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


# =========================
# Control / IK params
# =========================
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ
MAX_IK_ITERS = 30
TORQUE_LIMIT = 200.0

# If your MuJoCo actuators are position actuators, keep this True.
# If they are motor/torque actuators, set False and use PD torque.
USE_POSITION_CTRL = True
KP = 500.0
KD = 80.0


R_OFF_XYZ = np.array([-0.0073, 0.0039, -0.0720], dtype=float)
R_OFF_RPY = np.array([98.99, 0.30, 163.89], dtype=float)

L_OFF_XYZ = np.array([-0.0073, -0.0051, -0.0720], dtype=float)
L_OFF_RPY = np.array([-97.58, 1.71, 15.36], dtype=float)


# =========================
# Utility
# =========================

def ee_vec_to_se3(vec):
    vec = np.asarray(vec, dtype=float)

    pos = vec[:3]
    quat_xyzw = vec[3:7]

    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
    rot = R.from_quat(quat_xyzw).as_matrix()

    return pin.SE3(rot, pos)




def get_offset_se3(xyz, rpy_deg):
    """
    Same as C++:
    SE3(R, 0) * SE3(I, xyz)
    """
    rot = R.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    return pin.SE3(rot, np.zeros(3)) * pin.SE3(np.eye(3), xyz)


def ee_vec_to_se3(vec):
    """
    Expected format:
    [x, y, z, qx, qy, qz, qw, gripper]
    """
    vec = np.asarray(vec, dtype=float)

    pos = vec[:3]
    quat_xyzw = vec[3:7]

    rot = R.from_quat(quat_xyzw).as_matrix()
    return pin.SE3(rot, pos)


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

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )

        if actuator_id == -1:
            print(f"[WARN] Actuator not found: joint={joint_name}, expected={actuator_name}")
            continue

        actuator_map[joint_name] = actuator_id

    return actuator_map


def print_name_check(mj_model, pin_model, joint_names):
    print("\n========== Joint / actuator name check ==========")
    for name in joint_names:
        mj_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        pin_ok = pin_model.existJointName(name)
        print(f"{name:15s} | mujoco_joint={mj_jid:3d} | actuator={aid:3d} | pin_joint={pin_ok}")
    print("================================================\n")



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
            # Position actuator case
            mj_data.ctrl[aid] = q_des
        else:
            # Torque / motor actuator case
            if name not in mj_qpos_map:
                continue
            q_now = mj_data.qpos[mj_qpos_map[name]]

            # qvel address is usually same joint id mapping through jnt_dofadr
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


# =========================
# IK solver
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

    for _ in range(max_iters):
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)

        current = pin_data.oMf[frame_id]
        err = pin.log6(current.actInv(target_se3)).vector

        if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
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

        # Keep nominal posture in null space
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

        # Clamp single-DoF joint limits
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

    return q, True


def make_target_from_current_ee(current_ee, xyz_offset, rpy_offset_deg):
    dR = R.from_euler("xyz", rpy_offset_deg, degrees=True).as_matrix()
    target_rot = current_ee.rotation @ dR
    target_pos = current_ee.translation + xyz_offset
    return pin.SE3(target_rot, target_pos)


def print_mujoco_ee_debug(mj_model, mj_data):
    for side, cfg in ARM_CONFIG.items():
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, cfg["ee_body_name"])
        if body_id >= 0:
            pos = mj_data.xpos[body_id]
            mat = mj_data.xmat[body_id].reshape(3, 3)
            print(f"[{side}] MuJoCo EE pos: {pos}, rot first row: {mat[0]}")
        else:
            print(f"[WARN] MuJoCo EE body not found: {cfg['ee_body_name']}")


def apply_fixed_torso_to_q(
    q,
    pin_qidx_map,
):
    q_out = q.copy()

    for name, val in TORSO_INIT.items():

        if name in pin_qidx_map:
            q_out[pin_qidx_map[name]] = val

    return q_out



# =========================
# Main
# =========================
def main():
    # =========================
    # Load MuJoCo / Pinocchio
    # =========================
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data = mujoco.MjData(mj_model)

    pin_model = pin.buildModelFromUrdf(URDF_PATH)
    pin_data = pin_model.createData()

    # =========================
    # Load parquet trajectory
    # =========================
    print(f"[INFO] Loading trajectory: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    DATA_MODE = "observation"

    left_col = f"{DATA_MODE}.left_arm"
    right_col = f"{DATA_MODE}.right_arm"

    required_cols = [left_col, right_col]

    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"[ERROR] Missing column in parquet: {col}")

    print(f"[INFO] Loaded trajectory length: {len(df)}")
    print(f"[INFO] Using columns:")
    print(f"  left : {left_col}")
    print(f"  right: {right_col}")

    # =========================
    # Build offset transforms
    # =========================
    T_off_r_const = get_offset_se3(R_OFF_XYZ, R_OFF_RPY)
    T_off_l_const = get_offset_se3(L_OFF_XYZ, L_OFF_RPY)

    # =========================
    # Joint mappings
    # =========================
    arm_joint_names = get_all_arm_joints()
    torso_joint_names = get_torso_joints()

    joint_names = arm_joint_names + torso_joint_names

    pin_qidx_map = make_pin_joint_index_map(
        pin_model,
        joint_names,
    )

    mj_qpos_map = make_mujoco_joint_index_map(
        mj_model,
        joint_names,
    )

    actuator_map = make_actuator_index_map(
        mj_model,
        joint_names,
    )

    if len(actuator_map) == 0:
        print("[ERROR] No actuators found.")
        return

    locked_v_indices = get_locked_v_indices(pin_model)

    # =========================
    # Initialize pose
    # =========================
    q_current = pin.neutral(pin_model)

    set_initial_pose(
        mj_data,
        q_current,
        joint_names,
        mj_qpos_map,
        pin_qidx_map,
    )

    q_nominal = q_current.copy()

    mujoco.mj_forward(mj_model, mj_data)

    q_current = mujoco_q_to_pin_q(
        mj_data,
        pin_model,
        joint_names,
        mj_qpos_map,
        pin_qidx_map,
    )

    # =========================
    # Head frame check
    # =========================
    head_frame_name = "zed2i_left_camera_optical_frame"

    head_id = pin_model.getFrameId(head_frame_name)

    if head_id >= len(pin_model.frames):
        print(f"[ERROR] Frame not found: {head_frame_name}")
        return

    print(f"[INFO] Head frame found: {head_frame_name}")

    # =========================
    # Debug / timing
    # =========================
    POS_TOL = 0.01
    ROT_TOL = 0.05

    DEBUG_PRINT_DT = 0.1

    last_debug_time = time.time()
    last_time = time.time()

    frame_counter = 0

    print("\n========== Real EE trajectory playback ==========")
    print("[INFO] Using C++ teleop target logic")
    print("[INFO] target = (oM_head * controller_pose) * T_offset")
    print("[INFO] Close viewer manually to stop.")
    print("=================================================\n")

    # =========================
    # Viewer loop
    # =========================
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        while viewer.is_running():

            now = time.time()

            if now - last_time < DT:
                time.sleep(0.001)
                viewer.sync()
                continue

            last_time = now

            traj_idx = min(frame_counter, len(df) - 1)

            # =========================
            # Read parquet EE poses
            # =========================
            left_pose = ee_vec_to_se3(
                df.iloc[traj_idx][left_col]
            )

            right_pose = ee_vec_to_se3(
                df.iloc[traj_idx][right_col]
            )

            # =========================
            # Current robot FK
            # =========================
            mujoco.mj_forward(mj_model, mj_data)

            q_current = mujoco_q_to_pin_q(
                mj_data,
                pin_model,
                joint_names,
                mj_qpos_map,
                pin_qidx_map,
            )

            pin.forwardKinematics(
                pin_model,
                pin_data,
                q_current,
            )

            pin.updateFramePlacements(
                pin_model,
                pin_data,
            )

            # =========================
            # Head frame
            # =========================
            oM_head = pin_data.oMf[head_id]

            # =========================
            # C++ target logic
            # target = (oM_head * pose) * T_offset
            # =========================
            left_target = (
                (oM_head * left_pose)
                * T_off_l_const
            )

            right_target = (
                (oM_head * right_pose)
                * T_off_r_const
            )

            target_map = {
                "left": left_target,
                "right": right_target,
            }

            # =========================
            # Debug print timing
            # =========================
            debug_now = time.time()

            do_debug_print = (
                (debug_now - last_debug_time)
                > DEBUG_PRINT_DT
            )

            # =========================
            # IK solve
            # =========================
            q_des = q_current.copy()

            q_des = apply_fixed_torso_to_q(
                q_des,
                pin_qidx_map,
            )



            all_targets_reached = True

            for side, cfg in ARM_CONFIG.items():

                frame_name = cfg["pin_frame_name"]

                target = target_map[side]

                pin.forwardKinematics(
                    pin_model,
                    pin_data,
                    q_des,
                )

                pin.updateFramePlacements(
                    pin_model,
                    pin_data,
                )

                frame_id = pin_model.getFrameId(frame_name)

                if frame_id >= len(pin_model.frames):
                    print(f"[ERROR] Pinocchio frame not found: {frame_name}")
                    all_targets_reached = False
                    continue

                current_ee = pin_data.oMf[frame_id]

                other_arm = (
                    "left_arm"
                    if side == "right"
                    else "right_arm"
                )

                q_des, ok = solve_ik_single(
                    pin_model,
                    pin_data,
                    q_des,
                    q_nominal,
                    frame_name,
                    target,
                    locked_v_indices,
                    other_arm_keyword=other_arm,
                )

                pos_err = (
                    target.translation
                    - current_ee.translation
                )

                rot_err = pin.log3(
                    current_ee.rotation.T
                    @ target.rotation
                )

                pos_err_norm = np.linalg.norm(pos_err)
                rot_err_norm = np.linalg.norm(rot_err)

                if (
                    (not ok)
                    or pos_err_norm > POS_TOL
                    or rot_err_norm > ROT_TOL
                ):
                    all_targets_reached = False

                if do_debug_print:

                    print(f"\n[{side}] EE debug | traj_idx={traj_idx}")

                    print(f"  current pos : {current_ee.translation}")

                    print(f"  target  pos : {target.translation}")

                    print(f"  pos err     : {pos_err}")
                    print(f"  pos err norm: {pos_err_norm:.6f}")

                    print(f"  rot err     : {rot_err}")
                    print(f"  rot err norm: {rot_err_norm:.6f}")

                    print(f"  IK ok       : {ok}")

                if not ok:
                    print(f"[WARN] IK failed: {side}, traj_idx={traj_idx}")

            # =========================
            # Debug timer update
            # =========================
            if do_debug_print:
                last_debug_time = debug_now

            # =========================
            # Apply control
            # =========================
            apply_q_to_mujoco_ctrl(
                mj_model,
                mj_data,
                q_des,
                pin_qidx_map,
                mj_qpos_map,
                actuator_map,
                joint_names,
            )

            mujoco.mj_step(mj_model, mj_data)

            viewer.sync()

            # =========================
            # Next trajectory frame
            # =========================
            frame_counter += 1

            if frame_counter >= len(df):

                print("\n[INFO] Reached end of trajectory.")
                print("[INFO] Holding last pose.")

                frame_counter = len(df) - 1


if __name__ == "__main__":
    main()
