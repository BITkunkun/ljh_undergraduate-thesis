#!/usr/bin/env python3
"""
毕业设计 Gazebo：0415 局部异构收敛 + 0414 编队动态（移动中心）串联

流程：
1. 平地起飞、全维 PID 收敛到 MATLAB 扰动下的「近四面体」、定点保持后切换论文律（与 ``final_0415.py`` 一致）。
2. 异构量测闭环 + 上帝视角门控，收敛并连续保持于「正四面体」指标后，**不立即降落**。
3. 进入动态任务：以当前四机位置**均值**为 c0，按参数累加位移得 c1,c2,c3；时间轴（ROS/仿真时钟）：
   gap → 段1（c0→c1 + 扩张）→ gap → 段2（c1→c2 + 绕 Z 旋转）→ gap → 段3（c2→c3 + 收缩）→ gap → 降落。
   名义中心 ``c_nom(t)`` 仍按段内线性插值；**参考中心** ``c_ref`` 由一阶速度饱和跟踪 ``c_nom``（参数 ``mission_center_max_speed``），
   使平移参考受速度约束，减轻编队跟踪时瞬时拉大边距、逼近通讯/碰撞边界的趋势。
   该阶段采用**全维 PID** 跟踪 ``compose_desired_positions(c_ref, s, theta, …)``（与 ``final_0414.py`` 一致），
   并可用 ``mission_track_max_xy_speed`` 略收紧水平速度上限。

4. **全局轨迹记录**：自进入异构量测阶段起，每个控制周期将 ROS 时间与四机世界系三维位置写入
   ``trajectory_xyz_log_path``（默认 ``quad_trajectory_move.csv``），直至动态任务末段悬停结束（即将降落），供 ``复现论文+code/code_ljh/plot_2.m`` 绘制单机轨迹。
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

from reference_trajectory import compose_desired_positions

DIM = 3
N_AGENT = 4
N_EDGE = 6

BASE_POSITION_TEMPLATE = np.array(
    [
        [1.0, 2.0, 1.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 2.0, 2.0, 1.0],
    ],
    dtype=float,
)

H_INCIDENCE = np.array(
    [
        [1, 1, 1, 0, 0, 0],
        [-1, 0, 0, 1, 1, 0],
        [0, -1, 0, -1, 0, 1],
        [0, 0, -1, 0, -1, -1],
    ],
    dtype=float,
)

ADJ_FULL = np.array(
    [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ],
    dtype=float,
)

EDGE_LENGTH_MIN = 0.5
EDGE_LENGTH_MAX = 5.0


def edge_spacing_safety_violation(L: np.ndarray) -> Tuple[bool, int, float]:
    for j in range(N_EDGE):
        lj = float(L[j])
        if lj < EDGE_LENGTH_MIN or lj > EDGE_LENGTH_MAX:
            return True, j, lj
    return False, -1, 0.0


def rigidity_standard(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.zeros(N_EDGE, dtype=float)
    R = np.zeros((N_EDGE, N_AGENT * DIM), dtype=float)
    for j in range(N_EDGE):
        z = P @ H_INCIDENCE[:, j]
        L = float(np.linalg.norm(z))
        edges[j] = L
        if L < 1e-9:
            zhat = np.zeros(3, dtype=float)
        else:
            zhat = z / L
        for k in range(N_AGENT):
            R[j, DIM * k : DIM * (k + 1)] = H_INCIDENCE[k, j] * zhat
    return edges, R


def dgamma_scalar(
    j: int,
    edge_length_current: np.ndarray,
    edge_length_desired: np.ndarray,
    el_min: float = EDGE_LENGTH_MIN,
    el_max: float = EDGE_LENGTH_MAX,
) -> float:
    elc = float(edge_length_current[j])
    eld = float(edge_length_desired[j])
    eps = 1e-6
    elc = float(np.clip(elc, el_min + eps, el_max - eps))
    eld = float(np.clip(eld, el_min + eps, el_max - eps))
    return 0.001 * (
        2.0 * (elc - eld)
        - (el_min + el_max - 2.0 * elc) / ((el_max - elc) * (elc - el_min) + eps)
        + (el_min + el_max - 2.0 * eld) / ((el_max - elc) * (eld - el_min) + eps)
    )


def build_output_position_matrix() -> np.ndarray:
    d = np.zeros(12, dtype=float)
    d[0:3] = [0.0, 0.0, 1.0]
    d[3:6] = [0.0, 0.0, 0.0]
    d[6:9] = [1.0, 1.0, 0.0]
    d[9:12] = [1.0, 1.0, 1.0]
    return np.diag(d)


def build_heterogeneous_measurement(
    P: np.ndarray,
    edge_lengths: np.ndarray,
    output_pos: np.ndarray,
) -> np.ndarray:
    p_vec = P.reshape(-1, order="F")
    meas_pos = output_pos @ p_vec
    return np.concatenate([0.5 * edge_lengths**2, meas_pos])


def god_view_tetrahedron_metrics(
    P: np.ndarray,
    P_des: np.ndarray,
    el_des: np.ndarray,
) -> Tuple[float, float]:
    el_cur, _ = rigidity_standard(P)
    pos_max = max(float(np.linalg.norm(P[:, i] - P_des[:, i])) for i in range(N_AGENT))
    edge_max = float(np.max(np.abs(el_cur - el_des)))
    return pos_max, edge_max


def thesis_controller(
    agent_idx: int,
    edge_length_current: np.ndarray,
    edge_length_desired: np.ndarray,
    R_est: np.ndarray,
    estimation: np.ndarray,
    P_des: np.ndarray,
) -> np.ndarray:
    u = np.zeros(3, dtype=float)
    for j in range(N_EDGE):
        dg = dgamma_scalar(j, edge_length_current, edge_length_desired)
        row = R_est[j, agent_idx * DIM : (agent_idx + 1) * DIM]
        u -= abs(H_INCIDENCE[agent_idx, j]) * dg * row
    u -= 2.5 * (estimation[:, agent_idx] - P_des[:, agent_idx])
    return u


def thesis_observer(
    agent_idx: int,
    measure_current: np.ndarray,
    estimation: np.ndarray,
    P_des: np.ndarray,
    edge_length_desired: np.ndarray,
    output_matrix: np.ndarray,
) -> np.ndarray:
    measure_desired = np.concatenate(
        [0.5 * edge_length_desired**2, P_des.reshape(-1, order="F")]
    )
    C_i = output_matrix[:, agent_idx * DIM : (agent_idx + 1) * DIM]
    feedback = np.zeros(output_matrix.shape[0], dtype=float)
    for j in range(N_AGENT):
        C_j = output_matrix[:, j * DIM : (j + 1) * DIM]
        feedback -= ADJ_FULL[agent_idx, j] * (C_j @ (estimation[:, j] - P_des[:, j]))
    feedback += (measure_current - measure_desired) - C_i @ (estimation[:, agent_idx] - P_des[:, agent_idx])
    return 5.0 * (C_i.T @ feedback)


class ThesisHeteroThenMovingNode(Node):
    """0415 论文阶段完成后衔接 0414 式移动中心动态 PID 任务。"""

    def __init__(self) -> None:
        super().__init__("final_thesis_hetero_then_moving_0415_move")

        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("max_speed", 3.0)
        self.declare_parameter("takeoff_height", 2.0)
        self.declare_parameter("takeoff_tolerance", 0.08)

        self.declare_parameter("desired_edge_length", math.sqrt(2.0))
        self.declare_parameter("formation_tolerance", 0.12)
        self.declare_parameter("near_formation_tolerance", 0.15)

        self.declare_parameter("kp_xy", 0.95)
        self.declare_parameter("ki_xy", 0.02)
        self.declare_parameter("kd_xy", 0.25)
        self.declare_parameter("kp_z", 1.25)
        self.declare_parameter("ki_z", 0.02)
        self.declare_parameter("kd_z", 0.30)
        self.declare_parameter("integral_limit_xy", 2.0)
        self.declare_parameter("integral_limit_z", 1.0)
        self.declare_parameter("max_xy_speed", 2.5)
        self.declare_parameter("max_z_speed", 1.5)

        self.declare_parameter("matlab_perturb_low", -0.25)
        self.declare_parameter("matlab_perturb_high", 0.25)

        self.declare_parameter("loiter_duration", 5.0)

        self.declare_parameter("thesis_max_speed", 2.0)

        self.declare_parameter("god_view_pos_tol", 0.12)
        self.declare_parameter("god_view_edge_tol", 0.10)
        self.declare_parameter("god_view_hold_sec", 5.0)
        self.declare_parameter("god_view_debug_period", 0.0)

        # --- 动态段（与 final_0414 对齐；默认加长段时长、缩短位移以降低中心速度，利于通讯边界内运行）---
        self.declare_parameter("mission_segment_duration", 45.0)
        self.declare_parameter("mission_between_segment_duration", 5.0)
        # 名义时间轴上的中心 c_nom(t) 仍按段线性变化；参考中心 c_ref 由一阶速度饱和跟踪 c_nom，
        # 使编队 PID 跟踪的平移参考受速度约束，减轻收敛过程中边距/通讯裕度被瞬时拉大。
        self.declare_parameter("mission_center_max_speed", 0.10)
        # 动态段跟踪参考时略收紧水平速度上限（≤ max_xy_speed），可选与中心限速配合。
        self.declare_parameter("mission_track_max_xy_speed", 1.35)
        self.declare_parameter("mission_scale_start", 1.0)
        self.declare_parameter("mission_scale_expand", 2.0)
        self.declare_parameter("mission_scale_contract", 1.0)
        self.declare_parameter("mission_delta1_x", 1.5)
        self.declare_parameter("mission_delta1_y", -0.75)
        self.declare_parameter("mission_delta1_z", 0.0)
        self.declare_parameter("mission_delta2_x", 1.5)
        self.declare_parameter("mission_delta2_y", 0.75)
        self.declare_parameter("mission_delta2_z", 0.0)
        self.declare_parameter("mission_delta3_x", 1.5)
        self.declare_parameter("mission_delta3_y", -0.75)
        self.declare_parameter("mission_delta3_z", 0.0)

        self.declare_parameter("landing_speed", 0.35)
        self.declare_parameter("landing_tolerance", 0.08)

        _default_thesis_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_phase_data_move.txt")
        self.declare_parameter("thesis_data_log_path", _default_thesis_log)
        _default_traj_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quad_trajectory_move.csv")
        self.declare_parameter("trajectory_xyz_log_path", _default_traj_csv)

        self.dt = float(self.get_parameter("control_period").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.takeoff_height = float(self.get_parameter("takeoff_height").value)
        self.takeoff_tolerance = float(self.get_parameter("takeoff_tolerance").value)

        self.desired_edge_length = float(self.get_parameter("desired_edge_length").value)
        self.formation_tolerance = float(self.get_parameter("formation_tolerance").value)
        self.near_formation_tolerance = float(self.get_parameter("near_formation_tolerance").value)

        self.kp_xy = float(self.get_parameter("kp_xy").value)
        self.ki_xy = float(self.get_parameter("ki_xy").value)
        self.kd_xy = float(self.get_parameter("kd_xy").value)
        self.kp_z = float(self.get_parameter("kp_z").value)
        self.ki_z = float(self.get_parameter("ki_z").value)
        self.kd_z = float(self.get_parameter("kd_z").value)
        self.integral_limit_xy = float(self.get_parameter("integral_limit_xy").value)
        self.integral_limit_z = float(self.get_parameter("integral_limit_z").value)
        self.max_xy_speed = float(self.get_parameter("max_xy_speed").value)
        self.max_z_speed = float(self.get_parameter("max_z_speed").value)

        self.matlab_perturb_low = float(self.get_parameter("matlab_perturb_low").value)
        self.matlab_perturb_high = float(self.get_parameter("matlab_perturb_high").value)

        self.loiter_duration = float(self.get_parameter("loiter_duration").value)

        self.thesis_max_speed = float(self.get_parameter("thesis_max_speed").value)

        self.god_view_pos_tol = float(self.get_parameter("god_view_pos_tol").value)
        self.god_view_edge_tol = float(self.get_parameter("god_view_edge_tol").value)
        self.god_view_hold_sec = float(self.get_parameter("god_view_hold_sec").value)
        self.god_view_debug_period = float(self.get_parameter("god_view_debug_period").value)

        self.mission_segment_duration = float(self.get_parameter("mission_segment_duration").value)
        self.mission_between_segment_duration = float(
            self.get_parameter("mission_between_segment_duration").value
        )
        self.mission_center_max_speed = max(0.0, float(self.get_parameter("mission_center_max_speed").value))
        self.mission_track_max_xy_speed = min(
            float(self.get_parameter("mission_track_max_xy_speed").value),
            self.max_xy_speed,
        )
        self.mission_scale_start = float(self.get_parameter("mission_scale_start").value)
        self.mission_scale_expand = float(self.get_parameter("mission_scale_expand").value)
        self.mission_scale_contract = float(self.get_parameter("mission_scale_contract").value)

        self.landing_speed = float(self.get_parameter("landing_speed").value)
        self.landing_tolerance = float(self.get_parameter("landing_tolerance").value)

        self.thesis_data_log_path = str(self.get_parameter("thesis_data_log_path").value)
        self.trajectory_xyz_log_path = str(self.get_parameter("trajectory_xyz_log_path").value)

        self.agent_names = [f"quadrotor_{i}" for i in range(1, N_AGENT + 1)]
        self.positions: Dict[str, Optional[np.ndarray]] = {n: None for n in self.agent_names}
        self.initial_ground: Dict[str, np.ndarray] = {}
        self.hover_z = 0.0
        self.phase = "wait_poses"

        self.integral_error: Dict[str, np.ndarray] = {n: np.zeros(3, dtype=float) for n in self.agent_names}
        self.previous_error: Dict[str, np.ndarray] = {n: np.zeros(3, dtype=float) for n in self.agent_names}
        self.last_ros_time_sec: Optional[float] = None

        self.base_shape = self._build_base_shape_matrix()
        self.tetra_offsets = {n: self.base_shape[:, i].copy() for i, n in enumerate(self.agent_names)}

        rng = np.random.default_rng(1)
        span = self.matlab_perturb_high - self.matlab_perturb_low
        self._delta_near = self.matlab_perturb_low + span * rng.random((DIM, N_AGENT))

        self._near_targets: Optional[Dict[str, np.ndarray]] = None
        self._loiter_anchor: Optional[np.ndarray] = None
        self._loiter_t0: Optional[float] = None

        self._P_des: Optional[np.ndarray] = None
        self._edge_length_desired: Optional[np.ndarray] = None
        self._R_des: Optional[np.ndarray] = None
        self._output_matrix: Optional[np.ndarray] = None
        self._output_pos = build_output_position_matrix()
        self._estimation: Optional[np.ndarray] = None

        self._thesis_t0: Optional[float] = None
        self._god_view_ok_since: Optional[float] = None
        self._last_god_debug_time: Optional[float] = None

        self.mission_t0: Optional[float] = None
        self.mission_waypoints: Optional[List[np.ndarray]] = None
        self._mission_center_smooth: Optional[np.ndarray] = None

        self._last_stage_log_key = ""
        self._edge_safety_abort_done = False

        self._thesis_log_fp: Optional[TextIO] = None
        self._thesis_log_writer: Optional[Any] = None

        self._trajectory_log_fp: Optional[TextIO] = None
        self._trajectory_csv_writer: Optional[Any] = None
        self._trajectory_t0: Optional[float] = None

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE

        self.pubs = {}
        for name in self.agent_names:
            self.pubs[name] = self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            self.create_subscription(
                TFMessage, f"/{name}/world_pose", lambda msg, n=name: self.pose_callback(msg, n), qos
            )

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(
            "[启动] 0415_move：近四面体 → 论文异构收敛 → 移动中心动态 PID（c0…c3）→ 降落"
        )

    def _build_base_shape_matrix(self) -> np.ndarray:
        scale = self.desired_edge_length / math.sqrt(2.0)
        c = np.mean(BASE_POSITION_TEMPLATE, axis=1, keepdims=True)
        return (BASE_POSITION_TEMPLATE - c) * scale

    def _read_delta_vector(self, prefix: str) -> np.ndarray:
        x = float(self.get_parameter(f"{prefix}_x").value)
        y = float(self.get_parameter(f"{prefix}_y").value)
        z = float(self.get_parameter(f"{prefix}_z").value)
        return np.array([x, y, z], dtype=float)

    def pose_callback(self, msg: TFMessage, agent_name: str) -> None:
        if not msg.transforms:
            return
        t = msg.transforms[0].transform.translation
        self.positions[agent_name] = np.array([t.x, t.y, t.z], dtype=float)

    def position_matrix(self) -> Optional[np.ndarray]:
        if any(self.positions[n] is None for n in self.agent_names):
            return None
        return np.column_stack([self.positions[n] for n in self.agent_names])

    def publish_velocity(self, name: str, v: np.ndarray) -> None:
        sp = np.linalg.norm(v)
        cap = self.max_speed if self.phase != "thesis_hetero" else self.thesis_max_speed
        if sp > cap and sp > 1e-9:
            v = v * (cap / sp)
        m = Twist()
        m.linear.x = float(v[0])
        m.linear.y = float(v[1])
        m.linear.z = float(v[2])
        self.pubs[name].publish(m)

    def stop_all(self) -> None:
        z = np.zeros(3, dtype=float)
        for n in self.agent_names:
            self.publish_velocity(n, z)

    def _thesis_log_header_row(self) -> List[str]:
        cols = ["t_phase_ros_s", "ros_time_abs_s"]
        for prefix in ("act", "des", "ctrl", "est"):
            for i in range(1, N_AGENT + 1):
                for ax in ("x", "y", "z"):
                    cols.append(f"{prefix}_{i}_{ax}")
        return cols

    def _open_thesis_data_log(self, ros_abs_t0: float) -> None:
        self.close_thesis_log_if_open()
        try:
            fp = open(self.thesis_data_log_path, "w", newline="", encoding="utf-8")
        except OSError as exc:
            self.get_logger().error(f"无法打开论文阶段数据文件 {self.thesis_data_log_path}: {exc}")
            return
        self._thesis_log_fp = fp
        self._thesis_log_writer = csv.writer(fp)
        self._thesis_log_writer.writerow(self._thesis_log_header_row())
        self.get_logger().info(
            f"[数据记录] 异构量测阶段 → 动态任务前写入 {self.thesis_data_log_path} "
            f"(t0={ros_abs_t0:.6f}s)"
        )

    def _append_thesis_data_row(self, P: np.ndarray, now: float) -> None:
        if self._thesis_log_writer is None or self._thesis_t0 is None:
            return
        if self._P_des is None or self._estimation is None:
            return
        t_rel = float(now - self._thesis_t0)
        ros_abs = float(now)
        act = P.reshape(-1, order="F")
        des = self._P_des.reshape(-1, order="F")
        ctrl = (P - self._P_des).reshape(-1, order="F")
        est = (P - self._estimation).reshape(-1, order="F")
        row: List[float] = [t_rel, ros_abs]
        row.extend(float(x) for x in np.concatenate([act, des, ctrl, est]))
        self._thesis_log_writer.writerow(row)

    def close_thesis_log_if_open(self) -> None:
        if self._thesis_log_fp is not None:
            try:
                self._thesis_log_fp.flush()
                self._thesis_log_fp.close()
            except OSError:
                pass
            self._thesis_log_fp = None
            self._thesis_log_writer = None

    def _trajectory_header_row(self) -> List[str]:
        cols = ["t_rel_s", "ros_time_abs_s"]
        for i in range(1, N_AGENT + 1):
            cols.extend([f"p{i}_x_m", f"p{i}_y_m", f"p{i}_z_m"])
        return cols

    def _open_trajectory_log(self, ros_t0: float) -> None:
        """自异构量测阶段起记录四机世界系位置，直至动态任务结束（末段 gap 结束）。"""
        self.close_trajectory_log_if_open()
        self._trajectory_t0 = float(ros_t0)
        try:
            fp = open(self.trajectory_xyz_log_path, "w", newline="", encoding="utf-8")
        except OSError as exc:
            self.get_logger().error(f"无法打开轨迹 CSV {self.trajectory_xyz_log_path}: {exc}")
            self._trajectory_t0 = None
            return
        self._trajectory_log_fp = fp
        self._trajectory_csv_writer = csv.writer(fp)
        self._trajectory_csv_writer.writerow(self._trajectory_header_row())
        self.get_logger().info(
            f"[轨迹CSV] 四机全局坐标 (t_rel, ros_abs, xyz×4) → {self.trajectory_xyz_log_path}"
        )

    def _append_trajectory_row(self, P: np.ndarray, now: float) -> None:
        if self._trajectory_csv_writer is None or self._trajectory_t0 is None:
            return
        t_rel = float(now - self._trajectory_t0)
        row: List[float] = [t_rel, float(now)]
        for i in range(N_AGENT):
            row.extend(float(P[j, i]) for j in range(DIM))
        self._trajectory_csv_writer.writerow(row)

    def close_trajectory_log_if_open(self) -> None:
        if self._trajectory_log_fp is not None:
            try:
                self._trajectory_log_fp.flush()
                self._trajectory_log_fp.close()
            except OSError:
                pass
            self._trajectory_log_fp = None
            self._trajectory_csv_writer = None

    def reset_pid(self) -> None:
        for n in self.agent_names:
            self.integral_error[n] = np.zeros(3, dtype=float)
            self.previous_error[n] = np.zeros(3, dtype=float)

    def build_formation_targets(self, position_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        xy_c = np.mean(position_matrix[:2, :], axis=1)
        center = np.array([xy_c[0], xy_c[1], self.hover_z], dtype=float)
        return {n: center + self.tetra_offsets[n] for n in self.agent_names}

    def build_near_targets(self, position_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        ideal = self.build_formation_targets(position_matrix)
        out: Dict[str, np.ndarray] = {}
        for i, n in enumerate(self.agent_names):
            col = np.column_stack([ideal[n]])[:, 0]
            out[n] = col + self._delta_near[:, i]
        return out

    def compute_pid_velocity(
        self,
        agent_name: str,
        target: np.ndarray,
        current: np.ndarray,
        dt: float,
        max_xy_cap: Optional[float] = None,
    ) -> np.ndarray:
        error = target - current
        integral = self.integral_error[agent_name] + error * dt
        integral[0:2] = np.clip(integral[0:2], -self.integral_limit_xy, self.integral_limit_xy)
        integral[2] = float(np.clip(integral[2], -self.integral_limit_z, self.integral_limit_z))
        derivative = (error - self.previous_error[agent_name]) / max(dt, 1e-3)
        self.integral_error[agent_name] = integral
        self.previous_error[agent_name] = error

        v = np.zeros(3, dtype=float)
        v[0] = self.kp_xy * error[0] + self.ki_xy * integral[0] + self.kd_xy * derivative[0]
        v[1] = self.kp_xy * error[1] + self.ki_xy * integral[1] + self.kd_xy * derivative[1]
        v[2] = self.kp_z * error[2] + self.ki_z * integral[2] + self.kd_z * derivative[2]

        cap_xy = self.max_xy_speed if max_xy_cap is None else float(max_xy_cap)
        xy_sp = np.linalg.norm(v[:2])
        if xy_sp > cap_xy and xy_sp > 1e-9:
            v[:2] = v[:2] * (cap_xy / xy_sp)
        v[2] = float(np.clip(v[2], -self.max_z_speed, self.max_z_speed))
        return v

    def track_targets_pid(
        self,
        target_map: Dict[str, np.ndarray],
        dt: float,
        max_xy_cap: Optional[float] = None,
    ) -> Dict[str, float]:
        errors = {}
        for name in self.agent_names:
            cur = np.array(self.positions[name], dtype=float)
            vel = self.compute_pid_velocity(name, target_map[name], cur, dt, max_xy_cap=max_xy_cap)
            self.publish_velocity(name, vel)
            errors[name] = float(np.linalg.norm(target_map[name] - cur))
        return errors

    def log_stage_once(self, key: str, message: str) -> None:
        if self._last_stage_log_key != key:
            self._last_stage_log_key = key
            self.get_logger().info(message)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _init_thesis_globals(self, P_des: np.ndarray) -> None:
        el, R_d = rigidity_standard(P_des)
        out_top = R_d
        out_bot = np.zeros((N_AGENT * DIM, N_AGENT * DIM), dtype=float)
        self._P_des = P_des
        self._edge_length_desired = el
        self._R_des = R_d
        self._output_matrix = np.vstack([out_top, out_bot])

    def _desired_matrix_from_c_s_theta(self, c: np.ndarray, s: float, theta: float) -> np.ndarray:
        return compose_desired_positions(c, s, theta, self.base_shape)

    def _abort_edge_spacing_unsafe(self, L: np.ndarray, edge_idx: int, Lj: float) -> None:
        if self._edge_safety_abort_done:
            return
        self._edge_safety_abort_done = True
        self.get_logger().error(
            f"[安全停止] 边 {edge_idx} 间距 L={Lj:.4f} m 超出安全范围 "
            f"[{EDGE_LENGTH_MIN}, {EDGE_LENGTH_MAX}] m。已停止 cmd_vel 并退出。"
        )
        self.get_logger().error(f"当前六边长 (m): {np.array2string(L, precision=4)}")
        self.phase = "aborted"
        self.close_thesis_log_if_open()
        self.close_trajectory_log_if_open()
        self.stop_all()
        rclpy.shutdown()

    def _hetero_line_step(self, P: np.ndarray, dt_loop: float) -> None:
        assert self._P_des is not None
        assert self._estimation is not None
        assert self._output_matrix is not None
        assert self._edge_length_desired is not None

        el_cur, _ = rigidity_standard(P)
        el_cur = np.maximum(el_cur, EDGE_LENGTH_MIN + 1e-4)
        _, R_est = rigidity_standard(self._estimation)
        measure_current = build_heterogeneous_measurement(P, el_cur, self._output_pos)
        el_des = self._edge_length_desired

        est_dot_cols: List[np.ndarray] = []
        for i in range(N_AGENT):
            c = thesis_controller(
                i,
                el_cur,
                el_des,
                R_est,
                self._estimation,
                self._P_des,
            )
            o_raw = thesis_observer(
                i,
                measure_current,
                self._estimation,
                self._P_des,
                el_des,
                self._output_matrix,
            )
            local_O = self._output_pos[i * DIM : (i + 1) * DIM, i * DIM : (i + 1) * DIM]
            eye = np.eye(3, dtype=float)
            o_mid = (eye - local_O) @ o_raw + 10.0 * local_O @ (P[:, i] - self._estimation[:, i])
            self.publish_velocity(self.agent_names[i], c)
            est_dot_cols.append(c + o_mid)

        self._estimation = self._estimation + np.column_stack(est_dot_cols) * dt_loop

    def get_mission_nominal_c_s_theta(
        self, tau: float, T: float, Gb: float
    ) -> Optional[Tuple[np.ndarray, float, float, str, str, bool]]:
        """
        按仿真时间轴返回名义 (c_nom, s, theta) 与日志键；任务结束时返回 None。
        平移参考应由调用方对 c_nom 做速度饱和跟踪后再调用 compose_desired_positions。
        """
        assert self.mission_waypoints is not None
        w = self.mission_waypoints
        c0, c1, c2, c3 = w[0], w[1], w[2], w[3]

        if tau < Gb:
            s = self.mission_scale_start
            theta = 0.0
            return (c0.copy(), s, theta, "gap0", f"[间隔] 段前悬停 {Gb:.0f}s（ROS/仿真时间）…", True)

        if tau < Gb + T:
            alpha = (tau - Gb) / T
            c = c0 + alpha * (c1 - c0)
            s = self.mission_scale_start + alpha * (self.mission_scale_expand - self.mission_scale_start)
            theta = 0.0
            return (c, s, theta, "seg_move_expand", f"[段1/3] 中心 c0→c1 全维移动 + 扩张 (耗时 {T}s)", False)

        if tau < 2 * Gb + T:
            s = self.mission_scale_expand
            theta = 0.0
            return (c1.copy(), s, theta, "gap1", f"[间隔] 段间悬停 {Gb:.0f}s…", True)

        if tau < 2 * Gb + 2 * T:
            alpha = (tau - (2 * Gb + T)) / T
            c = c1 + alpha * (c2 - c1)
            s = self.mission_scale_expand
            theta = (math.pi / 2.0) * alpha
            return (c, s, theta, "seg_move_rotate", f"[段2/3] 中心 c1→c2 全维移动 + 绕 Z 旋转至 90° (耗时 {T}s)", False)

        if tau < 3 * Gb + 2 * T:
            s = self.mission_scale_expand
            theta = math.pi / 2.0
            return (c2.copy(), s, theta, "gap2", f"[间隔] 段间悬停 {Gb:.0f}s…", True)

        if tau < 3 * Gb + 3 * T:
            alpha = (tau - (3 * Gb + 2 * T)) / T
            c = c2 + alpha * (c3 - c2)
            s = self.mission_scale_expand + alpha * (self.mission_scale_contract - self.mission_scale_expand)
            theta = math.pi / 2.0
            return (c, s, theta, "seg_move_contract", f"[段3/3] 中心 c2→c3 全维移动 + 收缩 (耗时 {T}s)", False)

        if tau < 4 * Gb + 3 * T:
            s = self.mission_scale_contract
            theta = math.pi / 2.0
            return (c3.copy(), s, theta, "gap3", f"[间隔] 末段悬停 {Gb:.0f}s 后降落…", True)

        return None

    def _advance_mission_center_smooth(self, c_nom: np.ndarray, dt: float) -> np.ndarray:
        """一阶速度饱和：dc/dt 指向 c_nom，‖dc/dt‖ ≤ mission_center_max_speed。"""
        assert self._mission_center_smooth is not None
        v_cap = float(self.mission_center_max_speed)
        c_nom = np.asarray(c_nom, dtype=float).reshape(3)
        if v_cap <= 1e-9:
            self._mission_center_smooth = c_nom.copy()
            return self._mission_center_smooth
        err = c_nom - self._mission_center_smooth
        dist = float(np.linalg.norm(err))
        if dist < 1e-9:
            self._mission_center_smooth = c_nom.copy()
            return self._mission_center_smooth
        u = err / dist
        step_max = v_cap * max(dt, 1e-3)
        if dist <= step_max:
            self._mission_center_smooth = c_nom.copy()
        else:
            self._mission_center_smooth = self._mission_center_smooth + u * step_max
        return self._mission_center_smooth

    def _begin_mission_after_thesis(self, P: np.ndarray, now: float) -> None:
        c0 = np.mean(P, axis=1)
        d1 = self._read_delta_vector("mission_delta1")
        d2 = self._read_delta_vector("mission_delta2")
        d3 = self._read_delta_vector("mission_delta3")
        c1 = c0 + d1
        c2 = c1 + d2
        c3 = c2 + d3
        self.mission_waypoints = [c0.copy(), c1.copy(), c2.copy(), c3.copy()]
        self.mission_t0 = now
        self._mission_center_smooth = c0.copy()
        self.reset_pid()
        self.phase = "mission_moving_center"
        self._last_stage_log_key = ""
        Gb = max(self.mission_between_segment_duration, 0.0)
        self.log_stage_once(
            "mission_start",
            "[切换] 正四面体指标已保持 — 进入移动中心动态："
            f"c0=({c0[0]:.2f},{c0[1]:.2f},{c0[2]:.2f}) → c3=({c3[0]:.2f},{c3[1]:.2f},{c3[2]:.2f})，段间 {Gb:.0f}s；"
            f"参考中心限速 {self.mission_center_max_speed:.3f} m/s，动态段水平 PID 上限 {self.mission_track_max_xy_speed:.2f} m/s",
        )

    def control_loop(self) -> None:
        now = self.now_sec()
        if self.last_ros_time_sec is None:
            dt_loop = self.dt
        else:
            dt_loop = max(now - self.last_ros_time_sec, 1e-3)
        self.last_ros_time_sec = now

        P = self.position_matrix()
        if P is None:
            return

        if len(self.initial_ground) < N_AGENT:
            for n in self.agent_names:
                self.initial_ground[n] = np.array(self.positions[n], dtype=float).copy()

        if self.phase == "wait_poses":
            z_mean = float(np.mean([self.initial_ground[n][2] for n in self.agent_names]))
            self.hover_z = z_mean + self.takeoff_height
            self.phase = "takeoff"
            self.get_logger().info(f"[初始化] 起飞至 z={self.hover_z:.2f} m")

        if self.phase == "takeoff":
            ctrl = np.zeros((DIM, N_AGENT), dtype=float)
            for i in range(N_AGENT):
                e_z = self.hover_z - P[2, i]
                ctrl[2, i] = 1.2 * e_z
            for i, name in enumerate(self.agent_names):
                self.publish_velocity(name, ctrl[:, i])
            if all(abs(self.hover_z - P[2, i]) < self.takeoff_tolerance for i in range(N_AGENT)):
                self.phase = "formation_near"
                self.reset_pid()
                self._last_stage_log_key = ""
                self.log_stage_once(
                    "takeoff_done",
                    "[阶段] 起飞完成 — 全维 PID 收敛至近四面体…",
                )
            return

        if self.phase == "formation_near":
            targets = self.build_near_targets(P)
            err = self.track_targets_pid(targets, dt_loop)
            if max(err.values()) < self.near_formation_tolerance:
                self._near_targets = {k: v.copy() for k, v in targets.items()}
                self._loiter_anchor = np.column_stack([targets[n] for n in self.agent_names])
                self._loiter_t0 = now
                self.phase = "loiter"
                self.reset_pid()
                self._last_stage_log_key = ""
                self.log_stage_once(
                    "near_done",
                    f"[阶段] 近平衡点 — 定点保持 {self.loiter_duration:.0f}s 后切换论文律…",
                )
            return

        if self.phase == "loiter":
            assert self._loiter_anchor is not None
            assert self._loiter_t0 is not None
            t = now - self._loiter_t0
            targets = {self.agent_names[i]: self._loiter_anchor[:, i].copy() for i in range(N_AGENT)}
            self.track_targets_pid(targets, dt_loop)
            if t >= self.loiter_duration:
                Pm = self.position_matrix()
                assert Pm is not None
                xy_c = np.mean(Pm[:2, :], axis=1)
                center = np.array([xy_c[0], xy_c[1], self.hover_z], dtype=float)
                P_des = np.column_stack([center + self.tetra_offsets[n] for n in self.agent_names])
                self._init_thesis_globals(P_des)
                self._estimation = Pm.copy()
                self._thesis_t0 = now
                self._god_view_ok_since = None
                self._last_god_debug_time = None
                self.phase = "thesis_hetero"
                self._open_thesis_data_log(now)
                self._open_trajectory_log(now)
                self._last_stage_log_key = ""
                self.log_stage_once(
                    "thesis_start",
                    "[阶段] 异构量测控制；上帝视角达标并连续保持后进入移动中心任务…",
                )
            return

        if self.phase == "thesis_hetero":
            assert self._P_des is not None
            assert self._edge_length_desired is not None

            L_raw, _ = rigidity_standard(P)
            bad, ej, Lj = edge_spacing_safety_violation(L_raw)
            if bad:
                self._abort_edge_spacing_unsafe(L_raw, ej, Lj)
                return

            self._hetero_line_step(P, dt_loop)
            self._append_thesis_data_row(P, now)
            self._append_trajectory_row(P, now)

            pos_m, edge_m = god_view_tetrahedron_metrics(P, self._P_des, self._edge_length_desired)
            if self.god_view_debug_period > 0.0:
                if (
                    self._last_god_debug_time is None
                    or now - self._last_god_debug_time >= self.god_view_debug_period
                ):
                    self._last_god_debug_time = now
                    self.get_logger().info(
                        f"[上帝视角] max‖p-p*‖={pos_m:.4f} m  max|L-L*|={edge_m:.4f} m "
                        f"(门限 pos<{self.god_view_pos_tol}  edge<{self.god_view_edge_tol})"
                    )

            god_ok = pos_m < self.god_view_pos_tol and edge_m < self.god_view_edge_tol
            if god_ok:
                if self._god_view_ok_since is None:
                    self._god_view_ok_since = now
                elif now - self._god_view_ok_since >= self.god_view_hold_sec:
                    self.log_stage_once(
                        "god_view_ok",
                        f"[阶段] 上帝视角指标已满足并连续保持 {self.god_view_hold_sec:.0f}s — 衔接动态任务",
                    )
                    self.close_thesis_log_if_open()
                    self._begin_mission_after_thesis(P, now)
            else:
                self._god_view_ok_since = None

            return

        if self.phase == "mission_moving_center":
            L_raw, _ = rigidity_standard(P)
            bad, ej, Lj = edge_spacing_safety_violation(L_raw)
            if bad:
                self._abort_edge_spacing_unsafe(L_raw, ej, Lj)
                return

            assert self.mission_t0 is not None and self.mission_waypoints is not None
            tau = now - self.mission_t0
            T = max(self.mission_segment_duration, 1e-3)
            Gb = max(self.mission_between_segment_duration, 0.0)

            nominal = self.get_mission_nominal_c_s_theta(tau, T, Gb)

            if nominal is None:
                self._append_trajectory_row(P, now)
                self.close_trajectory_log_if_open()
                self.phase = "landing"
                self._last_stage_log_key = ""
                self.log_stage_once("landing_plan", "[阶段] 动态任务结束，轨迹 CSV 已关闭；准备同步降落")
                return

            c_nom, s, theta, log_key, log_msg, hold_current = nominal
            assert self._mission_center_smooth is not None
            c_ref = self._advance_mission_center_smooth(c_nom, dt_loop)
            p_des = self._desired_matrix_from_c_s_theta(c_ref, s, theta)

            self.log_stage_once(log_key, log_msg)

            if hold_current:
                target_map = {n: np.array(self.positions[n], dtype=float).copy() for n in self.agent_names}
            else:
                target_map = {name: p_des[:, i].copy() for i, name in enumerate(self.agent_names)}
            self.track_targets_pid(target_map, dt_loop, max_xy_cap=self.mission_track_max_xy_speed)
            self._append_trajectory_row(P, now)
            return

        if self.phase == "aborted":
            return

        if self.phase == "landing":
            ctrl = np.zeros((DIM, N_AGENT), dtype=float)
            for i in range(N_AGENT):
                ctrl[2, i] = -self.landing_speed
            for i, name in enumerate(self.agent_names):
                self.publish_velocity(name, ctrl[:, i])
            if all(P[2, i] < 0.2 for i in range(N_AGENT)):
                self.stop_all()
                self.get_logger().info("[完成] 0415_move：异构收敛 + 移动中心动态 + 降落结束。")
                rclpy.shutdown()
            return


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = ThesisHeteroThenMovingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断")
    finally:
        node.close_thesis_log_if_open()
        node.close_trajectory_log_if_open()
        node.stop_all()
        node.destroy_node()


if __name__ == "__main__":
    main()
