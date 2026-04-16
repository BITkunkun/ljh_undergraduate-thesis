#!/usr/bin/env python3
"""
毕业设计 Gazebo：局部收敛验证（全维 PID → 近四面体 → 定点保持 → 异构量测论文律）

需求对应关系（与 ``复现论文+code/code_ljh`` 中 ``main.m`` / ``dynamic.m`` 一致）：
1. **平地起飞**：与 ``final.py`` 相同，全维高度控制至 ``hover_z``。
2. **全维 PID 收敛到“近四面体”**：以编队中心 + 标准模板尺度得到理想四面体顶点，再叠加
   ``main.m`` 中 ``position_initial_matrix = position_desired_matrix - 0.25 + 0.5*rand``
   的扰动（本节点用 ``numpy.random.seed(1)`` 固定，与 MATLAB ``rng(1)`` 同分布族）。
3. **定点保持（原“盘旋”语义）**：到达近四面体平衡点后，四机在**固定世界坐标 XYZ** 上由全维 PID
   悬停不动，持续 ``loiter_duration`` 秒（默认 5s，**ROS / 仿真时钟**，与 ``now_sec()`` 一致），再切换控制律。
4. **双线检测（论文阶段）**：
   - **线1 — 异构量测闭环**：各机仅通过 ``output_positon_matrix`` 所定义的**部分量测**与分布式观测器/
     控制器驱动编队（与 ``main.m`` 一致：机 1 仅 Z、机 2 无位置、机 3 为 XY、机 4 全位置；边长项进入量测向量）。
     发送至仿真的速度为 ``controller`` 输出；估计器按 ``dynamic.m`` 积分。量测由仿真真值经该异构矩阵构造（无附加噪声）。
   - **线2 — 上帝视角**：在控制回路之外，用 **全部真实位置** 与 **全部边长真值** 相对期望四面体做指标，
     **仅用于判定**何时视为“正四面体收敛完成”，满足则经保持时间后进入 ``landing``；**不参与**控制律计算。
5. **降落**：由线2判定触发，与线1控制相互独立。编队指标（上帝视角）**首次**满足后，须**连续**保持
   ``god_view_hold_sec``（默认 **5 s**，ROS 仿真时钟）才作出降落决策；与 ``main.m`` 门控语义一致。
6. **安全停止**：论文阶段若任一边真实间距 **不在** ``[EDGE_LENGTH_MIN, EDGE_LENGTH_MAX]``（与 ``main.m`` 一致），
   判定为控制发散、机体过近/碰撞风险或等效异常，**立即停止**发布速度并结束节点。

刚性矩阵采用标准 6×12 行堆叠形式（每边一行，与 ``controller.m`` 中 ``dgamma * R(j,:)`` 梯度一致），
避免直接翻译 ``rigidity.m`` 中 ``blkdiag`` 维数与 6 边不匹配的问题。

**数据记录**：``thesis_phase_data.txt`` 仅在论文律阶段写入；自切换控制律时刻起，至上帝视角编队指标
**首次**满足后再 ``thesis_data_log_after_formation_sec``（默认 5 s）止，列仅为 ``t_phase_ros_s`` 与各机
控制误差（实际位置减期望）与估计误差（实际位置减估计）。Ignition 世界常见物理步长 0.004 s（250 Hz），本节点按
``control_period``（默认 0.02 s，即 50 Hz）定时采样位姿并写盘。
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

DIM = 3
N_AGENT = 4
N_EDGE = 6

# 与 ``final.py`` / MATLAB 模板一致 (列为智能体)
BASE_POSITION_TEMPLATE = np.array(
    [
        [1.0, 2.0, 1.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 2.0, 2.0, 1.0],
    ],
    dtype=float,
)

# main.m 有向图关联矩阵 H (4 x 6)
H_INCIDENCE = np.array(
    [
        [1, 1, 1, 0, 0, 0],
        [-1, 0, 0, 1, 1, 0],
        [0, -1, 0, -1, 0, 1],
        [0, 0, -1, 0, -1, -1],
    ],
    dtype=float,
)

# 全连通邻接（无自环）
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
EDGE_LENGTH_MAX = 2.0


def edge_spacing_safety_violation(L: np.ndarray) -> Tuple[bool, int, float]:
    """
    任一边长不在 ``[EDGE_LENGTH_MIN, EDGE_LENGTH_MAX]`` 内则视为异常（与 ``main.m`` 势场安全边一致）：
    过短可能碰撞/失控贴靠，过长可能发散或等效通讯/拓扑异常。
    返回 ``(是否越界, 边索引 j, 该边长度 L[j])``。
    """
    for j in range(N_EDGE):
        lj = float(L[j])
        if lj < EDGE_LENGTH_MIN or lj > EDGE_LENGTH_MAX:
            return True, j, lj
    return False, -1, 0.0


def rigidity_standard(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    边长 (6,) 与刚性矩阵 (6, 12)。第 j 行对应第 j 条边，列为 [agent0_xyz, agent1_xyz, ...]。
    """
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
    """与 ``dgamma.m`` 一致（对第 j 条边标量求导）；分母加小量避免除零。"""
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
    """main.m: diag([0 0 1, 0 0 0, 1 1 0, 1 1 1])，形状 (12, 12)。"""
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
    """
    线1 用量测向量：前半为 0.5*L_j^2（六边），后半为 ``output_pos @ vec(P)``（异构位置分量）。
    与 ``dynamic.m`` 中 ``measure_current`` 构造一致（仿真真值经 ``output_pos`` 掩码）。
    """
    p_vec = P.reshape(-1, order="F")
    meas_pos = output_pos @ p_vec
    return np.concatenate([0.5 * edge_lengths**2, meas_pos])


def god_view_tetrahedron_metrics(
    P: np.ndarray,
    P_des: np.ndarray,
    el_des: np.ndarray,
) -> Tuple[float, float]:
    """
    线2：上帝视角指标（全信息、不进入观测器）。
    返回 ``(max_i ‖p_i - p*_i‖, max_j |L_j - L*_j|)``。
    """
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


class ThesisLocalNode(Node):
    def __init__(self) -> None:
        super().__init__("final_thesis_local_hetero_0415")

        self.declare_parameter("control_period", 0.02)
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

        # MATLAB rng(1) 下扰动幅度
        self.declare_parameter("matlab_perturb_low", -0.25)
        self.declare_parameter("matlab_perturb_high", 0.25)

        # 切换论文律前：在当前 XYZ 定点保持（ROS/仿真时间，秒），默认 5s
        self.declare_parameter("loiter_duration", 5.0)

        self.declare_parameter("thesis_max_speed", 2.0)

        # 线2 上帝视角：全位置 + 全边长判定正四面体完成（不参与控制）
        self.declare_parameter("god_view_pos_tol", 0.12)
        self.declare_parameter("god_view_edge_tol", 0.10)
        # 编队指标满足后的“再保持”时间（秒，ROS 仿真时钟），默认 5 s，之后才触发降落决策
        self.declare_parameter("god_view_hold_sec", 5.0)
        # 上帝视角指标**首次**进入门限后，再继续记录数据的时长（秒），记录结束后关闭文件（降落指令开始前）
        self.declare_parameter("thesis_data_log_after_formation_sec", 5.0)
        self.declare_parameter("god_view_debug_period", 0.0)

        self.declare_parameter("landing_speed", 0.35)
        self.declare_parameter("landing_tolerance", 0.08)

        _default_thesis_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_phase_data.txt")
        self.declare_parameter("thesis_data_log_path", _default_thesis_log)

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
        self.thesis_data_log_after_formation_sec = float(
            self.get_parameter("thesis_data_log_after_formation_sec").value
        )
        self.god_view_debug_period = float(self.get_parameter("god_view_debug_period").value)

        self.landing_speed = float(self.get_parameter("landing_speed").value)
        self.landing_tolerance = float(self.get_parameter("landing_tolerance").value)

        self.thesis_data_log_path = str(self.get_parameter("thesis_data_log_path").value)

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

        # 与 MATLAB 一致的固定扰动 (rng(1))
        rng = np.random.default_rng(1)
        span = self.matlab_perturb_high - self.matlab_perturb_low
        self._delta_near = self.matlab_perturb_low + span * rng.random((DIM, N_AGENT))

        self._near_targets: Optional[Dict[str, np.ndarray]] = None
        self._loiter_anchor: Optional[np.ndarray] = None  # (3, 4)
        self._loiter_t0: Optional[float] = None

        self._P_des: Optional[np.ndarray] = None
        self._edge_length_desired: Optional[np.ndarray] = None
        self._R_des: Optional[np.ndarray] = None
        self._output_matrix: Optional[np.ndarray] = None
        self._output_pos = build_output_position_matrix()
        self._estimation: Optional[np.ndarray] = None

        self._thesis_t0: Optional[float] = None
        self._formation_first_ok_time: Optional[float] = None
        self._thesis_log_closed_by_window: bool = False
        self._god_view_ok_since: Optional[float] = None
        self._last_god_debug_time: Optional[float] = None

        self._last_stage_log_key = ""
        self._edge_safety_abort_done = False

        self._thesis_log_fp: Optional[TextIO] = None
        self._thesis_log_writer: Optional[Any] = None

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
            "[启动] 0415：… → 论文阶段双线（线1异构闭环 / 线2上帝视角判收敛）→ 降落"
        )

    def _build_base_shape_matrix(self) -> np.ndarray:
        scale = self.desired_edge_length / math.sqrt(2.0)
        c = np.mean(BASE_POSITION_TEMPLATE, axis=1, keepdims=True)
        return (BASE_POSITION_TEMPLATE - c) * scale

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
        cap = self.max_speed if self.phase not in ("thesis_hetero",) else self.thesis_max_speed
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
        cols = ["t_phase_ros_s"]
        for i in range(1, N_AGENT + 1):
            for ax in ("x", "y", "z"):
                cols.append(f"ctrl_{i}_{ax}")
        for i in range(1, N_AGENT + 1):
            for ax in ("x", "y", "z"):
                cols.append(f"est_{i}_{ax}")
        return cols

    def _open_thesis_data_log(self, ros_abs_t0: float) -> None:
        """论文阶段数据：自切换控制律起，至编队指标首次满足后再 thesis_data_log_after_formation_sec 秒止；仅写控制/估计误差。"""
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
            f"[数据记录] 论文律起 → 编队首次达标后 {self.thesis_data_log_after_formation_sec:.1f}s 止，写入 "
            f"{self.thesis_data_log_path}（仅 ctrl/est 误差；t_phase_ros_s 相对切换时刻；t0={ros_abs_t0:.6f}s）"
        )

    def _append_thesis_data_row(self, P: np.ndarray, now: float) -> None:
        """在观测器/估计一步更新之后记录：实际、期望、控制误差(实际-期望)、估计误差(实际-估计)。"""
        if self._thesis_log_writer is None or self._thesis_t0 is None:
            return
        if self._P_des is None or self._estimation is None:
            return
        t_rel = float(now - self._thesis_t0)
        ctrl = (P - self._P_des).reshape(-1, order="F")
        est = (P - self._estimation).reshape(-1, order="F")
        row: List[float] = [t_rel]
        row.extend(float(x) for x in np.concatenate([ctrl, est]))
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
        self, agent_name: str, target: np.ndarray, current: np.ndarray, dt: float
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

        xy_sp = np.linalg.norm(v[:2])
        if xy_sp > self.max_xy_speed and xy_sp > 1e-9:
            v[:2] = v[:2] * (self.max_xy_speed / xy_sp)
        v[2] = float(np.clip(v[2], -self.max_z_speed, self.max_z_speed))
        return v

    def track_targets_pid(self, target_map: Dict[str, np.ndarray], dt: float) -> Dict[str, float]:
        errors = {}
        for name in self.agent_names:
            cur = np.array(self.positions[name], dtype=float)
            vel = self.compute_pid_velocity(name, target_map[name], cur, dt)
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

    def _abort_edge_spacing_unsafe(self, L: np.ndarray, edge_idx: int, Lj: float) -> None:
        """边长越出安全区间：紧急停桨逻辑并退出。"""
        if self._edge_safety_abort_done:
            return
        self._edge_safety_abort_done = True
        self.get_logger().error(
            f"[安全停止] 边 {edge_idx} 间距 L={Lj:.4f} m 超出安全范围 "
            f"[{EDGE_LENGTH_MIN}, {EDGE_LENGTH_MAX}] m（与 main.m edge_length_min/max 一致）。"
            " 可能原因：控制发散、机体碰撞或过近、通讯/拓扑等效异常。已停止 cmd_vel 并退出。"
        )
        self.get_logger().error(f"当前六边长 (m): {np.array2string(L, precision=4)}")
        self.phase = "aborted"
        self.close_thesis_log_if_open()
        self.stop_all()
        rclpy.shutdown()

    def _hetero_line_step(self, P: np.ndarray, dt_loop: float) -> None:
        """
        线1：异构量测闭环。仅使用 ``build_heterogeneous_measurement`` 构造的量测与局部估计，
        不读取“上帝视角”收敛判据。
        """
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
                    "[阶段] 起飞完成 — 全维 PID 收敛至 MATLAB 初值附近的近四面体…",
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
                    f"[阶段] 已到达近平衡点 — 定点保持 XYZ {self.loiter_duration:.0f}s（ROS 时间）后切换论文律…",
                )
            return

        if self.phase == "loiter":
            assert self._loiter_anchor is not None
            assert self._loiter_t0 is not None
            t = now - self._loiter_t0
            # 目标为进入本阶段时锁定的固定点，全维 PID 保持 xyz 不变（不做小范围圆周机动）
            targets = {self.agent_names[i]: self._loiter_anchor[:, i].copy() for i in range(N_AGENT)}
            self.track_targets_pid(targets, dt_loop)
            if t >= self.loiter_duration:
                Pm = self.position_matrix()
                assert Pm is not None
                xy_c = np.mean(Pm[:2, :], axis=1)
                center = np.array([xy_c[0], xy_c[1], self.hover_z], dtype=float)
                P_des = np.column_stack([center + self.tetra_offsets[n] for n in self.agent_names])
                self._init_thesis_globals(P_des)

                # 无量测噪声时，估计初值与真实位置一致（无异构初值随机偏差）
                self._estimation = Pm.copy()

                self._thesis_t0 = now
                self._formation_first_ok_time = None
                self._thesis_log_closed_by_window = False
                self._god_view_ok_since = None
                self._last_god_debug_time = None
                self.phase = "thesis_hetero"
                self._open_thesis_data_log(now)
                self._last_stage_log_key = ""
                self.log_stage_once(
                    "thesis_start",
                    "[阶段] 线1 异构量测控制启动；线2 上帝视角独立监测全状态，达标后降落…",
                )
            return

        if self.phase == "thesis_hetero":
            assert self._P_des is not None
            assert self._edge_length_desired is not None

            # 安全停止：真实边长（未做 dgamma 数值夹紧）须在 [EDGE_LENGTH_MIN, EDGE_LENGTH_MAX] 内
            L_raw, _ = rigidity_standard(P)
            bad, ej, Lj = edge_spacing_safety_violation(L_raw)
            if bad:
                self._abort_edge_spacing_unsafe(L_raw, ej, Lj)
                return

            # 线1：部分量测驱动的分布式律（与线2 解耦）
            self._hetero_line_step(P, dt_loop)

            # 线2：上帝视角 — 全机真实位置 + 六边真实边长，仅用于 landing 门控
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
            if god_ok and self._formation_first_ok_time is None:
                self._formation_first_ok_time = now

            log_deadline: Optional[float] = None
            if self._formation_first_ok_time is not None:
                log_deadline = self._formation_first_ok_time + self.thesis_data_log_after_formation_sec

            if not self._thesis_log_closed_by_window:
                if log_deadline is None or now <= log_deadline:
                    self._append_thesis_data_row(P, now)
                else:
                    self._thesis_log_closed_by_window = True
                    self.close_thesis_log_if_open()
                    self.get_logger().info(
                        f"[数据记录] 编队指标首次满足后已记录 {self.thesis_data_log_after_formation_sec:.1f}s，"
                        "已关闭数据文件（降落指令开始前）"
                    )

            if god_ok:
                if self._god_view_ok_since is None:
                    self._god_view_ok_since = now
                elif now - self._god_view_ok_since >= self.god_view_hold_sec:
                    self.log_stage_once(
                        "god_view_ok",
                        f"[阶段] 线2 上帝视角：正四面体指标已满足（pos≤{self.god_view_pos_tol} m，"
                        f"|ΔL|≤{self.god_view_edge_tol} m）并连续保持 {self.god_view_hold_sec:.0f}s（ROS）— 降落决策生效",
                    )
                    # 本周期已在 thesis_hetero 内写完最后一行数据；此处关闭文件再进入降落，不包含降落阶段
                    self.close_thesis_log_if_open()
                    self.phase = "landing"
                    self._last_stage_log_key = ""
            else:
                self._god_view_ok_since = None

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
                self.get_logger().info("[完成] 0415 局部收敛验证流程结束。")
                rclpy.shutdown()
            return


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = ThesisLocalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断")
    finally:
        node.close_thesis_log_if_open()
        node.stop_all()
        node.destroy_node()


if __name__ == "__main__":
    main()
