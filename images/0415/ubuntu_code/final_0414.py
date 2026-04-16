#!/usr/bin/env python3
"""
毕业设计 Gazebo：全维 PID + 编队中心分三段移动 + 段间 5s（ROS/仿真时间）延迟

与 final.py 的差异：
- 进入任务时记录起点中心 c0（四机位置均值），并按参数累加三段位移得到 c1,c2,c3。
- 时间轴：gap(5s) → 移动段1（c0→c1 + 扩张）→ gap(5s) → 移动段2（c1→c2 + 旋转）
         → gap(5s) → 移动段3（c2→c3 + 收缩）→ gap(5s) → 降落。
- 每段移动内中心点沿直线匀速插值；尺度/转角与 final.py 原地任务相同。
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist

from reference_trajectory import compose_desired_positions
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

DIM = 3
N_AGENT = 4

BASE_POSITION_TEMPLATE = np.array(
    [
        [1.0, 2.0, 1.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 2.0, 2.0, 1.0],
    ],
    dtype=float,
)


class ThesisMovingCenterNode(Node):
    def __init__(self) -> None:
        super().__init__("final_thesis_moving_center_0414")

        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("max_speed", 3.0)
        self.declare_parameter("takeoff_height", 2.0)
        self.declare_parameter("takeoff_tolerance", 0.08)
        self.declare_parameter("use_filtered_transform", True)

        self.declare_parameter("desired_edge_length", math.sqrt(2.0))
        self.declare_parameter("formation_tolerance", 0.12)
        self.declare_parameter("shape_tolerance", 0.10)

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

        self.declare_parameter("mission_segment_duration", 20.0)
        # 段与段之间纯悬停（ROS/仿真时钟），默认 5s
        self.declare_parameter("mission_between_segment_duration", 5.0)
        self.declare_parameter("mission_gap_duration", 10.0)

        self.declare_parameter("mission_scale_start", 1.0)
        self.declare_parameter("mission_scale_expand", 2.0)
        self.declare_parameter("mission_scale_contract", 1.0)

        # 三段中心点位移（全维，单位 m），在任务开始时从当前中心累加
        self.declare_parameter("mission_delta1_x", 3)
        self.declare_parameter("mission_delta1_y", -1.5)
        self.declare_parameter("mission_delta1_z", 0.0)
        self.declare_parameter("mission_delta2_x", 3)
        self.declare_parameter("mission_delta2_y", 1.5)
        self.declare_parameter("mission_delta2_z", 0.0)
        self.declare_parameter("mission_delta3_x", 3)
        self.declare_parameter("mission_delta3_y", -1.5)
        self.declare_parameter("mission_delta3_z", 0.0)

        self.declare_parameter("landing_speed", 0.35)
        self.declare_parameter("landing_tolerance", 0.08)

        self.dt = float(self.get_parameter("control_period").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.takeoff_height = float(self.get_parameter("takeoff_height").value)
        self.takeoff_tolerance = float(self.get_parameter("takeoff_tolerance").value)
        self.use_filtered_transform = bool(self.get_parameter("use_filtered_transform").value)

        self.desired_edge_length = float(self.get_parameter("desired_edge_length").value)
        self.formation_tolerance = float(self.get_parameter("formation_tolerance").value)
        self.shape_tolerance = float(self.get_parameter("shape_tolerance").value)

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

        self.mission_segment_duration = float(self.get_parameter("mission_segment_duration").value)
        self.mission_between_segment_duration = float(
            self.get_parameter("mission_between_segment_duration").value
        )
        self.mission_gap_duration = float(self.get_parameter("mission_gap_duration").value)

        self.mission_scale_start = float(self.get_parameter("mission_scale_start").value)
        self.mission_scale_expand = float(self.get_parameter("mission_scale_expand").value)
        self.mission_scale_contract = float(self.get_parameter("mission_scale_contract").value)
        self.landing_speed = float(self.get_parameter("landing_speed").value)
        self.landing_tolerance = float(self.get_parameter("landing_tolerance").value)

        self.agent_names = [f"quadrotor_{i}" for i in range(1, N_AGENT + 1)]
        self.agent_pairs = list(itertools.combinations(self.agent_names, 2))
        self.positions: Dict[str, Optional[np.ndarray]] = {n: None for n in self.agent_names}
        self.initial_ground: Dict[str, np.ndarray] = {}
        self.hover_z = 0.0
        self.phase = "wait_poses"

        self.integral_error: Dict[str, np.ndarray] = {n: np.zeros(3, dtype=float) for n in self.agent_names}
        self.previous_error: Dict[str, np.ndarray] = {n: np.zeros(3, dtype=float) for n in self.agent_names}
        self.last_ros_time_sec: Optional[float] = None

        self.base_shape = self._build_base_shape_matrix()
        self.tetra_offsets = {n: self.base_shape[:, i].copy() for i, n in enumerate(self.agent_names)}

        self.mission_t0: Optional[float] = None
        self.mission_waypoints: Optional[List[np.ndarray]] = None
        self.formation_gap_t0: Optional[float] = None
        self._last_stage_log_key: str = ""

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE

        self.pubs = {}
        for name in self.agent_names:
            self.pubs[name] = self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            self.create_subscription(TFMessage, f"/{name}/world_pose", lambda msg, n=name: self.pose_callback(msg, n), qos)

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(
            "[启动] 全维 PID：中心分三段移动 + 段间 %.1fs，移动中扩张/旋转/收缩"
            % self.mission_between_segment_duration
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
        p = self.extract_position(msg, agent_name)
        if p is not None:
            self.positions[agent_name] = p

    def extract_position(self, msg: TFMessage, agent_name: str) -> Optional[np.ndarray]:
        if not msg.transforms:
            return None
        t = msg.transforms[0].transform.translation
        return np.array([t.x, t.y, t.z], dtype=float)

    def position_matrix(self) -> Optional[np.ndarray]:
        if any(self.positions[n] is None for n in self.agent_names):
            return None
        return np.column_stack([self.positions[n] for n in self.agent_names])

    def publish_velocity(self, name: str, v: np.ndarray) -> None:
        sp = np.linalg.norm(v)
        if sp > self.max_speed and sp > 1e-9:
            v = v * (self.max_speed / sp)
        m = Twist()
        m.linear.x = float(v[0])
        m.linear.y = float(v[1])
        m.linear.z = float(v[2])
        self.pubs[name].publish(m)

    def stop_all(self) -> None:
        z = np.zeros(3, dtype=float)
        for n in self.agent_names:
            self.publish_velocity(n, z)

    def reset_pid(self) -> None:
        for n in self.agent_names:
            self.integral_error[n] = np.zeros(3, dtype=float)
            self.previous_error[n] = np.zeros(3, dtype=float)

    def _desired_matrix_from_c_s_theta(self, c: np.ndarray, s: float, theta: float) -> np.ndarray:
        return compose_desired_positions(c, s, theta, self.base_shape)

    def build_formation_targets(self, position_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        xy_c = np.mean(position_matrix[:2, :], axis=1)
        center = np.array([xy_c[0], xy_c[1], self.hover_z], dtype=float)
        return {n: center + self.tetra_offsets[n] for n in self.agent_names}

    def compute_pid_velocity(self, agent_name: str, target: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
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

    def begin_landing(self) -> None:
        self.phase = "landing"
        self._last_stage_log_key = ""
        self.log_stage_once("landing_plan", "[阶段] 任务结束，准备同步降落")

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def get_mission_targets_moving(
        self, tau: float, T: float, Gb: float
    ) -> Tuple[Optional[np.ndarray], str, str, bool]:
        """
        tau: 自 mission_t0 起的 ROS/仿真时间。
        T: 每段机动时长；Gb: 段间悬停时长（默认 5s）。

        时间轴：
        [0, Gb)                     gap0 悬停
        [Gb, Gb+T)                  c0→c1 + 扩张
        [Gb+T, 2Gb+T)               gap1
        [2Gb+T, 2Gb+2T)             c1→c2 + 旋转
        [2Gb+2T, 3Gb+2T)            gap2
        [3Gb+2T, 3Gb+3T)            c2→c3 + 收缩
        [3Gb+3T, 4Gb+3T)            gap3
        """
        assert self.mission_waypoints is not None
        w = self.mission_waypoints
        c0, c1, c2, c3 = w[0], w[1], w[2], w[3]

        if tau < Gb:
            s = self.mission_scale_start
            theta = 0.0
            return (
                self._desired_matrix_from_c_s_theta(c0, s, theta),
                "gap0",
                f"[间隔] 段前悬停 {Gb:.0f}s（ROS/仿真时间）…",
                True,
            )

        if tau < Gb + T:
            alpha = (tau - Gb) / T
            c = c0 + alpha * (c1 - c0)
            s = self.mission_scale_start + alpha * (self.mission_scale_expand - self.mission_scale_start)
            theta = 0.0
            return (
                self._desired_matrix_from_c_s_theta(c, s, theta),
                "seg_move_expand",
                f"[段1/3] 中心 c0→c1 全维移动 + 扩张 (耗时 {T}s)",
                False,
            )

        if tau < 2 * Gb + T:
            s = self.mission_scale_expand
            theta = 0.0
            return (
                self._desired_matrix_from_c_s_theta(c1, s, theta),
                "gap1",
                f"[间隔] 段间悬停 {Gb:.0f}s…",
                True,
            )

        if tau < 2 * Gb + 2 * T:
            alpha = (tau - (2 * Gb + T)) / T
            c = c1 + alpha * (c2 - c1)
            s = self.mission_scale_expand
            theta = (math.pi / 2.0) * alpha
            return (
                self._desired_matrix_from_c_s_theta(c, s, theta),
                "seg_move_rotate",
                f"[段2/3] 中心 c1→c2 全维移动 + 绕 Z 旋转至 90° (耗时 {T}s)",
                False,
            )

        if tau < 3 * Gb + 2 * T:
            s = self.mission_scale_expand
            theta = math.pi / 2.0
            return (
                self._desired_matrix_from_c_s_theta(c2, s, theta),
                "gap2",
                f"[间隔] 段间悬停 {Gb:.0f}s…",
                True,
            )

        if tau < 3 * Gb + 3 * T:
            alpha = (tau - (3 * Gb + 2 * T)) / T
            c = c2 + alpha * (c3 - c2)
            s = self.mission_scale_expand + alpha * (self.mission_scale_contract - self.mission_scale_expand)
            theta = math.pi / 2.0
            return (
                self._desired_matrix_from_c_s_theta(c, s, theta),
                "seg_move_contract",
                f"[段3/3] 中心 c2→c3 全维移动 + 收缩 (耗时 {T}s)",
                False,
            )

        if tau < 4 * Gb + 3 * T:
            s = self.mission_scale_contract
            theta = math.pi / 2.0
            return (
                self._desired_matrix_from_c_s_theta(c3, s, theta),
                "gap3",
                f"[间隔] 末段悬停 {Gb:.0f}s 后降落…",
                True,
            )

        return None, "end", "end", False

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
                self.phase = "formation_full"
                self.reset_pid()
                self._last_stage_log_key = ""
                self.log_stage_once("takeoff_done", "[阶段] 起飞完成 — 聚集成正四面体编队…")
            return

        if self.phase == "formation_full":
            targets = self.build_formation_targets(P)
            err = self.track_targets_pid(targets, dt_loop)
            if max(err.values()) < self.formation_tolerance:
                self.phase = "formation_gap"
                self.formation_gap_t0 = now
                self.reset_pid()
                self._last_stage_log_key = ""
                self.log_stage_once(
                    "form_done",
                    f"[阶段] 编队完成 — formation_gap：保持 {self.mission_gap_duration:.0f}s 后进入移动中心任务…",
                )
            return

        if self.phase == "formation_gap":
            assert self.formation_gap_t0 is not None
            G_form = max(self.mission_gap_duration, 0.0)
            target_map = {n: np.array(self.positions[n], dtype=float).copy() for n in self.agent_names}
            self.track_targets_pid(target_map, dt_loop)
            if now - self.formation_gap_t0 >= G_form:
                c0 = np.mean(P, axis=1)
                d1 = self._read_delta_vector("mission_delta1")
                d2 = self._read_delta_vector("mission_delta2")
                d3 = self._read_delta_vector("mission_delta3")
                c1 = c0 + d1
                c2 = c1 + d2
                c3 = c2 + d3
                self.mission_waypoints = [c0.copy(), c1.copy(), c2.copy(), c3.copy()]
                self.mission_t0 = now
                self.phase = "mission_moving_center"
                self._last_stage_log_key = ""
                self.log_stage_once(
                    "mission_start",
                    "[切换] 进入移动中心任务：c0=(%.2f,%.2f,%.2f) → c3=(%.2f,%.2f,%.2f)，段间 %.1fs"
                    % (c0[0], c0[1], c0[2], c3[0], c3[1], c3[2], self.mission_between_segment_duration),
                )
            return

        if self.phase == "mission_moving_center":
            assert self.mission_t0 is not None and self.mission_waypoints is not None
            tau = now - self.mission_t0
            T = max(self.mission_segment_duration, 1e-3)
            Gb = max(self.mission_between_segment_duration, 0.0)

            p_des, log_key, log_msg, hold_current = self.get_mission_targets_moving(tau, T, Gb)

            if p_des is None:
                self.begin_landing()
                return

            self.log_stage_once(log_key, log_msg)

            if hold_current:
                target_map = {n: np.array(self.positions[n], dtype=float).copy() for n in self.agent_names}
            else:
                target_map = {name: p_des[:, i].copy() for i, name in enumerate(self.agent_names)}
            self.track_targets_pid(target_map, dt_loop)
            return

        if self.phase == "landing":
            ctrl = np.zeros((DIM, N_AGENT), dtype=float)
            for i in range(N_AGENT):
                ctrl[2, i] = -self.landing_speed
            for i, name in enumerate(self.agent_names):
                self.publish_velocity(name, ctrl[:, i])

            if all(P[2, i] < 0.2 for i in range(N_AGENT)):
                self.stop_all()
                self.get_logger().info("[完成] 仿真测试结束。")
                rclpy.shutdown()
            return


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = ThesisMovingCenterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断")
    finally:
        node.stop_all()
        node.destroy_node()


if __name__ == "__main__":
    main()
