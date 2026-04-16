#!/usr/bin/env python3
"""
毕业设计 Gazebo 复现（全局量测基准版）：全维 PID + 原地编队变换 + 10s 仿真间隔

阶段流程：
1. takeoff：全维 PID 起飞
2. formation_full：全维 PID 收敛至正四面体
3. formation_gap：编队达成后 ROS/仿真时间保持 mission_gap_duration（默认 10s）
4. mission_global_pid：【核心阶段】在进入任务瞬间锁定四机位置均值 c 为编队中心，全程 c 不变
   - 扩张：边长（尺度）由 1 过渡到 2，theta=0，耗时 mission_segment_duration
   - 间隔：当前位置悬停 mission_gap_duration（10s 仿真时间）
   - 旋转：尺度保持 2，theta 由 0 过渡到 π/2（与 MATLAB 参考轨迹 R_z 一致），耗时 mission_segment_duration
   - 间隔：悬停 mission_gap_duration
   - 收缩：尺度由 2 回到 1，theta=π/2，耗时 mission_segment_duration
   - 间隔：悬停 mission_gap_duration
5. landing：同步降落
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist

from reference_trajectory import compose_desired_positions, rot_z
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

DIM = 3
N_AGENT = 4
N_EDGE = 6

# MATLAB 模板顶点 (相对编队中心的坐标)
BASE_POSITION_TEMPLATE = np.array(
    [
        [1.0, 2.0, 1.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 2.0, 2.0, 1.0],
    ],
    dtype=float,
)

class ThesisFormationNode(Node):
    def __init__(self) -> None:
        super().__init__("final_thesis_formation_global_pid")

        # --- 基础控制参数 ---
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("max_speed", 3.0)         # 提速以适应大范围机动
        self.declare_parameter("takeoff_height", 2.0)
        self.declare_parameter("takeoff_tolerance", 0.08)
        self.declare_parameter("use_filtered_transform", True)

        # --- 编队 PID 参数 ---
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
        self.declare_parameter("max_xy_speed", 2.5)      # 提升XY机动能力
        self.declare_parameter("max_z_speed", 1.5)       # 提升Z轴爬升能力

        # --- 任务时间参数 ---
        # 每段移动耗时 20秒 (保证大范围平滑)
        self.declare_parameter("mission_segment_duration", 20.0) 
        # 每次变换前后的停留时间 10秒 (截图专用)
        self.declare_parameter("mission_gap_duration", 10.0)    

        # --- 原地任务：中心由进入任务时的四机位置均值锁定；下列 mission_p* 保留为参数占位，当前逻辑未使用 ---
        self.declare_parameter("mission_p0_x", 0.0)
        self.declare_parameter("mission_p0_y", 0.0)
        self.declare_parameter("mission_p0_z", 2.0)
        self.declare_parameter("mission_p1_x", 20.0)
        self.declare_parameter("mission_p1_y", 10.0)
        self.declare_parameter("mission_p1_z", 8.0)
        self.declare_parameter("mission_p2_x", 40.0)
        self.declare_parameter("mission_p2_y", -10.0)
        self.declare_parameter("mission_p2_z", 5.0)
        self.declare_parameter("mission_p3_x", 60.0)
        self.declare_parameter("mission_p3_y", 0.0)
        self.declare_parameter("mission_p3_z", 10.0)

        # --- 编队缩放尺度 ---
        self.declare_parameter("mission_scale_start", 1.0)
        self.declare_parameter("mission_scale_expand", 2.0)     # 扩张得更明显
        self.declare_parameter("mission_scale_contract", 1.0)
        
        self.declare_parameter("landing_speed", 0.35)
        self.declare_parameter("landing_tolerance", 0.08)

        # 加载所有参数
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

        # PID 状态
        self.integral_error: Dict[str, np.ndarray] = {n: np.zeros(3, dtype=float) for n in self.agent_names}
        self.previous_error: Dict[str, np.ndarray] = {n: np.zeros(3, dtype=float) for n in self.agent_names}
        self.last_ros_time_sec: Optional[float] = None

        self.base_shape = self._build_base_shape_matrix()
        self.tetra_offsets = {n: self.base_shape[:, i].copy() for i, n in enumerate(self.agent_names)}

        self.mission_t0: Optional[float] = None
        self.mission_center: Optional[np.ndarray] = None
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
        self.get_logger().info("[启动] 论文对比组：全维 PID，编队中心固定下的扩张/旋转/收缩")

    def _build_base_shape_matrix(self) -> np.ndarray:
        scale = self.desired_edge_length / math.sqrt(2.0)
        c = np.mean(BASE_POSITION_TEMPLATE, axis=1, keepdims=True)
        return (BASE_POSITION_TEMPLATE - c) * scale

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
        """与 MATLAB ``c + R_z(theta)*(s*base_shape)`` 及 ``reference_trajectory.compose_desired_positions`` 一致。"""
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
        self.log_stage_once("landing_plan", "[阶段] 任务彻底完成，准备同步降落！")

    def now_sec(self) -> float:
        """与 /clock 一致的 ROS 时间（开启 use_sim_time 时为仿真时间）。"""
        return self.get_clock().now().nanoseconds / 1e9

    def get_mission_targets(
        self, tau: float, T: float, G: float
    ) -> Tuple[Optional[np.ndarray], str, str, bool]:
        """
        原地编队变换时间轴（tau 为 ROS/仿真时间，与 /clock 一致）：
        编队中心 c 固定为进入任务时锁定的 ``self.mission_center``（四机位置均值）。

        0         ~ G         : 间隙 0（悬停）
        G         ~ G+T       : 扩张 s: start→expand，theta=0
        G+T       ~ 2G+T      : 间隙 1
        2G+T      ~ 2G+2T     : 旋转 s=expand，theta: 0→π/2
        2G+2T     ~ 3G+2T     : 间隙 2
        3G+2T     ~ 3G+3T     : 收缩 s: expand→contract，theta=π/2
        3G+3T     ~ 4G+3T     : 间隙 3
        最后一项为 True 时表示用实时位置作为目标（悬停段）。
        """
        assert self.mission_center is not None
        c = self.mission_center

        if tau < G:
            s = self.mission_scale_start
            theta = 0.0
            return (
                self._desired_matrix_from_c_s_theta(c, s, theta),
                "gap0",
                f"[间隔] 仿真时间 {G:.0f}s：编队中心固定，当前位置悬停…",
                True,
            )

        if tau < G + T:
            alpha = (tau - G) / T
            s = self.mission_scale_start + alpha * (self.mission_scale_expand - self.mission_scale_start)
            theta = 0.0
            return (
                self._desired_matrix_from_c_s_theta(c, s, theta),
                "seg_expand",
                f"[变换1/3] 原地扩张：边长尺度 {self.mission_scale_start}→{self.mission_scale_expand} (耗时 {T}s)",
                False,
            )

        if tau < 2 * G + T:
            s = self.mission_scale_expand
            theta = 0.0
            return (
                self._desired_matrix_from_c_s_theta(c, s, theta),
                "gap1",
                f"[间隔] 仿真时间 {G:.0f}s：悬停…",
                True,
            )

        if tau < 2 * G + 2 * T:
            alpha = (tau - (2 * G + T)) / T
            s = self.mission_scale_expand
            theta = (math.pi / 2.0) * alpha
            return (
                self._desired_matrix_from_c_s_theta(c, s, theta),
                "seg_rotate",
                f"[变换2/3] 原地绕 Z 轴旋转至 90° (耗时 {T}s)",
                False,
            )

        if tau < 3 * G + 2 * T:
            s = self.mission_scale_expand
            theta = math.pi / 2.0
            return (
                self._desired_matrix_from_c_s_theta(c, s, theta),
                "gap2",
                f"[间隔] 仿真时间 {G:.0f}s：悬停…",
                True,
            )

        if tau < 3 * G + 3 * T:
            alpha = (tau - (3 * G + 2 * T)) / T
            s = self.mission_scale_expand + alpha * (self.mission_scale_contract - self.mission_scale_expand)
            theta = math.pi / 2.0
            return (
                self._desired_matrix_from_c_s_theta(c, s, theta),
                "seg_contract",
                f"[变换3/3] 原地收缩：尺度回到 {self.mission_scale_contract} (耗时 {T}s)",
                False,
            )

        if tau < 4 * G + 3 * T:
            s = self.mission_scale_contract
            theta = math.pi / 2.0
            return (
                self._desired_matrix_from_c_s_theta(c, s, theta),
                "gap3",
                f"[间隔] 仿真时间 {G:.0f}s：悬停后降落…",
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
                self.log_stage_once("takeoff_done", "[阶段] 起飞完成 — 聚集成正四面体编队...")
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
                    f"[阶段] 编队完成 — formation_gap：ROS 时间保持 {self.mission_gap_duration:.0f}s 后进入任务…",
                )
            return

        if self.phase == "formation_gap":
            assert self.formation_gap_t0 is not None
            G_form = max(self.mission_gap_duration, 0.0)
            target_map = {n: np.array(self.positions[n], dtype=float).copy() for n in self.agent_names}
            self.track_targets_pid(target_map, dt_loop)
            if now - self.formation_gap_t0 >= G_form:
                self.mission_center = np.mean(P, axis=1)
                self.mission_t0 = now
                self.phase = "mission_global_pid"
                self._last_stage_log_key = ""
                self.log_stage_once(
                    "mission_start",
                    "[切换] 间隙结束，进入原地编队变换（中心锁定于 "
                    f"{self.mission_center[0]:.2f}, {self.mission_center[1]:.2f}, {self.mission_center[2]:.2f}）…",
                )
            return

        if self.phase == "mission_global_pid":
            assert self.mission_t0 is not None
            tau = now - self.mission_t0
            T = max(self.mission_segment_duration, 1e-3)  # 20秒
            G = max(self.mission_gap_duration, 0.0)       # 10秒

            p_des, log_key, log_msg, hold_current = self.get_mission_targets(tau, T, G)

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
            # 为简化展示，降落直接关闭速度指令让底层飞控自动接管或原地下降
            ctrl = np.zeros((DIM, N_AGENT), dtype=float)
            for i in range(N_AGENT):
                ctrl[2, i] = -self.landing_speed
            for i, name in enumerate(self.agent_names):
                self.publish_velocity(name, ctrl[:, i])
                
            if all(P[2, i] < 0.2 for i in range(N_AGENT)):
                self.stop_all()
                self.get_logger().info("[完成] 仿真测试圆满结束。")
                rclpy.shutdown()
            return

def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = ThesisFormationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断")
    finally:
        node.stop_all()
        node.destroy_node()

if __name__ == "__main__":
    main()