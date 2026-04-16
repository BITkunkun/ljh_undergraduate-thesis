#!/usr/bin/env python3

"""
Gazebo 四机 PID 正四面体编队演示节点。

控制流程：
1. 等待 4 架无人机的 `/world_pose` 绝对位置。
2. 4 架无人机一起起飞到统一高度。
3. 以 MATLAB 仿真中的正四面体相对构型为目标，使用 PID 收敛到对应队形。
4. 编队收敛后维持一段时间。
5. 所有无人机同步开始降落，并在同一结束时刻回到地面参考高度。

说明：
- 编队阶段不锁定固定的全局 x/y 目标点，只约束四机的相对构型。
- 正四面体相对位置直接对应 MATLAB 中的 `position_desired_matrix`。
"""

import itertools
import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


class PIDFormationDemo(Node):
    def __init__(self) -> None:
        super().__init__("demo_0412_pid_tetrahedron")

        # 统一起飞高度、目标边长、PID 参数与降落阶段都做成参数，方便后续调参。
        self.declare_parameter("takeoff_height", 2.0)
        self.declare_parameter("desired_edge_length", math.sqrt(2.0))
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("kp_xy", 0.85)
        self.declare_parameter("ki_xy", 0.02)
        self.declare_parameter("kd_xy", 0.18)
        self.declare_parameter("kp_z", 1.15)
        self.declare_parameter("ki_z", 0.02)
        self.declare_parameter("kd_z", 0.22)
        self.declare_parameter("integral_limit_xy", 2.0)
        self.declare_parameter("integral_limit_z", 1.0)
        self.declare_parameter("max_xy_speed", 1.0)
        self.declare_parameter("max_z_speed", 0.8)
        self.declare_parameter("takeoff_tolerance", 0.08)
        self.declare_parameter("formation_tolerance", 0.12)
        self.declare_parameter("shape_tolerance", 0.10)
        self.declare_parameter("hold_duration", 5.0)
        self.declare_parameter("landing_speed", 0.35)
        self.declare_parameter("landing_tolerance", 0.08)

        self.takeoff_height = float(self.get_parameter("takeoff_height").value)
        self.desired_edge_length = float(
            self.get_parameter("desired_edge_length").value
        )
        self.control_period = float(self.get_parameter("control_period").value)
        self.kp_xy = float(self.get_parameter("kp_xy").value)
        self.ki_xy = float(self.get_parameter("ki_xy").value)
        self.kd_xy = float(self.get_parameter("kd_xy").value)
        self.kp_z = float(self.get_parameter("kp_z").value)
        self.ki_z = float(self.get_parameter("ki_z").value)
        self.kd_z = float(self.get_parameter("kd_z").value)
        self.integral_limit_xy = float(
            self.get_parameter("integral_limit_xy").value
        )
        self.integral_limit_z = float(self.get_parameter("integral_limit_z").value)
        self.max_xy_speed = float(self.get_parameter("max_xy_speed").value)
        self.max_z_speed = float(self.get_parameter("max_z_speed").value)
        self.takeoff_tolerance = float(self.get_parameter("takeoff_tolerance").value)
        self.formation_tolerance = float(
            self.get_parameter("formation_tolerance").value
        )
        self.shape_tolerance = float(self.get_parameter("shape_tolerance").value)
        self.hold_duration = float(self.get_parameter("hold_duration").value)
        self.landing_speed = float(self.get_parameter("landing_speed").value)
        self.landing_tolerance = float(
            self.get_parameter("landing_tolerance").value
        )

        self.agent_names = [f"quadrotor_{idx}" for idx in range(1, 5)]
        self.agent_pairs = list(itertools.combinations(self.agent_names, 2))
        self.positions: Dict[str, Optional[np.ndarray]] = {
            name: None for name in self.agent_names
        }
        self.cmd_publishers: Dict[str, object] = {}
        self.pose_subs = []

        self.initial_positions: Dict[str, np.ndarray] = {}
        self.takeoff_targets: Dict[str, np.ndarray] = {}
        self.landing_start_positions: Dict[str, np.ndarray] = {}
        self.hover_height = 0.0
        self.tetrahedron_offsets = self.build_tetrahedron_offsets()

        self.integral_error: Dict[str, np.ndarray] = {
            name: np.zeros(3, dtype=float) for name in self.agent_names
        }
        self.previous_error: Dict[str, np.ndarray] = {
            name: np.zeros(3, dtype=float) for name in self.agent_names
        }

        self.phase = "bootstrap"
        self.last_control_time: Optional[float] = None
        self.last_status_time = 0.0
        self.hold_start_time: Optional[float] = None
        self.landing_start_time: Optional[float] = None
        self.landing_duration = 0.0

        # Gazebo 发布 /world_pose 时通常使用 BEST_EFFORT + VOLATILE。
        world_pose_qos = QoSProfile(depth=10)
        world_pose_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        world_pose_qos.durability = DurabilityPolicy.VOLATILE

        for name in self.agent_names:
            self.cmd_publishers[name] = self.create_publisher(
                Twist, f"/{name}/cmd_vel", 10
            )
            self.pose_subs.append(
                self.create_subscription(
                    TFMessage,
                    f"/{name}/world_pose",
                    lambda msg, agent_name=name: self.world_pose_callback(
                        msg, agent_name
                    ),
                    world_pose_qos,
                )
            )

        self.timer = self.create_timer(self.control_period, self.control_loop)
        self.get_logger().info(
            "demo_0412 已启动：四机同步起飞，收敛到 MATLAB 对应正四面体，"
            "保持队形后再同步降落。"
        )

    def build_tetrahedron_offsets(self) -> Dict[str, np.ndarray]:
        """根据 MATLAB 期望位置生成以质心为原点的正四面体相对偏移。"""
        desired_matrix = np.array(
            [
                [1.0, 2.0, 1.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 2.0, 2.0, 1.0],
            ],
            dtype=float,
        )
        desired_center = np.mean(desired_matrix, axis=1, keepdims=True)
        scale = self.desired_edge_length / math.sqrt(2.0)
        offset_matrix = (desired_matrix - desired_center) * scale
        return {
            agent_name: offset_matrix[:, index].copy()
            for index, agent_name in enumerate(self.agent_names)
        }

    def world_pose_callback(self, msg: TFMessage, agent_name: str) -> None:
        """接收单架无人机的世界坐标。"""
        position = self.extract_world_position(msg, agent_name)
        if position is not None:
            self.positions[agent_name] = position.copy()

    @staticmethod
    def extract_world_position(msg: TFMessage, agent_name: str) -> Optional[np.ndarray]:
        """从 `/world_pose` 中优先提取无人机本体的世界坐标。"""
        if not msg.transforms:
            return None

        filtered = []
        for transform in msg.transforms:
            child = transform.child_frame_id.strip("/")
            lower_child = child.lower()
            if any(
                token in lower_child
                for token in ("rotor", "propeller", "sensor", "camera", "slot", "optical")
            ):
                continue
            filtered.append(transform)

        for transform in filtered:
            child = transform.child_frame_id.strip("/")
            lower_child = child.lower()
            if agent_name in lower_child or lower_child.endswith("base_link"):
                translation = transform.transform.translation
                return np.array(
                    [translation.x, translation.y, translation.z], dtype=float
                )

        if filtered:
            translation = filtered[0].transform.translation
            return np.array([translation.x, translation.y, translation.z], dtype=float)

        translation = msg.transforms[0].transform.translation
        return np.array([translation.x, translation.y, translation.z], dtype=float)

    def current_position_matrix(self) -> Optional[np.ndarray]:
        """当 4 架无人机的位置都可用时，返回 3x4 位置矩阵。"""
        if any(self.positions[name] is None for name in self.agent_names):
            return None
        return np.column_stack([self.positions[name] for name in self.agent_names])

    def reset_pid_state(self) -> None:
        """阶段切换时清空 PID 内部状态，避免积分项残留。"""
        for name in self.agent_names:
            self.integral_error[name] = np.zeros(3, dtype=float)
            self.previous_error[name] = np.zeros(3, dtype=float)

    def set_phase(self, new_phase: str, message: str) -> None:
        """切换控制阶段并打印关键日志。"""
        if self.phase == new_phase:
            return
        self.phase = new_phase
        self.reset_pid_state()
        self.get_logger().info(message)

    def initialize_takeoff_targets(self) -> None:
        """记录初始位置，并生成四机同步起飞目标。"""
        for name in self.agent_names:
            current = np.array(self.positions[name], dtype=float)
            self.initial_positions[name] = current.copy()

        initial_z_values = [self.initial_positions[name][2] for name in self.agent_names]
        self.hover_height = float(np.mean(initial_z_values) + self.takeoff_height)

        for name in self.agent_names:
            target = self.initial_positions[name].copy()
            target[2] = self.hover_height
            self.takeoff_targets[name] = target

        self.set_phase(
            "takeoff_all",
            "已获取四架无人机初始位置，开始同步起飞到统一高度 "
            f"z={self.hover_height:.2f} m。",
        )

    def build_formation_targets(
        self, position_matrix: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        在当前水平质心附近构造正四面体目标。

        这里不锁定绝对 x/y 位置，只保留队形相对构型；
        但会把四面体中心高度固定在起飞后的 `hover_height`。
        """
        current_xy_center = np.mean(position_matrix[:2, :], axis=1)
        formation_center = np.array(
            [current_xy_center[0], current_xy_center[1], self.hover_height], dtype=float
        )

        target_map = {}
        for name in self.agent_names:
            target_map[name] = formation_center + self.tetrahedron_offsets[name]
        return target_map, formation_center

    def initialize_landing_profile(self, now: float) -> None:
        """
        记录降落起点，并生成同步降落的时间参数。

        降落阶段固定各自当前的 x/y，只让 z 沿统一时间进度回到初始地面高度，
        这样四架无人机可以一起开始降落，并在同一结束时刻落地。
        """
        self.landing_start_time = now
        self.landing_start_positions = {
            name: np.array(self.positions[name], dtype=float) for name in self.agent_names
        }

        max_drop_distance = 0.0
        for name in self.agent_names:
            start_z = self.landing_start_positions[name][2]
            ground_z = self.initial_positions[name][2]
            max_drop_distance = max(max_drop_distance, max(start_z - ground_z, 0.0))

        self.landing_duration = max(
            max_drop_distance / max(self.landing_speed, 1.0e-3), 1.0
        )
        self.set_phase(
            "landing",
            "编队保持完成，开始同步降落，预计 "
            f"{self.landing_duration:.1f} s 回到地面参考高度。",
        )

    def build_landing_targets(
        self, now: float
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """按照统一时间进度生成同步降落目标。"""
        assert self.landing_start_time is not None

        if self.landing_duration <= 1.0e-6:
            progress = 1.0
        else:
            progress = float(
                np.clip((now - self.landing_start_time) / self.landing_duration, 0.0, 1.0)
            )

        target_map = {}
        for name in self.agent_names:
            start_position = self.landing_start_positions[name]
            target = start_position.copy()
            target[2] = (
                (1.0 - progress) * start_position[2]
                + progress * self.initial_positions[name][2]
            )
            target_map[name] = target
        return target_map, progress

    def compute_pid_velocity(
        self,
        agent_name: str,
        target_position: np.ndarray,
        current_position: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """标准 PID：将位置误差直接映射成速度指令。"""
        error = target_position - current_position
        integral = self.integral_error[agent_name] + error * dt
        integral[0:2] = np.clip(
            integral[0:2], -self.integral_limit_xy, self.integral_limit_xy
        )
        integral[2] = float(
            np.clip(integral[2], -self.integral_limit_z, self.integral_limit_z)
        )

        derivative = (error - self.previous_error[agent_name]) / max(dt, 1.0e-3)
        self.integral_error[agent_name] = integral
        self.previous_error[agent_name] = error

        velocity = np.zeros(3, dtype=float)
        velocity[0] = (
            self.kp_xy * error[0]
            + self.ki_xy * integral[0]
            + self.kd_xy * derivative[0]
        )
        velocity[1] = (
            self.kp_xy * error[1]
            + self.ki_xy * integral[1]
            + self.kd_xy * derivative[1]
        )
        velocity[2] = (
            self.kp_z * error[2]
            + self.ki_z * integral[2]
            + self.kd_z * derivative[2]
        )

        xy_speed = np.linalg.norm(velocity[:2])
        if xy_speed > self.max_xy_speed and xy_speed > 1.0e-9:
            velocity[:2] = velocity[:2] * (self.max_xy_speed / xy_speed)
        velocity[2] = float(np.clip(velocity[2], -self.max_z_speed, self.max_z_speed))
        return velocity

    def publish_velocity(self, agent_name: str, velocity: np.ndarray) -> None:
        """向指定无人机发布速度控制指令。"""
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.linear.y = float(velocity[1])
        msg.linear.z = float(velocity[2])
        self.cmd_publishers[agent_name].publish(msg)

    def stop_all(self) -> None:
        """退出前给所有无人机发送零速度。"""
        zero = np.zeros(3, dtype=float)
        for name in self.agent_names:
            self.publish_velocity(name, zero)

    def track_targets(
        self, target_map: Dict[str, np.ndarray], dt: float
    ) -> Dict[str, float]:
        """跟踪当前阶段目标点，并返回每架无人机的误差范数。"""
        errors = {}
        for name in self.agent_names:
            current = np.array(self.positions[name], dtype=float)
            target = target_map[name]
            velocity = self.compute_pid_velocity(name, target, current, dt)
            self.publish_velocity(name, velocity)
            errors[name] = float(np.linalg.norm(target - current))
        return errors

    def compute_max_edge_error(self) -> float:
        """比较当前 6 条边与期望正四面体边长的误差。"""
        max_error = 0.0
        for first_name, second_name in self.agent_pairs:
            current_distance = float(
                np.linalg.norm(self.positions[first_name] - self.positions[second_name])
            )
            max_error = max(
                max_error, abs(current_distance - self.desired_edge_length)
            )
        return max_error

    def log_status(self, message: str, interval: float = 1.5) -> None:
        """限频打印状态信息，避免日志刷屏。"""
        now = time.time()
        if now - self.last_status_time >= interval:
            self.last_status_time = now
            self.get_logger().info(message)

    def control_loop(self) -> None:
        """主控制循环：同步起飞 -> 正四面体收敛 -> 保持 -> 同步降落。"""
        now = time.time()
        position_matrix = self.current_position_matrix()
        if position_matrix is None:
            missing_agents = [
                name for name in self.agent_names if self.positions[name] is None
            ]
            self.log_status(
                "正在等待四架无人机的 /world_pose 数据，"
                f"当前缺少: {', '.join(missing_agents)}"
            )
            return

        if self.last_control_time is None:
            dt = self.control_period
        else:
            dt = max(now - self.last_control_time, 1.0e-3)
        self.last_control_time = now

        if self.phase == "bootstrap":
            self.initialize_takeoff_targets()

        if self.phase == "takeoff_all":
            errors = self.track_targets(self.takeoff_targets, dt)
            max_position_error = max(errors.values())
            max_altitude_error = max(
                abs(self.takeoff_targets[name][2] - self.positions[name][2])
                for name in self.agent_names
            )
            self.log_status(
                "阶段1：同步起飞中，"
                f"最大位置误差={max_position_error:.3f} m，"
                f"最大高度误差={max_altitude_error:.3f} m"
            )
            if max_altitude_error < self.takeoff_tolerance:
                self.set_phase(
                    "formation",
                    "起飞完成，开始 PID 正四面体编队变换。"
                    f" 目标边长={self.desired_edge_length:.3f} m。",
                )
            return

        if self.phase == "formation":
            target_map, _ = self.build_formation_targets(position_matrix)
            errors = self.track_targets(target_map, dt)
            max_position_error = max(errors.values())
            max_edge_error = self.compute_max_edge_error()
            self.log_status(
                "阶段2：正四面体收敛中，"
                f"最大相对位置误差={max_position_error:.3f} m，"
                f"最大边长误差={max_edge_error:.3f} m"
            )
            if (
                max_position_error < self.formation_tolerance
                and max_edge_error < self.shape_tolerance
            ):
                self.hold_start_time = now
                self.set_phase(
                    "hold",
                    "正四面体编队已收敛，开始保持队形 "
                    f"{self.hold_duration:.1f} s。",
                )
            return

        if self.phase == "hold":
            target_map, _ = self.build_formation_targets(position_matrix)
            errors = self.track_targets(target_map, dt)
            max_position_error = max(errors.values())
            max_edge_error = self.compute_max_edge_error()
            elapsed = 0.0 if self.hold_start_time is None else now - self.hold_start_time
            remaining = max(self.hold_duration - elapsed, 0.0)
            self.log_status(
                "阶段3：保持正四面体队形，"
                f"剩余{remaining:.1f} s，"
                f"最大相对位置误差={max_position_error:.3f} m，"
                f"最大边长误差={max_edge_error:.3f} m"
            )
            if self.hold_start_time is not None and elapsed >= self.hold_duration:
                self.initialize_landing_profile(now)
            return

        if self.phase == "landing":
            target_map, progress = self.build_landing_targets(now)
            errors = self.track_targets(target_map, dt)
            max_position_error = max(errors.values())
            max_altitude_error = max(
                abs(target_map[name][2] - self.positions[name][2])
                for name in self.agent_names
            )
            self.log_status(
                "阶段4：同步降落中，"
                f"进度={progress * 100.0:.0f}% ，"
                f"最大位置误差={max_position_error:.3f} m，"
                f"最大高度误差={max_altitude_error:.3f} m"
            )
            if progress >= 1.0 and max_altitude_error < self.landing_tolerance:
                self.set_phase("completed", "所有无人机已完成同步降落。")
                self.stop_all()
            return

        if self.phase == "completed":
            self.stop_all()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PIDFormationDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("收到用户中断，准备发送零速度指令。")
    finally:
        node.stop_all()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
