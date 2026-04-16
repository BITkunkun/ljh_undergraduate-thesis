#!/usr/bin/env python3

"""
在 Gazebo 中复现 4 机正四面体编队收敛控制。

控制流程：
1. 4 架无人机都从地面初始高度起飞。
2. `quadrotor_4` 先单独上升到离地 2 m。
3. `quadrotor_1`、`quadrotor_2`、`quadrotor_3` 再上升到离地 1 m。
4. 最后切换到四面体编队收敛控制，只约束相对形状，不锁定固定绝对目标点。

说明：
- 四面体收敛阶段仍然保留平移自由度，只要求 6 条边收敛到目标边长。
- 额外加入局部空间软约束，避免整个编队漂出起飞区域太远。
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


class GazeboTetrahedronController(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_tetrahedron_controller")

        default_config = (
            "/home/ubuntu20/mbzirc_ws/install/share/mbzirc_ign/config/coast/config.yaml"
        )
        self.declare_parameter("config_path", default_config)
        self.declare_parameter("desired_edge_length", math.sqrt(2.0))
        self.declare_parameter("base_takeoff_height", 1.0)
        self.declare_parameter("quad4_takeoff_height", 2.0)
        self.declare_parameter("takeoff_gain", 1.0)
        self.declare_parameter("takeoff_max_speed", 0.8)
        self.declare_parameter("takeoff_tolerance", 0.08)
        self.declare_parameter("shape_gain", 1.0)
        self.declare_parameter("workspace_gain", 0.35)
        self.declare_parameter("workspace_xy_half_range", 4.0)
        self.declare_parameter("workspace_z_half_range", 2.0)
        self.declare_parameter("workspace_margin", 0.3)
        self.declare_parameter("max_speed", 1.0)
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("edge_length_min", 0.5)
        self.declare_parameter("edge_length_max", 2.0)
        self.declare_parameter("edge_tolerance", 0.08)

        self.config_path = self.get_parameter("config_path").value
        self.desired_edge_length = float(
            self.get_parameter("desired_edge_length").value
        )
        self.base_takeoff_height = float(
            self.get_parameter("base_takeoff_height").value
        )
        self.quad4_takeoff_height = float(
            self.get_parameter("quad4_takeoff_height").value
        )
        self.takeoff_gain = float(self.get_parameter("takeoff_gain").value)
        self.takeoff_max_speed = float(self.get_parameter("takeoff_max_speed").value)
        self.takeoff_tolerance = float(self.get_parameter("takeoff_tolerance").value)
        self.shape_gain = float(self.get_parameter("shape_gain").value)
        self.workspace_gain = float(self.get_parameter("workspace_gain").value)
        self.workspace_xy_half_range = float(
            self.get_parameter("workspace_xy_half_range").value
        )
        self.workspace_z_half_range = float(
            self.get_parameter("workspace_z_half_range").value
        )
        self.workspace_margin = float(self.get_parameter("workspace_margin").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.control_period = float(self.get_parameter("control_period").value)
        self.edge_length_min = float(self.get_parameter("edge_length_min").value)
        self.edge_length_max = float(self.get_parameter("edge_length_max").value)
        self.edge_tolerance = float(self.get_parameter("edge_tolerance").value)

        self.agent_names, self.spawn_positions = self.load_active_quadrotors(
            self.config_path
        )
        self.num_agents = len(self.agent_names)
        if self.num_agents < 4:
            raise RuntimeError(
                "当前四面体复现需要 4 架无人机，请在 config.yaml 中启用 quadrotor_1~4。"
            )
        if self.num_agents > 4:
            self.get_logger().warn(
                "论文与 MATLAB 代码对应的是 4 机四面体，这里只使用前 4 架无人机。"
            )
            self.agent_names = self.agent_names[:4]
            self.spawn_positions = self.spawn_positions[:4]
            self.num_agents = 4

        self.edges, self.incidence = self.build_complete_graph(self.num_agents)
        self.goal_edge_lengths = np.full(
            len(self.edges), self.desired_edge_length, dtype=float
        )
        self.publishers: Dict[str, object] = {}
        self.subscriptions = []
        self.positions: Dict[str, Optional[np.ndarray]] = {
            name: None for name in self.agent_names
        }
        self.reference_altitudes: Dict[str, float] = {}
        self.takeoff_targets: Dict[str, float] = {}
        self.workspace_center: Optional[np.ndarray] = None
        self.formation_initialized = False
        self.phase = "bootstrap"
        self.has_logged_converged = False
        self.start_time = time.time()

        world_pose_qos = QoSProfile(depth=1)
        world_pose_qos.reliability = ReliabilityPolicy.RELIABLE
        world_pose_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        for name in self.agent_names:
            self.publishers[name] = self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            self.subscriptions.append(
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
            f"已加载 {self.num_agents} 架无人机: {', '.join(self.agent_names)}"
        )
        self.get_logger().info(
            "控制阶段：4号机先起飞到离地 2m，1/2/3 再起飞到离地 1m，"
            "随后进入四面体收敛。"
        )

    @staticmethod
    def load_active_quadrotors(
        config_path: str,
    ) -> Tuple[List[str], np.ndarray]:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"未找到配置文件: {config_path}")

        with config_file.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or []

        agent_entries: List[Tuple[str, np.ndarray]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_name = item.get("model_name", "")
            if not model_name.startswith("quadrotor_"):
                continue
            xyz = item.get("position", {}).get("xyz", [0.0, 0.0, 0.0])
            if len(xyz) != 3:
                xyz = [0.0, 0.0, 0.0]
            agent_entries.append((model_name, np.array(xyz, dtype=float)))

        agent_entries.sort(key=lambda pair: int(pair[0].split("_")[-1]))
        if not agent_entries:
            raise RuntimeError("config.yaml 中没有找到启用的 quadrotor。")

        names = [item[0] for item in agent_entries]
        positions = np.vstack([item[1] for item in agent_entries])
        return names, positions

    @staticmethod
    def build_complete_graph(num_agents: int) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        edges: List[Tuple[int, int]] = []
        edge_count = num_agents * (num_agents - 1) // 2
        incidence = np.zeros((num_agents, edge_count), dtype=float)

        column = 0
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                edges.append((i, j))
                incidence[i, column] = 1.0
                incidence[j, column] = -1.0
                column += 1

        return edges, incidence

    def world_pose_callback(self, msg: TFMessage, agent_name: str) -> None:
        position = self.extract_world_position(msg, agent_name)
        if position is not None:
            self.positions[agent_name] = position

    @staticmethod
    def extract_world_position(msg: TFMessage, agent_name: str) -> Optional[np.ndarray]:
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
            if agent_name in child or child.endswith("base_link"):
                translation = transform.transform.translation
                return np.array(
                    [translation.x, translation.y, translation.z], dtype=float
                )

        if filtered:
            translation = filtered[0].transform.translation
            return np.array([translation.x, translation.y, translation.z], dtype=float)

        return None

    def current_position_matrix(self) -> Optional[np.ndarray]:
        if any(self.positions[name] is None for name in self.agent_names):
            return None
        return np.column_stack([self.positions[name] for name in self.agent_names])

    def initialize_takeoff_targets(self, position_matrix: np.ndarray) -> None:
        for agent_index, agent_name in enumerate(self.agent_names):
            current_z = float(position_matrix[2, agent_index])
            self.reference_altitudes[agent_name] = current_z
            climb_height = (
                self.quad4_takeoff_height
                if agent_name == "quadrotor_4"
                else self.base_takeoff_height
            )
            self.takeoff_targets[agent_name] = current_z + climb_height

        self.phase = "takeoff_quad4"
        self.get_logger().info(
            "已记录地面参考高度，先让 quadrotor_4 起飞到离地 2m。"
        )

    def initialize_formation(self, position_matrix: np.ndarray) -> None:
        self.workspace_center = np.mean(position_matrix, axis=1)
        self.formation_initialized = True
        self.has_logged_converged = False

        assert self.workspace_center is not None
        self.get_logger().info(
            "四面体收敛阶段已初始化，当前质心为 "
            f"({self.workspace_center[0]:.2f}, "
            f"{self.workspace_center[1]:.2f}, "
            f"{self.workspace_center[2]:.2f})，"
            f"目标边长为 {self.desired_edge_length:.3f} m。"
        )

    def rigidity(self, position_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        edge_lengths = np.zeros(len(self.edges), dtype=float)
        rigidity_matrix = np.zeros((len(self.edges), 3 * self.num_agents), dtype=float)

        for edge_index, (i, j) in enumerate(self.edges):
            diff = position_matrix[:, i] - position_matrix[:, j]
            edge_lengths[edge_index] = np.linalg.norm(diff)
            rigidity_matrix[edge_index, 3 * i : 3 * i + 3] = diff
            rigidity_matrix[edge_index, 3 * j : 3 * j + 3] = -diff

        return edge_lengths, rigidity_matrix

    def dgamma(self, current_length: float, desired_length: float) -> float:
        epsilon = 1.0e-3
        elc = float(
            np.clip(
                current_length,
                self.edge_length_min + epsilon,
                self.edge_length_max - epsilon,
            )
        )
        eld = float(
            np.clip(
                desired_length,
                self.edge_length_min + epsilon,
                self.edge_length_max - epsilon,
            )
        )
        elm1 = self.edge_length_min
        elm2 = self.edge_length_max

        denom_1 = max((elm2 - elc) * (elc - elm1), epsilon)
        denom_2 = max((elm2 - elc) * (eld - elm1), epsilon)

        return 0.001 * (
            2.0 * (elc - eld)
            - (elm1 + elm2 - 2.0 * elc) / denom_1
            + (elm1 + elm2 - 2.0 * eld) / denom_2
        )

    def clamp_velocity(self, velocity: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(velocity)
        if speed <= self.max_speed or speed < 1.0e-9:
            return velocity
        return self.max_speed * velocity / speed

    def altitude_velocity(self, current_z: float, target_z: float) -> float:
        vz = self.takeoff_gain * (target_z - current_z)
        return float(np.clip(vz, -self.takeoff_max_speed, self.takeoff_max_speed))

    def altitude_reached(self, current_z: float, target_z: float) -> bool:
        return abs(current_z - target_z) <= self.takeoff_tolerance

    def run_takeoff_phase(self, position_matrix: np.ndarray) -> bool:
        if self.phase == "takeoff_quad4":
            for agent_index, agent_name in enumerate(self.agent_names):
                current_z = float(position_matrix[2, agent_index])
                target_z = (
                    self.takeoff_targets[agent_name]
                    if agent_name == "quadrotor_4"
                    else self.reference_altitudes[agent_name]
                )
                vz = self.altitude_velocity(current_z, target_z)
                self.publish_velocity(agent_name, np.array([0.0, 0.0, vz], dtype=float))

            quad4_index = self.agent_names.index("quadrotor_4")
            quad4_z = float(position_matrix[2, quad4_index])
            if self.altitude_reached(quad4_z, self.takeoff_targets["quadrotor_4"]):
                self.phase = "takeoff_base"
                self.get_logger().info(
                    "quadrotor_4 已到达离地 2m，开始让 quadrotor_1~3 起飞到离地 1m。"
                )
            return True

        if self.phase == "takeoff_base":
            all_reached = True
            for agent_index, agent_name in enumerate(self.agent_names):
                current_z = float(position_matrix[2, agent_index])
                target_z = self.takeoff_targets[agent_name]
                vz = self.altitude_velocity(current_z, target_z)
                self.publish_velocity(agent_name, np.array([0.0, 0.0, vz], dtype=float))
                all_reached = all_reached and self.altitude_reached(current_z, target_z)

            if all_reached:
                self.phase = "formation"
                self.initialize_formation(position_matrix)
                self.get_logger().info("所有无人机已完成分阶段起飞，开始四面体收敛。")
            return True

        return False

    def workspace_correction(self, current_position: np.ndarray) -> np.ndarray:
        assert self.workspace_center is not None

        limits = np.array(
            [
                self.workspace_xy_half_range,
                self.workspace_xy_half_range,
                self.workspace_z_half_range,
            ],
            dtype=float,
        )
        safe_limits = np.maximum(limits - self.workspace_margin, 0.05)
        delta = current_position - self.workspace_center

        correction = np.zeros(3, dtype=float)
        for axis in range(3):
            overflow = abs(delta[axis]) - safe_limits[axis]
            if overflow > 0.0:
                correction[axis] = -self.workspace_gain * math.copysign(
                    overflow, delta[axis]
                )
        return correction

    def max_workspace_overflow(self, position_matrix: np.ndarray) -> float:
        assert self.workspace_center is not None

        limits = np.array(
            [
                self.workspace_xy_half_range,
                self.workspace_xy_half_range,
                self.workspace_z_half_range,
            ],
            dtype=float,
        )
        deltas = np.abs(position_matrix.T - self.workspace_center)
        overflow = np.maximum(deltas - limits, 0.0)
        return float(np.max(overflow))

    def publish_velocity(self, agent_name: str, velocity: np.ndarray) -> None:
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.linear.y = float(velocity[1])
        msg.linear.z = float(velocity[2])
        self.publishers[agent_name].publish(msg)

    def stop_all(self) -> None:
        zero = np.zeros(3, dtype=float)
        for agent_name in self.agent_names:
            self.publish_velocity(agent_name, zero)

    def control_loop(self) -> None:
        position_matrix = self.current_position_matrix()
        if position_matrix is None:
            self.get_logger().info(
                "等待所有无人机的 /world_pose 数据...", throttle_duration_sec=2.0
            )
            return

        if self.phase == "bootstrap":
            self.initialize_takeoff_targets(position_matrix)

        if self.run_takeoff_phase(position_matrix):
            return

        if not self.formation_initialized:
            self.initialize_formation(position_matrix)

        current_edge_lengths, rigidity_matrix = self.rigidity(position_matrix)

        max_command_speed = 0.0
        for agent_index, agent_name in enumerate(self.agent_names):
            current_position = position_matrix[:, agent_index]

            control_velocity = np.zeros(3, dtype=float)
            for edge_index, (source, target) in enumerate(self.edges):
                if agent_index not in (source, target):
                    continue
                dg_value = self.dgamma(
                    current_edge_lengths[edge_index],
                    self.goal_edge_lengths[edge_index],
                )
                gradient = rigidity_matrix[
                    edge_index, 3 * agent_index : 3 * agent_index + 3
                ]
                control_velocity -= self.shape_gain * dg_value * gradient

            control_velocity += self.workspace_correction(current_position)
            control_velocity = self.clamp_velocity(control_velocity)
            self.publish_velocity(agent_name, control_velocity)
            max_command_speed = max(max_command_speed, np.linalg.norm(control_velocity))

        max_edge_error = float(
            np.max(np.abs(current_edge_lengths - self.goal_edge_lengths))
        )
        formation_center = np.mean(position_matrix, axis=1)
        max_workspace_overflow = self.max_workspace_overflow(position_matrix)

        if (
            not self.has_logged_converged
            and max_edge_error < self.edge_tolerance
            and max_command_speed < 0.08
        ):
            elapsed = time.time() - self.start_time
            self.has_logged_converged = True
            self.get_logger().info(
                f"编队基本收敛，用时 {elapsed:.2f}s，"
                f"最大边长误差 {max_edge_error:.3f} m，"
                f"最大控制速度 {max_command_speed:.3f} m/s。"
            )

        self.get_logger().info(
            f"center=({formation_center[0]:.2f}, {formation_center[1]:.2f}, "
            f"{formation_center[2]:.2f}), "
            f"max_edge_err={max_edge_error:.3f} m, "
            f"max_cmd={max_command_speed:.3f} m/s, "
            f"workspace_overflow={max_workspace_overflow:.3f} m",
            throttle_duration_sec=1.5,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GazeboTetrahedronController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("收到中断信号，发送零速度并退出。")
    finally:
        node.stop_all()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
