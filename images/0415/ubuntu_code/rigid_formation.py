#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import time

class BasicPositionController(Node):
    def __init__(self):
        super().__init__('basic_closed_loop')
        self.get_logger().info("启动基础闭环位置控制器 (全局 /tf 监听版)...")

        self.num_agents = 3
        # 初始时将坐标设为 None，直到收到数据
        self.current_poses = {1: None, 2: None, 3: None} 
        
        # 目标坐标：等边三角形 (边长2米)，整体向前(X轴正向)平移约20米
        center_x, center_y, target_z = -1482.0, 0.0, 6.0
        self.target_poses = {
            1: np.array([center_x + 1.155, center_y + 0.0, target_z]),
            2: np.array([center_x - 0.577, center_y + 1.0, target_z]),
            3: np.array([center_x - 0.577, center_y - 1.0, target_z])
        }
        
        self.Kp = 0.8        
        self.max_vel = 2.0   
        self.start_time = time.time()

        # --- 1. 创建速度发布者 ---
        self.pubs = {}
        for i in range(1, self.num_agents + 1):
            self.pubs[i] = self.create_publisher(Twist, f'/quadrotor_{i}/cmd_vel', 10)
            
        # --- 2. 统一订阅全局的 /tf 话题 ---
        self.tf_sub = self.create_subscription(
            TFMessage, 
            '/tf', 
            self.tf_callback, 
            qos_profile_sensor_data # 兼容 Gazebo 的 Best Effort
        )

        self.timer = self.create_timer(0.05, self.control_loop)

    def tf_callback(self, msg):
        """遍历 /tf 树，揪出无人机本体的全局绝对坐标"""
        for t in msg.transforms:
            cid = t.child_frame_id
            
            # 在 /tf 中，父节点通常是 'earth', 'world' 或 'map'，子节点是 'quadrotor_X' 或 'quadrotor_X/base_link'
            # 我们过滤掉带有 'rotor' 和 'sensor' 的内部零件数据，只抓取无人机本体坐标
            if 'quadrotor_1' in cid and 'rotor' not in cid and 'sensor' not in cid:
                self.current_poses[1] = np.array([
                    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
                ])
            elif 'quadrotor_2' in cid and 'rotor' not in cid and 'sensor' not in cid:
                self.current_poses[2] = np.array([
                    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
                ])
            elif 'quadrotor_3' in cid and 'rotor' not in cid and 'sensor' not in cid:
                self.current_poses[3] = np.array([
                    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
                ])

    def control_loop(self):
        # 检查是否成功抓取到了 3 架飞机的全局坐标
        if any(pos is None for pos in self.current_poses.values()):
            self.get_logger().info("正在 /tf 话题中搜寻无人机的全局绝对坐标...", throttle_duration_sec=2.0)
            return

        elapsed = time.time() - self.start_time

        # ---- 控制逻辑 ----
        if elapsed < 35.0:
            phase = "飞向目标位置"
            
            for i in range(1, self.num_agents + 1):
                current = self.current_poses[i]
                target = self.target_poses[i]
                
                # 计算位置误差
                error = target - current
                
                # P 控制律：速度 = Kp * 误差
                vel = self.Kp * error
                
                # 速度限幅保护
                norm_vel = np.linalg.norm(vel)
                if norm_vel > self.max_vel:
                    vel = self.max_vel * (vel / norm_vel)
                
                msg = Twist()
                msg.linear.x = float(vel[0])
                msg.linear.y = float(vel[1])
                msg.linear.z = float(vel[2])
                self.pubs[i].publish(msg)

            # 打印当前进度（距离）
            if int(elapsed * 20) % 40 == 0:
                dist1 = np.linalg.norm(self.target_poses[1] - self.current_poses[1])
                dist2 = np.linalg.norm(self.target_poses[2] - self.current_poses[2])
                dist3 = np.linalg.norm(self.target_poses[3] - self.current_poses[3])
                self.get_logger().info(f"[{elapsed:.1f}s] {phase} | 距目标距离 - 机1: {dist1:.2f}m, 机2: {dist2:.2f}m, 机3: {dist3:.2f}m")

        elif elapsed < 40.0:
            for i in range(1, self.num_agents + 1):
                msg = Twist()
                msg.linear.z = -1.0 # 以 1m/s 的速度下降
                self.pubs[i].publish(msg)
            
            if int(elapsed * 20) % 40 == 0:
                self.get_logger().info("⬇️ 降落中...")

        else:
            self.get_logger().info("✅ 任务完成，关闭控制器！")
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = BasicPositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("控制器已停止。")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()