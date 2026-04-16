#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import math

class OpenLoopRotateNode(Node):
    def __init__(self):
        super().__init__('open_loop_rotate')
        self.get_logger().info("启动开环编队控制：平移 + 180度动态换位...")
        
        # 创建3架无人机的速度发布者
        self.pubs = {
            1: self.create_publisher(Twist, '/quadrotor_1/cmd_vel', 10),
            2: self.create_publisher(Twist, '/quadrotor_2/cmd_vel', 10),
            3: self.create_publisher(Twist, '/quadrotor_3/cmd_vel', 10),
        }
        time.sleep(1.0) # 等待节点连接

    def send_cmd(self, q_id, vx, vy, vz):
        """发送单台无人机的速度指令"""
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        msg.angular.z = 0.0  # 保持机头朝向不变，防止局部坐标系错乱
        self.pubs[q_id].publish(msg)

    def execute_takeoff(self, duration=3.0):
        self.get_logger().info("⬆️ 同步起飞中...")
        steps = int(duration * 10)
        for _ in range(steps):
            if not rclpy.ok(): break
            for i in range(1, 4):
                self.send_cmd(i, 0.0, 0.0, 1.5)
            time.sleep(0.1)

    def execute_land(self, duration=4.0):
        self.get_logger().info("⬇️ 同步降落中...")
        steps = int(duration * 10)
        for _ in range(steps):
            if not rclpy.ok(): break
            for i in range(1, 4):
                self.send_cmd(i, 0.0, 0.0, -1.0)
            time.sleep(0.1)

    def execute_hover(self, duration=2.0):
        self.get_logger().info("⏸️ 悬停稳定...")
        steps = int(duration * 10)
        for _ in range(steps):
            if not rclpy.ok(): break
            for i in range(1, 4):
                self.send_cmd(i, 0.0, 0.0, 0.0)
            time.sleep(0.1)

    def execute_maneuver(self, duration=10.0):
        self.get_logger().info(f"🔄 执行动态机动：平移20米 + 180度大回旋换位 ({duration}秒)...")
        
        hz = 10.0
        steps = int(duration * hz)
        dt = 1.0 / hz
        
        # 运动学参数
        V_cx = 2.0         # 整个编队向前平移的速度 (2.0 m/s * 10s = 20米)
        omega = math.pi / duration  # 角速度：10秒转完 180度 (pi 弧度)
        R = 2.0            # 1号机和3号机距离中心点(2号机)的半径是 2 米

        for step in range(steps):
            if not rclpy.ok(): break
            
            t = step * dt
            
            # --- 2号机 (中心点) ---
            # 只做纯平移
            self.send_cmd(2, V_cx, 0.0, 0.0)
            
            # --- 1号机 (初始在中心点前方) ---
            # 初始相位 0，速度叠加：平移速度 + 切向速度在 X, Y 轴的投影
            theta_1 = 0.0 + omega * t
            vx_1 = V_cx - R * omega * math.sin(theta_1)
            vy_1 = 0.0  + R * omega * math.cos(theta_1)
            self.send_cmd(1, vx_1, vy_1, 0.0)
            
            # --- 3号机 (初始在中心点后方) ---
            # 初始相位 pi (180度)，速度叠加机制同上
            theta_3 = math.pi + omega * t
            vx_3 = V_cx - R * omega * math.sin(theta_3)
            vy_3 = 0.0  + R * omega * math.cos(theta_3)
            self.send_cmd(3, vx_3, vy_3, 0.0)
            
            time.sleep(dt)

def main(args=None):
    rclpy.init(args=args)
    node = OpenLoopRotateNode()

    try:
        # 1. 垂直起飞
        node.execute_takeoff(duration=3.0)
        node.execute_hover(duration=1.0)

        # 2. 核心机动：一边往前飞，一边交叉换位
        node.execute_maneuver(duration=10.0)
        node.execute_hover(duration=2.0)

        # 3. 垂直降落
        node.execute_land(duration=4.0)

        node.get_logger().info("✅ 编队机动任务完美收工！")

    except KeyboardInterrupt:
        node.get_logger().info("检测到中断，下发急停指令...")
        node.execute_hover(duration=0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()