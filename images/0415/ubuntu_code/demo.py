#!/usr/bin/env python3
# 平动demo
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SimpleTranslationNode(Node):
    def __init__(self):
        super().__init__('simple_translation')
        self.get_logger().info("启动简易编队平飞节点 (3架无人机)...")
        
        # 创建3架无人机的速度发布者
        self.pubs = {
            1: self.create_publisher(Twist, '/quadrotor_1/cmd_vel', 10),
            2: self.create_publisher(Twist, '/quadrotor_2/cmd_vel', 10),
            3: self.create_publisher(Twist, '/quadrotor_3/cmd_vel', 10),
        }
        
        # 等待 ROS 2 网络连接
        time.sleep(1.0)

    def publish_velocity(self, vx, vy, vz, az):
        """同时给3架无人机发送相同的速度指令"""
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        msg.angular.z = float(az)
        
        for q_id in self.pubs:
            self.pubs[q_id].publish(msg)

    def execute_phase(self, phase_name, duration, vx, vy, vz, az):
        """执行特定动作阶段，10Hz频率发送指令防止超时掉落"""
        self.get_logger().info(f"▶ 正在执行: {phase_name} (持续 {duration} 秒)")
        
        hz = 10.0
        steps = int(duration * hz)
        sleep_time = 1.0 / hz
        
        for _ in range(steps):
            if not rclpy.ok():
                break
            self.publish_velocity(vx, vy, vz, az)
            time.sleep(sleep_time)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleTranslationNode()

    try:
        # ================= 动作序列编排 =================
        
        # 1. 起飞：Z轴速度 1.5 m/s，持续 3 秒
        node.execute_phase("同步起飞", duration=3.0, vx=0.0, vy=0.0, vz=1.5, az=0.0)
        
        # 2. 悬停稳定：速度归零，持续 1 秒
        node.execute_phase("悬停稳定", duration=1.0, vx=0.0, vy=0.0, vz=0.0, az=0.0)
        
        # 3. 编队平移：X轴正向速度 2.0 m/s，持续 10 秒 (2 m/s * 10 s = 20 m)
        node.execute_phase("向前平飞 20 米", duration=10.0, vx=2.0, vy=0.0, vz=0.0, az=0.0)
        
        # 4. 悬停稳定：到达目标点后刹车，持续 2 秒
        node.execute_phase("到达目标，悬停", duration=2.0, vx=0.0, vy=0.0, vz=0.0, az=0.0)
        
        # 5. 降落：Z轴速度 -1.0 m/s，持续 4 秒
        node.execute_phase("同步降落", duration=4.0, vx=0.0, vy=0.0, vz=-1.0, az=0.0)

        node.get_logger().info("✅ 任务完成！")

    except KeyboardInterrupt:
        node.get_logger().info("被用户中断，发送急停指令...")
        # 紧急悬停
        for _ in range(5):
            node.publish_velocity(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()