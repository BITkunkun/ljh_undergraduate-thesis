#!/usr/bin/env python3
# 旋转demo
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SwarmController(Node):
    def __init__(self):
        super().__init__('swarm_controller')
        self.get_logger().info("初始化多无人机编队控制节点...")
        
        # 创建三个无人机的 cmd_vel 发布者
        self.pubs = {
            1: self.create_publisher(Twist, '/quadrotor_1/cmd_vel', 10),
            2: self.create_publisher(Twist, '/quadrotor_2/cmd_vel', 10),
            3: self.create_publisher(Twist, '/quadrotor_3/cmd_vel', 10),
        }
        
        # 等待发布者与仿真环境建立连接
        time.sleep(1.0) 

    def send_cmd(self, q_id, vx, vy, vz, az):
        """发送单台无人机的速度指令"""
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        msg.angular.z = float(az)
        self.pubs[q_id].publish(msg)

    def run_phase(self, phase_name, duration, cmds):
        """
        执行具体的编队动作阶段
        :param phase_name: 动作名称
        :param duration: 持续时间 (秒)
        :param cmds: 包含各无人机速度的字典 {id: (vx, vy, vz, az)}
        """
        self.get_logger().info(f"▶ 正在执行: {phase_name} ({duration} 秒)")
        steps = int(duration * 10)  # 10Hz 的发布频率
        
        for _ in range(steps):
            if not rclpy.ok():
                break
            for q_id, cmd in cmds.items():
                self.send_cmd(q_id, cmd[0], cmd[1], cmd[2], cmd[3])
            time.sleep(0.1)  # 100ms 延时

    def stop_all(self):
        """悬停/停止所有无人机"""
        self.run_phase("悬停", 2.0, {
            1: (0.0, 0.0, 0.0, 0.0),
            2: (0.0, 0.0, 0.0, 0.0),
            3: (0.0, 0.0, 0.0, 0.0)
        })

def main(args=None):
    rclpy.init(args=args)
    node = SwarmController()

    try:
        # ---------------------------------------------------------
        # 动作编排：指令格式为 (vx, vy, vz, az)
        # ---------------------------------------------------------
        
        # 1. 起飞 (Z轴正向速度)
        node.run_phase("1. 编队起飞", 5.0, {
            1: (0.0, 0.0, 1.0, 0.0),
            2: (0.0, 0.0, 1.0, 0.0),
            3: (0.0, 0.0, 1.0, 0.0)
        })
        node.stop_all()

        # 2. 编队扩张 (1号向前，2号向左后，3号向右后)
        node.run_phase("2. 编队扩张", 3.0, {
            1: ( 0.5,  0.0, 0.0, 0.0),
            2: (-0.5,  0.5, 0.0, 0.0),
            3: (-0.5, -0.5, 0.0, 0.0)
        })
        node.stop_all()

        # 3. 编队旋转 (集体偏航旋转)
        node.run_phase("3. 原地旋转", 4.0, {
            1: (0.0, 0.0, 0.0, 1.0),
            2: (0.0, 0.0, 0.0, 1.0),
            3: (0.0, 0.0, 0.0, 1.0)
        })
        node.stop_all()

        # 4. 编队收缩 (反转扩张的速度方向)
        node.run_phase("4. 编队收缩", 3.0, {
            1: (-0.5,  0.0, 0.0, 0.0),
            2: ( 0.5, -0.5, 0.0, 0.0),
            3: ( 0.5,  0.5, 0.0, 0.0)
        })
        node.stop_all()

        # 5. 降落 (Z轴负向速度)
        node.run_phase("5. 编队降落", 5.0, {
            1: (0.0, 0.0, -0.8, 0.0),
            2: (0.0, 0.0, -0.8, 0.0),
            3: (0.0, 0.0, -0.8, 0.0)
        })

        node.get_logger().info("✅ 任务完成，断开连接。")

    except KeyboardInterrupt:
        node.get_logger().info("检测到中断，正在停止所有无人机...")
        node.stop_all()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()