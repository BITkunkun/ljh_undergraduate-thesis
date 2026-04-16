#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class AllPoseReader(Node):
    def __init__(self):
        super().__init__('all_pose_reader')
        
        # QoS 配置：必须与 Gazebo 发布端匹配
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE
        
        # 用于存储 4 架无人机的最新坐标
        self.poses = {1: None, 2: None, 3: None, 4: None}
        self.subs = []
        
        # 批量创建 4 架无人机的订阅器
        for i in range(1, 5):
            sub = self.create_subscription(
                TFMessage,
                f'/quadrotor_{i}/world_pose',
                # 使用 lambda 绑定当前的无人机编号
                lambda msg, agent_id=i: self.pose_callback(msg, agent_id),
                qos
            )
            self.subs.append(sub)
            
        # 设置一个定时器，每 0.5 秒刷新一次输出
        self.timer = self.create_timer(0.5, self.print_dashboard)
        self.get_logger().info("✅ 多机坐标监听节点已启动！")

    def pose_callback(self, msg: TFMessage, agent_id: int):
        """收到坐标时，更新对应无人机的字典数据"""
        if msg.transforms:
            x = msg.transforms[0].transform.translation.x
            y = msg.transforms[0].transform.translation.y
            z = msg.transforms[0].transform.translation.z
            self.poses[agent_id] = (x, y, z)

    def print_dashboard(self):
        """格式化输出，看起来像一个仪表盘"""
        # 使用普通的 print 打印，比 get_logger() 少了时间戳前缀，看起来更干净
        print("\n" + "="*40)
        print(f"{'无人机':<10} | {'X (m)':<8} | {'Y (m)':<8} | {'Z (m)':<8}")
        print("-" * 40)
        
        for i in range(1, 5):
            if self.poses[i] is not None:
                x, y, z = self.poses[i]
                print(f"quadrotor_{i} | {x:8.2f} | {y:8.2f} | {z:8.2f}")
            else:
                print(f"quadrotor_{i} | 正在等待数据流...")
        
        print("="*40)

def main(args=None):
    rclpy.init(args=args)
    node = AllPoseReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("监听已停止。")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()