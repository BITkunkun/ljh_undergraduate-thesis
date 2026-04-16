import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
import math

class MultiUAVTest(Node):
    def __init__(self):
        super().__init__('multi_uav_test_node')
        
        # 定义 5 架无人机的名称 (根据你的 yaml 配置)
        self.uavs = ['quadrotor_1', 'quadrotor_2', 'quadrotor_3', 'quadrotor_4', 'quadrotor_5']
        
        self.pubs = {}
        self.subs = {}
        self.poses = {uav: None for uav in self.uavs}
        
        # 设定编队的中心目标位置 (假设原起点是 -1500, 我们让它往前飞到 -1480, 高度 10 米)
        self.target_x = -1480.0
        self.target_y = 0.0
        self.target_z = 10.0
        
        # 状态机：0 = 强制起飞, 1 = 飞向目标点
        self.state = 0
        self.timer_count = 0
        
        for uav in self.uavs:
            # 1. 创建速度发布者
            self.pubs[uav] = self.create_publisher(Twist, f'/{uav}/cmd_vel', 10)
            
            # 2. 创建位置订阅者 
            # (注: 根据 MBZIRC Wiki，话题名为 pose。通常其类型为 PoseStamped)
            self.subs[uav] = self.create_subscription(
                PoseStamped,
                f'/{uav}/pose',
                lambda msg, name=uav: self.pose_callback(msg, name),
                10
            )
            
        # 以 10Hz 的频率运行控制循环
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('测试节点已启动！指令：全机强制起飞...')

    def pose_callback(self, msg, uav_name):
        """
        回调函数：实时更新每架无人机的当前坐标
        """
        self.poses[uav_name] = msg.pose

    def control_loop(self):
        self.timer_count += 1
        
        # --- 阶段 0：强制起飞 (持续 5 秒) ---
        if self.state == 0:
            if self.timer_count < 50:  # 0.1s * 50 = 5秒
                for uav in self.uavs:
                    msg = Twist()
                    msg.linear.z = 1.5  # 以 1.5 m/s 的速度爬升
                    self.pubs[uav].publish(msg)
            else:
                self.state = 1
                self.get_logger().info('起飞完成！切换至闭环位置控制，正在飞向目标点...')
                
        # --- 阶段 1：飞向目标点 (比例控制 P-Controller) ---
        elif self.state == 1:
            for i, uav in enumerate(self.uavs):
                current_pose = self.poses[uav]
                msg = Twist()
                
                # 如果还没收到位置数据，发送零速度悬停等待
                if current_pose is None:
                    self.pubs[uav].publish(msg)
                    continue
                
                # 为每架飞机分配目标偏移量，防止相撞（Y轴每架相隔 2 米）
                # 索引 0,1,2,3,4 对应的 Y 偏移为 -4, -2, 0, 2, 4
                y_offset = (i - 2) * 2.0 
                
                target_x = self.target_x
                target_y = self.target_y + y_offset
                target_z = self.target_z
                
                # 计算各轴误差 (目标值 - 当前值)
                err_x = target_x - current_pose.position.x
                err_y = target_y - current_pose.position.y
                err_z = target_z - current_pose.position.z
                
                # 设定比例增益 Kp (可以将其视为“反应速度”)
                kp = 0.5
                
                # 计算速度并限制最大速度 (防止积分器饱和或物理模型失控)
                # 水平最大 2.0 m/s，垂直最大 1.0 m/s
                msg.linear.x = max(min(kp * err_x, 2.0), -2.0)
                msg.linear.y = max(min(kp * err_y, 2.0), -2.0)
                msg.linear.z = max(min(kp * err_z, 1.0), -1.0)
                
                # 发送控制指令
                self.pubs[uav].publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MultiUAVTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('测试被手动终止。')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()