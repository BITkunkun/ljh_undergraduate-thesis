import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped  # pose 消息类型

class PoseSubscriber(Node):
    def __init__(self):
        super().__init__('pose_subscriber_node')
        
        # 订阅 /quadrotor_1/pose
        self.subscription = self.create_subscription(
            PoseStamped,
            '/quadrotor_1/pose',
            self.pose_callback,
            10
        )

    def pose_callback(self, msg):
        # 在这里拿到真实坐标！
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z  # 这就是无人机高度！

        # 打印
        self.get_logger().info(f'✅ 无人机真实位置：')
        self.get_logger().info(f'   X = {x:.2f} m')
        self.get_logger().info(f'   Y = {y:.2f} m')
        self.get_logger().info(f'   Z（高度）= {z:.2f} m')

def main(args=None):
    rclpy.init(args=args)
    node = PoseSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
