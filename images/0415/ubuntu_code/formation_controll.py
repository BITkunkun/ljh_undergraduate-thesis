import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ros_ign_interfaces.msg import Dataframe
import numpy as np
import struct
import math
import time

class UAVFormationNode(Node):
    def __init__(self, uav_id):
        super().__init__(f'formation_controller_q{uav_id}')
        self.uav_id = uav_id
        self.uav_name = f'quadrotor_{uav_id}'
        
        # --- 编队拓扑与期望位置定义 ---
        # 3架无人机，期望距离为 sqrt(2)
        self.num_agents = 3
        self.desired_distance = math.sqrt(2.0)
        
        # 初始期望位置 (等边三角形) [cite: 146]
        self.p_star_init = {
            1: np.array([0.0, 0.0, 2.0]),
            2: np.array([math.sqrt(2), 0.0, 2.0]),
            3: np.array([math.sqrt(2)/2, math.sqrt(1.5), 2.0])
        }
        self.p_star = self.p_star_init[self.uav_id].copy()
        
        # 邻居定义 (全连接图以保证刚性)
        all_ids = [1, 2, 3]
        all_ids.remove(self.uav_id)
        self.neighbors = all_ids
        
        # --- 局部弱可观测性测量矩阵 C_i (依据论文 [cite: 181, 194]) ---
        # UAV1 测量 X, Y; UAV2 测量 X; UAV3 不测量水平位置
        if self.uav_id == 1:
            self.C = np.diag([1.0, 1.0, 1.0])
        elif self.uav_id == 2:
            self.C = np.diag([1.0, 0.0, 1.0])
        else:
            self.C = np.diag([0.0, 0.0, 1.0])
            
        # 状态初始化
        self.p_true = np.zeros(3) # 由里程计获取的真实位置(仅用于模拟局部测量和计算相对距离)
        self.p_est = self.p_star + np.random.uniform(-0.5, 0.5, 3) # 初始估计位置加入扰动
        self.neighbor_p_est = {n: self.p_star_init[n].copy() for n in self.neighbors}
        
        # 算法增益参数 (参考 MATLAB main.m)
        self.k_c = 5.0    # 控制器增益 [cite: 236]
        self.k_o = 10.0   # 观测器增益 
        self.edge_min = 0.5
        self.edge_max = 2.0
        
        # --- ROS 2 通信接口设置 ---
        # 1. 发布速度控制指令
        self.cmd_pub = self.create_publisher(Twist, f'/{self.uav_name}/cmd_vel', 10)
        # 2. 订阅自身里程计 (用于获取自身真实位置以模拟传感器)
        self.odom_sub = self.create_subscription(Odometry, f'/{self.uav_name}/odometry', self.odom_cb, 10)
        # 3. MBZIRC Dataframe 发送与接收
        self.tx_pub = self.create_publisher(Dataframe, f'/{self.uav_name}/tx', 10)
        self.rx_sub = self.create_subscription(Dataframe, f'/{self.uav_name}/rx', self.rx_cb, 10)
        
        # 记录起飞与平移时间
        self.start_time = time.time()
        self.is_translating = True
        self.translation_duration = 10.0 # 平移10s
        self.translation_velocity = np.array([0.5, 0.0, 0.0]) # 沿X轴平移速度
        
        # 定时器：50Hz 执行控制和观测器更新
        self.dt = 0.02
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info(f"{self.uav_name} 初始化完成，C矩阵: \n{self.C}")

    def odom_cb(self, msg):
        """获取真实位置，用于计算距离和局部测量"""
        self.p_true[0] = msg.pose.pose.position.x
        self.p_true[1] = msg.pose.pose.position.y
        self.p_true[2] = msg.pose.pose.position.z

    def rx_cb(self, msg):
        """处理来自其他无人机的数据 [通过 MBZIRC Dataframe]"""
        src_name = msg.src_address
        src_id = int(src_name.split('_')[1])
        if src_id in self.neighbors:
            # 解包邻居发来的估计位置 (3个double)
            try:
                data = struct.unpack('3d', msg.data)
                self.neighbor_p_est[src_id] = np.array([data[0], data[1], data[2]])
            except Exception as e:
                self.get_logger().warn(f"数据解包失败: {e}")

    def dgamma(self, elc, eld):
        """
        对应论文中基于距离的势场导数约束控制律 [cite: 283, 291]
        移植自 MATLAB dgamma.m
        """
        elm1 = self.edge_min
        elm2 = self.edge_max
        # 避免分母为0的安全截断
        elc = np.clip(elc, elm1 + 0.05, elm2 - 0.05)
        
        term1 = 2 * (elc - eld)
        term2 = (elm1 + elm2 - 2 * elc) / ((elm2 - elc) * (elc - elm1))
        term3 = (elm1 + elm2 - 2 * eld) / ((elm2 - elc) * (eld - elm1))
        
        dg = 0.001 * (term1 - term2 + term3)
        return dg

    def control_loop(self):
        current_time = time.time() - self.start_time
        
        # --- 1. 动态更新期望位置 (实现10s平移) ---
        if current_time < self.translation_duration:
            # 起飞并平移
            self.p_star = self.p_star_init[self.uav_id] + self.translation_velocity * current_time
        else:
            if self.is_translating:
                self.get_logger().info("10s 平移结束，悬停保持编队。")
                self.is_translating = False
            # 保持最终位置
            self.p_star = self.p_star_init[self.uav_id] + self.translation_velocity * self.translation_duration

        # --- 2. 模拟局部测量 (y_i) ---
        y_i_star = self.C @ self.p_star
        y_i_current = self.C @ self.p_true # 基于传感器提取对应维度的状态
        
        # --- 3. 计算 Controller (式 17) [cite: 276] ---
        u_i = np.zeros(3)
        # 加上对邻居的梯度控制力
        for j in self.neighbors:
            # 这里通过真实位置获取相对距离 (模拟UWB传感器)
            # 注意：实际平台如果用传感器，这里用真实传感器获取的距离
            # 此处用 odometry 模拟理想距离加上轻微噪声
            true_distance = np.linalg.norm(self.p_true - self.neighbor_p_est[j]) # 简化：使用邻居的估计位置近似相对距离
            
            gamma_prime = self.dgamma(true_distance, self.desired_distance)
            u_i -= gamma_prime * (self.p_est - self.neighbor_p_est[j])
            
        # 加上跟踪目标位置的 P 控制项
        u_i -= self.k_c * (self.p_est - self.p_star)

        # --- 4. 计算 Observer (式 12)  ---
        obs_feedback = np.zeros(3)
        # 来自自身的局部测量误差
        obs_feedback += (y_i_current - y_i_star) - self.C @ (self.p_est - self.p_star)
        
        # 来自邻居的协同估计误差补偿
        for j in self.neighbors:
            # 根据论文，如果已知邻居的期望和C矩阵
            C_j = np.diag([1.0, 1.0, 1.0]) if j == 1 else (np.diag([1.0, 0.0, 1.0]) if j == 2 else np.diag([0.0, 0.0, 1.0]))
            p_j_star = self.p_star_init[j] + (self.translation_velocity * min(current_time, self.translation_duration))
            obs_feedback -= C_j @ (self.neighbor_p_est[j] - p_j_star)
            
        p_est_dot = u_i + self.k_o * (self.C.T @ obs_feedback)
        
        # 欧拉法更新自身估计位置
        self.p_est += p_est_dot * self.dt

        # --- 5. 发布控制指令到底层 (单积分器模型直接输出速度) ---
        twist = Twist()
        twist.linear.x = float(p_est_dot[0])
        twist.linear.y = float(p_est_dot[1])
        twist.linear.z = float(p_est_dot[2])
        self.cmd_pub.publish(twist)

        # --- 6. 通过 MBZIRC 网络向邻居广播自己的估计位置 ---
        for j in self.neighbors:
            msg = Dataframe()
            msg.src_address = self.uav_name
            msg.dst_address = f'quadrotor_{j}'
            # 将 numpy array 打包为 3个 double 字节流
            msg.data = struct.pack('3d', self.p_est[0], self.p_est[1], self.p_est[2])
            self.tx_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    # 这里可以使用多线程执行器在一个终端中同时运行3个无人机的逻辑
    # 也可以在真实环境中分为3个终端分别传入 1, 2, 3 启动
    import sys
    if len(sys.argv) > 1:
        uav_id = int(sys.argv[1])
        node = UAVFormationNode(uav_id)
        rclpy.spin(node)
        node.destroy_node()
    else:
        print("请提供无人机ID，例如: python3 formation_control.py 1")

    rclpy.shutdown()

if __name__ == '__main__':
    main()