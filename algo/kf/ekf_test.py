import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置随机种子保证结果可复现
np.random.seed(42)

# 1. 参数设置
total_time = 10  # 总时间（秒）
dt = 0.1         # 时间步长（秒）
steps = int(total_time / dt)  # 总步数

# 2. 真实物理过程（小车实际运动）
# 假设小车从(0,0)点开始，以1.5m/s的速度做匀速运动 (现在在2D平面上)
real_positions = np.zeros((steps, 2))  # 二维位置 [x, y]
real_velocity = 1.5  # 速度大小 (m/s)
real_angle = np.radians(45)  # 运动方向 (与x轴夹角45度)

# 生成真实位置（考虑真实世界的加速度和方向扰动）
acceleration_noise = 0.05  # 加速度扰动标准差 (m/s²)
angle_noise = np.radians(1)  # 方向扰动标准差 (弧度)

for t in range(1, steps):
    # 实际运动中有微小随机加速/减速和方向变化
    acceleration = np.random.normal(0, acceleration_noise)
    angle_change = np.random.normal(0, angle_noise)
    
    real_velocity += acceleration * dt
    real_angle += angle_change
    
    # 计算位移分量
    dx = real_velocity * np.cos(real_angle) * dt
    dy = real_velocity * np.sin(real_angle) * dt
    
    real_positions[t, 0] = real_positions[t-1, 0] + dx
    real_positions[t, 1] = real_positions[t-1, 1] + dy

# 3. 传感器测量（非线性测量：距离原点的距离）
def distance_measurement(position):
    """测量离原点的距离（非线性观测）"""
    return np.sqrt(position[0]**2 + position[1]**2)

# 生成距离测量数据（带噪声）
measurement_noise = 0.5  # 测量噪声标准差 (m)
distance_measurements = np.zeros(steps)

for t in range(steps):
    true_distance = distance_measurement(real_positions[t])
    distance_measurements[t] = true_distance + np.random.normal(0, measurement_noise)

# 4. 扩展卡尔曼滤波(EKF)初始化
# 状态向量：[位置x, 位置y, 速度大小, 角度]
state = np.array([0, 0, 1.0, np.radians(45)])  # 初始状态估计

# 状态协方差矩阵（初始不确定性）
P = np.diag([10.0, 10.0, 1.0, np.radians(10)])

# 状态转移矩阵（运动模型 - 匀速运动）
def state_transition(state, dt):
    """非线性状态转移函数"""
    x, y, v, theta = state
    new_x = x + v * np.cos(theta) * dt
    new_y = y + v * np.sin(theta) * dt
    return np.array([new_x, new_y, v, theta])

# 状态转移的雅可比矩阵（状态转移函数的导数）
def state_transition_jacobian(state, dt):
    """计算状态转移函数的雅可比矩阵"""
    x, y, v, theta = state
    J = np.eye(4)
    
    # dx/dtheta = -v * sin(theta) * dt
    J[0, 3] = -v * np.sin(theta) * dt
    
    # dy/dtheta = v * cos(theta) * dt
    J[1, 3] = v * np.cos(theta) * dt
    
    # dx/dv = cos(theta) * dt
    J[0, 2] = np.cos(theta) * dt
    
    # dy/dv = sin(theta) * dt
    J[1, 2] = np.sin(theta) * dt
    
    '''
        x, y, v, theta = state
        J = [ 
                [1, 0, dx/dv, dx/dtheta],
                [0, 1, dy/dv, dy/dtheta],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
    '''
    return J

# 过程噪声协方差矩阵（模型误差）
# 由于状态有不同单位，噪声也需要适当调整
Q = np.diag([0.01, 0.01, 0.02, np.radians(0.5)])**2

# 观测函数（从状态到测量）
def observation_model(state):
    """测量离原点的距离（非线性观测）"""
    x, y, v, theta = state
    return np.array([np.sqrt(x**2 + y**2)])

# 观测的雅可比矩阵（观测函数的导数）
def observation_jacobian(state):
    """计算观测函数的雅可比矩阵"""
    x, y, v, theta = state
    dist = np.sqrt(x**2 + y**2)
    
    # 避免除零错误
    if dist < 1e-6:
        return np.array([[0, 0, 0, 0]])
    
    # dz/dx = x / dist
    # dz/dy = y / dist
    # dz/dv = 0
    # dz/dtheta = 0
    return np.array([[x/dist, y/dist, 0, 0]])

# 测量噪声协方差（传感器精度）
R = np.array([[measurement_noise**2]])

# 存储卡尔曼滤波结果
ekf_positions = np.zeros((steps, 2))
ekf_velocities = np.zeros(steps)
ekf_angles = np.zeros(steps)

# 5. 扩展卡尔曼滤波(EKF)主循环
for t in range(steps):
    # ------------------ 预测步骤 ------------------
    # 使用非线性状态转移函数进行预测
    state_pred = state_transition(state, dt)
    
    # 计算状态转移的雅可比矩阵
    F = state_transition_jacobian(state, dt)
    
    # 协方差预测：使用一阶泰勒展开 P = F * P * F^T + Q
    P_pred = F @ P @ F.T + Q
    
    # ------------------ 更新步骤 ------------------
    if t > 0:  # 避免第一次测量更新时的除零错误
        # 使用非线性观测函数进行观测预测
        z_pred = observation_model(state_pred)
        
        # 计算观测的雅可比矩阵
        H = observation_jacobian(state_pred)
        
        # 计算卡尔曼增益
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # 计算测量残差（非线性观测差异）
        measurement_residual = distance_measurements[t] - z_pred
        
        # 状态更新：使用非线性观测差异
        state = state_pred + K.flatten() * measurement_residual
        
        # 协方差更新：使用一阶泰勒展开 P = (I - K*H) * P_pred
        I = np.eye(4)
        P = (I - K @ H) @ P_pred
    else:
        # 第一次没有测量，直接使用预测值
        state = state_pred
        P = P_pred
    
    # 存储结果
    ekf_positions[t, 0] = state[0]
    ekf_positions[t, 1] = state[1]
    ekf_velocities[t] = state[2]
    ekf_angles[t] = state[3]

estimated_distances = np.zeros(steps)
for t in range(steps):
    # 获取完整的状态估计 [x, y, v, theta]
    full_state = np.array([ekf_positions[t, 0], 
                          ekf_positions[t, 1], 
                          ekf_velocities[t], 
                          ekf_angles[t]])
    estimated_distances[t] = observation_model(full_state)

# 计算真实距离（用于比较）
true_distances = np.sqrt(real_positions[:,0]**2 + real_positions[:,1]**2)

# 6. 结果可视化
plt.figure(figsize=(15, 15))

# 2D轨迹图
plt.subplot(2, 2, 1)
plt.plot(real_positions[:,0], real_positions[:,1], 'g-', linewidth=2, label='真实轨迹')
plt.plot(ekf_positions[:,0], ekf_positions[:,1], 'b-', linewidth=1.5, label='EKF估计')
plt.scatter(0, 0, c='r', s=100, marker='*', label='原点')
plt.title('2D 轨迹估计')
plt.xlabel('X 位置 (米)')
plt.ylabel('Y 位置 (米)')
plt.legend()
plt.grid(True)
plt.axis('equal')

# 速度估计
plt.subplot(2, 2, 2)
plt.plot(np.arange(0, total_time, dt), np.ones(steps)*real_velocity, 'g-', linewidth=2, label='真实速度')
plt.plot(np.arange(0, total_time, dt), ekf_velocities, 'b-', linewidth=1.5, label='估计速度')
plt.title('速度估计')
plt.xlabel('时间 (秒)')
plt.ylabel('速度 (米/秒)')
plt.legend()
plt.grid(True)

# 角度估计
plt.subplot(2, 2, 3)
plt.plot(np.arange(0, total_time, dt), np.ones(steps)*real_angle, 'g-', linewidth=2, label='真实角度')
plt.plot(np.arange(0, total_time, dt), ekf_angles, 'b-', linewidth=1.5, label='估计角度')
plt.title('角度估计')
plt.xlabel('时间 (秒)')
plt.ylabel('角度 (弧度)')
plt.legend()
plt.grid(True)

# 距离测量与估计 - 这里使用我们预先计算好的estimated_distances
plt.subplot(2, 2, 4)
plt.plot(np.arange(0, total_time, dt), true_distances, 'g-', linewidth=2, label='真实距离')
plt.plot(np.arange(0, total_time, dt), distance_measurements, 'rx', alpha=0.4, label='传感器测量')
plt.plot(np.arange(0, total_time, dt), estimated_distances, 'b-', linewidth=1.5, label='EKF估计')
plt.title('距离原点估计')
plt.xlabel('时间 (秒)')
plt.ylabel('距离 (米)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 计算误差
position_errors = np.linalg.norm(ekf_positions - real_positions, axis=1)
sensor_errors = np.abs(distance_measurements - true_distances)
ekf_distance_errors = np.abs(estimated_distances - true_distances)
velocity_errors = np.abs(ekf_velocities - real_velocity)
angle_errors = np.abs(ekf_angles - real_angle)

# 打印性能统计数据
print("\n性能统计:")
print(f"传感器测量平均绝对误差: {np.mean(sensor_errors):.4f} 米")
print(f"EKF距离估计平均绝对误差: {np.mean(ekf_distance_errors):.4f} 米")
print(f"EKF位置平均绝对误差: {np.mean(position_errors):.4f} 米")
print(f"EKF速度平均绝对误差: {np.mean(velocity_errors):.4f} 米/秒")
print(f"EKF角度平均绝对误差: {np.mean(angle_errors):.4f} 弧度")
print(f"相当于角度误差: {np.degrees(np.mean(angle_errors)):.4f} 度")