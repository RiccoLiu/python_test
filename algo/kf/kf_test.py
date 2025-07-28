
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
# 假设小车从0点开始，以1.5m/s的速度做匀速运动
real_position = np.zeros(steps)
real_velocity = 1.5  # 恒定速度 (m/s)

# 生成真实位置（考虑真实世界的加速度扰动）
acceleration_noise = 0.05  # 加速度扰动标准差 (m/s²)
for t in range(1, steps):
    # 实际运动中有微小随机加速/减速
    acceleration = np.random.normal(0, acceleration_noise)
    real_velocity += acceleration * dt
    real_position[t] = real_position[t-1] + real_velocity * dt

# 3. 传感器测量（带噪声的位置测量）
measurement_noise = 0.8  # 测量噪声标准差 (m)
position_measurements = real_position + np.random.normal(0, measurement_noise, steps)

# 4. 卡尔曼滤波初始化
# 状态向量：[位置, 速度]
state = np.array([0, 0])  # 初始状态估计 [位置=0, 速度=0]
# 状态协方差矩阵（初始不确定性：位置不确定度大，速度不确定度大）
P = np.array([[10, 0], 
              [0, 10]])

# 状态转移矩阵（运动模型）
# 假设匀速运动：新位置 = 旧位置 + 速度*dt
F = np.array([[1, dt],
              [0, 1]])

# 过程噪声协方差矩阵（模型误差）
# 假设模型不完美（可能有未建模的加速度）
Q = np.array([[0.05, 0], 
              [0, 0.05]])

# 观测矩阵（我们只能观测到位置）
H = np.array([[1, 0]])

# 测量噪声协方差（传感器精度）
R = np.array([[measurement_noise**2]])

# 存储卡尔曼滤波结果
kalman_positions = np.zeros(steps)
kalman_velocities = np.zeros(steps)

# 5. 卡尔曼滤波主循环
for t in range(steps):
    # 预测步骤
    state = F @ state           # 状态预测：x = F * x
    P = F @ P @ F.T + Q         # 协方差预测：P = F * P * F^T + Q
    
    # 更新步骤（有传感器数据时）
    if t > 0:  # 避免第一次测量更新时的除零错误
        # 计算卡尔曼增益
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)  # K = P * H^T * (H*P*H^T + R)^-1
        
        # 状态更新：x = x + K * (测量值 - H*x)
        measurement_residual = position_measurements[t] - H @ state        
        state = state + K.flatten() * measurement_residual
        
        # 协方差更新：P = (I - K*H) * P
        I = np.eye(2)
        P = (I - K @ H) @ P

    # 存储结果
    kalman_positions[t] = state[0]
    kalman_velocities[t] = state[1]

# exit(-1)

# 6. 结果可视化
plt.figure(figsize=(15, 10))

# 位置随时间变化图
plt.subplot(2, 2, 1)
plt.plot(np.arange(0, total_time, dt), real_position, 'g-', linewidth=2, label='真实位置')
plt.plot(np.arange(0, total_time, dt), position_measurements, 'rx', alpha=0.4, label='传感器测量')
plt.plot(np.arange(0, total_time, dt), kalman_positions, 'b-', linewidth=1.5, label='卡尔曼滤波')
plt.title('位置估计')
plt.xlabel('时间 (秒)')
plt.ylabel('位置 (米)')
plt.legend()
plt.grid(True)

# 位置误差分布
plt.subplot(2, 2, 2)
position_errors = kalman_positions - real_position
sensor_errors = position_measurements - real_position
plt.hist(position_errors, bins=30, alpha=0.7, label='卡尔曼误差')
plt.hist(sensor_errors, bins=30, alpha=0.5, label='传感器误差')
plt.title('位置误差分布')
plt.xlabel('误差 (米)')
plt.ylabel('频次')
plt.legend()
plt.grid(True)

# 速度随时间变化图
plt.subplot(2, 2, 3)
plt.plot(np.arange(0, total_time, dt), np.ones(steps) * real_velocity, 'g-', linewidth=2, label='真实速度')
plt.plot(np.arange(0, total_time, dt), kalman_velocities, 'b-', linewidth=1.5, label='估计速度')
plt.title('速度估计')
plt.xlabel('时间 (秒)')
plt.ylabel('速度 (米/秒)')
plt.legend()
plt.grid(True)

# 位置估计误差随时间变化
plt.subplot(2, 2, 4)
plt.plot(np.arange(0, total_time, dt), position_errors, 'b-', label='卡尔曼误差')
plt.plot(np.arange(0, total_time, dt), sensor_errors, 'r-', alpha=0.4, label='传感器误差')
plt.title('位置误差随时间变化')
plt.xlabel('时间 (秒)')
plt.ylabel('误差 (米)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 打印性能统计数据
print("\n性能统计:")
print(f"传感器测量平均绝对误差: {np.mean(np.abs(sensor_errors)):.4f} 米")
print(f"卡尔曼滤波平均绝对误差: {np.mean(np.abs(position_errors)):.4f} 米")
print(f"卡尔曼滤波速度平均绝对误差: {np.mean(np.abs(kalman_velocities - real_velocity)):.4f} 米/秒")


