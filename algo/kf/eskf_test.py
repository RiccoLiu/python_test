import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# 设置随机种子保证结果可复现
np.random.seed(42)

# 1. 参数设置
total_time = 10  # 总时间（秒）
dt = 0.1         # 时间步长（秒）
steps = int(total_time / dt)  # 总步数

# 2. 真实物理过程（小车实际运动）
# 假设小车从(0,0)点开始，以1.5m/s的速度做匀速运动 (在2D平面上)
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

# 4. 误差状态卡尔曼滤波(ESKF)初始化
# 名义状态：[位置x, 位置y, 速度大小, 角度]
nominal_state = np.array([0, 0, 1.0, np.radians(45)])  # 初始名义状态估计

# 误差状态：[误差dx, 误差dy, 误差dv, 误差dtheta]
error_state = np.zeros(4)  # 初始误差状态为0

# 误差协方差矩阵
P = np.diag([1.0, 1.0, 0.5, np.radians(5)])**2  # 初始误差协方差

# 过程噪声协方差矩阵（模型误差）
Q = np.diag([0.01, 0.01, 0.02, np.radians(0.5)])**2

# 测量噪声协方差（传感器精度）
R = np.array([[measurement_noise**2]])

# 存储ESKF结果
eskf_positions = np.zeros((steps, 2))
eskf_velocities = np.zeros(steps)
eskf_angles = np.zeros(steps)

# 用于存储误差状态的中间值
error_states = np.zeros((steps, 4))

# 5. 误差状态卡尔曼滤波(ESKF)主循环
for t in range(steps):
    # =============== 预测步骤 ===============
    # 名义状态预测（使用非线性模型）
    x, y, v, theta = nominal_state
    nominal_state[0] = x + v * np.cos(theta) * dt  # x位置
    nominal_state[1] = y + v * np.sin(theta) * dt  # y位置
    # v和theta保持不变（由过程噪声模型覆盖变化）
    
    # 计算状态转移矩阵（对于误差状态）
    F = np.array([
        [1, 0, np.cos(theta)*dt, -v*np.sin(theta)*dt],
        [0, 1, np.sin(theta)*dt, v*np.cos(theta)*dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # 误差状态预测：dx_pred = F * dx
    error_state = F @ error_state
    
    # 误差协方差预测
    P = F @ P @ F.T + Q
    
    # =============== 更新步骤 ===============
    if t > 0:  # 避免第一次测量更新时的除零错误
        # 观测函数计算名义状态的预测测量值
        nominal_pos = nominal_state[:2]
        z_pred = distance_measurement(nominal_pos)
        
        # 计算观测雅可比矩阵（H矩阵）
        dist = distance_measurement(nominal_pos)
        if dist < 1e-6:
            H = np.array([0, 0, 0, 0])[None, :]  # 避免除零错误
        else:
            H = np.array([[nominal_pos[0]/dist, nominal_pos[1]/dist, 0, 0]])
        
        # 计算卡尔曼增益
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        
        # 实际的测量值
        z_actual = distance_measurements[t]
        
        # 计算测量残差
        residual = z_actual - z_pred
        
        # 更新误差状态
        error_state += K.flatten() * residual
        
        # 更新误差协方差
        I = np.eye(4)
        P = (I - K @ H) @ P
    
    # =============== 误差状态注入和重置 ===============
    # 注入误差状态到名义状态
    nominal_state[0] += error_state[0]  # 注入位置x误差
    nominal_state[1] += error_state[1]  # 注入位置y误差
    nominal_state[2] += error_state[2]  # 注入速度误差
    
    # 对于角度使用旋转矩阵形式注入
    dtheta = error_state[3]
    R_matrix = np.array([
        [np.cos(dtheta), -np.sin(dtheta)],
        [np.sin(dtheta), np.cos(dtheta)]
    ])
    
    # 更新名义位置和角度
    pos_vector = np.array([nominal_state[0], nominal_state[1]])
    new_pos = R_matrix @ pos_vector
    nominal_state[0], nominal_state[1] = new_pos[0], new_pos[1]
    nominal_state[3] = (nominal_state[3] + dtheta) % (2 * np.pi)
    
    # 重置误差状态为零（除了角度误差有特殊处理）
    error_state[:] = 0  # 重置所有误差状态
    # error_state[3] = 0  # 角度误差重置为零（已注入）
    
    # 存储误差状态（仅用于分析和可视化）
    error_states[t] = error_state
    
    # 存储结果
    eskf_positions[t, 0] = nominal_state[0]
    eskf_positions[t, 1] = nominal_state[1]
    eskf_velocities[t] = nominal_state[2]
    eskf_angles[t] = nominal_state[3]

# 计算真实距离（用于比较）
true_distances = np.sqrt(real_positions[:,0]**2 + real_positions[:,1]**2)

# 6. 结果可视化
plt.figure(figsize=(18, 15))

# 2D轨迹图
plt.subplot(3, 2, 1)
plt.plot(real_positions[:,0], real_positions[:,1], 'g-', linewidth=2, label='真实轨迹')
plt.plot(eskf_positions[:,0], eskf_positions[:,1], 'b-', linewidth=1.5, label='ESKF估计')
plt.scatter(0, 0, c='r', s=100, marker='*', label='原点')
plt.title('2D 轨迹估计 (ESKF)')
plt.xlabel('X 位置 (米)')
plt.ylabel('Y 位置 (米)')
plt.legend()
plt.grid(True)
plt.axis('equal')

# 速度估计
plt.subplot(3, 2, 2)
plt.plot(np.arange(0, total_time, dt), np.ones(steps)*real_velocity, 'g-', linewidth=2, label='真实速度')
plt.plot(np.arange(0, total_time, dt), eskf_velocities, 'b-', linewidth=1.5, label='估计速度')
plt.title('速度估计 (ESKF)')
plt.xlabel('时间 (秒)')
plt.ylabel('速度 (米/秒)')
plt.legend()
plt.grid(True)

# 角度估计
plt.subplot(3, 2, 3)
plt.plot(np.arange(0, total_time, dt), np.ones(steps)*real_angle, 'g-', linewidth=2, label='真实角度')
plt.plot(np.arange(0, total_time, dt), eskf_angles, 'b-', linewidth=1.5, label='估计角度')
plt.title('角度估计 (ESKF)')
plt.xlabel('时间 (秒)')
plt.ylabel('角度 (弧度)')
plt.legend()
plt.grid(True)

# 距离测量与估计
plt.subplot(3, 2, 4)
plt.plot(np.arange(0, total_time, dt), true_distances, 'g-', linewidth=2, label='真实距离')
plt.plot(np.arange(0, total_time, dt), distance_measurements, 'rx', alpha=0.4, label='传感器测量')
plt.title('距离原点估计 (ESKF)')
plt.xlabel('时间 (秒)')
plt.ylabel('距离 (米)')
plt.legend()
plt.grid(True)

# 位置误差随时间变化
plt.subplot(3, 2, 5)
position_errors = np.linalg.norm(eskf_positions - real_positions, axis=1)
plt.plot(np.arange(0, total_time, dt), position_errors, 'b-', label='位置误差')
plt.title('位置误差随时间变化 (ESKF)')
plt.xlabel('时间 (秒)')
plt.ylabel('位置误差 (米)')
plt.grid(True)

# 误差状态分量
plt.subplot(3, 2, 6)
plt.plot(np.arange(0, total_time, dt), error_states[:,0], 'r-', label='位置x误差')
plt.plot(np.arange(0, total_time, dt), error_states[:,1], 'b-', label='位置y误差')
plt.plot(np.arange(0, total_time, dt), error_states[:,2], 'g-', label='速度误差')
plt.plot(np.arange(0, total_time, dt), error_states[:,3], 'm-', label='角度误差')
plt.title('误差状态分量 (ESKF)')
plt.xlabel('时间 (秒)')
plt.ylabel('误差状态值')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 计算误差
position_errors = np.linalg.norm(eskf_positions - real_positions, axis=1)
sensor_errors = np.abs(distance_measurements - true_distances)
eskf_distance_errors = np.abs([distance_measurement(eskf_positions[t]) - true_distances[t] for t in range(steps)])
velocity_errors = np.abs(eskf_velocities - real_velocity)
angle_errors = np.abs(eskf_angles - real_angle)

# 打印性能统计数据
print("\nESKF性能统计:")
print(f"传感器测量平均绝对误差: {np.mean(sensor_errors):.4f} 米")
print(f"ESKF距离估计平均绝对误差: {np.mean(eskf_distance_errors):.4f} 米")
print(f"ESKF位置平均绝对误差: {np.mean(position_errors):.4f} 米")
print(f"ESKF速度平均绝对误差: {np.mean(velocity_errors):.4f} 米/秒")
print(f"ESKF角度平均绝对误差: {np.mean(angle_errors):.4f} 弧度")
print(f"相当于角度误差: {np.degrees(np.mean(angle_errors)):.4f} 度")


