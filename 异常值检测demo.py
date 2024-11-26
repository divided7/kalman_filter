import numpy as np
import matplotlib.pyplot as plt


def generate_quadratic_positions(a, b, c, num_points, noise=0.1, anomaly_indices=[]):
    x = np.linspace(-10, 10, num_points)
    y = a * x ** 2 + b * x + c
    positions = np.vstack((x, y)).T

    # 添加小量的高斯噪声
    positions += np.random.normal(0, noise, positions.shape)

    # 在指定索引处添加异常值
    for idx in anomaly_indices:
        positions[idx] += np.array([10, 10])  # 人为添加异常

    return positions


def kalman_filter_anomaly_detection(positions, threshold=5.0):
    frames, dim = positions.shape

    # 定义状态矩阵 [x, y, vx, vy] （位置和速度）
    state = np.zeros((4, 1))  # 初始状态
    state[:2, 0] = positions[0]  # 初始位置

    # 状态转移矩阵（假设简单的匀速模型）
    dt = 1  # 时间间隔
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # 观测矩阵，只观测位置 [x, y]
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    # 过程噪声协方差矩阵
    Q = np.eye(4) * 0.1

    # 观测噪声协方差矩阵
    R = np.eye(2) * 1.0

    # 初始误差协方差矩阵
    P = np.eye(4) * 500

    # 存储异常帧
    anomalies = []

    for i in range(1, frames):
        # 预测阶段
        state = F @ state
        P = F @ P @ F.T + Q

        # 计算残差 (观测值 - 预测值)
        z = positions[i].reshape(2, 1)
        y = z - H @ state
        S = H @ P @ H.T + R  # 残差协方差
        K = P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益

        # 更新状态和误差协方差矩阵
        state = state + K @ y
        P = (np.eye(4) - K @ H) @ P

        # 计算残差的欧氏距离
        residual = np.sqrt(y[0, 0] ** 2 + y[1, 0] ** 2)

        # 判断是否为异常值
        if residual > threshold:
            anomalies.append(i)

    return anomalies


# 生成二次曲线上的位置数据
positions = generate_quadratic_positions(a=0.5, b=-1.0, c=5.0, num_points=100, noise=0.1, anomaly_indices=[20, 50, 80])

# 使用卡尔曼滤波进行异常检测
anomalies = kalman_filter_anomaly_detection(positions, threshold=5.0)
print("异常帧索引：", anomalies)

# 可视化
plt.plot(positions[:, 0], positions[:, 1], label="Position")
plt.scatter(positions[anomalies, 0], positions[anomalies, 1], color='red', label="Anomalies")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Quadratic Curve with Anomalies")
plt.legend()
plt.show()
