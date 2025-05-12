# 机器人学

## 空间描述

**右乘连体左乘基**


1. 对于两个变换的叠加：\( M_2 M_1 \) 表示先进行 \( M_1 \) 变换，再进行 \( M_2 \) 变换，这里 \( M_1 \)、\( M_2 \) 都是自然基坐标系下。

2. 如果 \( M_2 \) 变换是在 \( M_1 \) 坐标系基础上进行的，那么根据相似矩阵把 \( M_2 \) 转换成自然基坐标系下：\( M_1 M_2 M_1^{-1} \)

3. 那么两个变换叠加就是：\( (M_1 M_2 M_1^{-1}) M_1 = M_1 M_2 \)

这是一个很有意思的现象，如果每个变换都是在上个变换基础上进行的，那么只要把矩阵顺序反过来即可：

- 所有变换都在自然基下：\( M_4 M_3 M_2 M_1 \)
- 每个变换在前一个变换后的坐标系下：\( M_1 M_2 M_3 M_4 \)

![alt text](<pic/截屏2025-04-25 14.28.28.png>)
![alt text](<pic/截屏2025-04-25 14.28.54.png>)

一个姿态若能被一组俯仰角绝对值大于90°的Z-Y-X欧拉角或X-Y-Z固定角描述，那么也能被另一组俯仰角绝对值不大于90°的Z-Y-X欧拉角或X-Y-Z固定角描述

**适用于齐次变换矩阵**

需要注意：

1）右乘是先平移、后旋转；

2）左乘是先旋转、后平移；

3）相对于基础坐标系的旋转（左乘旋转），可能会产生平移

任给一个姿态（或旋转），必有两组反号的欧拉参数与之对应

## 运动学

一个有N个关节的串联机构，有4N个运动学参量，其中3N个是连杆参数、N个是关节变量，它们包含了串联机构的全部空间几何信息

![alt text](<pic/截屏2025-04-25 15.09.03.png>)


## 四元数球面线性插值（Slerp）在机器人学中的应用

### 1. 四元数在机器人学中的作用
在机器人学中，四元数是一种高效且无奇异性的旋转表示方法，常用于描述和计算机器人末端执行器的姿态。相比欧拉角，四元数避免了万向节锁（Gimbal Lock）问题；相比旋转矩阵，四元数的存储和计算更加高效。

### 2. Slerp的基本原理
Slerp（Spherical Linear Interpolation）是一种在两个四元数之间进行插值的方法，用于平滑地过渡旋转。其特点是：
- 插值结果始终在单位四元数的球面上。
- 插值路径是球面上的大圆弧，保证了旋转的平滑性。
- 插值的角速度恒定，适合机器人运动规划中的平滑旋转。

公式如下：
\[
\text{Slerp}(q_1, q_2, t) = \frac{\sin((1-t)\theta)}{\sin(\theta)} q_1 + \frac{\sin(t\theta)}{\sin(\theta)} q_2
\]
其中：
- \( q_1 \) 和 \( q_2 \) 是起始和目标四元数。
- \( t \in [0, 1] \) 是插值因子。
- \( \theta \) 是 \( q_1 \) 和 \( q_2 \) 之间的夹角，计算公式为 \( \cos(\theta) = q_1 \cdot q_2 \)。

### 3. Slerp在笛卡尔空间规划中的应用
在机器人学中，笛卡尔空间规划是指在任务空间（通常是末端执行器的位姿空间）中规划路径。Slerp在笛卡尔空间规划中的主要应用包括：

 3.1 姿态插值
在机器人末端执行器从一个姿态过渡到另一个姿态时，使用Slerp可以生成平滑的旋转路径。

 3.2 平滑路径规划
在笛卡尔空间中，机器人路径通常由位置和姿态组成。通过结合线性插值（Lerp）计算位置过渡和Slerp计算姿态过渡，可以生成平滑的笛卡尔路径：
- 位置插值：使用线性插值计算起点和终点之间的平滑过渡。
- 姿态插值：使用Slerp计算起始和目标四元数之间的平滑旋转。

 3.3 避免奇异性
在笛卡尔路径规划中，使用欧拉角可能导致奇异性问题（如万向节锁）。通过使用四元数和Slerp，可以避免这些问题，确保路径的连续性和稳定性。

### 4. 示例：机器人末端执行器的笛卡尔路径规划
以下是一个结合位置插值和Slerp的笛卡尔路径规划示例：

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

def slerp(q1, q2, t):
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s1 = np.sin(theta_0 - theta) / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return s1 * q1 + s2 * q2

def cartesian_path(start_pos, end_pos, start_quat, end_quat, steps):
    path = []
    for t in np.linspace(0, 1, steps):
        # 位置插值
        position = (1 - t) * np.array(start_pos) + t * np.array(end_pos)
        # 姿态插值
        orientation = slerp(np.array(start_quat), np.array(end_quat), t)
        path.append((position, orientation))
    return path

# 示例输入
start_position = [0, 0, 0]
end_position = [1, 1, 1]
start_quaternion = [1, 0, 0, 0]
end_quaternion = [0, 1, 0, 0]

# 生成路径
path = cartesian_path(start_position, end_position, start_quaternion, end_quaternion, steps=10)
for pos, quat in path:
    print(f"Position: {pos}, Quaternion: {quat}")

# 可视化生成的路径
positions = np.array([p[0] for p in path])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='b', s=50)
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='r')
ax.set_title("Cartesian Path")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

```
在这个示例中，我们定义了一个 `slerp` 函数来计算两个四元数之间的球面线性插值。然后，我们使用 `cartesian_path` 函数生成从起始位置和姿态到目标位置和姿态的路径。最后，我们使用 Matplotlib 可视化生成的路径。

