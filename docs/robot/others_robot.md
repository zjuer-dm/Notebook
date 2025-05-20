# 卡尔曼滤波推导笔记

## 一、引言

卡尔曼滤波是20世纪60年代由R.E. Kalman提出的一种递归估计算法，用于从含噪声的测量中估计系统的状态，最初用于导弹导航系统。它通过融合预测（先验）结果和实际测量，对系统状态进行动态估计，适用于线性和高斯分布的系统，应用领域广泛，包括导航、自动控制、信号处理、金融等，在导航中用于提高位置估计精度，在自动控制中用于实时调整控制参数以维持系统稳定，其优势在于有效处理动态系统中的不确定性和噪声。

## 二、基本概念

在推导卡尔曼滤波之前，先说明系统的几个重要概念：

### 1、状态

状态是对系统的内部表示，包含描述系统行为的所有信息。在卡尔曼滤波中，使用状态向量表示系统状态，一般记作$x_k$，其中$k$表示时刻。例如匀速行驶的小车，其状态向量为：

$x_k = \begin{bmatrix} p_k \\ v_k \end{bmatrix}$

其中，$p_k$为小车在第$k$时刻的位置，$v_k$为小车在第$k$时刻的速度。

### 2、测量

测量是指通过传感器或其他手段获得的对系统状态的观测值，记作$Z_k$，通常包含误差即测量噪声，测量噪声服从均值为0，方差矩阵为$R_k$的高斯分布。

### 3、控制

控制是指系统中的确定性输入信息，如在自动驾驶场景中，车辆的油门和方向盘转角是已知的，这些信息可用于帮助估计系统状态。

### 4、卡尔曼滤波的预测和更新

卡尔曼滤波包含两个主要步骤：预测和更新。

## 三、卡尔曼滤波推导

### 1、系统模型

卡尔曼滤波基于以下线性动态系统模型：

状态方程：

$x_k = F_k x_{k-1} + B_k u_k + w_{k-1}$

测量方程：

$Z_k = H_k x_k + v_k$

其中：

- $x_k$：第$k$时刻的状态向量
- $F_k$：状态转移矩阵
- $B_k$：控制输入矩阵
- $u_k$：第$k$时刻的控制输入向量
- $w_{k-1}$：过程噪声，服从均值为0，协方差矩阵为$Q_{k-1}$的高斯分布
- $Z_k$：第$k$时刻的测量向量
- $H_k$：测量矩阵
- $v_k$：测量噪声，服从均值为0，协方差矩阵为$R_k$的高斯分布

### 2、预测步骤

预测步骤的目的是基于上一时刻的状态估计和控制输入，预测当前时刻的状态。

#### 先验状态估计

$x_k^- = F_k x_{k-1}^+ + B_k u_k$

其中，$x_k^-$为第$k$时刻的先验状态估计，$x_{k-1}^+$为第$k-1$时刻的后验状态估计。

#### 先验估计协方差矩阵

$P_k^- = F_k P_{k-1}^+ F_k^T + Q_{k-1}$

其中，$P_k^-$为第$k$时刻的先验估计协方差矩阵，$P_{k-1}^+$为第$k-1$时刻的后验估计协方差矩阵。

**推导说明：**

先验估计协方差矩阵的推导基于状态方程：

$x_k = F_k x_{k-1} + B_k u_k + w_{k-1}$

假设$x_{k-1}$的估计为$x_{k-1}^+$，其估计误差为$\tilde{x}_{k-1} = x_{k-1} - x_{k-1}^+$，则$x_k$的估计误差为：

$\tilde{x}_k = x_k - x_k^- = F_k \tilde{x}_{k-1} + w_{k-1}$

估计误差协方差矩阵为：

$E[\tilde{x}_k \tilde{x}_k^T] = F_k E[\tilde{x}_{k-1} \tilde{x}_{k-1}^T] F_k^T + E[w_{k-1} w_{k-1}^T] = F_k P_{k-1}^+ F_k^T + Q_{k-1}$

因此，得到先验估计协方差矩阵的公式：

$P_k^- = F_k P_{k-1}^+ F_k^T + Q_{k-1}$

### 3、更新步骤

更新步骤的目的是利用当前时刻的测量值修正先验状态估计，得到后验状态估计。

#### 后验状态估计

$x_k^+ = x_k^- + K_k (Z_k - H_k x_k^-)$

其中，$x_k^+$为第$k$时刻的后验状态估计，$K_k$为卡尔曼增益。

#### 卡尔曼增益

$K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^{-1}$

**推导说明：**

卡尔曼增益的计算公式可以通过最小化后验估计误差的协方差矩阵来推导。

后验估计误差为：

$\tilde{x}_k = x_k - x_k^+ = x_k - (x_k^- + K_k (Z_k - H_k x_k^-))$

代入测量方程$Z_k = H_k x_k + v_k$，得到：

$\tilde{x}_k = (I - K_k H_k)(x_k - x_k^-) - K_k v_k$

假设$x_k - x_k^-$和$v_k$是互不相关的，并且它们的协方差矩阵分别为$P_k^-$和$R_k$，则后验估计误差的协方差矩阵为：

$E[\tilde{x}_k \tilde{x}_k^T] = (I - K_k H_k) P_k^- (I - K_k H_k)^T + K_k R_k K_k^T$

为了最小化该协方差矩阵的迹（即最小化估计误差的方差），对$K_k$求导并令导数为零：

$\frac{\partial E[\tilde{x}_k \tilde{x}_k^T]}{\partial K_k} = -H_k P_k^- (I - K_k H_k)^T + K_k R_k + (I - K_k H_k) P_k^- H_k^T + R_k K_k^T = 0$

化简后得到：

$K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^{-1}$

#### 后验估计协方差矩阵

$P_k^+ = (I - K_k H_k) P_k^-$

**推导说明：**

将卡尔曼增益代入后验估计误差的协方差矩阵表达式：

$E[\tilde{x}_k \tilde{x}_k^T] = (I - K_k H_k) P_k^- (I - K_k H_k)^T + K_k R_k K_k^T$

代入$K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^{-1}$，并利用矩阵的性质，可以化简得到：

$P_k^+ = (I - K_k H_k) P_k^-$

### 4、递归过程

卡尔曼滤波的递归过程如下：

1. 初始化：给定初始状态估计$x_0^+$和初始估计协方差矩阵$P_0^+$。
2. 预测步骤：
   - 计算先验状态估计$x_k^-$。
   - 计算先验估计协方差矩阵$P_k^-$。
3. 更新步骤：
   - 计算卡尔曼增益$K_k$。
   - 计算后验状态估计$x_k^+$。
   - 计算后验估计协方差矩阵$P_k^+$。
4. 迭代：将$x_k^+$和$P_k^+$作为下一时刻的初始值，重复步骤2和3。

通过以上递归过程，卡尔曼滤波能够动态地估计系统状态，融合预测信息和测量信息，逐步减少估计误差，提高估计精度。

## EKF

扩展卡尔曼滤波（Extended Kalman Filter，EKF）通过局部线性来解决非线性的问题。将非线性的预测方程和观测方程进行求导，以切线代替的方式来线性化。其实就是在均值处进行一阶泰勒展开。

![alt text](<pic/Screenshot 2024-11-28 at 16.51.08.png>)

```m
function [] = ekf_localization()
 
    close all;
    clear all;

    disp('EKF Start!')

    time = 0;
    global endTime; % [sec]
    endTime = 60;
    global dt;
    dt = 0.1; % [sec]

    removeStep = 5;

    nSteps = ceil((endTime - time)/dt);

    estimation.time = [];
    estimation.u = [];
    estimation.GPS = [];
    estimation.xOdom = [];
    estimation.xEkf = [];
    estimation.xTruth = [];

    % 状态向量 [x y yaw]'
    xEkf = [0 0 0]';
    PxEkf = eye(3); % 初始协方差矩阵

    % 地面真实状态
    xTruth = xEkf;

    % 仅使用里程计
    xOdom = xTruth;

    % 观测向量 [x y yaw]'
    z = [0 0 0]';

    % 模拟参数
    global noiseQ
    noiseQ = diag([0.1 0 degreeToRadian(10)]).^2; %[Vx Vy yawrate]

    global noiseR
    noiseR = diag([0.5 0.5 degreeToRadian(5)]).^2; %[x y yaw]

    % 协方差矩阵
    convQ = noiseQ; % 运动模型的协方差
    convR = noiseR; % 观测模型的协方差

    % 主循环
    for i = 1:nSteps
        time = time + dt;
        % 输入
        u = robotControl(time);
        % 观测
        [z, xTruth, xOdom, u] = prepare(xTruth, xOdom, u);

        % ------ 卡尔曼滤波 --------
        % 预测
        xEkf = doMotion(xEkf, u); % 预测状态
        F = jacobF(xEkf, u); % 获取雅可比矩阵
        PxEkf = F * PxEkf * F' + convQ; % 更新协方差

        % 更新
        z_hat = doObservation(xEkf); % 预测观测
        H = jacobH(xEkf); % 获取观测模型的雅可比
        S = H * PxEkf * H' + convR; % 创新协方差
        K = PxEkf * H' / S; % 卡尔曼增益
        xEkf = xEkf + K * (z - z_hat); % 更新状态估计
        PxEkf = (eye(3) - K * H) * PxEkf; % 更新协方差

        % 模拟估计
        estimation.time = [estimation.time; time];
        estimation.xTruth = [estimation.xTruth; xTruth'];
        estimation.xOdom = [estimation.xOdom; xOdom'];
        estimation.xEkf = [estimation.xEkf; xEkf'];
        estimation.GPS = [estimation.GPS; z'];
        estimation.u = [estimation.u; u'];

        % 实时绘图
        if rem(i, removeStep) == 0
            plot(estimation.GPS(:,1), estimation.GPS(:,2), '*m', 'MarkerSize', 5); hold on;
            plot(estimation.xOdom(:,1), estimation.xOdom(:,2), '.k', 'MarkerSize', 10); hold on;
            plot(estimation.xEkf(:,1), estimation.xEkf(:,2), '.r', 'MarkerSize', 10); hold on;
            plot(estimation.xTruth(:,1), estimation.xTruth(:,2), '.b', 'MarkerSize', 10); hold on;
            axis equal;
            grid on;
            drawnow;
        end 
    end
    close
    
    finalPlot(estimation);
 
end

% 控制输入
function u = robotControl(time)
    global endTime;

    T = 10; % sec
    Vx = 1.0; % m/s
    Vy = 0.2; % m/s
    yawrate = 5; % deg/s
    
    if time > (endTime / 2)
        yawrate = -5;
    end
    
    u = [Vx * (1 - exp(-time / T)), Vy * (1 - exp(-time / T)), degreeToRadian(yawrate) * (1 - exp(-time / T))]';
end

% 准备观测
function [z, xTruth, xOdom, u] = prepare(xTruth, xOdom, u)
    global noiseQ;
    global noiseR;

    % 地面真实状态
    xTruth = doMotion(xTruth, u);
    % 添加运动噪声
    u = u + noiseQ * randn(3, 1);
    % 仅使用里程计
    xOdom = doMotion(xOdom, u);
    % 添加观测噪声
    z = xTruth + noiseR * randn(3, 1);
end

% 运动模型
function x = doMotion(x, u)
    global dt;
    
    theta = x(3);
    
    % 更新状态
    x(1) = x(1) + u(1) * cos(theta) * dt; % 更新 x 位置
    x(2) = x(2) + u(1) * sin(theta) * dt; % 更新 y 位置
    x(3) = x(3) + u(3) * dt;               % 更新偏航角

    % 确保偏航角在范围 [-pi, pi] 内
    x(3) = atan2(sin(x(3)), cos(x(3)));
end


function jF = jacobF(x, u)
    global dt;
    
    theta = x(3);
    
    jF = [1 0 -u(1) * sin(theta) * dt; 
          0 1  u(1) * cos(theta) * dt; 
          0 0  1];
end

% 观测模型
function z_hat = doObservation(xPred)
    z_hat = xPred; 
end


function jH = jacobH(x)
    jH = [1 0 0; 
          0 1 0; 
          0 0 1]; 
end


function [] = finalPlot(estimation)
    figure;
    
    plot(estimation.GPS(:,1), estimation.GPS(:,2), '*m', 'MarkerSize', 5); hold on;
    plot(estimation.xOdom(:,1), estimation.xOdom(:,2), '.k', 'MarkerSize', 10); hold on;
    plot(estimation.xEkf(:,1), estimation.xEkf(:,2), '.r', 'MarkerSize', 10); hold on;
    plot(estimation.xTruth(:,1), estimation.xTruth(:,2), '.b', 'MarkerSize', 10); hold on;
    legend('GPS 观测', '仅使用里程计', 'EKF 本地化', '地面真实值');

    xlabel('X (米)', 'fontsize', 12);
    ylabel('Y (米)', 'fontsize', 12);
    grid on;
    axis equal;
    
    
    error_odometry = estimation.xOdom - estimation.xTruth; % 里程计误差
    error_ekf = estimation.xEkf - estimation.xTruth; % EKF误差

    disp(['里程计均方误差: ', num2str(mean(vecnorm(error_odometry, 2, 2).^2))]);
    disp(['EKF均方误差: ', num2str(mean(vecnorm(error_ekf, 2, 2).^2))]);
end

function radian = degreeToRadian(degree)
    radian = degree / 180 * pi;
end
```

## 相机标定

在图像测量过程以及机器视觉应用中，为确定空间物体表面某点的三维几何位置与其在图像中对应点之间的相互关系，必须建立相机成像的几何模型，这些几何模型参数就是相机参数。

【1】进行摄像机标定的目的：求出相机的内、外参数，以及畸变参数。
【2】标定相机后通常是想做两件事：一个是由于每个镜头的畸变程度各不相同，通过相机标定可以校正这种镜头畸变矫正畸变，生成矫正后的图像；另一个是根据获得的图像重构三维场景。

