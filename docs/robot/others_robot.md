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

