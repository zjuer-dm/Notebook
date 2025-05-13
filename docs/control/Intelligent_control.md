# 智能控制
## 模糊控制
（1）模糊控制的基本思想是利用计算机来实现人的控制经验[1]；

（2）人工神经网络是一种应用类似于大脑神经突触联接的结构进行信息处理的数学模型[2]；

然而，对于复杂的系统，由于变量太多，往往难以正确的描述系统的动态，于是工程师便利用各种方法来简化系统动态，以达成控制的目的，但却不尽理想。换言之，传统的控制理论对于明确系统有强而有力的控制能力，但对于过于复杂或难以精确描述的系统，则显得无能为力了。因此便尝试着以模糊数学来处理这些控制问题。

模糊控制器包括四部分：

(1)模糊化。主要作用是选定模糊控制器的输入量，并将其转换为系统可识别的模糊量，具体包含以下三步：

第一，对输入量进行满足模糊控制需求的处理;

第二，对输入量进行尺度变换;

第三，确定各输入量的模糊语言取值和相应的隶属度函数。

(2)规则库。根据人类专家的经验建立模糊规则库。模糊规则库包含众多控制规则，是从实际控制经验过渡到模糊控制器的关键步骤。

(3)模糊推理。主要实现基于知识的推理决策。

(4)解模糊。主要作用是将推理得到的控制量转化为控制输出。

代码主要：MATLAB
```h
clear;
close all;
clc;
fis = mamfis("Name","tipper");
fis = addInput(fis,[0 10],"Name","service");
fis = addInput(fis,[0 10],"Name","food");

fis = addMF(fis,"service","gaussmf",[1.5 0],"Name","poor");
fis = addMF(fis,"service","gaussmf",[1.5 5],"Name","good");
fis = addMF(fis,"service","gaussmf",[1.5 10],"Name","excellent");

fis = addMF(fis,"food","trapmf",[-2 0 1 3],"Name","rancid");
fis = addMF(fis,"food","trapmf",[7 9 10 12],"Name","delicious");

fis = addOutput(fis,[0 30],"Name","tip");
fis = addMF(fis,"tip","trimf",[0 5 10],"Name","cheap");
fis = addMF(fis,"tip","trimf",[10 15 20],"Name","average");
fis = addMF(fis,"tip","trimf",[20 25 30],"Name","generous");

ruleList = [1 1 1 1 2;
            2 0 2 1 1;
            3 2 3 1 2];

fis = addRule(fis,ruleList);


fis2 = mamfis("Name","tipper");

fis2.Inputs(1) = fisvar;
fis2.Inputs(1).Name = "service";
fis2.Inputs(1).Range = [0 10];

fis2.Inputs(1).MembershipFunctions(1) = fismf;
fis2.Inputs(1).MembershipFunctions(1).Name = "poor";
fis2.Inputs(1).MembershipFunctions(1).Type = "gaussmf";
fis2.Inputs(1).MembershipFunctions(1).Parameters = [1.5 0];
fis2.Inputs(1).MembershipFunctions(2) = fismf;
fis2.Inputs(1).MembershipFunctions(2).Name = "good";
fis2.Inputs(1).MembershipFunctions(2).Type = "gaussmf";
fis2.Inputs(1).MembershipFunctions(2).Parameters = [1.5 5];
fis2.Inputs(1).MembershipFunctions(3) = fismf;
fis2.Inputs(1).MembershipFunctions(3).Name = "excellent";
fis2.Inputs(1).MembershipFunctions(3).Type = "gaussmf";
fis2.Inputs(1).MembershipFunctions(3).Parameters = [1.5 10];
fis2.Inputs(2) = fisvar([0 10],"Name","food");
fis2.Inputs(2).MembershipFunctions(1) = fismf("trapmf",[-2 0 1 3],...
                                              "Name","rancid");
fis2.Inputs(2).MembershipFunctions(2) = fismf("trapmf",[7 9 10 12],...
                                              "Name","delicious");

fis2.Outputs(1) = fisvar([0 30],"Name","tip");
mf1 = fismf("trimf",[0 5 10],"Name","cheap");
mf2 = fismf("trimf",[10 15 20],"Name","average");
mf3 = fismf("trimf",[20 25 30],"Name","generous");
fis2.Outputs(1).MembershipFunctions = [mf1 mf2 mf3];

rule1 = fisrule([1 1 1 1 2],2);
rule2 = fisrule([2 0 2 1 1],2);
rule3 = fisrule([3 2 3 1 2],2);
rules = [rule1 rule2 rule3];

rules = update(rules,fis2);
fis2.Rules = rules;
evalfis(fis,[1 2])

% 绘制输入 'service' 和 'food' 的隶属函数
figure;
subplot(2,1,1);
plotmf(fis, 'input', 1); % 绘制 'service' 的隶属函数
title('Membership Functions for Service');
subplot(2,1,2);
plotmf(fis, 'input', 2); % 绘制 'food' 的隶属函数
title('Membership Functions for Food');

% 绘制输出 'tip' 的隶属函数
figure;
plotmf(fis, 'output', 1); % 绘制 'tip' 的隶属函数
title('Membership Functions for Tip');
```

## 滑模控制

滑模控制是一种控制方法，它通过设计一个滑模面来控制系统的运动。

滑模控制理论（Sliding Mode Control, SMC）是一种建立在现代控制理论基础上的控制理论，主要数学核心为李雅普诺夫函数

滑模控制的核心思想是建立一个滑模面，将被控系统拉取到滑模面上来，使系统沿着滑模面运动。滑模控制的一个优势是无视外部扰动和不确定参数

考虑一个非线性系统：

$$\dot{x} = f(x) + g(x)u + d(t)$$

其中：
- $x$ 是系统状态向量
- $f(x)$ 是系统动态
- $g(x)$ 是控制输入矩阵
- $u$ 是控制输入
- $d(t)$ 是外部干扰

滑模面通常设计为：

$$s(x) = c_1x_1 + c_2x_2 + ... + c_{n-1}x_{n-1} + x_n = 0$$

其中 $c_i$ 是设计参数，需要满足 Hurwitz 条件。

滑模控制律通常包含两部分：

$$u = u_{eq} + u_{sw}$$

其中：
- $u_{eq}$ 是等效控制，用于维持系统在滑模面上
- $u_{sw}$ 是切换控制，用于驱动系统到达滑模面

$$V = \frac{1}{2}s^2$$

对时间求导：

$$\dot{V} = s\dot{s}$$

为了保证系统稳定性，需要满足：

$$\dot{V} = s\dot{s} < 0$$


以二阶系统为例：

$$\ddot{x} = f(x,\dot{x}) + g(x,\dot{x})u + d(t)$$

定义滑模面：

$$s = c_1x + \dot{x} = 0$$

对滑模面求导：

$$\dot{s} = c_1\dot{x} + \ddot{x} = c_1\dot{x} + f(x,\dot{x}) + g(x,\dot{x})u + d(t)$$

设计控制律：

$$u = u_{eq} + u_{sw} = -[g(x,\dot{x})]^{-1}[f(x,\dot{x}) + c_1\dot{x}] - k\cdot\text{sign}(s)$$

其中：
- $u_{eq} = -[g(x,\dot{x})]^{-1}[f(x,\dot{x}) + c_1\dot{x}]$ 是等效控制
- $u_{sw} = -k\cdot\text{sign}(s)$ 是切换控制
- $k$ 是切换增益，需要满足 $k > |d(t)|$
  
将控制律代入滑模面导数：

$$\dot{s} = -k\cdot\text{sign}(s) + d(t)$$

考虑李雅普诺夫函数：

$$V = \frac{1}{2}s^2$$

$$\dot{V} = s\dot{s} = s[-k\cdot\text{sign}(s) + d(t)] = -k|s| + s\cdot d(t) \leq -k|s| + |s|\cdot|d(t)| = -|s|(k - |d(t)|)$$

当 $k > |d(t)|$ 时，$\dot{V} < 0$，系统稳定。