## L2正则化

L2正则化是一种常用的正则化方法，用于防止模型过拟合。它通过在损失函数中添加一个正则化项来惩罚模型参数的大小。L2正则化的目标是最小化损失函数和正则化项的和，即：

$$
\min_{w} \frac{1}{2n} \sum_{i=1}^{n} (y_i - w^T x_i)^2 + \frac{\lambda}{2n} \sum_{j=1}^{d} w_j^2
$$

其中，$w$ 是模型参数，$x_i$ 是输入数据，$y_i$ 是输出数据，$n$ 是样本数量，$d$ 是参数数量，$\lambda$ 是正则化参数。

L2正则化项为：

$$
\frac{\lambda}{2n} \sum_{j=1}^{d} w_j^2
$$

## SGD

SGD是一种常用的优化算法，用于训练机器学习模型。它通过随机梯度下降来更新模型参数，从而最小化损失函数。SGD的更新公式为：

$$
w_{t+1} = w_t - \eta \nabla L(w_t)
$$

其中，$w_t$ 是第$t$次迭代时的模型参数，$\eta$ 是学习率，$\nabla L(w_t)$ 是损失函数在$w_t$处的梯度。

## Adam

Adam是一种常用的自适应学习率优化算法，用于训练机器学习模型。它通过计算梯度的一阶矩和二阶矩来更新模型参数，从而最小化损失函数。Adam的更新公式为：

$$
\begin{aligned}
    m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(w_t) \\
    v_t &= \beta_2 v_{t-1} + (1 - \beta_2) \nabla L(w_t)^2 \\
    w_{t+1} &= w_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

其中，$m_t$ 是梯度的一阶矩，$v_t$ 是梯度的二阶矩，$\beta_1$ 和 $\beta_2$ 是衰减系数，$\epsilon$ 是防止除零的小常数。

```py
def adam_update(parameters, gradients, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    for param, grad in zip(parameters, gradients):
        m[param] = beta1 * m[param] + (1 - beta1) * grad
        v[param] = beta2 * v[param] + (1 - beta2) * (grad ** 2)
        m_corrected = m[param] / (1 - beta1 ** t)
        v_corrected = v[param] / (1 - beta2 ** t)
        param_update = lr * m_corrected / (np.sqrt(v_corrected) + epsilon)
        param -= param_update
```

