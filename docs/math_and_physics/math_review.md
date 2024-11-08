# 微分方程
$$ y''  + py' +qy = f(x) $$

$$ 由\lambda ^2 + p \lambda  +q = 0 来解出\lambda _1 ,\lambda _2 $$

$$ y = C_1e^{\lambda _1 x} + C_2e^{\lambda _2 x} $$

$$ y = (C_1 +C_2 x ) e^{\lambda x} $$

$$ y = e^{\alpha x} (C_1cos\beta x + C_2sin\beta x) $$

* 一阶微分方程

$ y '  + g(x) y  = h(x) $

$$ y=e^{-\int g(x) d x}\left[C+\int h(x) e^{\int g(x) d x} d x\right]
$$

# 常用
$$ f(x) = \sum _{k = 0}^{n} \frac{f^{(k)}(0)}{k!}x^k + O(x^n) $$

* 由特殊到一般
* 洛必达
* 取对数
* 取倒数
* $ （1+x +x^2）(1-x) = 1 - x^3 $
* 向已知公式靠拢
* 画图，更直观
* 周期性质
* 构造函数
* 求幂级数的和函数（构造求导与积分）
* 拆分，构造
* 立体几何知识
* 参数代换

$$\begin{aligned}
e^{x}&=\sum_{n=0}^{\infty} \frac{1}{n !} x^{n}=1+x+\frac{1}{2 !} x^{2}+\cdots \in(-\infty,+\infty) \\
\sin x&=\sum_{n=0}^{\infty} \frac{(-1)^{n}}{(2 n+1) !} x^{2 n+1}=x-\frac{1}{3 !} x^{3}+\frac{1}{5 !} x^{5}+\cdots, x \in(-\infty,+\infty) \\
\cos x&=\sum_{n=0}^{\infty} \frac{(-1)^{n}}{(2 n) !} x^{2 n}=1-\frac{1}{2 !} x^{2}+\frac{1}{4 !} x^{4}+\cdots, x \in(-\infty,+\infty) \\
\ln (1+x)&=\sum_{n=0}^{\infty} \frac{(-1)^{n}}{n+1} x^{n+1}=x-\frac{1}{2} x^{2}+\frac{1}{3} x^{3}+\cdots, x \in(-1,1] \\
\frac{1}{1-x}&=\sum_{n=0}^{\infty} x^{n}=1+x+x^{2}+x^{3}+\cdots, x \in(-1,1) \\
\frac{1}{1+x}&=\sum_{n=0}^{\infty}(-1)^{n} x^{n}=1-x+x^{2}-x^{3}+\cdots, x \in(-1,1)\\
    (1+x)^{\alpha}&=1+\sum_{n=1}^{\infty} \frac{\alpha(\alpha-1) \cdots(\alpha-n+1)}{n !} x^{n}=1+\alpha x+\frac{\alpha(\alpha-1)}{2 !} x^{2}+\cdots, x \in(-1,1) \\
    \arctan x&=\sum_{n=0}^{\infty} \frac{(-1)^{n}}{2 n+1} x^{2 n+1}=x-\frac{1}{3} x^{3}+\frac{1}{5} x^{5}+\cdots+ x \in[-1,1] \\
    \arcsin x&=\sum_{n=0}^{\infty} \frac{(2 n) !}{4^{n}(n !)^{2}(2 n+1)} x^{2n+1}=x+\frac{1}{6} x^{3}+\frac{3}{40} x^{5}+\frac{5}{112} x^{7}+\frac{35}{1152} x^{9}+\cdots+, x \in(-1,1)\\
    \tan x&=\sum_{n=1}^{\infty} \frac{B_{2 n}(-4)^{n}\left(1-4^{n}\right)}{(2 n) !} x^{2 n-1}=x+\frac{1}{3} x^{3}+\frac{2}{15} x^{5}+\frac{17}{315} x^{7}+\frac{62}{2835} x^{9}+\frac{1382}{155925} x^{11}+\frac{21844}{6081075} x^{13}+\frac{929569}{638512875} x^{15}+\cdots,x\in (-\frac{\pi}{2},\frac{\pi}{2})
\end{aligned}$$

* 偶尔会用：
$$\begin{aligned}
f(x, y)= & f(0,0)+\left.\left(x f_{x}+y f_{y}\right)\right|_{(a, b)} \\
& +\left.\frac{1}{2!}\left(x^{2} f_{x x}+2 x y f_{x y}+y^{2} f_{y y}\right)\right|_{(a, b)} \\
& +\frac{1}{3!} x^{3} f_{x x x}+3 x^{2} y f_{x x y}+3 x y^{2} f_{x y y}+\left.y^{3} f_{y y y}\right|_{(a, b)} \\
& +\ldots+\left.\frac{1}{n!}\left(x \frac{\partial}{\partial x}+y \frac{\partial}{\partial y}\right)^{n} f\right|_{(a, b)} \\
& +\left.\frac{1}{(n+1)!}\left(x \frac{\partial}{\partial x}+y \frac{\partial}{\partial y}\right)^{n+1} f\right|_{(c x, c y)}
\end{aligned}$$

* Stolz定理 

$ \frac{*}{\infty}  型 $


定理1 

设  
$\left\{a_{n}\right\}$ 和 $\left\{b_{n}\right\}$
是两个实数列, 其中 $\left\{b_{n}\right\}$ 是严格单调的且趋向于无穷 $ (+\infty  或  -\infty) $ 。若极限 

$$ \lim _{n \rightarrow \infty} \frac{a_{n+1}-a_{n}}{b_{n+1}-b_{n}}=l $$ 

存在, 则 $ \lim_{n \rightarrow \infty}\frac{a_{n}}{b_{n}}=l $ 。

* 立体几何小点：
点到直线的距离: 设 $ P_{0}\left(x_{0}, y_{0}, z_{0}\right) $ 是空间一定点, 过点 $ P_{1}\left(x_{1}, y_{1}, z_{1}\right) $ 且方向向量为 $ s $ 的直线用 $ l\left(P_{1}, s\right) $ 表示; 点PO到的距离用 $ d\left(P_{0}, l\right) $ 表示。

设 $ \theta $ 为向量 $ s $ 与向量 $ P_1, P_O $ 的夹角, 则从图中可以得到有 $ d\left(P_{0}, l\right)=\left|P_{1} P_{0}\right| \sin \theta $
 又因为从外积公式得到  $ |s \times P 1 P 0|=|s||P 1 P 0| \sin \theta $ .
所以 
$$ d\left(P_{0}, l\right)=\frac{\left|s \times P_{1} P_{0}\right|}{|s|} $$ 

## 大题
1. 放缩

* 不等式

* 导数，类似 $ sinx \ge \frac{2}{\pi}x $ 当 $ 0\le x \le  \frac{\pi}{2}$

2. 向题目已知条件构造
3. 构造函数
4. 数列，裂项拆分，重组
5. 夹逼

(莱布尼茨定理) 如果交错级数 $ \sum_{n=1}(-1)^{n-1} u_{n} $ 满足条件:
(1) $ u_{n} \geqslant u_{n+1}(n=1,2,3, \cdots) $ ;
(2) $ \lim _{n \rightarrow \infty} u_{n}=0 $
 
则级数收敛, 且其和 $ s \leqslant u_{1} $ , 其余项 $ r_{n} $ 的绝对值 $ \left|r_{n}\right| \leqslant u_{n+1} $ .