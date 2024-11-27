# 九月练习
## 动态规划：
Partition Equal Subset Sum：

Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.

Example 1:

Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].
Example 2:

Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.

```c++
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum  = 0 ,maxele  = 0;
        for(int i = 0; i < nums.size() ; i++)
        {
            sum += nums[i] ;
            if(nums[i] > maxele)
            {
                maxele = nums[i];
            }
        }
        if(sum % 2 != 0 || maxele > sum/2)
            return false;
        
        vector<vector<int>> dp(nums.size() , vector<int>(sum/2 + 1, 0 )) ;
        for(int i = 0 ; i < nums.size() ; i++)
        {
            dp[i][0] = 1 ;
        }
        dp[0][nums[0]] = 1;
        for(int i = 1 ; i < nums.size() ; i++)
        {
            for(int j = 1 ; j <= sum/2 ; j++)
            {
                if(j <nums[i])
                {
                    dp[i][j] = dp[i-1][j];
                }
                else
                {
                    dp[i][j] = dp[i-1][j] | dp[i-1][j-nums[i]];
                }
            }
        }
        return dp[nums.size()-1][sum/2];
    }
};
```

动态规划的求解步骤：  
1. 分析最优子结构性质（递推关系）
2. 递归定义最优值（动态规划核心）
3. 自底向上的方式计算出最优值（动态规划的执行过程）
4. 根据计算最优值时得到的信息，构造最优解

记忆化搜索是一种“从顶至底”的方法：我们从原问题（根节点）开始，递归地将较大子问题分解为较小子问题，直至解已知的最小子问题（叶节点）。之后，通过回溯逐层收集子问题的解，构建出原问题的解。

与之相反，动态规划是一种“从底至顶”的方法：从最小子问题的解开始，迭代地构建更大子问题的解，直至得到原问题的解。

- 分治算法递归地将原问题划分为多个相互独立的子问题，直至最小子问题，并在回溯中合并子问题的解，最终得到原问题的解。
- 动态规划也对问题进行递归分解，但与分治算法的主要区别是，动态规划中的子问题是相互依赖的，在分解过程中会出现许多重叠子问题。
- 回溯算法在尝试和回退中穷举所有可能的解，并通过剪枝避免不必要的搜索分支。原问题的解由一系列决策步骤构成，我们可以将每个决策步骤之前的子序列看作一个子问题。
实际上，动态规划常用来求解最优化问题，它们不仅包含重叠子问题，还具有另外两大特性：最优子结构、无后效性。

最优子结构的含义：原问题的最优解是从子问题的最优解构建得来的。

无后效性是动态规划能够有效解决问题的重要特性之一，其定义为：给定一个确定的状态，它的未来发展只与当前状态有关，而与过去经历的所有状态无关。
### 记忆化搜索
在记忆化搜索中，当算法需要计算某个子问题的结果时，它首先检查是否已经计算过该问题。如果已经计算过，则直接返回已经存储的结果；否则，计算该问题，并将结果存储下来以备将来使用。


### 0-1 背包问题

比如现在有四个物品，要把这四个物品放入一个容量为8的背包之中，然后现在要求这个背包最大能够放入价值为多少的物品？

我们可以将 0-1 背包问题看作一个由 $n$ 轮决策组成的过程，对于每个物体都有不放入和放入两种决策，因此该问题满足决策树模型。

该问题的目标是求解“在限定背包容量下能放入物品的最大价值”，因此较大概率是一个动态规划问题。

**第一步：思考每轮的决策，定义状态，从而得到 $dp$ 表**

对于每个物品来说，不放入背包，背包容量不变；放入背包，背包容量减小。由此可得状态定义：当前物品编号 $i$ 和背包容量 $c$ ，记为 $[i, c]$ 。

状态 $[i, c]$ 对应的子问题为：**前 $i$ 个物品在容量为 $c$ 的背包中的最大价值**，记为 $dp[i, c]$ 。

待求解的是 $dp[n, cap]$ ，因此需要一个尺寸为 $(n+1) \times (cap+1)$ 的二维 $dp$ 表。

**第二步：找出最优子结构，进而推导出状态转移方程**

当我们做出物品 $i$ 的决策后，剩余的是前 $i-1$ 个物品决策的子问题，可分为以下两种情况。

- **不放入物品 $i$** ：背包容量不变，状态变化为 $[i-1, c]$ 。
- **放入物品 $i$** ：背包容量减少 $wgt[i-1]$ ，价值增加 $val[i-1]$ ，状态变化为 $[i-1, c-wgt[i-1]]$ 。

上述分析向我们揭示了本题的最优子结构：**最大价值 $dp[i, c]$ 等于不放入物品 $i$ 和放入物品 $i$ 两种方案中价值更大的那一个**。由此可推导出状态转移方程：

$$
dp[i, c] = \max(dp[i-1, c], dp[i-1, c - wgt[i-1]] + val[i-1])
$$

需要注意的是，若当前物品重量 $wgt[i - 1]$ 超出剩余背包容量 $c$ ，则只能选择不放入背包。

**第三步：确定边界条件和状态转移顺序**

当无物品或背包容量为 $0$ 时最大价值为 $0$ ，即首列 $dp[i, 0]$ 和首行 $dp[0, c]$ 都等于 $0$ 。

当前状态 $[i, c]$ 从上方的状态 $[i-1, c]$ 和左上方的状态 $[i-1, c-wgt[i-1]]$ 转移而来，因此通过两层循环正序遍历整个 $dp$ 表即可。

```
贪心是 DP 的子集。能贪心的一定可以 DP（但是时间复杂度可能不一样），能 DP 的不一定能贪心。

比如选或不选这个思路，DP 的想法是，枚举每个数选或不选，把所有的情况都枚举到，从中比较哪个最优，本质上是暴力的优化（不重复计算重叠子问题）；贪心的想法是（如果可以贪心），每一步都选最优的，选或不选哪个更好，可以直接算出来，有一种「人为介入」的感觉。
```

## BFS
给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。
两个相邻元素间的距离为 1 。

```c++
class Solution {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) 
    {
        int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int m = mat.size() , n = mat[0].size() ;
        vector<vector<int>> dist(m , vector<int> (n));
        vector<vector<int>> seen(m , vector<int> (n,0));
        queue<pair<int,int>> q;

        for(int i = 0 ; i  < m ; i++)
        {
            for(int j = 0 ;  j < n ; j++)
            {
                if(mat[i][j] == 0)
                {
                    q.emplace(i , j );
                    seen[i][j] = 1;
                }
            }
        }

        while( !q.empty() )
        {
            auto [i , j] = q.front();
            q.pop();
            for(int d = 0 ; d < 4 ; d++)
            {
                int ni = i + dirs[d][0];
                int nj = j + dirs[d][1];
                if(ni >= 0 && ni<m && nj>=0 &&nj <n &&!seen[ni][nj])
                {
                    dist[ni][nj] = dist[i][j]+1;
                    q.emplace(ni,nj);
                    seen[ni][nj] = 1;
                }
            }
        }
        return dist;
    }
};
```

给你一个 m x n 的迷宫矩阵 maze （下标从 0 开始），矩阵中有空格子（用 '.' 表示）和墙（用 '+' 表示）。同时给你迷宫的入口 entrance ，用 entrance = [entrancerow, entrancecol] 表示你一开始所在格子的行和列。

每一步操作，你可以往 上，下，左 或者 右 移动一个格子。你不能进入墙所在的格子，你也不能离开迷宫。你的目标是找到离 entrance 最近 的出口。出口 的含义是 maze 边界 上的 空格子。entrance 格子 不算 出口。

请你返回从 entrance 到最近出口的最短路径的 步数 ，如果不存在这样的路径，请你返回 -1 。

```c++
class Solution {
public:
    int nearestExit(vector<vector<char>>& maze, vector<int>& entrance) {
        int m = maze.size();
        int n = maze[0].size();
        // 上下左右四个相邻坐标对应的行列变化量
        vector<int> dx = {1, 0, -1, 0};
        vector<int> dy = {0, 1, 0, -1};
        queue<tuple<int, int, int>> q;
        // 入口加入队列并修改为墙
        q.emplace(entrance[0], entrance[1], 0);
        maze[entrance[0]][entrance[1]] = '+';
        while (!q.empty()){
            auto [cx, cy, d] = q.front();
            q.pop();
            // 遍历四个方向相邻坐标
            for (int k = 0; k < 4; ++k){
                int nx = cx + dx[k];
                int ny = cy + dy[k];
                // 新坐标合法且不为墙
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == '.'){
                    if (nx == 0 || nx == m - 1 || ny == 0 || ny == n - 1){
                        // 新坐标为出口，返回距离作为答案
                        return d + 1;
                    }
                    // 新坐标为空格子且不为出口，修改为墙并加入队列
                    maze[nx][ny] = '+';
                    q.emplace(nx, ny, d + 1);
                }
            }
        }
        // 不存在到出口的路径，返回 -1
        return -1;
    }
};


```

## 单调栈

```
单调栈（Monotone Stack）：一种特殊的栈。在栈的「先进后出」规则基础上，要求「从 栈顶 到 栈底 的元素是单调递增（或者单调递减）」。

所以单调栈一般用于解决一下几种问题：

寻找左侧第一个比当前元素大的元素。

寻找左侧第一个比当前元素小的元素。

寻找右侧第一个比当前元素大的元素。

寻找右侧第一个比当前元素小的元素。
```

给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。

```c++
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int size = temperatures.size();
        vector<int> ans(size);
        stack<int> st;
        for(int i = 0 ; i < size ; i++)
        {
            while(!st.empty() && temperatures[i] >temperatures[st.top()] )
            {
                ans[st.top()] = i - st.top();
                st.pop();
            }
            st.push(i);
        }
        return ans;
    }
};
```

模板：
``` py

def monotoneIncreasingStack(nums):
    stack = []
    for num in nums:
        while stack and num >= stack[-1]:
            stack.pop()
        stack.append(num)

or:

def monotoneDecreasingStack(nums):
    stack = []
    for num in nums:
        while stack and num <= stack[-1]:
            stack.pop()
        stack.append(num)


```

## 网格图（DFS/BFS/综合应用）

```
让我们一起来玩扫雷游戏！

给你一个大小为 m x n 二维字符矩阵 board ，表示扫雷游戏的盘面，其中：

'M' 代表一个 未挖出的 地雷，
'E' 代表一个 未挖出的 空方块，
'B' 代表没有相邻（上，下，左，右，和所有4个对角线）地雷的 已挖出的 空白方块，
数字（'1' 到 '8'）表示有多少地雷与这块 已挖出的 方块相邻，
'X' 则表示一个 已挖出的 地雷。
给你一个整数数组 click ，其中 click = [clickr, clickc] 表示在所有 未挖出的 方块（'M' 或者 'E'）中的下一个点击位置（clickr 是行下标，clickc 是列下标）。

根据以下规则，返回相应位置被点击后对应的盘面：

如果一个地雷（'M'）被挖出，游戏就结束了- 把它改为 'X' 。
如果一个 没有相邻地雷 的空方块（'E'）被挖出，修改它为（'B'），并且所有和其相邻的 未挖出 方块都应该被递归地揭露。
如果一个 至少与一个地雷相邻 的空方块（'E'）被挖出，修改它为数字（'1' 到 '8' ），表示相邻地雷的数量。
如果在此次点击中，若无更多方块可被揭露，则返回盘面。

```

**DFS**
```c++

class Solution
{
public:
    int dirx[8] = {0, 1, 0, -1, 1, 1, -1, -1};
    int diry[8] = {1, 0, -1, 0, 1, -1, 1, -1};

    void DFS(vector<vector<char>>& board , int x, int y)
    {
        int count = 0 ;
        for(int i = 0 ; i < 8 ; i++)
        {
            int tx = x + dirx[i];
            int ty = y + diry[i];
            if(tx < 0 || tx >= board.size() || ty < 0 || ty >= board[0].size() )
            {
                continue;
            }
            count += board[tx][ty] == 'M';
        }
        if(count)
        {
            board[x][y] = count + '0';
        }
        else
        {
            board[x][y] = 'B';
            for(int i = 0 ; i < 8 ; i++)
            {
                int tx = x + dirx[i];
                int ty = y + diry[i];
                if(tx < 0 || tx >= board.size() || ty < 0 || ty >= board[0].size() )
                {
                    continue;
                }
                DFS(board, x , y);
            }
        }
    }
    vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click) {
        int m = board.size() , n = board[0].size();
        if(board[click[0]] [click[1]] == 'M')
        {
            board[click[0]][click[1]] = 'X';
        }
        else if(board[click[0]][click[1]] == 'E')
        {
            DFS(board , click[0], click[1]);
        }
        return board;
    }
}

```

**BFS**
```c++
class Solution {
public:
    int dir_x[8] = {0, 1, 0, -1, 1, 1, -1, -1};
    int dir_y[8] = {1, 0, -1, 0, 1, -1, 1, -1};

    void bfs(vector<vector<char>>& board, int sx, int sy) {
        queue<pair<int, int>> Q;
        //队列是一种先进先出（FIFO, First In First Out）的数据结构，它允许在一端添加元素（称为队尾），并在另一端移除元素（称为队首）。
        vector<vector<int>> vis(board.size(), vector<int>(board[0].size(), 0));
        Q.push({sx, sy});
        vis[sx][sy] = true;
        while (!Q.empty()) {
            auto pos = Q.front();
            Q.pop();
            int cnt = 0, x = pos.first, y = pos.second;
            for (int i = 0; i < 8; ++i) {
                int tx = x + dir_x[i];
                int ty = y + dir_y[i];
                if (tx < 0 || tx >= board.size() || ty < 0 || ty >= board[0].size()) {
                    continue;
                }
                // 不用判断 M，因为如果有 M 的话游戏已经结束了
                cnt += board[tx][ty] == 'M';
            }
            if (cnt > 0) {
                // 规则 3
                board[x][y] = cnt + '0';
            } else {
                // 规则 2
                board[x][y] = 'B';
                for (int i = 0; i < 8; ++i) {
                    int tx = x + dir_x[i];
                    int ty = y + dir_y[i];
                    // 这里不需要在存在 B 的时候继续扩展，因为 B 之前被点击的时候已经被扩展过了
                    if (tx < 0 || tx >= board.size() || ty < 0 || ty >= board[0].size() || board[tx][ty] != 'E' || vis[tx][ty]) {
                        continue;
                    }
                    Q.push(make_pair(tx, ty));
                    vis[tx][ty] = true;
                }
            }
        }
    }

    vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click) {
        int x = click[0], y = click[1];
        if (board[x][y] == 'M') {
            // 规则 1
            board[x][y] = 'X';
        } else {
            bfs(board, x, y);
        }
        return board;
    }
};
```

你现在手里有一份大小为 n x n 的 网格 grid，上面的每个 单元格 都用 0 和 1 标记好了。其中 0 代表海洋，1 代表陆地。

请你找出一个海洋单元格，这个海洋单元格到离它最近的陆地单元格的距离是最大的，并返回该距离。如果网格上只有陆地或者海洋，请返回 -1。

我们这里说的距离是「曼哈顿距离」（ Manhattan Distance）：(x0, y0) 和 (x1, y1) 这两个单元格之间的距离是 |x0 - x1| + |y0 - y1| 。

```py
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        start = []
        for i in range(rows): # 将所有起点存入 start 数组
            for j in range(cols):
                if grid[i][j] == 1:
                    start.append((i, j, 0))
        
        if len(start) == 0 or len(start) == rows * cols: # 特判
            return -1

        queue = collections.deque(start) # 队列初始化
        dr = [0, 1, 0, -1] # 建立方向数组
        dc = [1, 0, -1, 0]
        while queue:
            i, j, dis = queue.popleft()
            for d in range(4): # 四个方向
                x = i + dr[d]
                y = j + dc[d]
                if x < 0 or y < 0 or x == rows or y == cols or grid[x][y] == 1: 
                    continue
                queue.append((x, y, dis + 1))
                grid[x][y] = 1 # 访问过的位置标记为 1
                
        return dis 
```

多源最短路 Dijkstra:
```c++
class Solution {
public:
    int maxDistance(vector<vector<int>>& g) {
        int n = g.size();
        int inf = INT_MAX - 1;
        vector<vector<int>> dist(n, vector<int>(n, inf));
        queue<pair<int, int>> q;
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                if(g[i][j]){
                    dist[i][j] = 0;
                    q.emplace(i, j);
                }
            }
        }
        if(q.size() == n*n || q.size() == 0)
            return -1;
        const int dirs[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        while(!q.empty()){
            auto [x, y] = q.front();
            q.pop();
            for(int i=0; i<4; i++){
                int nx = x + dirs[i][0], ny = y + dirs[i][1];
                if(nx < 0 || ny < 0 || nx >= n || ny >= n)
                    continue;
                if(dist[nx][ny] > dist[x][y] + 1){
                    dist[nx][ny] = dist[x][y] + 1;
                    q.emplace(nx, ny);
                }
            }
        }
        int ans = 0;
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                ans = max(ans, dist[i][j]);
            }
        }
        return ans;
    }
};
```