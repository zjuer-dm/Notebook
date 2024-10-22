# 十月练习
## 拓扑排序
```
树是一个无向图，其中任何两个顶点只通过一条路径连接。 换句话说，任何一个没有简单环路的连通图都是一棵树。

给你一棵包含 n 个节点的树，标记为 0 到 n - 1 。给定数字 n 和一个有 n - 1 条无向边的 edges 列表（每一个边都是一对标签），其中 edges[i] = [ai, bi] 表示树中节点 ai 和 bi 之间存在一条无向边。

可选择树中任何一个节点作为根。当选择节点 x 作为根节点时，设结果树的高度为 h 。在所有可能的树中，具有最小高度的树（即，min(h)）被称为 最小高度树 。

请你找到所有的 最小高度树 并按 任意顺序 返回它们的根节点标签列表。

树的 高度 是指根节点和叶子节点之间最长向下路径上边的数量。

```

首先找到所有度为 1 的节点压入队列，此时令节点剩余计数 remainNodes=n；

同时将当前 remainNodes 计数减去出度为 1 的节点数目，将最外层的度为 1 的叶子节点取出，并将与之相邻的节点的度减少，重复上述步骤将当前节点中度为 1 的节点压入队列中；

重复上述步骤，直到剩余的节点数组 remainNodes≤2 时，此时剩余的节点即为当前高度最小树的根节点。

```py
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]

        g = [[] for _ in range(n)]
        deg = [0] * n
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
            deg[x] += 1
            deg[y] += 1

        q = [i for i, d in enumerate(deg) if d == 1]
        remainNodes = n
        while remainNodes > 2:
            remainNodes -= len(q)
            tmp = q
            q = []
            for x in tmp:
                for y in g[x]:
                    deg[y] -= 1
                    if deg[y] == 1:
                        q.append(y)
        return q
```

```
2603

给你一个 n 个节点的无向无根树，节点编号从 0 到 n - 1 。给你整数 n 和一个长度为 n - 1 的二维整数数组 edges ，其中 edges[i] = [ai, bi] 表示树中节点 ai 和 bi 之间有一条边。再给你一个长度为 n 的数组 coins ，其中 coins[i] 可能为 0 也可能为 1 ，1 表示节点 i 处有一个金币。

一开始，你需要选择树中任意一个节点出发。你可以执行下述操作任意次：

收集距离当前节点距离为 2 以内的所有金币，或者
移动到树中一个相邻节点。
你需要收集树中所有的金币，并且回到出发节点，请你返回最少经过的边数。

如果你多次经过一条边，每一次经过都会给答案加一。

```


提示 1

定义一个点的度数为其邻居个数。如果一个点的度数为 1，那么这个点叫做叶子节点，例如示例 2 的 3,4,6,7 都是叶子节点。

如果叶子节点没有金币，我们有必要移动到叶子节点吗？没有必要。

那么可以先把这些没有金币的叶子节点去掉。如果去掉后又产生了新的没有金币的叶子节点，就继续去掉。

怎么实现？拓扑排序。一开始，把没有金币的叶子节点都加到队列中。然后不断循环直到队列为空。每次循环，弹出队首的节点 x，并删除 x 及其邻居之间的边。我们并不需要实际删除边，只需要把邻居的度数减少 1。如果一个邻居的度数减少为 1 且没有金币，就加到队列中，继续拓扑排序。

提示 2

看示例 2，在去掉节点 6 之后，现在每个叶子节点上都有金币。

由于可以「收集距离当前节点距离为 2 以内的所有金币」，我们没有必要移动到叶子节点再收集，而是移动到叶子节点的父节点的父节点，就能收集到叶子节点上的金币。

那么，去掉所有叶子，然后再去掉新产生的叶子，剩余节点就是必须要访问的节点。

提示 3

由于题目要求最后回到出发点，无论从哪个点出发，每条边都必须走两次。这是因为把出发点作为树根，递归遍历这棵树，那么往下「递」是一次，往上「归」又是一次，每条边都会经过两次。

所以答案就是剩余边数乘 2。当我们删除节点时，也可以看成是删除这个点到其父节点的边。

特别地，如果所有点都要被删除，那么当剩下两个点时，这两个点之间的边我们会删除两次，这会导致剩余边数等于 −1，而此时答案应该是 0。所以最后答案要和 0 取最大值。

代码实现时，由于我们不需要得到一个严格的拓扑序，所以简单地用栈或者数组代替队列，也是可以的。

## 区间 DP

```
给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。
```

```py
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        f = [[0]*n for _ in range(n)]

        for i in range(n-1,-1,-1):
            f[i][i] =1 
            for j in range(i+1,n):
                if s[i]==s[j]:
                    f[i][j] = f[i+1][j-1] + 2
                else:
                    f[i][j] = max(f[i][j-1] , f[i+1][j])
        
        return f[0][n-1]
                    

        # @cache
        # def DFS(i,j):
        #     if i == j:
        #         return 1
        #     if i > j:
        #         return 0
        #     if s[i]==s[j]:
        #         return DFS(i+1 , j - 1) +2 
        #     return max( DFS(i,j-1) , DFS(i+1, j))

        # return DFS(0 , n-1)
```


## 前缀和
### 一维前缀和

```
An array is considered special if every pair of its adjacent elements contains two numbers with different parity.

You are given an array of integer nums and a 2D integer matrix queries, where for queries[i] = [fromi, toi] your task is to check that 
subarray
 nums[fromi..toi] is special or not.

Return an array of booleans answer such that answer[i] is true if nums[fromi..toi] is special.


```

```c++
class Solution {
public:
    vector<bool> isArraySpecial(vector<int>& nums, vector<vector<int>>& queries) {
        vector<int> s(nums.size());
        for(int i = 1  ; i  < nums.size() ; i++)
        {
            s[i] = s[i-1] + (nums[i-1]%2 == nums[i]%2);

        }
        vector<bool> ans(queries.size());
        for(int i = 0 ; i < queries.size() ; i++)
        {
            auto &q = queries[i];
            ans[i] = s[q[1]]==s[q[0]];
        }
        return ans;
    }
};
```

### 二维前缀和
模板：
```py
class MatrixSum:
    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        s = [[0] * (n + 1) for _ in range(m + 1)]
        for i, row in enumerate(matrix):
            for j, x in enumerate(row):
                s[i + 1][j + 1] = s[i + 1][j] + s[i][j + 1] - s[i][j] + x
        self.s = s

    # 返回左上角在 (r1,c1) 右下角在 (r2-1,c2-1) 的子矩阵元素和（类似前缀和的左闭右开）
    def query(self, r1: int, c1: int, r2: int, c2: int) -> int:
        return self.s[r2][c2] - self.s[r2][c1] - self.s[r1][c2] + self.s[r1][c1]

    # 如果你不习惯左闭右开，也可以这样写
    # 返回左上角在 (r1,c1) 右下角在 (r2,c2) 的子矩阵元素和
    def query2(self, r1: int, c1: int, r2: int, c2: int) -> int:
        return self.s[r2 + 1][c2 + 1] - self.s[r2 + 1][c1] - self.s[r1][c2 + 1] + self.s[r1][c1]
```

```
Given a m x n matrix mat and an integer k, return a matrix answer where each answer[i][j] is the sum of all elements mat[r][c] for:

i - k <= r <= i + k,

j - k <= c <= j + k, and

(r, c) is a valid position in the matrix.
```

```c++
class Solution {
public:
    
    vector<vector<int>> matrixBlockSum(vector<vector<int>>& mat, int k) {
        int m = mat.size() , n = mat[0].size();
        vector<vector<int>> s(m+1,vector<int>(n+1));

        for(int i = 0 ; i < m; i++)
        {
            for(int j = 0 ; j < n ; j++)
            {
                s[i+1][j+1] = s[i][j+1] +s[i+1][j] -s[i][j] +mat[i][j];
            }
        }

        vector<vector<int>> ans(m , vector<int>(n));

        for(int i = 0 ; i < m ; i++)
        {
            for(int j = 0; j < n ; j++)
            {
                int x1 = max(min(i + k + 1,m),0);
                int x2 = max(min(i-k, m) , 0);
                int y1 = max(min(j + k + 1,n),0);
                int y2 = max(min(j-k, n) , 0);
                ans[i][j] = s[x1][y1] - s[x2][y1] - s[x1][y2] + s[x2][y2];
            }
        }
        return ans;
    }
};
```

```
给你一个正整数数组 nums 。

同时给你一个长度为 m 的整数数组 queries 。第 i 个查询中，你需要将 nums 中所有元素变成 queries[i] 。你可以执行以下操作 任意 次：

将数组里一个元素 增大 或者 减小 1 。
请你返回一个长度为 m 的数组 answer ，其中 answer[i]是将 nums 中所有元素变成 queries[i] 的 最少 操作次数。

注意，每次查询后，数组变回最开始的值。
```

```c++
class Solution {
public:
    vector<long long> minOperations(vector<int>& nums, vector<int>& queries) {
        ranges::sort(nums);
        int n = nums.size();
        vector<long long> sum(n + 1); // 前缀和
        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + nums[i];
        }

        int m = queries.size();
        vector<long long> ans(m);
        for (int i = 0; i < m; i++) {
            int q = queries[i];
            long long j = ranges::lower_bound(nums, q) - nums.begin();//二分查找
            long long left = q * j - sum[j]; // 蓝色面积
            long long right = sum[n] - sum[j] - q * (n - j); // 绿色面积
            ans[i] = left + right;
        }
        return ans;
    }
};

```

## 差分
对于数组 `a`,定义其差分数组(difference array)为:

$$
d[i] = \begin{cases} 
a[0], & \text{if } i = 0 \\
a[i] - a[i-1], & \text{if } i \geq 1 
\end{cases}
$$

 性质 1

从左到右累加 `d` 中的元素，可以得到数组 `a`。

性质 2

如下两个操作是等价的：
- 把 `a` 的子数组 `a[i], a[i+1], ⋯, a[j]` 都加上 `x`。
- 把 `d[i]` 增加 `x`，把 `d[j+1]` 减少 `x`。

操作步骤

利用性质 2，我们只需要 $O(1)$ 的时间就可以完成对 `a` 的子数组的操作。最后利用性质 1 从差分数组复原出数组 `a`。

**注**：也可以这样理解，`d[i]` 表示把下标 $≥i$ 的数都加上 `d[i]`。

### 一维差分

```
车上最初有 capacity 个空座位。车 只能 向一个方向行驶（也就是说，不允许掉头或改变方向）

给定整数 capacity 和一个数组 trips ,  trip[i] = [numPassengersi, fromi, toi] 表示第 i 次旅行有 numPassengersi 乘客，接他们和放
他们的位置分别是 fromi 和 toi 。这些位置是从汽车的初始位置向东的公里数。

当且仅当你可以在所有给定的行程中接送所有乘客时，返回 true，否则请返回 false。
```

```c++
class Solution {
public:
    bool carPooling(vector<vector<int>> &trips, int capacity) {
        int d[1001]{};
        for (auto &t : trips) {
            int num = t[0], from = t[1], to = t[2];
            d[from] += num;
            d[to] -= num;
        }
        int s = 0;
        for (int v : d) {
            s += v;
            if (s > capacity) {
                return false;
            }
        }
        return true;
    }
};

```
### 二维差分
```
给你一个 m x n 的二进制矩阵 grid ，每个格子要么为 0 （空）要么为 1 （被占据）。

给你邮票的尺寸为 stampHeight x stampWidth 。我们想将邮票贴进二进制矩阵中，且满足以下 限制 和 要求 ：

覆盖所有 空 格子。
不覆盖任何 被占据 的格子。
我们可以放入任意数目的邮票。
邮票可以相互有 重叠 部分。
邮票不允许 旋转 。
邮票必须完全在矩阵 内 。
如果在满足上述要求的前提下，可以放入邮票，请返回 true ，否则返回 false 。


```
```c++
class Solution {
public:
    bool possibleToStamp(vector<vector<int>> &grid, int stampHeight, int stampWidth) {
        int m = grid.size(), n = grid[0].size();

        // 1. 计算 grid 的二维前缀和
        vector<vector<int>> s(m + 1, vector<int>(n + 1));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                s[i + 1][j + 1] = s[i + 1][j] + s[i][j + 1] - s[i][j] + grid[i][j];
            }
        }

        // 2. 计算二维差分
        // 为方便第 3 步的计算，在 d 数组的最上面和最左边各加了一行（列），所以下标要 +1
        vector<vector<int>> d(m + 2, vector<int>(n + 2));
        for (int i2 = stampHeight; i2 <= m; i2++) {
            for (int j2 = stampWidth; j2 <= n; j2++) {
                int i1 = i2 - stampHeight + 1;
                int j1 = j2 - stampWidth + 1;
                if (s[i2][j2] - s[i2][j1 - 1] - s[i1 - 1][j2] + s[i1 - 1][j1 - 1] == 0) {
                    d[i1][j1]++;
                    d[i1][j2 + 1]--;
                    d[i2 + 1][j1]--;
                    d[i2 + 1][j2 + 1]++;
                }
            }
        }

        // 3. 还原二维差分矩阵对应的计数矩阵（原地计算）
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                d[i + 1][j + 1] += d[i + 1][j] + d[i][j + 1] - d[i][j];
                if (grid[i][j] == 0 && d[i + 1][j + 1] == 0) {
                    return false;
                }
            }
        }
        return true;
    }
};

```

## BFS
```
有 n 个人，每个人都有一个  0 到 n-1 的唯一 id 。

给你数组 watchedVideos  和 friends ，其中 watchedVideos[i]  和 friends[i] 分别表示 id = i 的人观看过的视频列表和他的好友列表。

Level 1 的视频包含所有你好友观看过的视频，level 2 的视频包含所有你好友的好友观看过的视频，以此类推。一般的，Level 为 k 的视频包含所有从你出发，最短距离为 k 的好友观看过的视频。

给定你的 id  和一个 level 值，请你找出所有指定 level 的视频，并将它们按观看频率升序返回。如果有频率相同的视频，请将它们按字母顺序从小到大排列。


```

```c++
class Solution {
public:
    vector<string> watchedVideosByFriends(vector<vector<string>>& wv,
                                          vector<vector<int>>& fs, 
                                          int id, 
                                          int level) {
        int n = wv.size();
        vector<int> dis(n, 1e9);
        dis[id] = 0;
        queue<int> q;
        q.push(id);
        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (auto& v: fs[u]) {
                if (dis[v] > dis[u] + 1) {
                    dis[v] = dis[u] + 1;
                    q.push(v); 
                }
            }
        }

        unordered_map<string, int> heap;
        for (int i = 0; i < n; i ++ ) {
            if (dis[i] == level) {
                for (auto& v: wv[i]) {
                    heap[v] ++ ;
                }
            }
        }

        vector<pair<int, string>> vp;
        for (auto& [k, v]: heap) vp.push_back({v, k});
        sort(vp.begin(), vp.end());

        vector<string> res;
        for (int i = 0; i < vp.size(); i ++ ) res.push_back(vp[i].second);

        return res;
    }
};
```
### 0-1 BFS
```
给你一个 m x n 的网格图 grid 。 grid 中每个格子都有一个数字，对应着从该格子出发下一步走的方向。 grid[i][j] 中的数字可能为以下几种情况：

1 ，下一步往右走，也就是你会从 grid[i][j] 走到 grid[i][j + 1]
2 ，下一步往左走，也就是你会从 grid[i][j] 走到 grid[i][j - 1]
3 ，下一步往下走，也就是你会从 grid[i][j] 走到 grid[i + 1][j]
4 ，下一步往上走，也就是你会从 grid[i][j] 走到 grid[i - 1][j]
注意网格图中可能会有 无效数字 ，因为它们可能指向 grid 以外的区域。

一开始，你会从最左上角的格子 (0,0) 出发。我们定义一条 有效路径 为从格子 (0,0) 出发，每一步都顺着数字对应方向走，最终在最右下角的格子 (m - 1, n - 1) 结束的路径。有效路径 不需要是最短路径 。

你可以花费 cost = 1 的代价修改一个格子中的数字，但每个格子中的数字 只能修改一次 。

请你返回让网格图至少有一条有效路径的最小代价。
```

```c++
class Solution {
private:
    static constexpr int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
public:
    int minCost(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<int> dist(m*n, INT_MAX);
        vector<int> seen(m*n,0);
        dist[0] = 0;
        deque<int> q;
        q.push_back(0);
        while(!q.empty())
        {
            auto cur = q.front();
            q.pop_front();
            if(seen[cur])
            {
                continue;
            }
            seen[cur] = 1;
            int x = cur/n;
            int y = cur%n;
            for(int i = 0 ; i < 4 ; i++)
            {
                int nx = x + dirs[i][0];
                int ny = y + dirs[i][1];
                int new_pos = nx * n + ny ; 
                int new_dis = dist[cur] + (grid[x][y]!=i+1);
                if(nx>=0&&ny>=0&&nx<m&&ny<n&&new_dis<dist[new_pos])
                {
                    dist[new_pos] = new_dis;
                    if(grid[x][y] == i+1)
                    {
                        q.push_front(new_pos);
                    }
                    else
                    {
                        q.push_back(new_pos);
                    }
                }
            }
        }
        return dist[m*n-1];
    }
};
```

## DFS
```
有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。

返回矩阵中 省份 的数量。
```

```c++
class Solution {
public:
    void DFS(vector<vector<int>>& isConnected , int city , int n,vector<int>& visited)
    {
        for(int i = 0 ; i < n ; i ++)
        {
            if(isConnected[city][i] && !visited[i])
            {
                visited[i] = 1;
                DFS(isConnected , i , n, visited);
            }
        }
    }
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        vector<int> visited(n);
        int prov = 0; 
        for(int i = 0 ; i < n ; i++)
        {
            if(!visited[i])
            {
                DFS(isConnected , i , n, visited);
                prov ++;
            }
        }
        return prov ;
    }
};
```

```
用以太网线缆将 n 台计算机连接成一个网络，计算机的编号从 0 到 n-1。线缆用 connections 表示，其中 connections[i] = [a, b] 连接了计算机 a 和 b。

网络中的任何一台计算机都可以通过网络直接或者间接访问同一个网络中其他任意一台计算机。

给你这个计算机网络的初始布线 connections，你可以拔开任意两台直连计算机之间的线缆，并用它连接一对未直连的计算机。请你计算并返回使所有计算机都连通所需的最少操作次数。如果不可能，则返回 -1 。 

```

```c++
class Solution {
private:
    vector<vector<int>> edges;
    vector<int> used;

public:
    void dfs(int u) {
        used[u] = true;
        for (int v: edges[u]) {
            if (!used[v]) {
                dfs(v);
            }
        }
    }
    
    int makeConnected(int n, vector<vector<int>>& connections) {
        if (connections.size() < n - 1) {
            return -1;
        }

        edges.resize(n);
        for (const auto& conn: connections) {
            edges[conn[0]].push_back(conn[1]);
            edges[conn[1]].push_back(conn[0]);
        }
        
        used.resize(n);
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            if (!used[i]) {
                dfs(i);
                ++ans;
            }
        }
        
        return ans - 1;
    }
};

```

## 并查集

主要用于解决一些元素分组的问题。它管理一系列不相交的集合，并支持两种操作：

* 合并（Union）：把两个不相交的集合合并为一个集合。
* 查询（Find）：查询两个元素是否在同一个集合中。


```
给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 equations[i] 的长度为 4，并采用两种不同的形式之一："a==b" 或 "a!=b"。在这里，a 和 b 是小写字母（不一定不同），表示单字母变量名。

只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 true，否则返回 false。 
```

```c++
class UnionFind{
private: 
    vector<int> parent;

public:
    UnionFind()
    {
        parent.resize(26);
        iota(parent.begin(),parent.end(),0);
    }
    int find(int index)
    {
        if(index == parent[index])
        {
            return index;
        }
        parent[index] = find(parent[index]);
        return parent[index];
    }
    void merge(int index1 , int index2)
    {
        parent[find(index1)] = find(index2);
    }
};
class Solution {
public:
    bool equationsPossible(vector<string>& equations) {
        UnionFind uf;
        for(auto & x : equations)
        {
            if(x[1] == '=')
            {
                int index1 = x[0] - 'a';
                int index2 = x[3] - 'a';
                uf.merge(index1 , index2);
            }
        }
        for(auto & x : equations)
        {
            if(x[1] == '!')
            {
                int index1 = x[0] - 'a';
                int index2 = x[3] - 'a';
                if(uf.find(index1) == uf.find(index2))
                {
                    return false;
                }
            }
        }
        return true;
    }
};
```

## 堆

堆通常是一个可以被看做一棵完全二叉树的数组对象。

堆满足下列性质：

* 堆中某个节点的值总是不大于或不小于其父节点的值。
* 堆总是一棵完全二叉树。

优先队列的主要特点如下：

1. 优先级决定出队顺序：在优先队列中，优先级高的元素会比优先级低的元素先出队，而不是按照加入队列的时间顺序。例如，假设一个任务的优先级为高，那么这个任务会先于其他低优先级的任务被处理。

2. 插入操作：优先队列允许插入元素，每个元素带有一个优先级。插入时可以按照队列的规则将元素插入合适的位置。

3. 出队操作：出队时，优先队列会选择优先级最高的元素进行出队。对于同样优先级的元素，一般会按照插入的顺序进行处理。

```
有一个无限大的二维平面。

给你一个正整数 k ，同时给你一个二维数组 queries ，包含一系列查询：

queries[i] = [x, y] ：在平面上坐标 (x, y) 处建一个障碍物，数据保证之前的查询 不会 在这个坐标处建立任何障碍物。
每次查询后，你需要找到离原点第 k 近 障碍物到原点的 距离 。

请你返回一个整数数组 results ，其中 results[i] 表示建立第 i 个障碍物以后，离原地第 k 近障碍物距离原点的距离。如果少于 k 个障碍物，results[i] == -1 。

注意，一开始 没有 任何障碍物。

坐标在 (x, y) 处的点距离原点的距离定义为 |x| + |y| 。
```

```c++
class Solution {
public:
    vector<int> resultsArray(vector<vector<int>>& queries, int k) {
        int size = queries.size();
        priority_queue<int> pq;
        vector<int> ans(size , -1);
        for(int i = 0 ; i < size ; i++)
        {
            pq.push(abs(queries[i][0])+abs(queries[i][1]));
            if(pq.size()>k)
            {
                pq.pop();
            }
            if(pq.size() == k)
            {
                ans[i] = pq.top();
            }
        }
        return ans;
    }
};
```

给你一个正整数数组 nums 。每一次操作中，你可以从 nums 中选择 任意 一个数并将它减小到 恰好 一半。（注意，在后续操作中你可以对减半过的数继续执行操作）

请你返回将 nums 数组和 至少 减少一半的 最少 操作数。

```c++
class Solution {
public:
    int halveArray(vector<int>& nums) {
        int op = 0 ; 
        priority_queue<double> pq(nums.begin(),nums.end());
        double sum = accumulate(nums.begin(),nums.end(),0.0) , sum2 = 0.0 ;
        while(sum2 < sum/2)
        {
            double temp = pq.top();
            sum2 += temp / 2 ;
            pq.pop();
            pq.push(temp/2);
            op++;
        }
        return op;
    }
};
```

## 最小生成树
最小生成树问题是图论中的经典问题，它在现实世界中有着广泛的应用，例如通信网络规划、电力传输网络规划等。在最小生成树问题中，我们需要找到一个连通图的子图，该子图包含了图中的所有节点，并且边的权重之和最小。
### Prim
 Prim 算法是一种用于寻找最小生成树的贪心算法。它从一个起始节点开始，逐步扩展生成树，直到包含图中的所有节点为止。算法维护一个候选边集合，每次从中选择一条最小权重的边，并将连接的节点加入生成树中。
```py
import heapq

def prim(graph, start):
    min_spanning_tree = []
    visited = set()
    priority_queue = [(0, start)]

    while priority_queue:
        weight, node = heapq.heappop(priority_queue)
        if node not in visited:
            visited.add(node)
            min_spanning_tree.append((weight, node))

            for neighbor, neighbor_weight in graph[node].items():
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (neighbor_weight, neighbor))

    return min_spanning_tree
```

### kruskal
 Kruskal 算法是一种用于寻找最小生成树的贪心算法。它将图中的所有边按照权重从小到大排序，然后依次将边加入生成树中，直到生成树包含了图中的所有节点。
```py
def find(parent, node):
    if parent[node] != node:
        parent[node] = find(parent, parent[node])
    return parent[node]

def union(parent, rank, node1, node2):
    root1 = find(parent, node1)
    root2 = find(parent, node2)

    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        elif rank[root1] < rank[root2]:
            parent[root1] = root2
        else:
            parent[root2] = root1
            rank[root1] += 1

def kruskal(graph):
    min_spanning_tree = []
    edges = []

    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            edges.append((weight, node, neighbor))

    edges.sort()

    parent = {node: node for node in graph}
    rank = {node: 0 for node in graph}

    for edge in edges:
        weight, node1, node2 = edge
        if find(parent, node1) != find(parent, node2):
            union(parent, rank, node1, node2)
            min_spanning_tree.append((weight, node1, node2))

    return min_spanning_tree
```

补充：堆
```c++
#include <iostream>
#include <queue>
#include <vector>

struct compare {
    bool operator()(int a, int b) {
        return a > b; // 定义最小堆
    }
};

int main() {
    // 创建一个自定义类型的优先队列，使用最小堆
    std::priority_queue<int, std::vector<int>, compare> pq_min;

    // 向优先队列中添加元素
    pq_min.push(30);
    pq_min.push(10);
    pq_min.push(50);
    pq_min.push(20);

    // 输出队列中的元素
    std::cout << "最小堆中的元素：" << std::endl;
    while (!pq_min.empty()) {
        std::cout << pq_min.top() << std::endl;
        pq_min.pop();
    }

    return 0;
}
```

## 栈
```
给你一个字符串 s 。它可能包含任意数量的 '*' 字符。你的任务是删除所有的 '*' 字符。

当字符串还存在至少一个 '*' 字符时，你可以执行以下操作：

删除最左边的 '*' 字符，同时删除该星号字符左边一个字典序 最小 的字符。如果有多个字典序最小的字符，你可以删除它们中的任意一个。
请你返回删除所有 '*' 字符以后，剩余字符连接而成的 
字典序最小的字符串。
```

```c++
class Solution {
public:
    string clearStars(string s) {
        vector<int> st[26];
        for (int i = 0; i < s.size(); i++) {
            if (s[i] != '*') {
                st[s[i] - 'a'].push_back(i);
                continue;
            }
            for (auto& p : st) {
                if (!p.empty()) {
                    p.pop_back();
                    break;
                }
            }
        }

        vector<int> idx;
        for (auto& p : st) {
            idx.insert(idx.end(), p.begin(), p.end());
        }
        ranges::sort(idx);

        string t(idx.size(), 0);
        for (int i = 0; i < idx.size(); i++) {
            t[i] = s[idx[i]];
        }
        return t;
    }
};

```

## 树

### 二叉树

```
给你一个二维整数数组 descriptions ，其中 descriptions[i] = [parenti, childi, isLefti] 表示 parenti 是 childi 在 二叉树 中的 父节点，二叉树中各节点的值 互不相同 。此外：

如果 isLefti == 1 ，那么 childi 就是 parenti 的左子节点。
如果 isLefti == 0 ，那么 childi 就是 parenti 的右子节点。
请你根据 descriptions 的描述来构造二叉树并返回其 根节点 。

测试用例会保证可以构造出 有效 的二叉树。

```
由于数组 descriptions 中用节点的数值表示对应节点，因此为了方便查找，我们用哈希表 nodes 来维护数值到对应节点的映射。

我们可以遍历数组 descriptions 来创建二叉树。具体地，当我们遍历到三元组 [p,c,left] 时，我们首先判断 nodes 中是否存在 p 与 c 对应的树节点，如果没有则我们新建一个数值为对应值的节点。随后，我们根据 left 的真假将 p 对应的节点的左或右子节点设为 c 对应的节点。当遍历完成后，我们就重建出了目标二叉树。

除此之外，我们还需要寻找二叉树的根节点。这个过程也可以在遍历和建树的过程中完成。我们可以同样用一个哈希表 isRoot 维护数值与是否为根节点的映射。在遍历时，我们需要将 isRoot[c] 设为 false（因为该节点有父节点）；而如果 p 在 isRoot 中不存在，则说明 p 暂时没有父节点，我们可以将 isRoot[c] 设为 true。最终在遍历完成后，一定有且仅有一个元素 root 在 isRoot 中的数值为 true，此时对应的 node[i] 为二叉树的根节点，我们返回该节点作为答案。


```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* createBinaryTree(vector<vector<int>>& descriptions) {
        unordered_map<int , bool> isRoot;
        unordered_map<int , TreeNode *> nodes;
        for(auto & x : descriptions)
        {
            int p = x[0] ; 
            int c = x[1] ;
            int left = x[2] ;
            if(!isRoot.count(p))
            {
                isRoot[p] = true;
            }
            isRoot[c] = false;
            if(!nodes.count(p))
            {
                nodes[p] = new TreeNode(p);
            }
            if(!nodes.count(c))
            {
                nodes[c] = new TreeNode(c);
            }
            if(left)
            {
                nodes[p]->left = nodes[c];
            }
            else
            {
                nodes[p]->right = nodes[c];
            }
        }
        int root = -1;
        for(auto & [node , val] :isRoot )
        {
            if(val == 1){
                root = node;
            }
        }
        return nodes[root];
    }
};
```

### 一般树

```
给你一棵有 n 个节点的无向树，节点编号为 0 到 n-1 ，它们中有一些节点有苹果。通过树上的一条边，需要花费 1 秒钟。你从 节点 0 出发，请你返回最少需要多少秒，可以收集到所有苹果，并回到节点 0 。

无向树的边由 edges 给出，其中 edges[i] = [fromi, toi] ，表示有一条边连接 from 和 toi 。除此以外，还有一个布尔数组 hasApple ，其中 hasApple[i] = true 代表节点 i 有一个苹果，否则，节点 i 没有苹果。

```

```c++
class Solution {
public:
    int minTime(int n, vector<vector<int>>& edges, vector<bool>& hasApple) {
        graph = vector<vector<int>>(n, vector<int>());
        for (const auto& e : edges) {
            graph[e[0]].push_back(e[1]);
            graph[e[1]].push_back(e[0]);
        }

        vector<int> parent(n, -1);
        buildReverseEdges(0, parent);
        vector<bool> visited(n, false);
        visited[0] = true;
        int res = 0;
        for (int i = 0; i < n; i++) {
            if (hasApple[i]) {
                res += dfs(i, parent, visited);
            }
        }

        return res << 1;
    }

private:
    vector<vector<int>> graph;
    void buildReverseEdges(int src, vector<int>& parent) {
        for (const auto& nei : graph[src]) {
            if (nei != 0 && parent[nei] == -1 /* parent not set yet*/) {
                parent[nei] = src;
                buildReverseEdges(nei, parent);
            }
        }
    }

    int dfs(int to, const vector<int>& parent, vector<bool>& visited) {
        int res = 0;
        if (!visited[to]) {
            visited[to] = true;
            res++;
            res += dfs(parent[to], parent, visited);
        }
        // if already visited, stop counting at intersection

        return res;
    }
};
```
buildReverseEdges:这个函数用于构建 parent 数组，它从根节点 0 开始，遍历所有邻居 nei，如果该邻居还没有被设置父节点（parent[nei] == -1），则将其父节点设置为当前节点 src，并递归调用 buildReverseEdges 继续处理该邻居的邻居。