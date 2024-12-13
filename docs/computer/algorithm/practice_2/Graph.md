你准备参加一场远足活动。给你一个二维 rows x columns 的地图 heights ，其中 heights[row][col] 表示格子 (row, col) 的高度。一开始你在最左上角的格子 (0, 0) ，且你希望去最右下角的格子 (rows-1, columns-1) （注意下标从 0 开始编号）。你每次可以往 上，下，左，右 四个方向之一移动，你想要找到耗费 体力 最小的一条路径。

一条路径耗费的 体力值 是路径上相邻格子之间 高度差绝对值 的 最大值 决定的。

请你返回从左上角走到右下角的最小 体力消耗值 。


* 将每个格子抽象成图中的一个点；

* 将每两个相邻的格子之间连接一条边，长度为这两个格子本身权值的差的绝对值；

* 需要找到一条从左上角到右下角的「最短路径」，其中路径的长度定义为路径上所有边的权值的最大值。

```c++
/*
「二分答案」：我们可以对最短路径的长度进行二分。当我们二分枚举到的长度为 x 时，我们
只保留所有长度 ≤x 的边。随后从左上角开始进行搜索
（深度优先搜索、广度优先搜索）均可，只需要判断是否能够到达右下角即可。
*/
class Solution {
private:
    const int dirs[4][2] = {{-1, 0} , {1 ,0} , {0, 1} , {0, -1}};
public:

    int minimumEffortPath(vector<vector<int>>& heights) {
        int m = heights.size();
        int n = heights[0].size();
        int left = 0, right = 999999 , ans = 0;
        while(left <= right)
        {
            int mid = (left + right) / 2;
            queue<pair<int,int>> q;
            q.emplace(0,0);
            vector<int> seen(m * n);
            seen[0] = 1;
            while(!q.empty())
            {
                auto [ x, y ] =q.front();
                q.pop();
                for(auto [dx,dy] : dirs)
                {
                    int nx = x + dx ;
                    int ny = y + dy ;
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && !seen[nx * n + ny] && abs(heights[x][y] - heights[nx][ny]) <= mid) {
                        q.emplace(nx, ny);
                        seen[nx * n + ny] = 1;
                    }
                }
            }
            if(seen[m*n -1])
            {
                ans = mid;
                right = mid-1;
            }
            else
            {
                //ans = mid;
                left = mid +1;
            }
        }
        return ans;
    }
};
```

```C++
/*
「并查集」：我们可以将所有边按照长度进行排序并依次添加进并查集，直到左上角和右下角连通为止。
*/

class UnionFind {
public:
    std::vector<int> parent;  // 存储每个元素的父节点
    std::vector<int> size;    // 存储每个集合的大小
    int n;                    // 元素的数量
    int component_count;      // 当前连通分量的数量
    
public:
    // 构造函数，初始化并设置每个元素的父节点为自己，初始时每个集合的大小为1
    UnionFind(int num_elements) : n(num_elements), component_count(num_elements), 
                                  parent(num_elements), size(num_elements, 1) {
        // 每个元素的父节点初始化为它自己
        std::iota(parent.begin(), parent.end(), 0);
    }
    
    // 查找元素x所在集合的代表元素（根节点），并进行路径压缩优化
    int find(int x) {
        // 如果x是集合的代表元素（根节点），直接返回
        if (parent[x] == x) {
            return x;
        }
        // 否则递归找到x的根节点，并在回溯过程中进行路径压缩
        return parent[x] = find(parent[x]);
    }
    
    // 合并两个元素x和y所在的集合
    void unite(int x, int y) {
        // 找到x和y的代表元素
        int root_x = find(x);
        int root_y = find(y);
        
        // 如果x和y已经在同一个集合中，不需要合并
        if (root_x == root_y) {
            return;
        }
        
        // 合并时，确保较小的集合并入较大的集合
        if (size[root_x] < size[root_y]) {
            std::swap(root_x, root_y);
        }
        
        // 将root_y的根节点指向root_x
        parent[root_y] = root_x;
        // 更新合并后集合的大小
        size[root_x] += size[root_y];
        
        // 合并操作减少了一个连通分量
        --component_count;
    }
    
    // 判断x和y是否在同一个集合中
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
    
    // 获取当前的连通分量数
    int getComponentCount() const {
        return component_count;
    }
};


struct Edge{
    int x,y,z;
    Edge(int _x , int _y , int _z): x(_x) , y(_y) ,z(_z) {}
    bool operator< (const Edge& that)const{
        return z < that.z;
    }
};

class Solution {
private:
    const int dirs[4][2] = {{-1, 0} , {1 ,0} , {0, 1} , {0, -1}};
public:
    int minimumEffortPath(vector<vector<int>>& heights) {
        int m = heights.size();
        int n = heights[0].size();
        vector<Edge> edges;
        for(int i = 0 ; i < m ; i ++)
        {
            for(int j = 0 ; j < n ; j ++)
            {
                int id = i * n + j;
                if(i > 0)
                    edges.emplace_back(id - n , id ,abs(heights[i][j] - heights[i-1][j]));
                if(j > 0)
                {
                    edges.emplace_back(id-1 , id , abs(heights[i][j] - heights[i][j-1]));
                }
            }
        }
        sort(edges.begin( ), edges.end());
        UnionFind uf(m * n);
        for(auto edge:edges)
        {
            uf.unite(edge.x , edge.y);
            if(uf.connected(0 , m*n-1))
            {
                return edge.z;
            }
        }
        return 0;
    }
};

```

有 n 个网络节点，标记为 1 到 n。

给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点， wi 是一个信号从源节点传递到目标节点的时间。

现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。

**适用于稠密图**
```c++
class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<vector<int>> g(n , vector<int>(n ,INT_MAX/2));
        for(auto & e : times)
        {
            g[e[0] -1 ][e[1] -1] = e[2];
        }

        vector<int> dist(n , INT_MAX/2);
        dist[k-1] = 0;
        int ans = 0;
        vector<int> ok(n);
        while(1)
        {
            int x = -1 ;
            for(int i = 0 ; i < n ; i++)
            {
                if(!ok[i] && (x < 0 || dist[i] < dist[x]))
                    x = i;
            }
            if(x < 0)
            {
                return ans;
            }
            if(dist[x] == INT_MAX/2)
            {
                return -1;
            }
            ans = dist[x];
            ok[x] = 1;
            for(int i = 0 ; i < n ; i++)
            {
                dist[i] = min(dist[i] , dist[x] + g[x][i]);
            }
        }
    }
};
```

**适用于稀疏图**

```c++
class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<vector<pair<int, int>>> g(n); // 邻接表
        for (auto& t : times) {
            g[t[0] - 1].emplace_back(t[1] - 1, t[2]);
        }

        vector<int> dis(n, INT_MAX);
        dis[k - 1] = 0;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        pq.emplace(0, k - 1);
        while (!pq.empty()) {
            auto [dx, x] = pq.top();
            pq.pop();
            if (dx > dis[x]) { // x 之前出堆过
                continue;
            }
            for (auto &[y, d] : g[x]) {
                int new_dis = dx + d;
                if (new_dis < dis[y]) {
                    dis[y] = new_dis; // 更新 x 的邻居的最短路
                    pq.emplace(new_dis, y);
                }
            }
        }
        int mx = ranges::max(dis);
        return mx < INT_MAX ? mx : -1;
    }
};
```

数据结构中对于稀疏图的定义为：有很少条边或弧（边的条数|E|远小于|V|²）的图称为稀疏图（sparse graph），
反之边的条数|E|接近|V|²，称为稠密图（dense graph）。