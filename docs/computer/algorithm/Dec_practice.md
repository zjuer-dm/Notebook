## 回溯
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

```py
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)

        ans = []
        path = [0] * n 

        def DFS(c , f):
            if c == n:
                ans.append(path.copy())
                return 
            for x in f:
                path[c] = x
                DFS(c+1 , f - {x})
        DFS(0 , set(nums))
        return ans
```

n 座城市，从 0 到 n-1 编号，其间共有 n-1 条路线。因此，要想在两座不同城市之间旅行只有唯一一条路线可供选择（路线网形成一颗树）。去年，交通运输部决定重新规划路线，以改变交通拥堵的状况。

路线用 connections 表示，其中 connections[i] = [a, b] 表示从城市 a 到 b 的一条有向路线。

今年，城市 0 将会举办一场大型比赛，很多游客都想前往城市 0 。

请你帮助重新规划路线方向，使每个城市都可以访问城市 0 。返回需要变更方向的最小路线数。

题目数据 保证 每个城市在重新规划路线方向后都能到达城市 0 。

    Treat the graph as undirected. Start a dfs from the root, 
    if you come across an edge in the forward direction, you need to reverse the edge.

```py
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        e = [[] for _ in range(n)]
        for edge in connections:
            e[edge[0]].append([edge[1] , 1])
            e[edge[1]].append([edge[0] , 0])
        
        def DFS(x , parent):
            res = 0
            for edge in e[x]:
                if edge[0] == parent:
                    continue
                res += DFS(edge[0] , x) + edge[1]
            return res

        return DFS(0 , -1 )
```

公司里有 n 名员工，每个员工的 ID 都是独一无二的，编号从 0 到 n - 1。公司的总负责人通过 headID 进行标识。

在 manager 数组中，每个员工都有一个直属负责人，其中 manager[i] 是第 i 名员工的直属负责人。对于总负责人，manager[headID] = -1。题目保证从属关系可以用树结构显示。

公司总负责人想要向公司所有员工通告一条紧急消息。他将会首先通知他的直属下属们，然后由这些下属通知他们的下属，直到所有的员工都得知这条紧急消息。

第 i 名员工需要 informTime[i] 分钟来通知它的所有直属下属（也就是说在 informTime[i] 分钟后，他的所有直属下属都可以开始传播这一消息）。

返回通知所有员工这一紧急消息所需要的 分钟数 。

```py
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        g = [[ ] for _ in range(n)]
        for i , x in enumerate(manager):
            if x >= 0:
                g[x].append(i)
        
        def DFS(m):
            ans = 0
            for i in g[m]:
                ans = max(ans , DFS(i))

            return ans + informTime[m]

        return DFS(headID)
```