## 拓扑排序
拓扑排序（Topological sorting）要解决的问题是如何给一个有向无环图的所有节点排序。

我们可以拿大学每学期排课的例子来描述这个过程，比如学习大学课程中有：「程序设计」，「算法语言」，「高等数学」，「离散数学」，「编译技术」，「普通物理」，「数据结构」，「数据库系统」等。按照例子中的排课，当我们想要学习「数据结构」的时候，就必须先学会「离散数学」，学习完这门课后就获得了学习「编译技术」的前置条件。当然，「编译技术」还有一个更加前的课程「算法语言」。这些课程就相当于几个顶点 $u$, 顶点之间的有向边 $(u,v)$ 就相当于学习课程的顺序。教务处安排这些课程，使得在逻辑关系符合的情况下排出课表，就是拓扑排序的过程。


```cpp
using Graph = vector<vector<int>>;  // 邻接表

struct TopoSort {
    enum class Status : uint8_t { to_visit, visiting, visited };

    const Graph& graph;
    const int n;
    vector<Status> status;
    vector<int> order;
    vector<int>::reverse_iterator it;

    TopoSort(const Graph& graph)
        : graph(graph),
        n(graph.size()),
        status(n, Status::to_visit),
        order(n),
        it(order.rbegin()) {}

    bool sort() {
    for (int i = 0; i < n; ++i) {
        if (status[i] == Status::to_visit && !dfs(i)) return false;
    }
    return true;
    }

    bool dfs(const int u) {
    status[u] = Status::visiting;
    for (const int v : graph[u]) {
        if (status[v] == Status::visiting) return false;
        if (status[v] == Status::to_visit && !dfs(v)) return false;
    }
    status[u] = Status::visited;
    *it++ = u;
    return true;
    }
};
```

```python
from enum import Enum, auto


class Status(Enum):
    to_visit = auto()
    visiting = auto()
    visited = auto()


def topo_sort(graph: list[list[int]]) -> list[int] | None:
    n = len(graph)
    status = [Status.to_visit] * n
    order = []

    def dfs(u: int) -> bool:
        status[u] = Status.visiting
        for v in graph[u]:
            if status[v] == Status.visiting:
                return False
            if status[v] == Status.to_visit and not dfs(v):
                return False
        status[u] = Status.visited
        order.append(u)
        return True

    for i in range(n):
        if status[i] == Status.to_visit and not dfs(i):
            return None

    return order[::-1]
```

时间复杂度：$O(E+V)$ 空间复杂度：$O(V)$

