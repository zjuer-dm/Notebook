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

## 字典树 (Trie)



字典树（Trie，也称为前缀树或查找树）是一种树形数据结构，通常用于存储动态集合或关联数组，其中的键通常是字符串。它能够实现字符串的快速检索、插入和删除，并且可以高效地查找具有相同前缀的字符串。



字典树由节点组成，每个节点包含以下信息：

- **子节点指针数组**：通常是一个大小为字符集大小（如26个字母或ASCII码）的数组，用于指向下一个字符对应的子节点。
- **结束标记**：一个布尔值，表示从根节点到当前节点路径所形成的字符串是否是一个完整的单词。



插入一个字符串时，从根节点开始，遍历字符串的每个字符。如果当前字符对应的子节点不存在，则创建新节点。遍历结束后，将最后一个字符对应节点的结束标记设置为真。

**示例**：插入 "apple" 和 "app"

```mermaid
graph TD
    root["根"]
    root --> a[a]
    a --> p1[p]
    p1 --> p2[p]
    p2 --> l[l]
    l --> e[e(结束)]
    p1 --> p3[p(结束)]
```



查找一个字符串时，从根节点开始，遍历字符串的每个字符。如果某个字符对应的子节点不存在，则表示该字符串不在字典树中。如果遍历完所有字符，并且最后一个字符对应节点的结束标记为真，则表示找到该字符串。



前缀查找与字符串查找类似，只是不需要检查最后一个字符对应节点的结束标记。只要能遍历完所有前缀字符，就表示该前缀存在。


