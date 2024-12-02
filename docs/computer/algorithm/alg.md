
其中许多有用知识在：
<a href="https://oi-wiki.org/">算法</a>

## 拓扑排序
在图论中，拓扑排序（Topological Sorting）是一个有向无环图（DAG, Directed Acyclic Graph）的所有顶点的线性序列。且该序列必须满足下面两个条件：

1. 每个顶点出现且只出现一次。
2. 若存在一条从顶点 A 到顶点 B 的路径，那么在序列中顶点 A 出现在顶点 B 的前面。

它是一个 DAG 图，那么如何写出它的拓扑排序呢？这里说一种比较常用的方法：

1. 从 DAG 图中选择一个 没有前驱（即入度为0）的顶点并输出。
2. 从图中删除该顶点和所有以它为起点的有向边。
3. 重复 1 和 2 直到当前的 DAG 图为空或当前图中不存在无前驱的顶点为止。后一种情况说明有向图中必然存在环。

```c++
#include<iostream>
#include <list>
#include <queue>
using namespace std;

/************************类声明************************/
class Graph
{
    int V;             // 顶点个数
    list<int> *adj;    // 邻接表
    queue<int> q;      // 维护一个入度为0的顶点的集合
    int* indegree;     // 记录每个顶点的入度
public:
    Graph(int V);                   // 构造函数
    ~Graph();                       // 析构函数
    void addEdge(int v, int w);     // 添加边
    bool topological_sort();        // 拓扑排序
};

/************************类定义************************/
Graph::Graph(int V)
{
    this->V = V;
    adj = new list<int>[V];

    indegree = new int[V];  // 入度全部初始化为0
    for(int i=0; i<V; ++i)
        indegree[i] = 0;
}

Graph::~Graph()
{
    delete [] adj;
    delete [] indegree;
}

void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); 
    ++indegree[w];
}

bool Graph::topological_sort()
{
    for(int i=0; i<V; ++i)
        if(indegree[i] == 0)
            q.push(i);         // 将所有入度为0的顶点入队

    int count = 0;             // 计数，记录当前已经输出的顶点数 
    while(!q.empty())
    {
        int v = q.front();      // 从队列中取出一个顶点
        q.pop();

        cout << v << " ";      // 输出该顶点
        ++count;
        // 将所有v指向的顶点的入度减1，并将入度减为0的顶点入栈
        list<int>::iterator beg = adj[v].begin();
        for( ; beg!=adj[v].end(); ++beg)
            if(!(--indegree[*beg]))
                q.push(*beg);   // 若入度为0，则入栈
    }

    if(count < V)
        return false;           // 没有输出全部顶点，有向图中有回路
    else
        return true;            // 拓扑排序成功
}
```

## 并查集：
并查集是一种用于管理元素所属集合的数据结构，实现为一个森林，其中每棵树表示一个集合，树中的节点表示对应集合中的元素。

顾名思义，并查集支持两种操作：

合并（Union）：合并两个元素所属集合（合并对应的树）

查询（Find）：查询某个元素所属集合（查询对应的树的根节点），这可以用于判断两个元素是否属于同一集合

并查集在经过修改后可以支持单个元素的删除、移动；使用动态开点线段树还可以实现可持久化并查集。
```c++
const int  N=1005					//指定并查集所能包含元素的个数（由题意决定）
int pre[N];     					//存储每个结点的前驱结点 
int rank[N];    					//树的高度 
void init(int n)     				//初始化函数，对录入的 n个结点进行初始化 
{
    for(int i = 0; i < n; i++){
        pre[i] = i;     			//每个结点的上级都是自己 
        rank[i] = 1;    			//每个结点构成的树的高度为 1 
    } 
}
int find(int x)     	 		    //查找结点 x的根结点 
{
    if(pre[x] == x) return x;  		//递归出口：x的上级为 x本身，则 x为根结点 
    return find(pre[x]); 			//递归查找 
} 
 
int find(int x)     				//改进查找算法：完成路径压缩，将 x的上级直接变为根结点，那么树的高度就会大大降低 
{
    if(pre[x] == x) return x;		//递归出口：x的上级为 x本身，即 x为根结点 
    return pre[x] = find(pre[x]);   //此代码相当于先找到根结点 rootx，然后 pre[x]=rootx 
} 

bool isSame(int x, int y)      		//判断两个结点是否连通 
{
    return find(x) == find(y);  	//判断两个结点的根结点（即代表元）是否相同 
}

bool join(int x,int y)
{
    x = find(x);						//寻找 x的代表元
    y = find(y);						//寻找 y的代表元
    if(x == y) return false;			//如果 x和 y的代表元一致，说明他们共属同一集合，则不需要合并，返回 false，表示合并失败；否则，执行下面的逻辑
    if(rank[x] > rank[y]) pre[y]=x;		//如果 x的高度大于 y，则令 y的上级为 x
    else								//否则
    {
        if(rank[x]==rank[y]) rank[y]++;	//如果 x的高度和 y的高度相同，则令 y的高度加1
        pre[x]=y;						//让 x的上级为 y
	}
	return true;						//返回 true，表示合并成功
}
```

## 单调栈
单调栈（Monotone Stack）：一种特殊的栈。在栈的「先进后出」规则基础上，
要求「从 栈顶 到 栈底 的元素是单调递增（或者单调递减）」。其中满足从栈顶到栈底的元素是单调递增的栈，叫做「单调递增栈」。满足从栈顶到栈底的元素是单调递减的栈，叫做「单调递减栈」。

**单调递增栈模板**

```py
def monotoneIncreasingStack(nums):
    stack = []
    for num in nums:
        while stack and num >= stack[-1]:
            stack.pop()
        stack.append(num)
```


## 回溯算法
回溯算法是对树形或者图形结构执行一次深度优先遍历，实际上类似枚举的搜索尝试过程，在遍历的过程中寻找问题的解。

深度优先遍历有个特点：当发现已不满足求解条件时，就返回，尝试别的路径。此时对象类型变量就需要重置成为和之前一样，称为「状态重置」。

给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。

```py
from typing import List

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        # 结果列表
        ans = []
        # 当前路径
        path = [0] * n
        # 是否已在路径中
        on_path = [False] * n
        
        # 对 nums 排序
        nums.sort()
        
        def DFS(c):
            # 递归结束条件
            if c == n:
                ans.append(path.copy())
                return
            for x in range(n):
                # 如果当前数字已经使用过，跳过
                if on_path[x]:
                    continue
                # 如果当前数字与前一个数字相同，且前一个数字还未被使用，跳过（避免重复）
                if x > 0 and nums[x] == nums[x - 1] and not on_path[x - 1]:
                    continue
                # 选择当前数字
                on_path[x] = True
                path[c] = nums[x]
                DFS(c + 1)
                # 撤销选择
                on_path[x] = False
        
        DFS(0)
        return ans
```