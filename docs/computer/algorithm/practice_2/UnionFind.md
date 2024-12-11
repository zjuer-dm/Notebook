# 并查集
给你两个整数数组 source 和 target ，长度都是 n 。还有一个数组 allowedSwaps ，其中每个 allowedSwaps[i] = [ai, bi] 表示你可以交换数组 source 中下标为 ai 和 bi（下标从 0 开始）的两个元素。注意，你可以按 任意 顺序 多次 交换一对特定下标指向的元素。

相同长度的两个数组 source 和 target 间的 汉明距离 是元素不同的下标数量。形式上，其值等于满足 source[i] != target[i] （下标从 0 开始）的下标 i（0 <= i <= n-1）的数量。

在对数组 source 执行 任意 数量的交换操作后，返回 source 和 target 间的 最小汉明距离 。


```c++
class Solution {
public:
    int find(vector<int>&f , int x)
    {
        if(f[x] == x)
        {
            return x;
        }
        return f[x] = find(f , f[x]);
    }
    void merge(vector<int>& f , int x , int y)
    {
        f[find(f, x)] = find(f , y);
    }
    int minimumHammingDistance(vector<int>& source, vector<int>& target, vector<vector<int>>& allowedSwaps) {
        int n = source.size();
        vector<int> f(n);
        for(int i = 0 ; i < n ; i++)
        {
            f[i] = i;
        }
        for(auto & e : allowedSwaps)
        {
            merge(f , e[0] , e[1]);
        }
        unordered_map<int ,unordered_multiset<int> > s , t;
        
        for(int i = 0 ; i < n ; i++)
        {
            int fa = find(f,i);
            s[fa].insert(source[i]);
            t[fa].insert(target[i]);
        }

        int ans = 0;

        for(int i = 0 ; i < n ; i++)
        {
            if(s.find(i) == s.end())
            {
                continue;
            }
            for(auto x : s[i])
            {
                if(t[i].find(x)==t[i].end())
                {
                    ans++;
                }
                else
                {
                    t[i].erase(t[i].find(x));
                }
            }
        }
        return ans;
    }
};

```