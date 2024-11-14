# 十一月练习
## 字典树（trie）
```
在英语中，我们有一个叫做 词根(root) 的概念，可以词根 后面 添加其他一些词组成另一个较长的单词——我们称这个词为 衍生词 (derivative)。例如，词根 help，跟随着 继承词 "ful"，可以形成新的单词 "helpful"。

现在，给定一个由许多 词根 组成的词典 dictionary 和一个用空格分隔单词形成的句子 sentence。你需要将句子中的所有 衍生词 用 词根 替换掉。如果 衍生词 有许多可以形成它的 词根，则用 最短 的 词根 替换它。

你需要输出替换之后的句子。
```

```py
class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        dictionarySet = set(dictionary)
        words = sentence.split(' ')
        for i, word in enumerate(words):
            for j in range(1, len(word) + 1):
                if word[:j] in dictionarySet:
                    words[i] = word[:j]
                    break
        return ' '.join(words)
```

字典树

```py
class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        # 构建字典树
        trie = {}
        for word in dictionary:
            cur = trie  # 指向当前节点，初始为根节点
            for c in word:  # 遍历词根的每个字符
                if c not in cur:  # 如果当前字符不在当前节点中
                    cur[c] = {}  # 创建新节点
                cur = cur[c]  # 移动到下一个节点
            cur['#'] = {}  # 在词根末尾标记结束

        # 处理句子
        words = sentence.split(' ')  # 按空格分割句子为单词
        for i, word in enumerate(words):  # 遍历每个单词
            cur = trie  # 重置当前节点为根节点
            for j, c in enumerate(word):  # 遍历当前单词的每个字符
                if '#' in cur:  # 如果找到了完整的词根
                    words[i] = word[:j]  # 替换为找到的词根
                    break  # 结束查找
                if c not in cur:  # 如果当前字符不在字典树中
                    break  # 结束查找
                cur = cur[c]  # 移动到下一个节点

        return ' '.join(words)  # 将处理后的单词连接成句子并返回
```

* 每日一题：
```
来自未来的体育科学家给你两个整数数组 energyDrinkA 和 energyDrinkB，数组长度都等于 n。这两个数组分别代表 A、B 两种不同能量饮料每小时所能提供的强化能量。

你需要每小时饮用一种能量饮料来 最大化 你的总强化能量。然而，如果从一种能量饮料切换到另一种，你需要等待一小时来梳理身体的能量体系（在那个小时里你将不会获得任何强化能量）。

返回在接下来的 n 小时内你能获得的 最大 总强化能量。

注意 你可以选择从饮用任意一种能量饮料开始。
```

```py
class Solution:
    def maxEnergyBoost(self, a: List[int], b: List[int]) -> int:
        n = len(a)
        f = [[0, 0] for _ in range(n + 2)]
        for i, (x, y) in enumerate(zip(a, b)):
            f[i + 2][0] = max(f[i + 1][0], f[i][1]) + x
            f[i + 2][1] = max(f[i + 1][1], f[i][0]) + y
        return max(f[-1])
```

* 每日一题：
```
在 LeetCode 商店中， 有 n 件在售的物品。每件物品都有对应的价格。然而，也有一些大礼包，每个大礼包以优惠的价格捆绑销售一组物品。

给你一个整数数组 price 表示物品价格，其中 price[i] 是第 i 件物品的价格。另有一个整数数组 needs 表示购物清单，其中 needs[i] 是需要购买第 i 件物品的数量。

还有一个数组 special 表示大礼包，special[i] 的长度为 n + 1 ，其中 special[i][j] 表示第 i 个大礼包中内含第 j 件物品的数量，且 special[i][n] （也就是数组中的最后一个整数）为第 i 个大礼包的价格。

返回 确切 满足购物清单所需花费的最低价格，你可以充分利用大礼包的优惠活动。你不能购买超出购物清单指定数量的物品，即使那样会降低整体价格。任意大礼包可无限次购买。
```

```py
class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        n = len(price)

        filter_special = []
        for sp in special:
            if sum(sp[i] for i in range(n)) > 0 and sum(sp[i] * price[i] for i in range(n)) > sp[-1]:
                filter_special.append(sp)

        @cache
        def DFS(cur):
            min_price = sum(need * price[i] for i , need in enumerate(cur))
            for cur_special in filter_special:
                special_price = cur_special[-1]
                next_need = []
                for i in range(n):
                    if cur_special[i] > cur[i]:
                        break
                    next_need.append(cur[i] - cur_special[i])
                if len(next_need) == n:
                    min_price = min(min_price , DFS(tuple(next_need)) + special_price)
            return min_price
        return DFS(tuple(needs))
```

```c++
class Solution {
public:
    map<vector<int>, int> memo;

    int shoppingOffers(vector<int>& price, vector<vector<int>>& special, vector<int>& needs) {
        int n = price.size();

        // 过滤不需要计算的大礼包，只保留需要计算的大礼包
        vector<vector<int>> filterSpecial;
        for (auto & sp : special) {
            int totalCount = 0, totalPrice = 0;
            for (int i = 0; i < n; ++i) {
                totalCount += sp[i];
                totalPrice += sp[i] * price[i];
            }
            if (totalCount > 0 && totalPrice > sp[n]) {
                filterSpecial.emplace_back(sp);
            }
        }

        return dfs(price, special, needs, filterSpecial, n);
    }

    // 记忆化搜索计算满足购物清单所需花费的最低价格
    int dfs(vector<int> price,const vector<vector<int>> & special, vector<int> curNeeds, vector<vector<int>> & filterSpecial, int n) {
        if (!memo.count(curNeeds)) {
            int minPrice = 0;
            for (int i = 0; i < n; ++i) {
                minPrice += curNeeds[i] * price[i]; // 不购买任何大礼包，原价购买购物清单中的所有物品
            }
            for (auto & curSpecial : filterSpecial) {
                int specialPrice = curSpecial[n];
                vector<int> nxtNeeds;
                for (int i = 0; i < n; ++i) {
                    if (curSpecial[i] > curNeeds[i]) { // 不能购买超出购物清单指定数量的物品
                        break;
                    }
                    nxtNeeds.emplace_back(curNeeds[i] - curSpecial[i]);
                }
                if (nxtNeeds.size() == n) { // 大礼包可以购买
                    minPrice = min(minPrice, dfs(price, special, nxtNeeds, filterSpecial, n) + specialPrice);
                }
            }
            memo[curNeeds] = minPrice;
        }
        return memo[curNeeds];
    }
};
```

每日一题：

有一根长度为 n 个单位的木棍，棍上从 0 到 n 标记了若干位置。例如，长度为 6 的棍子可以标记如下：

给你一个整数数组 cuts ，其中 cuts[i] 表示你需要将棍子切开的位置。

你可以按顺序完成切割，也可以根据需要更改切割的顺序。

每次切割的成本都是当前要切割的棍子的长度，切棍子的总成本是历次切割成本的总和。对棍子进行切割将会把一根木棍分成两根较小的木棍（这两根木棍的长度和就是切割前木棍的长度）。请参阅第一个示例以获得更直观的解释。

返回切棍子的 最小总成本 。

```py
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        cuts.sort()
        cuts = [0] + cuts + [n]
        m = len(cuts)
        
        dp = [[0] * m for _ in range(m)]
        for i in range(m-3, -1 , -1):
            for j in range(i+2 , m):
                dp[i][j] = min(dp[i][k] + dp[k][j] for k in range(i+1 , j)) +cuts[j] - cuts[i]
        return dp[0][-1]
```

## 贪心：
给定两个长度相等的数组 nums1 和 nums2，nums1 相对于 nums2 的优势可以用满足 nums1[i] > nums2[i] 的索引 i 的数目来描述。

返回 nums1 的任意排列，使其相对于 nums2 的优势最大化。

```py
class Solution:
    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()

        n = len(nums1)
        ids = sorted(range(n), key=lambda i: nums2[i])

        ans = [0] * n
        left, right = 0, n - 1
        for x in nums1:
            if x > nums2[ids[left]]:
                ans[ids[left]] = x  # 用下等马比下等马
                left += 1
            else:
                ans[ids[right]] = x  # 用下等马比上等马
                right -= 1
        return ans

```

```c++
class Solution {
public:
    vector<int> advantageCount(vector<int>& nums1, vector<int>& nums2) {
        sort(nums1.begin() , nums1.end());
        int left = 0 ;
        int right = nums1.size() - 1;
        vector<int> ans(right+1);
        vector<int> help (right+1);
        iota(help.begin(), help.end(), 0);
        ranges::sort(help, [&](int i, int j) { return nums2[i] < nums2[j]; });


        for(auto x : nums1)
        {
            if(x > nums2[help[left]])
            {
                ans[help[left] ]= x ;
                left++;
            }
            else
            {
                ans[help[right]] = x ;
                right --;
            }
        }

        return ans;
    }
};
```

在一个仓库里，有一排条形码，其中第 i 个条形码为 barcodes[i]。

请你重新排列这些条形码，使其中任意两个相邻的条形码不能相等。 你可以返回任何满足该要求的答案，此题保证存在答案。

```c++
class Solution {
public:
    vector<int> rearrangeBarcodes(vector<int>& barcodes) {
        unordered_map<int, int> count;
        for (int b : barcodes) {
            count[b]++;
        }
        priority_queue<pair<int, int>> q;
        for (const auto &[x, cx] : count) {
            q.push({cx, x});
        }
        vector<int> res;
        while (q.size()) {
            auto [cx, x] = q.top();
            q.pop();
            if (res.empty() || res.back() != x) {
                res.push_back(x);
                if (cx > 1) {
                    q.push({cx - 1, x});
                }
            } else {
                if (q.size() < 1) return res;
                auto [cy, y] = q.top();
                q.pop();
                res.push_back(y);
                if (cy > 1)  {
                    q.push({cy - 1, y});
                }
                q.push({cx, x});
            }
        }
        return res;
    }
};

```

**每日一题**
给你一个 二进制 字符串 s 和一个整数 k。

另给你一个二维整数数组 queries ，其中 queries[i] = [li, ri] 。

如果一个 二进制字符串 满足以下任一条件，则认为该字符串满足 k 约束：

字符串中 0 的数量最多为 k。
字符串中 1 的数量最多为 k。
返回一个整数数组 answer ，其中 answer[i] 表示 s[li..ri] 中满足 k 约束 的 
子字符串的数量。

*方法一：滑动窗口+前缀和+二分查找*
```py
class Solution:
    def countKConstraintSubstrings(self, s: str, k: int, queries: List[List[int]]) -> List[int]:
        n = len(s)
        pre = [0] * (n+1)
        left = [0] * n
        l = 0
        cnt = [0 , 0]
        for i , c in enumerate(s):
            cnt[ord(c)&1] += 1
            while cnt[0] > k and cnt[1] > k:
                cnt[ord(s[l])&1] -= 1
                l += 1
            left[i] = l
            pre[i+1] = pre[i] + i -l +1

        ans = [] 
        for l , r in queries:
            j = bisect_left(left , l , l ,r+1)
            ans.append(pre[r + 1] - pre[j] + (j - l + 1) * (j - l) // 2)
        return ans
```

```py
class Solution:
    def countKConstraintSubstrings(self, s: str, k: int, queries: List[List[int]]) -> List[int]:
        n = len(s)
        right = [n] * n
        pre = [0] * (n + 1)
        cnt = [0, 0]
        l = 0
        for i, c in enumerate(s):
            cnt[ord(c) & 1] += 1
            while cnt[0] > k and cnt[1] > k:
                cnt[ord(s[l]) & 1] -= 1
                right[l] = i
                l += 1
            pre[i + 1] = pre[i] + i - l + 1

        ans = []
        for l, r in queries:
            j = min(right[l], r + 1)
            ans.append(pre[r + 1] - pre[j] + (j - l + 1) * (j - l) // 2)
        return ans

```

## 堆

>最小堆可以看作是一种优先级队列的实现，有些应用场景需要从队列中获取最小的或者最大的元素，而且不要求数据全部有序，使用最小堆或者最大堆能很好的解决这类问题。
最小堆的元素是按完全二叉树的顺序存储方式存放在一维数组中。

给你一个整数 mountainHeight 表示山的高度。

同时给你一个整数数组 workerTimes，表示工人们的工作时间（单位：秒）。

工人们需要 同时 进行工作以 降低 山的高度。对于工人 i :

山的高度降低 x，需要花费 workerTimes[i] + workerTimes[i] * 2 + ... + workerTimes[i] * x 秒。例如：
山的高度降低 1，需要 workerTimes[i] 秒。
山的高度降低 2，需要 workerTimes[i] + workerTimes[i] * 2 秒，依此类推。
返回一个整数，表示工人们使山的高度降低到 0 所需的 最少 秒数。

```py
class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        h = [(t,t,t) for t in workerTimes]
        heapify(h)
        for _ in range(mountainHeight):
            nxt , delta , base = h[0]
            heapreplace(h , (nxt + delta + base , base + delta , base))
        return nxt
```

python中global和nonlocal:

第一，两者的功能不同。global关键字修饰变量后标识该变量是全局变量，对该变量进行修改就是修改全局变量，而nonlocal关键字修饰变量后标识该变量是上一级函数中的局部变量，如果上一级函数中不存在该局部变量，nonlocal位置会发生错误（最上层的函数使用nonlocal修饰变量必定会报错）。

第二，两者使用的范围不同。global关键字可以用在任何地方，包括最上层函数中和嵌套函数中，即使之前未定义该变量，global修饰后也可以直接使用，而nonlocal关键字只能用于嵌套函数中，并且外层函数中定义了相应的局部变量，否则会发生错误


    现有一棵无向树，树中包含 n 个节点，按从 0 到 n - 1 标记。树的根节点是节点 0 。给你一个长度为 n - 1 的二维整数数组 edges，其中 edges[i] = [ai, bi] 表示树中节点 ai 与节点 bi 之间存在一条边。

    如果一个节点的所有子节点为根的子树包含的节点数相同，则认为该节点是一个好节点。

    返回给定树中好节点的数量。

    子树指的是一个节点以及它所有后代节点构成的一棵树。

```py
class Solution:
    def countGoodNodes(self, edges: List[List[int]]) -> int:
        n = len(edges) + 1 
        g = [[] for _ in range(n) ]
        for x , y in edges:
            g[x].append(y)
            g[y].append(x)
        ans = 0
        def DFS(x : int , fa : int) -> int :
            sz0 = 0 
            flag = 1
            size = 1
            for y in g[x]:
                if y == fa:
                    continue
                sz = DFS(y , x)
                if sz0 == 0:
                    sz0 = sz
                elif sz0 != sz:
                    flag = 0
                size += sz
            nonlocal ans
            ans += flag
            return size
        
        DFS(0,-1)
        return ans
```

## 队列
给你一个整数数组 nums ，和一个表示限制的整数 limit，请你返回最长连续子数组的长度，该子数组中的任意两个元素之间的绝对差必须小于或者等于 limit 。

如果不存在满足条件的子数组，则返回 0 。

```py
from sortedcontainers import SortedList
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        n = len(nums)
        left , right  ,ans = 0, 0, 0
        s = SortedList()
        while  right < n:
            s.add(nums[right])
            if s[-1] - s[0] > limit:
                s.remove(nums[left])
                left += 1
            ans = max(ans , right - left +1)
            right += 1
        return ans
```

## dp
给你一个 下标从 1 开始的 整数数组 prices ，其中 prices[i] 表示你购买第 i 个水果需要花费的金币数目。

水果超市有如下促销活动：

如果你花费 prices[i] 购买了下标为 i 的水果，那么你可以免费获得下标范围在 [i + 1, i + i] 的水果。
注意 ，即使你 可以 免费获得水果 j ，你仍然可以花费 prices[j] 个金币去购买它以获得它的奖励。

请你返回获得所有水果所需要的 最少 金币数。

```py
class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        for i in range( (n + 1) //2 -1 , 0 , -1 ):
            prices[i - 1] += min(prices[i: i * 2 + 1])
        return prices[0]
```

```py
class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i: int) -> int:
            if i * 2 >= n:
                return prices[i - 1]  # i 从 1 开始
            return min(dfs(j) for j in range(i + 1, i * 2 + 2)) + prices[i - 1]
        return dfs(1)
```