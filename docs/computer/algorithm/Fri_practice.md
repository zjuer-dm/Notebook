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
### 区间
给你一个正整数 days，表示员工可工作的总天数（从第 1 天开始）。另给你一个二维数组 meetings，长度为 n，其中 meetings[i] = [start_i, end_i] 表示第 i 次会议的开始和结束天数（包含首尾）。

返回员工可工作且没有安排会议的天数。

注意：会议时间可能会有重叠。

```py
class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        meetings.sort(key = lambda a : a[0])
        left = 0
        right = -1
        count = 0
        for i , j in meetings:
            if i > right:
                
                
                count += right - left +1
                left = i
            right = max(right , j)
        count += right - left +1
        return days - count
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
这里有 n 个一样的骰子，每个骰子上都有 k 个面，分别标号为 1 到 k 。

给定三个整数 n、k 和 target，请返回投掷骰子的所有可能得到的结果（共有 kn 种方式），使得骰子面朝上的数字总和等于 target。

由于答案可能很大，你需要对 109 + 7 取模。

```py
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        if not ( n <= target and n*k >= target):
            return 0
        mod = 10**9 + 7
        @cache
        def DFS(i,j):
            
            if i ==0:
                return j==0
            res = 0
            for x in range(1,k+1):
                res += DFS(i-1 , j-x)
            return res % mod
        return DFS(n , target)
```

给你一棵二叉树的根节点 root ，二叉树中节点的值 互不相同 。另给你一个整数 start 。在第 0 分钟，感染 将会从值为 start 的节点开始爆发。

每分钟，如果节点满足以下全部条件，就会被感染：

节点此前还没有感染。
节点与一个已感染节点相邻。
返回感染整棵树需要的分钟数。
```py
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        if not ( n <= target and n*k >= target):
            return 0
        mod = 10**9 + 7
        f = [[0]*(target - n + 1) for _ in range(n+1)]
        f[0][0] = 1
        for i in range(1,n+1):
            for j in range(target - n + 1) :
                for x in range(min(k , j+1)):
                    f[i][j] = (f[i][j] + f[i-1][j-x]) % mod
        return f[n][-1]
```


```py
class Solution:
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        ans = 0
        def DFS(node:Optional[TreeNode]):
            if node is None:
                return 0 ,False
            l , l_found = DFS(node.left)
            r , r_found = DFS(node.right)
            nonlocal ans
            if node.val == start:
                ans = max(l , r)
                return 1,True

            if l_found or r_found:
                ans = max(ans , l+r)
                return (l if l_found else r) + 1 , True
            return max(l,r) + 1,False
        DFS(root)
        return ans
```

每日一题：

给你一个 m x n 的二进制矩阵 grid 。

如果矩阵中一行或者一列从前往后与从后往前读是一样的，那么我们称这一行或者这一列是 回文 的。

你可以将 grid 中任意格子的值 翻转 ，也就是将格子里的值从 0 变成 1 ，或者从 1 变成 0 。

请你返回 最少 翻转次数，使得矩阵中 所有 行和列都是 回文的 ，且矩阵中 1 的数目可以被 4 整除 。


```py
class Solution:
    def minFlips(self, a: List[List[int]]) -> int:
        m, n = len(a), len(a[0])
        ans = 0
        for i in range(m // 2):
            row, row2 = a[i], a[-1 - i]
            for j in range(n // 2):
                cnt1 = row[j] + row[-1 - j] + row2[j] + row2[-1 - j]
                ans += min(cnt1, 4 - cnt1)  # 全为 1 或全为 0

        if m % 2 and n % 2:
            # 正中间的数必须是 0
            ans += a[m // 2][n // 2]

        diff = cnt1 = 0
        if m % 2:
            # 统计正中间这一排
            row = a[m // 2]
            for j in range(n // 2):
                if row[j] != row[-1 - j]:
                    diff += 1
                else:
                    cnt1 += row[j] * 2
        if n % 2:
            # 统计正中间这一列
            for i in range(m // 2):
                if a[i][n // 2] != a[-1 - i][n // 2]:
                    diff += 1
                else:
                    cnt1 += a[i][n // 2] * 2

        return ans + (diff if diff else cnt1 % 4)
```

## 滑动窗口
在社交媒体网站上有 n 个用户。给你一个整数数组 ages ，其中 ages[i] 是第 i 个用户的年龄。

如果下述任意一个条件为真，那么用户 x 将不会向用户 y（x != y）发送好友请求：

ages[y] <= 0.5 * ages[x] + 7
ages[y] > ages[x]
ages[y] > 100 && ages[x] < 100
否则，x 将会向 y 发送一条好友请求。

注意，如果 x 向 y 发送一条好友请求，y 不必也向 x 发送一条好友请求。另外，用户不会向自己发送好友请求。

返回在该社交媒体网站上产生的好友请求总数。

```py
class Solution:
    def numFriendRequests(self, ages: List[int]) -> int:
        cnt = [0] * 121
        for age in ages:
            cnt[age] += 1

        ans = cnt_window = age_y = 0
        for age_x, c in enumerate(cnt):
            cnt_window += c
            if age_y * 2 <= age_x + 14:  # 不能发送好友请求
                cnt_window -= cnt[age_y]
                age_y += 1
            if cnt_window:  # 存在可以发送好友请求的用户
                ans += c * cnt_window - c
        return ans
```
给你一个整数数组 nums 和一个整数 k ，请你返回子数组内所有元素的乘积严格小于 k 的连续子数组的数目。

```py
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0
        count = 0
        left = 0
        pro = 1
        
        for right , num in enumerate(nums):
            pro *= num
            while pro >= k:
                pro //= nums[left]
                left+=1
            count += right - left + 1

        return count
```

## 数学：
给你一个整数 n 。如果两个整数 x 和 y 满足下述条件，则认为二者形成一个质数对：

1 <= x <= y <= n
x + y == n
x 和 y 都是质数
请你以二维有序列表的形式返回符合题目要求的所有 [xi, yi] ，列表需要按 xi 的 非递减顺序 排序。如果不存在符合要求的质数对，则返回一个空数组。

注意：质数是大于 1 的自然数，并且只有两个因子，即它本身和 1 。

```py
MX = 10 ** 6 + 1
primes = []
is_prime = [True] * MX
for i in range(2, MX):
    if is_prime[i]:
        primes.append(i)
        for j in range(i * i, MX, i):
            is_prime[j] = False
primes.extend((MX, MX))  

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n % 2:
            return [[2,n-2]] if is_prime[n-2] and n > 4 else []
        
        ans = []
        for x in primes:
            if is_prime[n-x] and n-x >= x:
                ans.append([x,n-x])
        return ans 
```

或者：线性筛

```py
MX = 10 ** 6 + 1
primes = []
is_prime = [True] * MX
for i in range(2, MX):
    if is_prime[i]:
        primes.append(i)
    for p in primes:
        if i * p >= MX: break
        is_prime[i * p] = False
        if i % p == 0: break
primes.extend((MX, MX))  # 保证下面下标不会越界
```

每日一题：

给你一个整数 n 和一个二维整数数组 queries。

有 n 个城市，编号从 0 到 n - 1。初始时，每个城市 i 都有一条单向道路通往城市 i + 1（ 0 <= i < n - 1）。

queries[i] = [ui, vi] 表示新建一条从城市 ui 到城市 vi 的单向道路。每次查询后，你需要找到从城市 0 到城市 n - 1 的最短路径的长度。

返回一个数组 answer，对于范围 [0, queries.length - 1] 中的每个 i，answer[i] 是处理完前 i + 1 个查询后，从城市 0 到城市 n - 1 的最短路径的长度。

```py
class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        edges = [ [i+1] for i in range(n)]
        edges[-1] = []
        ans = []

        def BFS(edges , n):
            dist = [-1]*(n)
            dist[0] = 0
            q = deque([0])
            while len(q)>0:
                x = q.popleft()
                for y in edges[x]:
                    if dist[y] > 0:
                        if y == n-1:
                            return dist[y]
                        continue
                    q.append(y)
                    dist[y] = dist[x] + 1
            return dist[n-1]
        
        for u, v in queries:
            edges[u].append(v)
            ans.append(BFS(edges , n))
        return ans
```

```py
class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:

        edges = [ [i-1] for i in range(n)]
        edges[0] = []
        ans = []
        dp = [i for i in range(n)]
        dp[0] = 0

        for (x , y) in queries:
            edges[y].append(x)
            for i in range(y , n):
                for ele in edges[i]:
                    dp[i] = min(dp[i] , dp[ele] + 1)
            ans.append(dp[-1])

        return ans
```

## dp综合

每日一题：并查集

给你一个整数 n 和一个二维整数数组 queries。

有 n 个城市，编号从 0 到 n - 1。初始时，每个城市 i 都有一条单向道路通往城市 i + 1（ 0 <= i < n - 1）。

queries[i] = [ui, vi] 表示新建一条从城市 ui 到城市 vi 的单向道路。每次查询后，你需要找到从城市 0 到城市 n - 1 的最短路径的长度。

所有查询中不会存在两个查询都满足 queries[i][0] < queries[j][0] < queries[i][1] < queries[j][1]。

返回一个数组 answer，对于范围 [0, queries.length - 1] 中的每个 i，answer[i] 是处理完前 i + 1 个查询后，从城市 0 到城市 n - 1 的最短路径的长度。


```py
class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        fa = list(range(n-1))

        def find(x: int)->int:
            rt = x
            while fa[rt] != rt:
                rt = fa[rt]

            while fa[x] != rt:
                fa[x] , x = rt , fa[x]
            
            return rt

        ans = []
        cnt = n-1

        for l , r in queries:
            fr = find(r-1)
            x = find(l)
            while x < r - 1:
                cnt -= 1
                fa[x] = fr
                x = find(x+1)
            ans.append(cnt)

        return ans
```

![alt text](<../pic/Screenshot 2024-11-20 at 10.54.24.png>)

```py
class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        nxt = list(range(1,n))
        cnt = n-1
        ans = []
        for l,r in queries:
            while nxt[l] < r:
                nxt[l] , l = r ,nxt[l]
                cnt -= 1
            ans.append(cnt)

        return ans
```

    给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。

    你可以对一个单词进行如下三种操作：

    插入一个字符
    删除一个字符
    替换一个字符

```py
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        @cache
        def DFS(i, j):
            if i < 0:
                return j + 1
            if j < 0 :
                return i+1
            if word1[i] == word2[j]:
                return DFS(i-1,j-1)
            return min(DFS(i-1,j) , DFS(i,j-1) , DFS(i-1,j-1))+1

        return DFS(len(word1)-1 , len(word2)-1)
```

```py
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n , m  = len(word1) , len(word2)
        f = [[0] * (m+1) for _ in range(n+1) ]
        f[0] = list(range(m+1))
        for i , x in enumerate(word1):
            f[i+1][0] = i+1
            for j , y in enumerate(word2):
                f[i+1][j+1] = f[i][j] if x == y else \
                min(f[i][j+1] , f[i+1][j] , f[i][j])+1
        return f[n][m]
```

```py
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        f = list(range(len(word2)+1))
        for x in word1:
            pre = f[0]
            f[0] += 1
            for j ,y in enumerate(word2):
                tmp = f[j+1]
                f[j+1] = pre if x == y else min(f[j+1], f[j] , pre)+1
                pre = tmp
        return f[-1]
```

### 新手
你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。

```py
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        dp1 = [0]* (n+1)
        dp2 = [0]* (n+1)
        for i in range(1,n):
            dp1[i+1] = max(dp1[i] , dp1[i-1] + nums[i-1])
            dp2[i+1] = max(dp2[i] , dp2[i-1] + nums[i])

        return max(dp1[-1] , dp2[-1])
```

```py
class Solution:
    # 198. 打家劫舍
    def rob1(self, nums: List[int]) -> int:
        f0 = f1 = 0
        for x in nums:
            f0, f1 = f1, max(f1, f0 + x)
        return f1

    def rob(self, nums: List[int]) -> int:
        return max(nums[0] + self.rob1(nums[2:-1]), self.rob1(nums[1:]))
```

一个魔法师有许多不同的咒语。

给你一个数组 power ，其中每个元素表示一个咒语的伤害值，可能会有多个咒语有相同的伤害值。

已知魔法师使用伤害值为 power[i] 的咒语时，他们就 不能 使用伤害为 power[i] - 2 ，power[i] - 1 ，power[i] + 1 或者 power[i] + 2 的咒语。

每个咒语最多只能被使用 一次 。

请你返回这个魔法师可以达到的伤害值之和的 最大值 。

```py
class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        cnt = Counter(power)
        a = sorted(cnt.keys())
        @cache
        def DFS(x : int)-> int:
            if x < 0:
                return 0
            i = a[x]
            j = x
            while j and a[j-1] >= i - 2:
                j -= 1

            return max(DFS(x-1) , DFS(j-1)+ cnt[i]*i)
        return DFS(len(a)-1)
```

```py
class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        cnt = Counter(power)
        a = sorted(cnt.keys())
        dp = [0] * (len(a)+1)
        j = 0
        for i , x in enumerate(a):
            while a[j] < x - 2:
                j += 1
            dp[i+1] = max(dp[i] , dp[j] + x*cnt[x]) 
        return dp[-1]
```
假设输入数组 power = [3, 4, 2, 3, 5, 6]：

cnt = Counter({3: 2, 4: 1, 2: 1, 5: 1, 6: 1}) 表示每个伤害值的出现次数。

a = [2, 3, 4, 5, 6] 是排序后的伤害值列表。

dp 数组最终会包含最大伤害值的累计结果，返回的 dp[-1] 即为最终的最大伤害值。
### 背包
给你两个 正 整数 n 和 x 。

请你返回将 n 表示成一些 互不相同 正整数的 x 次幂之和的方案数。换句话说，你需要返回互不相同整数 [n1, n2, ..., nk] 的集合数目，满足 n = n1^x + n2^x + ... + nk^x 。

由于答案可能非常大，请你将它对 10**9 + 7 取余后返回。

比方说，n = 160 且 x = 3 ，一个表示 n 的方法是 $ n = 2^3 + 3^3 + 5^3  $。
```py
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        num = []
        mod = 10**9 + 7
        for i in range(1, n+1):
            cur = pow(i , x)
            if cur > n:
                break
            num.append(cur)
        sz = len(num)
        @cache
        def DFS(n , i):
            if n == 0:
                return 1
            if i >= sz or n < 0:
                return 0
            
            return (DFS(n - num[i] , i+1) + DFS(n , i+1))%mod
        return DFS(n , 0)
```

这种做法超时。

```py
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        f = [1] + [0] * n
        for i in range(1 , n+1) : 
            v = i ** x
            if v > n:
                break
            for j in range(n , v-1 , -1):
                f[j] += f[j - v]
        return f[-1]%(10**9+7)
```

在两条独立的水平线上按给定的顺序写下 nums1 和 nums2 中的整数。

现在，可以绘制一些连接两个数字 nums1[i] 和 nums2[j] 的直线，这些直线需要同时满足：

 nums1[i] == nums2[j]
且绘制的直线不与任何其他连线（非水平线）相交。
请注意，连线即使在端点也不能相交：每个数字只能属于一条连线。

以这种方法绘制线条，并返回可以绘制的最大连线数。

```py
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        m = len(nums2)
        n = len(nums1)

        f = [[0] * (m+1) for _ in range(n+1)]
        for i , x  in enumerate(nums1):
            for j , y in enumerate(nums2):
                f[i+1][j+1] = f[i][j] + 1 if x == y else \
                                  max(f[i][j + 1], f[i + 1][j])

        return f[-1][-1]
```

### 状态机

给你一个下标从 0 开始的二进制字符串 s ，它表示一条街沿途的建筑类型，其中：

s[i] = '0' 表示第 i 栋建筑是一栋办公楼，
s[i] = '1' 表示第 i 栋建筑是一间餐厅。
作为市政厅的官员，你需要随机 选择 3 栋建筑。然而，为了确保多样性，选出来的 3 栋建筑 相邻 的两栋不能是同一类型。

比方说，给你 s = "001101" ，我们不能选择第 1 ，3 和 5 栋建筑，因为得到的子序列是 "011" ，有相邻两栋建筑是同一类型，所以 不合 题意。
请你返回可以选择 3 栋建筑的 有效方案数 。

```py
class Solution {
public:
    long long numberOfWays(string s) {
        //定义状态 0 1 01 10 在前面出现的次数
        //用0..3表示
        int n = s.size();
        long long ans = 0;
        vector<vector<long long>> dp(n + 1, vector<long long>(4));
        for (int i = 1; i <= n; i++)
        {
            if (s[i - 1] == '0')
            {
                ans += dp[i-1][2];
                dp[i][2] = dp[i - 1][2];
                dp[i][3] = dp[i - 1][3] + dp[i - 1][1];//前面的1可以和0组成新的10
                dp[i][0] = dp[i - 1][0] + 1;
                dp[i][1] = dp[i - 1][1];
            }
            else
            {
                ans += dp[i-1][3];
                dp[i][2] = dp[i - 1][2] + dp[i - 1][0];
                dp[i][3] = dp[i - 1][3];
                dp[i][0] = dp[i - 1][0];
                dp[i][1] = dp[i - 1][1] + 1;
            }
        }
        return ans;
    }
};
```

给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。

返回 你能获得的 最大 利润 。

```py
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [[0]*2 for _ in range(n+1)]
        
        dp[0][1] = -inf
        for i , price in enumerate(prices):
            dp[i+1][0] = max(dp[i][0] , dp[i][1] + price)
            dp[i+1][1] = max(dp[i][1] , dp[i+1][0]-price)
        return dp[n][0]
```

给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。


```py
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        buy1 = -prices[0]
        buy2 = -prices[0]
        s1 = 0 
        s2 = 0
        for p in prices:
            buy1 = max(buy1 , -p)
            s1 = max(s1 , buy1 + p)
            buy2 = max(buy2 , s1 - p)
            s2 = max(buy2 + p , s2)
        return s2
```

每日一题：

你有 k 个 非递减排列 的整数列表。找到一个 最小 区间，使得 k 个列表中的每个列表至少有一个数包含在其中。

我们定义如果 b-a < d-c 或者在 b-a == d-c 时 a < c，则区间 [a,b] 比 [c,d] 小。


```py
class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        h = [(arr[0] , i , 0 )for i ,arr in enumerate(nums)]
        heapify(h)

        ansl = h[0][0]
        ansr = r = max(arr[0] for arr in nums)
        while h[0][2] + 1 < len(nums[h[0][1]]):
            _, i ,j = h[0]
            x = nums[i][j+1]
            heapreplace(h , (x,i,j+1))
            r = max(r,x)
            l = h[0][0]
            if r - l < ansr - ansl:
                ansl , ansr = l , r
        return [ansl , ansr]
```

## 拓扑排序：

你有 n 道不同菜的信息。给你一个字符串数组 recipes 和一个二维字符串数组 ingredients 。第 i 道菜的名字为 recipes[i] ，如果你有它 所有 的原材料 ingredients[i] ，那么你可以 做出 这道菜。一道菜的原材料可能是 另一道 菜，也就是说 ingredients[i] 可能包含 recipes 中另一个字符串。

同时给你一个字符串数组 supplies ，它包含你初始时拥有的所有原材料，每一种原材料你都有无限多。

请你返回你可以做出的所有菜。你可以以 任意顺序 返回它们。

注意两道菜在它们的原材料中可能互相包含。
```c++
class Solution {
public:
    vector<string> findAllRecipes(vector<string>& recipes, vector<vector<string>>& ingredients, vector<string>& supplies) {
        int n = recipes.size();
        // 图
        unordered_map<string, vector<string>> depend;
        // 入度统计
        unordered_map<string, int> cnt;
        for (int i = 0; i < n; ++i) {
            for (const string& ing: ingredients[i]) {
                depend[ing].push_back(recipes[i]);
            }
            cnt[recipes[i]] = ingredients[i].size();
        }
        
        vector<string> ans;
        queue<string> q;
        // 把初始的原材料放入队列
        for (const string& sup: supplies) {
            q.push(sup);
        }
        // 拓扑排序
        while (!q.empty()) {
            string cur = q.front();
            q.pop();
            if (depend.count(cur)) {
                for (const string& rec: depend[cur]) {
                    --cnt[rec];
                    // 如果入度变为 0，说明可以做出这道菜
                    if (cnt[rec] == 0) {
                        ans.push_back(rec);
                        q.push(rec);
                    }
                }
            }
        }
        return ans;
    }
};
```
```py
class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        n = len(recipes)
        depend = defaultdict(list)
        cnt = Counter()
        for i in range(n):
            for r in ingredients[i]:
                depend[r].append(recipes[i])

            cnt[recipes[i]] = len(ingredients[i])
        
        ans = list()
        q = deque(supplies)
        while q:
            cur = q.popleft()
            if cur in depend:
                for x in depend[cur]:
                    cnt[x] -= 1
                    if cnt[x] == 0:
                        q.append(x)
                        ans.append(x)
        return ans
```

你总共需要上 numCourses 门课，课程编号依次为 0 到 numCourses-1 。你会得到一个数组 prerequisite ，其中 prerequisites[i] = [ai, bi] 表示如果你想选 bi 课程，你 必须 先选 ai 课程。

有的课会有直接的先修课程，比如如果想上课程 1 ，你必须先上课程 0 ，那么会以 [0,1] 数对的形式给出先修课程数对。
先决条件也可以是 间接 的。如果课程 a 是课程 b 的先决条件，课程 b 是课程 c 的先决条件，那么课程 a 就是课程 c 的先决条件。

你也得到一个数组 queries ，其中 queries[j] = [uj, vj]。对于第 j 个查询，您应该回答课程 uj 是否是课程 vj 的先决条件。

返回一个布尔数组 answer ，其中 answer[j] 是第 j 个查询的答案。

```py
class Solution:
    def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        G = defaultdict(list)
        in_degrees = [0] * numCourses

        for a, b in prerequisites:
            G[a].append(b)
            in_degrees[b] += 1

        deps = defaultdict(set)
        q = deque(i for i in range(numCourses) if in_degrees[i] == 0)

        while q:
            u = q.popleft()

            for v in G[u]:
                deps[v].add(u)
                deps[v] |= deps[u]
                in_degrees[v] -= 1

                if in_degrees[v] == 0:
                    q.append(v)


        return [a in deps[b] for a, b in queries]
```

```c++
class Solution {
public:
    vector<bool> checkIfPrerequisite(int numCourses, vector<vector<int>>& prerequisites, vector<vector<int>>& queries) {
        vector<vector<int>> g(numCourses);
        vector<int> indgree(numCourses, 0);
        vector<vector<bool>> isPre(numCourses, vector<bool>(numCourses, false));
        for (auto& p : prerequisites) {
            ++indgree[p[1]];
            g[p[0]].push_back(p[1]);
        }
        queue<int> q;
        for (int i = 0; i < numCourses; ++i) {
            if (indgree[i] == 0) {
                q.push(i);
            }
        }
        while (!q.empty()) {
            auto cur = q.front();
            q.pop();
            for (auto& ne : g[cur]) {
                isPre[cur][ne] = true;
                for (int i = 0; i < numCourses; ++i) {
                    isPre[i][ne] = isPre[i][ne] | isPre[i][cur];
                }
                --indgree[ne];
                if (indgree[ne] == 0) {
                    q.push(ne);
                }
            }
        }
        vector<bool> res;
        for (auto& query : queries) {
            res.push_back(isPre[query[0]][query[1]]);
        }
        return res;
    }
};
```

## 练习

给定一个字符串 s ，检查是否能重新排布其中的字母，使得两相邻的字符不同。

返回 s 的任意可能的重新排列。若不可行，返回空字符串 "" 。

```cpp
class Solution {
public:
    string reorganizeString(string s) {
        int n = s.length();
        unordered_map<char , int> count;
        for(auto x : s)
        {
            count[x] ++;
        }
        vector<pair<char , int>> a(count.begin() , count.end());
        ranges::sort(a, [](const auto& p , const auto & q){return p.second > q.second;});
        int m = a[0].second;
        if (m > n - m + 1)
        {
            return "";
        }
        string ans(n , 0);
        int i = 0 ;
        for(auto[x, y] : a)
        {
            while(y--)
            {
                ans[i] = x;
                i+=2;
                if(i >= n)
                {
                    i = 1;
                }
            }
        }
        return ans;
    }
};
```